#!/usr/bin/env python3
"""
SIRI-2 Reasoning Embedding Analysis
====================================
Analyzes whether models give consistent explanations across repeated
administrations by embedding their reasoning text and measuring
pairwise cosine similarity.

Two phases:
  1. Embedding generation and storage (run once, requires OpenAI API key)
  2. Consistency analysis (can be re-run on saved embeddings)

Usage:
    python reasoning_analysis.py

Requires:
    OPENAI_API_KEY environment variable (for embedding generation only)
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, linregress

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_RUNS = REPO_ROOT / "experiment-results" / "api_responses_raw.jsonl"
DEFAULT_EXPERT = REPO_ROOT / "instrument" / "siri2_expert_scores.csv"
DEFAULT_SUMMARY = REPO_ROOT / "experiment-results" / "model_scores_by_condition.csv"
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "embeddings"
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiment-results"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"


# ================================================================
# PHASE 1: EMBEDDING GENERATION AND STORAGE
# ================================================================

def init_embedder(model_name: str = "openai-small", api_key: str = None):
    """
    Initialize an embedding function.

    Args:
        model_name: 'openai-small' (text-embedding-3-small) or
                    'openai' (text-embedding-3-large)
        api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)

    Returns:
        A callable that takes a list of strings and returns a numpy array
        of shape (n_texts, embedding_dim).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    model = (
        "text-embedding-3-large" if model_name == "openai"
        else "text-embedding-3-small"
    )

    def embed(texts, batch_size=100):
        out = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Embedding",
            leave=False,
            disable=len(texts) <= batch_size,
        ):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            out.extend([e.embedding for e in resp.data])
            time.sleep(0.1)
        return np.array(out)

    return embed


def load_reasoning_data(
    jsonl_path: str,
    temperature: float,
    exclude_item14: bool = False,
) -> pd.DataFrame:
    """
    Load JSONL experiment results and extract reasoning entries.

    Filters to the 'detailed_w_reasoning' prompt variant (the only one
    that produces reasoning text) at the specified temperature.
    """
    data = []
    print(f"Loading data from {jsonl_path}")

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                entry.get("reasoning")
                and entry.get("prompt_variant") == "detailed_w_reasoning"
                and entry.get("temperature") == temperature
            ):
                data.append(entry)

    df = pd.DataFrame(data)
    df["item_id"] = df["item_id"].astype(str).str.zfill(2)

    if exclude_item14:
        df = df[df["item_id"] != "14"].copy()

    print(f"Loaded {len(df)} reasoning rows, {df['model'].nunique()} models")
    return df


def generate_and_save_embeddings(
    df: pd.DataFrame,
    embedder_func,
    output_dir: Path,
    embedding_model_name: str,
) -> None:
    """
    Generate embeddings for all reasoning texts and save to disk.

    Creates a subfolder under output_dir named after the embedding model,
    containing:
      - reasoning_embeddings_full.pkl  (DataFrame with embedding column)
      - embeddings_array.npy           (numpy array for fast loading)
      - reasoning_metadata.csv         (everything except the embedding vectors)
    """
    save_dir = Path(output_dir) / embedding_model_name.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating embeddings for {len(df)} reasoning texts...")
    all_texts = df["reasoning"].tolist()
    all_embeddings = embedder_func(all_texts)

    df = df.copy()
    df["embedding"] = list(all_embeddings)

    df.to_pickle(save_dir / "reasoning_embeddings_full.pkl")
    np.save(save_dir / "embeddings_array.npy", all_embeddings)
    df.drop(columns=["embedding"]).to_csv(
        save_dir / "reasoning_metadata.csv", index=False
    )

    print(f"Embeddings saved to {save_dir.resolve()}")


def load_embeddings(
    embedding_dir: Path,
    embedding_model_name: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load previously saved embeddings.

    Returns:
        (metadata DataFrame, embeddings array) with aligned row indices.
    """
    load_dir = Path(embedding_dir) / embedding_model_name.replace("/", "_")

    pkl_path = load_dir / "reasoning_embeddings_full.pkl"
    if pkl_path.exists():
        print(f"Loading embeddings from {pkl_path}")
        df = pd.read_pickle(pkl_path)
        embeddings = np.stack(df["embedding"].values)
        df = df.drop(columns=["embedding"]).reset_index(drop=True)
        return df, embeddings

    df = pd.read_csv(load_dir / "reasoning_metadata.csv")
    embeddings = np.load(load_dir / "embeddings_array.npy")
    print(f"Loaded {len(df)} reasoning entries with embeddings")
    return df.reset_index(drop=True), embeddings


# ================================================================
# PHASE 2: ANALYSIS FUNCTIONS
# ================================================================

def load_expert_scores(path: str, exclude_item14: bool = True) -> pd.DataFrame:
    """Load expert scores, returning a DataFrame with item_id, helper, M, SD columns."""
    expert = pd.read_csv(path)
    expert["item_id"] = expert["Item"].str.extract(r"(\d+)")[0].str.zfill(2)
    expert["helper"] = "helper_" + expert["Item"].str.extract(r"([AB])")[0].str.lower()

    if exclude_item14:
        expert = expert[expert["item_id"] != "14"].copy()

    return expert


def load_model_summaries(path: str, exclude_item14: bool = True) -> pd.DataFrame:
    """Load model performance summaries (mean/SD per model × item × condition)."""
    summ = pd.read_csv(path)
    summ["item_id"] = summ["item_id"].astype(str).str.zfill(2)

    if exclude_item14:
        summ = summ[summ["item_id"] != "14"].copy()

    # Ensure an 'Item' column exists for joining with expert scores
    if "Item" not in summ.columns and "helper" in summ.columns:
        helper_map = {
            "helper_a_score": "A", "helper_b_score": "B",
            "helper_a": "A", "helper_b": "B",
        }
        summ["Item"] = (
            summ["item_id"].str.lstrip("0")
            + summ["helper"].map(helper_map)
        )

    return summ


def _cosine_consistency(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute pairwise cosine similarity statistics for a set of embeddings.

    Returns dict with mean_similarity, std_similarity, min_similarity,
    max_similarity, and n_reasonings.
    """
    if len(embeddings) < 2:
        return {
            "mean_similarity": np.nan,
            "std_similarity": np.nan,
            "min_similarity": np.nan,
            "max_similarity": np.nan,
            "n_reasonings": len(embeddings),
        }

    sims = cosine_similarity(embeddings)
    upper = sims[np.triu_indices_from(sims, k=1)]

    return {
        "mean_similarity": float(upper.mean()),
        "std_similarity": float(upper.std()),
        "min_similarity": float(upper.min()),
        "max_similarity": float(upper.max()),
        "n_reasonings": len(embeddings),
    }


def analyze_consistency(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    exclude_item14: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Compute reasoning consistency for every model × item × helper group.

    The df and embeddings array must be row-aligned (same length, same order).

    Returns:
        Dict with 'detailed' (per-item results) and 'summary' (per-model)
        DataFrames.
    """
    # Apply item 14 filter once, up front
    if exclude_item14:
        keep = df["item_id"] != "14"
        df = df[keep].reset_index(drop=True)
        embeddings = embeddings[keep.values]

    results = []

    for (model, item_id, helper), group in tqdm(
        df.groupby(["model", "item_id", "helper"]),
        desc="Computing consistency",
    ):
        idx = group.index.values
        group_embeddings = embeddings[idx]
        consistency = _cosine_consistency(group_embeddings)
        scores = group["score"].values

        results.append({
            "model": model,
            "item_id": item_id,
            "helper": helper,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()) if len(scores) > 1 else 0.0,
            "score_range": float(np.ptp(scores)) if len(scores) > 1 else 0.0,
            **consistency,
        })

    detailed = pd.DataFrame(results)

    summary = (
        detailed.groupby("model")
        .agg(
            mean_similarity_avg=("mean_similarity", "mean"),
            mean_similarity_std=("mean_similarity", "std"),
            mean_similarity_min=("mean_similarity", "min"),
            score_std_avg=("score_std", "mean"),
            score_std_max=("score_std", "max"),
        )
        .round(3)
    )

    return {"detailed": detailed, "summary": summary}


def correlate_consistency_with_alignment(
    detailed: pd.DataFrame,
    expert_scores: pd.DataFrame,
    model_summaries: pd.DataFrame,
    prompt_variant: str = "detailed_w_reasoning",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Merge consistency results with expert alignment error and compute
    per-model correlations.

    Filters model_summaries to the same condition that reasoning was
    collected under (detailed_w_reasoning, T=1.0 by default) so that
    each consistency value pairs with exactly one alignment error,
    yielding n=48 independent observations per model.

    Returns a DataFrame with one row per model: correlation, p_value,
    n_items, mean_alignment_error, mean_consistency.
    """
    # Build Item key for joining
    df = detailed.copy()
    df["Item"] = (
        df["item_id"].str.lstrip("0")
        + df["helper"].map({"helper_a": "A", "helper_b": "B"})
    )

    # Merge expert scores
    df = df.merge(
        expert_scores[["Item", "M"]].rename(columns={"M": "expert_mean"}),
        on="Item",
        how="inner",
    )

    # Filter model summaries to the matching condition
    if "Item" in model_summaries.columns and "mean" in model_summaries.columns:
        summ = model_summaries.copy()
        if "prompt_variant" in summ.columns:
            summ = summ[summ["prompt_variant"] == prompt_variant]
        if "temperature" in summ.columns:
            summ = summ[summ["temperature"] == temperature]
        summ = summ[["model", "Item", "mean"]].rename(
            columns={"mean": "model_mean"}
        )
        df = df.merge(summ, on=["model", "Item"], how="inner")
    else:
        print("WARNING: Cannot compute alignment — model summaries missing "
              "required columns.")
        return pd.DataFrame()

    df["abs_error"] = (df["model_mean"] - df["expert_mean"]).abs()

    rows = []
    for model, group in df.groupby("model"):
        if len(group) < 10:
            continue
        r, p = pearsonr(group["mean_similarity"], group["abs_error"])
        rows.append({
            "model": model,
            "correlation": round(r, 3),
            "p_value": round(p, 4),
            "n_items": len(group),
            "mean_alignment_error": round(group["abs_error"].mean(), 3),
            "mean_consistency": round(group["mean_similarity"].mean(), 3),
        })

    return pd.DataFrame(rows)


def correlate_consistency_with_score_sd(
    detailed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-model Pearson correlation between item-level reasoning consistency
    and score SD across repetitions.
    """
    rows = []
    for model, group in detailed.groupby("model"):
        if len(group) < 10:
            continue
        r, p = pearsonr(group["mean_similarity"], group["score_std"])
        rows.append({
            "model": model,
            "correlation": round(r, 3),
            "p_value": round(p, 4),
            "n_items": len(group),
        })
    return pd.DataFrame(rows)


def identify_problem_items(
    detailed: pd.DataFrame,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Find items where reasoning consistency falls below a threshold."""
    low = detailed[detailed["mean_similarity"] < threshold]
    return (
        low.groupby(["model", "item_id"])
        .agg(
            mean_similarity=("mean_similarity", "mean"),
            score_range=("score_range", "mean"),
        )
        .reset_index()
        .sort_values("mean_similarity")
    )


# ================================================================
# VISUALIZATION
# ================================================================

def plot_consistency_summary(
    detailed: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """
    Two-panel figure:
      Left:  Violin plot of reasoning consistency by model
      Right: Scatter of consistency vs score SD (all models, colored)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: violin
    sns.violinplot(data=detailed, x="model", y="mean_similarity", ax=ax1)
    ax1.set_title("Reasoning Consistency by Model")
    ax1.set_ylabel("Mean Cosine Similarity (10 repetitions)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Right: scatter
    for model in sorted(detailed["model"].unique()):
        sub = detailed[detailed["model"] == model]
        ax2.scatter(
            sub["mean_similarity"], sub["score_std"],
            label=model, alpha=0.6, s=30,
        )
    ax2.set_xlabel("Reasoning Consistency (cosine similarity)")
    ax2.set_ylabel("Score SD (across 10 repetitions)")
    ax2.set_title("Reasoning Consistency vs Score Stability")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_consistency_violin(
    detailed: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Standalone violin plot of reasoning consistency by model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=detailed, x="model", y="mean_similarity", ax=ax)
    ax.set_title("Reasoning Consistency by Model")
    ax.set_ylabel("Mean Cosine Similarity (10 repetitions)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_consistency_vs_alignment(
    detailed: pd.DataFrame,
    expert_scores: pd.DataFrame,
    model_summaries: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Scatter of reasoning consistency vs alignment error, colored by model."""
    df = detailed.copy()
    df["Item"] = (
        df["item_id"].str.lstrip("0")
        + df["helper"].map({"helper_a": "A", "helper_b": "B"})
    )

    df = df.merge(
        expert_scores[["Item", "M"]].rename(columns={"M": "expert_mean"}),
        on="Item", how="inner",
    )

    if "Item" in model_summaries.columns and "mean" in model_summaries.columns:
        summ = model_summaries[["model", "Item", "mean"]].rename(
            columns={"mean": "model_mean"}
        )
        df = df.merge(summ, on=["model", "Item"], how="inner")
    else:
        print("WARNING: Skipping alignment plot — missing columns.")
        return

    df["abs_error"] = (df["model_mean"] - df["expert_mean"]).abs()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="mean_similarity", y="abs_error",
        hue="model", alpha=0.6, ax=ax,
    )
    ax.set_xlabel("Reasoning Consistency (cosine similarity)")
    ax.set_ylabel("Alignment Error |Model − Expert|")
    ax.set_title("Reasoning Consistency vs Expert Alignment")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_item14_exclusion(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    save_path: Optional[Path] = None,
):
    """
    Scatter of item-level consistency vs score variability (aggregated
    across models), with Item 14 highlighted as an outlier.

    This figure supports the decision to exclude Item 14 from main
    analyses — models also struggled with it, paralleling the original
    SIRI-2 validation where human raters could not reach consensus.

    Unlike analyze_consistency(), this function processes ALL items
    (including Item 14) so it can highlight the contrast.
    """
    results = []

    for (model, item_id, helper), group in df.groupby(
        ["model", "item_id", "helper"]
    ):
        if len(group) < 2:
            continue
        idx = group.index.values
        consistency = _cosine_consistency(embeddings[idx])
        scores = group["score"].values

        results.append({
            "model": model,
            "item_id": item_id,
            "helper": helper,
            "mean_similarity": consistency["mean_similarity"],
            "score_std": float(scores.std()),
        })

    per_item = pd.DataFrame(results)
    per_item["Item"] = (
        per_item["item_id"].str.zfill(2)
        + per_item["helper"].map({"helper_a": "A", "helper_b": "B"})
    )

    agg = (
        per_item.groupby("Item", as_index=False)
        .agg(
            mean_consistency=("mean_similarity", "mean"),
            mean_score_sd=("score_std", "mean"),
            item_id=("item_id", "first"),
        )
    )

    x = agg["mean_consistency"].values
    y = agg["mean_score_sd"].values
    is14 = agg["item_id"] == "14"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x[~is14], y[~is14], alpha=0.6, s=50, label="Items used in analysis")
    ax.scatter(
        x[is14], y[is14],
        color="red", s=100, edgecolors="darkred", linewidth=2,
        label="Item 14 (excluded)",
    )

    # Label all points with small item labels
    for idx in range(len(agg)):
        item_label = agg.iloc[idx]["Item"].lstrip("0")
        color = "darkred" if is14.iloc[idx] else "#555555"
        weight = "bold" if is14.iloc[idx] else "normal"
        fontsize = 9 if is14.iloc[idx] else 5.5
        ax.annotate(
            item_label, (x[idx], y[idx]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=fontsize, color=color, weight=weight,
        )

    # Trend line excluding Item 14
    if np.sum(~is14) > 1:
        slope, intercept, r_value, *_ = linregress(x[~is14], y[~is14])
        x_fit = np.linspace(x[~is14].min(), x[~is14].max(), 100)
        ax.plot(
            x_fit, intercept + slope * x_fit,
            "b-", linewidth=2, alpha=0.8,
            label=f"Trend (excl. Item 14), R²={r_value**2:.2f}",
        )

    ax.set_xlabel("Mean Reasoning Consistency Across Models", fontsize=11)
    ax.set_ylabel("Mean Score SD Across Models", fontsize=11)
    ax.set_title(
        "Item-Level Consistency vs Score Variability\n"
        "(Item 14 Highlighted for Exclusion)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_umap_projection(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    exclude_item14: bool = True,
    save_path: Optional[Path] = None,
):
    """
    UMAP projection of all reasoning embeddings, colored by model.

    Each point is one reasoning text (one model's explanation for one
    item on one repetition). Clustering reveals whether models reason
    similarly or differently about the same clinical scenarios.
    """
    import umap

    if exclude_item14:
        keep = df["item_id"] != "14"
        df = df[keep].reset_index(drop=True)
        embeddings = embeddings[keep.values]

    # Provider color mapping
    provider_colors = {
        "gpt-3.5-turbo-0125": "#1f77b4",
        "gpt-4o": "#0b4fa0",
        "claude-3-5-haiku-20241022": "#ff7f0e",
        "claude-3-5-sonnet-20241022": "#d45500",
        "claude-sonnet-4-20250514": "#c44e00",
        "claude-opus-4-20250514": "#8b3500",
        "gemini-2.0-flash": "#9467bd",
        "gemini-2.5-flash": "#7b4ea3",
        "gemini-2.5-pro": "#5e3788",
    }

    # Short display names
    display_names = {
        "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
        "gpt-4o": "GPT-4o",
        "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-opus-4-20250514": "Claude Opus 4",
        "gemini-2.0-flash": "Gemini 2.0 Flash",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
    }

    print("  Computing UMAP projection (this may take a moment)...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    for model in sorted(df["model"].unique()):
        mask = df["model"] == model
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=provider_colors.get(model, "#999999"),
            label=display_names.get(model, model),
            alpha=0.4, s=12, edgecolors="none",
        )

    # Label each item cluster at its centroid
    df = df.copy()
    df["item_label"] = (
        df["item_id"].str.lstrip("0")
        + df["helper"].str[-1].str.upper()
    )
    for label, group in df.groupby("item_label"):
        idx = group.index.values
        cx, cy = proj[idx, 0].mean(), proj[idx, 1].mean()
        ax.annotate(
            label, (cx, cy),
            fontsize=6, fontweight="bold", color="#333333",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7
            ),
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Projection of Model Reasoning Embeddings")
    ax.legend(
        fontsize=8, markerscale=3, framealpha=0.9,
        bbox_to_anchor=(1.02, 1), loc="upper left",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_similarity_heatmaps(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    save_path: Optional[Path] = None,
):
    """
    Side-by-side 10x10 pairwise similarity heatmaps for the highest-
    and lowest-consistency items (averaged across models).

    Makes the abstract concept of 'reasoning consistency' visually
    concrete: the high-consistency item is a solid warm block; the
    low-consistency item is noisy and patchy.
    """
    # Find highest and lowest consistency items (across all models)
    item_stats = []
    for (item_id, helper), group in df.groupby(["item_id", "helper"]):
        if item_id == "14":
            continue
        idx = group.index.values
        consistency = _cosine_consistency(embeddings[idx])
        item_stats.append({
            "item_id": item_id,
            "helper": helper,
            "label": f"{item_id.lstrip('0')}{helper[-1].upper()}",
            "mean_similarity": consistency["mean_similarity"],
        })

    stats_df = pd.DataFrame(item_stats)

    # Aggregate across models to find items that are globally high/low
    agg = (
        stats_df.groupby(["item_id", "helper", "label"], as_index=False)
        ["mean_similarity"].mean()
        .sort_values("mean_similarity")
    )

    low_item = agg.iloc[0]
    high_item = agg.iloc[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, item_row, title_prefix in [
        (axes[0], low_item, "Lowest Consistency"),
        (axes[1], high_item, "Highest Consistency"),
    ]:
        mask = (
            (df["item_id"] == item_row["item_id"])
            & (df["helper"] == item_row["helper"])
        )
        item_df = df[mask]
        item_emb = embeddings[mask.values]

        # Pick the model with the most repetitions for a clean 10x10
        model_counts = item_df["model"].value_counts()
        chosen_model = model_counts.index[0]
        model_mask = item_df["model"] == chosen_model
        model_emb = item_emb[model_mask.values]

        # Sort by repeat number for interpretable axes
        model_df = item_df[model_mask].copy()
        sort_order = model_df["repeat"].argsort().values
        model_emb = model_emb[sort_order]

        sims = cosine_similarity(model_emb)

        display_name = {
            "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
            "gpt-4o": "GPT-4o",
            "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
            "claude-sonnet-4-20250514": "Claude Sonnet 4",
            "claude-opus-4-20250514": "Claude Opus 4",
            "gemini-2.0-flash": "Gemini 2.0 Flash",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
        }.get(chosen_model, chosen_model)

        sns.heatmap(
            sims, vmin=0.5, vmax=1.0, cmap="YlOrRd", square=True,
            xticklabels=[f"R{i+1}" for i in range(len(sims))],
            yticklabels=[f"R{i+1}" for i in range(len(sims))],
            ax=ax,
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        )
        mean_sim = sims[np.triu_indices_from(sims, k=1)].mean()
        ax.set_title(
            f"{title_prefix}: Item {item_row['label']}\n"
            f"{display_name} · mean sim = {mean_sim:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Repetition")
        ax.set_ylabel("Repetition")

    plt.suptitle(
        "Pairwise Reasoning Similarity Across 10 Repetitions",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    EMBEDDING_MODEL = "openai-small"
    TEMPERATURE = 1.0

    model_dir = DEFAULT_EMBEDDING_DIR / EMBEDDING_MODEL.replace("/", "_")

    # ── Phase 1: generate embeddings (if not already saved) ──────────
    if not (model_dir / "embeddings_array.npy").exists():
        print("=" * 60)
        print("PHASE 1: Generating embeddings")
        print("=" * 60)

        embedder = init_embedder(EMBEDDING_MODEL)
        df_raw = load_reasoning_data(
            str(DEFAULT_RUNS), TEMPERATURE, exclude_item14=False
        )
        generate_and_save_embeddings(
            df_raw, embedder, DEFAULT_EMBEDDING_DIR, EMBEDDING_MODEL
        )
    else:
        print(f"Embeddings exist in {model_dir}, skipping generation.")

    # ── Phase 2: analysis ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Consistency analysis")
    print("=" * 60)

    df_all, embeddings_all = load_embeddings(DEFAULT_EMBEDDING_DIR, EMBEDDING_MODEL)

    # Load reference data
    try:
        expert_scores = load_expert_scores(str(DEFAULT_EXPERT), exclude_item14=True)
        model_summaries = load_model_summaries(str(DEFAULT_SUMMARY), exclude_item14=True)
    except FileNotFoundError:
        print("WARNING: Expert scores or summaries not found.")
        expert_scores = None
        model_summaries = None

    # Core consistency analysis (item 14 excluded)
    results = analyze_consistency(df_all, embeddings_all, exclude_item14=True)

    print("\n=== CONSISTENCY SUMMARY ===")
    print(results["summary"])

    # Correlations: consistency vs score SD
    sd_corr_df = correlate_consistency_with_score_sd(results["detailed"])
    if not sd_corr_df.empty:
        print("\n=== CONSISTENCY-SCORE SD CORRELATIONS ===")
        print(sd_corr_df.to_string(index=False))

    # Correlations with alignment error
    if expert_scores is not None and model_summaries is not None:
        corr_df = correlate_consistency_with_alignment(
            results["detailed"], expert_scores, model_summaries
        )
        if not corr_df.empty:
            print("\n=== CONSISTENCY-ALIGNMENT CORRELATIONS ===")
            print(corr_df.to_string(index=False))

    # ── Save results ─────────────────────────────────────────────────
    csv_dir = Path(DEFAULT_RESULTS_DIR)
    csv_dir.mkdir(parents=True, exist_ok=True)

    results["detailed"].to_csv(csv_dir / "reasoning_consistency_detailed.csv", index=False)
    results["summary"].to_csv(csv_dir / "reasoning_consistency_summary.csv")
    if not sd_corr_df.empty:
        sd_corr_df.to_csv(csv_dir / "consistency_score_sd_correlations.csv", index=False)

    problem = identify_problem_items(results["detailed"])
    problem.to_csv(csv_dir / "problem_items.csv", index=False)

    # ── Figures ──────────────────────────────────────────────────────
    fig_dir = Path(DEFAULT_FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    plot_consistency_summary(
        results["detailed"],
        save_path=fig_dir / "consistency_summary.png",
    )

    plot_consistency_violin(
        results["detailed"],
        save_path=fig_dir / "consistency_violin.png",
    )

    if expert_scores is not None and model_summaries is not None:
        plot_consistency_vs_alignment(
            results["detailed"], expert_scores, model_summaries,
            save_path=fig_dir / "consistency_vs_alignment.png",
        )

    # Item 14 plot uses ALL items (including 14) to show the contrast
    plot_item14_exclusion(
        df_all, embeddings_all,
        save_path=fig_dir / "item14_exclusion.png",
    )

    # UMAP projection of reasoning embeddings
    plot_umap_projection(
        df_all, embeddings_all,
        save_path=fig_dir / "umap_reasoning.png",
    )

    # Pairwise similarity heatmaps: highest vs lowest consistency items
    plot_similarity_heatmaps(
        df_all, embeddings_all,
        save_path=fig_dir / "similarity_heatmaps.png",
    )

    print(f"\nDone. CSVs saved to {csv_dir.resolve()}")
    print(f"      Figures saved to {fig_dir.resolve()}")
