#!/usr/bin/env python3
"""
SIRI-2 Analysis and Figure Generation
======================================
Computes consistency metrics, expert alignment, and generates
overview and per-question figures for the paper.

Usage:
    python analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')  # suppress seaborn/matplotlib deprecation warnings

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_SUMMARY = REPO_ROOT / "experiment-results" / "model_scores_by_condition.csv"
DEFAULT_EXPERT = REPO_ROOT / "instrument" / "siri2_expert_scores.csv"
DEFAULT_OUTPUT = REPO_ROOT / "figures"


def load_and_prepare_data(summary_path=DEFAULT_SUMMARY,
                         expert_path=DEFAULT_EXPERT,
                         exclude_item14=True):
    """
    Load and prepare both LLM and expert data for analysis.

    Args:
        summary_path: Path to LLM summary data (individual format)
        expert_path: Path to expert results
        exclude_item14: Whether to exclude item 14 from analysis

    Returns:
        Tuple of (llm_df, expert_df, merged_df)
    """
    llm_df = pd.read_csv(summary_path)
    expert_df = pd.read_csv(expert_path)

    # Prepare LLM data
    llm_df['item_id'] = llm_df['item_id'].astype(str).str.zfill(2)
    llm_df['helper_letter'] = llm_df['helper'].map({
        'helper_a': 'A',
        'helper_b': 'B'
    })
    llm_df['Item'] = llm_df['item_id'].astype(str).str.lstrip('0') + llm_df['helper_letter']

    # Prepare expert data
    expert_df['item_id'] = expert_df['Item'].str.extract(r'(\d+)')[0].str.zfill(2)
    expert_df['helper_letter'] = expert_df['Item'].str.extract(r'([AB])')[0]

    if exclude_item14:
        llm_df = llm_df[llm_df['item_id'] != '14'].copy()
        expert_df = expert_df[expert_df['item_id'] != '14'].copy()

    # Merge data
    merged_df = llm_df.merge(
        expert_df[['Item', 'M', 'SD']].rename(columns={'M': 'expert_mean', 'SD': 'expert_sd'}),
        on='Item',
        how='left'
    )

    # Calculate alignment metrics
    merged_df['abs_diff_from_expert'] = np.abs(merged_df['mean'] - merged_df['expert_mean'])
    merged_df['diff_from_expert'] = merged_df['mean'] - merged_df['expert_mean']

    print(f"Loaded {len(llm_df)} LLM rows and {len(expert_df)} expert rows")
    if exclude_item14:
        print("Item 14 excluded from analysis")

    return llm_df, expert_df, merged_df


def calculate_alignment_metrics(merged_df):
    """
    Calculate alignment metrics between models and experts.

    Returns:
        Tuple of (model_summary, condition_summary)
    """
    valid_df = merged_df[merged_df['expert_mean'].notna()].copy()

    # Model-level summary
    model_summary = []
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]

        mae = model_data['abs_diff_from_expert'].mean()
        rmse = np.sqrt((model_data['diff_from_expert'] ** 2).mean())
        mean_diff = model_data['diff_from_expert'].mean()
        corr, p_value = pearsonr(model_data['mean'], model_data['expert_mean'])
        within_1sd = (model_data['abs_diff_from_expert'] <= model_data['expert_sd']).mean() * 100

        model_summary.append({
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'mean_diff': mean_diff,
            'correlation': corr,
            'p_value': p_value,
            'pct_within_1sd': within_1sd
        })

    model_summary_df = pd.DataFrame(model_summary).sort_values('RMSE')

    # Condition-level summary (model x temperature x prompt)
    condition_summary = []
    for model in valid_df['model'].unique():
        for temp in valid_df['temperature'].unique():
            for prompt in valid_df['prompt_variant'].unique():
                cond_data = valid_df[
                    (valid_df['model'] == model) &
                    (valid_df['temperature'] == temp) &
                    (valid_df['prompt_variant'] == prompt)
                ]

                if len(cond_data) == 0:
                    continue

                mae = cond_data['abs_diff_from_expert'].mean()
                rmse = np.sqrt((cond_data['diff_from_expert'] ** 2).mean())
                self_consistency = cond_data['std'].mean()

                if len(cond_data) > 1:
                    corr, _ = pearsonr(cond_data['mean'], cond_data['expert_mean'])
                else:
                    corr = np.nan

                condition_summary.append({
                    'model': model,
                    'temperature': temp,
                    'prompt_variant': prompt,
                    'MAE': mae,
                    'RMSE': rmse,
                    'self_consistency_sd': self_consistency,
                    'correlation': corr,
                    'n_items': len(cond_data)
                })

    condition_summary_df = pd.DataFrame(condition_summary).sort_values(['model', 'temperature', 'prompt_variant'])

    return model_summary_df, condition_summary_df


def create_color_map(columns):
    """Create consistent color map for models."""
    color_map = {}

    openai_models = [m for m in columns if m.startswith("gpt") or m.startswith("o")]
    anthropic_models = [m for m in columns if m.startswith("claude")]
    gemini_models = [m for m in columns if m.startswith("gemini")]

    blues = plt.cm.Blues(np.linspace(0.45, 0.85, len(openai_models)))
    for m, c in zip(openai_models, blues):
        color_map[m] = c

    oranges = plt.cm.Oranges(np.linspace(0.45, 0.85, len(anthropic_models)))
    for m, c in zip(anthropic_models, oranges):
        color_map[m] = c

    purples = plt.cm.Purples(np.linspace(0.45, 0.85, len(gemini_models)))
    for m, c in zip(gemini_models, purples):
        color_map[m] = c

    color_map["Expert"] = "#2ca02c"

    return color_map


def get_bar_positions(n_items, columns):
    """Calculate bar positions with spacing between model groups."""
    expert_models = ["Expert"]
    gpt_models = ["gpt-3.5-turbo-0125", "gpt-4o"]
    claude_models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
                    "claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]

    available_gpt = [m for m in gpt_models if m in columns]
    available_claude = [m for m in claude_models if m in columns]
    available_gemini = [m for m in gemini_models if m in columns]

    positions = {}
    bar_w = 0.075
    group_spacing = 0.02

    x = np.arange(n_items)

    n_expert = 1 if "Expert" in columns else 0
    n_gpt = len(available_gpt)
    n_claude = len(available_claude)
    n_gemini = len(available_gemini)
    n_groups = sum([n_expert > 0, n_gpt > 0, n_claude > 0, n_gemini > 0])

    total_width = (n_expert + n_gpt + n_claude + n_gemini) * bar_w + (n_groups - 1) * group_spacing
    current_offset = -total_width / 2

    if "Expert" in columns:
        positions["Expert"] = x + current_offset + bar_w/2
        current_offset += bar_w
        if n_gpt > 0 or n_claude > 0 or n_gemini > 0:
            current_offset += group_spacing

    for model in available_gpt:
        positions[model] = x + current_offset + bar_w/2
        current_offset += bar_w

    if n_gpt > 0 and (n_claude > 0 or n_gemini > 0):
        current_offset += group_spacing

    for model in available_claude:
        positions[model] = x + current_offset + bar_w/2
        current_offset += bar_w

    if n_claude > 0 and n_gemini > 0:
        current_offset += group_spacing

    for model in available_gemini:
        positions[model] = x + current_offset + bar_w/2
        current_offset += bar_w

    return positions, bar_w


def plot_comprehensive_analysis(
    merged_df,
    llm_df=None,
    expert_df=None,
    save_dir=None,
    exclude_model=None,
    temp_to_plot=1.0,
    prompt_variant="detailed",
):
    """
    Generate all overview and per-question figures:
      - Fig A: RMSE + Consistency by model (provider colors)
      - Fig B: Temperature effects on RMSE + Consistency (red/blue)
      - Fig C: Prompt variant effects on RMSE + Consistency (red/blue/green)
      - Fig D: Per-question Expert vs Model means (4 stacked panels)
    """

    # Typography settings
    TITLE_FS = 13
    LABEL_FS = 11
    TICK_FS = 9
    LEGEND_FS = 10
    TITLE_WEIGHT = "bold"

    # Color palette
    RED = "#d64541"
    BLUE = "#2c82c9"
    GREEN = "#2ca02c"

    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=TICK_FS)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    merged_df_filtered = merged_df[merged_df["model"] != exclude_model].copy()
    merged_df_filtered["sq_err"] = (merged_df_filtered["mean"] - merged_df_filtered["expert_mean"]) ** 2

    model_order = (
        merged_df_filtered.groupby("model")["sq_err"].mean().pow(0.5).sort_values().index.tolist()
    )
    provider_color_map = create_color_map(model_order)

    # ===============================================================
    # FIG A: RMSE + Consistency by model (provider colors)
    # ===============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    desired_model_order = [
        "Expert",
        "gpt-3.5-turbo-0125", "gpt-4o",
        "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514", "claude-opus-4-20250514",
        "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    ]

    available_model_order = [m for m in desired_model_order if m in merged_df_filtered["model"].unique()]
    color_map_models = create_color_map(available_model_order)

    ax = axes[0]
    model_rmse = merged_df_filtered.groupby("model")["sq_err"].mean().pow(0.5).reindex(model_order)
    ax.barh(
        model_rmse.index,
        model_rmse.values,
        color=[color_map_models[m] for m in model_rmse.index],
        alpha=0.90,
    )
    ax.set_title("Model accuracy relative to expert consensus (RMSE)", fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    ax.set_xlabel("RMSE", fontsize=LABEL_FS)
    style_axis(ax)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    model_consistency = merged_df_filtered.groupby("model")["std"].mean().reindex(model_order)
    ax.barh(
        model_consistency.index,
        model_consistency.values,
        color=[color_map_models[m] for m in model_rmse.index],
        alpha=0.90,
    )
    ax.set_title("Model response variability (within-condition SD)", fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    ax.set_xlabel("Average within-condition SD", fontsize=LABEL_FS)
    style_axis(ax)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "overview_rmse_consistency.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ===============================================================
    # FIG B: Temperature effects (RMSE + consistency)
    # ===============================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    temp_rmse = (
        merged_df_filtered.groupby(["model", "temperature"])["sq_err"]
        .mean()
        .pow(0.5)
        .unstack()
        .reindex(model_order)
    )

    temp_cols = list(temp_rmse.columns)
    temp_color_map = {0: RED, 0.0: RED, 1: BLUE, 1.0: BLUE}
    temp_colors = [temp_color_map.get(c, BLUE) for c in temp_cols]

    temp_rmse.plot(kind="bar", ax=ax, width=0.70, color=temp_colors, alpha=0.85)
    ax.set_title("Effect of temperature on accuracy (RMSE)", fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    ax.set_ylabel("RMSE", fontsize=LABEL_FS)
    ax.set_xlabel("")
    style_axis(ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Temperature", fontsize=LEGEND_FS, title_fontsize=LEGEND_FS, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    temp_consistency = (
        merged_df_filtered.groupby(["model", "temperature"])["std"]
        .mean()
        .unstack()
        .reindex(model_order)
    )
    temp_consistency.plot(kind="bar", ax=ax, width=0.70, color=temp_colors, alpha=0.85)
    ax.set_title("Effect of temperature on response variability", fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    ax.set_ylabel("Average within-condition SD", fontsize=LABEL_FS)
    ax.set_xlabel("")
    style_axis(ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Temperature", fontsize=LEGEND_FS, title_fontsize=LEGEND_FS, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "overview_temperature_effects.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ===============================================================
    # FIG C: Prompt variant effects (RMSE + consistency)
    # ===============================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    base_prompt_colors = [RED, BLUE, GREEN]

    def extended_prompt_colors(n):
        if n <= 3:
            return base_prompt_colors[:n]
        extra = list(plt.cm.Set2(np.linspace(0.1, 0.9, n - 3)))
        return base_prompt_colors + extra

    ax = axes[0]
    prompt_rmse = (
        merged_df_filtered.groupby(["model", "prompt_variant"])["sq_err"]
        .mean()
        .pow(0.5)
        .unstack()
        .reindex(model_order)
    )
    prompt_colors = extended_prompt_colors(prompt_rmse.shape[1])

    prompt_rmse.plot(kind="bar", ax=ax, width=0.70, color=prompt_colors, alpha=0.85)
    ax.set_title("Effect of prompt variant on accuracy (RMSE)", fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    ax.set_ylabel("RMSE", fontsize=LABEL_FS)
    ax.set_xlabel("")
    style_axis(ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(
        title="Prompt variant",
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_FS,
        loc="upper left",
    )
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    prompt_consistency = (
        merged_df_filtered.groupby(["model", "prompt_variant"])["std"]
        .mean()
        .unstack()
        .reindex(model_order)
    )
    prompt_consistency.plot(kind="bar", ax=ax, width=0.70, color=prompt_colors, alpha=0.85)
    ax.set_title("Effect of prompt variant on response variability", fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
    ax.set_ylabel("Average within-condition SD", fontsize=LABEL_FS)
    ax.set_xlabel("")
    style_axis(ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(
        title="Prompt variant",
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_FS,
        loc="upper left",
    )
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "overview_prompt_effects.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ===============================================================
    # FIG D: Per-question bars (provider colors)
    # ===============================================================
    if llm_df is None or expert_df is None:
        print("Skipping per-question plot (llm_df and/or expert_df not provided).")
        return

    plot_df = llm_df[
        (llm_df["temperature"] == temp_to_plot)
        & (llm_df["prompt_variant"] == prompt_variant)
        & (llm_df["model"] != exclude_model)
    ].copy()

    pivot = plot_df.pivot_table(index="Item", columns="model", values="mean")
    pivot.insert(0, "Expert", expert_df.set_index("Item")["M"])

    items_order = []
    for i in range(1, 26):
        if i != 14:
            items_order.extend([f"{i}A", f"{i}B"])
    pivot = pivot.reindex([item for item in items_order if item in pivot.index])

    desired_model_order = [
        "Expert",
        "gpt-3.5-turbo-0125", "gpt-4o",
        "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514", "claude-opus-4-20250514",
        "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    ]
    pivot = pivot[[m for m in desired_model_order if m in pivot.columns]]

    color_map = create_color_map(pivot.columns)

    total_items = len(pivot)
    base_size = total_items // 4
    remainder = total_items % 4
    splits, start = [], 0
    for i in range(4):
        end = start + base_size + (1 if i < remainder else 0)
        splits.append((start, end))
        start = end
    pivot_parts = [pivot.iloc[s:e] for s, e in splits]

    fig, axes = plt.subplots(4, 1, figsize=(20, 24))
    titles = [
        f"Expert vs Model Means: Questions 1-6 ({prompt_variant} prompt, T = {temp_to_plot})",
        f"Expert vs Model Means: Questions 7-12 ({prompt_variant} prompt, T = {temp_to_plot})",
        f"Expert vs Model Means: Questions 13-18 ({prompt_variant} prompt, T = {temp_to_plot})",
        f"Expert vs Model Means: Questions 19-25 ({prompt_variant} prompt, T = {temp_to_plot})",
    ]

    for i, (data, ax, title) in enumerate(zip(pivot_parts, axes, titles)):
        x = np.arange(len(data))
        positions, bar_w = get_bar_positions(len(data), pivot.columns)

        for col in pivot.columns:
            ax.bar(
                positions[col],
                data[col].values,
                width=bar_w,
                color=color_map[col],
                label=col,
                alpha=0.90,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data.index, rotation=90, fontsize=TICK_FS)
        ax.set_ylabel("Mean SIRI-2 Score", fontsize=LABEL_FS)
        ax.set_title(title, fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
        style_axis(ax)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Shared legend organized by provider
    handles, labels = axes[0].get_legend_handles_labels()
    handle_map = {lab: h for lab, h in zip(labels, handles)}

    expert_row = [m for m in ["Expert"] if m in handle_map]
    openai_row = [m for m in ["gpt-3.5-turbo-0125", "gpt-4o"] if m in handle_map]
    anthropic_row = [m for m in [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ] if m in handle_map]
    gemini_row = [m for m in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"] if m in handle_map]

    y_expert = 0.995
    y_openai = 0.982
    y_anth   = 0.969
    y_gem    = 0.956

    def add_row(row, y):
        if not row:
            return
        fig.legend(
            [handle_map[m] for m in row],
            row,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=len(row),
            fontsize=LEGEND_FS,
            frameon=False,
            columnspacing=1.1,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

    add_row(expert_row, y_expert)
    add_row(openai_row, y_openai)
    add_row(anthropic_row, y_anth)
    add_row(gemini_row, y_gem)

    y_min = min(ax.get_ylim()[0] for ax in axes)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.subplots_adjust(
        left=0.05,
        right=0.995,
        bottom=0.06,
        top=0.925,
        hspace=0.28,
    )

    if save_dir:
        plt.savefig(
            save_dir / f"per_question_expert_vs_models_T{temp_to_plot}_prompt_{prompt_variant}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.show()

    # ===============================================================
    # FIG E: Per-question bars averaged across ALL conditions
    # ===============================================================
    if llm_df is None or expert_df is None:
        print("Skipping cross-condition per-question plot (llm_df and/or expert_df not provided).")
        return

    avg_df = (
        llm_df[llm_df["model"] != exclude_model]
        .groupby(["Item", "model"], as_index=False)["mean"]
        .mean()
    )

    pivot_avg = avg_df.pivot_table(index="Item", columns="model", values="mean")
    pivot_avg.insert(0, "Expert", expert_df.set_index("Item")["M"])

    items_order = []
    for i in range(1, 26):
        if i != 14:
            items_order.extend([f"{i}A", f"{i}B"])
    pivot_avg = pivot_avg.reindex([item for item in items_order if item in pivot_avg.index])
    pivot_avg = pivot_avg[[m for m in desired_model_order if m in pivot_avg.columns]]

    color_map_avg = create_color_map(pivot_avg.columns)

    total_items_avg = len(pivot_avg)
    base_size_avg = total_items_avg // 4
    remainder_avg = total_items_avg % 4
    splits_avg, start_avg = [], 0
    for i in range(4):
        end_avg = start_avg + base_size_avg + (1 if i < remainder_avg else 0)
        splits_avg.append((start_avg, end_avg))
        start_avg = end_avg
    pivot_parts_avg = [pivot_avg.iloc[s:e] for s, e in splits_avg]

    fig, axes = plt.subplots(4, 1, figsize=(20, 24))
    titles_avg = [
        "Expert vs Model Means: Questions 1-6 (averaged across all conditions)",
        "Expert vs Model Means: Questions 7-12 (averaged across all conditions)",
        "Expert vs Model Means: Questions 13-18 (averaged across all conditions)",
        "Expert vs Model Means: Questions 19-25 (averaged across all conditions)",
    ]

    for i, (data, ax, title) in enumerate(zip(pivot_parts_avg, axes, titles_avg)):
        x = np.arange(len(data))
        positions, bar_w = get_bar_positions(len(data), pivot_avg.columns)

        for col in pivot_avg.columns:
            ax.bar(
                positions[col],
                data[col].values,
                width=bar_w,
                color=color_map_avg[col],
                label=col,
                alpha=0.90,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data.index, rotation=90, fontsize=TICK_FS)
        ax.set_ylabel("Mean SIRI-2 Score", fontsize=LABEL_FS)
        ax.set_title(title, fontsize=TITLE_FS, fontweight=TITLE_WEIGHT)
        style_axis(ax)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    handle_map = {lab: h for lab, h in zip(labels, handles)}

    expert_row = [m for m in ["Expert"] if m in handle_map]
    openai_row = [m for m in ["gpt-3.5-turbo-0125", "gpt-4o"] if m in handle_map]
    anthropic_row = [m for m in [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ] if m in handle_map]
    gemini_row = [m for m in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"] if m in handle_map]

    add_row(expert_row, y_expert)
    add_row(openai_row, y_openai)
    add_row(anthropic_row, y_anth)
    add_row(gemini_row, y_gem)

    y_min = min(ax.get_ylim()[0] for ax in axes)
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    fig.subplots_adjust(
        left=0.05,
        right=0.995,
        bottom=0.06,
        top=0.925,
        hspace=0.28,
    )

    if save_dir:
        plt.savefig(
            save_dir / "per_question_expert_vs_models_all_conditions.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.show()


def run_comprehensive_analysis(
    summary_path=DEFAULT_SUMMARY,
    expert_path=DEFAULT_EXPERT,
    output_dir=DEFAULT_OUTPUT,
    exclude_item14=True,
    exclude_model=None,
    temp_to_plot=1.0,
    prompt_variant="detailed",
):
    """Run the full SIRI-2 analysis pipeline."""
    print("=" * 80)
    print("SIRI-2 COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading and preparing data...")
    llm_df, expert_df, merged_df = load_and_prepare_data(summary_path, expert_path, exclude_item14)

    llm_df_f = llm_df[llm_df["model"] != exclude_model].copy()
    merged_df_f = merged_df[merged_df["model"] != exclude_model].copy()

    print("\n2. Calculating expert alignment metrics...")
    model_summary, condition_summary = calculate_alignment_metrics(merged_df_f)
    print("\nModel-Level Summary (sorted by RMSE):")
    print(model_summary.round(3))

    print("\n3. Saving analysis tables...")
    model_summary.to_csv(output_dir / "model_alignment_summary.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_alignment_summary.csv", index=False)

    print("\n4. Creating visualizations...")

    plot_comprehensive_analysis(
        merged_df_f,
        llm_df=llm_df_f,
        expert_df=expert_df,
        save_dir=output_dir,
        exclude_model=exclude_model,
        temp_to_plot=temp_to_plot,
        prompt_variant=prompt_variant,
    )

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nTop 5 Models by Accuracy (lowest RMSE):")
    print(model_summary[["model", "RMSE", "MAE", "correlation"]].head().to_string(index=False))

    print("\nTemperature Effects on Accuracy (RMSE):")
    merged_df_f["sq_err"] = (merged_df_f["mean"] - merged_df_f["expert_mean"]) ** 2
    temp_rmse = merged_df_f.groupby("temperature")["sq_err"].mean() ** 0.5
    print(temp_rmse.round(4))

    print("\nPrompt Variant Effects on Accuracy (RMSE):")
    prompt_rmse = merged_df_f.groupby("prompt_variant")["sq_err"].mean() ** 0.5
    print(prompt_rmse.round(4))

    print("\nTop 5 Best Performing Conditions:")
    best_conditions = condition_summary.nsmallest(5, "RMSE")[["model", "temperature", "prompt_variant", "RMSE", "MAE"]]
    print(best_conditions.to_string(index=False))

    print(f"\nAnalysis complete. Results saved to {output_dir.resolve()}")


if __name__ == '__main__':
    run_comprehensive_analysis()
