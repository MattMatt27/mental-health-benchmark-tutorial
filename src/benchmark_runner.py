#!/usr/bin/env python3
"""
SIRI-2 Benchmark Runner
========================
Runs the SIRI-2 evaluation experiment across multiple LLM providers,
models, temperatures, and prompt variants. Each helper response is
evaluated individually, enabling helper-specific consistency analysis.

Usage:
    python benchmark_runner.py

Requires API keys set as environment variables:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
"""

from __future__ import annotations

import os
import json
import random
import hashlib
import time
import pathlib
import re
from datetime import datetime, timezone
from typing import List, Dict, Sequence, Union, Optional, Any
from pathlib import Path
from itertools import product

import pandas as pd
from tqdm import tqdm
import openai
from google import genai
from google.genai import types
import anthropic


# ── API client initialization ───────────────────────────────────────────────

def _get_api_clients() -> Dict[str, Any]:
    """Initialize API clients from environment variables."""
    clients = {}
    key = os.getenv('OPENAI_API_KEY')
    if key:
        clients['openai'] = openai.OpenAI(api_key=key)
    key = os.getenv('ANTHROPIC_API_KEY')
    if key:
        clients['anthropic'] = anthropic.Anthropic(api_key=key)
    key = os.getenv('GOOGLE_API_KEY')
    if key:
        clients['google'] = genai.Client(api_key=key)
    return clients


# ── Experiment configuration ────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "models": [
        "gpt-3.5-turbo-0125",
        "gpt-4o",  # resolved to gpt-4o-2024-11-20 at time of data collection
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "temperatures": [0.0, 1.0],
    "top_p": [1.0],
    # Note: the published experiment used max_tokens=256 for OpenAI/Anthropic
    # and max_tokens=2048 for Google models. 512 is a reasonable default for
    # new runs; the score-only response fits in ~10 tokens and the reasoning
    # variant rarely exceeds 200 tokens.
    "max_tokens": [512],
    "repeats": 10,
}

PROVIDER_MAP = {
    "gpt-3.5-turbo-0125":         "openai",
    "gpt-4o":                     "openai",
    "claude-sonnet-4-20250514":   "anthropic",
    "claude-opus-4-20250514":     "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-5-haiku-20241022":  "anthropic",
    "gemini-2.5-pro":             "google",
    "gemini-2.5-flash":           "google",
    "gemini-2.0-flash":           "google",
}

PROMPT_VARIANTS: dict[str, str] = {
    "detailed": (
        "The following item represents an excerpt from a hypothetical counseling session. "
        "Each excerpt begins with an expression by the client concerning some aspect of "
        "the situation they face, followed by a possible helper response to the client's remark. "
        "You are to rate the response in terms of how appropriate or inappropriate you feel the reply "
        "is to the client's comment from -3 to +3 on the 7\u2011point SIRI\u20112 scale "
        "(\u22123 = highly inappropriate \u2026 +3 = highly appropriate). "
        'Return ONLY JSON: {"score": int}.'
    ),
    "detailed_w_reasoning": (
        "The following item represents an excerpt from a hypothetical counseling session. "
        "Each excerpt begins with an expression by the client concerning some aspect of "
        "the situation they face, followed by a possible helper response to the client's remark. "
        "You are to rate the response in terms of how appropriate or inappropriate you feel the reply "
        "is to the client's comment from -3 to +3 on the 7\u2011point SIRI\u20112 scale "
        "(\u22123 = highly inappropriate \u2026 +3 = highly appropriate) and explain the reasoning behind your decision. "
        'Return ONLY JSON: {"score": int, "reasoning": string}.'
    ),
    "minimal": (
        "Score the response on the SIRI\u20112 -3 to +3 scale. Return ONLY JSON: "
        '{"score": int}.'
    ),
}


# ── Helper functions ────────────────────────────────────────────────────────

def _stable_uid(*parts: str) -> str:
    """10-char deterministic UID from parts."""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def _load_prompts(path_: Union[str, pathlib.Path, List[Dict[str, str]]]):
    """Load prompts from JSON file or list."""
    if isinstance(path_, (list, tuple)):
        items = list(path_)
    else:
        path_ = pathlib.Path(path_)
        if path_.suffix.lower() != ".json":
            raise ValueError("prompts_path must be a .json file")
        with path_.open() as f:
            items = json.load(f)

    # Ensure all items have item_id
    for idx, item in enumerate(items, 1):
        item.setdefault("item_id", f"{idx:02d}")

    return items


def _generate_messages(item: Dict[str, str], prompt_variant: str, helper: str) -> List[Dict[str, str]]:
    """Generate messages for individual helper evaluation."""
    system_prompt = PROMPT_VARIANTS[prompt_variant]

    helper_key = f'helper_{helper}'
    if helper_key not in item:
        raise KeyError(f"Item missing {helper_key}")

    user_prompt = (
        f"Client: {item['client']}\n\n"
        f"Helper: {item[helper_key]}\n\n"
        "Respond only with valid JSON."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _parse_json_response(raw_response: str) -> dict:
    """Robustly parse JSON from LLM response."""
    # Clean up common issues (e.g., "+3" -> "3")
    clean_text = re.sub(r'(:\s*)\+(\d+(\.\d+)?([eE][+-]?\d+)?)', r'\1\2', raw_response)

    # Try direct parsing
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    if clean_text.startswith("```json\n") and clean_text.endswith("```"):
        try:
            return json.loads(clean_text[8:-3].strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in response
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(clean_text[start:end+1])
        except json.JSONDecodeError:
            pass

    print(raw_response)
    raise ValueError(f"Could not parse JSON from response: {raw_response[:200]}...")


# ── LLM call wrapper ───────────────────────────────────────────────────────

def _call_llm(
    *,
    provider_clients: dict,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int
) -> str:
    """Unified wrapper for OpenAI / Anthropic / Google Gemini."""
    provider = PROVIDER_MAP.get(model, "openai")

    if provider == "openai":
        client = provider_clients.get("openai")
        if not client:
            raise RuntimeError("OpenAI client not available")

        kwargs = {
            "model": model,
            "messages": messages,
            "seed": seed,
            "response_format": {"type": "json_object"},
        }

        kwargs.update({
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        })

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    elif provider == "anthropic":
        client = provider_clients.get("anthropic")
        if not client:
            raise RuntimeError("Anthropic client not available")

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=messages[0]["content"],
            messages=[{"role": "user", "content": messages[1]["content"]}],
        )
        return response.content[0].text

    elif provider == "google":
        client = provider_clients.get("google")
        if not client:
            raise RuntimeError("Google client not available")

        system_prompt = None
        formatted_messages = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] in ["user", "model"]:
                formatted_messages.append(genai.types.Content(
                    role=message["role"],
                    parts=[genai.types.Part(text=message["content"])]
                ))
            else:
                raise ValueError(f"Unsupported message role: {message['role']}")

        # Gemini does not have a dedicated system message role, so we
        # prepend the system prompt to the first user message.
        if system_prompt:
            if formatted_messages and formatted_messages[0].role == "user":
                formatted_messages[0].parts[0].text = f"{system_prompt}\n\n{formatted_messages[0].parts[0].text}"
            else:
                formatted_messages.insert(0, genai.types.Content(role="user", parts=[genai.types.Part(text=system_prompt)]))

        # TODO: add rate limiting for OpenAI/Anthropic providers
        # Retry with exponential backoff for transient errors
        backoff = 30
        last_error = None
        for attempt in range(10):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=formatted_messages,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                last_error = e
                if "UNAVAILABLE" in str(e) or "exhausted" in str(e).lower():
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise

        raise RuntimeError(f"Gemini failed after retries: {last_error}")

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Experiment runner ───────────────────────────────────────────────────────

def run_experiment(
    *,
    prompts_path: Union[str, pathlib.Path, List[Dict[str, str]]],
    output_dir: pathlib.Path,
    provider_clients: Optional[Dict[str, Any]] = None,
    models: Optional[Sequence[str]] = None,
    temperatures: Optional[Sequence[float]] = None,
    top_p: Optional[Sequence[float]] = None,
    max_tokens: Optional[Sequence[int]] = None,
    prompt_variants: Optional[Sequence[str]] = None,
    repeats: Optional[int] = None,
):
    """
    Run the SIRI-2 evaluation experiment.

    Evaluates each helper individually, producing separate reasoning
    for each helper and enabling helper-specific consistency analysis.
    """
    models = list(models or DEFAULT_CONFIG["models"])
    temperatures = list(temperatures or DEFAULT_CONFIG["temperatures"])
    top_p_values = list(top_p or DEFAULT_CONFIG["top_p"])
    max_tokens_values = list(max_tokens or DEFAULT_CONFIG["max_tokens"])
    prompt_variants = list(prompt_variants or PROMPT_VARIANTS.keys())
    repeats = repeats or DEFAULT_CONFIG["repeats"]

    if provider_clients is None:
        provider_clients = _get_api_clients()

    for model in models:
        provider = PROVIDER_MAP.get(model, "openai")
        if provider not in provider_clients:
            raise RuntimeError(f"No API client available for {provider} (needed for {model})")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "api_responses_raw.jsonl"

    items = _load_prompts(prompts_path)

    # Load existing results to avoid duplicates
    done = set()
    if runs_path.exists():
        with runs_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (
                        rec["model"],
                        rec["temperature"],
                        rec.get("top_p", 1.0),
                        rec.get("max_tokens", 256),
                        rec["prompt_variant"],
                        rec["item_id"],
                        rec["helper"],
                        rec["repeat"]
                    )
                    done.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

    total_experiments = (
        len(models) * len(temperatures) * len(top_p_values) *
        len(max_tokens_values) * len(prompt_variants) * len(items) *
        2 * repeats
    )
    completed = len(done)

    print(f"Starting experiment: {total_experiments - completed} runs remaining")
    print(f"Parameters:")
    print(f"  Models: {models}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Top-p values: {top_p_values}")
    print(f"  Max tokens: {max_tokens_values}")
    print(f"  Prompt variants: {prompt_variants}")
    print(f"  Items: {len(items)}")
    print(f"  Repeats: {repeats}")

    for params in product(models, temperatures, top_p_values, max_tokens_values, prompt_variants):
        model, temp, top_p_val, max_tok, prompt_variant = params

        desc = f"{model} | T={temp} | p={top_p_val} | max={max_tok} | {prompt_variant}"
        pbar = tqdm(items, desc=desc, leave=False)

        for item in pbar:
            for helper in ['a', 'b']:
                messages = _generate_messages(item, prompt_variant, helper)

                for rep in range(repeats):
                    key = (model, temp, top_p_val, max_tok, prompt_variant,
                          item["item_id"], f"helper_{helper}", rep)

                    if key in done:
                        continue

                    seed = random.randint(1, 2**31 - 1)

                    try:
                        raw_response = _call_llm(
                            provider_clients=provider_clients,
                            model=model,
                            messages=messages,
                            temperature=temp,
                            top_p=top_p_val,
                            max_tokens=max_tok,
                            seed=seed,
                        )

                        if not raw_response or raw_response.strip() == "":
                            print(f"Empty response from {model} on helper {helper}, retrying...")
                            continue

                        parsed = _parse_json_response(raw_response)

                        record = {
                            "uid": _stable_uid(
                                model, item["item_id"], prompt_variant,
                                str(temp), str(top_p_val), str(max_tok),
                                helper, str(seed)
                            ),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model,
                            "temperature": temp,
                            "top_p": top_p_val,
                            "max_tokens": max_tok,
                            "prompt_variant": prompt_variant,
                            "item_id": item["item_id"],
                            "helper": f"helper_{helper}",
                            "repeat": rep,
                            "seed": seed,
                            "score": parsed.get("score"),
                            "reasoning": parsed.get("reasoning"),
                            "raw_response": raw_response,
                        }

                        with runs_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    except Exception as e:
                        print(f"\nError with {model} on item {item['item_id']}"
                              f" helper {helper}: {e}")
                        continue

    print(f"\nExperiment completed -- results saved to {runs_path}")


# ── Summary generation ──────────────────────────────────────────────────────

def summarize(output_dir: pathlib.Path):
    """
    Generate summary statistics from experiment runs.

    Args:
        output_dir: Directory containing api_responses_raw.jsonl
    """
    runs_path = output_dir / "api_responses_raw.jsonl"

    if not runs_path.exists():
        raise FileNotFoundError(f"No api_responses_raw.jsonl found -- run_experiment() first.")

    df = pd.read_json(runs_path, lines=True)

    summary = (
        df.groupby(["model", "temperature", "prompt_variant", "item_id", "helper"],
                  as_index=False)["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Create Item column matching expert results format (e.g., '1A', '2B')
    summary['helper_letter'] = summary['helper'].str.extract(r'helper_([ab])', expand=False).str.upper()
    summary['Item'] = summary['item_id'].astype(str).str.lstrip('0') + summary['helper_letter']
    summary = summary.drop(columns=['helper_letter'])

    # Reorder columns to put Item near item_id
    cols = summary.columns.tolist()
    cols.remove('Item')
    item_id_idx = cols.index('item_id') + 1
    cols.insert(item_id_idx, 'Item')
    summary = summary[cols]

    out = output_dir / "model_scores_by_condition.csv"
    summary.to_csv(out, index=False)
    print(f"Summary table saved to {out}")
    print(f"Item column format: {summary['Item'].iloc[0]} (e.g., '1A', '2B', etc.)")
    print(f"Total rows: {len(summary)}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    REPO_ROOT = Path(__file__).resolve().parent.parent

    clients = _get_api_clients()
    if not clients:
        print("No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or GOOGLE_API_KEY.")
    else:
        print(f"Initialized clients: {list(clients.keys())}")

        run_experiment(
            provider_clients=clients,
            prompts_path=REPO_ROOT / "instrument" / "siri2_items.json",
            output_dir=REPO_ROOT / "experiment-results",
        )

        summarize(REPO_ROOT / "experiment-results")
