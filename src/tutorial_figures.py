#!/usr/bin/env python3
"""
Paper-specific figures: directional bias and combined SIRI-2 comparison.

Figures produced:
  1. Directional bias by model and expert-rated category (bar chart)
  2. Combined SIRI-2 comparison: model ranges, human benchmarks, and
     chat-tool results on a single axis (range/dot plot)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "experiment-results"
EXPERT_CSV = REPO_ROOT / "instrument" / "siri2_expert_scores.csv"
SUMMARY_CSV = DATA_DIR / "model_scores_by_condition.csv"
SIRI2_SCORES_CSV = DATA_DIR / "siri2_total_scores.csv"
COMBINED_CSV = DATA_DIR / "siri2_combined_comparison.csv"
OUT_DIR = REPO_ROOT / "figures"

NAME_MAP = {
    'claude-3-5-haiku-20241022':  'Claude 3.5\nHaiku',
    'claude-3-5-sonnet-20241022': 'Claude 3.5\nSonnet',
    'claude-opus-4-20250514':     'Claude\nOpus 4',
    'claude-sonnet-4-20250514':   'Claude\nSonnet 4',
    'gemini-2.0-flash':           'Gemini 2.0\nFlash',
    'gemini-2.5-flash':           'Gemini 2.5\nFlash',
    'gemini-2.5-pro':             'Gemini 2.5\nPro',
    'gpt-3.5-turbo-0125':        'GPT-3.5\nTurbo',
    'gpt-4o':                     'GPT-4o',
}


# ── Load & prepare ──────────────────────────────────────────────────────────
def load_data():
    """Load model summaries and expert scores, compute error and category columns."""
    summary = pd.read_csv(SUMMARY_CSV)
    expert  = pd.read_csv(EXPERT_CSV)
    expert.columns = ['Item', 'expert_mean', 'expert_sd']
    expert['Item'] = expert['Item'].str.strip()

    merged = summary.merge(expert, on='Item', how='inner')
    merged['error']    = merged['mean'] - merged['expert_mean']
    merged['abs_error'] = merged['error'].abs()
    merged['sq_error'] = merged['error'] ** 2
    merged['expert_category'] = merged['expert_mean'].apply(
        lambda x: 'inappropriate' if x < -1 else ('appropriate' if x > 1 else 'ambiguous')
    )
    return merged


# ── Figure 1: Directional bias bar chart ────────────────────────────────────
def figure_bias_by_category(merged, out_path):
    """Paired bar chart: mean signed error on inappropriate vs appropriate items."""

    model_cat = merged.groupby(['model', 'expert_category']).agg(
        mean_error=('error', 'mean'),
        se_error=('error', lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()

    # Order by inappropriate error (most biased first)
    order_df = model_cat[model_cat['expert_category'] == 'inappropriate'] \
                   .sort_values('mean_error', ascending=False)
    model_order = order_df['model'].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_order))
    width = 0.35

    inapp_vals, inapp_ses, app_vals, app_ses = [], [], [], []
    for model in model_order:
        row_i = model_cat[(model_cat['model'] == model)
                          & (model_cat['expert_category'] == 'inappropriate')]
        row_a = model_cat[(model_cat['model'] == model)
                          & (model_cat['expert_category'] == 'appropriate')]
        inapp_vals.append(row_i['mean_error'].values[0])
        inapp_ses.append(row_i['se_error'].values[0])
        app_vals.append(row_a['mean_error'].values[0])
        app_ses.append(row_a['se_error'].values[0])

    ax.bar(x - width/2, inapp_vals, width, yerr=inapp_ses, capsize=3,
           label='Expert-rated inappropriate', color='#d64541', alpha=0.85)
    ax.bar(x + width/2, app_vals, width, yerr=app_ses, capsize=3,
           label='Expert-rated appropriate', color='#2c82c9', alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean signed error\n(positive = model rated higher than experts)',
                  fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([NAME_MAP.get(m, m) for m in model_order], fontsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('Directional bias in model ratings relative to expert consensus',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-1.8, 1.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 2: Combined SIRI-2 comparison (models, humans, chat tools) ────
def figure_combined_comparison(out_path,
                               scores_csv=SIRI2_SCORES_CSV,
                               combined_csv=COMBINED_CSV):
    """Range/dot plot showing all SIRI-2 scores on a single axis.

    - Models shown as horizontal ranges (min-max across conditions) with
      individual dots for each of the 6 experimental conditions.
    - Human benchmark groups shown as individual dots.
    - LLM chat-tool results shown as individual dots (different marker).
    - Expert panel shown as a vertical reference line.
    """
    from matplotlib.lines import Line2D

    scores = pd.read_csv(scores_csv)
    combined = pd.read_csv(combined_csv)

    # ── Model ranges (min / median / max across 6 conditions) ────────
    model_ranges = scores.groupby('model_display')['siri2_score'].agg(
        ['min', 'median', 'max']
    ).reset_index()
    model_ranges.columns = ['label', 'lo', 'mid', 'hi']
    model_ranges['sort_key'] = model_ranges['mid']
    model_ranges['entry_type'] = 'model'

    # Provider for color coding
    def _provider(label):
        if 'Claude' in label:
            return 'anthropic'
        if 'Gemini' in label:
            return 'google'
        return 'openai'
    model_ranges['provider'] = model_ranges['label'].apply(_provider)

    # ── Human benchmarks ─────────────────────────────────────────────
    humans = combined[combined['type'] == 'human'][['respondent', 'siri2_score', 'source']].copy()
    # Build label with citation: "Master's counselors, post (Neimeyer & Bonnelle 1997)"
    def _human_label(row):
        name = row['respondent']
        cite = row['source']
        # Move (pre)/(post) out of parens and add citation
        for tag in ['(post)', '(pre)']:
            if tag in name:
                name = name.replace(f' {tag}', f', {tag[1:-1]}')
        return f"{name} ({cite})"
    humans['label'] = humans.apply(_human_label, axis=1)
    humans['mid'] = humans['siri2_score']
    humans['lo'] = humans['mid']
    humans['hi'] = humans['mid']
    humans['sort_key'] = humans['mid']
    humans['entry_type'] = 'human'
    humans['provider'] = ''
    humans = humans[['label', 'lo', 'mid', 'hi', 'sort_key', 'entry_type', 'provider']]

    # ── Chat-tool results ────────────────────────────────────────────
    chat = combined[combined['type'] == 'llm_chat_tool'][['respondent', 'siri2_score', 'source']].copy()
    # Replace "(chat tool)" with citation: "Claude 3.5 Sonnet (McBain et al. 2025)"
    chat['label'] = chat.apply(
        lambda r: r['respondent'].replace('(chat tool)', f"({r['source']})"), axis=1)
    chat['mid'] = chat['siri2_score']
    chat['lo'] = chat['mid']
    chat['hi'] = chat['mid']
    chat['sort_key'] = chat['mid']
    chat['entry_type'] = 'chat_tool'
    chat['provider'] = ''
    chat = chat[['label', 'lo', 'mid', 'hi', 'sort_key', 'entry_type', 'provider']]

    # ── Expert panel ─────────────────────────────────────────────────
    expert_score = combined[combined['type'] == 'expert_panel']['siri2_score'].iloc[0]

    # ── Combine and sort ─────────────────────────────────────────────
    rows = pd.concat([model_ranges, humans, chat], ignore_index=True)
    rows = rows.sort_values('sort_key').reset_index(drop=True)

    # ── Colors ───────────────────────────────────────────────────────
    provider_colors = {
        'anthropic': '#e07b39',
        'openai':    '#4a90d9',
        'google':    '#7b68ae',
    }
    HUMAN_COLOR = '#5a6872'
    CHAT_COLOR = '#c53030'
    EXPERT_COLOR = '#2ca02c'

    # ── Marker shapes by prompt variant ────────────────────────────
    prompt_markers = {
        'minimal': 'o',
        'detailed': 's',
        'detailed_w_reasoning': '^',
    }
    # Sizes adjusted so shapes look visually equal
    # Filled gets +1.5 to compensate for white edge eating into fill area
    marker_sizes_filled = {'o': 7.5, 's': 7, '^': 8}
    marker_sizes_open   = {'o': 6, 's': 5.5, '^': 6.5}

    # ── Plot ─────────────────────────────────────────────────────────
    n_rows = len(rows)
    fig_h = max(6.5, n_rows * 0.28 + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_positions = np.arange(n_rows)

    for i, row in rows.iterrows():
        y = y_positions[i]
        etype = row['entry_type']

        if etype == 'model':
            color = provider_colors[row['provider']]
            # Range bar (connector)
            ax.plot([row['lo'], row['hi']], [y, y],
                    '-', color=color, linewidth=4, alpha=0.25, zorder=2,
                    solid_capstyle='round')
            # Best and worst condition dots: shape=prompt, fill=temperature
            model_conds = scores[scores['model_display'] == row['label']]
            best = model_conds.loc[model_conds['siri2_score'].idxmin()]
            worst = model_conds.loc[model_conds['siri2_score'].idxmax()]
            for cond in [best, worst]:
                marker = prompt_markers[cond['prompt_variant']]
                # Filled triangles sit slightly low; nudge up
                y_adj = y - 0.08 if (marker == '^' and cond['temperature'] == 0) else y
                if cond['temperature'] == 0:
                    ms = marker_sizes_filled[marker]
                    ax.plot(cond['siri2_score'], y_adj, marker, markersize=ms,
                            markerfacecolor=color, markeredgecolor='white',
                            markeredgewidth=0.6, zorder=4)
                else:
                    ms = marker_sizes_open[marker]
                    ax.plot(cond['siri2_score'], y_adj, marker, markersize=ms,
                            markerfacecolor='none', markeredgecolor=color,
                            markeredgewidth=1.3, zorder=4)
        elif etype == 'human':
            ax.plot(row['mid'], y, 's', markersize=5.5,
                    markerfacecolor=HUMAN_COLOR, markeredgecolor='white',
                    markeredgewidth=0.6, zorder=4)
        elif etype == 'chat_tool':
            ax.plot(row['mid'], y, 'D', markersize=6,
                    markerfacecolor='none', markeredgecolor=CHAT_COLOR,
                    markeredgewidth=1.8, zorder=4)

    # Expert reference line
    ax.axvline(expert_score, color=EXPERT_COLOR, linewidth=1.2,
               linestyle='--', alpha=0.7, zorder=1)
    ax.text(expert_score + 0.4, n_rows - 0.5,
            f'Expert panel ({expert_score})',
            fontsize=7.5, color=EXPERT_COLOR, va='top')

    # Y-axis labels
    ax.set_yticks(y_positions)
    labels = []
    for _, row in rows.iterrows():
        lab = row['label']
        if row['entry_type'] == 'model':
            lab = f"{lab}  ({row['lo']:.1f} - {row['hi']:.1f})"
        labels.append(lab)
    ax.set_yticklabels(labels, fontsize=8)

    # Style label colors by type
    for i, row in rows.iterrows():
        tick_label = ax.get_yticklabels()[i]
        etype = row['entry_type']
        if etype == 'model':
            tick_label.set_color(provider_colors[row['provider']])
            tick_label.set_fontweight('bold')
        elif etype == 'human':
            tick_label.set_color(HUMAN_COLOR)
        elif etype == 'chat_tool':
            tick_label.set_color(CHAT_COLOR)

    ax.invert_yaxis()
    ax.set_xlabel('SIRI-2 Total Score (lower = better alignment with experts)',
                  fontsize=10)
    ax.set_title('SIRI-2 Performance: LLM API Results, Chat Tools, and Human Benchmarks',
                 fontsize=11, fontweight='bold', pad=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)
    ax.tick_params(axis='y', length=0)

    # Legend with section headers
    blank = Line2D([], [], color='none', marker='None', linestyle='None')

    legend_elements = [
        # Section: LLM API conditions
        (blank, 'LLM API (best & worst condition):'),
        # Prompt shapes
        (Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
                markeredgecolor='#888', markersize=6.5), 'Minimal prompt'),
        (Line2D([0], [0], marker='s', color='w', markerfacecolor='#888',
                markeredgecolor='#888', markersize=6), 'Detailed prompt'),
        (Line2D([0], [0], marker='^', color='w', markerfacecolor='#888',
                markeredgecolor='#888', markersize=7), 'Detailed + reasoning'),
        # Spacer between shape and fill sections
        (blank, ''),
        (Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
                markeredgecolor='white', markeredgewidth=0.6, markersize=7.5),
         'Filled = temp 0'),
        (Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                markeredgecolor='#888', markeredgewidth=1.3, markersize=6),
         'Open = temp 1'),
        # Section: Benchmarks
        (blank, ''),
        (blank, 'Comparison benchmarks:'),
        (Line2D([0], [0], marker='s', color='w', markerfacecolor=HUMAN_COLOR,
                markeredgecolor='white', markeredgewidth=0.6, markersize=5.5),
         'Human benchmark group'),
        (Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
                markeredgecolor=CHAT_COLOR, markeredgewidth=1.8, markersize=6),
         'LLM chat tool'),
        (Line2D([0], [0], color=EXPERT_COLOR, linewidth=1.2, linestyle='--',
                alpha=0.7), 'Expert suicidologists'),
    ]
    handles, labels = zip(*legend_elements)
    leg = ax.legend(handles, labels, loc='upper right', fontsize=7,
                    framealpha=0.95, edgecolor='#ddd',
                    handletextpad=0.4, labelspacing=0.4)
    # Bold the section headers
    for text in leg.get_texts():
        if text.get_text().endswith(':'):
            text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Console output: verification tables ─────────────────────────────────────
def print_verification(merged):
    """Print directional bias tables and McBain chat-tool comparison."""
    print("=" * 80)
    print("BIAS ANALYSIS")
    print("=" * 80)

    inapp = merged[merged['expert_category'] == 'inappropriate']
    app   = merged[merged['expert_category'] == 'appropriate']

    print("\nPer-model mean signed error on INAPPROPRIATE items:")
    model_inapp = inapp.groupby('model')['error'].mean().sort_values(ascending=False)
    for model, err in model_inapp.items():
        short = NAME_MAP.get(model, model).replace('\n', ' ')
        print(f"  {short:22s}  {err:+.3f}")

    print(f"\nAll positive? {(model_inapp > 0).all()}")

    print("\nPer-model mean signed error on APPROPRIATE items:")
    model_app = app.groupby('model')['error'].mean().sort_values(ascending=False)
    for model, err in model_app.items():
        short = NAME_MAP.get(model, model).replace('\n', ' ')
        print(f"  {short:22s}  {err:+.3f}")

    # McBain comparison table
    print("\n" + "=" * 80)
    print("McBAIN COMPARISON (Claude 3.5 Sonnet & GPT-4o)")
    print("=" * 80)

    mcbain_vals = {
        'claude-3-5-sonnet-20241022': {'signed': 0.608, 'siri2': 36.65},
        'gpt-4o':                     {'signed': 0.865, 'siri2': 45.71},
    }

    for model_id, mc in mcbain_vals.items():
        short = NAME_MAP.get(model_id, model_id).replace('\n', ' ')
        print(f"\n  {short}")
        print(f"    McBain chat tool: signed={mc['signed']:+.3f}, SIRI-2={mc['siri2']}")

        sub = merged[merged['model'] == model_id]
        for pv in ['minimal', 'detailed', 'detailed_w_reasoning']:
            for temp in [0, 1.0]:
                cond = sub[(sub['prompt_variant'] == pv) & (sub['temperature'] == temp)]
                se  = cond['error'].mean()
                s2  = cond['abs_error'].sum()
                pct = (cond['error'] > 0).mean() * 100
                print(f"    API {pv:25s} t={temp:.1f} | signed={se:+.3f} | SIRI-2={s2:.1f} | over={pct:.0f}%")


# ── Figure 3: Prompt variant effects on SIRI-2 scores ────────────────────
def figure_prompt_effects(out_path, scores_csv=SIRI2_SCORES_CSV):
    """Grouped bar chart: SIRI-2 total score by model and prompt variant.

    Scores are averaged across both temperature settings for each
    model × prompt_variant combination.
    """
    scores = pd.read_csv(scores_csv)

    # Average across temperatures → one value per model × prompt_variant
    avg = scores.groupby(['model', 'model_display', 'prompt_variant', 'prompt_display'])\
                ['siri2_score'].mean().reset_index()

    # Sort models by overall average score (best first)
    model_order = avg.groupby('model_display')['siri2_score'].mean()\
                     .sort_values().index.tolist()

    # Map to raw model IDs for provider coloring
    display_to_raw = dict(zip(scores['model_display'], scores['model']))

    prompt_order = ['minimal', 'detailed', 'detailed_w_reasoning']
    prompt_labels = {'minimal': 'Minimal', 'detailed': 'Detailed',
                     'detailed_w_reasoning': 'Detailed + reasoning'}
    prompt_colors = {'minimal': '#5aaa46', 'detailed': '#2c82c9',
                     'detailed_w_reasoning': '#e07b39'}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_order))
    n_prompts = len(prompt_order)
    width = 0.25
    offsets = [-(n_prompts - 1) / 2 * width + i * width for i in range(n_prompts)]

    for i, pv in enumerate(prompt_order):
        vals = []
        for model_disp in model_order:
            row = avg[(avg['model_display'] == model_disp) &
                      (avg['prompt_variant'] == pv)]
            vals.append(row['siri2_score'].values[0] if len(row) else 0)
        ax.bar(x + offsets[i], vals, width, label=prompt_labels[pv],
               color=prompt_colors[pv], alpha=0.85)

    # Expert panel baseline
    expert_line = ax.axhline(y=32.5, color='#2ca02c', linewidth=1.2,
                             linestyle='--', alpha=0.7)

    ax.set_ylabel('SIRI-2 Total Score\n(lower = better alignment with experts)',
                  fontsize=11)
    ax.set_xticks(x)
    # Use single-line names for x-axis
    single_line_names = [n.replace('\n', ' ') for n in model_order]
    ax.set_xticklabels(single_line_names, fontsize=9, rotation=30, ha='right')

    # Add expert line to legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(expert_line)
    labels.append('Expert panel baseline (32.5)')
    ax.legend(handles, labels, title='Prompt variant', fontsize=10,
              title_fontsize=10, loc='upper left')
    ax.set_title('Effect of prompt variant on SIRI-2 performance',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    merged = load_data()
    print(f"Loaded {len(merged)} rows, {merged['model'].nunique()} models, "
          f"{merged['Item'].nunique()} items\n")

    print_verification(merged)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n\nGenerating figures...")
    figure_bias_by_category(merged, OUT_DIR / "fig_bias_by_category.png")
    figure_combined_comparison(OUT_DIR / "fig_combined_comparison.png")
    figure_prompt_effects(OUT_DIR / "fig_prompt_effects.png")

    print("\nDone.")
