# Benchmarking Language Models for Clinical Safety

Companion repository for **"Benchmarking Language Models for Clinical Safety: A Primer for Mental Health Professionals."**

Contains the full experiment code, raw results (27,000 API responses), and interactive notebooks for replicating and extending SIRI-2 benchmarking of language models from OpenAI, Anthropic, and Google.

## What This Repository Contains

We administered the [Suicide Intervention Response Inventory (SIRI-2)](https://doi.org/10.1080/074811897202137) to nine commercially available language models through their APIs, across three prompt variants, two temperature settings, and ten repeated administrations per condition. The results are scored against the original expert panel consensus and compared to published human benchmark groups spanning three decades of SIRI-2 research.

**Models evaluated:** GPT-3.5 Turbo, GPT-4o, Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude Sonnet 4, Claude Opus 4, Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 2.5 Pro

## Repository Structure

```
├── notebooks/
│   ├── 01_explore_our_results.ipynb   # Explore pre-computed results (no API keys needed)
│   └── 02_build_your_own_benchmark.ipynb  # Run your own evaluation
├── src/
│   ├── benchmark_runner.py            # Experiment runner (sends items to LLM APIs)
│   ├── siri2_scoring.py               # SIRI-2 total score computation and human benchmarks
│   ├── analysis.py                    # Consistency metrics and figure generation
│   ├── tutorial_figures.py            # Paper-specific figures (bias, comparison plots)
│   └── reasoning_analysis.py          # Reasoning embedding consistency analysis
├── experiment-results/
│   ├── api_responses_raw.jsonl        # All 27,000 raw API responses
│   ├── model_scores_by_condition.csv  # Per-item means by model x condition
│   ├── siri2_total_scores.csv         # SIRI-2 total scores per condition
│   ├── siri2_combined_comparison.csv  # Models + human benchmarks combined
│   └── reasoning_consistency_detailed.csv  # Per-item reasoning consistency metrics
├── instrument/
│   ├── siri2_items.json               # SIRI-2 scenarios (24 items, 48 helper responses)
│   └── siri2_expert_scores.csv        # Expert panel consensus scores (M, SD)
├── .env.example                       # API key template
├── requirements.txt
└── LICENSE
```

## Getting Started

### Explore the results (no API keys needed)

```bash
git clone https://github.com/MattMatt27/mental-health-benchmark-tutorial.git
cd mental-health-benchmarking-tutorial

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook notebooks/01_explore_our_results.ipynb
```

Notebook 01 walks through the pre-computed results: SIRI-2 scores, human benchmark comparisons, directional bias analysis, and per-item breakdowns. Everything runs locally on the included data files.

### Run your own benchmark

To run models yourself, you need at least one API key. Copy the template and add your keys:

```bash
cp .env.example .env
# Edit .env with your API keys (you don't need all three providers)
```

Then open Notebook 02, which walks through configuring and running a small-scale evaluation, scoring the results, and comparing to the published benchmarks.

## Experiment Design

| Parameter | Values |
|---|---|
| Models | 9 (3 providers x 2-4 models each) |
| Prompt variants | Minimal, Detailed, Detailed + reasoning |
| Temperatures | 0.0 (near-deterministic), 1.0 (stochastic) |
| Items | 25 scenarios x 2 helpers = 50 responses (48 scored, Item 14 excluded) |
| Repetitions | 10 per condition |
| **Total API calls** | **27,000** |

**Scoring:** SIRI-2 total score = sum of |model rating - expert mean| across 48 validated items. Lower scores indicate better alignment with expert suicidologists. The expert panel baseline is 32.5.

## Key Findings

- SIRI-2 scores ranged from **19.5** (Claude Opus 4, detailed prompt, T=0) to **84.0** (GPT-3.5 Turbo, minimal prompt, T=1)
- All nine models showed a **positive directional bias** on items experts rated as clinically inappropriate, rating harmful responses more favorably than experts did
- **Prompt design** had a larger effect on performance than temperature, with the detailed prompt reducing average scores from 50.2 to 41.0
- Configuration effects were often larger than model differences: a model's best settings could outperform a more capable model's worst
- The SIRI-2 is **approaching ceiling effects** for the most capable models — the top-performing configuration scored below the expert panel baseline, meaning the instrument can no longer differentiate further improvement. New, harder instruments are needed for frontier evaluation

## Regenerating Figures and Derived Data

Figures and derived CSVs are generated from the raw data by the scripts in `src/`. To regenerate:

```bash
python src/siri2_scoring.py          # SIRI-2 total scores and combined comparison table
python src/analysis.py               # Overview and per-item figures
python src/tutorial_figures.py       # Paper figures (bias chart, range plot, prompt effects)
python src/reasoning_analysis.py     # Reasoning embedding analysis (requires embeddings/)
```

The reasoning analysis requires pre-computed embeddings (not included in the repo due to size). To generate them, set `OPENAI_API_KEY` in your `.env` and run `reasoning_analysis.py` — it will embed the 4,500 reasoning texts using OpenAI's `text-embedding-3-small` model before running the analysis.

## Citation

If you use this code or data, please cite the companion paper:

> [Citation will be added upon publication]

## Contributing

This project is part of [MindBench.ai](https://mindbench.ai), a benchmarking initiative for mental health AI developed in collaboration with the National Alliance on Mental Illness (NAMI). If you are a clinician or researcher interested in contributing expert-rated scenarios, contact the corresponding author.

## License

MIT
