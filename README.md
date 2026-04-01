# Does Claude's Constitution Have a Culture?

**[Paper (arXiv)](https://arxiv.org/abs/2603.28123)**

An empirical evaluation of cultural bias in Claude's Constitutional AI, measuring whether Claude's value positions cluster with WEIRD (Western, Educated, Industrialized, Rich, Democratic) populations.

## Overview

This project selects 55 items from the World Values Survey (WVS) Wave 7 that map to Claude's constitutional principles across 6 domains: moral justifiability, gender & family, religion & values, tolerance, political authority, and economic values. Claude Sonnet is tested in two formats:

- **Format A (Direct Survey):** WVS items presented verbatim; responses compared to country-level data from 90 countries.
- **Format B (Advice-Seeking):** Each WVS item rephrased as a naturalistic advice-seeking prompt—the way real users interact with Claude—to measure whether WEIRD values surface in advice.

Cultural steerability is tested by prepending country context ("I'm writing to you from Egypt/Sweden/Nigeria...") across 12 countries spanning six Inglehart-Welzel cultural zones.

## Key Findings

- Claude's value profile most closely resembles Germany (r=0.861), the Netherlands, and New Zealand, but **extends beyond all 90 surveyed nations** on 31/47 items (66%).
- On the Inglehart-Welzel dimensions, Claude's Self-Expression score (1.881) **exceeds all 90 countries**; nearest is Denmark (1.869).
- 40/47 items produce **identical responses across 5 runs at temperature 1.0** — Claude's values are effectively deterministic.
- Cultural context produces **negligible shifts** in substantive positions (mean |Cohen's d| = 0.108). When shifts occur, they move toward the named country's WVS values (p=0.011), but 75% of the time there is no change.
- Rhetorical strategy varies by domain: gender items are 80% directive, moral justifiability items are 76% balanced-lean.
- Removing the system prompt increases refusals (44.4% vs 14.9%) but does not change the values expressed (r=0.98 with system version).
- Cross-model replication on Claude Haiku 4.5 confirms the same cultural profile (r=0.956).

## Key Analyses

1. **Cultural Clustering** — Where does Claude sit on the global cultural map? (UMAP projection, Pearson/Spearman correlations, JSD, hierarchical clustering)
2. **Beyond-Human Extremity** — Bootstrap CIs showing Claude falls outside all country ranges on a majority of items
3. **Steerability** — Can country context shift Claude's value positions? (Cohen's d, binomial directional test)
4. **Rhetorical Strategy** — When does Claude give directive advice vs. defer to the user's culture?
5. **System Prompt Ablation** — Are values in the weights or elicited by prompt framing?
6. **Cross-Model Consistency** — Do Sonnet and Haiku share the same cultural profile?

## Grounding Data

- [Anthropic GlobalOpinionQA Dataset](https://huggingface.co/datasets/Anthropic/llm_global_opinions) — WVS items with country-level response distributions
- [World Values Survey Wave 7](https://www.worldvaluessurvey.org/) (2017-2022)

## Prior Work

This project builds on:
- Atari et al. (2023), ["Which Humans?"](https://arxiv.org/abs/2209.12000) — Showed GPT models cluster with WEIRD populations on psychological instruments
- Tao et al. (2024), ["Cultural Bias and Cultural Alignment of Large Language Models"](https://arxiv.org/abs/2311.14096) — WVS evaluation of GPT models across 112 countries with cultural prompting
- Durmus et al. (2023), ["Towards Measuring the Representation of Subjective Global Opinions in Language Models"](https://arxiv.org/abs/2306.16388) — Introduced GlobalOpinionQA
- Santurkar et al. (2023), ["Whose Opinions Do Language Models Reflect?"](https://arxiv.org/abs/2303.17548) — OpinionQA showing US demographic misalignment
- Bai et al. (2022), ["Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073) — The Constitutional AI methodology
- Anthropic (2023), ["Collective Constitutional AI"](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — Sourced a constitution from ~1,000 U.S. adults
- Anthropic (2025), ["Values in the Wild"](https://www.anthropic.com/research/values-wild) — Large-scale analysis of values expressed in 700K Claude conversations
- Henrich et al. (2010), ["The weirdest people in the world?"](https://doi.org/10.1017/S0140525X0999152X) — The foundational WEIRD critique

**Key differentiation from Tao et al. (2024):** They tested GPT models with 10 WVS items in direct survey format only. This project tests Claude (Constitutional AI, a different alignment mechanism), uses 55 items across 6 domains, introduces naturalistic advice-seeking prompts (Format B), and analyzes rhetorical strategies (directive vs. balanced vs. deferral).

## Quick Start

```bash
# 1. Install dependencies
pip install anthropic datasets pandas numpy scipy matplotlib seaborn scikit-learn umap-learn

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Select WVS items
python src/item_selection.py

# 4. Generate prompts
python src/prompt_generation.py

# 5. Estimate cost
python src/api_runner.py estimate

# 6. Run the eval (sequential)
python src/api_runner.py run

# 7. Parse Format A responses
python src/response_parser.py results/raw_responses/<file>.jsonl

# 8. Code Format B responses with LLM-as-judge
python src/response_parser.py results/raw_responses/<file>.jsonl --code-b

# 9. Run analyses and generate tables
python src/analysis.py

# 10. Generate publication figures
python -c "from src.figures import generate_all; generate_all()"
```

## Project Structure

```
├── src/
│   ├── item_selection.py      # WVS item selection by cross-country variance
│   ├── prompt_generation.py   # Generate Format A & B prompts with conditions
│   ├── api_runner.py          # Sequential & batch API runner
│   ├── response_parser.py     # Parse Format A, LLM-code Format B
│   ├── analysis.py            # 10 analyses + exploratory figures
│   └── figures.py             # Publication-quality figures (canonical for paper)
├── data/
│   └── processed/             # Selected items and country means
├── prompts/
│   └── all_prompts.jsonl      # All generated prompts (770: 55 Format A + 715 Format B)
├── results/
│   ├── coded/                 # Parsed & coded responses
│   ├── figures/               # Generated plots
│   └── tables/                # CSV analysis outputs
├── references/                # Key papers (gitignored)
├── paper/                     # arXiv preprint (NeurIPS format)
└── blog/                      # Medium post
```

## Model

Primary: `claude-sonnet-4-20250514`. Replication: `claude-haiku-4-5-20251001`.

## Cost

~770 API calls × $0.005 avg ≈ **$3.85** (sequential) or **$1.93** (batch).

## License

MIT
