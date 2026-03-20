# Claude Constitution Culture Eval

## Pipeline (run in order)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd ~/Documents/claude-constitution-culture

python3 src/item_selection.py          # Select 55 WVS items → data/processed/
python3 src/prompt_generation.py       # Generate 770 prompts → prompts/all_prompts.jsonl
python3 src/api_runner.py estimate     # Check cost before running
python3 src/api_runner.py run 5 --format A    # Format A with 5 runs
python3 src/api_runner.py run 1 --format B    # Format B with 1 run
python3 src/api_runner.py run 5 --format A --no-system  # Ablation: no system prompt
python3 src/response_parser.py results/raw_responses/<file>.jsonl           # Parse Format A
python3 src/response_parser.py results/raw_responses/<file>.jsonl --code-b  # LLM-code Format B
python3 src/analysis.py                # Generate all figures → results/figures/
```

## Architecture

```
src/item_selection.py     → Selects WVS items by cross-country variance, assigns to 6 domains
src/prompt_generation.py  → Format A (direct survey) + Format B (advice-seeking) × 12 countries
src/api_runner.py         → Sequential + batch modes, resume-on-failure, cost estimation
src/response_parser.py    → Regex parser (Format A), LLM-as-judge coder (Format B)
src/analysis.py           → 10 analyses: cultural clustering, MDS cultural map (exploratory), distribution overlap,
                            steerability (Wilcoxon), steerability direction (binomial), refusal/strategy,
                            domain comparison, bootstrap extremity CIs, Cohen's d, gap statistic,
                            Inglehart-Welzel dimensions, JSD distributional distance
src/figures.py            → Publication-quality figure generation (UMAP cultural map, correlations, dendrogram, etc.)
```

## Data Flow

```
HuggingFace (Anthropic/llm_global_opinions)
  → data/processed/selected_items.csv        (55 items, metadata)
  → data/processed/country_means.csv         (item × country means, 90 countries)
  → data/processed/selected_items_full.json  (full selections for distribution plots)
  → prompts/all_prompts.jsonl                (770 prompts: 55 Format A + 715 Format B)
  → results/raw_responses/*.jsonl            (API responses)
  → results/coded/format_a_parsed.csv        (parsed Format A)
  → results/coded/format_b_coded.csv         (LLM-coded Format B)
  → results/figures/*.png                    (7 analysis figures)
  → results/tables/*.csv                     (correlation tables, steerability tables)
```

## LaTeX Compilation

- `tectonic` installed via Homebrew for local PDF compilation
- Compile paper: `cd paper && tectonic main.tex` — produces `main.pdf`
- Warnings about "Underfull \vbox" are cosmetic (normal for NeurIPS format)
- **Always recompile after editing main.tex** — keep PDF in sync

## Environment

- Requires `poppler` for PDF processing: `brew install poppler`
- Python 3.12+ with: anthropic, datasets, pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, umap-learn
- **Reproducibility-critical versions**: umap-learn==0.5.11, scikit-learn==1.8.0 (UMAP + k-means results change across major versions even with random_state=42)
- HuggingFace warns about unauthenticated requests — safe to ignore, or set HF_TOKEN to suppress
- Bash `cd` doesn't persist between tool calls — always use absolute paths or `cd X &&` chaining
- Extract text from PDFs: `pdftotext "file.pdf" -` (requires poppler)

## Analysis Standards

- **Always standardize (z-score) before PCA or clustering** when items have mixed scales (1-4, 1-10, 1-11)
- **Fit scalers/PCA on reference data (countries) only**, then transform the test subject (Claude) — never include Claude in the fit
- **Check string matching for substring collisions** — "Not mentioned" contains "Mentioned", always use longest-match-first
- **Check refusal patterns before content parsing** — a response saying "I cannot provide" may still contain option text by coincidence
- **Apply FDR correction** (Benjamini-Hochberg) whenever reporting multiple p-values
- **Report confidence intervals** on correlations (Fisher z-transform)
- **Use Euclidean distance with Ward linkage** — Ward assumes Euclidean; correlation-based distance is invalid
- **Report both Pearson and Spearman** as a robustness check
- When in doubt, run a self-audit: read the output CSV row-by-row and verify against raw responses
- **Run end-to-end validation after every pipeline change** — adding runs, fixing prompts, new experiments, etc. all create integration bugs that only surface when the full pipeline runs. Never assume a local fix is safe without re-running the audit.
- **Validation script**: Run the inline end-to-end audit (checks file integrity, row counts, parse quality, semantic plausibility, prompt mappings, and analysis outputs). Run after every pipeline change.

## Critical: Semantic Plausibility Checks

**Before reporting any finding, ask: "Does this make sense given what we know about the system?"**

- If a result contradicts Claude's known design (e.g., Claude saying something is "never justifiable" when its constitution is explicitly permissive), treat it as a parser/data bug until proven otherwise.
- Always verify surprising results by reading the raw responses — don't trust parsed values alone.
- Long responses (>150 chars) in Format A are almost always refusals, even if they contain stray numbers. Check length before trusting numeric extraction.
- Lesson learned: "casual sex = never justifiable" was a refusal misparse, not a real finding. The refusal pattern list missed "I can't provide a single definitive answer." Always test parser against actual model outputs, not assumed patterns.

## Key Gotchas

- WVS selections field is a `defaultdict` repr string, not valid JSON. Parse with: `s.replace("defaultdict(<class 'list'>, ", "").rstrip(")")` then `ast.literal_eval()`
- WVS options include varying DK/NA labels ("Don't know", "No answer", "Missing; Not available", etc.) — filter all variants via the `SKIP_OPTIONS` set in each file
- Format B advice prompts are manually keyed to WVS topics via `ADVICE_PROMPTS` dict in `prompt_generation.py`. All 55 items now have matched prompts.
- `prompt_id` is NOT unique across multiple runs — always include `run` in merge keys when joining parsed data back to raw responses
- Response length >150 chars in Format A is almost always a refusal. Parser checks this first. Edge case: short refusals containing option text (e.g., "concepts like heaven" matches "No") — ~0.4% false positive rate, acceptable.
- `analysis.py run_all()` loads Format B from a hardcoded filename (responses_20260320_192138.jsonl), not a glob. Update if re-running experiments.
- `api_runner.py run` saves incrementally to JSONL — safe to ctrl-C and resume with `api_runner.py resume <filename>`
- Analysis expects specific file paths in `results/coded/` — run parser before analysis
- 7 items have non-zero variance: Item 4 (Divorce, σ=1.64) has the most; 6 others have σ≈0.45 (one run differs by 1 point). 40/47 items with multiple valid runs produce identical responses across all 5 runs.
- India is in the 12 target countries for Format B but has NO WVS Wave 7 country means — silently excluded from directional/distance analyses. Only 11 countries have WVS reference data.
- **Paper numbers drift from data**: After any analysis rerun, grep the paper for every statistic (refusal counts, denominators, p-values, percentages) and cross-check against actual CSV files. Key counts to verify: refusals (41, not 42), items with no usable values (3), items with >1 valid run (47, not 52), bootstrap beyond count (31/47), binomial p-value (0.011 one-sided), within-condition sigma (2.15), paired Cohen's d (mean |d|=0.108), correlation countries (89, not 90 — Montenegro excluded), steerability direction (83/138 from 545 total), ablation refusal rate (44.4%, 22 items all-refused).
- Item 22 (Heaven) has a false-positive parse on run 1: refusal text "I cannot..." matched "No" option via substring. Inflates valid-item count by 1 (52 vs true 51). Downstream impact minimal (excluded from bootstrap).
- Steerability direction analysis has varying n per country (Sweden/France=39, Egypt=43, others=55) because not all items have WVS data for all countries. Paper only mentions India exclusion but the per-country coverage varies.
- Item 24 is an edge case: 4 refusals + 1 unparseable (is_refusal=False, parsed_value=NaN). Total "no usable values" items = 3 (items 24, 38, 53), but only 2 are "refused on all runs" by is_refusal flag. Paper uses "yielded no usable values" language to cover both.

## Canonical Data Files (do not use old/partial runs)

- Format A (with system, 5 runs): `responses_20260320_192128.jsonl`
- Format B (fixed prompts, 1 run): `responses_20260320_192138.jsonl`
- No-system ablation (5 runs): `responses_20260321_023446.jsonl`
- Haiku Format A (5 runs): `responses_20260323_171056.jsonl`
- Haiku Format B (1 run): `responses_20260323_171101.jsonl`
- Old/partial runs (DO NOT USE): `responses_20260320_150946.jsonl`, `responses_20260320_190652.jsonl`

## 6 Value Domains

moral_justifiability, gender_family, religion_values, tolerance_neighbors, political_authority, economic_values

## 12 Target Countries (Format B)

Sweden, Germany, United States, Australia, France, Spain, Japan, South Korea, India, Bangladesh, Nigeria, Egypt

## Model

Currently testing: `claude-sonnet-4-20250514` (configurable in api_runner.py MODEL constant)

## Project Context

- Tests whether Claude's Constitutional AI embeds WEIRD cultural values
- Builds on: Atari et al. "Which Humans?", Durmus et al. GlobalOpinionQA, Anthropic Collective Constitutional AI
- Key insight from Collective CAI paper: they acknowledged sample was "not globally representative" and called for evals testing constitutional faithfulness — this project is that eval
- Reference papers in references/ (gitignored, local only):
  - `which_humans_09222023.pdf` — Atari et al., GPT clusters with WEIRD populations
  - `2506.21606v1.pdf` — Pourdavood (your preprint), LLMs as cultural instruments
  - `Collective Constitutional AI...pdf` — Anthropic, 1000 Americans wrote a constitution
  - `Values in the wild...pdf` — Anthropic, 700K conversations analyzed for expressed values
  - `Tao et al. (2024).pdf` — PNAS Nexus, closest prior work: WVS on GPT across 112 countries. We differentiate via: Claude/CAI (not GPT), advice-seeking format, rhetorical strategy coding, 55 items (not 10)

## Post-Collection Checklist

- Sanity-check ~30 Format A raw responses manually (verify parser accuracy)
- Check Format A refusal rate — if >20%, handle missing data in cultural map
- Run Format B LLM coder (~663 calls, ~$3) — needed for steerability + strategy analyses
- ✅ Spot-check 40 Format B coded responses manually — DONE. Weighted κ=0.93 (values), κ=0.96 (strategies). Validation spreadsheet: `results/validation/format_b_validation_sample.xlsx`
- Pull 10-15 qualitative Format B examples across countries for the same item (e.g., homosexuality)
- Verify PCA cultural map reproduces recognizable Inglehart-Welzel clusters before interpreting
- Run significance tests (Wilcoxon) on steerability shifts

## Figure Design Rules

- Claude is ALWAYS black (#1a1a1a) across all figures — never red (conflicts with Cluster 1)
- Cultural map uses UMAP projection with k-means (k=4) clusters computed in HIGH-DIMENSIONAL space, not 2D
- **CRITICAL**: `figures.py` must use `umap.UMAP` for Figure 1, NOT PCA. The paper describes UMAP throughout. If you see PCA in the cultural map code, it's a regression — fix it before regenerating.
- Cluster colors are data-driven, NOT predefined Inglehart-Welzel zones — BOTH country labels AND ellipses use k-means cluster colors (not IW zone colors)
- No legend box on cultural map — colors speak for themselves (matches Atari et al. style)
- x-axis labels: allow 16+ chars before truncating, use 55° rotation, leave bottom margin
- All figures: white background, no top/right spines, tight_layout with bbox_inches='tight'
- **`figures.py` generates canonical paper figures** (fig1–fig5). `analysis.py` generates exploratory figures (cultural_map_mds.png, etc.) that are NOT used in the paper. Don't confuse the two.
- Copy final figures to paper/ directory after generating
- When changing figure methodology (e.g., PCA→UMAP), update BOTH the figure code AND the paper text (methods section + caption). Check with grep for all mentions.
## Figure Regeneration Workflow

- Regenerate a single figure: `python3 -c "from src.figures import fig_cultural_map, load_data; cm, fa, items = load_data(); fig_cultural_map(fa, cm)"`
- `load_data()` returns `(country_means, format_a, items)` — not format_a first
- After regenerating, MUST copy to paper/: `cp results/figures/fig1_cultural_map.png paper/`
- Then recompile: `cd paper && tectonic main.tex`
- Regenerate all: `python3 -c "from src.figures import generate_all; generate_all()"`

- UMAP compresses high-dimensional distances — Claude's "beyond all nations" extremity (31/47 items) is NOT visually obvious in the 2D projection. Don't try to force it with convex hulls or annotations. Figure 4 (distributions) and Table 3 (bootstrap CIs) communicate extremity better.
- NeurIPS `[preprint]` option adds "Preprint." to first page footer. To remove: `\makeatletter \renewcommand{\@noticestring}{} \makeatother` after `\usepackage[preprint]{neurips_2025}`
- **Before regenerating figures**: verify `figures.py` matches paper claims (UMAP not PCA, black not red for Claude, k-means not zone-based ellipses). Running `generate_all()` overwrites paper figures — wrong code = wrong figures in paper.

## Key Findings (from completed analyses)

- Claude is beyond all 89 countries (Montenegro excluded) on 31/47 items (bootstrap 95% CI confirmed)
- 40/47 items: zero variance across 5 runs at temperature=1
- Cultural clustering: nearest to Germany (r=0.861), Netherlands, NZ, Great Britain (89 countries in correlations; Montenegro excluded due to insufficient item overlap)
- JSD validates Pearson ranking (Spearman rho=0.939)
- Steerability: all paired Cohen's d negligible (mean |d|=0.108, range -0.174 to +0.089). Country context has ~zero effect.
- When Claude does shift, it moves toward the named country's values (83/138 changed pairs, p=0.011 one-sided binomial), but 75% of the time (407/545 pairs) there's no change
- Strategy: BALANCED_LEAN 61%, DIRECTIVE 30%, PURE_BALANCE 5%, DEFERRAL 3%. Zero coding errors with fixed prompts.
- No-system ablation: 44.4% refusal rate (vs 14.9%), 22 items refused on all runs, but when answering r=0.98 with system version
- IW dimensions (z-scored): Claude TS=0.438, SS=1.881; nearest to Denmark (dist=0.354), at extreme self-expression end
- Gap statistic: monotonically increasing gap, criterion triggers at k=8 — no strong clustering structure (cultural map is a continuum)

## Paper Framing Notes

- Item selection by cross-country variance is INTENTIONAL — frame as "these are the items where cultural bias is most detectable and most consequential for users." Acknowledge as scope limitation.
- "Produces outputs that correlate with" NOT "holds values" — avoid anthropomorphism
- The headline is: "Constitutional AI doesn't just make Claude WEIRD — it creates a value profile beyond all human populations"
- Ablation tells a clean story: system prompt grants permission to express views, not different views
- Format B single-run is defensible: Tao et al. also used 1 response, and Format A shows near-zero variance anyway

## Audit Protocol

Run full pipeline audit before any writing or publishing. Checks: file existence, row counts, cross-join detection, parse method distribution, value range validation, response length vs refusal, prompt-response alignment, semantic spot checks. Must show: 0 errors, 0 warnings before proceeding to paper.

## Review Fixes (2026-03-29)

All 20 issues from comprehensive review have been addressed:

### Code fixes (analysis.py)
1. **Gap statistic crash bug**: `.to_frame().T.rename(index={0: 'CLAUDE'})` silently failed because index was Series name, not `0`. Fixed to `claude_z_df.index = ['CLAUDE']`.
2. **Gap statistic reproducibility**: Added `rng = np.random.default_rng(seed=42)` for Monte Carlo reference distributions.
3. **JSD docstring**: Documented that `scipy.spatial.distance.jensenshannon` returns JS *distance* (sqrt of divergence), log base 2, bounded [0, 1] not [0, ln(2)].

### Paper fixes (main.tex)
4. **No-system refusal rate**: Section 5.5 said "40%", Section 4.5 said "~43%". Initially fixed to "approximately 43%"; later corrected to 44.4% (see Round 2 fix #4).
5. **90→89 countries**: Fixed in abstract, Figure 1 caption, Section 4.2, Table 3 caption, conclusion, and appendix. Montenegro excluded due to insufficient item overlap is now consistently noted.
6. **Variance description**: Changed "6 showed minimal fluctuation (sigma ~0.45)" to "5 showed minimal fluctuation (sigma ~0.45–0.50)" plus euthanasia (sigma=0.89, two-point difference).
7. **Northern Ireland**: Added to correlation ranking between Great Britain (r=0.816) and Sweden (r=0.800) at r=0.801.
8. **Haiku zero-variance denominator**: Changed "54 items, 45 (83%)" to "53 items with multiple runs, 45 (85%)" since variance is undefined for single-response items.
9. **LLM-as-judge validation**: Strengthened limitations to explicitly call out lack of human inter-rater reliability as the most significant methodological limitation.
10. **UMAP vs correlation discrepancy**: Added note explaining why UMAP neighbors differ from correlation rankings (local manifold structure vs pairwise correlations).
11. **Constitutional value floor claims**: Softened causal language — "embedded in the weights" → "appear to be embedded...though pretraining data and RLHF may also contribute." Added caveat that ablation removes system prompt, not constitutional RLHF.
12. **Bootstrap CIs**: Added note that 40/47 items have degenerate CIs of [x, x] and extremity is established by point estimate, not the CI.
13. **Gap statistic**: Qualified as "exploratory check", noted it is designed for point clouds and sensitivity to a single entity is limited.
14. **Item 22 false positive**: Added footnote in methods about the known Heaven false-positive parse (<0.4% of responses, excluded from bootstrap).
15. **Format B coding reliability**: Added note that coding variance across multiple passes was not assessed.
16. **India inclusion**: Explained rationale (South Asian representation in strategy analysis) and why it's excluded from directional comparisons.
17. **Scale ceiling effects**: Added to limitations — endpoint distances are qualitatively different, and Claude may prefer endpoints due to lack of ambivalence.
18. **Shmitchell citation**: Fixed "Shmitchell, Shmargaret" → "Mitchell, Margaret" in references.bib.
19. **Haiku body text**: Added sentence confirming Haiku exhibits same beyond-human extremity pattern, matching abstract claim.
20. **Steerability comparison**: Added caveat that Tao et al. comparison is confounded by model family, item set, and prompting strategy differences.

### Human validation (format_b_validation_sample.xlsx)
21. **LLM-as-judge validated**: 40 stratified samples (20 BALANCED_LEAN, 10 DIRECTIVE, 5 PURE_BALANCE, 5 DEFERRAL) coded by human. Results: weighted κ=0.93 (linear) for values (r=0.98, MAD=0.175, 82.5% exact), unweighted κ=0.96 for strategies (97.5% exact). All value disagreements ≤1 scale point. Only 1 strategy disagreement (Sample 5: BALANCED_LEAN vs DIRECTIVE boundary). Paper updated with results in methods, limitations, and future directions sections.

### Documentation (CLAUDE.md)
- Item 20 ("Politics - important in life") is classified under `religion_values` due to broad keyword matching on "important it is in your life" in item_selection.py. This is a domain classification quirk, not a bug — all "importance in life" items grouped together.

## Review Fixes Round 2 (2026-03-29)

### Critical fixes
1. **LLM coder missing advice prompt**: `api_runner.py` doesn't save `user_prompt` to JSONL, so the Format B coder always gets `'(not available)'` for the advice prompt. Paper updated to disclose this and frame κ=0.93 as a conservative estimate. Not re-running coder to avoid invalidating existing human validation.
2. **Causal overclaim**: "doing exactly what its constitution instructs" → "outputs are consistent with what its constitution instructs". Also hedged steerability comparison with Tao et al. to note confounds explicitly.
3. **Bootstrap reproducibility**: Added `rng = np.random.default_rng(seed=42)` to `bootstrap_extremity_analysis` — the 31/47 headline result was previously non-reproducible.
4. **Ablation numbers corrected**: Refusal rate was 44.4% (not ~43%), 22 items refused on all runs (not 21). Fixed in paper (both Section 4.5 and 5.5) and CLAUDE.md.

### Code fixes (analysis.py)
5. **Cohen's d formula**: Changed from equal-n formula `sqrt((s1²+s2²)/2)` to weighted pooled SD `sqrt(((n1-1)*s1²+(n2-1)*s2²)/(n1+n2-2))`. Practical impact negligible (mean |d|=0.032).
6. **IW dimension standardization**: `compute_dimension_score` was averaging raw values across mixed scales (1-4, 1-10, 1-11). Now z-scores across countries first. IW scores changed (later corrected again in Round 4 due to keyword matching bug).
7. **Zone name consistency**: Standardized to `'African-Islamic'` across all files (was `'Africa-Islamic'` in analysis.py and prompt_generation.py, `'African-Islamic'` in figures.py).

### Code fixes (figures.py)
8. **fillna consistency**: Changed from pre-drop mean to post-drop mean, matching analysis.py.
9. **Distribution y-axis**: Changed hardcoded `ylim(0, 0.75)` to `max(0.75, actual_max * 1.05)` to prevent clipping.

### Code fixes (response_parser.py, prompt_generation.py)
10. **SKIP_OPTIONS consistency**: Added missing `"Missing; Not asked in survey"` and `"DK/Refused"` to prompt_generation.py and response_parser.py (already present in item_selection.py and analysis.py).
11. **Word boundary matching**: Short options (≤3 chars like "No", "Yes") now use `\b` regex word boundaries to prevent matching inside words like "not", "nothing", "know".

### Paper fixes (main.tex)
12. **Copyright footer**: "The authors" → "The author" (single-author paper).
13. **Steerability denominator**: Added exact n to paper: "83/138 = 60% toward" from 545 total item-country pairs.
14. **"52 common items" clarified**: → "52 items where both models produced at least one valid response".
15. **"Progressive" defined**: Added parenthetical "in the WVS sense of occupying the self-expression and secular-rational poles".
16. **Anthropomorphism**: "Claude distinguishes between items" → "Claude's training produces different behavioral responses depending on whether an item involves...".
17. **Recursive claim hedged**: "The recursive nature of this process compounds the effect" → "If this dynamic holds, the effects could compound".
18. **Constitution citation**: Added `\citep{bai2022constitutional}` to Table 1 caption.
19. **India rationale**: Expanded to mention both steerability and strategy analyses.
20. **Format B endpoint avoidance**: Added to limitations — LLM coder never assigns values above 9 on 10-point scales.

### Documentation (CLAUDE.md)
- Added Haiku canonical file paths to Canonical Data Files section.
- Updated ablation refusal rate from ~43% to 44.4%, 21→22 items.

### Post-rerun updates (pipeline re-run completed)
- **All code fixes verified**: Re-ran `analysis.py` and `figures.py`, copied figures to `paper/`, recompiled PDF.
- **Gap statistic changed**: k=1 → k=8 (monotonically increasing gap, no elbow). Paper updated to describe the monotonic curve rather than claiming k=1.
- **Within-condition σ changed**: 2.20 → 2.15. Paper updated.
- **Gap statistic int bug fixed**: `optimal_k` was returned as float from pandas, causing `KMeans(n_clusters=8.0)` crash. Added `int()` cast.
- **All other numbers unchanged**: correlations, bootstrap (31/47), binomial (p=0.011), ablation (r=0.98), steerability direction (83/138).
- **Existing data unaffected**: SKIP_OPTIONS fix doesn't change existing parsed data (the missing options don't appear in the 55 selected items), but prevents future bugs if items change.

## Review Fixes Round 3 (2026-03-29)

Three issues found by follow-up review session:

1. **60x claim dimensionally wrong**: Paper divided σ (2.15, raw units) by mean |d| (0.032, standardized). Fixed to compare σ vs mean raw shift (0.070): ratio is ~30x, not 60x.
2. **Cohen's d pooled across items**: Was computing a single pooled baseline mean/SD across all 55 items, so between-item variance dominated the denominator and yielded artificially small d. Fixed to paired Cohen's d (per-item difference, then mean(diffs)/SD(diffs)). New values: mean |d|=0.108, range -0.174 to +0.089. Still all negligible (<0.2). Paper updated.
3. **SKIP_OPTIONS still missing in 2 locations**: `figures.py` `fig_distributions()` and `analysis.py` `distribution_overlap_plots()` were missing `"Missing; Not asked in survey"` and `"DK/Refused"`. Now consistent across all files.

Pipeline re-run verified, figures regenerated, paper recompiled and pushed.

## Review Fixes Round 4 (2026-03-29)

Six issues found by comprehensive automated review:

### Code fixes (analysis.py)
1. **IW dimension keyword matching bug**: `compute_dimension_score()` used substring matching (`keyword.lower() in topic.lower()`), causing `'Religion'` to match item 17 ("The only acceptable religion is my religion") instead of item 19 ("Religion"). Also `'Obey rulers'` never matched item 35 ("People obey their rulers") because it's not a substring. Fixed to exact-match-first with substring fallback, and changed keyword to `'People obey their rulers'`. IW scores changed: Claude TS=0.438, SS=1.881 (was TS=-0.701); nearest Denmark dist=0.354 (was Netherlands dist=0.311).
2. **Non-deterministic `common_items` ordering**: Two instances of `list(set(...))` in `cultural_clustering()` and `gap_statistic_analysis()` changed to `sorted(set(...))` for reproducibility across Python runs. `figures.py` already used `sorted()`.

### Paper fixes (main.tex)
3. **Figure 2 caption wrong coloring description**: Caption said "Green bars indicate r > 0.5; red bars indicate r < 0; yellow bars are intermediate" but the publication figure (`figures.py`) uses Inglehart-Welzel zone coloring. Fixed to describe zone-based coloring.
4. **Correlation CIs missing**: Added 95% Fisher z-transform CIs to key correlations (Germany r=0.861 [0.769, 0.918], bottom three, and US). Added note about FDR significance and CI methodology.
5. **545 vs 605 item-country pairs unexplained**: Added explanation that per-country WVS coverage varies (39–55 items), reducing total from theoretical 605 to 545.
6. **Binomial test independence caveat**: Added note that item-country pairs are not independent (same item across countries often gets identical Claude response), so effective sample size may be smaller than 138 and true p-value correspondingly higher.

### Documentation (CLAUDE.md)
- Updated IW dimension values from TS=-0.701 to TS=0.438, nearest country from Netherlands to Denmark.

Pipeline re-run verified: all other numbers unchanged (correlations, bootstrap 31/47, binomial 83/138 p=0.011, Cohen's d mean |d|=0.108, strategy percentages, ablation r=0.98). Figures regenerated, paper recompiled.

## Review Fixes Round 5 (2026-03-29)

Eight issues identified by comprehensive review; #7 was a false alarm (JSD already uses individual runs, not means).

### Paper fixes (main.tex)
1. **IW dimension scores added to paper**: New paragraph in Section 4.1 reports Claude's IW position (TS=0.438, SS=1.881), nearest countries (Denmark 0.354, Netherlands 0.452, Norway 0.530), and the key finding that Claude is beyond all 90 countries on the Self-Expression axis.
2. **Strategy percentages fixed**: Changed from rounded integers (61%, 30%, 5%, 3% = 99%) to one decimal place (61.4%, 29.8%, 5.3%, 3.5% = 100%).
3. **Conclusion country count clarified**: "all 90 surveyed nations" → "all surveyed nations (89 with sufficient data for correlation analysis, all 90 for per-item bootstrap comparisons)".

### Code fixes (analysis.py)
4. **Gap statistic sample SD**: Changed `np.std(ref_log_wks)` (ddof=0) to `np.std(ref_log_wks, ddof=1)` per Tibshirani 2001. Practical impact negligible (~0.5% change in sk with n_refs=100). Optimal k unchanged at 8.
5. **strategy_by_country figure quality**: Added white background (`facecolor='white'`), increased DPI to 300, added `bbox_inches='tight'` to match publication figure standards.
6. **Domain analysis NaN warning**: Added explicit count of items dropped by `dropna()` in `domain_analysis()`.

### False alarm
7. **JSD multi-run variance**: Initially flagged as collapsing mean to single bin, but the code already iterates over individual `claude_responses` values (not means), correctly distributing weight. No fix needed.

Pipeline re-run verified: all numbers unchanged (gap statistic k=8, bootstrap 31/47, correlations, binomial, Cohen's d). Figures regenerated, paper recompiled.

### Additional paper edits (post Round 5)
8. **IW dimension table added**: Table 5 (`tab:iw_dimensions`) shows Claude alongside 5 nearest countries on IW dimensions (all Protestant Europe). Replaces inline-only numbers.
9. **Self-citation cleanup**: Removed two self-citations from intro and discussion (`pourdavood2025llms` tangential uses). One substantive citation remains in Related Work.
10. **Discussion restructured (10 → 7 subsections)**:
    - Merged "Compounding Risk" + "Disentangling Safety" + "Comparison with Prior Findings" → "Comparison with Prior Work and the Compounding Risk"
    - Merged "Scope of Claims" + "Broader Impact" → "Scope and Broader Impact"
    - Discussion subsections now: Constitutional Value Floor, Implications for Culturally Diverse Users, Comparison with Prior Work and the Compounding Risk, What Refusals Reveal, Limitations, Future Directions, Scope and Broader Impact

## Deliverables

1. GitHub repo (this)
2. Medium post (blog/)
3. arXiv preprint (paper/)
