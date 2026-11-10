# CrediStance — Implementation Guide

This document explains the code in `credistance/`, how the experiments are run, and how every number in the report is produced. Everything runs on one GPU; all credal, conformal and decision computations run on CPU from cached model outputs.

## 1. Repository layout

```
major_project/
├── credistance/
│   ├── prepare_data.py       # raw datasets + SemEval raw votes -> unified JSONL + human label distributions
│   ├── train.py              # trains ONE ensemble member of any type; caches outputs on every eval set
│   ├── credal.py             # credal core: bounds, entropies, EffCre, conformal, metrics (NumPy only)
│   ├── make_jobs.py          # writes jobs.txt (the whole experiment grid)
│   ├── run_jobs.sh           # runs jobs.txt with N parallel workers, skips finished jobs
│   ├── llm_prompt_credal.py  # zero-shot prompt-ensemble credal sets with an instruction LLM
│   └── analysis.py           # builds every method from cached outputs; all metrics, tables, figures
├── data/
│   ├── raw/                  # unzipped SemEval-2016, P-Stance, COVID-19-Stance
│   ├── semeval_raw_annotations/   # SemEval2016-Task6-raw-annotations-stance.csv (per-annotator votes)
│   └── processed/            # unified JSONL splits (see §2)
├── runs/<experiment>/<method>_s<seed>/<evalset>.npz   # cached logits / intervals per member
├── results/results.json, results/tables.md, results/fig/*.png
└── logs/
```

## 2. Data preparation (`prepare_data.py`)

**Unified record** (one JSON per line):

```json
{"id": "semeval-10001", "source": "semeval", "text": "...", "target": "Atheism",
 "label": 1, "split": "test", "label_dist": [0.125, 0.875, 0.0], "n_votes": 8, "human_entropy": 0.54}
```

Labels: `FAVOR=0, AGAINST=1, NONE=2` (SemEval's raw `NEUTRAL` answer is mapped to `NONE`, as in the official label derivation).

**Human label distributions.** The SemEval-2016 organisers released every crowd worker's stance label (`Worker ID, Instance ID, Target, Tweet, Stance, Opinion towards`). For each tweet with votes we compute

```
λ_x  = (n_FAVOR, n_AGAINST, n_NONE) / n
H(λ_x) = -Σ_k λ_xk log2 λ_xk            # 0 … 1.585 bits
```

Rows with an empty `Instance ID` are tweets the organisers dropped from the shared task because annotator agreement was below 60%. They become the **ambiguity stress set** (`semeval_ambig.jsonl`, 1,385 tweets, mean entropy 1.445 bits). They were never part of any training set.

| File | Items | With votes | Role |
|---|---|---|---|
| `semeval_train.jsonl` | 2,250 | 1,934 | member training (80% of official train, stratified by target) |
| `semeval_cal.jsonl` | 564 | 485 | conformal calibration, temperature scaling, EffCre likelihood reference; **never trained on** |
| `semeval_test.jsonl` | 1,249 | 1,080 | official test set |
| `semeval_ambig.jsonl` | 1,385 | 1,385 | low-agreement pool (4 seen targets + Donald Trump) |
| `pstance_test.jsonl` | 2,157 | – | cross-dataset unseen targets (Trump/Biden/Sanders, binary) |
| `covid_test.jsonl` | 800 | – | cross-dataset unseen targets (4 COVID targets) |

Pre-processing: strip `#SemST`, replace URLs with `HTTPURL` and mentions with `@USER`, collapse whitespace. The climate-change target has no raw votes (annotated internally by the organisers), so those items carry hard labels only.

## 3. Training one member (`train.py`)

Backbone: `microsoft/deberta-v3-base`, `[CLS]` pooling, dropout 0.1, linear head. Input is the sentence pair `(target, text)` so the target is never truncated. AdamW, lr 2e-5, weight decay 0.01, 10% linear warm-up, 5 epochs, batch 32, max length 128, bf16 autocast with **fp32 master weights** (`dtype=torch.float32` at load; transformers ≥5 otherwise loads this checkpoint in fp16 and fine-tuning silently produces NaN).

One script trains every member type; the only difference is the loss:

| `--method` | Loss on a batch of size B | Role |
|---|---|---|
| `ce` | mean cross-entropy on hard labels | deep-ensemble member → Credal Wrapper |
| `soft` | mean cross-entropy against `label_dist` where it exists (one-hot otherwise) | label-distribution learning |
| `dro` | CE on the **top ⌊δ·B⌋ hardest** samples of the batch (CVaR-DRO, closed form) | CreDRO member; members differ in δ |
| `softdro` | same top-δ rule, but against soft human distributions | **proposed combination** |
| `crenet` | interval head (2K outputs → midpoint m, half-length h = softplus) → interval softmax → `CE(q_U) + top-δ CE(q_L)` | Credal Deep Ensemble member |
| `edl` | evidential (Dirichlet) loss of Sensoy et al. with KL annealing | baseline |

Diversity for every ensemble: seed *i* and an 80% bootstrap sample of the training set per member. For `dro`/`softdro` the DRO fraction is spread uniformly over the members, δ_i = 0.5 + 0.5·i/(M−1), as in CreDRO.

**Interval softmax** (CreNet, numerically stable form used here):

```
q_L,i = sigmoid(a_L,i − logsumexp_{k≠i} m_k),   q_U,i = sigmoid(a_U,i − logsumexp_{k≠i} m_k)
```

which satisfies q_L ≤ q_U and Σ q_L ≤ 1 ≤ Σ q_U.

Every run saves, for each evaluation set, `logits` (N,K) (or `qL,qU` for CreNet, `alpha` for EDL), the labels, the targets and, for the seed-0 CE member, 10 MC-dropout samples `mc_logits` (T,N,K).

Leave-one-target-out: `--exclude_target "<target>"` removes a target from training; the same five evaluation sets are scored so seen/unseen items are separated at analysis time.

## 4. Credal core (`credal.py`)

All functions take NumPy arrays; `(M,N,K)` = members × items × classes.

| Function | What it computes |
|---|---|
| `credal_bounds(P)` | `l_k = min_m p_mk`, `u_k = max_m p_mk` (Credal Wrapper) |
| `reachable(l,u)` | tightens to attainable bounds: `u_k ← min(u_k, 1−Σ_{j≠k} l_j)`, `l_k ← max(l_k, 1−Σ_{j≠k} u_j)` |
| `intersection_probability(l,u)` | `p* = l + α(u−l)`, `α = (1−Σl)/Σ(u−l)` — the point prediction |
| `upper_entropy(l,u)` | `max_{p∈C} H(p)`: water-filling `p_k = clip(τ,l_k,u_k)`, τ by bisection |
| `lower_entropy(l,u)` | `min_{p∈C} H(p)`: entropy is concave ⇒ minimum at a vertex; all ≤12 vertices enumerated |
| `decompose(l,u)` | `TU = H̄`, `AU = H̲`, `EU = H̄ − H̲`, width `Σ(u−l)` |
| `nondominated(l,u)` | interval dominance: keep class k iff `u_k ≥ max_j l_j` |
| `bayesian_decomposition(P)` | ensemble baseline: `TU = H(p̄)`, `AU = mean H(p_m)`, `EU = mutual information` |
| `effcre_bounds(train_logits, y, test_logits, α)` | EffCre 1-D variant: per class, shift logits by `t·e_k`; feasible `t` keep the (held-out) log-likelihood within `log α` of its maximum; endpoints by bisection; bounds = `p_k(x; t_k^±)` |
| `conformal_credal(g_cal, λ_cal, g_test, α)` | conformalized credal set: score `TV(λ, g)`, quantile `q̂`, TV-ball bounds `[g−q̂, g+q̂]` |
| `credal_conformal(l,u,λ…)` | our variant: score = distance of λ to the predicted box (0 if inside); calibrated set = box dilated by `q̂` |
| metrics | `ece`, `nll`, `brier`, `macro_f1`, `f_avg` (SemEval official), `aurc`, `auroc`, `u65`, `u80`, `coverage_tv`, `temperature_scale` |

## 5. Methods assembled in `analysis.py`

| Name | Built from | Point prediction | TU / AU / EU |
|---|---|---|---|
| Softmax | `ce_s0` | softmax | entropy / entropy / 1−max p |
| TempScale | `ce_s0` + T fitted on cal | softmax(z/T) | same |
| MCD-MI | 10 dropout samples of `ce_s0` | mean | Bayesian decomposition |
| MCD-Credal | same samples | intersection prob. | credal decomposition |
| EDL | `edl_s0` | α/S | entropy / entropy / vacuity K/S |
| DE-MI | `ce_s0..4` | mean | Bayesian decomposition |
| CreWrap-DE | `ce_s0..4` | intersection prob. | credal decomposition |
| EffCre(α) | `ce_s0` only + cal logits | intersection prob. | credal decomposition |
| CreDE | `crenet_s0..4`, averaged reachable intervals | intersection prob. | credal decomposition |
| CreWrap-Soft | `soft_s0..4` | intersection prob. | credal decomposition |
| CreDRO | `dro_s0..4` (δ spread) | intersection prob. | credal decomposition |
| **CreDRO-Soft** (proposed) | `softdro_s0..4` | intersection prob. | credal decomposition |
| Prompt-Credal (LLM) | 8 prompt paraphrases of Qwen2.5-3B-Instruct | intersection prob. | credal decomposition |

## 6. Experiments and where each number comes from

| Exp. | Question | Data | Code path |
|---|---|---|---|
| E1 | in-target accuracy, calibration, selective prediction, set-valued quality | `runs/intarget`, SemEval test | `point_metrics` |
| E4a | does EU rise on an unseen target? | `runs/loto_*`, SemEval test split into seen/unseen by the held-out target | `auroc(EU, unseen)` per fold, mean ± std over 5 folds |
| E4b | cross-dataset unseen targets | in-target models scored on P-Stance / COVID test vs SemEval test | `auroc(EU, is_other_dataset)` |
| E5 | does AU track human disagreement? | SemEval test items with votes | Spearman ρ(AU, H), ρ(EU, H), JSD(p*, λ) |
| E5b | ambiguity stress test | 1,101 low-agreement tweets on the 4 seen targets vs test | `auroc(AU, is_ambig)`, mean AU/EU |
| E6 | conformal coverage of λ | cal ∪ test voted items, 100 random 50/50 re-splits | `conformal_eval` |
| E7 | decision cost | SemEval test | `cost_curve` (c_err=1, c_defer=0.3, c_set=0.15/extra class) |
| E8 | LLM prompt-credal | `runs/llm_qwen3b` | same metrics |

## 7. Running everything

```bash
python3 credistance/prepare_data.py              # ~5 s
python3 credistance/make_jobs.py 5               # M = 5 members -> jobs.txt (151 jobs)
bash credistance/run_jobs.sh 3                   # 3 parallel workers on one GPU (~45 s per job)
python3 credistance/llm_prompt_credal.py --model Qwen/Qwen2.5-3B-Instruct --prompts 8
python3 credistance/analysis.py                  # results/results.json, results/tables.md, results/fig/
```

Re-running `run_jobs.sh` skips jobs whose `config.json` exists, so the grid can be resumed. Changing `M` in `make_jobs.py` and `members()` in `analysis.py` is all that is needed for larger ensembles.

## 8. Design decisions worth knowing

- **Calibration set is held out from training.** Temperature scaling, EffCre's likelihood reference and conformal quantiles all use `semeval_cal`, which no member ever sees, so the conformal guarantee is not contaminated by training.
- **Same members, different wrappers.** `DE-MI` and `CreWrap-DE` are built from identical trained networks; any difference between them is purely the credal representation.
- **EffCre uses the held-out likelihood.** The original method uses the training likelihood. A fine-tuned transformer fits its training set almost perfectly, which makes that likelihood extremely peaked; we therefore evaluate the relative likelihood on the calibration split and state this deviation.
- **Reachable bounds before any uncertainty computation.** Interval bounds from min/max are not necessarily attainable in the simplex; `reachable()` is applied before entropies, dominance and the intersection probability.
- **The ambiguity set excludes Donald Trump.** The low-agreement pool contains 284 Trump tweets; Trump was never a training target, so those would confound ambiguity with target novelty. The E5b comparison uses the 1,101 tweets on the four seen targets.
- **P-Stance is binary.** When scoring P-Stance accuracy the `NONE` column is dropped before the argmax; uncertainty scores use the full 3-class credal set.

## 9. Additional analysis script (`ablation_stats.py`)

Ensemble-size ablation (first *M* members of each in-target ensemble), 1,000-resample bootstrap confidence intervals on the in-target test set, paired *t* / Wilcoxon tests of unseen-target AUROC across the five LOTO folds, the per-fold table, and the AU-vs-width map figure. Output: `results/ablation_stats.md`, `results/fig/ensemble_size.png`, `results/fig/au_eu_map.png`.

## 10. Where the results live

| File | Content |
|---|---|
| `results/results.json` | every metric for every method, experiment and fold |
| `results/tables.md` | E1, E4a, E4b, E5, E5b, E6, E7 tables (auto-generated) |
| `results/ablation_stats.md` | ensemble-size ablation, bootstrap CIs, paired fold tests |
| `results/fig/*.png` | figures used in the report |
| `FINAL_REPORT.md` / `CrediStance_Final_Report.tex` | the written report (Markdown and LaTeX) |

Two pipeline issues found and fixed during the run, recorded here so they are not rediscovered: (1) transformers ≥ 5 loads `deberta-v3-base` in fp16 → NaN fine-tuning unless `dtype=torch.float32` is passed; (2) `xargs` strips quotes, so multi-word `--exclude_target` values need `xargs -d '\n'` in `run_jobs.sh` (the first LOTO pass silently trained on the full data for four of five folds and had to be re-run).

## 11. Diagrams

The architecture and concept diagrams in the reports (`results/fig/arch/*.png`) are rendered from the SVG React components in `docs/diagrams.jsx`, which also power the presentation at https://rakeshtirlangi.github.io/Credi-Stance/. To regenerate them, serve `docs/` locally and screenshot `figure.html`:

```bash
cd docs && python3 -m http.server 8124 &
google-chrome --headless=new --hide-scrollbars --force-device-scale-factor=2 --window-size=1580,600 \
  --screenshot=../results/fig/arch/pipeline.png "http://localhost:8124/figure.html?d=Pipeline&theme=light"
```

Available `d` values: SameVector, CredalIntro, AmbigVsIgnor, DecompDiag, ClassicalVsModern, Pipeline, CreDROSoftDiagram, Dominance, ConformalDiag, LOTO, EUPathology, Members. `theme=dark` gives the deck colours.
