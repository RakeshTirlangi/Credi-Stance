# Credal Uncertainty Quantification for Reliable Stance Detection on Unseen Targets

**Final project report — problem definition, alternative credal architectures, proposed method, experiments and results**

---

## Abstract

Stance detection assigns FAVOR, AGAINST or NONE to a (text, target) pair. Every deployed stance detector outputs one probability vector, which cannot tell *ambiguity* (the text is contested and humans disagree) from *ignorance* (the model has never seen this target). We study **credal sets** — convex sets of probability distributions, summarised as lower and upper class probabilities — as the output representation for stance detection, and compare six ways of building them: post-hoc credal wrapping of deep ensembles and MC-dropout, Credal Deep Ensembles with interval heads, distributionally-robust credal ensembles (CreDRO), single-model credal sets through likelihood decalibration (EffCre), and prompt-paraphrase credal sets from instruction-tuned LLMs. We propose a new combination for stance, **CreDRO-Soft**: distributionally-robust members trained on *human label distributions* recovered from the publicly released per-annotator votes of SemEval-2016, so that the aleatoric part of the credal set is grounded in real human disagreement while the epistemic part is driven by distributional robustness. No new annotation is collected. On SemEval-2016 (in-target, five leave-one-target-out folds), P-Stance and COVID-19-Stance (cross-dataset unseen targets), 1,101 low-agreement tweets excluded from the original shared task, and 100 conformal re-splits, we report accuracy, calibration, selective-prediction risk, unseen-target detection, alignment with human disagreement, conformal coverage of the human label distribution, and decision cost. Results tables are reproduced exactly from `results/tables.md`.

---

## 1. Problem definition

### 1.1 Task

Given a text $x$ and a target $t$, predict $y \in \mathcal{Y} = \{\textsc{favor}, \textsc{against}, \textsc{none}\}$ [Mohammad et al., 2016; Küçük & Can, 2020].

### 1.2 What is wrong with one probability vector

A classifier returns $p(y \mid x, t) \in \Delta_3$. Two very different situations produce the same vector:

| Situation | Example | Right action |
|---|---|---|
| **Ambiguity (aleatoric)** — the text is contested; annotators disagree | *"Oh great, another 'climate emergency' summit. I'm sure the private jets will fix it."* | flag as contested; do not count it as a clean vote |
| **Ignorance (epistemic)** — the target/domain is new to the model | a tweet about *Digital euro* when training covered abortion, feminism, atheism | defer to a human; collect labels for the target |

SemEval-2016 itself shows how much ambiguity there is: the organisers dropped 1,385 tweets from the shared task because annotator agreement fell below 60 % [Mohammad et al., 2017]. VAST reports Krippendorff's α of only 0.427 [Allaway & McKeown, 2020]. A single probability vector offers no basis for choosing between "contested" and "unknown".

### 1.3 Problem statement

> Existing stance detectors output a single distribution $p(y\mid x,t)$ that conflates aleatoric and epistemic uncertainty. Consequently they are over-confident on unseen targets and cannot justify abstention. We seek a stance detector that outputs a **credal set** $\mathcal{C}(x,t) \subseteq \Delta_3$, represented by class-wise lower and upper probabilities $[\underline{P}(y), \overline{P}(y)]$, such that (a) the set's *location* reflects ambiguity and its *size* reflects ignorance, (b) the set can be calibrated to cover the true human label distribution with a distribution-free guarantee, and (c) the set supports a predict / set-valued / defer decision. The method must need **no new annotation**, must run on one GPU, and must be evaluated on in-target, cross-target and cross-dataset stance data.

### 1.4 Research questions

| | Question | Hypothesis |
|---|---|---|
| RQ1 | Do credal stance models give better uncertainty estimates than softmax, temperature scaling, MC-dropout, deep ensembles and evidential deep learning at equal accuracy? | better calibration (ECE) and selective risk (AURC) at ±1 pt macro-F1 |
| RQ2 | Does the credal decomposition separate ambiguity from ignorance? | the epistemic score rises on unseen targets; the aleatoric score tracks human label entropy |
| RQ3 | Can credal sets be calibrated to cover the human label distribution? | conformal credal sets attain 1−α coverage; soft-label-trained sets are more efficient |
| RQ4 | Does credal uncertainty improve decisions? | lower expected cost than confidence thresholding at every deferral budget |
| RQ5 | Do LLM prompt-ensemble credal sets beat single-prompt LLM confidence? | lower ECE, higher error-detection AUROC |

---

## 2. Background

### 2.1 Credal sets and the two kinds of uncertainty

A credal set is a closed convex set $\mathcal{C} \subseteq \Delta_K$ of candidate distributions for an imprecisely known ground truth [Walley, 1991]. We use probability-interval credal sets $\mathcal{C}(l,u) = \{p \in \Delta_K : l_k \le p_k \le u_k\}$. With Shannon entropy $H$ in bits, the standard measures are [Abellán et al., 2006; Hüllermeier & Waegeman, 2021]

$$\mathrm{TU} = \overline{H} = \max_{p\in\mathcal{C}} H(p), \qquad \mathrm{AU} = \underline{H} = \min_{p\in\mathcal{C}} H(p), \qquad \mathrm{EU} = \overline{H} - \underline{H}.$$

A *small* set around a spread-out distribution is "confident that the text is ambiguous" (high AU, low EU); a *large* set is "do not know which distribution is right" (high EU). Sale et al. [2023] show that the set's volume is a sound epistemic measure only for binary problems, which is why the entropy decomposition (and, as we will see, interval width) is used for three classes. Gonzalez-Garcia et al. [2026] propose distance-based (total-variation) measures as a multiclass alternative; we report interval width as its simplest proxy.

### 2.2 Classical versus modern credal models — can credal methods handle text?

| | Classical credal models | Modern credal deep learning |
|---|---|---|
| Examples | credal networks [Cozman, 2000], Naive Credal Classifier [Corani & Zaffalon, 2008] | Credal Wrapper [Wang et al., 2025a], Credal Deep Ensembles [Wang et al., 2024], CreINNs [Wang et al., 2025b], conformal credal sets [Javanmardi et al., 2024], CreRL [Löhr et al., 2025], EffCre [Hofman et al., 2026], CreDRO [2026], credal LLMs [Manchingal et al., 2026] |
| Where the set lives | inside a graphical model over discrete variables | at the network output |
| Input | tabular / discrete | anything a neural encoder accepts |

The folk claim that "credal networks only work on tabular data" is true of the classical family, where exact inference limits models to a handful of discrete variables. Modern credal deep learning only needs probability vectors; the encoder handles the raw text. Credal sets have already been evaluated on ChaosNLI, an NLI benchmark with ~100 annotators per item [Javanmardi et al., 2024; Hofman et al., 2026], and on LLMs via LoRA ensembles [Manchingal et al., 2026]. Stance detection has the same structure as NLI (a text pair, three classes, high human disagreement), and EZ-Stance even reformulates zero-shot stance as NLI [Zhao & Caragea, 2024].

---

## 3. Alternative credal architectures considered

We compared six families for building the credal set, all implemented on the same DeBERTa-v3-base backbone so that only the credal construction differs.

### A1. Post-hoc credal wrapper over a deep ensemble (CreWrap-DE) [Wang et al., 2025a]
Train $M$ standard members (seed + 80 % bootstrap). $l_k = \min_m p_{m,k}$, $u_k = \max_m p_{m,k}$. Point prediction by intersection probability $p^*_k = l_k + \alpha(u_k-l_k)$, $\alpha = (1-\sum l)/\sum(u-l)$. Plug-and-play; cost $M$ trainings. Same members as the Bayesian deep-ensemble baseline (DE-MI), so the comparison isolates the credal representation.

### A2. Credal wrapper over MC-dropout (MCD-Credal) [Gal & Ghahramani, 2016; Wang et al., 2025a]
One model, $T=10$ dropout samples, same wrapper. Cost: one training.

### A3. Credal Deep Ensembles (CreDE) [Wang et al., 2024]
Each member has an interval head: midpoint $m$ and half-length $h = \mathrm{softplus}(\cdot)$ give deterministic intervals $[m-h, m+h]$, mapped to probability intervals by the *interval softmax*
$q_{L,i} = \sigma\big(a_{L,i} - \mathrm{LSE}_{k\ne i}\, m_k\big)$, $q_{U,i} = \sigma\big(a_{U,i} - \mathrm{LSE}_{k\ne i}\, m_k\big)$ (numerically stable form of their Eq. 4). Loss $=\mathrm{CE}(q_U) + \mathrm{CE}(q_L)$ on the hardest $\delta=0.5$ fraction of each batch (DRO-inspired). The ensemble averages the members' *reachable* intervals.

### A4. Distributionally-robust credal ensembles (CreDRO) [2026]
Members are ordinary networks that differ only in a CVaR-DRO fraction $\delta_i$: each batch is sorted by per-sample loss and only the top $\lfloor\delta_i B\rfloor$ samples receive gradient; $\delta_i$ is spread uniformly over $[0.5, 1]$. Credal set by min/max as in A1. Same cost as a deep ensemble; the members disagree *by design* on the hard, shifted part of the data, which is exactly what unseen targets look like.

### A5. Single-model credal sets by decalibration (EffCre) [Hofman et al., 2026; Löhr et al., 2025]
From one trained model, shift the logits by $t\,e_k$ and keep every $t$ whose log-likelihood on a reference set stays within $\log\alpha$ of the maximum; bounds are $p_k(x; t_k^{\pm})$, found by bisection. No ensemble, no retraining. We use the held-out calibration split as the likelihood reference (a fine-tuned transformer fits its training set almost perfectly, which collapses the training-likelihood budget) and sweep $\alpha \in \{0.5, 10^{-2}, 10^{-4}, 10^{-6}\}$.

### A6. Prompt-paraphrase credal sets from an LLM (Prompt-Credal) [Manchingal et al., 2026; Xiong et al., 2024]
Zero-shot Qwen2.5-Instruct (3B and 7B) with 8 system/instruction paraphrases; class probabilities from the restricted softmax over the verbalizer tokens *favor / against / none*; credal set by min/max over prompts. Baselines: a single prompt, and the prompt-mixture (mean + mutual information).

### Proposed: human-distribution-supervised DRO credal ensemble (**CreDRO-Soft**)

Two deficiencies of A1–A5 motivate the proposal. First, every method above is trained on **hard labels**, so nothing in the training signal tells the network how *contested* an item is; the lower entropy (AU) of the resulting credal set is an artefact of the backbone rather than a model of human disagreement. Second, seed/bootstrap diversity (A1) is incidental: members disagree where they happen to, not where the data is shifted.

CreDRO-Soft therefore trains CreDRO members (A4) against the **empirical human label distribution** $\lambda_x$ wherever per-annotator votes exist (SemEval released ~8 votes per tweet), i.e. the per-sample loss is $-\sum_k \lambda_{x,k}\log p_k$, and keeps the CVaR-DRO member diversity. The intended division of labour is:

- **location of the credal set ← human disagreement** (soft labels pull every member towards $\lambda_x$, so $\underline{H}$ tracks human entropy);
- **size of the credal set ← distributional robustness** (members with small $\delta$ concentrate on the hardest tail, members with $\delta=1$ fit everything; they diverge most on items unlike the training distribution).

We also evaluate the two ingredients separately (CreWrap-Soft = soft labels without DRO; CreDRO = DRO without soft labels), which turns the proposal into a 2×2 ablation.

### Calibration and decision layers (shared by every credal method)

- **Conformal credal calibration** [Javanmardi et al., 2024]: non-conformity $s_i = \mathrm{TV}(\lambda_i, g(x_i))$ on the held-out calibration split, quantile $\hat q$ at level $1-\alpha$, calibrated set $= \{\lambda : \mathrm{TV}(\lambda, g(x)) \le \hat q\}$ with class-wise bounds $[g_k-\hat q,\, g_k+\hat q]$. Guarantee $P(\lambda_x \in \hat{\mathcal{C}}(x)) \ge 1-\alpha$ under exchangeability. We add a **credal-dilation** variant whose score is the distance of $\lambda$ to the predicted box (zero if inside), so the calibrated set is the model's own box dilated by $\hat q$.
- **Interval dominance** [Troffaes, 2007]: $\mathrm{ND}(x) = \{k : u_k \ge \max_j l_j\}$. Routing: one non-dominated class → predict; several and low EU → set-valued answer; high EU → defer.

---

## 4. System architecture

```
(text, target) ──► DeBERTa-v3-base ──► M members {CE | soft | DRO | soft+DRO | CreNet}
                                              │  p_1 … p_M  (or q_L, q_U per member)
                                              ▼
                               credal layer: l = min_m p_m, u = max_m p_m  → reachable(l,u)
                                              │
              ┌───────────────────────────────┼─────────────────────────────┐
              ▼                               ▼                             ▼
   decomposition                conformal calibration (λ_x on cal)     interval dominance
   TU = H̄, AU = H̲, EU = H̄−H̲      q̂ → [g−q̂, g+q̂] or box ⊕ q̂            ND(x) → predict / set / defer
```

Single-model variants (EffCre, MCD-Credal) and the LLM prompt-credal share everything from the credal layer down.

---

## 5. Data

All numbers below were verified on the files used (October–November 2026).

| Split | Items | With human votes | Role |
|---|---|---|---|
| SemEval-2016 train (80 %) | 2,250 | 1,934 | member training |
| SemEval-2016 cal (20 % of train, stratified by target, never trained on) | 564 | 485 | temperature, EffCre reference, conformal calibration |
| SemEval-2016 test (official) | 1,249 | 1,080 | in-target evaluation; human alignment |
| SemEval low-agreement pool (dropped from the task) | 1,385 (1,101 on the four seen targets) | 1,385 | ambiguity stress test |
| P-Stance test (Trump, Biden, Sanders; binary) | 2,157 | – | cross-dataset unseen targets |
| COVID-19-Stance test (face masks, Fauci, stay-at-home, school closures) | 800 | – | cross-dataset unseen targets |

Human label distributions: $\lambda_x = (n_{\textsc{favor}}, n_{\textsc{against}}, n_{\textsc{none}})/n$, median $n = 8$; NEUTRAL votes mapped to NONE. The *Climate Change* target has no released votes and carries hard labels only. Leave-one-target-out (LOTO): five folds, each excluding one SemEval target from training; the official test set is then split into seen and unseen items.

## 6. Experimental protocol

- Backbone `microsoft/deberta-v3-base`, `[CLS]` pooling, AdamW 2e-5, 5 epochs, batch 32, max length 128, bf16 autocast with fp32 master weights; $M=5$ members, seed $i$ + 80 % bootstrap; MC-dropout $T=10$; CreNet $\delta=0.5$; CreDRO $\delta_i \in \{0.5, 0.625, 0.75, 0.875, 1.0\}$.
- 151 member trainings (26 in-target + 5 × 25 LOTO) on one RTX PRO 5000, ≈45 s each; all credal/conformal/decision computations on CPU from cached outputs.
- Metrics: accuracy, $F_{avg}$ (official SemEval: mean F1 of FAVOR and AGAINST), macro-F1, ECE (15 bins), NLL; misclassification AUROC and AURC with TU; unseen-target AUROC with EU, width and TU; determinacy, set accuracy, $u_{65}$ [Zaffalon et al., 2012]; Spearman ρ between AU/EU/TU and human entropy; JSD/TV between $p^*$ and $\lambda_x$; conformal coverage and efficiency over 100 random 50/50 re-splits of the voted cal ∪ test items; expected decision cost with $c_{err}=1$, $c_{defer}=0.3$, $c_{set}=0.15$ per extra class.


---

## 7. Results

All tables are copied from `results/tables.md` and `results/ablation_stats.md` (generated by `credistance/analysis.py` and `credistance/ablation_stats.py`). Backbone DeBERTa-v3-base, $M=5$ unless stated. "TU" is the upper entropy for credal methods and the predictive entropy otherwise; "EU" is $\overline H-\underline H$ for credal methods, mutual information for DE/MCD, vacuity for EDL and $1-\max p$ for single models; "width" is $\sum_k (u_k-l_k)$.

### 7.1 E1 — In-target accuracy, calibration, selective prediction (SemEval-2016 test, n = 1,249)

| Method | Acc | F_avg | macro-F1 | ECE | NLL | AUROC-err (TU) | AURC (TU) | Determ. | Set acc | u65 | width |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Softmax | 0.742 | 0.756 | 0.718 | 0.154 | 0.696 | **0.789** | 0.103 | - | - | - | - |
| TempScale | 0.742 | 0.756 | 0.718 | 0.077 | 0.590 | 0.785 | 0.104 | - | - | - | - |
| MCD-MI | 0.757 | 0.771 | 0.729 | 0.087 | 0.583 | 0.754 | 0.105 | - | - | - | - |
| MCD-Credal | 0.755 | 0.766 | 0.725 | 0.050 | **0.571** | 0.741 | 0.110 | 0.676 | 0.886 | 0.759 | 0.678 |
| EDL | 0.718 | 0.733 | 0.698 | 0.088 | 0.702 | 0.786 | 0.116 | - | - | - | - |
| DE-MI | 0.761 | 0.773 | **0.740** | 0.086 | 0.611 | 0.754 | 0.102 | - | - | - | - |
| CreWrap-DE | 0.754 | 0.769 | 0.727 | 0.066 | 0.607 | 0.746 | 0.109 | 0.698 | 0.869 | 0.757 | 0.620 |
| CreDE | 0.757 | 0.772 | 0.732 | 0.068 | 0.576 | 0.770 | **0.100** | 0.985 | 0.765 | 0.760 | 0.031 |
| CreWrap-Soft | 0.758 | 0.772 | 0.736 | **0.029** | 0.578 | 0.729 | 0.113 | 0.692 | 0.874 | 0.758 | 0.587 |
| CreDRO | **0.762** | **0.775** | **0.740** | 0.046 | 0.583 | 0.733 | 0.110 | 0.688 | 0.876 | 0.758 | 0.646 |
| CreDRO-Soft | 0.753 | 0.763 | 0.736 | 0.041 | 0.602 | 0.723 | 0.117 | 0.701 | 0.861 | 0.749 | 0.612 |
| EffCre(α=0.5) | 0.747 | 0.761 | 0.724 | 0.144 | 0.685 | 0.777 | 0.104 | 0.970 | 0.761 | 0.751 | 0.081 |
| EffCre(α=1e-6) | 0.748 | 0.762 | 0.724 | 0.127 | 0.650 | 0.775 | 0.104 | 0.880 | 0.802 | **0.761** | 0.302 |
| LLM-single (Qwen2.5-3B) | 0.635 | 0.630 | 0.580 | 0.274 | 2.063 | 0.655 | 0.258 | - | - | - | - |
| LLM-promptMI (Qwen2.5-3B) | 0.632 | 0.628 | 0.581 | 0.215 | 1.651 | 0.647 | 0.262 | - | - | - | - |
| LLM-PromptCredal (Qwen2.5-3B) | 0.620 | 0.627 | 0.578 | 0.143 | 1.500 | 0.621 | 0.279 | 0.535 | 0.810 | 0.618 | 0.995 |

Bootstrap 95 % CIs (1,000 resamples): ECE Softmax 0.154 [0.130, 0.177], DE-MI 0.091 [0.071, 0.112], CreWrap-DE 0.072 [0.054, 0.093], CreDRO 0.054 [0.036, 0.072], CreDRO-Soft 0.051 [0.035, 0.069], CreWrap-Soft 0.038 [0.023, 0.054]; AURC Softmax 0.102 [0.087, 0.120] vs CreDRO-Soft 0.118 [0.101, 0.135]; accuracy CIs all overlap (0.74–0.76).

**Reading.** (i) Accuracy is unchanged (all CIs overlap); the official $F_{avg}$ of 0.756–0.775 is in the range reported for BERT-family models on this benchmark [Garg & Caragea, 2024]. (ii) **Calibration improves monotonically with the amount of credal/soft structure**: Softmax 0.154 → temperature 0.077 → ensemble 0.086 → credal wrapper 0.066 → DRO 0.046 → soft labels 0.029. The soft-label gain is significant (non-overlapping CIs with every hard-label method). (iii) **Selective prediction in-target is *not* improved**: the softmax entropy remains the best error detector (AUROC 0.789) and AURC differences are within CIs. Credal TU is a worse *error* score in-target because it is deliberately high on contested items, which the models often get right by majority. (iv) Set-valued prediction: wrapper-based credal sets abstain to a 2–3-class set on ~30 % of items and are then right 87 % of the time, with utility-discounted accuracy $u_{65}$ ≈ 0.76 ≥ single accuracy; CreDE and EffCre produce very narrow intervals (width 0.03–0.3), consistent with the "intervals too tight" observation of the CreDRO authors.

### 7.2 E4a — Unseen-target detection, leave-one-target-out (mean ± std over 5 folds)

| Method | AUROC unseen (EU) | AUROC unseen (width) | AUROC unseen (TU) | Acc unseen | F_avg unseen | AURC unseen (TU) | mean AU seen→unseen | mean EU seen→unseen |
|---|---|---|---|---|---|---|---|---|
| Softmax | 0.786 ± 0.052 | 0.786 ± 0.052 | 0.816 ± 0.065 | 0.572 ± 0.135 | 0.511 ± 0.124 | 0.339 ± 0.129 | 0.56→1.04 | 0.15→0.31 |
| TempScale | 0.795 ± 0.055 | 0.795 ± 0.055 | 0.828 ± 0.065 | 0.572 ± 0.135 | 0.511 ± 0.124 | 0.341 ± 0.129 | 0.79→1.22 | 0.21→0.38 |
| DE-MI | 0.743 ± 0.051 | 0.743 ± 0.051 | 0.875 ± 0.037 | 0.501 ± 0.106 | 0.442 ± 0.121 | 0.439 ± 0.110 | 0.58→1.05 | 0.12→0.22 |
| CreWrap-DE | 0.623 ± 0.051 | 0.767 ± 0.040 | 0.871 ± 0.042 | 0.506 ± 0.092 | 0.453 ± 0.113 | 0.494 ± 0.087 | 0.28→0.65 | 0.68→0.85 |
| CreDE | 0.520 ± 0.101 | **0.878 ± 0.039** | 0.890 ± 0.033 | 0.562 ± 0.106 | 0.501 ± 0.119 | 0.390 ± 0.107 | 0.73→1.24 | 0.05→0.05 |
| CreWrap-Soft | 0.505 ± 0.043 | 0.760 ± 0.033 | 0.878 ± 0.034 | 0.528 ± 0.087 | 0.477 ± 0.102 | 0.448 ± 0.089 | 0.53→0.98 | 0.53→0.54 |
| CreDRO | 0.542 ± 0.082 | 0.787 ± 0.025 | 0.897 ± 0.016 | 0.503 ± 0.130 | 0.445 ± 0.146 | 0.469 ± 0.140 | 0.37→0.75 | 0.72→0.76 |
| **CreDRO-Soft** | 0.337 ± 0.106 | 0.759 ± 0.058 | **0.905 ± 0.021** | 0.523 ± 0.083 | 0.476 ± 0.108 | 0.428 ± 0.087 | 0.64→1.06 | 0.60→0.48 |
| EffCre(α=1e-6) | 0.692 ± 0.045 | 0.810 ± 0.076 | 0.828 ± 0.078 | 0.563 ± 0.130 | 0.502 ± 0.122 | 0.369 ± 0.125 | 0.36→0.72 | 0.44→0.60 |

Per-fold unseen AUROC (TU) for CreDRO-Soft: Atheism 0.908, Climate 0.938, Feminism 0.873, Clinton 0.902, Abortion 0.902 (Softmax: 0.814 / 0.916 / 0.763 / 0.733 / 0.854). Paired tests over folds: CreDRO-Soft > Softmax in 5/5 folds (+0.089, paired t p = 0.025); CreDRO-Soft > DE-MI in 4/5 (+0.030, p = 0.16); CreDRO-Soft > CreDRO in 4/5 (+0.008, n.s.).

**Reading.** (i) The **upper entropy of a credal set is the best unseen-target detector** and the most *stable* one: CreDRO-Soft reaches 0.905 ± 0.021 and CreDRO 0.897 ± 0.016, against 0.816 ± 0.065 for softmax entropy and 0.875 ± 0.037 for the deep-ensemble entropy; the fold-to-fold standard deviation of the credal methods is 3–4× smaller. (ii) **The additive entropy decomposition breaks on unseen targets.** On an unseen target every member becomes nearly uniform, so the *lower* entropy jumps (0.64 → 1.06 bits for CreDRO-Soft, out of a maximum 1.585) and $\overline H-\underline H$ *shrinks* (0.60 → 0.48): the EU score is anti-correlated with novelty (AUROC 0.34). This is the "AU saturates, EU has no room" pathology of entropy-based decompositions discussed by Wimmer et al. [2023]; the simple **interval width** does not suffer from it (0.76–0.88) and the trained-interval CreDE width is the best epistemic score (0.878). (iii) Unseen-target *accuracy* collapses for every method (0.50–0.57 vs 0.73–0.79 on seen targets), which is why detection and deferral matter.

### 7.3 E4b — Cross-dataset unseen targets (SemEval-trained models; unseen = P-Stance / COVID-19 test)

| Method | AUROC P-Stance (EU) | (width) | (TU) | Acc P-Stance | AUROC COVID (EU) | (width) | (TU) | Acc COVID | macro-F1 COVID |
|---|---|---|---|---|---|---|---|---|---|
| Softmax | 0.594 | 0.594 | 0.597 | 0.724 | 0.874 | 0.874 | 0.895 | 0.585 | 0.586 |
| TempScale | 0.596 | 0.596 | 0.602 | 0.724 | 0.885 | 0.885 | **0.910** | 0.585 | 0.586 |
| MCD-MI | 0.450 | 0.450 | 0.519 | 0.674 | 0.535 | 0.535 | 0.846 | 0.554 | 0.557 |
| EDL | 0.539 | 0.539 | 0.551 | 0.673 | 0.763 | 0.763 | 0.808 | 0.554 | 0.544 |
| DE-MI | 0.646 | 0.646 | 0.657 | 0.710 | 0.708 | 0.708 | 0.856 | 0.600 | 0.598 |
| CreWrap-DE | **0.651** | 0.641 | 0.680 | 0.681 | 0.671 | 0.737 | 0.867 | 0.595 | 0.596 |
| CreDE | 0.627 | 0.642 | 0.655 | 0.726 | 0.716 | **0.902** | 0.848 | 0.609 | **0.610** |
| CreWrap-Soft | 0.629 | **0.664** | **0.713** | 0.712 | 0.501 | 0.730 | 0.877 | **0.610** | 0.609 |
| CreDRO | 0.615 | 0.627 | 0.678 | 0.690 | 0.651 | 0.751 | 0.867 | 0.583 | 0.583 |
| CreDRO-Soft | 0.437 | 0.624 | 0.711 | 0.707 | 0.398 | 0.749 | 0.888 | 0.579 | 0.579 |
| EffCre(α=1e-6) | 0.579 | 0.592 | 0.597 | 0.713 | 0.790 | 0.890 | 0.904 | 0.585 | 0.588 |
| LLM-PromptCredal (Qwen2.5-3B) | 0.463 | 0.444 | 0.458 | **0.744** | 0.481 | 0.475 | 0.475 | 0.532 | 0.517 |

**Reading.** COVID tweets (health domain, 2020) are far from the SemEval training targets and every supervised method detects them well with TU (0.85–0.91); P-Stance tweets are political like SemEval and are much harder to tell apart (0.60–0.71; best CreWrap-Soft 0.713 and CreDRO-Soft 0.711). The LLM's prompt-disagreement carries no information about *our* training distribution (AUROC < 0.5), as expected: epistemic uncertainty is relative to what a model was trained on. Zero-shot transfer accuracy is 0.58–0.61 on COVID (3-class) and 0.68–0.75 on P-Stance (binary).

### 7.4 E5 — Alignment with human disagreement (SemEval test items with votes, n = 1,080)

| Method | ρ(AU, H) | ρ(EU, H) | ρ(TU, H) | JSD(p*, λ) | TV(p*, λ) |
|---|---|---|---|---|---|
| Softmax | 0.318 | 0.311 | 0.318 | 0.162 | 0.289 |
| TempScale | 0.323 | 0.315 | 0.323 | 0.133 | 0.267 |
| MCD-MI | 0.334 | 0.321 | 0.341 | 0.133 | 0.263 |
| EDL | 0.318 | 0.327 | 0.318 | 0.155 | 0.295 |
| DE-MI | 0.325 | 0.277 | 0.321 | 0.144 | 0.271 |
| CreWrap-DE | 0.324 | 0.258 | 0.316 | 0.142 | 0.274 |
| CreDE | 0.327 | 0.106 | 0.327 | 0.136 | 0.265 |
| CreWrap-Soft | **0.345** | 0.169 | 0.334 | **0.124** | **0.260** |
| CreDRO | 0.302 | 0.214 | 0.289 | 0.138 | 0.270 |
| CreDRO-Soft | 0.304 | **0.068** | 0.319 | 0.129 | 0.266 |
| EffCre(α=1e-6) | 0.317 | 0.323 | 0.328 | 0.149 | 0.279 |
| LLM-PromptCredal (Qwen2.5-3B) | 0.082 | 0.140 | 0.145 | 0.209 | 0.348 |

All ρ(AU, H) for supervised models are significant (p < 10⁻²⁰). **Reading.** Soft-label training gives the best match to human disagreement (ρ = 0.345, lowest JSD 0.124). More important is *discriminant validity*: for credal methods the epistemic score is almost uncorrelated with human entropy (ρ(EU,H) = 0.07 for CreDRO-Soft, 0.11 for CreDE, 0.17 for CreWrap-Soft) while the aleatoric score correlates at 0.30–0.35, i.e. the two parts of the credal set carry different information. For a single softmax (and EffCre, which only dilates it) both scores are the *same* ranking, so no separation is possible. The zero-shot LLM's uncertainty barely tracks human disagreement (ρ ≈ 0.08–0.15).

### 7.5 E5b — Ambiguity stress test (1,101 low-agreement tweets on the four seen targets vs the test set)

| Method | AUROC ambig (AU) | AUROC ambig (EU) | AUROC ambig (width) | AUROC ambig (TU) | mean AU test→ambig | mean EU test→ambig |
|---|---|---|---|---|---|---|
| Softmax | 0.652 | 0.648 | 0.648 | 0.652 | 0.45→0.65 | 0.11→0.18 |
| DE-MI | 0.660 | 0.619 | 0.619 | 0.656 | 0.50→0.68 | 0.12→0.16 |
| CreWrap-DE | 0.652 | 0.597 | – | 0.648 | 0.25→0.36 | 0.60→0.73 |
| CreDE | 0.663 | 0.530 | – | 0.664 | 0.66→0.88 | 0.04→0.04 |
| CreWrap-Soft | **0.668** | 0.521 | – | 0.653 | 0.55→0.73 | 0.54→0.56 |
| CreDRO | 0.638 | 0.589 | – | 0.652 | 0.30→0.42 | 0.63→0.72 |
| CreDRO-Soft | 0.653 | **0.486** | – | 0.652 | 0.65→0.81 | 0.54→0.54 |
| EffCre(α=1e-6) | 0.648 | 0.636 | – | 0.654 | 0.28→0.42 | 0.38→0.49 |
| LLM-PromptCredal (Qwen2.5-3B) | 0.578 | 0.575 | – | 0.578 | 0.04→0.06 | 0.73→0.86 |

(width columns for credal methods are in `results/tables.md`.) **Reading.** Tweets that humans could not agree on are only moderately separable from the test set by any single score (AUROC 0.64–0.67): many *test* tweets are also contested (mean human entropy 0.63 bits). The decisive observation is the *pattern*: for CreDRO-Soft and CreWrap-Soft the aleatoric score rises on the ambiguous pool (0.65 → 0.81) while the epistemic score stays flat (0.54 → 0.54; AUROC 0.49–0.52, i.e. chance), whereas for the hard-label wrapper both rise together (EU 0.60 → 0.73). Soft-label supervision is what stops "contested" from leaking into "unknown".

### 7.6 E6 — Conformal coverage of the human label distribution (100 random re-splits of the 1,565 voted cal ∪ test items)

| Method | cov@0.9 (TV ball) | q̂@0.9 | set size@0.9 | cov@0.8 | q̂@0.8 | set size@0.8 | cov@0.9 (credal dilation) | q̂ | set size |
|---|---|---|---|---|---|---|---|---|---|
| Softmax | 0.903 | 0.724 | 3.00 | 0.799 | 0.546 | 2.98 | - | - | - |
| TempScale | 0.903 | 0.642 | 3.00 | 0.798 | 0.478 | 2.92 | - | - | - |
| DE-MI | 0.902 | 0.635 | 3.00 | 0.798 | 0.472 | 2.50 | - | - | - |
| CreWrap-DE | 0.901 | 0.602 | 3.00 | 0.798 | 0.466 | 2.46 | 0.901 | 0.473 | 2.66 |
| CreDE | 0.900 | 0.617 | 3.00 | 0.798 | 0.449 | **2.38** | 0.900 | 0.609 | 3.00 |
| CreWrap-Soft | 0.901 | 0.546 | 3.00 | 0.797 | **0.436** | 2.58 | 0.900 | 0.420 | 2.64 |
| CreDRO | 0.901 | 0.586 | 3.00 | 0.797 | 0.458 | 2.56 | 0.901 | 0.445 | 2.68 |
| CreDRO-Soft | 0.900 | **0.542** | 3.00 | 0.799 | 0.444 | 2.89 | 0.900 | **0.416** | 2.90 |
| EffCre(α=1e-6) | 0.903 | 0.698 | 3.00 | 0.799 | 0.517 | 2.89 | 0.902 | 0.596 | 3.00 |
| LLM-PromptCredal (Qwen2.5-3B) | 0.899 | 0.757 | 3.00 | 0.799 | 0.598 | 3.00 | 0.900 | 0.705 | 3.00 |

**Reading.** The guarantee holds for every predictor (empirical coverage 0.90 / 0.80 at the nominal levels). Efficiency is where the predictors differ: the TV radius needed for 90 % coverage falls from 0.724 (softmax) to 0.542 (CreDRO-Soft) and 0.546 (CreWrap-Soft) — **soft-label training produces the most efficient conformal credal sets**. At 90 % the radius is still so large that every class stays non-dominated (set size 3): with a median of only 8 votes per tweet, $\lambda_x$ is itself a noisy estimate (standard error up to 0.18 per class), so a distribution-free 90 % guarantee must be loose. At 80 % the sets become informative (2.4–2.6 classes). The credal-dilation variant needs a smaller additive radius (0.416 for CreDRO-Soft) because the model's own box already covers part of the discrepancy.

### 7.7 E7 — Decision value (expected cost; $c_{err}=1$, $c_{defer}=0.3$, $c_{set}=0.15$ per extra class)

In-target test:

| Method | conf@0 | conf@5 % | conf@10 % | conf@20 % | credal@0 | credal@5 % | credal@10 % | credal@20 % |
|---|---|---|---|---|---|---|---|---|
| Softmax | 0.258 | 0.242 | 0.228 | 0.213 | - | - | - | - |
| DE-MI | 0.239 | 0.223 | 0.220 | 0.202 | - | - | - | - |
| CreWrap-DE | 0.246 | 0.233 | 0.222 | 0.212 | 0.185 | 0.190 | 0.196 | 0.200 |
| CreDE | 0.243 | 0.232 | 0.222 | 0.202 | 0.238 | 0.237 | 0.235 | 0.236 |
| CreWrap-Soft | 0.242 | 0.233 | 0.228 | 0.217 | 0.182 | 0.190 | 0.199 | 0.214 |
| CreDRO | 0.238 | 0.228 | 0.220 | 0.208 | **0.181** | **0.184** | **0.189** | **0.197** |
| CreDRO-Soft | 0.247 | 0.241 | 0.236 | 0.218 | 0.194 | 0.200 | 0.210 | 0.223 |
| EffCre(α=1e-6) | 0.252 | 0.239 | 0.231 | 0.213 | 0.218 | 0.213 | 0.206 | 0.203 |

Unseen targets (LOTO folds, 10 % deferral budget), credal routing vs confidence thresholding: CreDRO 0.246/0.310/0.345/0.283/0.304 vs 0.407/0.685/0.444/0.341/0.512; CreDRO-Soft 0.259/0.278/0.352/0.254/0.332 vs 0.448/0.555/0.436/0.345/0.509; Softmax (confidence only) 0.353/0.324/0.398/0.294/0.673.

**Reading.** Set-valued credal routing lowers the expected cost by about a quarter in-target (0.181 vs 0.238–0.258) *without any deferral*, because interval dominance turns most would-be errors into a cheap two-class answer. On unseen targets the gap widens to 30–50 % at the same 10 % deferral budget. Narrow-interval methods (CreDE, EffCre) gain little because they are almost always determinate.

### 7.8 E8 — LLM prompt-credal (Qwen2.5-3B-Instruct, zero-shot, 8 prompt paraphrases)

Single-prompt accuracy varies from 0.52 to 0.63 across paraphrases on the test set; the credal set over prompts halves ECE (0.274 → 0.143) and reduces NLL (2.06 → 1.50) at unchanged accuracy (0.62–0.64), but prompts disagree so much that only 53 % of items get a single-class answer (width 0.995), and the LLM's uncertainty neither detects our unseen datasets (AUROC < 0.5) nor tracks human disagreement (ρ ≈ 0.1). Fine-tuned 184 M-parameter encoders beat the zero-shot 3B LLM by 12 accuracy points in-target. A Qwen2.5-7B run with the same protocol is included in `results/tables.md` when present.

### 7.9 Ensemble-size ablation (`results/ablation_stats.md`)

| Method | M | ECE | LOTO unseen AUROC (TU) | COVID unseen AUROC (width) | mean width |
|---|---|---|---|---|---|
| CreWrap-DE | 1 / 2 / 3 / 5 | 0.154 / 0.126 / 0.109 / 0.066 | 0.816 / 0.835 / 0.854 / 0.871 | – / 0.768 / 0.750 / 0.737 | 0 / 0.22 / 0.37 / 0.62 |
| CreDRO | 1 / 2 / 3 / 5 | 0.093 / 0.076 / 0.070 / 0.046 | 0.877 / 0.888 / 0.906 / 0.897 | – / 0.775 / 0.740 / 0.751 | 0 / 0.26 / 0.41 / 0.65 |
| CreDRO-Soft | 1 / 2 / 3 / 5 | 0.066 / 0.045 / 0.047 / 0.041 | 0.888 / 0.910 / 0.919 / 0.905 | – / 0.695 / 0.726 / 0.749 | 0 / 0.23 / 0.39 / 0.61 |
| CreWrap-Soft | 1 / 2 / 3 / 5 | 0.077 / 0.059 / 0.052 / 0.029 | 0.843 / 0.857 / 0.875 / 0.878 | – / 0.720 / 0.713 / 0.730 | 0 / 0.21 / 0.34 / 0.59 |

Extending the in-target ensembles to M = 8 brings no further gain (ECE 0.064 / 0.046 / 0.045 / 0.025 for CreWrap-DE / CreDRO / CreDRO-Soft / CreWrap-Soft; COVID unseen AUROC (TU) 0.859 / 0.864 / 0.881 / 0.879) while the mean width keeps growing (0.69–0.78), so M = 5 is used for every main table.

**Reading.** A *single* soft-label DRO model (M = 1) already has ECE 0.066 and unseen-target AUROC 0.888 — better than a five-member hard-label ensemble on both counts — so most of the benefit comes from the training signal, not from the ensemble. Width-based epistemic scores need M ≥ 2 and saturate at M = 3; three CreDRO-Soft members reach the best LOTO AUROC in the study (0.919).

### 7.10 Figures

- `results/fig/eu_seen_vs_unseen.png` — epistemic-score histograms, seen vs unseen target (Clinton fold).
- `results/fig/risk_coverage_intarget.png` — selective risk vs coverage, in-target.
- `results/fig/au_vs_human_entropy.png` — predicted aleatoric uncertainty vs human label entropy (binned).
- `results/fig/au_eu_map.png` — mean (AU, width) of each credal method on seen test / ambiguous / unseen-target items: ambiguity moves points right, novelty moves them up.
- `results/fig/ensemble_size.png` — ablation over M.

---

## 8. Discussion

**What the credal representation buys for stance.** Three things, each measured: (1) *calibrated* class probabilities without a post-hoc temperature (ECE 0.03–0.05 vs 0.15), (2) the *most stable* unseen-target detector in the study (upper entropy, 0.90 ± 0.02 vs 0.82 ± 0.07 for softmax entropy), and (3) a *set-valued* answer that cuts decision cost by ~25 % in-target and 30–50 % on unseen targets without deferring anything. It does **not** buy a better in-target error detector; softmax entropy remains best there, because credal total uncertainty is intentionally high on contested-but-correct items.

**Which construction.** Post-hoc wrapping of a deep ensemble is the baseline; replacing seed diversity by DRO diversity (CreDRO) improves calibration and unseen-target AUROC at identical cost; replacing hard labels by human label distributions (Soft) improves calibration further, gives the best human alignment and the most efficient conformal sets, and — the key qualitative result — keeps the epistemic score flat on ambiguous items. Combining both (CreDRO-Soft) gives the best unseen-target detection and the cleanest ambiguity/novelty separation, at the price of slightly higher in-target AURC. Trained-interval heads (CreDE) give excellent *width*-based novelty detection but near-degenerate sets; the single-model EffCre is the cheapest option and matches softmax-level novelty detection on COVID but cannot separate the two kinds of uncertainty (its two scores are the same ranking).

**A measurement result.** The textbook decomposition EU $= \overline H - \underline H$ is unusable for unseen-target detection in three-class stance: because every member collapses to near-uniform predictions on a new target, the lower entropy saturates and the difference *shrinks*. Interval width (or the distance-based measures of Gonzalez-Garcia et al. [2026]) should be used for the epistemic part, and the upper entropy for "total" uncertainty.

**Human label distributions with 8 votes.** They are informative enough to train on (every soft-label gain above) and to calibrate on (coverage holds), but too noisy for a 90 % distribution-free guarantee to be tight. ChaosNLI-style data with ~100 votes would close that gap; collecting it was out of scope by design.

## 9. Limitations

- One backbone size (DeBERTa-v3-base) and $M = 5$; the ablation suggests $M = 3$ suffices, but a large backbone was not tested.
- Human votes exist only for four SemEval targets; human-alignment claims are scoped to SemEval.
- Conformal guarantees assume exchangeability between calibration and test items; cross-dataset sets are not guaranteed.
- The LLM experiments are zero-shot; LoRA-ensemble credal LLMs [Manchingal et al., 2026] were not trained.
- Unseen-target accuracy is low for every method (≈0.5); this work improves *knowing* when the model is out of its depth, not the stance prediction itself.

## 10. Conclusion

We built and evaluated six ways of giving a stance detector a credal output and proposed one that combines distributional robustness with supervision from publicly available per-annotator votes. On SemEval-2016, P-Stance and COVID-19-Stance the proposal (i) calibrates as well as the best method in the study, (ii) detects unseen targets with the highest and most stable AUROC (0.905 ± 0.021, 5/5 folds over softmax, p = 0.025), (iii) is the only family whose epistemic score stays flat on human-contested tweets, (iv) yields the most efficient conformal credal sets, and (v) reduces expected decision cost by 25–50 % through set-valued answers. Along the way we show that the standard entropy-based epistemic measure is unsuitable for this task and that a single soft-label DRO model is a strong cheap alternative when an ensemble is not affordable. No new annotation was needed.

---

## References

- Abellán, J., Klir, G. J., Moral, S. (2006). Disaggregated total uncertainty measure for credal sets. *International Journal of General Systems*, 35(1), 29–44.
- Allaway, E., McKeown, K. (2020). Zero-shot stance detection: A dataset and model using generalized topic representations. *EMNLP*.
- Angelopoulos, A. N., Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4).
- Corani, G., Zaffalon, M. (2008). Learning reliable classifiers from small or incomplete data sets: The naive credal classifier 2. *JMLR*, 9, 581–621.
- Cozman, F. G. (2000). Credal networks. *Artificial Intelligence*, 120(2), 199–233.
- Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
- Garg, K., Caragea, C. (2024). Stanceformer: Target-aware transformer for stance detection. *Findings of EMNLP*. (BERT/BERTweet reference numbers on SemEval-2016.)
- Glandt, K., Khanal, S., Li, Y., Caragea, D., Caragea, C. (2021). Stance detection in COVID-19 tweets. *ACL*.
- Gonzalez-Garcia, X., Chau, S. L., Rodemann, J., Caprio, M., Muandet, K., Bustince, H., Destercke, S., Hüllermeier, E., Sale, Y. (2026). Quantification of credal uncertainty: A distance-based approach. *arXiv:2603.27270*.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- He, P., Gao, J., Chen, W. (2023). DeBERTaV3. *ICLR*.
- Hofman, P., Löhr, T., Muschalik, M., Sale, Y., Hüllermeier, E. (2026). Efficient credal prediction through decalibration. *ICLR*.
- Hüllermeier, E., Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning. *Machine Learning*, 110, 457–506.
- Javanmardi, A., Stutz, D., Hüllermeier, E. (2024). Conformalized credal set predictors. *NeurIPS*.
- Küçük, D., Can, F. (2020). Stance detection: A survey. *ACM Computing Surveys*, 53(1).
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Learning credal ensembles via distributionally robust optimization (CreDRO). (2026). *arXiv:2602.08470*.
- Li, Y., Sosea, T., Sawant, A., Nair, A. J., Inkpen, D., Caragea, C. (2021). P-Stance: A large dataset for stance detection in political domain. *Findings of ACL*.
- Löhr, T., Hofman, P., Mohr, F., Hüllermeier, E. (2025). Credal prediction based on relative likelihood. *NeurIPS*.
- Manchingal, S. K., Nikolenko, S., Cuzzolin, F. (2026). Credal large language models for semantic commitment under uncertainty. *arXiv:2608.23244*.
- Mohammad, S. M., Kiritchenko, S., Sobhani, P., Zhu, X., Cherry, C. (2016). SemEval-2016 Task 6: Detecting stance in tweets. *SemEval*.
- Mohammad, S. M., Sobhani, P., Kiritchenko, S. (2017). Stance and sentiment in tweets. *ACM TOIT*, 17(3). Raw annotations: saifmohammad.com/WebPages/StanceDataset.htm.
- Nie, Y., Zhou, X., Bansal, M. (2020). What can we learn from collective human opinions on NLI data? (ChaosNLI). *EMNLP*.
- Plank, B. (2022). The "problem" of human label variation. *EMNLP*.
- Sale, Y., Caprio, M., Hüllermeier, E. (2023). Is the volume of a credal set a good measure for epistemic uncertainty? *UAI*.
- Sensoy, M., Kaplan, L., Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS*.
- Troffaes, M. C. M. (2007). Decision making under uncertainty using imprecise probabilities. *IJAR*, 45(1), 17–29.
- Walley, P. (1991). *Statistical Reasoning with Imprecise Probabilities*. Chapman & Hall.
- Wang, K., Cuzzolin, F., Manchingal, S. K., Shariatmadar, K., Moens, D., Hallez, H. (2024). Credal deep ensembles for uncertainty quantification. *NeurIPS*.
- Wang, K., Cuzzolin, F., et al. (2025a). Credal wrapper of model averaging for uncertainty estimation in classification. *ICLR*.
- Wang, K., et al. (2025b). CreINNs: Credal-set interval neural networks. *Neural Networks*.
- Wimmer, L., Sale, Y., Hofman, P., Bischl, B., Hüllermeier, E. (2023). Quantifying aleatoric and epistemic uncertainty in machine learning: Are conditional entropy and mutual information appropriate measures? *UAI*.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? *ICLR*.
- Zaffalon, M., Corani, G., Mauá, D. (2012). Evaluating credal classifiers by utility-discounted predictive accuracy. *IJAR*, 53(8).
- Zhao, C., Caragea, C. (2024). EZ-STANCE: A large dataset for English zero-shot stance detection. *ACL*.
