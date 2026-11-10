# CrediStance: Credal Uncertainty Quantification for Reliable Stance Detection on Unseen Targets

**Project type:** Major project, single stage (6 months)
**Domain:** Natural Language Processing · Uncertainty Quantification · Imprecise Probabilities
**Version:** 1.0 · October 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Background: Credal Sets in One Page](#3-background-credal-sets-in-one-page)
4. [Do Credal Methods Work on Text? (Feasibility)](#4-do-credal-methods-work-on-text-feasibility)
5. [Research Questions and Hypotheses](#5-research-questions-and-hypotheses)
6. [System Architecture Overview](#6-system-architecture-overview)
7. [Module M1: Data Layer](#7-module-m1-data-layer)
8. [Module M2: Human Label Distributions from Existing Raw Annotations](#8-module-m2-human-label-distributions-from-existing-raw-annotations)
9. [Module M3: Input Encoding](#9-module-m3-input-encoding)
10. [Module M4: Backbone Ensembles](#10-module-m4-backbone-ensembles)
11. [Module M5: Credal Layer](#11-module-m5-credal-layer)
12. [Module M6: Uncertainty Decomposition](#12-module-m6-uncertainty-decomposition)
13. [Module M7: Conformal Credal Calibration](#13-module-m7-conformal-credal-calibration)
14. [Module M8: Decision and Routing Layer](#14-module-m8-decision-and-routing-layer)
15. [Module M9 (Stretch): Concept-Level Credal Network](#15-module-m9-stretch-concept-level-credal-network)
16. [Module M10: Explanation Dashboard](#16-module-m10-explanation-dashboard)
17. [End-to-End Worked Examples](#17-end-to-end-worked-examples)
18. [Baselines](#18-baselines)
19. [Evaluation Protocol and Metrics](#19-evaluation-protocol-and-metrics)
20. [Experiment Matrix](#20-experiment-matrix)
21. [Ablation Studies](#21-ablation-studies)
22. [Implementation Details](#22-implementation-details)
23. [Reference Implementation of the Credal Core](#23-reference-implementation-of-the-credal-core)
24. [Six-Month Timeline](#24-six-month-timeline)
25. [Risks and Mitigations](#25-risks-and-mitigations)
26. [Ethical Considerations](#26-ethical-considerations)
27. [Expected Contributions and Deliverables](#27-expected-contributions-and-deliverables)
28. [References](#28-references)

---

## 1. Executive Summary

Stance detection decides whether the author of a text is **in FAVOR of**, **AGAINST**, or **NEUTRAL/NONE** towards a given target (a person, policy, or claim) [19, 40, 41]. Current models, from fine-tuned transformers to LLMs, output **a single probability vector**. That one number cannot tell two very different situations apart:

| Situation | Example | What a reliable system should say |
|---|---|---|
| **Ambiguity** (aleatoric) | *"Oh great, another 'climate emergency' summit. I'm sure the private jets will fix it."* (sarcasm: favor or against climate action?) | "Humans themselves disagree here; this is contested." |
| **Ignorance** (epistemic) | A tweet about a target never seen in training, e.g. *"Digital euro"* | "I have not learned enough about this target; defer or ask for labels." |

**CrediStance** replaces the single probability vector with a **credal set**: a convex set of plausible probability distributions, summarised as a **lower and upper probability for each class**. From the credal set we:

1. **decompose** uncertainty into aleatoric (ambiguity) and epistemic (ignorance) parts [5, 6];
2. **calibrate** it with conformal prediction to get a **distribution-free coverage guarantee** [11];
3. **decide** with imprecise-probability decision rules (interval dominance [17]), so the system can predict, return a set of plausible stances, or defer to a human;
4. **validate** the aleatoric part against **real human disagreement**, using the *publicly released raw annotations* of SemEval-2016 (~8 annotators per tweet, plus 1,385 low-agreement tweets) [19, 48]. **No new annotation is required.**

To our knowledge (literature search, October 2026), **no published work applies credal sets or imprecise probabilities to stance detection**. The closest work applies credal sets to natural language inference (NLI) [11, 12], which has the same structure as stance detection (a text pair and three classes).

---

## 2. Problem Statement

> **Title:** *Knowing What the Model Doesn't Know About Stance: Credal Uncertainty Quantification for Reliable Zero-Shot and Cross-Target Stance Detection.*
>
> Given a text *x* and a target *t*, existing stance detectors output a single probability vector **p**(y | x, t) over {FAVOR, AGAINST, NONE}. That one vector mixes up (i) **aleatoric uncertainty**, which comes from sarcasm, implicit stance, missing context and genuine human disagreement (VAST reports Krippendorff's α = 0.427 [20]), with (ii) **epistemic uncertainty**, which comes from unseen targets, domain shift and limited data. As a result, models are **over-confident on unseen targets** and **cannot justify abstaining**.
>
> We propose **CrediStance**, a stance detector that outputs a **credal set** of distributions (class-wise lower and upper probabilities) built from ensembles of transformer and LLM backbones. CrediStance (a) separates aleatoric from epistemic uncertainty, (b) gives conformal coverage guarantees for the human label distribution, and (c) uses this uncertainty to **abstain, return set-valued predictions, or defer** to humans. We evaluate on in-target, cross-target and zero-shot benchmarks, and against human label distributions taken from existing raw multi-annotator stance data.

**Scope boundaries (what this project is not):**
- It is not about getting a new state-of-the-art macro-F1. The main claim is about **reliability**: how well uncertainty is represented and how good the decisions are.
- It does not infer the beliefs of individual users. It classifies texts only (see §26).

---

## 3. Background: Credal Sets in One Page

**Notation.** There are K = 3 classes 𝒴 = {FAVOR, AGAINST, NONE}, and Δ_K is the probability simplex.

**Credal set.** A closed convex set 𝒞 ⊆ Δ_K of candidate distributions for an imprecisely known ground-truth distribution [1, 6].

**Probability-interval credal set.** Given lower bounds **l** and upper bounds **u** per class:

```
𝒞(l, u) = { p ∈ Δ_K : l_k ≤ p_k ≤ u_k  for all k }
```

**Lower / upper probability.** P̲(k) = min_{p∈𝒞} p_k and P̄(k) = max_{p∈𝒞} p_k.

**Why two kinds of uncertainty can be read from a credal set** [5, 6]:
- If the set is **small but centred** (e.g. around [0.45, 0.42, 0.13]), the model is confident *about* a distribution that is itself spread out. This is **aleatoric** uncertainty.
- If the set is **large** (e.g. P̄(FAVOR) = 0.80 but P̲(FAVOR) = 0.15), the model does not know which distribution is right. This is **epistemic** uncertainty.

**Standard entropy-based measures** (Abellán, Klir & Moral [5]; Hüllermeier & Waegeman [6]):

```
Total uncertainty      TU(𝒞) = H̄(𝒞) = max_{p∈𝒞} H(p)        (upper entropy)
Aleatoric uncertainty  AU(𝒞) = H̲(𝒞) = min_{p∈𝒞} H(p)        (lower entropy)
Epistemic uncertainty  EU(𝒞) = H̄(𝒞) − H̲(𝒞)                  (non-specificity part)
```

Sale et al. [7] show that the **volume** of a credal set is a reasonable epistemic measure for binary problems but not for multi-class ones. That is one reason we use the entropy-based decomposition above with K = 3.

**Classical vs. modern credal models.**

| | Classical (1990s–2010s) | Modern credal deep learning (2024–2026) |
|---|---|---|
| Examples | Credal networks [2], Naive Credal Classifier [3, 4], credal decision trees | Credal Wrapper [8], Credal Deep Ensembles [9], CreINNs [10], Conformal credal sets [11], EffCre [12], Credal LLMs [13], Credal Transformer [14] |
| Where the credal set lives | Inside the model (imprecise conditional probability tables over discrete variables) | At the **output** (lower/upper class probabilities) |
| Input data | Discrete / tabular | **Any**: images, text, graphs, LLM tokens |

---

## 4. Do Credal Methods Work on Text? (Feasibility)

**Short answer: yes.** The common claim that "credal networks only work on tabular data" is true of *classical credal networks* [2] and the *Naive Credal Classifier* [3, 4]. Both need a small number of discrete variables, because exact inference becomes intractable otherwise. You cannot build a credal network over a 30,000-word vocabulary.

**Modern credal methods do not depend on the input type.** They need only a model that produces probability vectors over classes. A neural encoder turns the text into those vectors, and the credal construction never sees raw features. Published evidence on language data:

| Work | Venue | Text / language evidence |
|---|---|---|
| Conformalized Credal Set Predictors [11] | NeurIPS 2024 | Credal sets on **ChaosNLI** (3,113 NLI pairs, ~100 annotators each) using transformer-language-model embeddings |
| Efficient Credal Prediction through Decalibration [12] | ICLR 2026 | Credal sets on **ChaosNLI**, CLIP, TabPFN and in-context learning |
| Credal LLMs [13] | arXiv 2026 | LoRA-ensemble credal sets on Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B for QA, calibration and selective prediction |
| Credal Transformer [14] | arXiv 2025 | Credal attention inside the Transformer architecture |
| Credal Wrapper [8] | ICLR 2025 | Model-agnostic post-hoc wrapper (tested on ViT/ResNet; applies unchanged to any ensemble) |

**Stance detection has the same structure as NLI:**

| | NLI | Stance |
|---|---|---|
| Input | (premise, hypothesis) | (text, target) |
| Labels | entailment / neutral / contradiction | favor / none / against |
| Human disagreement | high (ChaosNLI [24]) | high (VAST α = 0.427 [20]) |

EZ-Stance [21] even reformulates zero-shot stance detection *as* NLI. The credal machinery shown to work for NLI therefore carries over directly.

**On data size.** Credal sets are *most* useful when data is scarce, because that is when epistemic uncertainty is large and lower and upper probabilities separate [3, 4]. Unseen and few-shot targets are exactly this case.

---

## 5. Research Questions and Hypotheses

| ID | Research question | Hypothesis | Tested in |
|---|---|---|---|
| RQ1 | Do credal stance models give better uncertainty estimates than standard UQ methods? | **H1:** CrediStance improves misclassification-AUROC and AURC over softmax, MC Dropout, Deep Ensembles and EDL, at equal macro-F1 (±1 point). | E1–E3 |
| RQ2 | Does the decomposition separate ambiguity from ignorance? | **H2a:** EU is significantly higher on unseen than on seen targets (unseen-target AUROC ≥ 0.75). **H2b:** AU correlates with human-label entropy (Spearman ρ ≥ 0.4) more strongly than EU does. | E4, E5 |
| RQ3 | Can we guarantee coverage of the human label distribution? | **H3:** Conformal credal sets reach the target coverage 1−α (±2 %) on HLV-test (SemEval-2016 raw annotations), and are smaller (more efficient) when built on the credal score than on the softmax score. | E6 |
| RQ4 | Does credal uncertainty lead to better decisions? | **H4:** Credal routing (predict / set-valued / defer) gives lower expected cost than confidence thresholding, at every deferral budget. | E7 |
| RQ5 | Do credal sets from LLMs beat LLM confidence baselines? | **H5:** LoRA-ensemble credal LLMs are better calibrated than verbalised and sampling-based LLM confidence [44]. | E8 |

---

## 6. System Architecture Overview

```
                        ┌──────────────────────────────────────────────────────┐
                        │                 M1  DATA LAYER                        │
                        │ SemEval-16 · VAST · EZ-Stance · COVID-19-Stance        │
                        │ unified schema → splits (in-target / LOTO / zero-shot) │
                        └───────────────┬───────────────────────┬──────────────┘
                                        │                       │
                     ┌──────────────────▼─────┐   ┌─────────────▼──────────────────┐
                     │ M3  INPUT ENCODING      │   │ M2  HUMAN LABEL DISTRIBUTIONS  │
                     │ encoder template /      │   │ SemEval-16 raw (~8 ann./tweet) │
                     │ LLM prompt template     │   │ → label distributions λ_x      │
                     └──────────┬─────────────┘   └───────┬─────────────┬──────────┘
                                │                         │ calibration │ test
             ┌──────────────────▼──────────────────┐      │             │
             │ M4  BACKBONE ENSEMBLES               │      │             │
             │ Path A: DeBERTa-v3-large × M=8       │      │             │
             │ Path B: Qwen2.5-7B + 8 LoRA adapters │      │             │
             │ → M probability vectors p_1 … p_M    │      │             │
             └──────────────────┬──────────────────┘      │             │
                                │                          │             │
             ┌──────────────────▼──────────────────┐      │             │
             │ M5  CREDAL LAYER                     │      │             │
             │ l_k = min p_mk , u_k = max p_mk      │      │             │
             │ intersection probability p*          │      │             │
             └───────┬───────────────────┬─────────┘      │             │
                     │                   │                │             │
     ┌───────────────▼──────┐  ┌─────────▼──────────────▼───┐         │
     │ M6  UNCERTAINTY       │  │ M7  CONFORMAL CREDAL       │         │
     │ DECOMPOSITION         │  │ CALIBRATION                │         │
     │ TU = H̄, AU = H̲,       │  │ score TV(λ_x, p*)          │         │
     │ EU = H̄ − H̲           │  │ → radius q̂, guarantee 1−α │         │
     └───────────────┬──────┘  └─────────┬──────────────────┘         │
                     │                   │                              │
             ┌───────▼───────────────────▼─────────┐                    │
             │ M8  DECISION & ROUTING               │◄──── evaluation ───┘
             │ interval dominance → {predict,       │
             │ set-valued, defer-to-human}          │
             └───────────────────┬─────────────────┘
                                 │
     ┌───────────────────────────▼───────────────┐   ┌──────────────────────────────┐
     │ M10  EXPLANATION DASHBOARD                 │◄──┤ M9 (stretch) CONCEPT-LEVEL   │
     │ lower/upper bars, AU/EU gauges, routing    │   │ CREDAL NETWORK               │
     └───────────────────────────────────────────┘   └──────────────────────────────┘
```

**Data flow summary:** text + target → M ensemble members → M probability vectors → credal set (l, u) → {p*, TU, AU, EU} → conformal radius q̂ → calibrated bounds → decision.

---

## 7. Module M1: Data Layer

### 7.1 Benchmarks

| Dataset | Ref | Domain | Targets | Size (≈) | Labels | Role in project |
|---|---|---|---|---|---|---|
| SemEval-2016 Task 6A | [19] | Tweets | 5 (Atheism, Climate Change is a Real Concern, Feminist Movement, Hillary Clinton, Legalization of Abortion) | 4,163 | FAVOR / AGAINST / NONE | In-target + leave-one-target-out (LOTO) cross-target |
| VAST | [20] | NYT "Room for Debate" comments | ~5,600 topics | ~18,500 | pro / con / neutral | Zero-shot & few-shot |
| EZ-Stance | [21] | Tweets | Noun-phrase + claim targets, many domains | 47,316 | favor / against / none | Zero-shot (target-based and domain-based subtasks) |
| COVID-19-Stance | [23] | Tweets | 4 (Fauci, stay-at-home orders, face masks, school closures) | ~6,100 | in-favor / against / neither | Temporal / domain-shift stress test (optional) |
| P-Stance | [22] | Tweets | 3 (Trump, Biden, Sanders) | ~21,500 | favor / against (binary) | Optional binary sanity check |

> Sizes are approximate. Verify them against the official releases when you download. Twitter datasets are distributed as tweet IDs plus labels, and some tweets can no longer be retrieved. Record the number actually retrieved.

### 7.2 Unified label schema

| Unified label | SemEval-16 | VAST | EZ-Stance | COVID-19-Stance |
|---|---|---|---|---|
| `FAVOR` (0) | FAVOR | pro | favor | in-favor |
| `AGAINST` (1) | AGAINST | con | against | against |
| `NONE` (2) | NONE | neutral | none | neither |

### 7.3 Unified record format (JSONL)

```json
{
  "id": "semeval16-climate-00412",
  "source": "semeval2016",
  "text": "We need to act NOW. The science is settled #ClimateAction",
  "target": "Climate Change is a Real Concern",
  "target_type": "noun_phrase",
  "label": 0,
  "label_dist": null,
  "split": "train",
  "seen_target": true,
  "domain": "politics/environment",
  "meta": {"orig_label": "FAVOR", "lang": "en"}
}
```

`label_dist` is filled for every SemEval-2016 item that has raw annotations, e.g. `[0.625, 0.250, 0.125]` (see §8).

### 7.4 Splits

| Split protocol | Construction | Purpose |
|---|---|---|
| **In-target** | Official SemEval train/test | Sanity check: credal methods must not hurt F1 |
| **LOTO cross-target** | 5 folds; train on 4 SemEval targets, test on the 5th | Controlled epistemic shift |
| **Zero-shot** | VAST official zero-shot test; EZ-Stance target-based and domain-based splits | Strong epistemic shift |
| **Seen/unseen pool** | Mix the in-target test set (seen = 0) with zero-shot test items (unseen = 1) | Unseen-target detection (E4) |
| **HLV-cal / HLV-test / HLV-unseen / HLV-ambig** | Built from SemEval-2016 raw annotations (§8.3) | Conformal calibration / human-alignment evaluation |

A **dev set** (10 % of train, stratified by target) is held out for early stopping and hyperparameter selection. **Never** tune on the zero-shot test sets.

### 7.5 Pre-processing

1. Replace URLs with `HTTPURL` and user mentions with `@USER` (the BERTweet convention [47]).
2. Keep hashtags, but also add a segmented copy: `#ClimateAction` → `#ClimateAction (climate action)`.
3. Normalise unicode, keep emojis (they carry stance signal), and collapse repeated whitespace.
4. Truncate to 128 tokens for tweets and 256 tokens for VAST comments; log truncation rates.
5. De-duplicate exact (text, target) pairs across splits to prevent leakage.

---

## 8. Module M2: Human Label Distributions from Existing Raw Annotations

**This project collects no new annotations.** All human label distributions come from datasets that already publish each annotator's individual label. This module turns those raw labels into a label distribution λ_x for every item. That distribution is the ground truth for aleatoric uncertainty and the calibration target for the conformal guarantee [11].

### 8.1 Sources

| Source | Ref | What is released | Size (verified Oct 2026) | Role |
|---|---|---|---|---|
| **SemEval-2016 raw stance annotations** | [19, 48] | Every CrowdFlower worker's stance label per tweet (`Worker ID, Instance ID, Target, Tweet, Stance, Opinion towards`) | **4,306** shared-task tweets, **median 8 annotators** per tweet; targets Atheism, Feminist Movement, Hillary Clinton, Legalization of Abortion, Donald Trump | **Primary HLV source** (calibration + test) |
| **SemEval-2016 low-agreement pool** | [19, 48] | Same file: tweets *excluded* from the shared task because annotator agreement was < 60 % | **1,385** tweets, median 8 annotators, mean label entropy **1.445 bits** (maximum possible: 1.585) | **Ambiguity stress test** (never used for training) |
| **ChaosNLI** | [24] | ~100 labels per NLI item | 3,113 SNLI/MNLI items | Secondary check of conformal coverage with dense labels, via the stance-as-NLI reformulation of [21] |
| Gezici et al. news-stance (optional) | [49] | 3 annotators per document, used in [27] | ~1,000 documents | Optional cross-domain check (long news articles) |

> Download: https://saifmohammad.com/WebPages/StanceDataset.htm ("Raw annotations by individual annotators for stance"). The licence forbids redistribution, so the repo ships a download script, not the data. The *Climate Change* target was annotated internally by the organisers and is **not** in the raw file.

### 8.2 Processing the raw annotations

1. **Label mapping.** The raw file has four answers: FAVOR, AGAINST, NEUTRAL, NONE. Map `NEUTRAL → NONE`, matching how the official SemEval-2016 labels were derived [19].
2. **Grouping.** Group rows by `Instance ID` (shared-task tweets). Rows with an empty ID are the low-agreement pool; group those by (target, tweet text).
3. **Minimum-annotator filter.** Keep items with **n ≥ 5** annotators. Report how many are dropped.
4. **Label distribution:**
   ```
   λ_x  = (n_FAVOR, n_AGAINST, n_NONE) / n          # empirical human label distribution
   H(λ_x) = − Σ_k λ_xk log2 λ_xk                    # human disagreement (0 … 1.585 bits)
   majority(x) = argmax_k λ_xk                       # for hard-label metrics
   ```
5. **Small-n correction.** With only ~8 votes, λ_x is a *noisy estimate* of the true human distribution: the standard error per class is up to ≈ 0.18. Report two versions. One is raw λ_x. The other is a Dirichlet-smoothed version, (n_k + 0.5) / (n + 1.5). Also report the conformal results on the n ≥ 10 subset.
6. **Bonus signal.** The `Opinion towards` field (TARGET / OTHER / NO ONE) flags **implicit-stance** tweets, where the opinion is about something other than the target. Use it to slice error and uncertainty analyses (§19.5).

### 8.3 Splits built from the raw annotations

| Split | Construction | Used for |
|---|---|---|
| **HLV-train** | Official SemEval train tweets (4 Task-A targets in the raw file) minus HLV-cal | Training members; also soft-label training (ablation A5) |
| **HLV-cal** | 20 % of official SemEval train, stratified by target; *held out from member training* | Conformal calibration (§13) |
| **HLV-test** | Official SemEval test tweets with raw annotations | Human-alignment (E5) and coverage (E6) |
| **HLV-unseen** | All *Donald Trump* tweets: a target never used in training | Human-labelled **unseen-target** test (E4, E5) |
| **HLV-ambig** | The 1,385-tweet low-agreement pool | Ambiguity stress test: AU should be high, and EU should *not* be driven by it |

### 8.4 Example processed record

(Illustrative text. Real tweets cannot be redistributed.)

```json
{
  "id": "semeval16-raw-XXXX",
  "text": "Oh sure, because 'equality' means only one side gets to talk. Great job, everyone.",
  "target": "Feminist Movement",
  "counts": {"FAVOR": 3, "AGAINST": 4, "NONE": 1},
  "n_annotators": 8,
  "label_dist": [0.375, 0.500, 0.125],
  "label_dist_smoothed": [0.368, 0.474, 0.158],
  "human_entropy_bits": 1.406,
  "opinion_towards": {"TARGET": 5, "OTHER": 2, "NO ONE": 1},
  "split": "hlv-test"
}
```

---

## 9. Module M3: Input Encoding

### 9.1 Encoder path (DeBERTa-v3)

Sentence-pair input. The target goes first, so it is never truncated:

```
[CLS] Climate Change is a Real Concern [SEP] We need to act NOW. The science is settled #ClimateAction [SEP]
```

### 9.2 LLM path (Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct)

```text
<|system|>
You are an expert annotator for stance detection.
<|user|>
Target: {target}
Text: {text}
What is the author's stance towards the target? Answer with exactly one word:
favor, against, or none.
<|assistant|>
Answer:
```

**Classification readout.** Take the next-token logits of the three verbalizer tokens (`" favor"`, `" against"`, `" none"`; check that each is a single token in the chosen tokenizer) and apply a softmax restricted to those three. This gives a proper 3-class probability vector with no free-text generation and no parsing errors.

**Prompt variants** (used only for the zero-shot *prompt-credal* baseline, B9): 8 paraphrased instructions × 3 different few-shot demonstration sets.

---

## 10. Module M4: Backbone Ensembles

The credal set is built from **M diverse members**. Diversity is what makes the credal width meaningful [8, 9, 32].

### 10.1 Path A: Encoder deep ensemble (primary, cheap, fast)

| Setting | Value |
|---|---|
| Backbone | `microsoft/deberta-v3-large` (~435 M params) [29] |
| Members | M = 8, full fine-tuning |
| Diversity sources | Different random seeds (head init + data order); each member trained on an 80 % bootstrap of the training set |
| Head | Linear layer on the `[CLS]` pooled output → 3 logits |
| Loss | Cross-entropy (hard labels). Ablation A5: soft-label cross-entropy where `label_dist` exists |
| Optimiser | AdamW, lr 1e-5, weight decay 0.01, linear warm-up 10 %, linear decay |
| Batch / epochs | 32 / up to 6, early stopping on dev macro-F1 (patience 2) |
| Max length | 128 (tweets), 256 (VAST) |
| Precision | bf16 |

### 10.2 Path B: LLM LoRA ensemble (credal LLM [13, 31])

| Setting | Value |
|---|---|
| Backbone | `Qwen2.5-7B-Instruct` (alternative: `Llama-3.1-8B-Instruct`), frozen |
| Adapters | M = 8 LoRA adapters [30], each trained independently |
| LoRA config | r = 16, α = 32, dropout = 0.05, target modules = all attention + MLP projections |
| Diversity | Seed + 80 % bootstrap + shuffled order; optionally 2 instruction paraphrases per member |
| Loss | Cross-entropy on the restricted 3-token softmax (§9.2) |
| Optimiser | AdamW, lr 1e-4, cosine schedule, warm-up 3 %, 2 epochs |
| Batch | 16 (gradient accumulation as needed) |
| Precision | bf16; 4-bit (QLoRA) optional if memory-bound |

### 10.3 Output of M4

For each input (x, t), M4 returns the matrix of member predictions:

```
P(x, t) ∈ ℝ^{M×K},   row m = p_m = softmax(f_m(x, t)),   K = 3
```

---

## 11. Module M5: Credal Layer

### 11.1 Primary construction: Credal Wrapper [8]

From the M member distributions, take the class-wise extremes:

```
l_k = min_{m=1..M} p_{m,k}        (lower probability of class k)
u_k = max_{m=1..M} p_{m,k}        (upper probability of class k)
𝒞(x) = { p ∈ Δ_3 : l ≤ p ≤ u }
```

Every bound l_k and u_k is attained by some member p_m, and each p_m ∈ 𝒞. The bounds are therefore *reachable*, and no extra tightening step is needed.

### 11.2 Point prediction: intersection probability [8]

To get one representative distribution from the credal set (for F1 and ECE):

```
α  = (1 − Σ_k l_k) / Σ_k (u_k − l_k)
p*_k = l_k + α · (u_k − l_k)
ŷ = argmax_k p*_k
```

### 11.3 Alternative constructions (compared in ablation A2)

| Variant | Ref | Idea | Cost |
|---|---|---|---|
| **CW-Ens** (primary) | [8] | min/max over an ensemble | M forward passes |
| **CW-MCD** | [8, 33] | min/max over M MC-Dropout samples of one model | 1 model, M passes |
| **CreDE** | [9] | Each network directly predicts lower/upper class probabilities, trained with a loss inspired by distributionally robust optimisation (use the authors' official code) | M interval networks |
| **CreINN** | [10] | Interval-weight network outputs probability intervals | 1 interval network |
| **EffCre** | [12] | Credal set from a *single* model via "decalibration" (no ensemble) | 1 model |
| **Credal distillation** | [15] | Distil the ensemble credal set into one student | 1 pass at inference |

---

## 12. Module M6: Uncertainty Decomposition

### 12.1 Measures

| Measure | Definition | Computation (K = 3) |
|---|---|---|
| **TU** (upper entropy) | H̄ = max_{p∈𝒞} H(p) | Concave maximisation over a box-constrained simplex. The solution is p_k = clip(τ, l_k, u_k), with τ found by bisection so that Σ p_k = 1 ("water-filling", from the KKT conditions) |
| **AU** (lower entropy) | H̲ = min_{p∈𝒞} H(p) | Entropy is concave, so the minimum is at a **vertex** of the polytope. Enumerate the vertices (≤ K·2^{K−1} = 12 candidates for K = 3) |
| **EU** | H̄ − H̲ | Difference of the two |
| Width (diagnostic) | W = Σ_k (u_k − l_k) | Closed form |
| Ensemble MI (baseline) | H(p̄) − (1/M) Σ_m H(p_m) | Standard Bayesian decomposition [6, 32]; used only for comparison |

All entropies are in **bits** (log₂), so the maximum for K = 3 is log₂3 ≈ 1.585.

### 12.2 Interpretation table

| AU | EU | Meaning | Typical cause |
|---|---|---|---|
| low | low | Confident, clear | Explicit stance on a seen target |
| high | low | **Ambiguous / contested** | Sarcasm, mixed stance; humans also disagree |
| low / any | high | **Unfamiliar / out-of-knowledge** | Unseen target, new domain, emerging event |
| high | high | Both | Ambiguous text on an unseen target |

---

## 13. Module M7: Conformal Credal Calibration

Ensemble credal sets have **no formal guarantee**. Following Javanmardi, Stutz & Hüllermeier [11], we calibrate them on HLV-cal (§8.3), where the target is the full **human label distribution** λ_x rather than a single label.

### 13.1 Procedure (split conformal, first-order predictor)

1. **Predictor:** g(x) = p*(x), the intersection probability from M5.
2. **Non-conformity score** on each calibration item (x_i, λ_i) ∈ HLV-cal:
   ```
   s_i = TV(λ_i, g(x_i)) = ½ Σ_k |λ_ik − g_k(x_i)|
   ```
   ([11] also evaluates Wasserstein, KL and 1 − inner product. We compare TV and KL in ablation A4.)
3. **Quantile:** with n calibration items and miscoverage α (default 0.1):
   ```
   q̂ = the ⌈(n+1)(1−α)⌉-th smallest value of {s_1,…,s_n}
   ```
4. **Conformal credal set** for a new input:
   ```
   Ĉ(x) = { λ ∈ Δ_3 : TV(λ, g(x)) ≤ q̂ }
   ```
5. **Guarantee** (exchangeability; [11, 36]): P( λ_x ∈ Ĉ(x) ) ≥ 1 − α.

### 13.2 Closed-form bounds of the TV ball

The class-wise lower and upper probabilities of a TV ball of radius q̂ around g are:

```
l̂_k = max(0, g_k − q̂)
û_k = min(1, g_k + q̂)
```

These calibrated bounds feed the decision layer (M8). This combines the *shape* information from the ensemble (through g) with a *guaranteed size* (through q̂).

### 13.3 Label-level conformal sets (secondary)

For datasets with only hard labels, we also report standard conformal prediction *sets* of labels (LAC and APS [36, 37]) as a baseline, and compare their size and coverage with sets induced by credal dominance.

---

## 14. Module M8: Decision and Routing Layer

### 14.1 Interval dominance [17]

Class j **dominates** class k if l_j > u_k, meaning j is more probable than k under *every* distribution in the credal set. The **non-dominated set** is:

```
ND(x) = { k : u_k ≥ max_j l_j }
```

### 14.2 Routing policy

```
if |ND(x)| == 1:
        if AU(x) ≤ τ_AU:              → PREDICT ŷ                       (confident)
        else:                          → PREDICT ŷ + flag "contested"
elif EU(x) > τ_EU:                     → DEFER to human expert / request labels for target
else:                                  → SET-VALUED output ND(x) + flag "ambiguous"
```

- τ_EU and τ_AU are chosen on the dev set to meet a deferral budget (e.g. ≤ 10 % of traffic deferred).
- **Cost-sensitive variant (E7):** the costs are c_err = 1 for a wrong single label, c_defer = 0.3, and c_set = 0.15 × (|ND| − 1). Report the expected cost against the deferral rate.

### 14.3 Why this matters downstream

- **Content moderation / misinformation pipelines** [41]: unsure cases go to human reviewers.
- **Opinion analytics:** do not count "contested" texts as clean votes.
- **Active learning for new targets:** high-EU items are the most informative ones to label (ablation A6).

---

## 15. Module M9 (Stretch): Concept-Level Credal Network

This module is optional. It is included for interpretability, and because it uses a *classical* credal network [2] where one fits: on **discrete concepts extracted from text**, not on raw text.

### 15.1 Concept extraction (LLM-based, discrete)

| Concept node | Values |
|---|---|
| `TargetMention` | explicit / implicit / absent |
| `Sentiment` (towards target) | positive / negative / neutral |
| `Sarcasm` | yes / no |
| `Negation` | yes / no |
| `StanceHashtag` | pro / con / none |

### 15.2 Credal network

- **Structure:** a fixed DAG with the concept nodes as parents of `Stance`; `Sarcasm` also points to `Sentiment`'s effect on `Stance`.
- **Parameters:** imprecise conditional probability tables learnt with the **Imprecise Dirichlet Model** (s = 2) [46]. Sparse configurations get wide intervals automatically.
- **Inference:** exact inference with the CREMA library [43]. Output: lower and upper P(Stance | concepts).

### 15.3 Use

Explain *why* a prediction is uncertain. For example: "Sarcasm = yes, and P̲(FAVOR | concepts) = 0.21 but P̄(FAVOR | concepts) = 0.64: this configuration is rare in training." Compare its credal intervals with the neural credal set on the same items.

---

## 16. Module M10: Explanation Dashboard

A small web demo (Gradio or Streamlit):

```
Target: [ Climate Change is a Real Concern ]
Text:   [ Oh great, another 'climate emergency' summit... ]

             lower ────────── upper
FAVOR    0.42 |███████▌ |  0.45
AGAINST  0.40 |███████  |  0.43
NONE     0.14 |██▌      |  0.15

Aleatoric (AU): ███████████████░  1.44 bits   ← ambiguous
Epistemic (EU): ▏                0.02 bits   ← familiar target
Decision: SET-VALUED {FAVOR, AGAINST}  · flag: contested
Ensemble members: [table of 8 member predictions]
```

---

## 17. End-to-End Worked Examples

The numbers below are computed with the reference code in §23 (M = 5 members for brevity).

### Example A: clear stance, seen target

*Target:* Climate Change is a Real Concern · *Text:* "We need to act NOW. The science is settled #ClimateAction"

| Member | FAVOR | AGAINST | NONE |
|---|---|---|---|
| p₁ | 0.90 | 0.05 | 0.05 |
| p₂ | 0.88 | 0.07 | 0.05 |
| p₃ | 0.92 | 0.04 | 0.04 |
| p₄ | 0.89 | 0.06 | 0.05 |
| p₅ | 0.91 | 0.05 | 0.04 |
| **l** | 0.88 | 0.04 | 0.04 |
| **u** | 0.92 | 0.07 | 0.05 |
| **p\*** | 0.900 | 0.055 | 0.045 |

TU = 0.647, AU = 0.482, **EU = 0.165** bits. ND = {FAVOR} → **PREDICT FAVOR**.

### Example B: sarcasm (aleatoric)

*Text:* "Oh great, another 'climate emergency' summit. I'm sure the private jets will fix it."

| Member | FAVOR | AGAINST | NONE |
|---|---|---|---|
| p₁ | 0.44 | 0.41 | 0.15 |
| p₂ | 0.42 | 0.43 | 0.15 |
| p₃ | 0.45 | 0.40 | 0.15 |
| p₄ | 0.43 | 0.42 | 0.15 |
| p₅ | 0.44 | 0.42 | 0.14 |
| **l** | 0.42 | 0.40 | 0.14 |
| **u** | 0.45 | 0.43 | 0.15 |
| **p\*** | 0.437 | 0.417 | 0.146 |

TU = 1.460, **AU = 1.443**, **EU = 0.017** bits. ND = {FAVOR, AGAINST} and EU is low → **SET-VALUED {FAVOR, AGAINST}, flagged "contested"**. A softmax model would simply output FAVOR with p = 0.44 and give no indication that the text is contested. On real data, this behaviour is checked against λ_x from the SemEval-2016 raw annotations: items like the record in §8.4 (λ = [0.375, 0.500, 0.125], H = 1.41 bits) are exactly where annotators disagree.

### Example C: unseen target (epistemic)

*Target:* Digital euro (unseen) · *Text:* "Finally a way to pay without banks taking a cut. Bring it on."

| Member | FAVOR | AGAINST | NONE |
|---|---|---|---|
| p₁ | 0.80 | 0.10 | 0.10 |
| p₂ | 0.15 | 0.75 | 0.10 |
| p₃ | 0.30 | 0.20 | 0.50 |
| p₄ | 0.70 | 0.20 | 0.10 |
| p₅ | 0.20 | 0.65 | 0.15 |
| **l** | 0.15 | 0.10 | 0.10 |
| **u** | 0.80 | 0.75 | 0.50 |
| **p\*** | 0.399 | 0.349 | 0.253 |

TU = 1.585, AU = 0.922, **EU = 0.663** bits. ND = {FAVOR, AGAINST, NONE} and EU > τ_EU → **DEFER** and add "Digital euro" to the active-learning queue. Note that averaging the ensemble gives [0.43, 0.38, 0.19]. That looks like a mild FAVOR and hides the fact that the members disagree.

### Example D: conformal calibration

Suppose g(x) = [0.70, 0.20, 0.10] and the calibrated radius is q̂ = 0.15. Then l̂ = [0.55, 0.05, 0.00] and û = [0.85, 0.35, 0.25]. FAVOR dominates (0.55 > 0.35), so the system **PREDICTS FAVOR**, and with ≥ 90 % probability the true human distribution lies within TV distance 0.15 of [0.70, 0.20, 0.10].

---

## 18. Baselines

| ID | Baseline | Ref | Notes |
|---|---|---|---|
| B1 | Single model, softmax confidence | – | Max-probability and entropy scores |
| B2 | Temperature scaling | [34] | Fitted on dev |
| B3 | MC Dropout (M = 8 samples) | [33] | Predictive entropy + MI |
| B4 | Deep Ensemble (M = 8), mean prediction | [32] | Predictive entropy + MI; **the key control**: same members as CrediStance |
| B5 | Evidential deep learning (Dirichlet head) | [35] | Vacuity as EU |
| B6 | Label-level conformal (LAC, APS) | [36, 37] | Set size / coverage |
| B7 | LoRA ensemble (mean) | [31] | LLM-path equivalent of B4 |
| B8 | LLM verbalised confidence / sampling consistency | [44] | Zero-shot LLM |
| B9 | Prompt-credal (zero-shot LLM, 24 prompt × demo variants) | ours | No fine-tuning: tests whether prompt variation alone gives useful credal sets |
| B10 | FACTUAL calibration | [28] | Stance-specific LLM debiasing/calibration |
| B11 | Multi-perspective soft-label training | [27] | Human-label-variation baseline |

**Fairness rule.** B4 and CrediStance use **exactly the same trained members**. Any gain then comes from the credal representation and decision rule, not from extra compute.

---

## 19. Evaluation Protocol and Metrics

### 19.1 Predictive quality

| Metric | Definition | Notes |
|---|---|---|
| F_avg (SemEval) | (F1_FAVOR + F1_AGAINST) / 2 | Official SemEval-2016 metric [19] |
| Macro-F1 | Mean F1 over 3 classes | VAST, EZ-Stance |
| Accuracy | – | – |

### 19.2 Calibration

| Metric | Definition |
|---|---|
| ECE | 15 equal-width bins on max p* |
| NLL | −log p*_y |
| Brier | Σ_k (p*_k − 1[y=k])² |

### 19.3 Uncertainty quality

| Metric | Positive class | Score |
|---|---|---|
| Misclassification AUROC / AUPR | ŷ ≠ y | TU (and EU, AU separately) |
| **Unseen-target AUROC** | item from an unseen target | EU (H2a) |
| AURC (risk–coverage) [38] | – | Reject by TU; area under the risk–coverage curve |

### 19.4 Set-valued (credal) prediction quality [16]

| Metric | Definition |
|---|---|
| Determinacy | Fraction of items with \|ND\| = 1 |
| Single accuracy | Accuracy on determinate items |
| Set accuracy | Fraction where y ∈ ND on indeterminate items |
| **u65 / u80** | Utility-discounted accuracy: u65 = 1.6/\|S\| − 0.6/\|S\|² and u80 = 2.2/\|S\| − 1.2/\|S\|² when y ∈ S, else 0 [16] |

### 19.5 Human alignment (HLV-test, HLV-unseen, HLV-ambig)

| Metric | Expected direction |
|---|---|
| Spearman ρ(AU, H(λ_x)) | Positive and significant (H2b) |
| Spearman ρ(EU, H(λ_x)) | Weaker than ρ(AU, ·) (discriminant validity) |
| JSD(p*, λ_x), TV(p*, λ_x) | Lower is better |
| Human-entropy-binned accuracy | Accuracy should drop as human entropy rises |
| AU on HLV-ambig vs. HLV-test | AU significantly higher on the low-agreement pool |
| Implicit-stance slice | Uncertainty broken down by `Opinion towards` = TARGET vs. OTHER |

### 19.6 Conformal validity

| Metric | Definition |
|---|---|
| Empirical coverage | Fraction of HLV-test items with TV(λ_x, g(x)) ≤ q̂ (target 1−α for α ∈ {0.05, 0.1, 0.2}) |
| Efficiency | Mean q̂ (smaller is better) and mean \|ND\| after calibration |
| Conditional coverage | Coverage per source / per human-entropy bin |

### 19.7 Statistical rigour

- **3 independent ensemble pools** (3 × 8 members with disjoint seeds). Report mean ± std.
- **95 % bootstrap CIs** (1,000 resamples) on all test metrics.
- **Paired permutation tests** (10,000 permutations) between CrediStance and B4, with Holm–Bonferroni correction across datasets.
- Conformal results are averaged over **100 random calibration/test re-splits** of HLV.

---

## 20. Experiment Matrix

| Exp | Question | Data | Systems | Primary metrics |
|---|---|---|---|---|
| **E1** | Does credal hurt accuracy in-target? | SemEval-16 in-target | B1–B6, CrediStance-A | F_avg, ECE, AURC |
| **E2** | Cross-target reliability | SemEval-16 LOTO (5 folds) | B1–B6, CrediStance-A | Macro-F1, misclassification AUROC, AURC |
| **E3** | Zero-shot reliability | VAST, EZ-Stance (target- and domain-based) | B1–B6, CrediStance-A/B | Macro-F1, ECE, AURC, u65/u80 |
| **E4** | Does EU detect unseen targets? | Seen/unseen pool | EU vs. MI (B4), vacuity (B5), max-prob (B1) | Unseen-target AUROC |
| **E5** | Does AU match human disagreement? | HLV-test, HLV-unseen, HLV-ambig | All | ρ(AU, H(λ)), JSD |
| **E6** | Conformal coverage | HLV-cal → HLV-test | CrediStance vs. conformal on B1/B4 scores | Coverage, efficiency |
| **E7** | Decision value | All test sets | Credal routing vs. confidence thresholding | Expected cost vs. deferral rate |
| **E8** | LLM path | VAST, EZ-Stance, HLV-test | CrediStance-B vs. B7–B10 | All of the above |

---

## 21. Ablation Studies

| ID | Ablation | Values |
|---|---|---|
| A1 | Ensemble size M | 2, 4, 8, 16 |
| A2 | Credal construction | CW-Ens, CW-MCD, CreDE, CreINN, EffCre, distilled |
| A3 | Diversity source | seed-only / bootstrap-only / both / + prompt paraphrase (LLM) |
| A4 | Conformal score | TV, KL, Wasserstein, 1 − inner product [11] |
| A5 | Training targets | hard labels vs. soft human labels λ_x (HLV-train) |
| A6 | Active learning on new targets | EU-sampling vs. TU vs. random, 50/100/200-label budgets on a held-out EZ-Stance target |
| A7 | Backbone | DeBERTa-v3-large vs. BERTweet-large [47] vs. Qwen2.5-7B vs. Llama-3.1-8B |
| A8 | Uncertainty measure | Entropy decomposition vs. width W vs. volume [7] |

---

## 22. Implementation Details

### 22.1 Tech stack

| Purpose | Tool |
|---|---|
| Deep learning | PyTorch 2.x |
| Models | Hugging Face `transformers`, `peft` (LoRA), `accelerate`, `bitsandbytes` (optional QLoRA) |
| Credal core | NumPy (own module, §23) |
| Classical credal network (M9) | CREMA (Java) [43] |
| Config | Hydra |
| Tracking | Weights & Biases or MLflow |
| Stats | `scipy.stats`, `scikit-learn` |
| Demo | Gradio |

### 22.2 Repository layout

```
credistance/
├── configs/                  # Hydra configs (data, model, credal, conformal, exp)
├── data/
│   ├── raw/                  # original releases (gitignored)
│   ├── processed/            # unified JSONL (§7.3)
│   └── hlv/                  # download script for SemEval-16 raw annotations + aggregated λ_x
├── credistance/
│   ├── data/                 # loaders, label mapping, splits, preprocessing
│   ├── models/
│   │   ├── encoder.py        # DeBERTa member
│   │   └── llm_lora.py       # LoRA member + verbalizer readout
│   ├── train_ensemble.py     # trains M members, saves logits
│   ├── credal/
│   │   ├── core.py           # bounds, intersection prob, entropies, dominance (§23)
│   │   ├── conformal.py      # nonconformity scores, quantile, TV-ball bounds
│   │   └── variants/         # CreDE, CreINN, EffCre adapters
│   ├── decision/routing.py   # §14 policy + cost evaluation
│   ├── eval/                 # metrics (§19), bootstrap, permutation tests
│   └── concept_cn/           # M9 stretch module
├── scripts/                  # run_e1.sh … run_e8.sh
├── notebooks/                # analysis & figures
├── demo/app.py               # Gradio dashboard
└── tests/                    # unit tests for credal core (vertex enumeration, entropies)
```

**Design principle:** members are trained once and their **logits are cached** to disk (`logits/{dataset}/{split}/member_{m}.npy`). Every credal, conformal and decision experiment then runs on CPU in seconds, which makes the ablations cheap.

### 22.3 Compute estimate (rough)

| Job | Hardware | Estimate |
|---|---|---|
| 8 × DeBERTa-v3-large on SemEval / VAST / EZ-Stance | 1 × A100-80GB | ~1–3 GPU-h per member per dataset |
| 8 × LoRA adapters on Qwen2.5-7B | 1 × A100-80GB | ~3–8 GPU-h per adapter per dataset |
| 3 ensemble pools (for variance) | – | Multiply the above by 3 |
| Credal / conformal / decision experiments | CPU | Minutes (cached logits) |

> Treat these as planning figures only. Benchmark one member early and rescale.

### 22.4 Reproducibility checklist

- [ ] Fixed seeds; log package versions (`pip freeze`)
- [ ] Cache and version all member logits
- [ ] Release the λ_x aggregation script (not the data: the SemEval licence forbids redistribution)
- [ ] Report every hyperparameter searched and the search budget
- [ ] Unit tests: upper entropy = log₂3 on the full simplex; lower entropy = 0 on the full simplex; p* sums to 1; coverage ≥ 1−α on synthetic exchangeable data

---

## 23. Reference Implementation of the Credal Core

Tested NumPy implementation (Python ≥ 3.9, NumPy only). It produces the numbers in §17.

```python
import itertools
import numpy as np

EPS = 1e-12

def entropy(p):
    p = np.clip(p, EPS, 1.0)
    return float(-(p * np.log2(p)).sum())

def credal_bounds(P):
    """P: (M, K) ensemble member probabilities -> lower, upper (K,)"""
    return P.min(axis=0), P.max(axis=0)

def intersection_probability(l, u):
    w = u - l
    if w.sum() < EPS:
        return l / l.sum()
    alpha = (1.0 - l.sum()) / w.sum()
    return l + alpha * w

def vertices(l, u):
    """Vertices of {p : l <= p <= u, sum p = 1} (small K): fix K-1 coords at a bound, solve the last."""
    K = len(l); V = []
    for free in range(K):
        others = [j for j in range(K) if j != free]
        for choice in itertools.product([0, 1], repeat=K - 1):
            p = np.empty(K)
            for j, c in zip(others, choice):
                p[j] = u[j] if c else l[j]
            p[free] = 1.0 - p[others].sum()
            if l[free] - 1e-9 <= p[free] <= u[free] + 1e-9:
                V.append(p)
    return np.unique(np.round(np.array(V), 10), axis=0)

def upper_entropy(l, u, iters=100):
    # max-entropy point of a probability-interval credal set is p_k = clip(t, l_k, u_k)
    # ("water-filling", from the KKT conditions); find t by bisection so that sum(p) = 1
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        t = (lo + hi) / 2
        if np.clip(t, l, u).sum() > 1.0:
            hi = t
        else:
            lo = t
    return entropy(np.clip((lo + hi) / 2, l, u))

def lower_entropy(l, u):
    # entropy is concave -> minimum over a polytope is attained at a vertex
    return min(entropy(v) for v in vertices(l, u))

def decompose(l, u):
    H_up, H_lo = upper_entropy(l, u), lower_entropy(l, u)
    return {"TU": H_up, "AU": H_lo, "EU": H_up - H_lo}

def tv_ball_bounds(g, q):
    """Class-wise lower/upper probabilities of the TV ball {p : TV(p,g) <= q}."""
    return np.maximum(0.0, g - q), np.minimum(1.0, g + q)

def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[min(k, n) - 1]

def nondominated(l, u):
    """Interval dominance: keep class k unless some j has l_j > u_k."""
    return [k for k in range(len(l)) if u[k] >= l.max()]
```

**Usage on cached logits:**

```python
logits = np.stack([np.load(f"logits/vast/test/member_{m}.npy") for m in range(8)])  # (M, N, K)
P = np.exp(logits - logits.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)

for i in range(P.shape[1]):
    l, u = credal_bounds(P[:, i, :])
    p_star = intersection_probability(l, u)
    unc = decompose(l, u)
    nd = nondominated(l, u)
```

**Conformal calibration on HLV:**

```python
def tv(a, b): return 0.5 * np.abs(a - b).sum(-1)

scores = tv(lambda_cal, g_cal)           # (n,)
q_hat = conformal_quantile(scores, alpha=0.1)
l_hat, u_hat = tv_ball_bounds(g_test_i, q_hat)
coverage = (tv(lambda_test, g_test) <= q_hat).mean()
```

---

## 24. Six-Month Timeline

| Weeks | Work package | Deliverable / milestone |
|---|---|---|
| **1–2** | Systematic literature review (confirm novelty); download all datasets incl. SemEval-16 raw annotations; set up repo, configs, logging | Lit-review document; data inventory |
| **3–4** | Data pipeline (§7) + raw-annotation processing (§8): λ_x, HLV splits, agreement stats; B1–B3 baselines on SemEval | Baseline table; HLV statistics |
| **5–6** | Train DeBERTa ensemble pools (in-target, LOTO, HLV-train, VAST, EZ-Stance); cache logits | Cached logits for Path A |
| **7–8** | Credal core (§23) + unit tests; E1, E2, E3 for Path A; B4–B6 | **Milestone 1:** first credal vs. baseline results |
| **9–10** | E4 unseen-target detection; decision layer (§14); E7 cost curves | Routing results |
| **11–12** | E5 human alignment (HLV-test / unseen / ambig); E6 conformal coverage (+ ChaosNLI check) | **Mid-term review:** working system + coverage guarantee |
| **13–15** | Path B: LoRA ensembles on Qwen2.5-7B; B7–B10; E8 | LLM credal results |
| **16–18** | Ablations A1–A8; credal variants (CreDE / EffCre); 3-pool variance runs; significance tests | Full results tables |
| **19–20** | Error analysis by human-entropy bin and `Opinion towards` slice (implicit stance); M9 stretch if time; Gradio demo | Demo v1; qualitative analysis |
| **21–23** | Paper writing; figures; dataset card + datasheet; code clean-up | Paper draft |
| **24–26** | Internal review, revisions, submission (ARR / UncertaiNLP / WASSA); final report and viva | **Final:** paper submitted, code + dataset released, final presentation |

---

## 25. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Prior credal stance work appears | Low–Med | High | Weeks 1–2 literature check; the credal decomposition + conformal + decision layer for stance stay novel |
| Credal gives no F1 gain | High | Low | The claim is about reliability (AURC, unseen-target AUROC, coverage), not F1 |
| Ensemble members too similar → narrow credal sets | Med | Med | Bootstrap + seed + (LLM) prompt diversity; A3 ablation; CreDE (trained intervals) |
| Only ~8 votes per item → noisy λ_x | High | Med | Dirichlet smoothing; report the n ≥ 10 subset; confirm on ChaosNLI (~100 votes) |
| Raw annotations cover only SemEval targets | Certain | Med | Human-alignment claims are scoped to SemEval; VAST/EZ-Stance are evaluated with hard-label metrics only |
| Tweet hydration losses | Med | Low | Use the versions currently available and report counts; VAST/EZ-Stance are less affected |
| LLM ensemble too slow | Low (no resource limit) | Med | QLoRA; EffCre single-model variant [12]; credal distillation [15] |
| Conformal exchangeability violated (calibrate on one source, test on another) | Med | Med | Calibrate per source; report conditional coverage; discuss as a limitation |

---

## 26. Ethical Considerations

- **No individual profiling.** The system classifies the stance of *texts*. It must not be used to infer or score the beliefs of identifiable people. Large-scale stance inference raises surveillance risks, and the SICI analysis [42] explicitly cautions against inferring beliefs from sparse text.
- **Uncertainty as a safeguard.** The defer and "contested" outputs are designed to *reduce* harmful over-confident labelling of political speech.
- **Perspectivism.** Disagreement is treated as signal, not noise [25, 26]. No new human data is collected. Raw annotator IDs in SemEval are already anonymised and are not used to profile workers.
- **Data licences.** Respect the original dataset licences. Release tweets as IDs only, following platform terms. Do not redistribute the SemEval raw annotations; point users to the official page.

---

## 27. Expected Contributions and Deliverables

1. **CrediStance:** to our knowledge, the first credal uncertainty framework for stance detection, with an aleatoric/epistemic decomposition and imprecise-probability decision making.
2. **A human-label-distribution benchmark for stance** built from the existing SemEval-2016 raw annotations (HLV-test / unseen / ambig splits). To our knowledge, these raw annotations have not been used to evaluate stance uncertainty before. We release the processing scripts.
3. **Conformal credal guarantees** for stance: coverage of the *human label distribution*, not just a single label.
4. **An empirical study** across in-target, cross-target and zero-shot settings, and across encoder and LLM backbones, against 11 baselines.
5. **Open-source code** (cached-logit pipeline, credal core with tests) and an **interactive demo**.

**Target venues:** ACL / EMNLP / NAACL (via ARR); workshops UncertaiNLP and WASSA; dataset track options (e.g. LREC-COLING).

---

## 28. References

**Imprecise probabilities and credal sets**

[1] P. Walley. *Statistical Reasoning with Imprecise Probabilities*. Chapman & Hall, 1991.
[2] F. G. Cozman. Credal networks. *Artificial Intelligence*, 120(2):199–233, 2000.
[3] M. Zaffalon. The naive credal classifier. *Journal of Statistical Planning and Inference*, 105(1):5–21, 2002.
[4] G. Corani, M. Zaffalon. Learning reliable classifiers from small or incomplete data sets: the Naive Credal Classifier 2. *JMLR*, 9:581–621, 2008. https://jmlr.org/papers/v9/corani08a.html
[5] J. Abellán, G. J. Klir, S. Moral. Disaggregated total uncertainty measure for credal sets. *International Journal of General Systems*, 35(1):29–44, 2006.
[6] E. Hüllermeier, W. Waegeman. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. *Machine Learning*, 110:457–506, 2021.
[7] Y. Sale, M. Caprio, E. Hüllermeier. Is the volume of a credal set a good measure for epistemic uncertainty? *UAI 2023*, PMLR 216. https://arxiv.org/abs/2306.09586

**Credal deep learning**

[8] K. Wang, F. Cuzzolin, et al. Credal wrapper of model averaging for uncertainty estimation in classification. *ICLR 2025*. https://arxiv.org/abs/2405.15047
[9] K. Wang, F. Cuzzolin, S. Kudukkil Manchingal, K. Shariatmadar, D. Moens, H. Hallez. Credal deep ensembles for uncertainty quantification. *NeurIPS 2024*. https://proceedings.neurips.cc/paper_files/paper/2024/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract.html
[10] K. Wang et al. CreINNs: Credal-set interval neural networks for uncertainty estimation in classification tasks. *Neural Networks*, 2025. https://arxiv.org/abs/2401.05043
[11] A. Javanmardi, D. Stutz, E. Hüllermeier. Conformalized credal set predictors. *NeurIPS 2024*. https://arxiv.org/abs/2402.10723
[12] P. Hofman, T. Löhr, M. Muschalik, Y. Sale, E. Hüllermeier. Efficient credal prediction through decalibration. *ICLR 2026*. https://proceedings.iclr.cc/paper_files/paper/2026/file/9c76b2ec87a0a62d256e4e4c01e3d8dd-Paper-Conference.pdf
[13] S. Kudukkil Manchingal, S. Nikolenko, F. Cuzzolin. Credal large language models for semantic commitment under uncertainty. *arXiv:2608.23244*, 2026.
[14] Credal Transformer: a principled approach for quantifying and mitigating hallucinations in large language models. *arXiv:2510.12137*, 2025.
[15] Credal ensemble distillation for uncertainty quantification. *arXiv:2511.13766*, 2025.
[16] M. Zaffalon, G. Corani, D. Mauá. Evaluating credal classifiers by utility-discounted predictive accuracy. *International Journal of Approximate Reasoning*, 53(8):1282–1301, 2012.
[17] M. C. M. Troffaes. Decision making under uncertainty using imprecise probabilities. *International Journal of Approximate Reasoning*, 45(1):17–29, 2007.
[18] *(reserved)*

**Stance detection**

[19] S. M. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, C. Cherry. SemEval-2016 Task 6: Detecting stance in tweets. *SemEval 2016*.
[20] E. Allaway, K. McKeown. Zero-shot stance detection: a dataset and model using generalized topic representations. *EMNLP 2020*. https://arxiv.org/abs/2010.03640
[21] C. Zhao, C. Caragea. EZ-STANCE: a large dataset for English zero-shot stance detection. *ACL 2024*. https://aclanthology.org/2024.acl-long.838
[22] Y. Li, T. Sosea, A. Sawant, A. J. Nair, D. Inkpen, C. Caragea. P-Stance: a large dataset for stance detection in political domain. *Findings of ACL 2021*.
[23] K. Glandt, S. Khanal, Y. Li, D. Caragea, C. Caragea. Stance detection in COVID-19 tweets. *ACL 2021*.
[40] D. Küçük, F. Can. Stance detection: a survey. *ACM Computing Surveys*, 53(1), 2020.
[41] A survey of stance detection on social media: new directions and perspectives. *arXiv:2409.15690*, 2024.
[42] SICI: a semantic-pragmatic complexity index reveals regime shifts in LLM stance detection. *arXiv:2606.13189*, 2026.
[28] A. Li et al. Mitigating biases of large language models in stance detection with counterfactual augmented calibration (FACTUAL). *arXiv:2402.14296*.

**Human label variation**

[24] Y. Nie, X. Zhou, M. Bansal. What can we learn from collective human opinions on natural language inference data? (ChaosNLI). *EMNLP 2020*. https://arxiv.org/abs/2010.03532
[25] B. Plank. The "problem" of human label variation: on ground truth in data, modeling and evaluation. *EMNLP 2022*.
[26] A. Uma, T. Fornaciari, D. Hovy, S. Paun, B. Plank, M. Poesio. Learning from disagreement: a survey. *JAIR*, 72:1385–1470, 2021.
[27] B. Muscato, P. Bushipaka, G. Gezici, L. Passaro, F. Giannotti. Multi-perspective stance detection. *arXiv:2411.08752*, 2024.
[39] K. Krippendorff. *Content Analysis: An Introduction to Its Methodology*. Sage, 2004.

**Models and uncertainty baselines**

[29] P. He, J. Gao, W. Chen. DeBERTaV3: improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*.
[30] E. J. Hu et al. LoRA: low-rank adaptation of large language models. *ICLR 2022*.
[31] X. Wang, L. Aitchison, M. Rudolph. LoRA ensembles for large language model fine-tuning. *arXiv:2310.00035*, 2023.
[32] B. Lakshminarayanan, A. Pritzel, C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*.
[33] Y. Gal, Z. Ghahramani. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML 2016*.
[34] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. *ICML 2017*.
[35] M. Sensoy, L. Kaplan, M. Kandemir. Evidential deep learning to quantify classification uncertainty. *NeurIPS 2018*.
[36] A. N. Angelopoulos, S. Bates. Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 2023.
[37] Y. Romano, M. Sesia, E. Candès. Classification with valid and adaptive coverage (APS). *NeurIPS 2020*.
[38] Y. Geifman, R. El-Yaniv. Selective classification for deep neural networks. *NeurIPS 2017*.
[43] D. Huber, R. Cabañas, A. Antonucci, M. Zaffalon. CREMA: a Java library for credal network inference. *PGM 2020*.
[44] M. Xiong et al. Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR 2024*.
[46] P. Walley. Inferences from multinomial data: learning about a bag of marbles. *Journal of the Royal Statistical Society B*, 58(1):3–57, 1996.
[47] D. Q. Nguyen, T. Vu, A. T. Nguyen. BERTweet: a pre-trained language model for English tweets. *EMNLP 2020 (System Demonstrations)*.
[48] S. M. Mohammad, P. Sobhani, S. Kiritchenko. Stance and sentiment in tweets. *ACM Transactions on Internet Technology*, 17(3), 2017. Raw annotations: https://saifmohammad.com/WebPages/StanceDataset.htm
[49] G. Gezici, A. Lipani, Y. Saygın, E. Yilmaz. Evaluation metrics for measuring bias in search engine results. *Information Retrieval Journal*, 24:85–113, 2021.

> **Citation hygiene.** Before final submission, re-check every entry against DBLP or the ACL Anthology: page numbers, final venue (several 2025–2026 works are still arXiv preprints), and the full author lists for [8], [10], [14], [15], [28], [41], [42], [44] and [49].
