# Autonomous Machine Learning Research Agent for Recommender Systems

> TechJam problem statement (Problem #2). Last updated by organizers: 26 August 2026, 6:33 PM.
>
> Notes from organizers:
> - Added downloadable `kuairand-starter-kit.zip` under "Starter Kit".
> - Technical Workshop Webinar with Q&A: 28 Aug, 2:00–2:45 PM.

---

## 2.1 Background

### Motivation

Machine learning engineers (MLEs) spend much of their time on a single activity: taking a dataset and a set of metrics, then iterating on a model again and again to push the score higher. This work is inherently cyclic — every round repeats the same loop.

**Figure 1. The MLE iteration loop.** A closed cycle of five core stages, plus a reflection step that feeds the next round:

1. **Read the problem** — understand the given dataset and the target metrics.
2. **Inspect data** — study data distribution through exploratory data analysis (EDA).
3. **Engineer features** — build and select input features (see Appendix A.5).
4. **Train + tune** — choose a model, set the loss function, and tune hyperparameters.
5. **Evaluate** — read the metrics, check for overfitting, and consult the leaderboard.

The result of the evaluate stage drives a **reflect + revise** step, which decides what to change and loops back into the next iteration — re-inspecting the data and adjusting the features. The cycle repeats until the score plateaus.

Two of these stages — **engineer features** and **train + tune** — are carried out almost entirely in code. Each turn of the loop produces and modifies code, which is what makes the loop a natural target for automation: it is structured and repeatable, yet writing and revising that code is exactly the kind of task a code-generating LLM can take on.

### Prior Work

Over the past two years, a new line of work has set out to automate this loop: the **Autonomous ML Research Agent**, an LLM-driven agent that runs the cycle on its own. Representative systems:

- **MLE-Bench** [1] (OpenAI) — a benchmark of 75 Kaggle competitions, now a standard evaluation suite for such agents.
- **AIDE** [2] (Weco AI) — a state-of-the-art agent that frames ML engineering as code optimization and explores the solution space via tree search.
- **AI-Scientist-v2** [3] (Sakana AI) — an end-to-end agent for autonomous scientific and ML research, using agentic tree search to form hypotheses, run experiments, and write up results. *(This is the base framework this repo builds on.)*

### This Challenge

Design an autonomous ML research agent. Given a public ML dataset and a set of metrics, the agent must autonomously run the full loop of Figure 1 — read the problem, engineer features, train and tune the model, evaluate, then reflect and iterate — to reach the highest possible score across the test sets. Writing the code for each stage is part of the agent's job, not something provided in advance.

> **New to recommender systems?** All benchmarks come from the recommendation domain (the KuaiRand family). If terms such as CTR, multi-task learning, NDCG, or Recall@K are unfamiliar, start with **Appendix A: A Primer on Recommender Systems**.

---

## 2.2 Problem Statement

### The Task

Design and implement an **Autonomous ML Research Agent**. For each benchmark, the agent must autonomously:

1. **Reproduce the official baseline.** Stand up a working end-to-end pipeline and confirm it reaches the official baseline's reported validation score. (The official baseline is a fixed, organizer-provided reference — see Benchmarks. Any starter pipeline the agent builds for itself is an internal step, not the reference it is scored against.)
2. **Iterate on the pipeline.** Autonomously draw on established methods from both industry and academia to improve each stage of the pipeline, and apply those improvements in code. The agent develops using only the training split and the public validation feedback — it never has access to the hidden test set.
3. **Improve over the baseline.** Through repeated iterations, drive the validation score above the official baseline. Improvement need not be strictly monotonic, but the agent should show a clear, sustained ability to keep improving. Final ranking is computed once, on the hidden test set, using the submission the agent designates as final.

### Task Requirements

1. **Runs end-to-end and aims to beat the baseline.** The agent must run the full pipeline on the required benchmark (**KuaiRand-Pure**) and reach a converged result; the bonus benchmark (KuaiRand-1k & KuaiRand-27k) is optional. The target is a hidden-test score that exceeds the official baseline; the actual delta achieved — positive or negative — feeds into the Primary metric scoring, so falling short is scored continuously rather than treated as a disqualifying failure.
2. **Iterates autonomously across the full stack.** Improvements may target any part of the algorithmic stack — not just model architecture, but every upstream and downstream module. The goal is to minimize human intervention; a fully autonomous run is ideal, but a well-instrumented semi-automated pipeline needing only a handful of interventions is acceptable. We measure how little human intervention a run requires (e.g. number of manual interventions).
3. **Robust operation.** Robustness is about how the agent handles difficulty, not how often it succeeds. When a step fails (a code error, a timeout, an unexpected input), the agent can recover, retry, or route around it, and long iterative runs neither crash, stall, nor diverge.

---

## 2.3 Constraints & Scope

| Category | Details |
|---|---|
| **In scope** | Any open-source library or framework (PyTorch, RecBole, TorchRec, LightGBM, …); any papers, public solutions, or pretrained weights; changes to any pipeline stage — not just the model. |
| **Out of scope** | No external training data or pretrained weights trained on these benchmarks' test labels; no hidden-test access during development (train + validation only). |
| **Limits** | KuaiRand-Pure: NDCG@10 / Recall@50, click = positive (fixed) — **Required**. KuaiRand-1k & KuaiRand-27k: same task and metrics — **Bonus**. Hidden test scored once, on the final submission. Compute budget: TBD. |
| **Allowed assumptions** | Fixed train / validation / hidden-test split per dataset; official baseline, scores & evaluation script (incl. convergence rule); example submission + output schema. |

---

## 2.4 Available Resources & Data

### Starter Kit

To lower the barrier to entry, the challenge provides a standard starting point. Download: `kuairand-starter-kit.zip` — **numpy only** (no torch / pandas / scikit-learn); `python3 baseline.py --model fm` reproduces the official baseline in about **40 s on a single CPU core**. It contains:

1. **Fixed data splits** (date-based, from `log_standard_4_08_to_4_21_pure.csv` & `log_standard_4_22_to_5_08_pure.csv`):
   - train = 20220408–20220421 (**1,141,112 rows**)
   - validation = 20220422–20220428 (**124,909 rows**)
   - test = 20220429–20220508 (**170,588 rows**)
   - Teams develop on train + validation only; the hidden test set is scored once. Splitting by date avoids tie-breaking ambiguity on equal timestamps.
2. **Official baseline**: a fixed reference pipeline — a **Factorization Machine (k=16, lr=0.001, 5 categorical fields)**, numpy only, ~40 s on CPU.
   - Published **hidden-test** scores: GAUC 0.6610 / nDCG@5 0.5282 / **primary 0.5946** (mean over 5 seeds, std 0.0008).
   - **Validation**: GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016.
   - Reference rungs for harness self-check — random scoring: primary 0.4753; item popularity: primary 0.5715.
3. **Evaluation script** (`evaluate.py`): the exact scoring code (GAUC / nDCG@5). Model-agnostic — takes only `(user_ids, labels, scores)`.
   - Pinned conventions: users with zero positives count as nDCG = 0 and are included in the average; GAUC counts only users with `0 < positives < impressions`, weighted by positive count; nDCG gain = `2^rel − 1`.
   - **Convergence rule**: ε = 0.002, N = 3 — converged when the validation primary score has not improved by more than ε over the last N consecutive iterations (ε ≈ 2.5σ of the baseline's 5-seed std of 0.0008).
4. **Submission format**: a CSV with header `row_id,user_id,video_id,score`, one line per evaluation-split row.
   - `row_id` is a 0-based, strictly increasing index into the split as produced by `data.load()`; `user_id` / `video_id` are redundant fields used only to verify alignment; `score` is any real number (only relative order matters), NaN / Inf rejected.
   - `row_id` is required because `(user_id, video_id)` is not unique in the evaluation split — 3.06% of test rows are repeated pairs, up to 12 times.
   - Generate a runnable example with `python3 submit.py --make` and validate with `--check`.
5. **Run-log requirements**: each iteration should record its **hypothesis**, the **code diff**, the resulting **metrics**, and any **error / recovery** events. These logs are how judges assess Autonomy and Robustness.
6. **LLM coding agent**: use whatever you like, or Trae from ByteDance (7-day free trial for new users).

### Benchmarks

**KuaiRand-Pure is required and determines 100% of the primary score.** KuaiRand-1k and KuaiRand-27k are bonus datasets (extra credit, optional).

**Resource policy.** External resources are open by default: any open-source library, any papers/docs/public solutions, and pretrained weights freely. **One hard rule: no external training data.** Training must rely only on the KuaiRand datasets — no augmenting, joining, or pre-training on any other dataset, and no pretrained model whose weights were trained on these benchmarks' test labels.

| Dataset | Domain & Description | Metrics | Scale |
|---|---|---|---|
| **KuaiRand** (Kuaishou) | Short-video feed. 12 feedback signals (click / like / follow / comment / forward / long_view / play_time …) plus a randomized-exposure intervention supporting counterfactual evaluation. Default task: **click** = positive relevance label; reports **NDCG@10 / Recall@50**. | NDCG@10 / Recall@50 | Pure: 1.4M interactions (27K users × 7.6K items). 1k: 11.7M. 27k: 322M. |

Link: [KuaiRand — https://kuairand.com](https://kuairand.com). KuaiRand's randomized-exposure data also enables off-policy / counterfactual evaluation (OPE).

---

## 2.5 Expected Deliverables

1. **Written Project Description (via Devpost)** — how the solution addresses the problem statement; development tools; APIs; libraries and frameworks; datasets and assets used.
2. **Public Code/GitHub Repository** — well-structured, commented code covering all components; a README with project overview, setup/installation, steps to reproduce, a reflection on limitations & future work, and team member contributions.
3. **Run & Iteration Logs** — per-iteration log covering the hypothesis, the code diff, the resulting metrics (NDCG@10 / Recall@50), and any error/recovery events; plus a short summary of the number of manual interventions during the run.
4. **Final Submission & Results Summary** — final model output/checkpoint for KuaiRand-Pure (Starter Kit schema); a results table with validation-best NDCG@10 / Recall@50 and absolute delta over the official baseline; reported resource usage (total input+output tokens, total GPU-hours).

---

## 2.6 Judging Criteria

| Criterion | Weight |
|---|---|
| Technical Execution | 35% |
| Innovation & Problem Insight | 20% |
| Impact & Relevance | 20% |
| Feasibility & Practicality | 15% |
| Presentation & Communication (Final Event Only) | 10% |

**Technical Execution — Primary Metric & Robustness.** Score the *converged* result, not the peak and not the intermediate trajectory. Converged = validation score has not improved by more than ε over the last N iterations (ε, N from Starter Kit), or the run hits the compute/wall-clock budget — whichever comes first. The validation-best checkpoint at that point is evaluated once on the hidden test set.

- KuaiRand-Pure determines 100% of the Primary metric; 1k/27k earn bonus points.
- Per-dataset metrics → NDCG@10 / Recall@50. Within each dataset, the score is the equal-weighted average of each metric's **absolute improvement** over the official baseline on the hidden test set:

  ```
  delta(m)      = score_agent(m) − score_baseline(m)
  score_dataset = mean over m of delta(m)
  ```

- **Robustness** — judged by how the agent handles a failure (recovering, retrying, routing around), not by whether it ever fails.

**Innovation & Problem Insight** — what the agent chose to target across the full stack and the reasoning behind it; originality in drawing on published methods, papers, or public solutions.

**Impact & Relevance — Autonomy** — how much of the improvement loop the agent drives on its own; measured primarily by the number of manual interventions (fewer = higher; fully autonomous = highest).

**Feasibility & Practicality — Resource Consumption** — total input+output tokens across the run, and total GPU-hours to reach the converged result.

---

## 2.7 References

1. J. S. Chan et al., "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering," OpenAI, 2024. arXiv:2410.07095.
2. Z. Jiang et al., "AIDE: AI-Driven Exploration in the Space of Code," 2025. arXiv:2502.13138.
3. Y. Yamada et al., "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search," 2025. arXiv:2504.08066.

---

## 2.8 Appendix A — A Primer on Recommender Systems

### A.1 The Big Picture: The Recommendation Pipeline

A modern industrial recommender runs a funnel, each stage narrowing the candidate set:

```
Recall   →   Pre-ranking   →   Ranking   →   Re-ranking
millions      thousands        hundreds       final list
```

- **Recall / Retrieval**: cheaply retrieve a few thousand candidates from millions.
- **Pre-ranking**: a lightweight model trims further.
- **Ranking**: a heavy, accurate model scores each candidate. **This challenge mostly lives here.**
- **Reranking**: adjust final ordering for diversity, business rules, etc.

The KuaiRand benchmarks are ranking/prediction tasks, not full end-to-end pipelines.

### A.2 Core Tasks: CTR and the Feedback Funnel

- **CTR (Click-Through Rate)** — P(click | impression).
- **CVR (Conversion Rate)** — P(conversion | click). E-commerce only; not a task here.
- The funnel: impression → click → deeper engagement. Two known problems: **sample selection bias** (post-click signal only observed on clicked items, yet must be predicted for all impressions) and **data sparsity** (post-click signals like long_view/like are far rarer than clicks).

KuaiRand has no purchase label, so CVR is never scored. But the same two problems reappear on its post-click signals; ESMM-style multi-task modelling (A.3) is a legitimate approach.

### A.3 Multi-Task & Multi-Feedback Learning

Predicting the many user signals jointly (rather than a separate model per signal) shares representations and tends to improve every task. KuaiRand provides 12 feedback signals, so a multi-task model can learn from several jointly even though only **click** is scored. The key idea: balance **shared parameters** (transfer knowledge across tasks) against **task-specific parameters** (prevent conflicting tasks from hurting one another — the "seesaw" problem).

### A.4 Evaluation Metrics

| Metric | Intuition | Used for |
|---|---|---|
| **AUC** | Probability a random positive is ranked above a random negative. Threshold-free, robust to class imbalance. | CTR/CVR in general (not scored here) |
| **NDCG** | Quality of a ranked list, rewarding relevant items near the top (with position discount). | Ranking quality (KuaiRand) |
| **Recall** | Fraction of all relevant items that appear in the returned list. | Coverage (KuaiRand) |

**Offline vs. online**: a higher offline metric does not always mean better real-world performance (distribution shift, feedback loops). This competition is evaluated offline, but the gap is worth knowing.

### A.5 Feature Engineering Basics

- **ID features**: user ID, item ID, category ID — high-cardinality discrete features.
- **Embedding**: map each discrete ID to a learnable dense vector. Foundation of all deep recommenders.
- **Feature crossing**: combine features (e.g. user × category) to capture interactions. FM and DeepFM automate this.

### A.6 Annotated Reading List

Read just one of the following for the overview:

- Google, *Recommendation Systems (Machine Learning Crash Course)*, Overview section — https://developers.google.com/machine-learning/recommendation (Google calls the ranking stage "scoring" — same thing).
- Wang Shusen, *Recommender Systems*, Chapter 1 (Overview) — https://github.com/wangshusen/RecommenderSystem (most beginner-friendly Chinese resource).
