# ByteRush — Autonomous ML Research Agent for Recommender Systems

*TikTok TechJam · Problem #2: Autonomous Machine Learning Research Agent for Recommender Systems*
*Benchmark: KuaiRand-Pure · Written Project Description (Deliverable 2.5.1)*

---

## Inspiration & Summary

Machine-learning engineers spend most of their time on one repetitive loop: read a dataset and its metrics, engineer features, train and tune a model, read the score, then reflect and try again. **ByteRush** turns that loop into an autonomous agent. It connects Sakana AI's **AI‑Scientist‑v2** agentic tree‑search engine to the organizer‑provided **KuaiRand‑Pure Factorization Machine (FM)** baseline, so a large language model can propose an experiment, write the code, run it in an isolated workspace, read back validation metrics, and decide what to try next — with no human in the loop and no access to the hidden test set.

---

## How the Solution Addresses the Problem Statement

The problem statement (2.2) asks for an agent that can autonomously **reproduce the official baseline**, **iterate on the pipeline** using only train + public‑validation feedback, and **improve over the baseline**, while running end‑to‑end and remaining robust across long runs. ByteRush addresses each requirement directly:

**1. Reproduces the official baseline.** The agent stands up a working FM pipeline through the controlled `run_fm_experiment.py` interface and confirms it reaches the published reference. In the selected **2026-08-30 22:11** run, our verified reproduction reached **GAUC 0.667366 / nDCG@5 0.535944 / primary 0.601655** (`primary = (GAUC + nDCG@5) / 2`), matching the official FM validation primary of approximately 0.6016 and establishing that the loop is correct and reproducible before search begins.

**2. Iterates autonomously across the stack.** ByteRush runs the full Figure‑1 loop on its own. Each iteration the LLM: (a) proposes a hypothesis and generates code, (b) executes it in a sandboxed interpreter, (c) parses back GAUC, nDCG@5 and the primary metric, (d) writes a node into a **best‑first tree search (BFTS)** journal, (e) reproduces promising nodes under an independent seed, and (f) auto‑selects the best node and checkpoints it. Over an end‑to‑end campaign the agent explored FM hyper‑parameter tuning, an embedding model, Wide&Deep, Embedding+MLP, DeepFM, BPR and DIN‑style attention — improvements spanning features, model architecture and loss functions, not just the model box.

**3. Improves over the baseline, and freezes a final submission for hidden‑test scoring.** Two numbers matter here, and the challenge separates them deliberately:

- **What we can measure (validation).** During development the only feedback available is the validation primary. Against the reproduced FM baseline (**GAUC 0.667366 / nDCG@5 0.535944 / primary 0.601655**), the frozen **Wide & Deep** winner reached **GAUC 0.671028 / nDCG@5 0.537670 / primary 0.604349**, an improvement of **+0.003662 GAUC / +0.001726 nDCG@5 / +0.002694 primary (+0.448%)**. The winner was selected using validation only and verified across 5 seeds. We report this against the FM bar, not against a `random` sanity check, and we flag in our run log that an earlier "+0.12" figure was misleading because it compared to random.
- **What determines the ranking (hidden test).** Per the problem statement, **the final ranking is computed once by the organizers on the hidden test set**, using the single submission the agent designates as final. The agent therefore selects and freezes its model *exclusively on validation*, produces predictions for all **170,588** hidden-test rows in a `submission.csv` with the official `row_id,user_id,video_id,score` schema, and validates it with `submit.py --check`. We never see a test score ourselves, so we do **not** self‑report one; the published reference bar on hidden test is the official FM's **primary 0.5946** (GAUC 0.6610 / nDCG@5 0.5282). The reproduced FM validation primary is **0.007055** above that hidden-test reference (0.601655 → 0.5946), so our **+0.002694** validation lead is the honest, measurable signal — the final hidden‑test delta is resolved at judging time.

**4. Never touches the hidden test set.** A hard boundary in the code prevents the agent from ever seeing test labels or test metrics, so the validation lead above is not contaminated. `evaluate.py` is the single scoring authority; the research loop selects nodes purely on **validation** primary. The test split is read only once — after the model is frozen — to emit the submission file, and its labels/metrics are never returned to the agent.

**5. Robust operation.** Long runs neither crash nor diverge: failed code nodes are caught, parsed and routed around; the FM interface rejects unknown or unsafe parameters; and `data.py` / `evaluate.py` are guarded by **SHA‑256 integrity checks** before and after every run so the agent cannot silently alter the data split, the labels, the row order, or the evaluator. A no‑LLM **smoke test** validates the interpreter, metric structure, primary computation and protected‑file hashes as a fast regression gate.

**Instrumentation & transparency.** Every run records its hypothesis, code diff, resulting metrics and any error/recovery event into an auto‑generated **run log** (`reports/RUN_LOG.md`), and a live **Streamlit dashboard** visualizes the search tree, per‑stage stats and the best node. For the selected **2026-08-30 22:11** run, the recorded usage is **1,470,813 LLM tokens across 257 calls, 4,915.8 seconds (81.9 min) wall‑clock, and 1,017.7 seconds (17.0 min) GPU‑active**. The run executed 46 iterations, performed 4 seed evaluations, and reached 5,696 MiB peak GPU memory.

---

## Development Tools

- **AI‑Scientist‑v2 AgentManager / BFTS engine** (Sakana AI) — the tree‑search orchestrator that drives plan → code → execute → analyze → select. Launched via `launch_scientist_bfts.py` with a project‑specific `bfts_config_kuairand.yaml`.
- **Python 3.11 / 3.12** on Linux, with CUDA **PyTorch 2.5.1+cu124** provided by the **SeeTacloud / AutoDL** cloud GPU image (FM also runs CPU‑only).
- **tmux** for durable, detachable long‑running agent sessions on the remote server.
- **Streamlit dashboard** (`dashboard/app.py`) for live monitoring of the search tree, stage statistics and the current best node.
- **Git** for version control, with a strict pre‑upload checklist that keeps API keys, `.env`, raw KuaiRand data, experiment directories, checkpoints and logs out of the repository.
- **Model routing via environment variables** — the current agent configuration routes six specialized AI‑Scientist roles across three models: `gpt-5.6-sol` for code generation, `gpt-5.6-terra` for feedback, summarization and node selection, and `gpt-5.6-luna` for visual analysis and report generation. The routing is controlled entirely through the six `AI_SCIENTIST_*_MODEL` variables, with no Python/YAML edits.
- **`black`, `py_compile`, `pip check`** as lightweight code‑hygiene and dependency‑consistency gates.

---

## APIs

The current API configuration uses role-specialized model routing through the AI‑Scientist model-client interface. All six roles are bound through environment variables, so the complete model stack can be changed without modifying the research pipeline.

| Environment variable | Model | Agent role |
|---|---|---|
| `AI_SCIENTIST_CODE_MODEL` | `gpt-5.6-sol` | Code generation and implementation |
| `AI_SCIENTIST_FEEDBACK_MODEL` | `gpt-5.6-terra` | Experiment feedback and analysis |
| `AI_SCIENTIST_SUMMARY_MODEL` | `gpt-5.6-terra` | Research-state summarization |
| `AI_SCIENTIST_SELECT_MODEL` | `gpt-5.6-terra` | Best-node selection |
| `AI_SCIENTIST_VLM_MODEL` | `gpt-5.6-luna` | Plot and figure interpretation |
| `AI_SCIENTIST_REPORT_MODEL` | `gpt-5.6-luna` | Final report generation |

The configuration used to launch the agent is:

```bash
cd /root/autodl-tmp/ByteRush

export AI_SCIENTIST_CODE_MODEL=gpt-5.6-sol
export AI_SCIENTIST_FEEDBACK_MODEL=gpt-5.6-terra
export AI_SCIENTIST_SUMMARY_MODEL=gpt-5.6-terra
export AI_SCIENTIST_SELECT_MODEL=gpt-5.6-terra
export AI_SCIENTIST_VLM_MODEL=gpt-5.6-luna
export AI_SCIENTIST_REPORT_MODEL=gpt-5.6-luna
```

The final **Wide & Deep** winner was selected exclusively on validation and verified across **5 seeds**. It reached **GAUC 0.671028 / nDCG@5 0.537670 / primary 0.604349**, improving over the reproduced FM baseline (**GAUC 0.667366 / nDCG@5 0.535944 / primary 0.601655**) by **+0.003662 GAUC / +0.001726 nDCG@5 / +0.002694 primary (+0.448%)**. The frozen model is submission-ready and produces predictions for all **170,588** hidden-test rows.

Structured tool/function calls return machine-readable `AI_SCIENTIST_RESULT` records containing validation GAUC, nDCG@5 and primary metrics. Additional provider routes inherited from upstream AI‑Scientist‑v2 remain available in the code but are not used by the current configuration.

*No credentials are stored in the repo; keys are loaded only into the shell/tmux session that launches the agent.*

---

## Libraries & Frameworks

**Agent & experiment runtime**
- `numpy` (the FM baseline and evaluator are **numpy‑only**, per the starter kit)
- `PyTorch` (CUDA) for the neural iterations the agent explores (Wide&Deep, DeepFM, DIN attention, etc.)
- `omegaconf` (agent/experiment configuration), `jsonschema` + `genson` (schema validation of tool outputs), `dataclasses-json`, `funcy`, `coolname`, `humanize`, `shutup`
- `tiktoken` (token accounting / budget tracking), `backoff` (retry with exponential back‑off), `python-igraph` (search‑tree graph handling), `rich` + `tqdm` (console/log UX), `black` (auto‑formatting of generated code)

**LLM client**
- The AI‑Scientist model-client abstraction routes requests according to the six `AI_SCIENTIST_*_MODEL` environment variables: `gpt-5.6-sol` handles code generation; `gpt-5.6-terra` handles feedback, summaries and selection; and `gpt-5.6-luna` handles visual analysis and reporting. Alternate provider clients remain as inherited upstream dependencies but are not used by the current configuration.

**Visualization & reporting**
- `matplotlib`, `seaborn` for plots; `streamlit`, `plotly`, `streamlit-autorefresh` for the live dashboard; `pypdf` / `pymupdf4llm` for document handling (inherited from upstream)

**Base framework**
- **AI‑Scientist‑v2** (Sakana AI) — AgentManager, Interpreter, parallel agent and tree‑search modules under `ai_scientist/treesearch/`.

Dependencies are pinned in `requirements_kuairand.txt` (a trimmed, verified subset) with the broader upstream set in `requirements.txt`.

---

## Datasets & Assets Used

- **KuaiRand‑Pure** (Kuaishou short‑video feedback dataset) — the required benchmark. Source: [kuairand.com](https://kuairand.com), downloaded from Zenodo (`KuaiRand-Pure.tar.gz`). Fixed, date‑based chronological splits from the organizer starter kit:
  - **train** 2022‑04‑08 → 04‑21 — 1,141,112 rows
  - **validation** 2022‑04‑22 → 04‑28 — 124,909 rows
  - **test (hidden)** 2022‑04‑29 → 05‑08 — 170,588 rows
  - Task: in‑user re‑ranking of exposed videos; relevance label = `long_view` (0/1). Raw data is never committed to Git.
- **Official KuaiRand starter kit** (organizer‑provided assets):
  - `data.py` — official loader/splitter (protected, hash‑verified)
  - `evaluate.py` — official GAUC / nDCG@5 scoring code, the single source of truth (protected)
  - `baseline.py` — random / item‑popularity / FM reference baselines
  - `run_fm_experiment.py` + `fm_experiment_config.json` — the validation‑only controlled FM experiment interface exposed to the agent
  - `submit.py` — submission‑CSV generation and format validation (`row_id,user_id,video_id,score`)
  - `AGENT_FM_INTERFACE.md`, `kuairand_ranking.md` — the agent's interface contract and research‑task prompt
- **Reference scores.** Development bar (validation, what the agent optimizes against): the selected run reproduced FM at **GAUC 0.667366 / nDCG@5 0.535944 / primary 0.601655**; item‑popularity 0.5807; random 0.4834 (sanity check only); oracle ceiling 0.8484 (theoretical max). The final 5-seed-verified Wide & Deep winner reached **GAUC 0.671028 / nDCG@5 0.537670 / primary 0.604349**, improving by **+0.003662 / +0.001726 / +0.002694 (+0.448%)**, respectively. Judging bar (hidden test, scored once by organizers): FM official **primary 0.5946** (GAUC 0.6610 / nDCG@5 0.5282, mean over 5 seeds, std 0.0008), **0.007055** below the reproduced validation primary — this is the number the final submission is ultimately ranked against.
- **No external training data or test‑trained pretrained weights** were used — training relies solely on the KuaiRand splits, per the challenge's one hard resource rule.

---

## Built On / Credits

ByteRush is built on **AI‑Scientist‑v2** by Sakana AI ([repo](https://github.com/SakanaAI/AI-Scientist-v2), [paper](https://arxiv.org/abs/2504.08066)) and retains its license and responsible‑use requirements; because the system executes LLM‑generated code, it is run only in isolated, controlled environments. Benchmark data courtesy of the **KuaiRand** team ([kuairand.com](https://kuairand.com)).
