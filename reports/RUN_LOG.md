# KuaiRand Agent Run Log

*ByteRush · TechJam — KuaiRand-Pure long_view ranking · primary = mean(GAUC, nDCG@5)*  
*Auto-generated 2026-08-30 16:56 · 16 rows / 29 scored of 37 launched (8 unscored/aborted hidden, 13 FM-level duplicates collapsed)*

## Reference scores (validation)

The bar to beat is **`fm_official`** — `random` is only a sanity check. `oracle_ceiling` is the theoretical max (nDCG capped by all-negative users) and the denominator for headroom.

| Reference | Primary | Note |
|---|---|---|
| random | `0.4834` | sanity check only — **not** the baseline |
| item_popularity | `0.5807` | official non-trained baseline |
| **fm_official** | **`0.6016`** | **official FM baseline — the real bar** |
| oracle_ceiling | `0.8484` | theoretical upper bound |

## Summary

| Metric | Value |
|---|---|
| FM baseline (fm_official) | `0.6016` |
| **Best validation primary** | **`0.6053`** (+0.0037 vs FM baseline · 1.5% of oracle headroom) (2026-08-29 16:09) |
| Scored runs | 29 of 37 launched |
| Total LLM tokens | 7.53M across ~1,876 calls |
| Agent wall-clock | 6.4 h |
| GPU-active training | 109 min |

> **Reality check:** measured against the real FM baseline (`0.6016`), the agent's best is only **+0.0037** — essentially matching the provided FM. Early runs reproduced FM at ~0.6014; later architecture changes (DeepFM, BPR, DIN attention) added <0.004. The '+0.12' figure from an earlier draft compared against `random` and was misleading.

## Per-run ledger

Chronological. `Primary` = best validation node in that run. `Stage` = which BFTS stage that node came from (`S3` exact; `~S3` approximate — peak was an intermediate node, so the deepest selected stage is shown). `vs prev` compares to the previous scored run. Shipped submissions use a 3-seed mean.

| Start | Architecture / change | Dur | Train | Tokens | Calls | Iters | Seeds | Primary | Stage | vs base | vs prev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-28 01:49 | FM baseline level — 14 early runs collapsed | — | — | — | — | — | — | `0.6015` | — | −0.0001 | — |
| 2026-08-29 15:19 | FM (tuned) | — | — | — | — | — | — | `0.6018` | ~S2 | +0.0002 | +0.0004 |
| 2026-08-29 16:09 | embedding model | — | — | — | — | — | — | `0.6053` | ~S3 | +0.0037 | +0.0035 |
| 2026-08-29 19:18 | Wide&Deep · BCE | 43m | 3m | 652k | 234 | 15 | 12 | `0.6039` | ~S4 | +0.0023 | −0.0014 |
| 2026-08-29 21:12 | Embedding + MLP · BCE | 42m | 3m | 616k | 217 | 14 | 9 | `0.6048` | ~S4 | +0.0032 | +0.0009 |
| 2026-08-30 00:35 | DeepFM + Wide&Deep · BCE | 36m | 11m | 1.10M | 289 | 23 | 30 | `0.6042` | ~S4 | +0.0026 | −0.0006 |
| 2026-08-30 03:02 | embedding model · BCE | 5m | 44s | 66k | 15 | 3 | 0 | `0.6017` | S2 | +0.0001 | −0.0025 |
| 2026-08-30 03:13 | embedding model · BCE | 4m | 28s | 48k | 13 | 2 | 0 | `0.6017` | S2 | +0.0001 | ±0 |
| 2026-08-30 03:22 | embedding model · BCE | 4m | 27s | 48k | 15 | 2 | 0 | `0.6017` | S2 | +0.0001 | ±0 |
| 2026-08-30 03:34 | embedding model · BCE | 57m | 9m | 1.12M | 281 | 28 | 9 | `0.6040` | ~S2 | +0.0024 | +0.0024 |
| 2026-08-30 05:03 | embedding model · BCE | 19m | 2m | 478k | 59 | 16 | 0 | `0.6035` | ~S3 | +0.0019 | −0.0005 |
| 2026-08-30 05:33 | embedding model · BPR pairwise | 28m | 9m | 746k | 123 | 22 | 0 | `0.6049` | S3 | +0.0033 | +0.0013 |
| 2026-08-30 06:02 | embedding model · BCE | — | — | — | — | — | — | `0.6040` | ~S3 | +0.0024 | −0.0008 |
| 2026-08-30 07:26 | DIN target-attention · BPR pairwise + history | 86m | 51m | 931k | 178 | 25 | 21 | `0.6044` | ~S4 | +0.0028 | +0.0003 |
| 2026-08-30 15:30 | embedding model · BCE | 27m | 8m | 702k | 176 | 16 | 18 | `0.6040` | ~S4 | +0.0024 | −0.0004 |
| 2026-08-30 16:17 | Embedding + MLP · BCE | 36m | 13m | 1.02M | 276 | 22 | 27 | `0.6047` | S3 | +0.0031 | +0.0006 |

## Model usage

| Model | Role | Tokens | Calls |
|---|---|---|---|
| `deepseek-v4-pro` | plan + code | 3.73M | 406 |
| `deepseek-v4-flash-vision-exp` | plot / figure reading | 1.92M | 269 |
| `deepseek-v4-flash` | feedback + select + summary | 1.87M | 1,201 |

## Workflow (per run)

AI-Scientist-v2 tree search, `bfts_config_kuairand.yaml` (max_stage 4, 1 worker):

1. **Stage 1 — Initial implementation:** run trusted starting code, confirm a working candidate on GPU.
2. **Stage 2 — Baseline tuning:** learning-rate / regularization sweeps.
3. **Stage 3 — Creative research:** architecture changes (MLP, DeepFM, Wide&Deep, ranking losses, attention).
4. **Stage 4 — Ablation + freeze:** leave-one-component-out ablation, 3-seed re-validation, freeze checkpoint → `submission.csv`.

---
*Sources: per-run `resource_summary.json` (timing/tokens/GPU), `history.json` (metrics), `best_solution_*.py` (architecture). Metrics are validation-only; test labels are held out. Regenerate: `python reports/build_run_log.py`.*
