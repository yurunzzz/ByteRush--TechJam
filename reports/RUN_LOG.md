# KuaiRand Agent Run Log

*ByteRush · TechJam — KuaiRand-Pure long_view ranking · primary = mean(GAUC, nDCG@5)*  
*Auto-generated 2026-08-30 16:27 · 33 runs shown / 37 launched (4 sub-minute aborts hidden)*

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
| Total LLM tokens | 6.51M across ~1,600 calls |
| Agent wall-clock | 5.9 h |
| GPU-active training | 96 min |

> **Reality check:** measured against the real FM baseline (`0.6016`), the agent's best is only **+0.0037** — essentially matching the provided FM. Early runs reproduced FM at ~0.6014; later architecture changes (DeepFM, BPR, DIN attention) added <0.004. The '+0.12' figure from an earlier draft compared against `random` and was misleading.

## Per-run ledger

Chronological. `Primary` = best validation node in that run. `vs prev` compares to the previous scored run. Shipped submissions use a more conservative 3-seed mean.

| Start | Architecture / change | Dur | Train | Tokens | Calls | Iters | Seeds | Primary | vs base | vs prev |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-28 01:49 | — | — | — | — | — | — | — | `0.6015` | −0.0001 | — |
| 2026-08-28 01:55 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 01:58 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 02:06 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 03:00 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 21:58 | candidate model | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 22:13 | candidate model | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 22:28 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 22:33 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-28 22:57 | candidate model | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-29 01:42 | DeepFM + BPR pairwise loss | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-29 02:35 | candidate model | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-29 02:50 | candidate model | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-29 03:20 | candidate model | — | — | — | — | — | — | `0.6015` | −0.0001 | ±0 |
| 2026-08-29 15:19 | candidate model | — | — | — | — | — | — | `0.6018` | +0.0002 | +0.0004 |
| 2026-08-29 16:09 | DeepFM | — | — | — | — | — | — | `0.6053` | +0.0037 | +0.0035 |
| 2026-08-29 19:10 | — · aborted | 0s | 0s | 0 | — | 0 | 0 | `—` | — | — |
| 2026-08-29 19:18 | Wide&Deep | 43m | 3m | 652k | 234 | 15 | 12 | `0.6039` | +0.0023 | −0.0014 |
| 2026-08-29 20:58 | — · aborted | 39s | 13s | 0 | — | 1 | 0 | `—` | — | — |
| 2026-08-29 21:12 | Embedding + MLP interaction | 42m | 3m | 616k | 217 | 14 | 9 | `0.6048` | +0.0032 | +0.0009 |
| 2026-08-30 00:35 | DeepFM + Wide&Deep | 36m | 11m | 1.10M | 289 | 23 | 30 | `0.6042` | +0.0026 | −0.0006 |
| 2026-08-30 03:02 | candidate model | 5m | 44s | 66k | 15 | 3 | 0 | `0.6017` | +0.0001 | −0.0025 |
| 2026-08-30 03:13 | candidate model | 4m | 28s | 48k | 13 | 2 | 0 | `0.6017` | +0.0001 | ±0 |
| 2026-08-30 03:22 | candidate model | 4m | 27s | 48k | 15 | 2 | 0 | `0.6017` | +0.0001 | ±0 |
| 2026-08-30 03:34 | candidate model | 57m | 9m | 1.12M | 281 | 28 | 9 | `0.6040` | +0.0024 | +0.0024 |
| 2026-08-30 05:03 | candidate model | 19m | 2m | 478k | 59 | 16 | 0 | `0.6035` | +0.0019 | −0.0005 |
| 2026-08-30 05:33 | DeepFM + BPR pairwise loss | 28m | 9m | 746k | 123 | 22 | 0 | `0.6049` | +0.0033 | +0.0013 |
| 2026-08-30 06:02 | candidate model | — | — | — | — | — | — | `0.6040` | +0.0024 | −0.0008 |
| 2026-08-30 07:26 | DIN target-attention + history | 86m | 51m | 931k | 178 | 25 | 21 | `0.6044` | +0.0028 | +0.0003 |
| 2026-08-30 15:14 | — · aborted | 0s | 0s | 0 | — | 0 | 0 | `—` | — | — |
| 2026-08-30 15:23 | — · aborted | 43s | 15s | 0 | — | 1 | 0 | `—` | — | — |
| 2026-08-30 15:30 | DeepFM | 27m | 8m | 702k | 176 | 16 | 18 | `0.6040` | +0.0024 | −0.0004 |
| 2026-08-30 16:17 | candidate model | — | — | — | — | — | — | `0.6040` | +0.0024 | −0.0001 |

## Model usage

| Model | Role | Tokens | Calls |
|---|---|---|---|
| `deepseek-v4-pro` | plan + code | 3.32M | 350 |
| `deepseek-v4-flash` | feedback + select + summary | 1.63M | 1,030 |
| `deepseek-v4-flash-vision-exp` | plot / figure reading | 1.56M | 220 |

## Workflow (per run)

AI-Scientist-v2 tree search, `bfts_config_kuairand.yaml` (max_stage 4, 1 worker):

1. **Stage 1 — Initial implementation:** run trusted starting code, confirm a working candidate on GPU.
2. **Stage 2 — Baseline tuning:** learning-rate / regularization sweeps.
3. **Stage 3 — Creative research:** architecture changes (MLP, DeepFM, Wide&Deep, ranking losses, attention).
4. **Stage 4 — Ablation + freeze:** leave-one-component-out ablation, 3-seed re-validation, freeze checkpoint → `submission.csv`.

---
*Sources: per-run `resource_summary.json` (timing/tokens/GPU), `history.json` (metrics), `best_solution_*.py` (architecture). Metrics are validation-only; test labels are held out. Regenerate: `python reports/build_run_log.py`.*
