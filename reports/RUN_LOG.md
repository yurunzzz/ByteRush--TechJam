# KuaiRand Agent Run Log

*ByteRush · TechJam — KuaiRand-Pure long_view ranking · primary = mean(GAUC, nDCG@5)*  
*Auto-generated 2026-08-30 16:11 · 12 runs shown / 36 launched*

## Summary

| Metric | Value |
|---|---|
| FM baseline (starter kit) | `0.4834` |
| **Best validation primary** | **`0.6049`** (+0.1215 vs baseline) |
| Scored runs | 12 of 36 launched |
| Total LLM tokens | 6.51M across ~1,600 calls |
| Agent wall-clock | 5.8 h |
| GPU-active training | 96 min |

## Per-run ledger

Chronological. `Primary` = best validation node in that run. `vs prev` compares to the previous scored run. Shipped submissions use a more conservative 3-seed mean.

| Start | Architecture / change | Dur | Train | Tokens | Calls | Iters | Seeds | Primary | vs base | vs prev |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-29 19:18 | Wide&Deep | 43m | 3m | 652k | 234 | 15 | 12 | `0.6039` | +0.1205 | — |
| 2026-08-29 21:12 | Embedding + MLP interaction | 42m | 3m | 616k | 217 | 14 | 9 | `0.6048` | +0.1214 | +0.0009 |
| 2026-08-30 00:35 | DeepFM + Wide&Deep | 36m | 11m | 1.10M | 289 | 23 | 30 | `0.6042` | +0.1208 | −0.0006 |
| 2026-08-30 03:02 | candidate model | 5m | 44s | 66k | 15 | 3 | 0 | `0.6017` | +0.1183 | −0.0025 |
| 2026-08-30 03:13 | candidate model | 4m | 28s | 48k | 13 | 2 | 0 | `0.6017` | +0.1183 | ±0 |
| 2026-08-30 03:22 | candidate model | 4m | 27s | 48k | 15 | 2 | 0 | `0.6017` | +0.1183 | ±0 |
| 2026-08-30 03:34 | candidate model | 57m | 9m | 1.12M | 281 | 28 | 9 | `0.6040` | +0.1206 | +0.0024 |
| 2026-08-30 05:03 | candidate model | 19m | 2m | 478k | 59 | 16 | 0 | `0.6035` | +0.1201 | −0.0005 |
| 2026-08-30 05:33 | DeepFM + BPR pairwise loss | 28m | 9m | 746k | 123 | 22 | 0 | `0.6049` | +0.1215 | +0.0013 |
| 2026-08-30 06:02 | candidate model | — | — | — | — | — | — | `0.6040` | +0.1206 | −0.0008 |
| 2026-08-30 07:26 | DIN target-attention + history | 86m | 51m | 931k | 178 | 25 | 21 | `0.6044` | +0.1210 | +0.0003 |
| 2026-08-30 15:30 | DeepFM | 27m | 8m | 702k | 176 | 16 | 18 | `0.6040` | +0.1206 | −0.0004 |

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
