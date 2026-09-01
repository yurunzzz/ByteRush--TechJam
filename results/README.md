# Reproducible Results

This directory contains compact evidence from the completed Stage 1-4 run `2026-08-31_18-02-46_kuairand_fm_validation_baseline_attempt_0` and the frozen Wide & Deep model selected using validation data only. Interrupted runs, failed candidate workspaces, caches, and redundant checkpoints are intentionally excluded.

## Final result

| Metric | FM baseline | Wide & Deep | Absolute gain |
|---|---:|---:|---:|
| GAUC | 0.667366 | 0.671028 | +0.003662 |
| nDCG@5 | 0.535944 | 0.537670 | +0.001726 |
| Primary | 0.601655 | 0.604349 | +0.002694 |

The final values are means from five successful validation seeds. No test label or test metric entered model selection. `final_model/` was reconstructed from the preserved successful Stage 4 outputs because the shared artifact directory was subsequently reused by later research runs. The source model, matching best-seed checkpoint, training history, hashes, and source node identifiers are retained for auditability.

## Layout

- `runs/2026-08-31_18-02-46_kuairand_fm_validation_baseline_attempt_0/`: configuration and compact search/resource evidence for the complete run.
- `final_model/`: submission-ready Wide & Deep source, checkpoint, manifest, validation seed metrics, and predictions.

## Validate the submission

```bash
python kuairand-starter-kit/submit.py results/final_model/submission.csv --check --split test --data_dir kuairand-starter-kit/KuaiRand-Pure/data
```
