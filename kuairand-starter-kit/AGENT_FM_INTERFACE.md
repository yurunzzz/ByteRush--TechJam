# AI Scientist-v2 ↔ FM validation-only interface

## Purpose

`run_fm_experiment.py` is the trusted experiment harness. It lets an agent run one configurable Factorization Machine experiment while exposing only validation metrics. It does not encode, score, print, or save test metrics.

The official `data.py` split definition and `evaluate.py` metric implementation remain authoritative. The harness records and verifies SHA-256 hashes for both files before and after an experiment.

## Run one experiment

From the Starter Kit directory:

```bash
/opt/anaconda3/bin/python3 run_fm_experiment.py \
  --config fm_experiment_config.json \
  --data-dir KuaiRand-Pure/data \
  --output-dir experiments/fm_validation_baseline
```

Use the Python executable that has NumPy installed. The system Python on some machines may not include NumPy.

## Agent-readable result

The last stdout line always begins with:

```text
AI_SCIENTIST_RESULT 
```

On success, the remainder is one JSON object containing `GAUC`, `nDCG@5`, `primary`, checkpoint location, runtime, row counts, seed, and protected-file hashes. The only reported split is `valid` and `metric_direction` is `maximize`.

On failure, the same prefix is followed by a JSON object with `status: error`, `error_type`, and `message`. The process exits nonzero.

The experiment directory contains:

```text
config.json       exact validated configuration
history.json      validation metrics and train loss for every epoch
metrics.json      best validation result for node selection
best_model.npz    best-primary FM parameters
```

AI Scientist-v2 should use `metrics.json.primary` as its node-selection scalar and retain `GAUC` and `nDCG@5` for diagnosis.

## Allowed configuration

Only these keys are accepted:

| Key | Meaning |
|---|---|
| `experiment_name` | Safe experiment identifier |
| `seed` | NumPy random seed |
| `embedding_dim` | FM embedding dimension |
| `learning_rate` | Adam learning rate |
| `l2` | FM L2 regularization |
| `batch_size` | Training batch size |
| `max_epochs` | Maximum epochs |
| `early_stopping_patience` | Validation early-stopping patience |
| `min_delta` | Minimum primary change counted as an epoch improvement |

Unknown keys and unsafe values are rejected. The config cannot replace the label, split, evaluator, data loader, metric, or execute arbitrary code.

## Required AI Scientist-v2 integration

Configure the experiment runner to invoke this command and parse only the final `AI_SCIENTIST_RESULT` line or `metrics.json`. During Stage 2 the agent should alter only the allowed hyperparameters. Stage 3 model-code research will require a separate controlled model-plugin interface; it must preserve the same validation-only result contract.

Do not expose `baseline.py` test output to the research loop. Test prediction generation must be a separate final-inference command, run only after the validation-best checkpoint is frozen. `submit.py --check --split test` may then be used for schema and row-alignment validation; test scores must not be returned to the agent.
