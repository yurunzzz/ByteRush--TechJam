# KuaiRand Bonus Experiment Report

This report summarizes the converged KuaiRand-1K and KuaiRand-27K bonus runs. Model selection used validation metrics only; no test metric was computed or exposed to the agent.

## Final selected results

| Dataset | Selected model | Validation GAUC | Validation nDCG@5 | Validation primary |
| --- | --- | ---: | ---: | ---: |
| KuaiRand-1K | MLP (`2_multi_root_tuning_3_mlp`) | 0.517147958 | 0.490641564 | 0.503894761 |
| KuaiRand-27K | FM (`2_multi_root_tuning_1_fm`) | 0.521361083 | 0.430474535 | 0.475917809 |

Each selected result is the mean of two successful seeds. The metric named `primary` is `(GAUC + nDCG@5) / 2`.

The 1K Stage 3 search observed an adversarial-embedding MLP with primary 0.504521966 (GAUC 0.516865551, nDCG@5 0.492178380), but its gain of 0.000627 was below the configured 0.002 promotion threshold and it won only one of two seeds. The verified Stage 2 MLP therefore remains the selected converged incumbent. On 27K, the best Stage 3 candidate reached primary 0.472337641, below the selected Stage 2 FM.

## Resources to convergence

The accounting chain combines (1) Attempt 2 from launch through creation of the verified Stage 2 snapshot and (2) the complete Attempt 4 Stage 3 resume. Seed reruns are listed separately and are not counted toward the competition iteration limit.

| Dataset | Prompt tokens | Completion tokens | Total tokens | Agent wall time | Counted iterations | Seed evaluations | GPU | GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| KuaiRand-1K | 771,431 | 302,363 | 1,073,794 | 3,797.886 s (63m 17.886s) | 21 / 50 | 10 | NVIDIA RTX A4000 | 1.054969 |
| KuaiRand-27K | 519,867 | 230,264 | 750,131 | 2,707.849 s (45m 07.849s) | 15 / 50 | 6 | NVIDIA GeForce RTX 4080 SUPER | 0.752180 |
| Sum | 1,291,298 | 532,627 | 1,823,925 | 6,505.735 s | 36 | 16 | two single-GPU runs | 1.807149 |

GPU-hours use the conservative reproducible accounting convention `number of allocated GPUs × agent wall-clock hours`. Attempt 4's one-second sampler additionally observed 0.057097 active GPU-hours on 1K and 0.039879 active GPU-hours on 27K; the earlier Stage 2 process was paused before its sampler wrote a final active-time summary, so sampled active time is not used as the official total.

Both runs satisfy the competition limits of less than six hours and at most 50 counted iterations.

## Search outcome

- KuaiRand-1K: five valid Stage 3 candidates were evaluated. None passed the configured promotion rule. The strongest unpromoted candidate added bounded adversarial embedding perturbations during training.
- KuaiRand-27K: two valid Stage 3 candidates were evaluated. Neither improved on the FM incumbent; the strongest used a DeepFM residual interaction tower.
- Attempt 4 stopped during final artifact freezing because the resumed Stage 2 journal contained an aggregate incumbent without Stage 4 seed-child nodes. Search records and checkpoints were already intact. This finalization-only defect has since been fixed; it did not change any recorded metric.

## Submission status

These are bonus profiles, and both checked-in `dataset_profile.json` files specify `submission_export: false`. The profiles expose train/validation data to the agent but intentionally disable test prediction export. In particular, the 27K cache contains 114,832,239 test rows, so a CSV would also be several gigabytes and unsuitable for ordinary Git storage.

No test metric was computed, and no misleading pseudo-submission was generated. The frozen validation-best experiment artifacts are included in the matching experiment branch. A genuine organizer submission must be generated from a main KuaiRand-Pure validation-best checkpoint with `kuairand-starter-kit/submit.py --make-best`, followed by `--check` on the official test split.

