"""Editable KuaiRand candidate used by AI Scientist-v2.

The trusted input directory owns data loading and evaluation. Stage 2 may edit
CONFIG only. Stage 3 may replace CandidateModel or its training loss while
preserving the step/predict/state interface and validation-only protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

execution_dir = Path.cwd()
input_candidates = (execution_dir / "input", execution_dir.parent / "input")
input_dir = next((p for p in input_candidates if p.is_dir()), None)
if input_dir is None:
    raise FileNotFoundError(f"trusted input directory not found: {input_candidates}")
sys.path.insert(0, str(input_dir))

import baseline as baseline_module  # noqa: E402
import data as data_module  # noqa: E402
import evaluate as evaluate_module  # noqa: E402

# Stage 2 search space: keep CandidateModel unchanged and tune these values.
CONFIG = {
    "seed": int(os.environ.get("AI_SCIENTIST_SEED", "0")),
    "embedding_dim": 16,
    "learning_rate": 0.001,
    "l2": 1e-6,
    "batch_size": 8192,
    "max_epochs": 40,
    "patience": 4,
    "min_delta": 1e-5,
}

# Stage 4 interface. Stage 3 must register every newly introduced component as
# a literal True entry and guard its use with component_enabled(name). The
# baseline has no optional research component, so the manifest starts empty.
ABLATION_COMPONENTS = {}
ABLATION_TARGET = os.environ.get("AI_SCIENTIST_ABLATION_TARGET", "full")
if ABLATION_TARGET != "full" and ABLATION_TARGET not in ABLATION_COMPONENTS:
    raise ValueError(
        f"unknown ablation target {ABLATION_TARGET!r}; registered components: "
        f"{sorted(ABLATION_COMPONENTS)}"
    )


def component_enabled(name: str) -> bool:
    if name not in ABLATION_COMPONENTS:
        raise KeyError(f"unregistered ablation component: {name}")
    return bool(ABLATION_COMPONENTS[name]) and ABLATION_TARGET != name


# Stage 3 extension point. X is a dense integer ID array of shape (batch, 5),
# one categorical feature ID per field. It is NOT a one-hot/scipy matrix:
# use embedding lookup V[X], never X @ V. Replace this class with a compatible
# candidate while keeping __init__, step, predict, state_dict and
# load_state_dict. Vectorize over the whole batch: Python loops over samples are
# forbidden. New Stage 3 prototypes should use at most 12 epochs. Never load
# test data.
class CandidateModel(baseline_module.FM):
    def state_dict(self):
        return {"V": self.V.copy(), "W": self.W.copy(), "b": np.asarray(self.b).copy()}

    def load_state_dict(self, state):
        self.V = state["V"].copy()
        self.W = state["W"].copy()
        self.b = np.float32(state["b"])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


protected = {name: file_hash(input_dir / name) for name in ("data.py", "evaluate.py")}
started = time.monotonic()
loaded = data_module.load(str(input_dir / "KuaiRand-Pure" / "data"))
splits = {"train": loaded["train"], "valid": loaded["valid"]}
del loaded
encoded, feature_dimension = data_module.encode(splits)
train_x, train_y, _ = encoded["train"]
valid_x, valid_y, valid_users = encoded["valid"]

model = CandidateModel(
    feature_dimension,
    k=CONFIG["embedding_dim"],
    lr=CONFIG["learning_rate"],
    l2=CONFIG["l2"],
    seed=CONFIG["seed"],
)
rng = np.random.default_rng(CONFIG["seed"])
best_primary = -np.inf
best_epoch = 0
best_state = None
bad_epochs = 0
history = []

for epoch in range(1, CONFIG["max_epochs"] + 1):
    indices = rng.permutation(len(train_y))
    losses = []
    for start in range(0, len(indices), CONFIG["batch_size"]):
        batch = indices[start : start + CONFIG["batch_size"]]
        losses.append(model.step(train_x[batch], train_y[batch]))
    scores = np.asarray(model.predict(valid_x), dtype=np.float64)
    if scores.shape != (len(valid_y),) or not np.isfinite(scores).all():
        raise RuntimeError("candidate produced invalid validation predictions")
    metrics = evaluate_module.evaluate(valid_users, valid_y, scores)
    primary = float(metrics["primary"])
    history.append({
        "epoch": epoch,
        "train_loss": float(np.mean(losses)),
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": primary,
    })
    print(
        f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
        f"valid_GAUC={metrics['GAUC']:.6f} valid_nDCG@5={metrics['nDCG@5']:.6f} "
        f"valid_primary={primary:.6f}", flush=True,
    )
    if primary > best_primary + CONFIG["min_delta"]:
        best_primary, best_epoch = primary, epoch
        best_state = model.state_dict()
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= CONFIG["patience"]:
            break

if best_state is None:
    raise RuntimeError("candidate training produced no valid checkpoint")
model.load_state_dict(best_state)
best_scores = np.asarray(model.predict(valid_x), dtype=np.float64)
best_metrics = evaluate_module.evaluate(valid_users, valid_y, best_scores)
after = {name: file_hash(input_dir / name) for name in ("data.py", "evaluate.py")}
if after != protected:
    raise RuntimeError("protected data/evaluation infrastructure changed")

working_dir = execution_dir / "working"
working_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(working_dir / "candidate_checkpoint.npz", **best_state)
(working_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

gauc = float(best_metrics["GAUC"])
ndcg5 = float(best_metrics["nDCG@5"])
primary = (gauc + ndcg5) / 2.0
experiment_data = {
    "KuaiRand-Pure": {
        "metrics": {
            "validation primary": [primary],
            "validation GAUC": [gauc],
            "validation nDCG@5": [ndcg5],
        },
        "losses": {"train": [row["train_loss"] for row in history], "val": []},
        "predictions": [],
        "ground_truth": [],
        "metadata": {
            "metric_to_optimize": "validation primary",
            "maximize": True,
            "best_epoch": best_epoch,
            "seed": CONFIG["seed"],
            "runtime_seconds": time.monotonic() - started,
            "candidate_class": type(model).__name__,
            "ablation_target": ABLATION_TARGET,
            "registered_ablation_components": ABLATION_COMPONENTS,
            "active_ablation_components": {
                name: component_enabled(name) for name in ABLATION_COMPONENTS
            },
        },
    }
}
np.save(working_dir / "experiment_data.npy", experiment_data)
result_payload = {
    "split": "validation",
    "target": "long_view",
    "GAUC": gauc,
    "nDCG@5": ndcg5,
    "primary": primary,
    "best_epoch": best_epoch,
    "seed": CONFIG["seed"],
    "ablation_target": ABLATION_TARGET,
    "active_ablation_components": {
        name: component_enabled(name) for name in ABLATION_COMPONENTS
    },
}
print("AI_SCIENTIST_RESULT=" + json.dumps(result_payload, sort_keys=True))
print(f"validation GAUC: {gauc:.9f}")
print(f"validation nDCG@5: {ndcg5:.9f}")
print(f"validation primary: {primary:.9f}")
