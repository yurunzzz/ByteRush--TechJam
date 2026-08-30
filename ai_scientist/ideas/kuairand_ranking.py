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
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

execution_dir = Path.cwd()
input_candidates = (execution_dir / "input", execution_dir.parent / "input")
input_dir = next((p for p in input_candidates if p.is_dir()), None)
if input_dir is None:
    raise FileNotFoundError(f"trusted input directory not found: {input_candidates}")
sys.path.insert(0, str(input_dir))

import baseline as baseline_module  # noqa: E402
import data as data_module  # noqa: E402
import evaluate as evaluate_module  # noqa: E402
import research_data as research_data_module  # noqa: E402

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


def build_research_schema(splits, feature_state=None):
    """Expose schema v2 while keeping the official encoder authoritative."""
    return research_data_module.build_schema_v2(
        splits,
        data_module=data_module,
        feature_state=feature_state,
    )


def build_features(splits, feature_state=None):
    """Return the lossless legacy FM view of trusted research schema v2."""
    schema, feature_dimension, fitted_state = build_research_schema(
        splits,
        feature_state=feature_state,
    )
    encoded = research_data_module.LegacyFMAdapter.to_legacy(schema)
    return encoded, feature_dimension, fitted_state


# Stage 3 extension point. X is a dense integer ID array of shape (batch, 5),
# one categorical feature ID per field. The baseline is deliberately implemented
# in PyTorch so neural candidates inherit a real CUDA execution path.
class CandidateModel(nn.Module):
    def __init__(self, feature_dimension, k=16, lr=0.001, l2=1e-6, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.V = nn.Embedding(feature_dimension, k)
        self.W = nn.Embedding(feature_dimension, 1)
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float32))
        nn.init.normal_(self.V.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.W.weight, mean=0.0, std=0.01)
        self.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.loss_fn = nn.BCEWithLogitsLoss()

    @property
    def device(self):
        return self.b.device

    def forward(self, x):
        embeddings = self.V(x)
        linear = self.W(x).sum(dim=1).squeeze(-1) + self.b
        summed = embeddings.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction

    def step(self, x, y):
        self.train()
        x_tensor = torch.as_tensor(x, dtype=torch.long, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.forward(x_tensor)
        loss = self.loss_fn(logits, y_tensor)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, x, batch_size=65536):
        self.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                x_tensor = torch.as_tensor(
                    x[start : start + batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                predictions.append(self.forward(x_tensor).cpu().numpy())
        return np.concatenate(predictions)

    def state_dict(self):
        return {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in super().state_dict().items()
        }

    def load_state_dict(self, state):
        tensor_state = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in state.items()
        }
        return super().load_state_dict(tensor_state)


def create_model(feature_dimension, config=None):
    """Create one candidate from the complete frozen training configuration."""
    effective_config = dict(CONFIG)
    if config is not None:
        effective_config.update(config)
    return CandidateModel(
        feature_dimension,
        k=int(effective_config["embedding_dim"]),
        lr=float(effective_config["learning_rate"]),
        l2=float(effective_config["l2"]),
        seed=int(effective_config["seed"]),
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"checkpoint metadata is not JSON serializable: {type(value)}")


def _state_as_numpy(model):
    state = {}
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        state[name] = np.asarray(value).copy()
    return state


def save_candidate_checkpoint(
    path,
    model,
    feature_state,
    config,
    feature_dimension,
    metadata=None,
):
    """Save everything needed to reproduce inference without retraining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "feature_dimension": int(feature_dimension),
        "feature_state": _json_safe(feature_state),
        "config": _json_safe(config),
        "metadata": _json_safe(metadata or {}),
    }
    arrays = {
        f"state::{name}": value for name, value in _state_as_numpy(model).items()
    }
    arrays["__metadata_json__"] = np.asarray(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(path, **arrays)
    return path


def load_candidate_checkpoint(path):
    """Load a frozen model, feature mapping, config, and audit metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"candidate checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata_json__" not in archive.files:
            raise ValueError("checkpoint has no frozen feature/config metadata")
        payload = json.loads(str(archive["__metadata_json__"].item()))
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"unsupported checkpoint schema: {payload.get('schema_version')}"
            )
        model = create_model(
            int(payload["feature_dimension"]),
            config=payload["config"],
        )
        state = {
            name.removeprefix("state::"): torch.as_tensor(
                archive[name], device=model.device
            )
            for name in archive.files
            if name.startswith("state::")
        }
    if not state:
        raise ValueError("checkpoint contains no model weights")
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "feature_state": payload["feature_state"],
        "config": payload["config"],
        "feature_dimension": int(payload["feature_dimension"]),
        "metadata": payload.get("metadata", {}),
    }


def run_training():
    """Train on train, select by validation only, and save the best checkpoint."""
    protected = {
        name: file_hash(input_dir / name) for name in ("data.py", "evaluate.py")
    }
    started = time.monotonic()
    loaded = data_module.load(str(input_dir / "KuaiRand-Pure" / "data"))
    splits = {"train": loaded["train"], "valid": loaded["valid"]}
    del loaded
    encoded, feature_dimension, feature_state = build_features(splits)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]

    effective_config = dict(CONFIG)
    model = create_model(feature_dimension, config=effective_config)
    rng = np.random.default_rng(int(effective_config["seed"]))
    best_primary = -np.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, int(effective_config["max_epochs"]) + 1):
        indices = rng.permutation(len(train_y))
        losses = []
        for start in range(0, len(indices), int(effective_config["batch_size"])):
            batch = indices[start : start + int(effective_config["batch_size"])]
            losses.append(model.step(train_x[batch], train_y[batch]))
        scores = np.asarray(model.predict(valid_x), dtype=np.float64)
        if scores.shape != (len(valid_y),) or not np.isfinite(scores).all():
            raise RuntimeError("candidate produced invalid validation predictions")
        metrics = evaluate_module.evaluate(valid_users, valid_y, scores)
        primary = float(metrics["primary"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "GAUC": float(metrics["GAUC"]),
                "nDCG@5": float(metrics["nDCG@5"]),
                "primary": primary,
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
            f"valid_GAUC={metrics['GAUC']:.6f} "
            f"valid_nDCG@5={metrics['nDCG@5']:.6f} "
            f"valid_primary={primary:.6f}",
            flush=True,
        )
        if primary > best_primary + float(effective_config["min_delta"]):
            best_primary, best_epoch = primary, epoch
            best_state = _state_as_numpy(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(effective_config["patience"]):
                break

    if best_state is None:
        raise RuntimeError("candidate training produced no valid checkpoint")
    model.load_state_dict(best_state)
    best_scores = np.asarray(model.predict(valid_x), dtype=np.float64)
    best_metrics = evaluate_module.evaluate(valid_users, valid_y, best_scores)
    after = {
        name: file_hash(input_dir / name) for name in ("data.py", "evaluate.py")
    }
    if after != protected:
        raise RuntimeError("protected data/evaluation infrastructure changed")

    gauc = float(best_metrics["GAUC"])
    ndcg5 = float(best_metrics["nDCG@5"])
    primary = (gauc + ndcg5) / 2.0
    active_components = {
        name: component_enabled(name) for name in ABLATION_COMPONENTS
    }
    result_payload = {
        "split": "validation",
        "target": "long_view",
        "GAUC": gauc,
        "nDCG@5": ndcg5,
        "primary": primary,
        "best_epoch": best_epoch,
        "seed": int(effective_config["seed"]),
        "ablation_target": ABLATION_TARGET,
        "active_ablation_components": active_components,
    }

    working_dir = execution_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    save_candidate_checkpoint(
        working_dir / "candidate_checkpoint.npz",
        model=model,
        feature_state=feature_state,
        config=effective_config,
        feature_dimension=feature_dimension,
        metadata={
            "best_epoch": best_epoch,
            "validation": {"GAUC": gauc, "nDCG@5": ndcg5, "primary": primary},
            "ablation_target": ABLATION_TARGET,
            "active_ablation_components": active_components,
        },
    )
    (working_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )

    experiment_data = {
        "KuaiRand-Pure": {
            "metrics": {
                "validation primary": [primary],
                "validation GAUC": [gauc],
                "validation nDCG@5": [ndcg5],
            },
            "losses": {
                "train": [row["train_loss"] for row in history],
                "val": [],
            },
            "predictions": [],
            "ground_truth": [],
            "metadata": {
                "metric_to_optimize": "validation primary",
                "maximize": True,
                "best_epoch": best_epoch,
                "seed": int(effective_config["seed"]),
                "runtime_seconds": time.monotonic() - started,
                "candidate_class": type(model).__name__,
                "python_executable": sys.executable,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(model.device),
                "gpu_name": (
                    torch.cuda.get_device_name(model.device)
                    if model.device.type == "cuda"
                    else None
                ),
                "cuda_peak_memory_mib": (
                    torch.cuda.max_memory_allocated(model.device) / (1024 ** 2)
                    if model.device.type == "cuda"
                    else 0.0
                ),
                "ablation_target": ABLATION_TARGET,
                "registered_ablation_components": ABLATION_COMPONENTS,
                "active_ablation_components": active_components,
            },
        }
    }
    np.save(working_dir / "experiment_data.npy", experiment_data)
    print("AI_SCIENTIST_RESULT=" + json.dumps(result_payload, sort_keys=True))
    print(f"validation GAUC: {gauc:.9f}")
    print(f"validation nDCG@5: {ndcg5:.9f}")
    print(f"validation primary: {primary:.9f}")
    return {
        "model": model,
        "feature_state": feature_state,
        "feature_dimension": feature_dimension,
        "config": effective_config,
        "history": history,
        "result_payload": result_payload,
    }


model = None
feature_state = None
result_payload = None
if os.environ.get("AI_SCIENTIST_INFERENCE_ONLY") != "1":
    training_outputs = run_training()
    model = training_outputs["model"]
    feature_state = training_outputs["feature_state"]
    result_payload = training_outputs["result_payload"]
