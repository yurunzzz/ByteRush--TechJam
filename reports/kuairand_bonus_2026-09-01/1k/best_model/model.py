"""Schema-v2-compatible lightweight MLP root for KuaiRand-1K."""

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
DATASET_NAME = os.getenv("KUAIRAND_DATASET_NAME", "KuaiRand-1K")
print(f"Using device: {DEVICE}", flush=True)

execution_dir = Path.cwd()
input_candidates = (execution_dir / "input", execution_dir.parent / "input")
input_dir = next((p for p in input_candidates if p.is_dir()), None)
if input_dir is None:
    raise FileNotFoundError(f"trusted input directory not found: {input_candidates}")
sys.path.insert(0, str(input_dir))

import baseline as baseline_module  # noqa: E402,F401
import data as data_module  # noqa: E402
import evaluate as evaluate_module  # noqa: E402
import research_data as research_data_module  # noqa: E402

RESEARCH_MANIFEST = {
    "candidate_id": "architecture_mlp",
    "role": "architecture_mlp",
    "group": "architecture_exploration",
    "category": "model_architecture",
    "model_family": "mlp",
    "research_family": "architecture",
    "loss_family": "pointwise_bce",
    "parent_node_id": "60828fa8b73a4ab9aadbea3e82dece1a",
    "parent_model_family": "fm",
    "input_schema_version": 2,
    "assignment_id": "stage1b:1:architecture_mlp",
    "assignment_kind": "bootstrap",
    "autonomous": False,
    "objective": (
        "Replace the FM scorer with one lightweight mlp model while keeping "
        "schema v2, validation-only evaluation, and primary unchanged."
    ),
    "hypothesis": (
        "A shallow embedding MLP over the trusted user/video categorical ID "
        "pair can learn nonlinear long_view relevance patterns distinct from FM."
    ),
    "mechanism": (
        "Replace the FM scorer with one lightweight mlp model while keeping "
        "schema v2, validation-only evaluation, and primary unchanged."
    ),
    "mechanism_ids": ["architecture_mlp"],
    "modified_symbols": [
        "build_features",
        "CandidateModel",
        "create_model",
        "run_training",
    ],
    "expected_metric": ["GAUC", "nDCG@5", "primary"],
    "tunable_parameters": [
        "embedding_dim",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "l2",
    ],
    "ablation_components": ["architecture_mlp_component"],
    "combination_compatibility": (
        "Compatible with train-only causal factors that emit categorical IDs; "
        "this bootstrap uses only the trusted user/video ID pair."
    ),
    "change_scope": "one principal research mechanism",
    "component_dependencies": {
        "architecture_mlp_component": ["custom_trusted_id_pair"]
    },
    "evidence": [
        {
            "source_type": "dependency",
            "reference": "dependency:executed_guarded_component",
            "supports": ["architecture_mlp_component"],
        }
    ],
}

FACTOR_SELECTION = {
    "considered_factor_ids": ["static_user_profile"],
    "selected_factor_ids": [],
    "selection_reason": (
        "The trusted cache exposes only user_id and video_id categorical inputs; "
        "the selected card records their lossless schema-v2 adapter path without "
        "introducing unavailable metadata or learned aggregates."
    ),
    "rejected_reasons": {
        "static_user_profile": (
            "The trusted cache does not expose additional static profile fields."
        )
    },
    "created_factor_cards": [
        {
            "factor_id": "custom_trusted_id_pair",
            "semantics": (
                "The two globally encoded categorical IDs corresponding to the "
                "trusted user_id and video_id fields."
            ),
            "helps_when": (
                "A compact neural scorer can learn nonlinear transformations of "
                "the available identity representation."
            ),
            "model_fit": (
                "Concatenate two learned embeddings and score with a shallow MLP."
            ),
            "avoid_when": (
                "The trusted adapter does not emit exactly two categorical fields."
            ),
            "data_cost": "none beyond the trusted schema-v2 encoding",
            "leakage_rule": (
                "Fit categorical mappings on train only and reuse the frozen "
                "feature_state unchanged on validation and inference."
            ),
        }
    ],
}

FEATURE_FACTORS = [
    {
        "library_id": "custom_trusted_id_pair",
        "name": "Trusted user-video categorical ID pair",
        "raw_fields": ["user_id", "video_id"],
        "transform": (
            "Use build_schema_v2 followed by LegacyFMAdapter and retain the "
            "resulting dense int32 array with shape (N, 2)."
        ),
        "output_fields": ["categorical_id_pair"],
        "state_policy": (
            "Fit the trusted categorical mapping on train only; serialize it in "
            "feature_state and reuse it unchanged for validation and inference."
        ),
    }
]

CONFIG = {
    "seed": int(os.environ.get("AI_SCIENTIST_SEED", "0")),
    "embedding_dim": 8,
    "hidden_dim": 32,
    "dropout": 0.0,
    "num_fields": 2,
    "learning_rate": 0.001,
    "l2": 1e-6,
    "batch_size": 8192,
    "max_epochs": 5,
    "patience": 3,
    "min_delta": 1e-5,
}

ABLATION_COMPONENTS = {"architecture_mlp_component": True}
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
    """Build trusted schema v2, fitting state on train when state is absent."""
    return research_data_module.build_schema_v2(
        splits,
        data_module=data_module,
        feature_state=feature_state,
    )


def build_features(splits, feature_state=None):
    """Expose the trusted two-field categorical pair through the legacy contract."""
    schema, feature_dimension, fitted_state = build_research_schema(
        splits,
        feature_state=feature_state,
    )
    legacy = research_data_module.LegacyFMAdapter.to_legacy(schema)
    encoded = {}

    for split_name, payload in legacy.items():
        x, y, users = payload
        features = {}
        features["categorical_id_pair"] = np.asarray(x, dtype=np.int32)

        categorical_id_pair = features["categorical_id_pair"]
        if categorical_id_pair.ndim != 2:
            raise ValueError(
                f"{split_name} categorical_id_pair must be rank 2, got "
                f"{categorical_id_pair.shape}"
            )
        if categorical_id_pair.shape[1] != 2:
            raise ValueError(
                f"{split_name} expected trusted user/video width 2, got "
                f"{categorical_id_pair.shape[1]}"
            )
        if len(categorical_id_pair) and (
            categorical_id_pair.min() < 0
            or categorical_id_pair.max() >= int(feature_dimension)
        ):
            raise ValueError(
                f"{split_name} categorical IDs fall outside "
                f"[0, {int(feature_dimension)})"
            )

        if component_enabled("architecture_mlp_component"):
            model_input = features["categorical_id_pair"]
        else:
            model_input = features["categorical_id_pair"]

        encoded[split_name] = (
            model_input,
            np.asarray(y, dtype=np.float32),
            np.asarray(users),
        )

    return encoded, feature_dimension, fitted_state


class CandidateModel(nn.Module):
    """Two-field embedding MLP with a simple linear ablation path."""

    def __init__(
        self,
        feature_dimension,
        num_fields=2,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.10,
        lr=0.001,
        l2=1e-6,
        seed=0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.num_fields = int(num_fields)
        self.embedding_dim = int(embedding_dim)
        self.mlp_enabled = component_enabled("architecture_mlp_component")

        self.embedding = nn.Embedding(
            int(feature_dimension),
            self.embedding_dim,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.num_fields * self.embedding_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.linear_ablation = nn.Embedding(int(feature_dimension), 1)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear_ablation.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.to(DEVICE)
        if torch.cuda.is_available() and self.bias.device.type != "cuda":
            raise RuntimeError("CUDA is visible but CandidateModel remains on CPU")
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(lr),
            weight_decay=float(l2),
        )

    @property
    def device(self):
        return self.bias.device

    def _validate_tensor(self, x):
        if x.ndim != 2 or x.shape[1] != self.num_fields:
            raise ValueError(
                f"MLP expected shape (batch, {self.num_fields}), got "
                f"{tuple(x.shape)}"
            )

    def forward(self, x):
        self._validate_tensor(x)
        if component_enabled("architecture_mlp_component"):
            embeddings = self.embedding(x)
            flattened = embeddings.reshape(x.shape[0], -1)
            return self.mlp(flattened).squeeze(-1) + self.bias
        return self.linear_ablation(x).sum(dim=1).squeeze(-1) + self.bias

    def step(self, x, y):
        self.train()
        x_tensor = torch.as_tensor(x, dtype=torch.long, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.forward(x_tensor)
        loss = self.loss_fn(logits, y_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError("candidate produced a non-finite training loss")
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def predict(self, x, batch_size=65536):
        self.eval()
        if len(x) == 0:
            return np.empty(0, dtype=np.float32)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x), int(batch_size)):
                x_tensor = torch.as_tensor(
                    x[start : start + int(batch_size)],
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
    """Create the MLP from the complete frozen training configuration."""
    effective_config = dict(CONFIG)
    if config is not None:
        effective_config.update(config)
    return CandidateModel(
        feature_dimension=int(feature_dimension),
        num_fields=int(effective_config["num_fields"]),
        embedding_dim=int(effective_config["embedding_dim"]),
        hidden_dim=int(effective_config["hidden_dim"]),
        dropout=float(effective_config["dropout"]),
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
    """Save model weights, feature mapping, config, and audit metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "feature_dimension": int(feature_dimension),
        "feature_state": _json_safe(feature_state),
        "config": _json_safe(config),
        "metadata": _json_safe(metadata or {}),
    }
    arrays = {f"state::{name}": value for name, value in _state_as_numpy(model).items()}
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
                archive[name],
                device=model.device,
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
    """Train on train and select exclusively with validation primary."""
    protected = {
        name: file_hash(input_dir / name) for name in ("data.py", "evaluate.py")
    }
    started = time.monotonic()

    loaded = data_module.load(str(input_dir / "data"))
    splits = {"train": loaded["train"], "valid": loaded["valid"]}
    del loaded

    encoded, feature_dimension, feature_state = build_features(splits)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]

    if train_x.ndim != 2 or valid_x.ndim != 2:
        raise ValueError("trusted categorical inputs must be rank-two arrays")
    if train_x.shape[1] != valid_x.shape[1]:
        raise ValueError("train and validation categorical widths differ")
    if train_x.shape[1] != 2:
        raise ValueError(
            f"expected trusted schema-v2 adapter width 2, got {train_x.shape[1]}"
        )

    effective_config = dict(CONFIG)
    effective_config["num_fields"] = int(train_x.shape[1])
    if int(effective_config["max_epochs"]) > 5:
        raise ValueError("bootstrap training may use at most 5 epochs")

    model = create_model(feature_dimension, config=effective_config)

    smoke_count = min(8, len(train_x))
    if smoke_count == 0:
        raise RuntimeError("empty training split")
    smoke_scores = model.predict(train_x[:smoke_count])
    if smoke_scores.shape != (smoke_count,) or not np.isfinite(smoke_scores).all():
        raise RuntimeError("candidate failed pre-training prediction smoke test")

    rng = np.random.default_rng(int(effective_config["seed"]))
    best_primary = -np.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, int(effective_config["max_epochs"]) + 1):
        indices = rng.permutation(len(train_y))
        losses = []
        batch_size = int(effective_config["batch_size"])
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            losses.append(model.step(train_x[batch], train_y[batch]))

        scores = np.asarray(model.predict(valid_x), dtype=np.float64)
        if scores.shape != (len(valid_y),) or not np.isfinite(scores).all():
            raise RuntimeError("candidate produced invalid validation predictions")

        metrics = evaluate_module.evaluate(valid_users, valid_y, scores)
        gauc_epoch = float(metrics["GAUC"])
        ndcg_epoch = float(metrics["nDCG@5"])
        primary = (gauc_epoch + ndcg_epoch) / 2.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "GAUC": gauc_epoch,
                "nDCG@5": ndcg_epoch,
                "primary": primary,
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
            f"valid_GAUC={gauc_epoch:.6f} "
            f"valid_nDCG@5={ndcg_epoch:.6f} "
            f"valid_primary={primary:.6f}",
            flush=True,
        )

        if primary > best_primary + float(effective_config["min_delta"]):
            best_primary = primary
            best_epoch = epoch
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
    if best_scores.shape != (len(valid_y),) or not np.isfinite(best_scores).all():
        raise RuntimeError("best checkpoint produced invalid validation predictions")
    best_metrics = evaluate_module.evaluate(valid_users, valid_y, best_scores)

    after = {name: file_hash(input_dir / name) for name in ("data.py", "evaluate.py")}
    if after != protected:
        raise RuntimeError("protected data/evaluation infrastructure changed")

    gauc = float(best_metrics["GAUC"])
    ndcg5 = float(best_metrics["nDCG@5"])
    primary = (gauc + ndcg5) / 2.0
    active_components = {name: component_enabled(name) for name in ABLATION_COMPONENTS}
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
    checkpoint_path = save_candidate_checkpoint(
        working_dir / "candidate_checkpoint.npz",
        model=model,
        feature_state=feature_state,
        config=effective_config,
        feature_dimension=feature_dimension,
        metadata={
            "best_epoch": best_epoch,
            "validation": {
                "GAUC": gauc,
                "nDCG@5": ndcg5,
                "primary": primary,
            },
            "research_manifest": RESEARCH_MANIFEST,
            "factor_selection": FACTOR_SELECTION,
            "feature_factors": FEATURE_FACTORS,
            "ablation_target": ABLATION_TARGET,
            "active_ablation_components": active_components,
        },
    )

    reloaded = load_candidate_checkpoint(checkpoint_path)
    roundtrip_count = min(64, len(valid_x))
    original_roundtrip = model.predict(valid_x[:roundtrip_count])
    loaded_roundtrip = reloaded["model"].predict(valid_x[:roundtrip_count])
    if not np.allclose(
        original_roundtrip,
        loaded_roundtrip,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("checkpoint round-trip changed candidate predictions")
    del reloaded

    (working_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    experiment_data = {
        DATASET_NAME: {
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
                "dataset": DATASET_NAME,
                "maximize": True,
                "best_epoch": best_epoch,
                "seed": int(effective_config["seed"]),
                "runtime_seconds": time.monotonic() - started,
                "candidate_class": type(model).__name__,
                "model_family": RESEARCH_MANIFEST["model_family"],
                "loss_family": RESEARCH_MANIFEST["loss_family"],
                "input_schema_version": 2,
                "num_fields": int(train_x.shape[1]),
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
                    torch.cuda.max_memory_allocated(model.device) / (1024**2)
                    if model.device.type == "cuda"
                    else 0.0
                ),
                "ablation_target": ABLATION_TARGET,
                "registered_ablation_components": ABLATION_COMPONENTS,
                "active_ablation_components": active_components,
                "research_manifest": RESEARCH_MANIFEST,
                "factor_selection": FACTOR_SELECTION,
                "feature_factors": FEATURE_FACTORS,
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
