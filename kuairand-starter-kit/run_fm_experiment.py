#!/usr/bin/env python3
"""Run one reproducible FM experiment and expose validation results only.

This is the trusted boundary between an experiment agent and the KuaiRand
Starter Kit. The agent may choose values in a small JSON configuration, but it
does not choose the data split, label, evaluator, or metric implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import baseline as baseline_module
import data as data_module
import evaluate as evaluate_module


RESULT_PREFIX = "AI_SCIENTIST_RESULT "
CONFIG_KEYS = {
    "experiment_name",
    "seed",
    "embedding_dim",
    "learning_rate",
    "l2",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "min_delta",
}
DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "fm_validation_baseline",
    "seed": 0,
    "embedding_dim": 16,
    "learning_rate": 0.001,
    "l2": 1e-6,
    "batch_size": 8192,
    "max_epochs": 40,
    "early_stopping_patience": 4,
    "min_delta": 1e-5,
}
PROTECTED_FILES = ("data.py", "evaluate.py")


class ConfigError(ValueError):
    """Raised when an experiment config violates the interface contract."""


def _require_int(config: dict[str, Any], key: str, lo: int, hi: int) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ConfigError(f"{key} must be an integer in [{lo}, {hi}], got {value!r}")


def _require_number(config: dict[str, Any], key: str, lo: float, hi: float) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a finite number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ConfigError(f"{key} must be in [{lo}, {hi}], got {value!r}")
    config[key] = value


def load_config(path: Path) -> dict[str, Any]:
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read valid JSON config {path}: {exc}") from exc
    if not isinstance(supplied, dict):
        raise ConfigError("config root must be a JSON object")
    unknown = sorted(set(supplied) - CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")

    config = {**DEFAULT_CONFIG, **supplied}
    name = config["experiment_name"]
    if not isinstance(name, str) or not name or len(name) > 80:
        raise ConfigError("experiment_name must be a non-empty string of at most 80 characters")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name):
        raise ConfigError("experiment_name may contain only letters, digits, '_' and '-'")

    _require_int(config, "seed", 0, 2**32 - 1)
    _require_int(config, "embedding_dim", 1, 512)
    _require_int(config, "batch_size", 32, 1_000_000)
    _require_int(config, "max_epochs", 1, 500)
    _require_int(config, "early_stopping_patience", 1, 100)
    _require_number(config, "learning_rate", 1e-7, 1.0)
    _require_number(config, "l2", 0.0, 1.0)
    _require_number(config, "min_delta", 0.0, 0.1)
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(repo_dir: Path) -> dict[str, str]:
    return {name: sha256(repo_dir / name) for name in PROTECTED_FILES}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finite_predictions(scores: np.ndarray, expected_rows: int) -> None:
    if scores.shape != (expected_rows,):
        raise RuntimeError(f"prediction shape is {scores.shape}, expected ({expected_rows},)")
    if not np.isfinite(scores).all():
        raise RuntimeError("validation predictions contain NaN or infinity")


def run_experiment(config: dict[str, Any], data_dir: Path, output_dir: Path) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    before_hashes = protected_hashes(repo_dir)
    started = time.monotonic()

    # data.load() follows the official chronological split. Only train and valid
    # are retained by this harness; the test split is never encoded or evaluated.
    loaded = data_module.load(str(data_dir))
    splits = {"train": loaded["train"], "valid": loaded["valid"]}
    del loaded
    encoded, feature_dimension = data_module.encode(splits)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]

    model = baseline_module.FM(
        feature_dimension,
        k=config["embedding_dim"],
        lr=config["learning_rate"],
        l2=config["l2"],
        seed=config["seed"],
    )
    rng = np.random.default_rng(config["seed"])
    best_primary = -np.inf
    best_epoch = 0
    best_state: tuple[np.ndarray, np.ndarray, np.float32] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config["max_epochs"] + 1):
        epoch_started = time.monotonic()
        indices = rng.permutation(len(train_y))
        losses = []
        for start in range(0, len(indices), config["batch_size"]):
            batch = indices[start : start + config["batch_size"]]
            losses.append(model.step(train_x[batch], train_y[batch]))

        validation_scores = model.predict(valid_x)
        finite_predictions(validation_scores, len(valid_y))
        metrics = evaluate_module.evaluate(valid_users, valid_y, validation_scores)
        gauc = float(metrics["GAUC"])
        ndcg5 = float(metrics["nDCG@5"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "GAUC": gauc,
            "nDCG@5": ndcg5,
            "primary": (gauc + ndcg5) / 2.0,
            "duration_seconds": round(time.monotonic() - epoch_started, 6),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={epoch_record['train_loss']:.6f} "
            f"valid_GAUC={epoch_record['GAUC']:.6f} "
            f"valid_nDCG@5={epoch_record['nDCG@5']:.6f} "
            f"valid_primary={epoch_record['primary']:.6f}",
            flush=True,
        )

        if metrics["primary"] > best_primary + config["min_delta"]:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= config["early_stopping_patience"]:
                break

    if best_state is None:
        raise RuntimeError("training completed without a valid checkpoint")
    model.V, model.W, model.b = best_state
    best_scores = model.predict(valid_x)
    finite_predictions(best_scores, len(valid_y))
    best_metrics = evaluate_module.evaluate(valid_users, valid_y, best_scores)

    after_hashes = protected_hashes(repo_dir)
    if after_hashes != before_hashes:
        raise RuntimeError("protected infrastructure changed during the experiment")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.npz"
    np.savez_compressed(
        checkpoint_path,
        V=model.V,
        W=model.W,
        b=np.asarray(model.b),
        feature_dimension=np.asarray(feature_dimension),
        best_epoch=np.asarray(best_epoch),
    )
    atomic_json(output_dir / "config.json", config)
    atomic_json(output_dir / "history.json", {"epochs": history})

    final_gauc = float(best_metrics["GAUC"])
    final_ndcg5 = float(best_metrics["nDCG@5"])
    result = {
        "schema_version": 1,
        "status": "success",
        "experiment_name": config["experiment_name"],
        "split": "valid",
        "label": data_module.LABEL,
        "metric_direction": "maximize",
        "GAUC": final_gauc,
        "nDCG@5": final_ndcg5,
        "primary": (final_gauc + final_ndcg5) / 2.0,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "seed": config["seed"],
        "train_rows": len(train_y),
        "validation_rows": len(valid_y),
        "validation_users": int(best_metrics["users"]),
        "runtime_seconds": round(time.monotonic() - started, 6),
        "checkpoint": str(checkpoint_path.resolve()),
        "protected_file_sha256": after_hashes,
    }
    atomic_json(output_dir / "metrics.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one controlled FM experiment and report validation metrics only."
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON experiment config")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("KuaiRand-Pure/data"), help="official data directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/fm_validation"), help="artifact directory"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = run_experiment(config, args.data_dir, args.output_dir)
    except Exception as exc:
        error = {
            "schema_version": 1,
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(RESULT_PREFIX + json.dumps(error, sort_keys=True), flush=True)
        return 1

    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
