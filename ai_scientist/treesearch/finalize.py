"""Freeze the statistically best Stage-4 candidate as a reproducible artifact."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _metric_value(node: Any) -> float:
    if node.metric is None:
        return float("nan")
    return float(node.metric.get_mean_value())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_metrics(node: Any) -> dict[str, float] | None:
    """Read trusted validation metrics saved for a node, when available."""
    if not node.exp_results_dir:
        return None
    files = list(Path(node.exp_results_dir).rglob("experiment_data.npy"))
    if not files:
        return None
    try:
        data = np.load(files[0], allow_pickle=True).item()
        dataset_payload = next(
            payload
            for payload in data.values()
            if isinstance(payload, dict) and "metrics" in payload
        )
        metrics = dataset_payload["metrics"]
        return {
            "GAUC": float(metrics["validation GAUC"][0]),
            "nDCG@5": float(metrics["validation nDCG@5"][0]),
            "primary": float(metrics["validation primary"][0]),
        }
    except (KeyError, IndexError, TypeError, ValueError, OSError):
        return None


def freeze_stage4_winner(
    journal: Any,
    output_dir: str | Path,
    source_stage: str,
    required_seeds: int = 3,
) -> dict[str, Any]:
    """Select by mean validation primary across Stage-4 seed reruns and freeze code.

    Only non-buggy, non-seed full/ablation candidates with at least
    ``required_seeds`` successful seed children are eligible.
    """
    output_dir = Path(output_dir).resolve()
    journal_nodes = set(journal.nodes)
    candidates = []
    for node in journal.nodes:
        if node.is_seed_node or node.is_buggy or node.metric is None:
            continue
        seeds = [
            child
            for child in node.children
            if child in journal_nodes
            and child.is_seed_node
            and not child.is_seed_agg_node
            and not child.is_buggy
            and math.isfinite(_metric_value(child))
        ]
        if len(seeds) < required_seeds:
            continue
        seed_results = [
            (seed, metrics)
            for seed in seeds
            if (metrics := _validation_metrics(seed)) is not None
        ]
        if len(seed_results) < required_seeds:
            continue
        primary_values = [metrics["primary"] for _, metrics in seed_results]
        candidates.append((float(np.mean(primary_values)), node, seed_results))

    if not candidates:
        raise RuntimeError(
            f"Stage 4 produced no candidate with {required_seeds} successful "
            "trusted multi-seed evaluations; final model was not frozen"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, winner, seed_results = candidates[0]
    seed_metrics = [metrics for _, metrics in seed_results]
    summary = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        values = [metric[name] for metric in seed_metrics]
        summary[name] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }

    best_seed_node, best_seed_metrics = max(
        seed_results,
        key=lambda item: item[1]["primary"],
    )
    if not best_seed_node.exp_results_dir:
        raise RuntimeError(
            f"best Stage-4 seed node {best_seed_node.id} has no result directory"
        )
    checkpoint_source = (
        Path(best_seed_node.exp_results_dir) / "candidate_checkpoint.npz"
    )
    if not checkpoint_source.is_file():
        raise RuntimeError(
            f"best Stage-4 seed node {best_seed_node.id} has no complete "
            f"checkpoint: {checkpoint_source}"
        )
    history_source = Path(best_seed_node.exp_results_dir) / "history.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "model.py",
        "manifest.json",
        "source_node_id.txt",
        "checkpoint.npz",
        "training_history.json",
        "export_checkpoint.npz",
        "export_history.json",
        "submission.csv",
        "submission.csv.metadata.json",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    model_path = output_dir / "model.py"
    model_path.write_text(winner.code)
    checkpoint_path = output_dir / "checkpoint.npz"
    shutil.copy2(checkpoint_source, checkpoint_path)
    if history_source.is_file():
        shutil.copy2(history_source, output_dir / "training_history.json")
    model_digest = hashlib.sha256(winner.code.encode("utf-8")).hexdigest()
    checkpoint_digest = _file_sha256(checkpoint_path)
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage": source_stage,
        "source_node_id": winner.id,
        "ablation_name": winner.ablation_name,
        "ablation_target": winner.ablation_name or "full",
        "required_successful_seeds": required_seeds,
        "successful_seed_count": len(seed_metrics),
        "validation": summary,
        "selection_metric": "validation primary mean across seeds",
        "model_sha256": model_digest,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_source_node_id": best_seed_node.id,
        "checkpoint_validation_primary": best_seed_metrics["primary"],
        "test_metrics_used_for_selection": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "source_node_id.txt").write_text(winner.id + "\n")
    return manifest
