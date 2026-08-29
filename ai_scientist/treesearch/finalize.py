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


def _validation_metrics(node: Any) -> dict[str, float] | None:
    """Read trusted validation metrics saved for a node, when available."""
    if not node.exp_results_dir:
        return None
    files = list(Path(node.exp_results_dir).rglob("experiment_data.npy"))
    if not files:
        return None
    try:
        data = np.load(files[0], allow_pickle=True).item()
        metrics = data["KuaiRand-Pure"]["metrics"]
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
        seed_metrics = [_validation_metrics(seed) for seed in seeds]
        seed_metrics = [metric for metric in seed_metrics if metric is not None]
        if len(seed_metrics) < required_seeds:
            continue
        primary_values = [metric["primary"] for metric in seed_metrics]
        candidates.append((float(np.mean(primary_values)), node, seeds, seed_metrics))

    if not candidates:
        raise RuntimeError(
            f"Stage 4 produced no candidate with {required_seeds} successful "
            "trusted multi-seed evaluations; final model was not frozen"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, winner, seed_nodes, seed_metrics = candidates[0]
    summary = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        values = [metric[name] for metric in seed_metrics]
        summary[name] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "model.py",
        "manifest.json",
        "source_node_id.txt",
        "checkpoint.npz",
        "training_history.json",
        "export_checkpoint.npz",
        "export_history.json",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    model_path = output_dir / "model.py"
    model_path.write_text(winner.code)
    digest = hashlib.sha256(winner.code.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage": source_stage,
        "source_node_id": winner.id,
        "ablation_name": winner.ablation_name,
        "ablation_target": winner.ablation_name or "full",
        "required_successful_seeds": required_seeds,
        "successful_seed_count": len(seed_metrics),
        "validation": summary,
        "selection_metric": "validation primary mean across seeds",
        "model_sha256": digest,
        "test_metrics_used_for_selection": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "source_node_id.txt").write_text(winner.id + "\n")

    # Preserve the checkpoint from the best successful seed run for audit and
    # optional deployment. The exporter still defaults to deterministic
    # retraining so it can verify code/data compatibility end to end.
    best_index = int(np.argmax([metric["primary"] for metric in seed_metrics]))
    best_seed_node = seed_nodes[best_index]
    if best_seed_node.exp_results_dir:
        checkpoint = Path(best_seed_node.exp_results_dir) / "candidate_checkpoint.npz"
        history = Path(best_seed_node.exp_results_dir) / "history.json"
        if checkpoint.exists():
            shutil.copy2(checkpoint, output_dir / "checkpoint.npz")
            manifest["checkpoint_source_node_id"] = best_seed_node.id
        if history.exists():
            shutil.copy2(history, output_dir / "training_history.json")
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return manifest
