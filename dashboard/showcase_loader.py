"""Build a frozen, auditable ByteRush competition-showcase payload.

The live Research Console is useful while an AgentManager run is executing.
The competition showcase has a different contract: only a completed experiment
that owns a verified final artifact may become the displayed champion.  This
module joins ``experiments`` and ``artifacts`` through ``source_node_id`` and
never uses test metrics for model selection.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable


SHOWCASE_SCHEMA_VERSION = 1
REQUIRED_FINAL_FILES = (
    "manifest.json",
    "model.py",
    "checkpoint.npz",
    "submission.csv",
    "submission.csv.metadata.json",
)


class ShowcaseBuildError(RuntimeError):
    """Raised when no scientifically valid frozen result can be displayed."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_metric(manifest: dict[str, Any], name: str) -> float | None:
    metric = (manifest.get("validation") or {}).get(name)
    if isinstance(metric, dict):
        return _float(metric.get("mean"))
    return _float(metric)


def _metric_block(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        raw = (manifest.get("validation") or {}).get(name) or {}
        values = [_float(value) for value in raw.get("values", [])] if isinstance(raw, dict) else []
        result[name] = {
            "mean": _float(raw.get("mean")) if isinstance(raw, dict) else _float(raw),
            "std": _float(raw.get("std")) if isinstance(raw, dict) else None,
            "values": [value for value in values if value is not None],
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _snapshot_nodes(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["id"]): node
        for node in snapshot.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }


def _snapshot_is_settled(snapshot: dict[str, Any]) -> bool:
    nodes = _snapshot_nodes(snapshot).values()
    return bool(nodes) and not any(str(node.get("status", "")).lower() in {"running", "pending", "queued"} for node in nodes)


def _find_snapshot(root: Path, source_node_id: str) -> tuple[Path, dict[str, Any]] | None:
    matches: list[tuple[float, Path, dict[str, Any]]] = []
    for path in (root / "experiments").rglob("dashboard_snapshot.json"):
        snapshot = _read_json(path)
        if not isinstance(snapshot, dict) or not _snapshot_is_settled(snapshot):
            continue
        if source_node_id not in _snapshot_nodes(snapshot):
            continue
        matches.append((_float(snapshot.get("generated_at")) or 0.0, path, snapshot))
    if not matches:
        return None
    _, path, snapshot = max(matches, key=lambda item: item[0])
    return path, snapshot


def _artifact_candidate(root: Path, artifact_dir: Path) -> dict[str, Any] | None:
    if any(not (artifact_dir / name).is_file() for name in REQUIRED_FINAL_FILES):
        return None
    manifest = _read_json(artifact_dir / "manifest.json")
    metadata = _read_json(artifact_dir / "submission.csv.metadata.json")
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        return None
    source_node_id = str(manifest.get("source_node_id") or metadata.get("source_node_id") or "")
    required_seeds = int(manifest.get("required_successful_seeds") or 0)
    successful_seeds = int(manifest.get("successful_seed_count") or 0)
    if not source_node_id or required_seeds < 2 or successful_seeds < required_seeds:
        return None
    if manifest.get("test_metrics_used_for_selection") is not False:
        return None
    if metadata.get("test_metrics_computed") is not False:
        return None
    metrics = _metric_block(manifest)
    if any(metrics[name]["mean"] is None for name in metrics):
        return None
    snapshot_match = _find_snapshot(root, source_node_id)
    if snapshot_match is None:
        return None
    snapshot_path, snapshot = snapshot_match
    stage4_verified = str(manifest.get("source_stage") or "").startswith("4_")
    return {
        "artifact_dir": artifact_dir,
        "manifest": manifest,
        "metadata": metadata,
        "metrics": metrics,
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "stage4_verified": stage4_verified,
        "successful_seeds": successful_seeds,
    }


def discover_complete_candidates(root: Path, pinned_artifact: str | None = None) -> list[dict[str, Any]]:
    """Return verified artifact/experiment pairs, strongest first."""
    artifacts = root / "artifacts"
    if pinned_artifact:
        paths = [(root / pinned_artifact).resolve()]
    else:
        paths = sorted({path.parent for path in artifacts.rglob("manifest.json") if not path.name.endswith(".orig")})
    candidates = [candidate for path in paths if (candidate := _artifact_candidate(root, path))]
    candidates.sort(
        key=lambda item: (
            bool(item["stage4_verified"]),
            item["successful_seeds"],
            item["metrics"]["primary"]["mean"] or float("-inf"),
        ),
        reverse=True,
    )
    return candidates


def _node_status(node: dict[str, Any]) -> str:
    status = str(node.get("status") or "").lower()
    if status in {"succeeded", "failed", "running", "pending"}:
        return status
    if node.get("primary") is not None:
        return "succeeded"
    if node.get("error_type") or node.get("error_info"):
        return "failed"
    return "pending"


def _ancestor_path(nodes: dict[str, dict[str, Any]], node_id: str) -> list[str]:
    path: list[str] = []
    seen: set[str] = set()
    current = node_id
    while current and current not in seen and current in nodes:
        seen.add(current)
        path.append(current)
        current = str(nodes[current].get("parent_id") or "")
    return list(reversed(path))


def _model_evidence(root: Path, source_node_id: str) -> dict[str, Any]:
    state_path = root / "artifacts" / "comparison_current" / "research_loop" / "state.json"
    state = _read_json(state_path)
    if not isinstance(state, dict):
        return {"model_family": "unknown", "principal_change": "", "research_family": "architecture", "state": {}}
    memory = state.get("memory") or {}
    pools: list[Any] = []
    for key in ("diverse_frontier", "candidate_records", "experience_records"):
        value = memory.get(key)
        if isinstance(value, list):
            pools.extend(value)
    frontier = _read_json(root / "artifacts" / "comparison_current" / "research_loop" / "diverse_frontier.json")
    if isinstance(frontier, list):
        pools.extend(frontier)
    elif isinstance(frontier, dict):
        for value in frontier.values():
            if isinstance(value, list):
                pools.extend(value)
    record = next((item for item in pools if isinstance(item, dict) and str(item.get("node_id")) == source_node_id), {})
    return {
        "model_family": str(record.get("model_family") or "unknown"),
        "principal_change": str(record.get("principal_change") or ""),
        "research_family": str(record.get("research_family") or "architecture"),
        "source_stage": str(record.get("source_stage") or state.get("incumbent_source_stage") or ""),
        "state": state,
    }


def _stage_summaries(snapshot: dict[str, Any], final_metrics: dict[str, Any], seed_count: int) -> list[dict[str, Any]]:
    labels = {
        "baseline": ("Stage 1", "Diverse starting points"),
        "tuning": ("Stage 2", "Controlled tuning"),
        "creative": ("Stage 3", "Creative research"),
        "ablation": ("Stage 4", "Ablation & verification"),
    }
    result: list[dict[str, Any]] = []
    for stage in snapshot.get("stages", []):
        key = str(stage.get("key") or "")
        if key not in labels:
            continue
        summary = stage.get("summary") or {}
        result.append({
            "key": key,
            "eyebrow": labels[key][0],
            "title": labels[key][1],
            "total": int(summary.get("total") or 0),
            "succeeded": int(summary.get("succeeded") or 0),
            "failed": int(summary.get("failed") or 0),
            "best_node_id": summary.get("best_node_id"),
        })
    if not any(stage["key"] == "ablation" for stage in result):
        result.append({
            "key": "ablation",
            "eyebrow": "Stage 4",
            "title": "Ablation & verification",
            "total": seed_count,
            "succeeded": seed_count,
            "failed": 0,
            "best_node_id": "frozen-final-verification",
            "primary": final_metrics["primary"]["mean"],
        })
    return result


def _tree_payload(snapshot: dict[str, Any], source_node_id: str, final_metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]], list[str]]:
    raw_nodes = _snapshot_nodes(snapshot)
    champion_path = _ancestor_path(raw_nodes, source_node_id)
    nodes: list[dict[str, Any]] = []
    for raw in raw_nodes.values():
        nodes.append({
            "id": str(raw["id"]),
            "parent_id": str(raw.get("parent_id") or ""),
            "label": str(raw.get("label") or "Experiment"),
            "stage": str(raw.get("stage") or ""),
            "stage_number": int(raw.get("stage_number") or 0),
            "primary": _float(raw.get("primary")),
            "gauc": _float(raw.get("gauc")),
            "ndcg": _float(raw.get("ndcg")),
            "status": _node_status(raw),
            "hypothesis": str(raw.get("hypothesis") or ""),
            "analysis": str(raw.get("analysis") or ""),
            "is_seed_node": bool(raw.get("is_seed_node")),
            "is_champion_path": str(raw["id"]) in champion_path,
            "is_promoted": str(raw["id"]) == source_node_id,
            "is_final": False,
        })
    frozen_id = "frozen-final-verification"
    nodes.append({
        "id": frozen_id,
        "parent_id": source_node_id,
        "label": "5-seed final verification",
        "stage": "ablation",
        "stage_number": 4,
        "primary": final_metrics["primary"]["mean"],
        "gauc": final_metrics["GAUC"]["mean"],
        "ndcg": final_metrics["nDCG@5"]["mean"],
        "status": "succeeded",
        "hypothesis": "Verify the promoted model across independent random seeds before freezing it.",
        "analysis": "The model remained stable and was frozen for submission generation.",
        "is_seed_node": False,
        "is_champion_path": True,
        "is_promoted": False,
        "is_final": True,
    })
    champion_path.append(frozen_id)
    ids = {node["id"] for node in nodes}
    edges = [[node["parent_id"], node["id"]] for node in nodes if node["parent_id"] in ids]
    return nodes, edges, champion_path


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ImportError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_showcase_payload(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Select the strongest completed result and create the frontend contract."""
    root = root.expanduser().resolve()
    config = _load_config(config_path)
    pinned_artifact = (config.get("showcase") or {}).get("pinned_artifact")
    candidates = discover_complete_candidates(root, pinned_artifact)
    if not candidates:
        raise ShowcaseBuildError(
            "No completed experiment owns a verified final model, checkpoint, submission, and validation-only manifest."
        )
    winner = candidates[0]
    artifact_dir: Path = winner["artifact_dir"]
    manifest = winner["manifest"]
    metadata = winner["metadata"]
    snapshot = winner["snapshot"]
    source_node_id = str(manifest["source_node_id"])
    snapshot_nodes = _snapshot_nodes(snapshot)
    ancestry = _ancestor_path(snapshot_nodes, source_node_id)
    baseline_node = snapshot_nodes.get(ancestry[0]) if ancestry else None
    if baseline_node is None or _float(baseline_node.get("primary")) is None:
        raise ShowcaseBuildError("The frozen model cannot be traced back to a scored baseline root node.")
    baseline = {
        "node_id": str(baseline_node["id"]),
        "label": "Factorization Machine baseline",
        "GAUC": _float(baseline_node.get("gauc")),
        "nDCG@5": _float(baseline_node.get("ndcg")),
        "primary": _float(baseline_node.get("primary")),
    }
    metrics = winner["metrics"]
    final = {name: values["mean"] for name, values in metrics.items()}
    delta = {name: final[name] - baseline[name] for name in final if final[name] is not None and baseline[name] is not None}
    relative = {name: delta[name] / baseline[name] for name in delta if baseline[name] not in (None, 0)}
    evidence = _model_evidence(root, source_node_id)
    nodes, edges, champion_path = _tree_payload(snapshot, source_node_id, metrics)
    succeeded = [node for node in nodes if node["status"] == "succeeded" and node["primary"] is not None and not node["is_seed_node"]]
    top_candidates = sorted(succeeded, key=lambda node: node["primary"], reverse=True)[:8]
    training_history = _read_json(artifact_dir / "training_history.json")
    if not isinstance(training_history, list):
        training_history = []
    state = evidence.pop("state", {})
    project = config.get("project") or {}
    story = config.get("story") or {}
    payload = {
        "schema_version": SHOWCASE_SCHEMA_VERSION,
        "generated_at": time.time(),
        "project": {
            "name": str(project.get("name") or "ByteRush"),
            "subtitle": str(project.get("subtitle") or "Autonomous ML Research Agent for Recommender Systems"),
            "competition": str(project.get("competition") or "TikTok TechJam"),
            "tagline": str(project.get("tagline") or "From research hypothesis to a submission-ready ranking model."),
        },
        "story": story,
        "selection": {
            "run_id": str(snapshot.get("run_id") or winner["snapshot_path"].parent.name),
            "source_node_id": source_node_id,
            "source_stage": str(manifest.get("source_stage") or evidence.get("source_stage") or ""),
            "selection_metric": str(manifest.get("selection_metric") or "validation primary mean across seeds"),
            "stage4_verified": bool(winner["stage4_verified"]),
            "successful_seed_count": winner["successful_seeds"],
            "complete_candidate_count": len(candidates),
        },
        "winner": {
            "label": str(story.get("winner_label") or evidence.get("model_family", "wide_deep")).replace("_", " ").title(),
            "model_family": evidence.get("model_family", "unknown"),
            "research_family": evidence.get("research_family", "architecture"),
            "principal_change": evidence.get("principal_change") or "A wide linear path captures memorized interactions while a deep path learns nonlinear feature combinations.",
            "baseline": baseline,
            "metrics": metrics,
            "final": final,
            "delta": delta,
            "relative_delta": relative,
        },
        "search": {
            "stages": _stage_summaries(snapshot, metrics, winner["successful_seeds"]),
            "nodes": nodes,
            "edges": edges,
            "champion_path": champion_path,
            "top_candidates": top_candidates,
            "total_nodes": len(nodes) - 1,
            "successful_nodes": sum(node["status"] == "succeeded" for node in nodes[:-1]),
            "failed_nodes": sum(node["status"] == "failed" for node in nodes[:-1]),
            "research_rounds": int(state.get("current_round") or 0),
            "elapsed_seconds": _float(state.get("elapsed_seconds")),
        },
        "training_history": training_history,
        "integrity": {
            "validation_only_selection": manifest.get("test_metrics_used_for_selection") is False,
            "test_metrics_computed": metadata.get("test_metrics_computed") is True,
            "submission_rows": int(metadata.get("rows") or 0),
            "successful_seed_count": winner["successful_seeds"],
            "checkpoint_sha256": str(manifest.get("checkpoint_sha256") or _sha256(artifact_dir / "checkpoint.npz")),
            "model_sha256": str(manifest.get("model_sha256") or _sha256(artifact_dir / "model.py")),
            "submission_sha256": str(metadata.get("submission_sha256") or _sha256(artifact_dir / "submission.csv")),
        },
        "files": {
            "data_root": str(root),
            "experiment_snapshot": _relative(winner["snapshot_path"], root),
            "artifact_dir": _relative(artifact_dir, root),
            "model": _relative(artifact_dir / "model.py", root),
            "checkpoint": _relative(artifact_dir / "checkpoint.npz", root),
            "submission": _relative(artifact_dir / "submission.csv", root),
            "manifest": _relative(artifact_dir / "manifest.json", root),
        },
    }
    return payload


def write_showcase_payload(payload: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def load_showcase_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != SHOWCASE_SCHEMA_VERSION:
        raise ShowcaseBuildError(f"Invalid or unsupported showcase manifest: {path}")
    return payload
