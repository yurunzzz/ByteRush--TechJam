"""Canonical, atomic dashboard snapshots for ByteRush experiment runs.

The AgentManager owns the only authoritative view of stages and journals.  This
module serializes that view after every callback so visual layers never have to
infer a stage from a hypothesis, an arbitrary file timestamp, or a copied log.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SNAPSHOT_SCHEMA_VERSION = 1
STAGE_KEYS = {1: "baseline", 2: "tuning", 3: "creative", 4: "ablation"}
STAGE_DESCRIPTIONS = {
    "baseline": "Runs the protected FM reference implementation and records the validation-only comparison point.",
    "tuning": "Explores controlled FM hyperparameters without changing the trusted data or evaluator.",
    "creative": "Tests new, interpretable ranking hypotheses, model structures, features, and learning objectives.",
    "ablation": "Removes or isolates registered components to verify each contribution under the same validation protocol.",
}
_RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_.+")
_STAGE_RE = re.compile(r"^(?:stage_)?(\d+)_")
_RESULT_MARKER_RE = re.compile(r"AI_SCIENTIST_RESULT(?:\"\s*:\s*|\s*:\s*|\s*=\s*|\s+)", re.IGNORECASE)
_GAUC_RE = re.compile(r"(?:validation|valid)[_\s]*GAUC\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
_NDCG_RE = re.compile(r"(?:validation|valid)[_\s]*nDCG@?5\s*[:=]\s*([0-9.]+)", re.IGNORECASE)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def stage_number(stage_name: str) -> int:
    match = _STAGE_RE.match(stage_name)
    return int(match.group(1)) if match else 0


def _infer_run_id(path: Path) -> str:
    for parent in (path, *path.parents):
        if _RUN_ID_RE.match(parent.name):
            return parent.name
    return path.name


def _result_payload(output: str) -> dict[str, Any]:
    marker = _RESULT_MARKER_RE.search(output)
    if not marker:
        return {}
    try:
        payload, _ = json.JSONDecoder().raw_decode(output[marker.end() :].lstrip())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _metric_primary(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return _float(metric)
    value = metric.get("value", metric)
    if not isinstance(value, dict):
        return _float(value)
    for definition in value.get("metric_names", []):
        if "primary" not in str(definition.get("metric_name", "")).lower():
            continue
        for datum in definition.get("data", []):
            return _float(datum.get("best_value", datum.get("final_value")))
    for key in ("primary", "validation primary", "validation_primary"):
        if key in value:
            return _float(value[key])
    return None


def _metrics(node: dict[str, Any]) -> tuple[float | None, float | None, float | None, str | None]:
    output = "".join(str(item) for item in (node.get("_term_out") or []))
    payload = _result_payload(output)
    primary = _metric_primary(node.get("metric")) or _payload_value(payload, "primary", "validation_primary", "validation primary")
    # Older runs emitted validation_* keys, while newer protected evaluators
    # emit their short aliases.  Both are the same validated result; avoid
    # falling back to an incidental training metric embedded earlier in a log.
    gauc = _payload_value(payload, "GAUC", "validation_GAUC", "validation_gauc", "validation GAUC") or _match(_GAUC_RE, output)
    ndcg = _payload_value(payload, "nDCG@5", "validation_nDCG@5", "validation_ndcg@5", "validation nDCG@5") or _match(_NDCG_RE, output)
    checkpoint = payload.get("checkpoint")
    return primary, gauc, ndcg, str(checkpoint) if checkpoint else None


def _payload_value(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(payload.get(key))
        if value is not None:
            return value
    return None


def _match(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return _float(match.group(1)) if match else None


def _relative_or_absolute(path: Path | None, snapshot_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(snapshot_dir.resolve()))
    except ValueError:
        return str(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _dashboard_index(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Small companion index used for fast run switching and trend rendering.

    It is a pure projection of the full, same-timestamp snapshot.  The
    dashboard only opens the large node/code/log payload for the selected run.
    """
    summary = snapshot.get("run_summary") or {}
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": "dashboard_index",
        "run_id": snapshot.get("run_id"),
        "generated_at": snapshot.get("generated_at"),
        "run_summary": {
            key: summary.get(key)
            for key in ("best_node_id", "best_label", "best_stage", "primary", "gauc", "ndcg", "baseline_primary")
        },
        "stages": [
            {
                "key": stage.get("key"),
                "number": stage.get("number"),
                "summary": stage.get("summary") or {},
            }
            for stage in snapshot.get("stages", [])
        ],
    }


def _node_status(node: dict[str, Any], primary: float | None) -> str:
    persisted = str(node.get("status") or "").lower()
    if persisted in {"pending", "queued", "not_started"}:
        return "pending"
    if node.get("is_buggy") is True or node.get("exc_type") or node.get("exc_info"):
        return "failed"
    return "succeeded" if primary is not None else "running"


def _node_label(node: dict[str, Any], ordinal: int) -> str:
    """Give every node a stable, useful label without changing its payload."""
    explicit = node.get("hyperparam_name") or node.get("ablation_name")
    if explicit:
        return str(explicit)
    plan = str(node.get("plan") or "").strip().replace("\n", " ")
    for prefix in ("Hypothesis:", "hypothesis:", "Experiment:", "experiment:"):
        if plan.startswith(prefix):
            plan = plan[len(prefix) :].strip()
            break
    if plan:
        return plan[:54] + ("…" if len(plan) > 54 else "")
    return f"Experiment {ordinal}"


def _layout(node_ids: list[str], parents: dict[str, str | None]) -> dict[str, list[float]]:
    """Build a deterministic, stage-local tree layout without external graph libs."""
    node_set = set(node_ids)
    depths: dict[str, int] = {}

    def depth(node_id: str, seen: set[str] | None = None) -> int:
        if node_id in depths:
            return depths[node_id]
        seen = seen or set()
        parent = parents.get(node_id)
        if parent not in node_set or parent in seen:
            depths[node_id] = 0
        else:
            depths[node_id] = depth(parent, seen | {node_id}) + 1
        return depths[node_id]

    buckets: dict[int, list[str]] = defaultdict(list)
    for node_id in node_ids:
        buckets[depth(node_id)].append(node_id)
    maximum = max(buckets, default=0)
    layout: dict[str, list[float]] = {}
    for level, members in buckets.items():
        for index, node_id in enumerate(members):
            layout[node_id] = [(index + 1) / (len(members) + 1), level / max(maximum, 1)]
    return layout


def build_snapshot(
    stage_payloads: Iterable[dict[str, Any]],
    *,
    run_id: str,
    snapshot_dir: Path,
) -> dict[str, Any]:
    """Create a canonical snapshot from serialized journal payloads.

    ``stage_payloads`` are ordered source journals. Every stage is grouped by
    the numeric stage prefix, while node IDs are deduplicated globally. A node
    therefore has exactly one origin stage even when a winning parent is carried
    into a later stage.
    """
    ordered = sorted(stage_payloads, key=lambda item: (item["stage_number"], item.get("sequence", 0)))
    groups: dict[int, dict[str, Any]] = {}
    nodes: dict[str, dict[str, Any]] = {}
    parents: dict[str, str | None] = {}
    seen: set[str] = set()

    for source in ordered:
        number = int(source["stage_number"])
        if number not in STAGE_KEYS:
            continue
        key = STAGE_KEYS[number]
        group = groups.setdefault(
            number,
            {
                "key": key,
                "number": number,
                "stage_names": [],
                "description": source.get("description") or STAGE_DESCRIPTIONS[key],
                "goals": source.get("goals") or [],
                "node_ids": [],
                "new_node_ids": [],
                "artifact_dir": source["artifact_dir"],
            },
        )
        group["stage_names"].append(source["stage_name"])
        group["artifact_dir"] = source["artifact_dir"]
        raw = source["journal"]
        parent_lookup = raw.get("node2parent", {})
        for raw_node in raw.get("nodes", []):
            node_id = str(raw_node.get("id"))
            if not node_id:
                continue
            if node_id not in group["node_ids"]:
                group["node_ids"].append(node_id)
            parent_id = parent_lookup.get(node_id) or raw_node.get("parent_id")
            parents.setdefault(node_id, str(parent_id) if parent_id else None)
            if node_id in seen:
                continue
            seen.add(node_id)
            primary, gauc, ndcg, checkpoint = _metrics(raw_node)
            artifact_dir = Path(source["artifact_dir"])
            solution = next(iter(sorted(artifact_dir.glob("best_solution_*.py"))), None)
            nodes[node_id] = {
                "id": node_id,
                "stage": key,
                "stage_number": number,
                "label": _node_label(raw_node, len(nodes) + 1),
                "hypothesis": raw_node.get("plan") or "",
                "analysis": raw_node.get("analysis") or "",
                "code": raw_node.get("code") or "",
                "terminal_output": "".join(str(item) for item in (raw_node.get("_term_out") or [])),
                "primary": primary,
                "gauc": gauc,
                "ndcg": ndcg,
                "status": _node_status(raw_node, primary),
                "error_type": raw_node.get("exc_type"),
                "error_info": _json_safe(raw_node.get("exc_info")),
                "parent_id": parents[node_id],
                "is_seed_node": bool(raw_node.get("is_seed_node")),
                "ablation_name": raw_node.get("ablation_name"),
                "hyperparam_name": raw_node.get("hyperparam_name"),
                "created_at": raw_node.get("ctime"),
                "artifact_paths": {
                    "stage_dir": _relative_or_absolute(artifact_dir, snapshot_dir),
                    "config": _relative_or_absolute(artifact_dir / "config.yaml", snapshot_dir),
                    "solution": _relative_or_absolute(solution, snapshot_dir),
                    "checkpoint": checkpoint,
                },
            }
            group["new_node_ids"].append(node_id)

    stages: list[dict[str, Any]] = []
    for number in sorted(groups):
        group = groups[number]
        group_nodes = [nodes[node_id] for node_id in group["node_ids"] if node_id in nodes]
        candidates = [nodes[node_id] for node_id in group["new_node_ids"] if node_id in nodes]
        succeeded = [node for node in candidates if node["status"] == "succeeded"]
        best = max(succeeded, key=lambda node: node["primary"] or float("-inf"), default=None)
        layout = _layout(group["node_ids"], parents)
        edges = [
            [parent, node_id]
            for node_id in group["node_ids"]
            if (parent := parents.get(node_id)) in set(group["node_ids"])
        ]
        stages.append(
            {
                "key": group["key"],
                "number": group["number"],
                "stage_names": group["stage_names"],
                "description": group["description"],
                "goals": group["goals"],
                "node_ids": group["node_ids"],
                "new_node_ids": group["new_node_ids"],
                "edges": edges,
                "layout": layout,
                "summary": {
                    "total": len(candidates),
                    "succeeded": len(succeeded),
                    "failed": sum(node["status"] == "failed" for node in candidates),
                    "running": sum(node["status"] == "running" for node in candidates),
                    "pending": sum(node["status"] == "pending" for node in candidates),
                    "best_node_id": best["id"] if best else None,
                },
            }
        )

    completed = [node for node in nodes.values() if node["status"] == "succeeded"]
    best = max(completed, key=lambda node: node["primary"] or float("-inf"), default=None)
    baseline_stage = next((stage for stage in stages if stage["key"] == "baseline"), None)
    baseline_node = None
    if baseline_stage:
        baseline_candidates = [nodes[node_id] for node_id in baseline_stage["new_node_ids"] if node_id in nodes and nodes[node_id]["status"] == "succeeded"]
        baseline_node = max(baseline_candidates, key=lambda node: node["primary"] or float("-inf"), default=None)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": time.time(),
        "stages": stages,
        "nodes": list(nodes.values()),
        "run_summary": {
            "best_node_id": best["id"] if best else None,
            "best_label": best["label"] if best else None,
            "best_stage": best["stage"] if best else None,
            "primary": best["primary"] if best else None,
            "gauc": best["gauc"] if best else None,
            "ndcg": best["ndcg"] if best else None,
            "baseline_primary": baseline_node["primary"] if baseline_node else None,
        },
    }


def write_dashboard_snapshot(cfg: Any, manager: Any) -> Path:
    """Write one atomic run-level snapshot directly from AgentManager state."""
    # Keep the historic backfill utility lightweight: it reads already-serialized
    # journals and does not require the AgentManager dependency stack.
    from . import serialize

    snapshot_dir = Path(cfg.log_dir)
    payloads = []
    for sequence, stage in enumerate(manager.stages):
        journal = manager.journals.get(stage.name)
        if journal is None:
            continue
        payloads.append(
            {
                "stage_name": stage.name,
                "stage_number": stage.stage_number,
                "description": stage.description,
                "goals": list(stage.goals),
                "sequence": sequence,
                "artifact_dir": str(snapshot_dir / f"stage_{stage.name}"),
                "journal": json.loads(serialize.dumps_json(journal)),
            }
        )
    snapshot = build_snapshot(payloads, run_id=_infer_run_id(snapshot_dir), snapshot_dir=snapshot_dir)
    path = snapshot_dir / "dashboard_snapshot.json"
    _atomic_json(path, snapshot)
    _atomic_json(snapshot_dir / "dashboard_index.json", _dashboard_index(snapshot))
    return path


def rebuild_dashboard_snapshot(run_dir: Path) -> Path | None:
    """Build the same schema for a historic run using its saved journal files."""
    payloads = []
    for sequence, journal_path in enumerate(run_dir.rglob("journal.json")):
        number = stage_number(journal_path.parent.name)
        if number not in STAGE_KEYS:
            continue
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        latest_ctime = max((node.get("ctime") or 0 for node in journal.get("nodes", [])), default=0)
        payloads.append(
            {
                "stage_name": journal_path.parent.name.removeprefix("stage_"),
                "stage_number": number,
                "description": STAGE_DESCRIPTIONS[STAGE_KEYS[number]],
                "goals": [],
                "sequence": (number, latest_ctime, sequence),
                "artifact_dir": str(journal_path.parent),
                "journal": journal,
            }
        )
    if not payloads:
        return None
    snapshot = build_snapshot(payloads, run_id=run_dir.name, snapshot_dir=run_dir)
    path = run_dir / "dashboard_snapshot.json"
    _atomic_json(path, snapshot)
    _atomic_json(run_dir / "dashboard_index.json", _dashboard_index(snapshot))
    return path
