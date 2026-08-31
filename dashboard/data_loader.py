"""Read canonical ByteRush dashboard snapshots.

Snapshots are written atomically by AgentManager. This module deliberately does
not infer a stage from a hypothesis or select a run by copied file mtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STAGE_META = {
    "baseline": {"label": "Baseline", "description": "Runs the protected FM reference implementation and records the validation-only comparison point."},
    "tuning": {"label": "Tuning", "description": "Explores controlled FM hyperparameters without changing the trusted data or evaluator."},
    "creative": {"label": "Creative research", "description": "Tests new, interpretable ranking hypotheses, model structures, features, and learning objectives."},
    "ablation": {"label": "Ablation", "description": "Removes or isolates registered components to verify each contribution under the same validation protocol."},
}
SNAPSHOT_SCHEMA_VERSION = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve(snapshot_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else snapshot_dir / path


@dataclass
class ExperimentNode:
    uid: str
    index: int
    stage: str
    label: str
    plan: str
    analysis: str
    code: str
    terminal_output: str
    primary: float | None
    gauc: float | None
    ndcg: float | None
    error_type: str | None
    error_info: str | None
    execution_status: str
    is_best: bool
    x: float
    y: float
    artifact_dir: Path
    ablation_name: str = ""
    hyperparam_name: str = ""
    parent_id: str | None = None
    is_stage_candidate: bool = True
    artifact_paths: dict[str, Path | None] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.execution_status in {"pending", "queued", "not_started"}:
            return "pending"
        if self.error_type or self.error_info:
            return "failed"
        if self.primary is not None:
            return "succeeded"
        return "running"


@dataclass
class TreeSnapshot:
    stage: str
    path: Path
    nodes: list[ExperimentNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def _index_from_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility projection for a snapshot created before fast indexes."""
    summary = payload.get("run_summary") or {}
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": "dashboard_index",
        "run_id": payload.get("run_id"),
        "generated_at": payload.get("generated_at"),
        "run_summary": summary,
        "stages": [
            {"key": stage.get("key"), "number": stage.get("number"), "summary": stage.get("summary") or {}}
            for stage in payload.get("stages", [])
        ],
    }


def _run_indexes(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load only compact run indexes; the full node payload stays lazy."""
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    experiments = root / "experiments"
    if not experiments.exists():
        return []
    indexed_runs: set[str] = set()
    for path in experiments.rglob("dashboard_index.json"):
        payload = _read_json(path)
        if not payload or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION or payload.get("kind") != "dashboard_index":
            continue
        run_id = str(payload.get("run_id") or path.parent.name)
        indexed_runs.add(run_id)
        previous = chosen.get(run_id)
        if previous is None or float(payload.get("generated_at") or 0) >= float(previous[1].get("generated_at") or 0):
            chosen[run_id] = (path, payload)
    # Existing copies remain usable before their one-time backfill.  New runs
    # always have the lightweight index and skip this expensive fallback.
    for path in experiments.rglob("dashboard_snapshot.json"):
        payload = _read_json(path)
        if not payload or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            continue
        run_id = str(payload.get("run_id") or path.parent.name)
        if run_id in indexed_runs:
            continue
        chosen[run_id] = (path, _index_from_snapshot(payload))
    return sorted(chosen.values(), key=lambda item: str(item[1].get("run_id", "")))


def _snapshot_for_run(root: Path, run_id: str | None = None) -> tuple[Path, dict[str, Any]] | None:
    snapshots = _run_indexes(root)
    if run_id:
        return next((item for item in snapshots if str(item[1].get("run_id")) == run_id), None)
    return snapshots[-1] if snapshots else None


def _full_snapshot(index_path: Path, index: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    """Open the large payload only for the run being inspected."""
    path = index_path if index_path.name == "dashboard_snapshot.json" else index_path.with_name("dashboard_snapshot.json")
    payload = _read_json(path)
    if not payload or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        return None
    if str(payload.get("run_id")) != str(index.get("run_id")):
        return None
    if abs(float(payload.get("generated_at") or 0) - float(index.get("generated_at") or 0)) > 1e-6:
        return None
    return path, payload


def available_runs(root: Path, *, indexes: list[tuple[Path, dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Return run-scoped metadata for the dashboard's run selector.

    The selector deliberately uses snapshot run IDs, never timestamps from
    copied files.  The newest run is listed first so a running experiment is
    immediately visible while older completed four-stage runs remain reviewable.
    """
    records: list[dict[str, Any]] = []
    for _, snapshot in reversed(indexes if indexes is not None else _run_indexes(root)):
        summary = snapshot.get("run_summary") or {}
        stages = [str(stage.get("key")) for stage in snapshot.get("stages", []) if stage.get("key") in STAGE_META]
        records.append({
            "run_id": str(snapshot.get("run_id")),
            "primary": _float(summary.get("primary")),
            "gauc": _float(summary.get("gauc")),
            "ndcg": _float(summary.get("ndcg")),
            "baseline_primary": _float(summary.get("baseline_primary")),
            "stage_keys": stages,
        })
    return records


def _node_from_snapshot(raw: dict[str, Any], *, snapshot_dir: Path, stage: dict[str, Any], index: int) -> ExperimentNode:
    layout = stage.get("layout", {}).get(raw["id"], [0.5, 0.5])
    artifacts = raw.get("artifact_paths") or {}
    paths = {name: _resolve(snapshot_dir, value) for name, value in artifacts.items()}
    return ExperimentNode(
        uid=str(raw["id"]), index=index, stage=str(stage["key"]),
        label=str(raw.get("label") or f"Experiment {index + 1}"),
        plan=str(raw.get("hypothesis") or ""), analysis=str(raw.get("analysis") or ""),
        code=str(raw.get("code") or ""), terminal_output=str(raw.get("terminal_output") or ""),
        primary=_float(raw.get("primary")), gauc=_float(raw.get("gauc")), ndcg=_float(raw.get("ndcg")),
        error_type=str(raw.get("error_type") or ""), error_info=str(raw.get("error_info") or ""),
        execution_status=str(raw.get("status") or ""),
        is_best=str(stage.get("summary", {}).get("best_node_id")) == str(raw["id"]),
        x=float(layout[0]), y=float(layout[1]), artifact_dir=paths.get("stage_dir") or snapshot_dir,
        ablation_name=str(raw.get("ablation_name") or ""), hyperparam_name=str(raw.get("hyperparam_name") or ""),
        parent_id=str(raw["parent_id"]) if raw.get("parent_id") else None,
        is_stage_candidate=str(raw["id"]) in set(stage.get("new_node_ids") or []), artifact_paths=paths,
    )


def find_latest_trees(experiments_root: Path, run_id: str | None = None) -> dict[str, TreeSnapshot]:
    selected_index = _snapshot_for_run(experiments_root.parent, run_id)
    if selected_index is None:
        return {}
    full_snapshot = _full_snapshot(*selected_index)
    if full_snapshot is None:
        return {}
    path, payload = full_snapshot
    raw_nodes = {str(node["id"]): node for node in payload.get("nodes", []) if isinstance(node, dict) and node.get("id")}
    trees: dict[str, TreeSnapshot] = {}
    for stage in payload.get("stages", []):
        key = stage.get("key")
        if key not in STAGE_META:
            continue
        node_ids = [str(node_id) for node_id in stage.get("node_ids", []) if str(node_id) in raw_nodes]
        nodes = [_node_from_snapshot(raw_nodes[node_id], snapshot_dir=path.parent, stage=stage, index=index) for index, node_id in enumerate(node_ids)]
        trees[key] = TreeSnapshot(
            stage=key, path=path, nodes=nodes,
            edges=[(str(left), str(right)) for left, right in stage.get("edges", []) if str(left) in raw_nodes and str(right) in raw_nodes],
            progress=stage.get("summary") or {},
            description=str(stage.get("description") or STAGE_META[key]["description"]),
        )
    return trees


def load_ledger(root: Path, *, indexes: list[tuple[Path, dict[str, Any]]] | None = None) -> tuple[float | None, list[dict[str, Any]]]:
    """Build the evolution series from the same snapshots as the rest of the UI."""
    rows: list[dict[str, Any]] = []
    for _, snapshot in indexes if indexes is not None else _run_indexes(root):
        summary = snapshot.get("run_summary") or {}
        primary = _float(summary.get("primary"))
        if primary is None:
            continue
        rows.append({
            "run_id": str(snapshot.get("run_id")), "label": str(summary.get("best_label") or snapshot.get("run_id")),
            "stage": str(summary.get("best_stage") or ""), "primary": primary,
            "gauc": _float(summary.get("gauc")), "ndcg": _float(summary.get("ndcg")),
            "baseline_primary": _float(summary.get("baseline_primary")), "sort_key": str(snapshot.get("run_id")),
            "stage_reached": str(summary.get("best_stage") or ""),
        })
    baseline = rows[-1].get("baseline_primary") if rows else None
    return baseline, rows


def load_overview(root: Path) -> tuple[float | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the compact index set once for the top cards, selector and trend."""
    indexes = _run_indexes(root)
    baseline, ledger = load_ledger(root, indexes=indexes)
    return baseline, ledger, available_runs(root, indexes=indexes)


def best_node(trees: dict[str, TreeSnapshot]) -> ExperimentNode | None:
    nodes = [node for tree in trees.values() for node in tree.nodes if node.is_stage_candidate and node.primary is not None]
    return max(nodes, key=lambda node: node.primary or float("-inf"), default=None)


def stage_stats(stage: str, tree: TreeSnapshot | None, baseline: float | None) -> dict[str, Any]:
    if tree is None:
        return {"total": 0, "success": 0, "failed": 0, "best": None, "delta": None, "progress": None}
    nodes = [node for node in tree.nodes if node.is_stage_candidate]
    succeeded = [node for node in nodes if node.status == "succeeded"]
    failed = [node for node in nodes if node.status == "failed"]
    best = max(succeeded, key=lambda node: node.primary or float("-inf"), default=None)
    delta = (best.primary - baseline) if best and baseline is not None and best.primary is not None else None
    return {"total": len(nodes), "success": len(succeeded), "failed": len(failed), "best": best, "delta": delta, "progress": tree.progress}


def artifact_files(node: ExperimentNode) -> dict[str, Path | None]:
    solution, config, checkpoint = (node.artifact_paths.get(name) for name in ("solution", "config", "checkpoint"))
    if solution and not solution.exists():
        solution = next(iter(sorted(node.artifact_dir.glob("best_solution_*.py"))), None)
    if config and not config.exists():
        config = node.artifact_dir / "config.yaml"
    if checkpoint and not checkpoint.exists():
        checkpoint = next(iter(sorted(node.artifact_dir.rglob("candidate_checkpoint.npz"))), None)
    return {"config": config if config and config.exists() else None, "solution": solution if solution and solution.exists() else None, "checkpoint": checkpoint if checkpoint and checkpoint.exists() else None}
