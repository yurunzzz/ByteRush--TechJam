"""Generate explicit, backward-compatible code-diff logs for experiment journals."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _make_record(node: dict[str, Any], parent_id: str | None,
                 nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    node_id = str(node["id"])
    code = str(node.get("code") or "")
    parent = nodes.get(parent_id) if parent_id else None
    if parent_id and parent is None:
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "status": "missing_parent",
            "note": "Parent code was unavailable; no potentially misleading diff was generated.",
            "parent_code_sha256": None,
            "node_code_sha256": _sha256(code),
            "added_lines": None,
            "deleted_lines": None,
            "code_diff": "",
            "hypothesis": str(node.get("plan") or node.get("overall_plan") or ""),
            "ctime": node.get("ctime"),
        }

    parent_code = str(parent.get("code") or "") if parent else ""
    from_name = f"parent/{parent_id}.py" if parent_id else "/dev/null"
    to_name = f"node/{node_id}.py"
    diff = "".join(
        difflib.unified_diff(
            parent_code.splitlines(keepends=True),
            code.splitlines(keepends=True),
            fromfile=from_name,
            tofile=to_name,
            n=3,
        )
    )
    if not parent_id:
        status = "initial_code"
        note = "Root implementation compared with /dev/null."
    elif parent_code == code:
        status = "no_code_change"
        note = "No Python code change; this may be a configuration-only iteration."
    else:
        status = "changed"
        note = "Explicit unified diff against the recorded parent node."
    lines = diff.splitlines()
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "status": status,
        "note": note,
        "parent_code_sha256": _sha256(parent_code) if parent_id else None,
        "node_code_sha256": _sha256(code),
        "added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in lines),
        "deleted_lines": sum(line.startswith("-") and not line.startswith("---") for line in lines),
        "code_diff": diff,
        "hypothesis": str(node.get("plan") or node.get("overall_plan") or ""),
        "ctime": node.get("ctime"),
    }


def build_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build records from a serialized journal, preferring top-level node2parent."""
    raw_nodes = payload.get("nodes", [])
    nodes = {str(node["id"]): node for node in raw_nodes}
    parent_map = {str(k): str(v) for k, v in payload.get("node2parent", {}).items()}
    records = []
    for node in raw_nodes:
        node_id = str(node["id"])
        parent_id = parent_map.get(node_id)
        if parent_id is None and node.get("parent_id") is not None:
            parent_id = str(node["parent_id"])
        records.append(_make_record(node, parent_id, nodes))
    return records


def _document(records: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "summary": {"iterations": len(records), "status_counts": counts},
        "iterations": records,
    }


def write_stage_sidecar(journal_path: Path) -> Path:
    journal_path = Path(journal_path)
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    document = _document(
        build_records(payload), source_journal=journal_path.name,
        format="unified_diff", context_lines=3,
    )
    output = journal_path.with_name("code_diffs.json")
    _atomic_write(output, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return output


def _collect_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    nodes: dict[str, dict[str, Any]] = {}
    parents: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    journal_paths = sorted(Path(run_dir).rglob("journal.json"))
    for journal_path in journal_paths:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        relative = str(journal_path.relative_to(run_dir))
        parent_map = {str(k): str(v) for k, v in payload.get("node2parent", {}).items()}
        for node in payload.get("nodes", []):
            node_id = str(node["id"])
            if node_id in nodes and nodes[node_id].get("code") != node.get("code"):
                raise ValueError(f"conflicting code snapshots for node {node_id}")
            nodes.setdefault(node_id, node)
            sources.setdefault(node_id, []).append(relative)
            candidate_parent = parent_map.get(node_id)
            if candidate_parent is None and node.get("parent_id") is not None:
                candidate_parent = str(node["parent_id"])
            if candidate_parent:
                if node_id in parents and parents[node_id] != candidate_parent:
                    raise ValueError(f"conflicting parents for node {node_id}")
                parents[node_id] = candidate_parent
    ordered = sorted(nodes.values(), key=lambda node: (node.get("ctime") or 0, str(node["id"])))
    records = []
    for node in ordered:
        record = _make_record(node, parents.get(str(node["id"])), nodes)
        record["source_journals"] = sources[str(node["id"])]
        records.append(record)
    return records, [str(path.relative_to(run_dir)) for path in journal_paths]


def _one_line(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def render_markdown(document: dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Explicit Iteration Code Diffs", "",
        f"Run: `{document['run']}`  ",
        f"Generated: `{document['generated_at_utc']}`  ",
        f"Unique iterations: **{summary['iterations']}**  ",
        "Comparison source: serialized `node2parent` relationships; root code is compared with `/dev/null`.", "",
        "This is an additive sidecar. Existing journals, metrics, checkpoints, and configurations are unchanged.", "",
    ]
    for index, record in enumerate(document["iterations"], 1):
        lines.extend([
            f"## {index}. `{record['node_id']}`", "",
            f"- Parent: `{record['parent_id'] or 'ROOT'}`",
            f"- Status: `{record['status']}`",
            f"- Added/deleted lines: `{record['added_lines']}` / `{record['deleted_lines']}`",
            f"- Node SHA256: `{record['node_code_sha256']}`",
            f"- Hypothesis: {_one_line(record.get('hypothesis') or '') or 'Not recorded'}",
            f"- Note: {record['note']}", "",
        ])
        if record["code_diff"]:
            lines.extend(["````diff", record["code_diff"].rstrip(), "````", ""])
        else:
            lines.extend(["_No textual code diff._", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_run_logs(run_dir: Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    records, journals = _collect_run(run_dir)
    document = _document(
        records, run=run_dir.name, source_journals=journals,
        format="unified_diff", context_lines=3,
    )
    json_path = run_dir / "iteration_code_diffs.json"
    markdown_path = run_dir / "ITERATION_CODE_DIFFS.md"
    _atomic_write(json_path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(markdown_path, render_markdown(document))
    return json_path, markdown_path


def find_experiment_root(path: Path) -> Path | None:
    resolved = Path(path).resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.parent.name == "experiments":
            return candidate
    return None
