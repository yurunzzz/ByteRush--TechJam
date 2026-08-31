from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .showcase_loader import ShowcaseBuildError, build_showcase_payload, discover_complete_candidates
except ImportError:
    from showcase_loader import ShowcaseBuildError, build_showcase_payload, discover_complete_candidates


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ShowcaseLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = "winner-node"
        snapshot = {
            "schema_version": 1,
            "run_id": "complete-run",
            "generated_at": 20,
            "nodes": [
                {"id": "fm-root", "parent_id": None, "stage": "baseline", "stage_number": 1, "label": "FM", "status": "succeeded", "primary": .60, "gauc": .66, "ndcg": .54},
                {"id": "wide-root", "parent_id": "fm-root", "stage": "baseline", "stage_number": 1, "label": "Wide Deep", "status": "succeeded", "primary": .605, "gauc": .67, "ndcg": .54},
                {"id": self.source, "parent_id": "wide-root", "stage": "tuning", "stage_number": 2, "label": "Tuned", "status": "succeeded", "primary": .606, "gauc": .671, "ndcg": .541},
            ],
            "stages": [
                {"key": "baseline", "summary": {"total": 2, "succeeded": 2, "failed": 0, "best_node_id": "wide-root"}},
                {"key": "tuning", "summary": {"total": 1, "succeeded": 1, "failed": 0, "best_node_id": self.source}},
                {"key": "creative", "summary": {"total": 2, "succeeded": 0, "failed": 2, "best_node_id": None}},
            ],
        }
        _write_json(self.root / "experiments" / "complete-run" / "dashboard_snapshot.json", snapshot)
        artifact = self.root / "artifacts" / "comparison_current" / "final_model"
        artifact.mkdir(parents=True)
        manifest = {
            "source_node_id": self.source,
            "source_stage": "4_final_incumbent_confirmation",
            "required_successful_seeds": 3,
            "successful_seed_count": 3,
            "test_metrics_used_for_selection": False,
            "selection_metric": "validation primary mean across seeds",
            "validation": {
                "GAUC": {"mean": .672, "std": .001, "values": [.671, .672, .673]},
                "nDCG@5": {"mean": .542, "std": .001, "values": [.541, .542, .543]},
                "primary": {"mean": .607, "std": .001, "values": [.606, .607, .608]},
            },
        }
        _write_json(artifact / "manifest.json", manifest)
        _write_json(artifact / "submission.csv.metadata.json", {"source_node_id": self.source, "test_metrics_computed": False, "rows": 2})
        _write_json(artifact / "training_history.json", [{"epoch": 1, "primary": .607}])
        (artifact / "model.py").write_text("# model", encoding="utf-8")
        (artifact / "checkpoint.npz").write_bytes(b"checkpoint")
        (artifact / "submission.csv").write_text("id,prediction\n1,.5\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_build_joins_final_artifact_to_completed_run(self) -> None:
        payload = build_showcase_payload(self.root)
        self.assertEqual(payload["selection"]["run_id"], "complete-run")
        self.assertEqual(payload["winner"]["baseline"]["node_id"], "fm-root")
        self.assertAlmostEqual(payload["winner"]["delta"]["primary"], .007)
        self.assertEqual(payload["search"]["champion_path"], ["fm-root", "wide-root", self.source, "frozen-final-verification"])
        self.assertTrue(payload["integrity"]["validation_only_selection"])

    def test_interrupted_snapshot_is_rejected(self) -> None:
        snapshot_path = self.root / "experiments" / "complete-run" / "dashboard_snapshot.json"
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["nodes"].append({"id": "still-running", "status": "running"})
        _write_json(snapshot_path, snapshot)
        self.assertEqual(discover_complete_candidates(self.root), [])
        with self.assertRaises(ShowcaseBuildError):
            build_showcase_payload(self.root)

    def test_missing_submission_is_rejected(self) -> None:
        (self.root / "artifacts" / "comparison_current" / "final_model" / "submission.csv").unlink()
        self.assertEqual(discover_complete_candidates(self.root), [])


if __name__ == "__main__":
    unittest.main()
