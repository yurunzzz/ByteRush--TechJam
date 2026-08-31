import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.treesearch.utils.code_diff import (
    build_records,
    find_experiment_root,
    write_run_logs,
    write_stage_sidecar,
)


class CodeDiffLoggingTests(unittest.TestCase):
    def payload(self):
        return {
            "nodes": [
                {"id": "root", "code": "x = 1\n", "plan": "initial", "ctime": 1},
                {"id": "child", "code": "x = 2\ny = 3\n", "plan": "change", "ctime": 2,
                 "parent_id": None},
                {"id": "same", "code": "x = 2\ny = 3\n", "plan": "config", "ctime": 3,
                 "parent_id": None},
            ],
            "node2parent": {"child": "root", "same": "child"},
        }

    def test_root_changed_and_no_change(self):
        records = {record["node_id"]: record for record in build_records(self.payload())}
        self.assertEqual(records["root"]["status"], "initial_code")
        self.assertIn("--- /dev/null", records["root"]["code_diff"])
        self.assertEqual(records["child"]["status"], "changed")
        self.assertIn("-x = 1", records["child"]["code_diff"])
        self.assertIn("+x = 2", records["child"]["code_diff"])
        self.assertEqual(records["same"]["status"], "no_code_change")
        self.assertEqual(records["same"]["code_diff"], "")

    def test_missing_parent_is_explicit(self):
        payload = {"nodes": [{"id": "orphan", "code": "pass\n"}],
                   "node2parent": {"orphan": "missing"}}
        record = build_records(payload)[0]
        self.assertEqual(record["status"], "missing_parent")
        self.assertEqual(record["code_diff"], "")

    def test_stage_sidecar_does_not_modify_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / "journal.json"
            original = json.dumps(self.payload(), separators=(",", ":"))
            journal.write_text(original, encoding="utf-8")
            output = write_stage_sidecar(journal)
            self.assertEqual(journal.read_text(encoding="utf-8"), original)
            self.assertEqual(len(json.loads(output.read_text())["iterations"]), 3)

    def test_run_log_deduplicates_nodes_across_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "experiments" / "demo_run"
            stage_one = run / "logs" / "stage_1"
            stage_two = run / "logs" / "stage_2"
            stage_one.mkdir(parents=True)
            stage_two.mkdir(parents=True)
            first = self.payload()
            second = {"nodes": [first["nodes"][0], first["nodes"][1]],
                      "node2parent": {"child": "root"}}
            (stage_one / "journal.json").write_text(json.dumps(first), encoding="utf-8")
            (stage_two / "journal.json").write_text(json.dumps(second), encoding="utf-8")
            json_path, markdown_path = write_run_logs(run)
            document = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(document["summary"]["iterations"], 3)
            root = next(item for item in document["iterations"] if item["node_id"] == "root")
            self.assertEqual(len(root["source_journals"]), 2)
            self.assertIn("Explicit Iteration Code Diffs", markdown_path.read_text())
            self.assertEqual(find_experiment_root(stage_two), run.resolve())

    def test_conflicting_duplicate_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "experiments" / "demo_run"
            one = run / "one"
            two = run / "two"
            one.mkdir(parents=True)
            two.mkdir(parents=True)
            (one / "journal.json").write_text(
                json.dumps({"nodes": [{"id": "n", "code": "a\n"}]}), encoding="utf-8")
            (two / "journal.json").write_text(
                json.dumps({"nodes": [{"id": "n", "code": "b\n"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicting code snapshots"):
                write_run_logs(run)


if __name__ == "__main__":
    unittest.main()
