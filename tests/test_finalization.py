import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_scientist.treesearch.finalize import freeze_stage4_winner
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.utils.metric import MetricValue


def _metric(score):
    return MetricValue(
        value=score,
        maximize=True,
        name="validation primary",
    )


def _write_validation_metrics(path, primary):
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "KuaiRand-Pure": {
            "metrics": {
                "validation GAUC": [primary],
                "validation nDCG@5": [primary],
                "validation primary": [primary],
            }
        }
    }
    np.save(path / "experiment_data.npy", payload)


def _seed(parent, result_dir, score):
    return Node(
        code=parent.code,
        metric=_metric(score),
        parent=parent,
        exp_results_dir=str(result_dir),
        is_buggy=False,
        is_buggy_plots=False,
        is_seed_node=True,
    )


class FinalizationTests(unittest.TestCase):
    def test_checkpoint_follows_the_matching_best_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            winner = Node(
                code="MODEL = 'winner'\n",
                metric=_metric(0.65),
                is_buggy=False,
                is_buggy_plots=False,
            )
            missing_metrics = _seed(winner, root / "seed_missing", 0.50)
            lower = _seed(winner, root / "seed_lower", 0.60)
            higher = _seed(winner, root / "seed_higher", 0.70)
            _write_validation_metrics(Path(lower.exp_results_dir), 0.60)
            _write_validation_metrics(Path(higher.exp_results_dir), 0.70)
            (Path(lower.exp_results_dir) / "candidate_checkpoint.npz").write_bytes(
                b"lower-checkpoint"
            )
            (Path(higher.exp_results_dir) / "candidate_checkpoint.npz").write_bytes(
                b"higher-checkpoint"
            )
            journal = Journal(nodes=[winner, missing_metrics, lower, higher])

            manifest = freeze_stage4_winner(
                journal,
                root / "final_model",
                source_stage="4_ablation",
                required_seeds=2,
            )

            checkpoint = root / "final_model" / "checkpoint.npz"
            self.assertEqual(checkpoint.read_bytes(), b"higher-checkpoint")
            self.assertEqual(manifest["checkpoint_source_node_id"], higher.id)
            self.assertEqual(
                manifest["checkpoint_sha256"],
                hashlib.sha256(b"higher-checkpoint").hexdigest(),
            )
            self.assertAlmostEqual(manifest["validation"]["primary"]["mean"], 0.65)
            self.assertEqual(
                json.loads((root / "final_model" / "manifest.json").read_text()),
                manifest,
            )

    def test_missing_best_checkpoint_stops_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            winner = Node(
                code="MODEL = 'winner'\n",
                metric=_metric(0.65),
                is_buggy=False,
                is_buggy_plots=False,
            )
            seeds = []
            for index, primary in enumerate((0.60, 0.61, 0.70)):
                seed = _seed(winner, root / f"seed_{index}", primary)
                _write_validation_metrics(Path(seed.exp_results_dir), primary)
                if primary < 0.70:
                    (
                        Path(seed.exp_results_dir) / "candidate_checkpoint.npz"
                    ).write_bytes(f"checkpoint-{index}".encode())
                seeds.append(seed)
            journal = Journal(nodes=[winner, *seeds])

            with self.assertRaisesRegex(RuntimeError, "no complete checkpoint"):
                freeze_stage4_winner(
                    journal,
                    root / "final_model",
                    source_stage="4_ablation",
                    required_seeds=3,
                )
            self.assertFalse((root / "final_model").exists())


if __name__ == "__main__":
    unittest.main()
