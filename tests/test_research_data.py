import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_module = _load("trusted_data", ROOT / "kuairand-starter-kit" / "data.py")
research_data = _load(
    "trusted_research_data",
    ROOT / "kuairand-starter-kit" / "research_data.py",
)


class ResearchDataTests(unittest.TestCase):
    def _splits(self):
        return {
            "train": [
                (20220408, "u1", "v1", "a1", "1", 1000.0, 1),
                (20220409, "u1", "v2", "a2", "1", 2000.0, 0),
                (20220410, "u2", "v1", "a1", "0", 1000.0, 1),
                (20220411, "u2", "v2", "a2", "0", 2000.0, 0),
            ],
            "valid": [
                (20220422, "u1", "v3", "a3", "1", 1500.0, 0),
            ],
            "test": [
                (20220429, "u1", "v4", "a4", "1", 1500.0, 1),
            ],
        }

    def test_schema_v2_legacy_adapter_is_exact(self):
        splits = self._splits()
        official, dimension, state = data_module.encode(
            splits, return_state=True
        )
        schema, adapted_dimension, adapted_state = research_data.build_schema_v2(
            splits, data_module=data_module
        )
        adapted = research_data.LegacyFMAdapter.to_legacy(schema)
        research_data.assert_legacy_equivalence(official, adapted)
        self.assertEqual(dimension, adapted_dimension)
        self.assertEqual(state, adapted_state)
        for split_name in official:
            self.assertEqual(
                official[split_name][0].tobytes(),
                adapted[split_name][0].tobytes(),
            )
            self.assertEqual(
                official[split_name][1].tobytes(),
                adapted[split_name][1].tobytes(),
            )

    def test_history_is_past_only_and_frozen_after_train(self):
        splits = self._splits()
        schema, _, _ = research_data.build_schema_v2(
            splits, data_module=data_module
        )
        research_data.attach_causal_history(schema, splits, max_length=3)
        train_mask = schema["splits"]["train"]["inputs"]["history_mask"]
        valid_items = schema["splits"]["valid"]["inputs"]["history_item_ids"]
        test_items = schema["splits"]["test"]["inputs"]["history_item_ids"]
        self.assertFalse(train_mask[0].any())
        self.assertTrue(train_mask[1].any())
        np.testing.assert_array_equal(valid_items, test_items)

    def test_pair_builder_never_crosses_users(self):
        users = ["u1", "u1", "u2", "u2"]
        labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
        positive, negative = research_data.same_user_pair_indices(users, labels)
        self.assertGreater(len(positive), 0)
        for pos, neg in zip(positive, negative):
            self.assertEqual(users[pos], users[neg])
            self.assertGreater(labels[pos], labels[neg])


if __name__ == "__main__":
    unittest.main()
