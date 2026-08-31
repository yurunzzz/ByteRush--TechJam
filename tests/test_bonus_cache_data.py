import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_data_module():
    path = ROOT / "kuairand-starter-kit" / "data.py"
    spec = importlib.util.spec_from_file_location("bonus_cache_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BonusCacheDataTests(unittest.TestCase):
    def test_cache_mode_is_deterministic_and_never_exposes_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.npz"
            arrays = {"num_users": np.asarray(8), "num_items": np.asarray(20)}
            for name, size in (("train", 12), ("valid", 10), ("test", 9)):
                arrays[f"{name}_users"] = np.arange(size) % 8
                arrays[f"{name}_items"] = np.arange(size) % 20
                arrays[f"{name}_labels"] = np.arange(size) % 2
                arrays[f"{name}_history_end"] = np.arange(size)
            np.savez(cache, **arrays)
            environment = {
                "KUAIRAND_CACHE_PATH": str(cache),
                "KUAIRAND_DATASET_NAME": "KuaiRand-1K",
                "KUAIRAND_MAX_TRAIN_ROWS": "5",
                "KUAIRAND_MAX_VALID_ROWS": "4",
                "KUAIRAND_SAMPLE_SEED": "123",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                module = _load_data_module()
                first = module.load("unused")
                second = module.load("unused")
                encoded, dimension, state = module.encode(
                    first, return_state=True
                )

        self.assertEqual(set(first), {"train", "valid"})
        self.assertEqual(len(first["train"].labels), 5)
        self.assertEqual(len(first["valid"].labels), 4)
        np.testing.assert_array_equal(first["train"].users, second["train"].users)
        self.assertEqual(encoded["train"][0].shape, (5, 2))
        self.assertGreater(dimension, int(encoded["train"][0].max()))
        self.assertEqual(state["fields"], ["user_id", "video_id"])


if __name__ == "__main__":
    unittest.main()
