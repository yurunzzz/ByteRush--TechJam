import tempfile
import unittest
from pathlib import Path

from ai_scientist.treesearch.factor_context import build_factor_context


class FactorContextTests(unittest.TestCase):
    def test_context_lists_raw_fields_and_leakage_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "KuaiRand-Pure" / "data"
            data_dir.mkdir(parents=True)
            headers = {
                "log_standard_4_08_to_4_21_pure.csv": (
                    "user_id,video_id,date,time_ms,long_view,is_click,play_time_ms\n"
                ),
                "user_features_pure.csv": "user_id,user_active_degree\n",
                "video_features_basic_pure.csv": "video_id,author_id,tag\n",
                "video_features_statistic_pure.csv": "video_id,show_cnt\n",
            }
            for name, header in headers.items():
                (data_dir / name).write_text(header)

            context = build_factor_context(
                Path(tmp),
                incumbent_code=(
                    "FIELDS = ['user_id', 'video_id']\n"
                    "ABLATION_COMPONENTS = {'history': True}\n"
                ),
                round_number=1,
            )

        self.assertIn("user_active_degree", context)
        self.assertIn("Current registered components: history", context)
        self.assertIn("earlier time_ms", context)
        self.assertIn("Never read validation/test outcomes", context)
        self.assertIn("outcome-derived history at the end of train", context)
        self.assertIn("forbidden inputs", context)
        self.assertIn("video_features_statistic_pure.csv as unsafe", context)


if __name__ == "__main__":
    unittest.main()
