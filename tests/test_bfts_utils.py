import tempfile
import unittest
from pathlib import Path

import yaml

from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file
from ai_scientist.treesearch.research_loop import (
    estimate_max_experiment_runs,
    research_loop_config_from_mapping,
)


class BftsConfigTests(unittest.TestCase):
    def test_kuairand_budget_stays_below_competition_limits(self):
        config_path = Path(__file__).resolve().parents[1] / "bfts_config_kuairand.yaml"
        config = yaml.safe_load(config_path.read_text())
        loop = research_loop_config_from_mapping(
            config["agent"]["research_loop"]
        )
        budget = estimate_max_experiment_runs(
            loop,
            stage4_max_components=int(config["agent"]["ablation"]["max_components"]),
        )

        self.assertLess(loop.max_wall_clock_seconds, 6 * 60 * 60)
        self.assertLess(budget["maximum_search_iterations"], 50)
        self.assertLess(budget["maximum_total"], 50)
        self.assertEqual(loop.max_research_rounds, 1)

    def test_final_model_directory_is_scoped_to_timestamped_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            idea_dir = root / "run"
            idea_dir.mkdir()
            idea = idea_dir / "idea.md"
            idea.write_text("idea")
            source = root / "config.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "data_dir": str(root / "input"),
                        "agent": {"final_model_dir": "artifacts/final_model"},
                    }
                )
            )

            output = Path(
                edit_bfts_config_file(str(source), str(idea_dir), str(idea))
            )
            config = yaml.safe_load(output.read_text())

        self.assertEqual(
            config["agent"]["final_model_dir"],
            str((idea_dir / "artifacts" / "final_model").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
