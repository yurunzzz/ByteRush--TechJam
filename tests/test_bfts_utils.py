import tempfile
import unittest
from pathlib import Path

import yaml

from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file


class BftsConfigTests(unittest.TestCase):
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
