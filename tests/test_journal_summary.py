import unittest
from unittest.mock import patch

from ai_scientist.treesearch.backend.utils import compile_prompt_to_md
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.utils.metric import MetricValue


class JournalSummaryTests(unittest.TestCase):
    def test_large_journal_is_bounded_and_keeps_best_recent_and_failed_nodes(self):
        journal = Journal()
        for index in range(30):
            metric = 0.99 if index == 2 else 0.50 + index / 1000
            journal.append(
                Node(
                    id=f"good-{index}",
                    plan=f"PLAN-{index} " + "p" * 6000,
                    analysis=f"ANALYSIS-{index} " + "a" * 9000,
                    code=f"CODE-{index} " + "c" * 9000,
                    metric=MetricValue(
                        value=metric,
                        maximize=True,
                        name="validation primary",
                    ),
                    is_buggy=False,
                    is_buggy_plots=False,
                )
            )

        for index in range(10):
            journal.append(
                Node(
                    id=f"failed-{index}",
                    plan=f"FAILED-PLAN-{index} " + "p" * 6000,
                    analysis=f"FAILED-ANALYSIS-{index} " + "a" * 9000,
                    code=f"FAILED-CODE-{index} " + "c" * 9000,
                    metric=None,
                    exc_type="RuntimeError",
                    is_buggy=True,
                    is_buggy_plots=False,
                )
            )

        captured = {}

        def fake_query(**kwargs):
            captured.update(kwargs)
            return "bounded summary"

        with patch("ai_scientist.treesearch.journal.query", side_effect=fake_query):
            result = journal.generate_summary(include_code=True, model="test-model")

        compiled_prompt = compile_prompt_to_md(captured["system_message"])
        self.assertEqual(result, "bounded summary")
        self.assertLess(len(compiled_prompt), 54000)
        self.assertIn("good-2", compiled_prompt)
        self.assertIn("good-29", compiled_prompt)
        self.assertIn("failed-9", compiled_prompt)
        self.assertNotIn("good-5 (step 5)", compiled_prompt)


if __name__ == "__main__":
    unittest.main()
