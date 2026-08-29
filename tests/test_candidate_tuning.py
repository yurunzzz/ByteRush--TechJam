import unittest
from types import SimpleNamespace

from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import (
    HyperparamTuningIdea,
    MinimalAgent,
    ParallelAgent,
    _is_hyperparam_tuning_stage,
)
from ai_scientist.treesearch.utils.metric import MetricValue


def _good_node(code: str, score: float = 0.6) -> Node:
    return Node(
        code=code,
        metric=MetricValue(value=score, maximize=True),
        is_buggy=False,
        is_buggy_plots=False,
    )


class CandidateTuningTests(unittest.TestCase):
    def test_stage2_remains_a_tuning_stage(self):
        self.assertTrue(_is_hyperparam_tuning_stage("2_baseline_tuning_1_first", None))

    def test_normal_stage3_is_not_misclassified(self):
        self.assertFalse(
            _is_hyperparam_tuning_stage("3_creative_research_1_first", None)
        )

    def test_kuairand_tuning_prompt_preserves_frozen_model_contract(self):
        parent = _good_node("candidate-code")
        agent = object.__new__(MinimalAgent)
        agent.task_desc = "KuaiRand-Pure validation primary and nDCG@5"
        captured = {}

        def capture(prompt):
            captured.update(prompt)
            return "plan", parent.code

        agent.plan_and_code_query = capture
        result = agent._generate_hyperparam_tuning_node(
            parent,
            HyperparamTuningIdea("learning rate", "tune CONFIG"),
        )

        guideline = "\n".join(
            captured["Instructions"]["Implementation guideline"]
        )
        self.assertIn("Change only literal values inside CONFIG", guideline)
        self.assertIn("build_features(splits, feature_state=None)", guideline)
        self.assertIn("AI_SCIENTIST_INFERENCE_ONLY=1", guideline)
        self.assertEqual(result.code, parent.code)

    def test_explicit_candidate_becomes_the_tuning_parent(self):
        candidate = _good_node("candidate-code")
        agent = object.__new__(ParallelAgent)
        agent.num_workers = 1
        agent.journal = Journal(nodes=[candidate])
        agent.cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(num_drafts=0, debug_prob=0.0, max_debug_depth=0)
            )
        )
        agent.stage_name = "candidate_tuning"
        agent.best_stage3_node = None
        agent.best_stage1_node = None
        agent.tuning_base_node = candidate
        agent.research_base_node = None
        agent.is_hyperparam_tuning = True

        self.assertEqual(agent._select_parallel_nodes(), [candidate])

    def test_parallel_selection_respects_remaining_budget(self):
        candidate = _good_node("candidate-code")
        agent = object.__new__(ParallelAgent)
        agent.num_workers = 3
        agent.journal = Journal(nodes=[candidate])
        agent.cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(num_drafts=0, debug_prob=0.0, max_debug_depth=0)
            )
        )
        agent.stage_name = "candidate_tuning"
        agent.best_stage3_node = None
        agent.best_stage1_node = None
        agent.tuning_base_node = candidate
        agent.research_base_node = None
        agent.is_hyperparam_tuning = True

        self.assertEqual(
            agent._select_parallel_nodes(max_nodes=2), [candidate, candidate]
        )

    def test_research_branches_are_siblings_from_the_incumbent(self):
        incumbent = _good_node("incumbent-code")
        previous_candidate = _good_node("previous-candidate")
        previous_candidate.parent = incumbent
        incumbent.children.add(previous_candidate)
        agent = object.__new__(ParallelAgent)
        agent.num_workers = 1
        agent.journal = Journal(nodes=[incumbent, previous_candidate])
        agent.cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(num_drafts=0, debug_prob=0.0, max_debug_depth=0)
            )
        )
        agent.stage_name = "3_creative_research_1_round"
        agent.best_stage3_node = None
        agent.tuning_base_node = None
        agent.is_hyperparam_tuning = False
        agent.research_base_node = incumbent

        self.assertEqual(agent._select_parallel_nodes(), [incumbent])

    def _tuning_agent(self, base: Node) -> ParallelAgent:
        agent = object.__new__(ParallelAgent)
        agent.is_hyperparam_tuning = True
        agent.tuning_base_node = base
        return agent

    def test_better_result_becomes_progressive_tuning_base(self):
        base = _good_node("base", score=0.60)
        better = _good_node("better", score=0.61)
        agent = self._tuning_agent(base)

        self.assertTrue(agent._maybe_promote_tuning_base(better))
        self.assertIs(agent.tuning_base_node, better)

    def test_worse_result_keeps_current_tuning_base(self):
        base = _good_node("base", score=0.60)
        worse = _good_node("worse", score=0.59)
        agent = self._tuning_agent(base)

        self.assertFalse(agent._maybe_promote_tuning_base(worse))
        self.assertIs(agent.tuning_base_node, base)

    def test_buggy_or_nonfinite_result_keeps_current_tuning_base(self):
        base = _good_node("base", score=0.60)
        buggy = _good_node("buggy", score=0.70)
        buggy.is_buggy = True
        nonfinite = _good_node("nonfinite", score=float("nan"))
        agent = self._tuning_agent(base)

        self.assertFalse(agent._maybe_promote_tuning_base(buggy))
        self.assertIs(agent.tuning_base_node, base)
        self.assertFalse(agent._maybe_promote_tuning_base(nonfinite))
        self.assertIs(agent.tuning_base_node, base)


if __name__ == "__main__":
    unittest.main()
