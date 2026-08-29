import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.candidate_contract import config_assignment
from ai_scientist.treesearch.backend import compile_prompt_to_md
from ai_scientist.treesearch.parallel_agent import (
    GPUManager,
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
    def test_gpu_manager_allows_bounded_shared_gpu_slots(self):
        manager = GPUManager(num_gpus=1, max_workers_per_gpu=3)

        self.assertEqual(manager.acquire_gpu("worker-1"), 0)
        self.assertEqual(manager.acquire_gpu("worker-2"), 0)
        self.assertEqual(manager.acquire_gpu("worker-3"), 0)
        with self.assertRaises(RuntimeError):
            manager.acquire_gpu("worker-4")

        manager.release_gpu("worker-2")
        self.assertEqual(manager.acquire_gpu("worker-4"), 0)
        self.assertEqual(manager.gpu_loads[0], 3)

    def test_five_seed_evaluation_runs_in_gpu_sized_batches(self):
        class ImmediateFuture:
            def __init__(self, value):
                self.value = value

            def result(self, timeout=None):
                return self.value

        class ImmediateExecutor:
            def submit(self, fn, *args):
                return ImmediateFuture(fn(*args))

        parent = _good_node("MODEL = 'candidate'\n")
        journal = Journal(nodes=[parent])
        manager = GPUManager(num_gpus=1, max_workers_per_gpu=3)
        concurrent_assignments = []
        acquire_gpu = manager.acquire_gpu

        def tracked_acquire(process_id):
            gpu_id = acquire_gpu(process_id)
            concurrent_assignments.append(len(manager.gpu_assignments))
            return gpu_id

        manager.acquire_gpu = tracked_acquire
        agent = object.__new__(ParallelAgent)
        agent.cfg = SimpleNamespace()
        agent.stage_name = "final_confirmation"
        agent.num_workers = 3
        agent.gpu_manager = manager
        agent.executor = ImmediateExecutor()
        agent.timeout = 1
        agent.task_desc = "KuaiRand"
        agent.evaluation_metrics = None
        agent.journal = journal

        def process_with_parent(node_data, *args):
            result = _good_node(node_data["code"], score=0.61)
            result.parent = parent
            return result.to_dict()

        agent._process_node_wrapper = process_with_parent

        seeds = agent._run_multi_seed_evaluation(parent, num_seeds=5)

        self.assertEqual(len(seeds), 5)
        self.assertEqual(max(concurrent_assignments), 3)
        self.assertEqual(manager.gpu_loads, {0: 0})

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

    def test_kuairand_tuning_node_restores_runtime_seed(self):
        parent_code = (
            "import os\n"
            "CONFIG = {'seed': int(os.environ.get('AI_SCIENTIST_SEED', '0')), "
            "'learning_rate': 0.001, 'max_epochs': 40}\n"
        )
        proposed = parent_code.replace(
            "int(os.environ.get('AI_SCIENTIST_SEED', '0'))",
            "0",
        ).replace("'max_epochs': 40", "'max_epochs': 12")
        parent = _good_node(parent_code)
        agent = object.__new__(MinimalAgent)
        agent.task_desc = "KuaiRand-Pure validation primary and nDCG@5"
        agent.plan_and_code_query = lambda prompt: ("plan", proposed)

        result = agent._generate_hyperparam_tuning_node(
            parent,
            HyperparamTuningIdea("epochs", "tune CONFIG"),
        )
        config = config_assignment(result.code)

        self.assertIn("__ast_expression__", config["seed"])
        self.assertEqual(config["max_epochs"], 12)

    def test_scored_config_history_compiles_for_next_tuning_idea(self):
        base = _good_node("CONFIG = {'learning_rate': 0.001}\n")
        agent = object.__new__(ParallelAgent)
        agent.stage_name = "2_baseline_tuning_1_closed_loop"
        agent.tuning_base_node = base
        agent._hyperparam_tuning_state = {
            "tried_hyperparams": {"config_baseline_001"},
            "tried_configurations": [
                {
                    "config": {
                        "seed": {"__ast_expression__": "runtime seed"},
                        "learning_rate": 0.001,
                    },
                    "validation_primary": 0.6016,
                }
            ],
        }
        agent.cfg = SimpleNamespace(
            agent=SimpleNamespace(
                code=SimpleNamespace(model="test-model", temp=0.0)
            )
        )

        def compile_then_answer(system_message, **kwargs):
            compiled = compile_prompt_to_md(system_message)
            self.assertIn("validation_primary", compiled)
            return (
                "HYPERPARAM NAME: config_baseline_002\n"
                "DESCRIPTION: change learning_rate to 0.0005"
            )

        with patch(
            "ai_scientist.treesearch.parallel_agent.query",
            side_effect=compile_then_answer,
        ):
            idea = agent._generate_hyperparam_tuning_idea()

        self.assertEqual(idea.name, "config_baseline_002")

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
        agent.max_search_workers = 1
        self.assertEqual(
            agent._select_parallel_nodes(max_nodes=2),
            [candidate],
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
