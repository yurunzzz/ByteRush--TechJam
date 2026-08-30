import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from ai_scientist.treesearch.agent_manager import AgentManager, Stage
from ai_scientist.treesearch.closed_loop import (
    ClosedLoopRunner,
    _node_validation_metrics,
)
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import (
    _extract_enabled_ablation_components,
)
from ai_scientist.treesearch.utils.metric import MetricValue


def _node(code, score, *, parent=None, ablation_name=None, is_seed=False):
    return Node(
        code=code,
        metric=MetricValue(
            value=score,
            maximize=True,
            name="validation primary",
        ),
        parent=parent,
        is_buggy=False,
        is_buggy_plots=False,
        is_seed_node=is_seed,
        ablation_name=ablation_name,
    )


class FakeManager:
    def __init__(self, data_dir, output_dir):
        self.cfg = OmegaConf.create(
            {
                "data_dir": str(data_dir),
                "agent": {
                    "search": {
                        "num_drafts": 1,
                        "debug_prob": 0.0,
                        "max_debug_depth": 0,
                    },
                    "research_loop": {
                        "enabled": True,
                        "max_research_rounds": 2,
                        "patience": 1,
                        "stage1_validation_iterations": 1,
                        "baseline_tuning_iterations": 2,
                        "candidate_branches": 2,
                        "candidate_parallel_workers": 2,
                        "stage3_generation_attempts": 2,
                        "candidate_tuning_iterations": 2,
                        "candidate_refinement_top_k": 2,
                        "candidate_refinement_iterations": 2,
                        "candidate_finalist_top_k": 2,
                        "finalist_top_k": 2,
                        "finalist_num_seeds": 3,
                        "final_confirmation_num_seeds": 5,
                        "ablation_candidate_top_k": 2,
                        "ablation_synergy_pairs": 1,
                        "min_primary_gain": 0.002,
                        "required_seed_wins": 2,
                    },
                    "ablation": {"max_components": 1},
                    "final_model_dir": str(output_dir),
                },
            }
        )
        initial = Stage(
            name="1_initial_implementation_1_preliminary",
            description="preliminary",
            goals="validate FM",
            max_iterations=1,
            num_drafts=1,
            stage_number=1,
        )
        self.current_stage_number = 1
        self.current_stage = initial
        self.stages = [initial]
        self.journals = {initial.name: Journal()}
        self.main_stage_goals = {
            1: "validate FM",
            2: "tune FM",
            3: "research",
            4: "ablate",
        }
        self.checkpoint_calls = 0

    def _curate_task_desc(self, stage):
        return "KuaiRand-Pure validation metric nDCG@5"

    def parse_stage_names(self, stage_name):
        numbers = [int(value) for value in re.findall(r"\d+", stage_name)]
        parts = re.split(r"\d+", stage_name)
        return (
            numbers[0],
            parts[1].strip("_"),
            numbers[1],
            parts[-1].strip("_"),
        )

    def _save_checkpoint(self):
        self.checkpoint_calls += 1


class FakeAgent:
    created_stage_names = []
    created_task_descs = {}

    def __init__(self, **kwargs):
        self.journal = kwargs["journal"]
        self.stage_name = kwargs["stage_name"]
        self.tuning_base = kwargs["tuning_base_node"]
        self.research_base = kwargs["research_base_node"]
        self.stage3_base = kwargs["best_stage3_node"]
        self.candidate_contexts = list(kwargs.get("candidate_contexts") or [])
        self.created_stage_names.append(self.stage_name)
        self.created_task_descs[self.stage_name] = kwargs["task_desc"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def step(self, exec_callback, max_nodes=None):
        count = int(max_nodes or 1)
        if self.stage_name.startswith("1_"):
            self.journal.append(
                _node(
                    "ABLATION_COMPONENTS = {}\nBASELINE = 'FM'\n",
                    0.600,
                )
            )
            return

        if self.stage_name.startswith("2_"):
            for index in range(count):
                score = 0.601 + index * 0.001
                self.journal.append(
                    _node(
                        self.tuning_base.code + f"# fm tuning {index}\n",
                        score,
                        parent=self.tuning_base,
                    )
                )
            return

        if self.stage_name.startswith("3_creative"):
            base_score = self.research_base.metric.get_mean_value()
            second_round = "round_two" in self.stage_name
            for index in range(count):
                if second_round:
                    score = base_score - 0.001 - index * 0.001
                else:
                    score = base_score + 0.004 - index * 0.001
                context = self.candidate_contexts[index]
                role = re.search(r"Candidate role \d+/\d+: ([^\n]+)", context).group(1)
                group = re.search(r"Major research group: ([^\n]+)", context).group(1)
                category = re.search(r"Research category: ([^\n]+)", context).group(1)
                factor_component = f"factor_{role}"
                model_component = f"model_{role}"
                output_field = f"feature_{role}"
                code = (
                    "CONFIG = {'learning_rate': 0.001, 'epochs': 6}\n"
                    "RESEARCH_MANIFEST = {"
                    f"'candidate_id': {role!r}, 'role': {role!r}, "
                    f"'group': {group!r}, 'category': {category!r}, "
                    "'hypothesis': 'role-specific mechanism improves ranking', "
                    "'mechanism': 'factor consumed by matching model path', "
                    f"'mechanism_ids': [{role!r}], "
                    "'modified_symbols': ['build_features', 'create_model'], "
                    "'expected_metric': ['GAUC', 'nDCG@5'], "
                    "'tunable_parameters': ['learning_rate'], "
                    f"'ablation_components': [{factor_component!r}, {model_component!r}], "
                    "'combination_compatibility': 'model consumes factor', "
                    f"'change_scope': 'candidate', 'component_dependencies': {{{model_component!r}: [{factor_component!r}]}}, "
                    "'evidence': ["
                    f"{{'source_type': 'dependency', 'reference': 'dependency:model_needs_factor', 'supports': [{factor_component!r}, {model_component!r}]}}]}}\n"
                    "FEATURE_FACTORS = [{"
                    f"'name': {output_field!r}, 'raw_fields': ['user_id', 'video_id'], "
                    f"'transform': 'causal role transform', 'output_fields': [{output_field!r}], "
                    "'state_policy': 'train_only_frozen'}]\n"
                    f"ABLATION_COMPONENTS = {{{factor_component!r}: True, {model_component!r}: True}}\n"
                    "def component_enabled(name):\n"
                    "    return ABLATION_COMPONENTS[name]\n"
                    "def build_features(splits, feature_state=None):\n"
                    f"    if component_enabled({factor_component!r}):\n"
                    f"        output_field = {output_field!r}\n"
                    "    return splits, feature_state\n"
                    "def create_model(feature_dimension, config=None):\n"
                    f"    if component_enabled({model_component!r}):\n"
                    f"        return ('candidate_{index}', feature_dimension)\n"
                    "    return ('fm', feature_dimension)\n"
                )
                self.journal.append(
                    _node(code, score, parent=self.research_base)
                )
            return

        if self.stage_name.startswith("3_candidate_tuning"):
            base_score = self.tuning_base.metric.get_mean_value()
            second_round = "round_two" in self.stage_name
            increment = 0.0 if second_round else 0.001
            for index in range(count):
                score = base_score + increment * (index + 1)
                self.journal.append(
                    _node(
                        self.tuning_base.code + f"# candidate tuning {index}\n",
                        score,
                        parent=self.tuning_base,
                    )
                )
            return

        if self.stage_name.startswith("4_"):
            components = _extract_enabled_ablation_components(
                self.stage3_base.code
            )
            for component in components[:count]:
                code = (
                    "import os\n"
                    "os.environ['AI_SCIENTIST_ABLATION_TARGET'] = "
                    f"{component!r}\n"
                    + self.stage3_base.code
                )
                self.journal.append(
                    _node(
                        code,
                        self.stage3_base.metric.get_mean_value() + 0.0005,
                        parent=self.stage3_base,
                        ablation_name=component,
                    )
                )
            return

        raise AssertionError(f"unexpected stage {self.stage_name}")

    def _run_multi_seed_evaluation(self, node, num_seeds=None):
        count = int(num_seeds)
        offsets = [
            (index - (count - 1) / 2) * 0.0001 for index in range(count)
        ]
        results = []
        for offset in offsets:
            seed = _node(
                node.code,
                node.metric.get_mean_value() + offset,
                parent=node,
                is_seed=True,
            )
            self.journal.append(seed)
            results.append(seed)
        return results



class ConcurrentFakeAgent(FakeAgent):
    lock = threading.Lock()
    active_candidate_branches = 0
    max_active_candidate_branches = 0

    def step(self, exec_callback, max_nodes=None):
        if not self.stage_name.startswith("3_candidate_tuning"):
            return super().step(exec_callback, max_nodes=max_nodes)

        with self.lock:
            type(self).active_candidate_branches += 1
            type(self).max_active_candidate_branches = max(
                type(self).max_active_candidate_branches,
                type(self).active_candidate_branches,
            )
        try:
            time.sleep(0.02)
            return super().step(exec_callback, max_nodes=max_nodes)
        finally:
            with self.lock:
                type(self).active_candidate_branches -= 1


class NoImprovementFakeAgent(FakeAgent):
    def step(self, exec_callback, max_nodes=None):
        before = len(self.journal.nodes)
        super().step(exec_callback, max_nodes=max_nodes)
        if not self.stage_name.startswith("3_creative"):
            return

        base_score = self.research_base.metric.get_mean_value()
        for index, node in enumerate(self.journal.nodes[before:]):
            node.metric = MetricValue(
                value=base_score - 0.004 - index * 0.001,
                maximize=True,
                name="validation primary",
            )


class ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        FakeAgent.created_stage_names = []
        FakeAgent.created_task_descs = {}
        ConcurrentFakeAgent.active_candidate_branches = 0
        ConcurrentFakeAgent.max_active_candidate_branches = 0

    def _data_dir(self, root):
        data_dir = root / "KuaiRand-Pure" / "data"
        data_dir.mkdir(parents=True)
        headers = {
            "log_standard_4_08_to_4_21_pure.csv": (
                "user_id,video_id,date,time_ms,long_view\n"
            ),
            "user_features_pure.csv": "user_id,user_active_degree\n",
            "video_features_basic_pure.csv": "video_id,author_id,tag\n",
            "video_features_statistic_pure.csv": "video_id,show_cnt\n",
        }
        for name, header in headers.items():
            (data_dir / name).write_text(header)
        return root

    def test_experience_metrics_are_loaded_from_nested_validation_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "candidate" / "run_0"
            result_dir.mkdir(parents=True)
            np.save(
                result_dir / "experiment_data.npy",
                {
                    "KuaiRand-Pure": {
                        "metrics": {
                            "validation GAUC": [0.71],
                            "validation nDCG@5": [0.63],
                            "validation primary": [0.67],
                        }
                    }
                },
            )
            node = _node("MODEL = 'candidate'", 0.67)
            node.exp_results_dir = str(Path(tmp) / "candidate")

            metrics = _node_validation_metrics(node)

        self.assertEqual(
            metrics,
            {"GAUC": 0.71, "nDCG@5": 0.63, "primary": 0.67},
        )

    def test_complete_loop_promotes_stage4_winner_then_reuses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "artifacts" / "final_model",
            )
            finalized = {}

            def finalizer(journal, **kwargs):
                finalized["journal"] = journal
                finalized.update(kwargs)
                return {"source_stage": kwargs["source_stage"]}

            exported = {}

            def submission_exporter(**kwargs):
                exported.update(kwargs)
                return {"path": "submission.csv", "checked": True}

            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                agent_factory=ConcurrentFakeAgent,
                finalizer=finalizer,
                submission_exporter=submission_exporter,
            )
            result = runner.run()
            memory_dir = root / "artifacts" / "research_loop"
            latest_summary = json.loads(
                (memory_dir / "round_summary.json").read_text()
            )
            archived_round_one = json.loads(
                (memory_dir / "round_summaries" / "round_001.json").read_text()
            )
            prompt_summary = (
                memory_dir / "round_summary.prompt.txt"
            ).read_text()

        self.assertEqual(result["state"].current_round, 2)
        self.assertEqual(result["state"].no_improvement_rounds, 1)
        self.assertAlmostEqual(result["incumbent"].score, 0.6105)
        self.assertIn("'factor_causal_history_interest': False", result["incumbent"].node.code)
        self.assertEqual(manager.checkpoint_calls, 2)
        self.assertTrue(
            any(
                name.startswith("4_ablation_studies_1_round_one")
                for name in FakeAgent.created_stage_names
            )
        )
        self.assertGreaterEqual(
            ConcurrentFakeAgent.max_active_candidate_branches, 2
        )
        self.assertFalse(
            any(
                name.startswith("4_ablation_studies_1_round_two")
                for name in FakeAgent.created_stage_names
            )
        )
        self.assertIs(finalized["journal"], result["incumbent"].journal)
        self.assertEqual(finalized["required_seeds"], 5)
        self.assertTrue(result["submission"]["checked"])
        self.assertEqual(latest_summary["round_number"], 2)
        self.assertEqual(archived_round_one["round_number"], 1)
        self.assertFalse(latest_summary["test_metrics_used"])
        self.assertIn("validation-only", prompt_summary)
        self.assertIn(
            "Structured validation-only candidate experience",
            FakeAgent.created_task_descs["3_creative_research_1_round_two"],
        )
        self.assertEqual(
            exported["output_dir"],
            Path(manager.cfg.agent.final_model_dir).resolve(),
        )
        self.assertEqual(
            exported["starter_kit"],
            Path(manager.cfg.data_dir).resolve(),
        )

        round_two_journal = manager.journals[
            "3_creative_research_1_round_two"
        ]
        root_node = round_two_journal.nodes[0]
        siblings = [
            node for node in round_two_journal.nodes if node.parent is root_node
        ]
        self.assertEqual(len(siblings), 2)
        self.assertIn(
            "'factor_causal_history_interest': False",
            root_node.code,
        )

    def test_agent_manager_uses_closed_loop_only_when_enabled(self):
        manager = object.__new__(AgentManager)
        manager.cfg = OmegaConf.create(
            {"agent": {"research_loop": {"enabled": True}}}
        )
        sentinel = {"closed": True}

        from unittest.mock import patch

        with patch(
            "ai_scientist.treesearch.closed_loop.ClosedLoopRunner"
        ) as runner_type:
            runner_type.return_value.run.return_value = sentinel
            result = AgentManager.run(manager, exec_callback=lambda: None)

        self.assertIs(result, sentinel)
        runner_type.assert_called_once()

    def test_baseline_gets_five_seed_confirmation_when_no_candidate_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "artifacts" / "final_model",
            )
            finalized = {}

            def finalizer(journal, **kwargs):
                finalized["journal"] = journal
                finalized.update(kwargs)
                return {"source_stage": kwargs["source_stage"]}

            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                agent_factory=NoImprovementFakeAgent,
                finalizer=finalizer,
                submission_exporter=lambda **kwargs: {
                    "path": "submission.csv",
                    "checked": True,
                },
            )
            result = runner.run()

        self.assertEqual(result["state"].current_round, 1)
        self.assertEqual(len(result["incumbent"].seed_scores), 5)
        self.assertEqual(len(result["state"].incumbent_seed_scores), 5)
        self.assertEqual(finalized["required_seeds"], 5)
        self.assertIn(
            "4_final_incumbent_confirmation_1_five_seed",
            NoImprovementFakeAgent.created_stage_names,
        )


if __name__ == "__main__":
    unittest.main()
