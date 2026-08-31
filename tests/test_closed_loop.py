import hashlib
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
from ai_scientist.treesearch.candidate_contract import extract_assignment_contract
from ai_scientist.treesearch.closed_loop import (
    ClosedLoopRunner,
    EvaluatedConfiguration,
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
                        "stage1b_enabled": False,
                        "baseline_tuning_iterations": 2,
                        "stage2_num_seeds": 2,
                        "candidate_branches": 5,
                        "candidate_parallel_workers": 2,
                        "stage3_generation_attempts": 2,
                        "candidate_tuning_top_k": 2,
                        "candidate_tuning_iterations": 1,
                        "candidate_refinement_top_k": 1,
                        "candidate_refinement_iterations": 2,
                        "candidate_finalist_top_k": 1,
                        "finalist_top_k": 1,
                        "finalist_num_seeds": 1,
                        "final_confirmation_num_seeds": 2,
                        "ablation_candidate_top_k": 1,
                        "ablation_synergy_pairs": 0,
                        "min_primary_gain": 0.002,
                        "required_seed_wins": 1,
                        "checkpoint_submission_each_incumbent": False,
                        "smoke_test_enabled": False,
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
        self.completed_stages = []
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
    creative_batch_sizes = []

    def __init__(self, **kwargs):
        self.journal = kwargs["journal"]
        self.stage_name = kwargs["stage_name"]
        self.tuning_base = kwargs["tuning_base_node"]
        self.research_base = kwargs["research_base_node"]
        self.research_bases = list(kwargs.get("research_base_nodes") or [])
        self.stage3_base = kwargs["best_stage3_node"]
        self.candidate_contexts = list(kwargs.get("candidate_contexts") or [])
        self.num_workers = int(
            kwargs["cfg"].agent.research_loop.get("candidate_parallel_workers", 1)
        )
        self.max_search_workers = kwargs.get("max_search_workers")
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
            self.creative_batch_sizes.append(count)
            created = []
            research_families = (
                "training_strategy",
                "evidence_synthesis",
                "feature_engineering",
                "history_interest",
                "context_interaction",
            )
            for index in range(count):
                if second_round:
                    score = base_score - 0.001 - index * 0.001
                else:
                    score = base_score + 0.004 - index * 0.001
                context = self.candidate_contexts[index]
                assignment = extract_assignment_contract(context)
                if not assignment or not assignment.get("assignment_id"):
                    raise AssertionError("missing autonomous assignment marker")
                assignment_id = assignment["assignment_id"]
                slot = int(assignment_id.rsplit(":", 1)[-1])
                round_token = "round2" if second_round else "round1"
                token = f"{round_token}_slot{slot}"
                component = f"autonomous_{token}"
                research_family = research_families[
                    (slot - 1) % len(research_families)
                ]
                parent = (
                    self.research_bases[index]
                    if self.research_bases
                    else self.research_base
                )
                code = (
                    "CONFIG = {'learning_rate': 0.001, 'epochs': 6}\n"
                    "RESEARCH_MANIFEST = {"
                    f"'candidate_id': {assignment_id!r}, 'role': 'autonomous_stage3', "
                    "'group': 'autonomous_research', 'category': 'open_choice', "
                    "'model_family': 'fm', "
                    f"'research_family': {research_family!r}, "
                    "'loss_family': 'pointwise_bce', "
                    f"'parent_node_id': {parent.id!r}, "
                    "'parent_model_family': 'fm', 'input_schema_version': 2, "
                    f"'hypothesis': 'independent hypothesis {token} improves ranking', "
                    f"'mechanism': 'guarded autonomous mechanism {token}', "
                    f"'mechanism_ids': [{token!r}], "
                    "'modified_symbols': ['create_model'], "
                    "'expected_metric': ['GAUC', 'nDCG@5'], "
                    "'tunable_parameters': ['learning_rate'], "
                    f"'ablation_components': [{component!r}], "
                    "'combination_compatibility': 'independent component', "
                    "'change_scope': 'one autonomous mechanism', "
                    "'component_dependencies': {}, "
                    "'evidence': ["
                    f"{{'source_type': 'dependency', 'reference': 'dependency:guarded_path', 'supports': [{component!r}]}}]}}\n"
                    "FACTOR_SELECTION = {"
                    "'considered_factor_ids': ['static_user_profile'], "
                    "'selected_factor_ids': [], "
                    "'selection_reason': 'this hypothesis does not need a new factor', "
                    "'rejected_reasons': {'static_user_profile': 'not required'}, "
                    "'created_factor_cards': []}\n"
                    "FEATURE_FACTORS = []\n"
                    f"ABLATION_COMPONENTS = {{{component!r}: True}}\n"
                    "def component_enabled(name):\n"
                    "    return ABLATION_COMPONENTS[name]\n"
                    "def build_features(splits, feature_state=None):\n"
                    "    return splits, feature_state\n"
                    "def create_model(feature_dimension, config=None):\n"
                    f"    if component_enabled({component!r}):\n"
                    f"        return ({token!r}, feature_dimension)\n"
                    "    return ('fm', feature_dimension)\n"
                )
                node = _node(code, score, parent=parent)
                node.assignment_id = assignment_id
                self.journal.append(node)
                created.append(node)
            return created

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


class MultiRootTuningFakeAgent(FakeAgent):
    def step(self, exec_callback, max_nodes=None):
        if not self.stage_name.startswith("2_multi_root_tuning"):
            return super().step(exec_callback, max_nodes=max_nodes)
        count = int(max_nodes or 1)
        base_score = self.tuning_base.metric.get_mean_value()
        for index in range(count):
            self.journal.append(
                _node(
                    self.tuning_base.code + f"# root tuning {index}\n",
                    base_score + 0.001 * (index + 1),
                    parent=self.tuning_base,
                )
            )


class ClosedLoopTests(unittest.TestCase):
    @staticmethod
    def _stage2_snapshot(root: Path) -> Path:
        snapshot = root / "parent_run" / "artifacts" / "incumbent_snapshots" / "001_stage2"
        snapshot.mkdir(parents=True)
        model = (
            "RESEARCH_MANIFEST = {'model_family': 'fm', "
            "'research_family': 'baseline', 'loss_family': 'pointwise_bce'}\n"
            "ABLATION_COMPONENTS = {}\n"
        )
        checkpoint = b"frozen-stage2-checkpoint"
        (snapshot / "model.py").write_text(model)
        (snapshot / "checkpoint.npz").write_bytes(checkpoint)
        (snapshot / "training_history.json").write_text("[]\n")
        (snapshot / "source_node_id.txt").write_text("stage2-node\n")
        manifest = {
            "schema_version": 2,
            "source_stage": "2_multi_root_tuning_1_fm",
            "source_node_id": "stage2-node",
            "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            "test_metrics_used_for_selection": False,
            "validation": {
                "GAUC": {"mean": 0.62, "values": [0.61, 0.63]},
                "nDCG@5": {"mean": 0.58, "values": [0.57, 0.59]},
                "primary": {"mean": 0.60, "values": [0.59, 0.61]},
            },
        }
        (snapshot / "manifest.json").write_text(json.dumps(manifest))
        return snapshot

    def test_stage2_resume_verifies_and_materializes_independent_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._stage2_snapshot(root)
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "continuation" / "artifacts" / "final_model",
            )
            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                resume_from_stage2=snapshot,
                agent_factory=FakeAgent,
                finalizer=lambda *args, **kwargs: {},
                submission_exporter=lambda **kwargs: {},
            )

            resumed = runner._load_stage2_resume(snapshot)
            copied = root / "continuation" / "artifacts" / "resume_parent_stage2"
            provenance = json.loads(
                (root / "continuation" / "artifacts" / "resume_provenance.json").read_text()
            )
            copied_checkpoint_exists = (copied / "checkpoint.npz").is_file()

        self.assertEqual(resumed.node.id, "stage2-node")
        self.assertEqual(resumed.seed_scores, [0.59, 0.61])
        self.assertAlmostEqual(resumed.score, 0.60)
        self.assertTrue(copied_checkpoint_exists)
        self.assertEqual(provenance["source_stage"], "2_multi_root_tuning_1_fm")
        self.assertFalse(provenance["test_metrics_used_for_selection"])
        self.assertEqual(runner.state.verified_incumbent_id, "stage2-node")

    def test_stage2_resume_rejects_hash_mismatch_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = self._stage2_snapshot(root)
            (snapshot / "checkpoint.npz").write_bytes(b"tampered")
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "continuation" / "artifacts" / "final_model",
            )
            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                resume_from_stage2=snapshot,
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner._load_stage2_resume(snapshot)
            self.assertFalse(
                (root / "continuation" / "artifacts" / "resume_parent_stage2").exists()
            )

    def setUp(self):
        FakeAgent.created_stage_names = []
        FakeAgent.created_task_descs = {}
        FakeAgent.creative_batch_sizes = []
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

    def test_node_assignment_id_survives_worker_serialization(self):
        node = _node("MODEL = 'candidate'", 0.67)
        node.assignment_id = "round1:autonomous:3"

        restored = Node.from_dict(dict(node.to_dict()))

        self.assertEqual(restored.assignment_id, node.assignment_id)

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

    def test_five_autonomous_slots_run_in_real_worker_sized_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "artifacts" / "final_model",
            )
            manager.cfg.agent.research_loop.candidate_branches = 5
            manager.cfg.agent.research_loop.stage3_generation_attempts = 5
            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                agent_factory=FakeAgent,
                finalizer=lambda *args, **kwargs: {},
                submission_exporter=lambda **kwargs: {},
            )
            incumbent = _node(
                "ABLATION_COMPONENTS = {}\nBASELINE = 'FM'\n",
                0.600,
            )

            candidates = runner._generate_candidates(incumbent, 1)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(FakeAgent.creative_batch_sizes, [2, 2, 1])
        self.assertEqual(
            [node.assignment_id for node in candidates],
            [f"round1:autonomous:{index}" for index in range(1, 6)],
        )
        self.assertFalse(
            any(
                item.status == "worker_failed"
                for item in runner.state.memory.attempts
            )
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
        self.assertAlmostEqual(result["incumbent"].score, 0.6095)
        self.assertIn("'autonomous_round1_slot1': False", result["incumbent"].node.code)
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
        self.assertEqual(finalized["required_seeds"], 2)
        self.assertTrue(result["submission"]["checked"])
        self.assertEqual(latest_summary["round_number"], 2)
        self.assertEqual(archived_round_one["round_number"], 1)
        self.assertFalse(latest_summary["test_metrics_used"])
        self.assertIn("validation-only", prompt_summary)
        self.assertIn(
            "Structured validation-only candidate experience",
            FakeAgent.created_task_descs["3_creative_research_1_round_two_autonomous"],
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
            "3_creative_research_1_round_two_autonomous"
        ]
        root_node = round_two_journal.nodes[0]
        siblings = [
            node for node in round_two_journal.nodes if node.parent is root_node
        ]
        self.assertEqual(len(siblings), 2)
        self.assertIn(
            "'autonomous_round1_slot1': False",
            root_node.code,
        )

    def test_stage2_fairly_tunes_stage1b_root_and_can_replace_fm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "artifacts" / "final_model",
            )
            manager.cfg.agent.research_loop.stage2_root_top_k = 2
            manager.cfg.agent.research_loop.stage2_num_seeds = 2
            manager.cfg.agent.research_loop.min_primary_gain = 0.002
            baseline = _node("ABLATION_COMPONENTS = {}\nBASELINE = 'FM'\n", 0.600)
            wide_deep = _node(
                "RESEARCH_MANIFEST = {'model_family': 'wide_deep', "
                "'research_family': 'architecture', 'loss_family': 'pointwise_bce', "
                "'parent_node_id': 'fm', 'parent_model_family': 'fm', "
                "'role': 'architecture_wide_deep'}\n"
                "FACTOR_SELECTION = {'considered_factor_ids': ['static_user_profile'], "
                "'selected_factor_ids': [], 'selection_reason': 'architecture only', "
                "'rejected_reasons': {'static_user_profile': 'not required'}, "
                "'created_factor_cards': []}\n"
                "ABLATION_COMPONENTS = {'wide_deep_block': True}\n"
                "def component_enabled(name): return ABLATION_COMPONENTS.get(name, False)\n",
                0.604,
            )
            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                agent_factory=MultiRootTuningFakeAgent,
                finalizer=lambda *args, **kwargs: {},
                submission_exporter=lambda **kwargs: {},
            )

            winner = runner._run_stage2(baseline, [wide_deep])

        self.assertEqual(winner.model_family, "wide_deep")
        self.assertEqual(runner.state.verified_incumbent_id, winner.node.id)
        standardized = [
            record
            for record in runner.state.memory.records
            if record.source_phase == "stage1b_standardized_stage2"
        ]
        self.assertEqual(len(standardized), 1)
        self.assertEqual(standardized[0].status, "verified_incumbent")

    def test_incumbent_snapshot_exports_submission_and_updates_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = FakeManager(
                self._data_dir(root / "starter"),
                root / "artifacts" / "final_model",
            )
            manager.cfg.agent.research_loop.checkpoint_submission_each_incumbent = True
            frozen = {}
            exported = {}

            def finalizer(journal, **kwargs):
                frozen.update(kwargs)
                Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
                return {"source_stage": kwargs["source_stage"]}

            def submission_exporter(**kwargs):
                exported.update(kwargs)
                return {
                    "path": str(Path(kwargs["output_dir"]) / "submission.csv"),
                    "checked": True,
                }

            runner = ClosedLoopRunner(
                manager,
                exec_callback=lambda *args: None,
                agent_factory=FakeAgent,
                finalizer=finalizer,
                submission_exporter=submission_exporter,
            )
            journal = Journal()
            node = _node("ABLATION_COMPONENTS = {}\nBASELINE = 'FM'\n", 0.602)
            journal.append(node)
            runner.incumbent = EvaluatedConfiguration(
                node=node,
                journal=journal,
                stage_name="2_baseline_tuning_1_test",
                score=0.602,
                seed_scores=[0.601, 0.603],
                metrics={"GAUC": 0.61, "nDCG@5": 0.594, "primary": 0.602},
            )

            pointer = runner._snapshot_best_available("verified test")
            pointer_file = root / "artifacts" / "best_available.json"

            self.assertIsNotNone(pointer)
            self.assertTrue(pointer_file.is_file())
            self.assertTrue(pointer["submission"]["checked"])
            self.assertEqual(frozen["required_seeds"], 2)
            self.assertEqual(exported["output_dir"], frozen["output_dir"])
            self.assertIn("incumbent_snapshots", str(exported["output_dir"]))

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

    def test_baseline_gets_configured_seed_confirmation_when_no_candidate_improves(self):
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
        self.assertEqual(len(result["incumbent"].seed_scores), 2)
        self.assertEqual(len(result["state"].incumbent_seed_scores), 2)
        self.assertEqual(finalized["required_seeds"], 2)
        self.assertNotIn(
            "4_final_incumbent_confirmation_1_configured_seeds",
            NoImprovementFakeAgent.created_stage_names,
        )


if __name__ == "__main__":
    unittest.main()
