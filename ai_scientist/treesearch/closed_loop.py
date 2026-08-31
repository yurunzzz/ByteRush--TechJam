"""Deterministic Stage 1-4 orchestration for the ByteRush research loop."""
from __future__ import annotations

import ast
import copy
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional, Sequence

from rich import print

from .candidate_contract import (
    CandidateRole,
    bootstrap_candidate_roles,
    candidate_semantic_signature,
    component_guard_calls,
    config_assignment,
    format_factor_change,
    literal_assignment,
    model_family_from_code,
    normalize_candidate_metadata,
    research_family_for_role,
    validate_candidate_contract,
    select_candidate_roles,
)
from .factor_context import build_factor_context
from .factor_library import factor_library_prompt
from .finalize import freeze_stage4_winner
from .journal import Journal, Node
from .parallel_agent import (
    ParallelAgent,
    _extract_enabled_ablation_components,
    _load_kuairand_validation_metric,
    get_gpu_count,
)
from .research_loop import (
    AblationEvidence,
    ExperimentRecord,
    FrontierRecord,
    PromotionPolicy,
    ResearchLoopConfig,
    ResearchLoopState,
    candidate_fingerprint,
    estimate_max_experiment_runs,
    extract_primary_score,
    research_loop_config_from_mapping,
)


@dataclass
class EvaluatedConfiguration:
    node: Node
    journal: Journal
    stage_name: str
    score: float
    seed_scores: list[float]
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    seed_metrics: list[dict[str, Optional[float]]] = field(default_factory=list)
    metric_means: dict[str, float] = field(default_factory=dict)
    role: Optional[CandidateRole] = None
    candidate_id: Optional[str] = None
    model_family: str = "fm"
    research_family: str = ""
    loss_family: str = "pointwise_bce"
    parent_node_id: str = ""
    parent_model_family: str = "fm"


@dataclass(frozen=True)
class CandidateAssignment:
    role: CandidateRole
    parent: Node
    parent_model_family: str
    source: str


def export_and_check_submission(
    output_dir: str | Path,
    starter_kit: str | Path,
) -> dict[str, Any]:
    """Use the organizer submission entry point for export and schema checking."""
    output_dir = Path(output_dir).resolve()
    starter_kit = Path(starter_kit).resolve()
    submit_script = starter_kit / "submit.py"
    data_dir = starter_kit / "KuaiRand-Pure" / "data"
    if not submit_script.is_file():
        raise FileNotFoundError(f"official submit.py not found: {submit_script}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"KuaiRand-Pure data not found: {data_dir}")

    submission = output_dir / "submission.csv"
    make_command = [
        sys.executable,
        str(submit_script),
        str(submission),
        "--make-best",
        "--artifact_dir",
        str(output_dir),
    ]
    print("[cyan]Loading frozen checkpoint and generating submission.csv[/cyan]")
    subprocess.run(make_command, cwd=starter_kit, check=True)

    check_command = [
        sys.executable,
        str(submit_script),
        str(submission),
        "--check",
        "--data_dir",
        str(data_dir),
        "--split",
        "test",
    ]
    print("[cyan]Running official submit.py --check[/cyan]")
    subprocess.run(check_command, cwd=starter_kit, check=True)

    metadata_path = submission.with_suffix(submission.suffix + ".metadata.json")
    metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    )
    return {
        "path": str(submission),
        "metadata_path": str(metadata_path),
        "rows": metadata.get("rows"),
        "sha256": metadata.get("submission_sha256"),
        "checked": True,
    }


def _cfg_get(container: Any, name: str, default: Any) -> Any:
    if container is None:
        return default
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(name, default)
    return getattr(container, name, default)


def _clone_root(node: Node) -> Node:
    clone = copy.deepcopy(node)
    clone.parent = None
    clone.children = set()
    clone.step = None
    return clone


def _node_score(node: Node) -> Optional[float]:
    if node.is_buggy or node.metric is None:
        return None
    result = extract_primary_score(node, allow_fallback=False)
    if result is None or not math.isfinite(result.score):
        return None
    return result.score


def _node_validation_metrics(node: Node) -> dict[str, Optional[float]]:
    """Read only trusted KuaiRand validation metrics for experience memory."""
    score = _node_score(node)
    if score is None:
        return {}
    if node.exp_results_dir:
        result_dir = Path(node.exp_results_dir)
        metric_dirs = [result_dir]
        if result_dir.is_dir():
            metric_dirs.extend(
                path.parent for path in result_dir.rglob("experiment_data.npy")
            )
        term_out = "".join(node._term_out or [])
        seen_dirs = set()
        for metric_dir in metric_dirs:
            resolved = metric_dir.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            try:
                gauc, ndcg5, primary = _load_kuairand_validation_metric(
                    str(resolved), term_out
                )
                return {
                    "GAUC": float(gauc),
                    "nDCG@5": float(ndcg5),
                    "primary": float(primary),
                }
            except (OSError, KeyError, TypeError, ValueError):
                continue
    # Generic tests may expose only the selection score. Never infer component
    # metrics from arbitrary output, where test-like values could appear.
    return {"GAUC": None, "nDCG@5": None, "primary": float(score)}


def _mean_validation_metrics(
    metrics: Sequence[dict[str, Optional[float]]],
) -> dict[str, Optional[float]]:
    result = {}
    for name in ("GAUC", "nDCG@5", "primary"):
        values = [
            float(item[name])
            for item in metrics
            if item.get(name) is not None and math.isfinite(float(item[name]))
        ]
        result[name] = mean(values) if values else None
    return result


def _principal_change(node: Node) -> str:
    components = _extract_enabled_ablation_components(node.code)
    component_text = ", ".join(components) if components else "unregistered component"
    narrative = next(
        (
            str(value).strip()
            for value in (node.overall_plan, node.plan, node.analysis)
            if value and str(value).strip()
        ),
        "controlled candidate change",
    )
    narrative = " ".join(narrative.split())[:240]
    return f"{component_text}: {narrative}"


def _factor_selection(
    node: Node,
) -> tuple[list[str], list[str], str, dict[str, str], list[dict[str, Any]]]:
    selection = literal_assignment(node.code, "FACTOR_SELECTION")
    if not isinstance(selection, dict):
        return [], [], "factor library was not inspected", {}, []
    considered = selection.get("considered_factor_ids", [])
    considered_factor_ids = (
        [str(item) for item in considered]
        if isinstance(considered, (list, tuple))
        else []
    )
    selected = selection.get("selected_factor_ids", [])
    factor_ids = (
        [str(item) for item in selected]
        if isinstance(selected, (list, tuple))
        else []
    )
    created = selection.get("created_factor_cards", [])
    rejected = selection.get("rejected_reasons", {})
    rejected_reasons = (
        {str(key): str(value) for key, value in rejected.items()}
        if isinstance(rejected, dict)
        else {}
    )
    factor_cards = (
        [dict(item) for item in created if isinstance(item, dict)]
        if isinstance(created, (list, tuple))
        else []
    )
    return (
        considered_factor_ids,
        factor_ids,
        str(selection.get("selection_reason", "")).strip(),
        rejected_reasons,
        factor_cards,
    )


def _node_metrics(node: Node) -> dict[str, float]:
    """Expose finite trusted validation metrics to Stage 4 evidence logic."""
    return {
        name: float(value)
        for name, value in _node_validation_metrics(node).items()
        if value is not None and math.isfinite(float(value))
    }


def _mean_metrics(nodes: Sequence[Node]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for node in nodes:
        for name, value in _node_metrics(node).items():
            values.setdefault(name, []).append(value)
    return {name: mean(items) for name, items in values.items() if items}


def _candidate_artifacts(node: Node) -> dict[str, Any]:
    """Describe reloadable validation artifacts without reading any test outcome."""
    metadata: dict[str, Any] = {
        "valid": bool(node.code.strip()) and _node_score(node) is not None,
        "code_path": "",
        "checkpoint_path": "",
        "validation_split": "fixed_train_validation",
        "training_epochs": None,
    }
    config = config_assignment(node.code) or {}
    epochs = config.get("max_epochs", config.get("epochs"))
    if isinstance(epochs, int):
        metadata["training_epochs"] = epochs

    if not node.exp_results_dir:
        # Unit-test/in-memory nodes have no on-disk artifacts. Production nodes
        # always carry exp_results_dir and are checked below.
        return metadata
    result_dir = Path(node.exp_results_dir)
    if not result_dir.is_dir():
        metadata["valid"] = False
        return metadata
    code_files = sorted(result_dir.rglob("experiment_code.py"))
    if not code_files:
        code_files = sorted(result_dir.rglob("runfile.py"))
    checkpoints = sorted(result_dir.rglob("*checkpoint*.npz"))
    metric_files = sorted(result_dir.rglob("experiment_data.npy"))
    metadata["code_path"] = str(code_files[0]) if code_files else ""
    metadata["checkpoint_path"] = str(checkpoints[0]) if checkpoints else ""
    metadata["valid"] = bool(
        metadata["valid"] and code_files and checkpoints and metric_files
    )
    return metadata


def _ranked_nodes(journal: Journal, limit: Optional[int] = None) -> list[Node]:
    candidates = []
    for node in journal.nodes:
        if node.is_seed_node:
            continue
        score = _node_score(node)
        if score is not None:
            candidates.append((score, node))
    candidates.sort(key=lambda item: item[0], reverse=True)
    nodes = [node for _, node in candidates]
    return nodes if limit is None else nodes[:limit]


def _materialize_ablation_winner(node: Node) -> Node:
    """Turn a winning leave-one-out node into a stable next-round parent."""
    if not node.ablation_name:
        return _clone_root(node)
    try:
        tree = ast.parse(node.code)
    except SyntaxError:
        return _clone_root(node)

    components = tuple(
        component for component in node.ablation_name.split("+") if component
    )
    changed_manifest = False
    cleaned_body = []
    for statement in tree.body:
        is_target_assignment = (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
            and isinstance(statement.targets[0].value, ast.Attribute)
            and isinstance(statement.targets[0].value.value, ast.Name)
            and statement.targets[0].value.value.id == "os"
            and statement.targets[0].value.attr == "environ"
            and isinstance(statement.targets[0].slice, ast.Constant)
            and statement.targets[0].slice.value == "AI_SCIENTIST_ABLATION_TARGET"
        )
        if is_target_assignment:
            continue
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ABLATION_COMPONENTS"
            for target in statement.targets
        ):
            try:
                manifest = ast.literal_eval(statement.value)
            except (ValueError, TypeError, SyntaxError):
                manifest = None
            if isinstance(manifest, dict) and all(
                component in manifest for component in components
            ):
                for component in components:
                    manifest[component] = False
                statement.value = ast.parse(repr(manifest), mode="eval").body
                changed_manifest = True
        cleaned_body.append(statement)
    if not changed_manifest:
        return _clone_root(node)

    tree.body = cleaned_body
    ast.fix_missing_locations(tree)
    result = _clone_root(node)
    result.code = ast.unparse(tree) + "\n"
    result.ablation_name = None
    return result


def _joint_ablation_node(full_model: Node, components: Sequence[str]) -> Node:
    working = _clone_root(full_model)
    for component in components:
        working.ablation_name = component
        working = _materialize_ablation_winner(working)
    label = "+".join(components)
    return Node(
        plan=(
            "Controlled joint ablation. Disable exactly "
            + ", ".join(components)
            + "; preserve every other setting."
        ),
        code=working.code,
        parent=full_model,
        is_buggy=False,
        ablation_name=label,
    )


class ClosedLoopRunner:
    """Run the production ByteRush loop while preserving the generic runner."""

    def __init__(
        self,
        manager: Any,
        exec_callback: Callable,
        step_callback: Optional[Callable] = None,
        *,
        agent_factory: Callable[..., Any] = ParallelAgent,
        finalizer: Callable[..., dict[str, Any]] = freeze_stage4_winner,
        submission_exporter: Callable[..., dict[str, Any]] = (
            export_and_check_submission
        ),
    ):
        self.manager = manager
        self.exec_callback = exec_callback
        self.step_callback = step_callback
        self.agent_factory = agent_factory
        self.finalizer = finalizer
        self.submission_exporter = submission_exporter

        raw = _cfg_get(manager.cfg.agent, "research_loop", {})
        self.config = research_loop_config_from_mapping(raw)
        policy = PromotionPolicy(
            min_improvement=float(_cfg_get(raw, "min_primary_gain", 0.002)),
            required_seeds=self.config.finalist_num_seeds,
            required_seed_wins=int(_cfg_get(raw, "required_seed_wins", 2)),
            maximize=True,
            max_component_regression=float(
                _cfg_get(raw, "max_component_regression", 0.001)
            ),
        )
        self.state = ResearchLoopState(config=self.config, policy=policy)
        self.ablation_cfg = _cfg_get(manager.cfg.agent, "ablation", {})
        self.max_ablation_components = int(
            _cfg_get(self.ablation_cfg, "max_components", 6)
        )
        self.output_dir = Path(
            _cfg_get(manager.cfg.agent, "final_model_dir", "artifacts/final_model")
        ).resolve()
        self.candidate_role_by_node_id: dict[str, CandidateRole] = {}
        self.candidate_origin_by_node_id: dict[str, tuple[Any, Journal]] = {}
        self.candidate_parent_by_node_id: dict[str, Node] = {}
        self.frontier_nodes_by_id: dict[str, Node] = {}
        self.bootstrap_node_ids: set[str] = set()
        self.memory_dir = self.output_dir.parent / "research_loop"
        self.generated_fingerprints: set[str] = set()
        self.generated_semantic_signatures: set[str] = set()
        self.incumbent: Optional[EvaluatedConfiguration] = None

    def _new_stage(
        self,
        *,
        name: str,
        goals: str,
        max_iterations: int,
        num_drafts: int = 0,
    ) -> tuple[Any, Journal]:
        stage_type = type(self.manager.stages[0])
        self.manager.current_stage_number += 1
        stage = stage_type(
            name=name,
            description=name,
            goals=goals,
            max_iterations=max_iterations,
            num_drafts=num_drafts,
            stage_number=self.manager.current_stage_number,
        )
        journal = Journal()
        self.manager.stages.append(stage)
        self.manager.journals[name] = journal
        self.manager.current_stage = stage
        return stage, journal

    def _task_desc(self, stage: Any, extra_context: str = "") -> str:
        task_desc = self.manager._curate_task_desc(stage)
        main_stage, main_name, substage, substage_name = (
            self.manager.parse_stage_names(stage.name)
        )
        task_desc += (
            f"\n\nCurrent Main Stage: {main_name}\n"
            f"Sub-stage: {substage} - {substage_name}\n"
            f"Sub-stage goals: {stage.goals}\n"
        )
        if extra_context:
            task_desc += "\n" + extra_context + "\n"
        return task_desc

    def _agent(
        self,
        stage: Any,
        journal: Journal,
        *,
        tuning_base: Optional[Node] = None,
        research_base: Optional[Node] = None,
        research_bases: Optional[Sequence[Node]] = None,
        stage3_base: Optional[Node] = None,
        task_context: str = "",
        num_workers: Optional[int] = None,
        max_search_workers: Optional[int] = None,
        candidate_contexts: Optional[Sequence[str]] = None,
    ):
        stage_cfg = (
            self.manager.cfg.copy()
            if hasattr(self.manager.cfg, "copy")
            else copy.deepcopy(self.manager.cfg)
        )
        stage_cfg.agent.search.num_drafts = stage.num_drafts
        if num_workers is not None:
            stage_cfg.agent.num_workers = max(1, int(num_workers))
        return self.agent_factory(
            task_desc=self._task_desc(stage, task_context),
            cfg=stage_cfg,
            journal=journal,
            stage_name=stage.name,
            best_stage3_node=stage3_base,
            best_stage2_node=research_base,
            best_stage1_node=tuning_base if stage.name.startswith("2_") else None,
            tuning_base_node=tuning_base,
            research_base_node=research_base,
            research_base_nodes=research_bases,
            max_search_workers=max_search_workers,
            candidate_contexts=candidate_contexts,
        )

    def _notify(self, stage: Any, journal: Journal) -> None:
        if self.step_callback:
            self.step_callback(stage, journal)

    @staticmethod
    def _experiment_count(journal: Journal) -> int:
        return sum(not node.is_seed_node for node in journal.nodes)

    def _run_budget(
        self,
        agent: Any,
        stage: Any,
        journal: Journal,
        budget: int,
        *,
        inherited_nodes: int = 0,
        stop_after_first_valid: bool = False,
    ) -> int:
        completed = max(0, self._experiment_count(journal) - inherited_nodes)
        while completed < budget:
            before = self._experiment_count(journal)
            max_nodes = 1 if stop_after_first_valid else budget - completed
            agent.step(self.exec_callback, max_nodes=max_nodes)
            after = self._experiment_count(journal)
            added = after - before
            if added <= 0:
                break
            completed += added
            self._notify(stage, journal)
            if stop_after_first_valid and _ranked_nodes(journal, 1):
                break
        return completed

    def _evaluate_nodes(
        self,
        agent: Any,
        stage: Any,
        journal: Journal,
        nodes: Sequence[Node],
        *,
        num_seeds: Optional[int] = None,
        role: Optional[CandidateRole] = None,
        candidate_id: Optional[str] = None,
    ) -> list[EvaluatedConfiguration]:
        evaluated = []
        required = int(num_seeds or self.config.finalist_num_seeds)
        for node in nodes:
            seed_nodes = agent._run_multi_seed_evaluation(
                node, num_seeds=required
            )
            self._notify(stage, journal)
            seed_scores = [
                score
                for score in (_node_score(seed) for seed in seed_nodes)
                if score is not None
            ]
            if len(seed_scores) != required:
                continue
            seed_metrics = [_node_validation_metrics(seed) for seed in seed_nodes]
            metrics = _mean_validation_metrics(seed_metrics)
            evaluated.append(
                EvaluatedConfiguration(
                    node=node,
                    journal=journal,
                    stage_name=stage.name,
                    score=mean(seed_scores),
                    seed_scores=seed_scores,
                    metrics=metrics,
                    seed_metrics=seed_metrics,
                    metric_means={
                        name: float(value)
                        for name, value in metrics.items()
                        if value is not None
                    },
                    role=role,
                    candidate_id=candidate_id,
                )
            )
        return evaluated

    def _evaluate_top_k(
        self,
        agent: Any,
        stage: Any,
        journal: Journal,
        *,
        num_seeds: Optional[int] = None,
    ) -> Optional[EvaluatedConfiguration]:
        finalists = _ranked_nodes(journal, self.config.finalist_top_k)
        results = self._evaluate_nodes(
            agent,
            stage,
            journal,
            finalists,
            num_seeds=num_seeds,
        )
        return max(results, key=lambda result: result.score) if results else None

    def _run_stage1(self) -> Node:
        stage = self.manager.current_stage
        stage.max_iterations = self.config.stage1_validation_iterations
        journal = self.manager.journals[stage.name]
        with self._agent(stage, journal) as agent:
            self._run_budget(
                agent,
                stage,
                journal,
                self.config.stage1_validation_iterations,
                stop_after_first_valid=True,
            )
        valid = _ranked_nodes(journal, 1)
        if not valid:
            raise RuntimeError("Stage 1 did not produce a valid FM baseline")
        return _clone_root(valid[0])

    def _stage2_roots(
        self, baseline: Node, stage1b_roots: Sequence[Node]
    ) -> list[Node]:
        """Keep the FM control plus the strongest bootstrap root per family."""
        limit = max(1, int(self.config.stage2_root_top_k))
        selected = [baseline]
        seen_families = {"fm"}
        for node in sorted(
            stage1b_roots,
            key=lambda item: _node_score(item) or float("-inf"),
            reverse=True,
        ):
            family = model_family_from_code(node.code)
            if family in seen_families:
                continue
            selected.append(node)
            seen_families.add(family)
            if len(selected) >= limit:
                break
        return selected

    def _run_stage2_root(
        self, root: Node, root_number: int
    ) -> Optional[EvaluatedConfiguration]:
        family = model_family_from_code(root.code)
        stage, journal = self._new_stage(
            name=f"2_multi_root_tuning_{root_number}_{family}",
            goals=(
                self.manager.main_stage_goals[2]
                + "\n- The assigned root family is "
                + family
                + ". Keep this architecture fixed and tune it under exactly the "
                "same budget used for every competing Stage 1 root."
            ),
            max_iterations=self.config.baseline_tuning_iterations,
        )
        tuning_base = _clone_root(root)
        journal.append(tuning_base)
        role = self.candidate_role_by_node_id.get(root.id)
        with self._agent(
            stage,
            journal,
            tuning_base=tuning_base,
            max_search_workers=1,
        ) as agent:
            self._run_budget(
                agent,
                stage,
                journal,
                self.config.baseline_tuning_iterations,
                inherited_nodes=1,
            )
            finalists = _ranked_nodes(journal, self.config.finalist_top_k)
            evaluated = self._evaluate_nodes(
                agent,
                stage,
                journal,
                finalists,
                num_seeds=self.config.stage2_num_seeds,
                role=role,
                candidate_id=root.id,
            )
        if not evaluated:
            return None
        winner = max(evaluated, key=lambda result: result.score)
        metadata = self._manifest_metadata(winner.node)
        winner.model_family = metadata["model_family"] or family
        winner.research_family = metadata["research_family"]
        winner.loss_family = metadata["loss_family"] or "pointwise_bce"
        winner.parent_node_id = metadata["parent_node_id"]
        winner.parent_model_family = metadata["parent_model_family"] or family
        return winner

    def _run_stage2(
        self, baseline: Node, stage1b_roots: Sequence[Node]
    ) -> EvaluatedConfiguration:
        """Fairly tune/replicate FM and diverse roots, then select one incumbent."""
        roots = self._stage2_roots(baseline, stage1b_roots)
        evaluated = [
            result
            for root_number, root in enumerate(roots, 1)
            if (result := self._run_stage2_root(root, root_number)) is not None
        ]
        if not evaluated:
            raise RuntimeError("Stage 2 produced no root with valid standardized seeds")
        fm_results = [item for item in evaluated if item.model_family == "fm"]
        if not fm_results:
            raise RuntimeError("Stage 2 lost the required FM control root")
        fm_control = max(fm_results, key=lambda result: result.score)
        self.state.set_incumbent(
            fm_control.node.id,
            fm_control.score,
            fm_control.seed_scores,
            metrics=fm_control.metrics,
            source_stage=fm_control.stage_name,
        )

        stage2_policy = PromotionPolicy(
            min_improvement=self.state.policy.min_improvement,
            required_seeds=self.config.stage2_num_seeds,
            required_seed_wins=max(1, math.ceil(self.config.stage2_num_seeds / 2)),
            maximize=self.state.policy.maximize,
            max_component_regression=self.state.policy.max_component_regression,
        )
        provisional: list[EvaluatedConfiguration] = []
        for result in evaluated:
            if result is fm_control:
                continue
            artifact = _candidate_artifacts(result.node)
            role = result.role
            (
                considered_factor_ids,
                factor_ids,
                factor_selection_reason,
                factor_rejected_reasons,
                factor_cards,
            ) = _factor_selection(result.node)
            metadata = self._manifest_metadata(result.node)
            decision = self.state.evaluate_candidate(
                node_id=result.node.id,
                fingerprint=candidate_fingerprint(result.node.code),
                score=result.score,
                seed_scores=result.seed_scores,
                metrics=result.metrics,
                seed_metrics=result.seed_metrics,
                principal_change=_principal_change(result.node),
                components=_extract_enabled_ablation_components(result.node.code),
                role=role.name if role else metadata["role"],
                category=role.group if role else "architecture_exploration",
                factor_ids=factor_ids,
                considered_factor_ids=considered_factor_ids,
                factor_selection_reason=factor_selection_reason,
                factor_rejected_reasons=factor_rejected_reasons,
                factor_cards=factor_cards,
                model_family=metadata["model_family"],
                research_family=metadata["research_family"],
                loss_family=metadata["loss_family"],
                parent_node_id=metadata["parent_node_id"],
                parent_model_family=metadata["parent_model_family"],
                source_stage=result.stage_name,
                source_phase="stage1b_standardized_stage2",
                code_path=artifact["code_path"],
                checkpoint_path=artifact["checkpoint_path"],
                validation_split=artifact["validation_split"],
                random_seeds=range(len(result.seed_scores)),
                training_epochs=artifact["training_epochs"],
                artifacts_valid=bool(artifact["valid"]),
                round_number=0,
                policy_override=stage2_policy,
            )
            signature = self._semantic_signature_for_node(result.node)
            self._register_frontier(
                result.node,
                source_stage=result.stage_name,
                round_number=0,
                semantic_signature=signature,
            )
            print(
                f"[cyan]Standardized Stage 1B root {result.model_family}: "
                f"{decision.reason}[/cyan]"
            )
            if decision.promoted:
                provisional.append(result)

        if provisional:
            winner = max(provisional, key=lambda result: result.score)
            self.incumbent = winner
            self.state.set_incumbent(
                winner.node.id,
                winner.score,
                winner.seed_scores,
                metrics=winner.metrics,
                source_stage=winner.stage_name,
            )
            self.state.memory.mark_verified_incumbent(winner.node.id)
            print(
                f"[green]Stage 2 verified incumbent is {winner.model_family} "
                f"with primary={winner.score:.6f}[/green]"
            )
            return winner

        self.incumbent = fm_control
        print(
            f"[green]Stage 2 retained FM control with primary="
            f"{fm_control.score:.6f}[/green]"
        )
        return fm_control

    def _generate_candidates_single_parent_legacy(
        self, incumbent: Node, round_number: int
    ) -> list[Node]:
        """Compatibility implementation retained for old checkpoints only."""
        roles = select_candidate_roles(
            self.state.memory.direction_summary(),
            round_number=round_number,
            branch_count=self.config.candidate_branches,
            preferred_role_names=self.config.initial_candidate_roles,
            reserved_role_name=self.config.reserved_candidate_role,
        )
        role_by_name = {role.name: role for role in roles}
        word = self._word(round_number)
        stage, journal = self._new_stage(
            name=f"3_creative_research_1_round_{word}",
            goals=self.manager.main_stage_goals[3],
            max_iterations=self.config.stage3_generation_attempts,
        )
        research_base = _clone_root(incumbent)
        journal.append(research_base)
        context = build_factor_context(
            self.manager.cfg.data_dir,
            incumbent_code=research_base.code,
            round_number=round_number,
            failed_hypotheses=self.state.memory.failed_hypotheses,
        )
        context += "\n\n" + self.state.experience_prompt(
            before_round=round_number
        )
        context += (
            "\nDirection evidence from previous rounds:\n"
            + json.dumps(
                self.state.memory.direction_summary(),
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        accepted: dict[str, Node] = {}
        retry_feedback: dict[str, list[str]] = {
            role.name: [] for role in roles
        }
        attempts = 0
        seen_node_ids = {research_base.id}

        def role_prompt(role: CandidateRole, index: int) -> str:
            evidence_memory = (
                "Local direction evidence: "
                + self.state.memory.prompt_evidence(role.group)
                + "\nCross-branch portfolio lessons: "
                + self.state.memory.portfolio_lessons()
            )
            return role.prompt(
                index,
                len(roles),
                retry_feedback=retry_feedback[role.name],
                evidence_memory=evidence_memory,
                factor_library_context=factor_library_prompt(
                    role_group=role.group,
                    role_category=role.category,
                    observed_evidence=self.state.memory.factor_summary(),
                    discovered_cards=self.state.memory.discovered_factor_cards(),
                ),
            )

        with self._agent(
            stage,
            journal,
            research_base=research_base,
            task_context=context,
            candidate_contexts=[
                role_prompt(role, index)
                for index, role in enumerate(roles, 1)
            ],
        ) as agent:
            while attempts < self.config.stage3_generation_attempts:
                pending = [role for role in roles if role.name not in accepted]
                if not pending:
                    break
                assigned = pending[: min(
                    self.config.stage3_generation_attempts - attempts,
                    len(pending),
                )]
                agent.candidate_contexts = [
                    role_prompt(role, roles.index(role) + 1)
                    for role in assigned
                ]
                before = self._experiment_count(journal)
                agent.step(
                    self.exec_callback,
                    max_nodes=len(assigned),
                )
                after = self._experiment_count(journal)
                added = after - before
                if added <= 0:
                    break
                attempts += added
                self._notify(stage, journal)
                new_nodes = [
                    node for node in journal.nodes if node.id not in seen_node_ids
                ]
                for expected_role, node in zip(assigned, new_nodes):
                    seen_node_ids.add(node.id)
                    if _node_score(node) is None:
                        detail = str(getattr(node, "analysis", "execution failed"))
                        reason = f"execution failed: {detail[:300]}"
                        retry_feedback[expected_role.name].append(reason)
                        self.state.memory.failed_hypotheses.append(
                            f"{expected_role.name}: {reason}"
                        )
                        continue
                    node.code = normalize_candidate_metadata(
                        node.code,
                        expected_role,
                        expected_parent_id=research_base.id,
                        expected_parent_model_family=model_family_from_code(
                            research_base.code
                        ),
                    )
                    manifest = literal_assignment(node.code, "RESEARCH_MANIFEST")
                    role_name = (
                        manifest.get("role") if isinstance(manifest, dict) else None
                    )
                    role = role_by_name.get(str(role_name))
                    if role is None or role.name != expected_role.name:
                        reason = (
                            f"expected role {expected_role.name!r}, received "
                            f"manifest role {role_name!r}"
                        )
                        retry_feedback[expected_role.name].append(reason)
                        self.state.memory.failed_hypotheses.append(reason)
                        print(f"[yellow]Rejected Stage 3 candidate: {reason}[/yellow]")
                        continue
                    contract = validate_candidate_contract(
                        research_base.code,
                        node.code,
                        role,
                    )
                    if not contract.valid:
                        reason = "; ".join(contract.reasons)
                        retry_feedback[role.name].append(reason)
                        self.state.memory.failed_hypotheses.append(
                            f"{role.name}: {reason}"
                        )
                        print(f"[yellow]Rejected Stage 3 candidate: {reason}[/yellow]")
                        continue
                    semantic_signature = candidate_semantic_signature(contract)
                    if semantic_signature in self.generated_semantic_signatures:
                        reason = "semantic duplicate of an earlier mechanism/factor bundle"
                        retry_feedback[role.name].append(reason)
                        self.state.memory.failed_hypotheses.append(
                            f"{role.name}: {reason}"
                        )
                        continue
                    fingerprint = candidate_fingerprint(node.code)
                    if fingerprint in self.generated_fingerprints:
                        reason = "duplicate code fingerprint"
                        retry_feedback[role.name].append(reason)
                        self.state.memory.failed_hypotheses.append(
                            f"{role.name}: {reason}"
                        )
                        continue
                    self.generated_fingerprints.add(fingerprint)
                    self.generated_semantic_signatures.add(semantic_signature)
                    accepted[role.name] = node
                    self.candidate_role_by_node_id[node.id] = role
                    self.candidate_origin_by_node_id[node.id] = (stage, journal)
                    print(
                        f"[green]Accepted {role.group}/{role.name} candidate "
                        f"{node.id}[/green]"
                    )
                    factor_change = format_factor_change(contract)
                    if factor_change:
                        print(f"[cyan]Generated factors:\n{factor_change}[/cyan]")
        return [accepted[role.name] for role in roles if role.name in accepted]

    def _manifest_metadata(self, node: Node) -> dict[str, str]:
        manifest = literal_assignment(node.code, "RESEARCH_MANIFEST")
        if not isinstance(manifest, dict):
            return {
                "model_family": model_family_from_code(node.code),
                "research_family": "baseline",
                "loss_family": "pointwise_bce",
                "parent_node_id": "",
                "parent_model_family": "",
                "role": "",
            }
        return {
            name: str(manifest.get(name, ""))
            for name in (
                "model_family",
                "research_family",
                "loss_family",
                "parent_node_id",
                "parent_model_family",
                "role",
            )
        }

    def _register_frontier(
        self,
        node: Node,
        *,
        source_stage: str,
        round_number: int,
        semantic_signature: str,
    ) -> None:
        score = _node_score(node)
        if score is None:
            return
        metadata = self._manifest_metadata(node)
        record = FrontierRecord(
            node_id=node.id,
            source_stage=source_stage,
            round_number=round_number,
            score=score,
            metrics=_node_validation_metrics(node),
            model_family=metadata["model_family"],
            research_family=metadata["research_family"],
            loss_family=metadata["loss_family"],
            parent_node_id=metadata["parent_node_id"],
            parent_model_family=metadata["parent_model_family"],
            fingerprint=candidate_fingerprint(node.code),
            semantic_signature=semantic_signature,
            principal_change=_principal_change(node),
            role=metadata["role"],
        )
        self.frontier_nodes_by_id[node.id] = node
        self.state.memory.add_frontier(
            record,
            limit=self.config.frontier_max_size,
        )

    def _semantic_signature_for_node(self, node: Node) -> str:
        manifest = literal_assignment(node.code, "RESEARCH_MANIFEST")
        selection = literal_assignment(node.code, "FACTOR_SELECTION")
        payload = {
            "model_family": (manifest or {}).get("model_family") if isinstance(manifest, dict) else None,
            "research_family": (manifest or {}).get("research_family") if isinstance(manifest, dict) else None,
            "loss_family": (manifest or {}).get("loss_family") if isinstance(manifest, dict) else None,
            "mechanism_ids": sorted((manifest or {}).get("mechanism_ids", [])) if isinstance(manifest, dict) else [],
            "factor_ids": sorted((selection or {}).get("selected_factor_ids", [])) if isinstance(selection, dict) else [],
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def _generate_assigned_candidates(
        self,
        assignments: Sequence[CandidateAssignment],
        *,
        stage_name: str,
        round_number: int,
        max_attempts: int,
        stage1b: bool = False,
    ) -> list[Node]:
        if not assignments:
            return []
        stage, journal = self._new_stage(
            name=stage_name,
            goals=(
                "Build low-cost, schema-v2-compatible non-FM research roots. "
                "Each root is evidence, not an automatic replacement for the FM baseline."
                if stage1b
                else self.manager.main_stage_goals[3]
            ),
            max_iterations=max_attempts,
        )
        parents_by_id: dict[str, Node] = {}
        for assignment in assignments:
            parents_by_id.setdefault(assignment.parent.id, _clone_root(assignment.parent))
        for parent in parents_by_id.values():
            journal.append(parent)
        normalized = [
            CandidateAssignment(
                role=assignment.role,
                parent=parents_by_id[assignment.parent.id],
                parent_model_family=assignment.parent_model_family,
                source=assignment.source,
            )
            for assignment in assignments
        ]
        base_context = build_factor_context(
            self.manager.cfg.data_dir,
            incumbent_code=normalized[0].parent.code,
            round_number=max(1, round_number),
            failed_hypotheses=self.state.memory.failed_hypotheses,
        )
        base_context += "\n\n" + self.state.experience_prompt(
            before_round=max(1, round_number)
        )
        if stage1b:
            base_context += (
                "\nStage 1B bootstrap constraints:\n"
                f"- Train at most {self.config.stage1b_max_epochs} epochs.\n"
                "- Change exactly one model architecture family and keep pointwise BCE.\n"
                "- A valid root need not beat FM; it enters the diverse frontier for later search.\n"
            )
        accepted: dict[str, Node] = {}
        retry_feedback = {item.role.name: [] for item in normalized}
        role_index = {
            item.role.name: index for index, item in enumerate(normalized, 1)
        }
        seen_node_ids = set(parents_by_id)
        attempts = 0

        def prompt_for(item: CandidateAssignment, index: int) -> str:
            return item.role.prompt(
                index,
                len(normalized),
                retry_feedback=retry_feedback[item.role.name],
                evidence_memory=(
                    "Parent source: " + item.source + "\n"
                    + self.state.memory.prompt_evidence(item.role.group)
                ),
                factor_library_context=factor_library_prompt(
                    role_group=item.role.group,
                    role_category=item.role.category,
                    observed_evidence=self.state.memory.factor_summary(),
                    discovered_cards=self.state.memory.discovered_factor_cards(),
                ),
                parent_node_id=item.parent.id,
                parent_model_family=item.parent_model_family,
            )

        with self._agent(
            stage,
            journal,
            research_base=normalized[0].parent,
            research_bases=[item.parent for item in normalized],
            task_context=base_context,
            candidate_contexts=[
                prompt_for(item, index)
                for index, item in enumerate(normalized, 1)
            ],
        ) as agent:
            while attempts < max_attempts:
                pending = [
                    item for item in normalized if item.role.name not in accepted
                ]
                if not pending:
                    break
                assigned = pending[: min(max_attempts - attempts, len(pending))]
                agent.research_base_nodes = [item.parent for item in assigned]
                agent.research_base_node = assigned[0].parent
                agent.candidate_contexts = [
                    prompt_for(item, role_index[item.role.name]) for item in assigned
                ]
                before = self._experiment_count(journal)
                agent.step(self.exec_callback, max_nodes=len(assigned))
                added = self._experiment_count(journal) - before
                if added <= 0:
                    break
                attempts += added
                self._notify(stage, journal)
                new_nodes = [
                    node for node in journal.nodes if node.id not in seen_node_ids
                ]
                for expected, node in zip(assigned, new_nodes):
                    seen_node_ids.add(node.id)
                    if _node_score(node) is None:
                        retry_feedback[expected.role.name].append("execution failed")
                        continue
                    node.code = normalize_candidate_metadata(
                        node.code,
                        expected.role,
                        expected_parent_id=expected.parent.id,
                        expected_parent_model_family=expected.parent_model_family,
                    )
                    contract = validate_candidate_contract(
                        expected.parent.code,
                        node.code,
                        expected.role,
                        expected_parent_id=expected.parent.id,
                        expected_parent_model_family=expected.parent_model_family,
                    )
                    if not contract.valid:
                        reason = "; ".join(contract.reasons)
                        retry_feedback[expected.role.name].append(reason)
                        self.state.memory.failed_hypotheses.append(
                            f"{expected.role.name}: {reason}"
                        )
                        print(f"[yellow]Rejected candidate: {reason}[/yellow]")
                        continue
                    if stage1b:
                        config = config_assignment(node.code) or {}
                        epochs = config.get("max_epochs", config.get("epochs"))
                        if not isinstance(epochs, int) or epochs > self.config.stage1b_max_epochs:
                            reason = (
                                "Stage 1B CONFIG must use a literal max_epochs/epochs <= "
                                f"{self.config.stage1b_max_epochs}"
                            )
                            retry_feedback[expected.role.name].append(reason)
                            self.state.memory.failed_hypotheses.append(
                                f"{expected.role.name}: {reason}"
                            )
                            continue
                    signature = candidate_semantic_signature(contract)
                    fingerprint = candidate_fingerprint(node.code)
                    if signature in self.generated_semantic_signatures:
                        retry_feedback[expected.role.name].append("semantic duplicate")
                        continue
                    if fingerprint in self.generated_fingerprints:
                        retry_feedback[expected.role.name].append("code duplicate")
                        continue
                    self.generated_semantic_signatures.add(signature)
                    self.generated_fingerprints.add(fingerprint)
                    accepted[expected.role.name] = node
                    self.candidate_role_by_node_id[node.id] = expected.role
                    self.candidate_origin_by_node_id[node.id] = (stage, journal)
                    self.candidate_parent_by_node_id[node.id] = expected.parent
                    self._register_frontier(
                        node,
                        source_stage=stage.name,
                        round_number=round_number,
                        semantic_signature=signature,
                    )
                    if stage1b:
                        self.bootstrap_node_ids.add(node.id)
                    print(
                        f"[green]Accepted {expected.role.name} from "
                        f"{expected.source}/{expected.parent_model_family}: {node.id}[/green]"
                    )
        self._save_memory()
        return [
            accepted[item.role.name]
            for item in normalized
            if item.role.name in accepted
        ]

    def _run_stage1b(self, baseline: Node) -> list[Node]:
        if not self.config.stage1b_enabled:
            return []
        roles = bootstrap_candidate_roles(self.config.stage1b_model_families)
        assignments = [
            CandidateAssignment(role, baseline, "fm", "stage1a_fm")
            for role in roles
        ]
        roots = self._generate_assigned_candidates(
            assignments,
            stage_name="1_diverse_roots_2_schema_v2_bootstrap",
            round_number=0,
            max_attempts=self.config.stage1b_generation_attempts,
            stage1b=True,
        )
        baseline_score = _node_score(baseline)
        baseline_metrics = _node_validation_metrics(baseline)
        for node in roots:
            score = _node_score(node)
            if score is None:
                continue
            role = self.candidate_role_by_node_id.get(node.id)
            metadata = self._manifest_metadata(node)
            metrics = _node_validation_metrics(node)
            artifact = _candidate_artifacts(node)
            (
                considered_factor_ids,
                factor_ids,
                factor_selection_reason,
                factor_rejected_reasons,
                factor_cards,
            ) = _factor_selection(node)
            fingerprint = candidate_fingerprint(
                node.code,
                config={"evaluation_protocol": "stage1b_bootstrap"},
            )
            self.state.memory.register_candidate(fingerprint)
            self.state.memory.record(
                ExperimentRecord(
                    round_number=0,
                    node_id=node.id,
                    fingerprint=fingerprint,
                    score=score,
                    seed_scores=[score],
                    promoted=False,
                    reason=(
                        "bootstrap-only validation result; queued for equal-budget "
                        "Stage 2 tuning and multi-seed standardization"
                    ),
                    metrics=metrics,
                    metric_deltas={
                        name: (
                            float(metrics[name]) - float(baseline_metrics[name])
                            if metrics.get(name) is not None
                            and baseline_metrics.get(name) is not None
                            else None
                        )
                        for name in ("GAUC", "nDCG@5", "primary")
                    },
                    seed_metrics=[metrics],
                    seed_mean=score,
                    seed_std=0.0,
                    principal_change=_principal_change(node),
                    components=_extract_enabled_ablation_components(node.code),
                    role=role.name if role else metadata["role"],
                    category=role.group if role else "architecture_exploration",
                    improvement=(
                        score - baseline_score if baseline_score is not None else 0.0
                    ),
                    factor_ids=factor_ids,
                    considered_factor_ids=considered_factor_ids,
                    factor_selection_reason=factor_selection_reason,
                    factor_rejected_reasons=factor_rejected_reasons,
                    factor_cards=factor_cards,
                    model_family=metadata["model_family"],
                    research_family=metadata["research_family"],
                    loss_family=metadata["loss_family"],
                    parent_node_id=metadata["parent_node_id"],
                    parent_model_family=metadata["parent_model_family"],
                    source_stage="1_diverse_roots_2_schema_v2_bootstrap",
                    source_phase="stage1b_bootstrap",
                    status="bootstrap_pending_standardization",
                    code_path=artifact["code_path"],
                    checkpoint_path=artifact["checkpoint_path"],
                    validation_split=artifact["validation_split"],
                    training_epochs=artifact["training_epochs"],
                    artifacts_valid=bool(artifact["valid"]),
                )
            )
        self._save_memory()
        return roots

    def _stage3_parent_pool(self) -> list[tuple[Node, str]]:
        if self.incumbent is None:
            raise RuntimeError("Stage 3 requires an incumbent")
        pool: list[tuple[Node, str]] = []
        pool.extend(
            (self.incumbent.node, "incumbent")
            for _ in range(max(0, self.config.stage3_incumbent_parent_slots))
        )
        used_frontier_nodes: set[str] = set()

        def select_records(records, count):
            selected = []
            seen_families = set()
            for record in sorted(records, key=lambda item: item.score, reverse=True):
                node = self.frontier_nodes_by_id.get(record.node_id)
                if (
                    node is None
                    or record.node_id in used_frontier_nodes
                    or record.model_family in seen_families
                ):
                    continue
                selected.append((node, record.source_stage))
                seen_families.add(record.model_family)
                used_frontier_nodes.add(record.node_id)
                if len(selected) >= count:
                    break
            return selected

        non_bootstrap = [
            item for item in self.state.memory.frontier
            if item.node_id not in self.bootstrap_node_ids
        ]
        pool.extend(
            select_records(
                non_bootstrap or self.state.memory.frontier,
                max(0, self.config.stage3_frontier_parent_slots),
            )
        )
        bootstrap = [
            item for item in self.state.memory.frontier
            if item.node_id in self.bootstrap_node_ids
        ]
        pool.extend(
            select_records(
                bootstrap,
                max(0, self.config.stage3_bootstrap_parent_slots),
            )
        )
        all_frontier = select_records(
            self.state.memory.frontier,
            self.config.candidate_branches,
        )
        fill_index = 0
        while len(pool) < self.config.candidate_branches:
            if all_frontier:
                pool.append(all_frontier[fill_index % len(all_frontier)])
                fill_index += 1
            else:
                pool.append((self.incumbent.node, "incumbent_fallback"))
        return pool[: self.config.candidate_branches]

    def _generate_candidates(self, incumbent: Node, round_number: int) -> list[Node]:
        roles = select_candidate_roles(
            self.state.memory.direction_summary(),
            round_number=round_number,
            branch_count=self.config.candidate_branches,
            preferred_role_names=self.config.initial_candidate_roles,
            reserved_role_name=self.config.reserved_candidate_role,
        )
        parents = self._stage3_parent_pool()
        assignments = []
        for role, (parent, source) in zip(roles, parents):
            parent_family = model_family_from_code(parent.code)
            constrained_role = replace(
                role,
                allowed_model_families=(parent_family,),
            )
            assignments.append(
                CandidateAssignment(
                    constrained_role,
                    parent,
                    parent_family,
                    source,
                )
            )
        return self._generate_assigned_candidates(
            assignments,
            stage_name=(
                f"3_creative_research_1_round_{self._word(round_number)}_multi_parent"
            ),
            round_number=round_number,
            max_attempts=self.config.stage3_generation_attempts,
        )

    def _prepare_candidate_tuning(
        self,
        candidate: Node,
        round_number: int,
        branch_number: int,
        *,
        phase: str,
        budget: int,
        role: CandidateRole,
        candidate_id: str,
    ) -> tuple[Any, Journal, Node, int, CandidateRole, str]:
        substage = 2 if phase == "targeted" else 3
        stage, journal = self._new_stage(
            name=(
                f"3_candidate_tuning_{substage}_{phase}_round_"
                f"{self._word(round_number)}_branch_{self._word(branch_number)}"
            ),
            goals=(
                "Keep this candidate architecture, feature builder, loss, and "
                "split fixed; tune one complete CONFIG per node. "
                + (
                    "Make one evidence-based correction to the most sensitive new parameter."
                    if phase == "targeted"
                    else (
                        "Concentrate on candidate-specific parameters such as factor fusion, "
                        "auxiliary-loss weight, temperature, or regularization. Inherit Stage 2 "
                        "backbone parameters unless the training trace shows a concrete problem."
                    )
                )
            ),
            max_iterations=budget,
        )
        tuning_base = _clone_root(candidate)
        journal.append(tuning_base)
        return stage, journal, tuning_base, budget, role, candidate_id

    def _execute_candidate_tuning(
        self,
        stage: Any,
        journal: Journal,
        tuning_base: Node,
        budget: int,
        role: CandidateRole,
        candidate_id: str,
    ) -> Optional[EvaluatedConfiguration]:
        # Each branch gets one worker so progressive tuning remains serial
        # inside the branch while independent A/B/C branches run concurrently.
        with self._agent(
            stage,
            journal,
            tuning_base=tuning_base,
            num_workers=1,
        ) as agent:
            self._run_budget(
                agent,
                stage,
                journal,
                budget,
                inherited_nodes=1,
            )
        best = _ranked_nodes(journal, 1)
        if not best:
            return None
        score = _node_score(best[0])
        if score is None:
            return None
        return EvaluatedConfiguration(
            node=best[0],
            journal=journal,
            stage_name=stage.name,
            score=score,
            seed_scores=[],
            metrics=_node_validation_metrics(best[0]),
            metric_means=_node_metrics(best[0]),
            role=role,
            candidate_id=candidate_id,
        )

    def _execute_tuning_jobs(
        self,
        jobs: Sequence[tuple[Any, Journal, Node, int, CandidateRole, str]],
    ) -> list[Optional[EvaluatedConfiguration]]:
        if not jobs:
            return []
        workers = min(
            max(1, int(self.config.candidate_parallel_workers)),
            len(jobs),
        )
        if workers == 1:
            return [self._execute_candidate_tuning(*job) for job in jobs]
        print(
            f"[cyan]Tuning {len(jobs)} candidate branches with "
            f"{workers} parallel workers[/cyan]"
        )
        stagger_delay = max(
            0.0, float(self.config.stage3_startup_stagger_seconds)
        )
        should_stagger = stagger_delay > 0 and get_gpu_count() == 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for index, job in enumerate(jobs):
                futures.append(executor.submit(self._execute_candidate_tuning, *job))
                if should_stagger and index < len(jobs) - 1:
                    print(
                        f"[cyan]Staggering next Stage 3 candidate start by "
                        f"{stagger_delay:.1f}s; active candidates keep running "
                        "in parallel[/cyan]"
                    )
                    time.sleep(stagger_delay)
            # Consume futures in branch order so promotion semantics do not
            # depend on which candidate happens to finish first.
            return [future.result() for future in futures]

    def _tune_candidates(
        self,
        candidates: Sequence[Node],
        round_number: int,
    ) -> list[EvaluatedConfiguration]:
        screened: list[EvaluatedConfiguration] = []
        for candidate in candidates:
            role = self.candidate_role_by_node_id.get(candidate.id)
            origin = self.candidate_origin_by_node_id.get(candidate.id)
            score = _node_score(candidate)
            if role is None or origin is None or score is None:
                continue
            stage, journal = origin
            screened.append(
                EvaluatedConfiguration(
                    node=candidate,
                    journal=journal,
                    stage_name=stage.name,
                    score=score,
                    seed_scores=[],
                    metrics=_node_validation_metrics(candidate),
                    metric_means=_node_metrics(candidate),
                    role=role,
                    candidate_id=candidate.id,
                )
            )
        if not screened:
            return []
        screened.sort(key=lambda result: result.score, reverse=True)

        targeted_bases = screened[: self.config.candidate_tuning_top_k]
        targeted_jobs = [
            self._prepare_candidate_tuning(
                result.node,
                round_number,
                branch_number,
                phase="targeted",
                budget=self.config.candidate_tuning_iterations,
                role=result.role,
                candidate_id=result.candidate_id,
            )
            for branch_number, result in enumerate(targeted_bases, 1)
            if result.role is not None and result.candidate_id is not None
        ]
        targeted = {
            result.candidate_id: result
            for result in self._execute_tuning_jobs(targeted_jobs)
            if result is not None and result.candidate_id is not None
        }
        after_targeted = [
            targeted.get(result.candidate_id, result) for result in screened
        ]
        after_targeted.sort(key=lambda result: result.score, reverse=True)

        refinement_bases = after_targeted[
            : self.config.candidate_refinement_top_k
        ]
        refinement_jobs = [
            self._prepare_candidate_tuning(
                result.node,
                round_number,
                branch_number,
                phase="top1",
                budget=self.config.candidate_refinement_iterations,
                role=result.role,
                candidate_id=result.candidate_id,
            )
            for branch_number, result in enumerate(refinement_bases, 1)
            if result.role is not None and result.candidate_id is not None
        ]
        refined = {
            result.candidate_id: result
            for result in self._execute_tuning_jobs(refinement_jobs)
            if result is not None and result.candidate_id is not None
        }
        merged = [
            refined.get(result.candidate_id, result) for result in after_targeted
        ]
        merged.sort(key=lambda result: result.score, reverse=True)
        finalists = merged[: self.config.candidate_finalist_top_k]

        confirmed: dict[str, EvaluatedConfiguration] = {}
        for result in finalists:
            stage = next(
                stage for stage in self.manager.stages
                if stage.name == result.stage_name
            )
            with self._agent(
                stage,
                result.journal,
                tuning_base=result.node,
                num_workers=self.config.finalist_num_seeds,
            ) as agent:
                repeated = self._evaluate_nodes(
                    agent,
                    stage,
                    result.journal,
                    [result.node],
                    role=result.role,
                    candidate_id=result.candidate_id,
                )
            if repeated and result.candidate_id is not None:
                confirmed[result.candidate_id] = repeated[0]

        results = [
            confirmed.get(result.candidate_id, result) for result in merged
        ]
        results = sorted(results, key=lambda result: result.score, reverse=True)
        for result in results:
            self._register_frontier(
                result.node,
                source_stage=result.stage_name,
                round_number=round_number,
                semantic_signature=self._semantic_signature_for_node(result.node),
            )
        return results

    def _run_stage4(
        self,
        promoted: EvaluatedConfiguration,
        round_number: int,
        candidate_number: int,
    ) -> Optional[EvaluatedConfiguration]:
        components = _extract_enabled_ablation_components(promoted.node.code)[
            : self.max_ablation_components
        ]
        guarded = component_guard_calls(promoted.node.code)
        components = [component for component in components if component in guarded]
        if not components:
            return promoted
        stage, journal = self._new_stage(
            name=(
                f"4_ablation_studies_1_round_{self._word(round_number)}_"
                f"candidate_{self._word(candidate_number)}"
            ),
            goals=self.manager.main_stage_goals[4],
            max_iterations=len(components),
        )
        full_model = _clone_root(promoted.node)
        journal.append(full_model)
        with self._agent(
            stage,
            journal,
            stage3_base=full_model,
        ) as agent:
            self._run_budget(
                agent,
                stage,
                journal,
                len(components),
                inherited_nodes=1,
            )
            single_nodes = [
                node
                for node in journal.nodes
                if node.ablation_name in components and _node_score(node) is not None
            ]
            results = self._evaluate_nodes(
                agent,
                stage,
                journal,
                [full_model, *single_nodes],
                role=promoted.role,
                candidate_id=promoted.candidate_id,
            )
            if not results:
                return None
            full_result = next(
                (result for result in results if result.node.ablation_name is None),
                None,
            )
            if full_result is None:
                return None

            single_results = {
                result.node.ablation_name: result
                for result in results
                if result.node.ablation_name in components
            }
            ranked_components = sorted(
                single_results,
                key=lambda component: abs(
                    full_result.score - single_results[component].score
                ),
                reverse=True,
            )
            important = ranked_components[: min(4, len(ranked_components))]
            pair_specs = list(combinations(important, 2))[
                : self.config.ablation_synergy_pairs
            ]
            pair_nodes = [
                _joint_ablation_node(full_model, pair) for pair in pair_specs
            ]
            for node in pair_nodes:
                journal.append(node)
            pair_results = self._evaluate_nodes(
                agent,
                stage,
                journal,
                pair_nodes,
                role=promoted.role,
                candidate_id=promoted.candidate_id,
            )

        category = promoted.role.group if promoted.role else ""
        full_metrics = full_result.metric_means
        for component, result in single_results.items():
            contribution = full_result.score - result.score
            seed_wins = sum(
                full > ablated
                for full, ablated in zip(
                    full_result.seed_scores,
                    result.seed_scores,
                )
            )
            verdict = (
                "positive"
                if contribution > 0.0002
                else "harmful"
                if contribution < -0.0002
                else "neutral"
            )
            self.state.memory.record_ablation(
                AblationEvidence(
                    round_number=round_number,
                    candidate_id=promoted.candidate_id or promoted.node.id,
                    component=component,
                    category=category,
                    full_score=full_result.score,
                    ablated_score=result.score,
                    primary_contribution=contribution,
                    seed_wins=seed_wins,
                    verdict=verdict,
                    gauc_contribution=(
                        full_metrics["GAUC"] - result.metric_means["GAUC"]
                        if "GAUC" in full_metrics and "GAUC" in result.metric_means
                        else None
                    ),
                    ndcg5_contribution=(
                        full_metrics["nDCG@5"] - result.metric_means["nDCG@5"]
                        if "nDCG@5" in full_metrics
                        and "nDCG@5" in result.metric_means
                        else None
                    ),
                )
            )

        for result in pair_results:
            pair = tuple(
                component
                for component in (result.node.ablation_name or "").split("+")
                if component in single_results
            )
            if len(pair) != 2:
                continue
            contributions = [
                full_result.score - single_results[item].score for item in pair
            ]
            joint_contribution = full_result.score - result.score
            synergy = joint_contribution - sum(contributions)
            self.state.memory.record_ablation(
                AblationEvidence(
                    round_number=round_number,
                    candidate_id=promoted.candidate_id or promoted.node.id,
                    component="+".join(pair),
                    category=category,
                    full_score=full_result.score,
                    ablated_score=result.score,
                    primary_contribution=joint_contribution,
                    seed_wins=sum(
                        full > ablated
                        for full, ablated in zip(
                            full_result.seed_scores,
                            result.seed_scores,
                        )
                    ),
                    verdict=(
                        "positive_synergy"
                        if synergy > 0.0002
                        else "negative_synergy"
                        if synergy < -0.0002
                        else "additive"
                    ),
                    interaction_with=list(pair),
                    synergy=synergy,
                )
            )

        all_results = [*results, *pair_results]
        return max(all_results, key=lambda result: result.score)

    def _confirm_candidate(
        self,
        candidate: EvaluatedConfiguration,
        round_number: int,
        candidate_number: int,
        *,
        stage_name: Optional[str] = None,
    ) -> Optional[EvaluatedConfiguration]:
        stable = _materialize_ablation_winner(candidate.node)
        if stable.metric is None:
            stable.metric = next(
                (
                    copy.deepcopy(child.metric)
                    for child in candidate.node.children
                    if child.is_seed_node and _node_score(child) is not None
                ),
                None,
            )
        stage, journal = self._new_stage(
            name=stage_name
            or (
                f"4_final_confirmation_2_round_{self._word(round_number)}_"
                f"candidate_{self._word(candidate_number)}"
            ),
            goals=(
                "Run the frozen validation-best candidate across five independent "
                "seeds. Do not change code or CONFIG."
            ),
            max_iterations=0,
        )
        journal.append(stable)
        with self._agent(
            stage,
            journal,
            stage3_base=stable,
            num_workers=self.config.candidate_parallel_workers,
        ) as agent:
            results = self._evaluate_nodes(
                agent,
                stage,
                journal,
                [stable],
                num_seeds=self.config.final_confirmation_num_seeds,
                role=candidate.role,
                candidate_id=candidate.candidate_id,
            )
        return results[0] if results else None

    def _ensure_final_confirmation(self) -> None:
        if self.incumbent is None:
            raise RuntimeError("cannot confirm without an incumbent")
        required = self.config.final_confirmation_num_seeds
        if len(self.incumbent.seed_scores) >= required:
            return

        confirmed = self._confirm_candidate(
            self.incumbent,
            round_number=max(self.state.current_round, 1),
            candidate_number=1,
            stage_name="4_final_incumbent_confirmation_1_configured_seeds",
        )
        if confirmed is None or len(confirmed.seed_scores) != required:
            raise RuntimeError(
                "final incumbent did not complete the configured seed confirmation"
            )
        self.incumbent = confirmed
        self.state.set_incumbent(
            confirmed.node.id,
            confirmed.score,
            confirmed.seed_scores,
            metrics=confirmed.metrics,
            source_stage=confirmed.stage_name,
        )
        self._save_memory()

    def _save_memory(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "policy": asdict(self.state.policy),
            "current_round": self.state.current_round,
            "no_improvement_rounds": self.state.no_improvement_rounds,
            "incumbent_node_id": self.state.incumbent_node_id,
            "incumbent_score": self.state.incumbent_score,
            "incumbent_seed_scores": self.state.incumbent_seed_scores,
            "incumbent_metrics": self.state.incumbent_metrics,
            "incumbent_source_stage": self.state.incumbent_source_stage,
            "verified_incumbent_id": self.state.verified_incumbent_id,
            "provisional_candidate_id": self.state.provisional_candidate_id,
            "provisional_candidate_score": self.state.provisional_candidate_score,
            "memory": self.state.memory.to_dict(),
        }
        (self.memory_dir / "state.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        (self.memory_dir / "diverse_frontier.json").write_text(
            json.dumps(
                [asdict(item) for item in self.state.memory.frontier],
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _save_round_summary(self, round_number: int) -> None:
        summary = self.state.round_summary(round_number)
        archive_dir = self.memory_dir / "round_summaries"
        archive_dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        (archive_dir / f"round_{round_number:03d}.json").write_text(serialized)
        (self.memory_dir / "round_summary.json").write_text(serialized)
        prompt = self.state.experience_prompt(before_round=round_number + 1)
        (self.memory_dir / "round_summary.prompt.txt").write_text(prompt + "\n")

    def _finalize(self) -> dict[str, Any]:
        if self.incumbent is None:
            raise RuntimeError("cannot finalize without an incumbent")
        return self.finalizer(
            self.incumbent.journal,
            output_dir=self.output_dir,
            source_stage=self.incumbent.stage_name,
            required_seeds=self.config.final_confirmation_num_seeds,
        )

    @staticmethod
    def _word(number: int) -> str:
        words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
        }
        return words.get(number, f"index_{number}")

    def run(self) -> dict[str, Any]:
        budget = estimate_max_experiment_runs(
            self.config,
            stage4_max_components=self.max_ablation_components,
        )
        print(f"[cyan]Closed-loop maximum execution budget: {budget}[/cyan]")

        baseline = self._run_stage1()
        stage1b_roots = self._run_stage1b(baseline)
        self.incumbent = self._run_stage2(baseline, stage1b_roots)
        self._save_memory()

        while not self.state.should_stop:
            round_number = self.state.start_round()
            print(f"[green]Starting research round {round_number}[/green]")
            candidates = self._generate_candidates(
                self.incumbent.node, round_number
            )
            tuned_results = self._tune_candidates(candidates, round_number)
            promoted_results: dict[str, EvaluatedConfiguration] = {}
            for result in tuned_results:
                candidate_id = result.candidate_id or result.node.id
                role = result.role
                (
                    considered_factor_ids,
                    factor_ids,
                    factor_selection_reason,
                    factor_rejected_reasons,
                    factor_cards,
                ) = _factor_selection(result.node)
                artifact = _candidate_artifacts(result.node)
                decision = self.state.evaluate_candidate(
                    node_id=candidate_id,
                    fingerprint=candidate_fingerprint(result.node.code),
                    score=result.score,
                    seed_scores=result.seed_scores,
                    metrics=result.metrics,
                    seed_metrics=result.seed_metrics,
                    principal_change=_principal_change(result.node),
                    components=_extract_enabled_ablation_components(
                        result.node.code
                    ),
                    role=role.name if role else "",
                    category=role.group if role else "",
                    factor_ids=factor_ids,
                    considered_factor_ids=considered_factor_ids,
                    factor_selection_reason=factor_selection_reason,
                    factor_rejected_reasons=factor_rejected_reasons,
                    factor_cards=factor_cards,
                    model_family=self._manifest_metadata(result.node)["model_family"],
                    research_family=self._manifest_metadata(result.node)["research_family"],
                    loss_family=self._manifest_metadata(result.node)["loss_family"],
                    parent_node_id=self._manifest_metadata(result.node)["parent_node_id"],
                    parent_model_family=self._manifest_metadata(result.node)["parent_model_family"],
                    source_stage=result.stage_name,
                    source_phase="stage3",
                    code_path=artifact["code_path"],
                    checkpoint_path=artifact["checkpoint_path"],
                    validation_split=artifact["validation_split"],
                    random_seeds=range(len(result.seed_scores)),
                    training_epochs=artifact["training_epochs"],
                    artifacts_valid=bool(artifact["valid"]),
                )
                print(
                    f"[cyan]Candidate {candidate_id}: "
                    f"{decision.reason}[/cyan]"
                )
                if decision.promoted:
                    promoted_results[candidate_id] = result

            promoted = sorted(
                promoted_results.values(),
                key=lambda result: result.score,
                reverse=True,
            )[: self.config.ablation_candidate_top_k]
            stage4_results = [
                result
                for candidate_number, promoted_result in enumerate(promoted, 1)
                if (
                    result := self._run_stage4(
                        promoted_result,
                        round_number,
                        candidate_number,
                    )
                )
                is not None
            ]
            confirmed = [
                result
                for candidate_number, stage4_result in enumerate(
                    stage4_results, 1
                )
                if (
                    result := self._confirm_candidate(
                        stage4_result,
                        round_number,
                        candidate_number,
                    )
                )
                is not None
            ]
            confirmed.sort(key=lambda result: result.score, reverse=True)
            for result in confirmed:
                candidate_id = result.candidate_id or result.node.id
                confirmation = self.state.accept_pending_candidate(
                    node_id=candidate_id,
                    final_node_id=result.node.id,
                    score=result.score,
                    seed_scores=result.seed_scores,
                    metrics=result.metrics,
                )
                print(
                    f"[cyan]Final confirmation for {candidate_id}: "
                    f"{confirmation.reason}[/cyan]"
                )
                if confirmation.promoted:
                    self.incumbent = result
                    break

            self.state.finish_round()
            self._save_round_summary(round_number)
            self._save_memory()
            self.manager._save_checkpoint()

        self._ensure_final_confirmation()
        manifest = self._finalize()
        submission = self.submission_exporter(
            output_dir=self.output_dir,
            starter_kit=Path(self.manager.cfg.data_dir).resolve(),
        )
        self._save_memory()
        self.manager.current_stage = None
        return {
            "manifest": manifest,
            "submission": submission,
            "state": self.state,
            "budget": budget,
            "incumbent": self.incumbent,
        }
