"""Deterministic Stage 1-4 orchestration for the ByteRush research loop."""
from __future__ import annotations

import ast
import copy
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional, Sequence

from rich import print

from .candidate_contract import (
    CandidateRole,
    candidate_semantic_signature,
    component_guard_calls,
    format_factor_change,
    literal_assignment,
    validate_candidate_contract,
    select_candidate_roles,
)
from .factor_context import build_factor_context
from .finalize import freeze_stage4_winner
from .journal import Journal, Node
from .parallel_agent import ParallelAgent, _extract_enabled_ablation_components
from .research_loop import (
    AblationEvidence,
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
    metric_means: dict[str, float] = field(default_factory=dict)
    role: Optional[CandidateRole] = None
    candidate_id: Optional[str] = None


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


def _node_metrics(node: Node) -> dict[str, float]:
    """Read trusted validation metrics already attached to a completed node."""
    metrics: dict[str, float] = {}
    primary = _node_score(node)
    if primary is not None:
        metrics["primary"] = primary
    output = getattr(node, "parse_term_out", "")
    if isinstance(output, (list, tuple)):
        output = "\n".join(map(str, output))
    for name, pattern in {
        "GAUC": r"validation\s+GAUC\s*:\s*([-+0-9.eE]+)",
        "nDCG@5": r"validation\s+nDCG@5\s*:\s*([-+0-9.eE]+)",
    }.items():
        match = re.search(pattern, str(output), flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if math.isfinite(value):
                metrics[name] = value
    return metrics


def _mean_metrics(nodes: Sequence[Node]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for node in nodes:
        for name, value in _node_metrics(node).items():
            values.setdefault(name, []).append(value)
    return {name: mean(items) for name, items in values.items() if items}


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
            evaluated.append(
                EvaluatedConfiguration(
                    node=node,
                    journal=journal,
                    stage_name=stage.name,
                    score=mean(seed_scores),
                    seed_scores=seed_scores,
                    metric_means=_mean_metrics(seed_nodes),
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
    ) -> Optional[EvaluatedConfiguration]:
        finalists = _ranked_nodes(journal, self.config.finalist_top_k)
        results = self._evaluate_nodes(agent, stage, journal, finalists)
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

    def _run_stage2(self, baseline: Node) -> EvaluatedConfiguration:
        stage, journal = self._new_stage(
            name="2_baseline_tuning_1_closed_loop",
            goals=self.manager.main_stage_goals[2],
            max_iterations=self.config.baseline_tuning_iterations,
        )
        tuning_base = _clone_root(baseline)
        journal.append(tuning_base)
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
            winner = self._evaluate_top_k(agent, stage, journal)
        if winner is None:
            raise RuntimeError("Stage 2 produced no configuration with three valid seeds")
        return winner

    def _generate_candidates(
        self, incumbent: Node, round_number: int
    ) -> list[Node]:
        roles = select_candidate_roles(
            self.state.memory.direction_summary(),
            round_number=round_number,
            branch_count=self.config.candidate_branches,
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
                    print(
                        f"[green]Accepted {role.group}/{role.name} candidate "
                        f"{node.id}[/green]"
                    )
                    factor_change = format_factor_change(contract)
                    if factor_change:
                        print(f"[cyan]Generated factors:\n{factor_change}[/cyan]")
        return [accepted[role.name] for role in roles if role.name in accepted]

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
        substage = 2 if phase == "coarse" else 3
        stage, journal = self._new_stage(
            name=(
                f"3_candidate_tuning_{substage}_{phase}_round_"
                f"{self._word(round_number)}_branch_{self._word(branch_number)}"
            ),
            goals=(
                "Keep this candidate architecture, feature builder, loss, and "
                "split fixed; tune one complete CONFIG per node. "
                + (
                    "Cover the plausible range broadly."
                    if phase == "coarse"
                    else "Refine near the strongest coarse configuration."
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._execute_candidate_tuning, *job)
                for job in jobs
            ]
            return [future.result() for future in futures]

    def _tune_candidates(
        self,
        candidates: Sequence[Node],
        round_number: int,
    ) -> list[EvaluatedConfiguration]:
        coarse_jobs = []
        for branch_number, candidate in enumerate(candidates, 1):
            role = self.candidate_role_by_node_id.get(candidate.id)
            if role is None:
                continue
            coarse_jobs.append(
                self._prepare_candidate_tuning(
                    candidate,
                    round_number,
                    branch_number,
                    phase="coarse",
                    budget=self.config.candidate_tuning_iterations,
                    role=role,
                    candidate_id=candidate.id,
                )
            )
        coarse_results = [
            result
            for result in self._execute_tuning_jobs(coarse_jobs)
            if result is not None
        ]
        if not coarse_results:
            return []
        coarse_results.sort(key=lambda result: result.score, reverse=True)
        refinement_bases = coarse_results[
            : self.config.candidate_refinement_top_k
        ]
        refinement_jobs = [
            self._prepare_candidate_tuning(
                result.node,
                round_number,
                branch_number,
                phase="refinement",
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
        merged = [refined.get(result.candidate_id, result) for result in coarse_results]
        merged.sort(key=lambda result: result.score, reverse=True)
        finalists = merged[: self.config.candidate_finalist_top_k]

        evaluated: list[EvaluatedConfiguration] = []
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
            evaluated.extend(repeated)
        return sorted(evaluated, key=lambda result: result.score, reverse=True)

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
            name=(
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
            "memory": self.state.memory.to_dict(),
        }
        (self.memory_dir / "state.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

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
        self.incumbent = self._run_stage2(baseline)
        self.state.set_incumbent(
            self.incumbent.node.id,
            self.incumbent.score,
            self.incumbent.seed_scores,
        )
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
                decision = self.state.evaluate_candidate(
                    node_id=candidate_id,
                    fingerprint=candidate_fingerprint(result.node.code),
                    score=result.score,
                    seed_scores=result.seed_scores,
                    role=role.name if role else "",
                    category=role.group if role else "",
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
                )
                print(
                    f"[cyan]Final confirmation for {candidate_id}: "
                    f"{confirmation.reason}[/cyan]"
                )
                if confirmation.promoted:
                    self.incumbent = result
                    break

            self.state.finish_round()
            self._save_memory()
            self.manager._save_checkpoint()

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
