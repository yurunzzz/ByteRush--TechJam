"""Deterministic Stage 1-4 orchestration for the ByteRush research loop."""
from __future__ import annotations

import ast
import copy
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional, Sequence

from rich import print

from .factor_context import build_factor_context
from .finalize import freeze_stage4_winner
from .journal import Journal, Node
from .parallel_agent import (
    ParallelAgent,
    _extract_enabled_ablation_components,
    _load_kuairand_validation_metric,
    get_gpu_count,
)
from .research_loop import (
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
    metrics: dict[str, Optional[float]]
    seed_metrics: list[dict[str, Optional[float]]]


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

    component = node.ablation_name
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
            if isinstance(manifest, dict) and component in manifest:
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
        self.memory_dir = self.output_dir.parent / "research_loop"
        self.generated_fingerprints: set[str] = set()
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
    ) -> list[EvaluatedConfiguration]:
        evaluated = []
        required = self.config.finalist_num_seeds
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
        candidates = []
        attempts = 0
        seen_node_ids = {research_base.id}
        with self._agent(
            stage,
            journal,
            research_base=research_base,
            task_context=context,
        ) as agent:
            while (
                attempts < self.config.stage3_generation_attempts
                and len(candidates) < self.config.candidate_branches
            ):
                before = self._experiment_count(journal)
                remaining_attempts = self.config.stage3_generation_attempts - attempts
                remaining_candidates = self.config.candidate_branches - len(candidates)
                agent.step(
                    self.exec_callback,
                    max_nodes=min(remaining_attempts, remaining_candidates),
                )
                after = self._experiment_count(journal)
                added = after - before
                if added <= 0:
                    break
                attempts += added
                self._notify(stage, journal)
                for node in journal.nodes:
                    if node.id in seen_node_ids:
                        continue
                    seen_node_ids.add(node.id)
                    if _node_score(node) is None:
                        continue
                    fingerprint = candidate_fingerprint(node.code)
                    if fingerprint in self.generated_fingerprints:
                        continue
                    components = _extract_enabled_ablation_components(node.code)
                    if not components:
                        self.state.memory.failed_hypotheses.append(
                            f"node {node.id} omitted ABLATION_COMPONENTS"
                        )
                        continue
                    self.generated_fingerprints.add(fingerprint)
                    candidates.append(node)
                    if len(candidates) >= self.config.candidate_branches:
                        break
        return candidates

    def _prepare_candidate_tuning(
        self,
        candidate: Node,
        round_number: int,
        branch_number: int,
    ) -> tuple[Any, Journal, Node]:
        stage, journal = self._new_stage(
            name=(
                "3_candidate_tuning_2_round_"
                f"{self._word(round_number)}_branch_{self._word(branch_number)}"
            ),
            goals=(
                "Keep this candidate architecture, feature builder, loss, and "
                "split fixed; tune one complete configuration per node."
            ),
            max_iterations=self.config.candidate_tuning_iterations,
        )
        tuning_base = _clone_root(candidate)
        journal.append(tuning_base)
        return stage, journal, tuning_base

    def _execute_candidate_tuning(
        self,
        stage: Any,
        journal: Journal,
        tuning_base: Node,
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
                self.config.candidate_tuning_iterations,
                inherited_nodes=1,
            )
            return self._evaluate_top_k(agent, stage, journal)

    def _tune_candidate(
        self,
        candidate: Node,
        round_number: int,
        branch_number: int,
    ) -> Optional[EvaluatedConfiguration]:
        job = self._prepare_candidate_tuning(
            candidate,
            round_number,
            branch_number,
        )
        return self._execute_candidate_tuning(*job)

    def _tune_candidates(
        self,
        candidates: Sequence[Node],
        round_number: int,
    ) -> list[Optional[EvaluatedConfiguration]]:
        jobs = []
        for branch_number, candidate in enumerate(candidates, 1):
            jobs.append(
                self._prepare_candidate_tuning(
                    candidate, round_number, branch_number
                )
            )
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

    def _run_stage4(
        self,
        promoted: EvaluatedConfiguration,
        round_number: int,
    ) -> Optional[EvaluatedConfiguration]:
        components = _extract_enabled_ablation_components(promoted.node.code)[
            : self.max_ablation_components
        ]
        if not components:
            return None
        stage, journal = self._new_stage(
            name=f"4_ablation_studies_1_round_{self._word(round_number)}",
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
            variants = _ranked_nodes(journal)
            results = self._evaluate_nodes(agent, stage, journal, variants)
        return max(results, key=lambda result: result.score) if results else None

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
            "memory": self.state.memory.to_dict(),
        }
        (self.memory_dir / "state.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
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
            required_seeds=self.config.finalist_num_seeds,
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
            metrics=self.incumbent.metrics,
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
                if result is None:
                    continue
                decision = self.state.evaluate_candidate(
                    node_id=result.node.id,
                    fingerprint=candidate_fingerprint(result.node.code),
                    score=result.score,
                    seed_scores=result.seed_scores,
                    metrics=result.metrics,
                    seed_metrics=result.seed_metrics,
                    principal_change=_principal_change(result.node),
                    components=_extract_enabled_ablation_components(
                        result.node.code
                    ),
                )
                print(
                    f"[cyan]Candidate {result.node.id}: "
                    f"{decision.reason}[/cyan]"
                )
                if decision.promoted:
                    promoted_results[result.node.id] = result

            pending_id = self.state.pending_candidate_id
            if pending_id and pending_id in promoted_results:
                stage4_winner = self._run_stage4(
                    promoted_results[pending_id], round_number
                )
                if stage4_winner is not None:
                    next_parent = _materialize_ablation_winner(
                        stage4_winner.node
                    )
                    confirmation = self.state.accept_pending_candidate(
                        node_id=pending_id,
                        final_node_id=next_parent.id,
                        score=stage4_winner.score,
                        seed_scores=stage4_winner.seed_scores,
                        metrics=stage4_winner.metrics,
                    )
                    print(
                        f"[cyan]Stage 4 confirmation for {pending_id}: "
                        f"{confirmation.reason}[/cyan]"
                    )
                    if confirmation.promoted:
                        self.incumbent = EvaluatedConfiguration(
                            node=next_parent,
                            journal=stage4_winner.journal,
                            stage_name=stage4_winner.stage_name,
                            score=stage4_winner.score,
                            seed_scores=stage4_winner.seed_scores,
                            metrics=stage4_winner.metrics,
                            seed_metrics=stage4_winner.seed_metrics,
                        )

            self.state.finish_round()
            self._save_round_summary(round_number)
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
