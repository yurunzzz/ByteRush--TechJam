from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence


def _normalized_metric_name(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    source: str
    components: Mapping[str, float] = field(default_factory=dict)


def _metric_payload(metric_or_node: Any) -> Any:
    metric = getattr(metric_or_node, "metric", metric_or_node)
    return getattr(metric, "value", metric)


def _collect_named_metrics(payload: Any) -> dict[str, float]:
    named: dict[str, list[float]] = {}

    if not isinstance(payload, Mapping):
        return {}

    metric_entries = payload.get("metric_names")
    if isinstance(metric_entries, list):
        for entry in metric_entries:
            if not isinstance(entry, Mapping):
                continue
            metric_name = _normalized_metric_name(entry.get("metric_name", ""))
            if not metric_name:
                continue
            values = []
            for data_point in entry.get("data", []):
                if not isinstance(data_point, Mapping):
                    continue
                value = _finite_float(data_point.get("best_value"))
                if value is None:
                    value = _finite_float(data_point.get("final_value"))
                if value is not None:
                    values.append(value)
            if values:
                named.setdefault(metric_name, []).extend(values)

    for key, raw_value in payload.items():
        if key == "metric_names":
            continue
        value = _finite_float(raw_value)
        if value is not None:
            named.setdefault(_normalized_metric_name(key), []).append(value)

    return {name: mean(values) for name, values in named.items() if values}


def extract_primary_score(
    metric_or_node: Any,
    *,
    primary_name: str = "primary",
    component_names: Sequence[str] = ("GAUC", "nDCG@5"),
    allow_fallback: bool = True,
) -> Optional[ScoreBreakdown]:
    """Extract the deterministic validation score used by the research loop.

    ByteRush uses ``primary = (GAUC + nDCG@5) / 2``. The fallback keeps the
    generic AI-Scientist workflow compatible with tasks that expose one metric.
    """

    payload = _metric_payload(metric_or_node)
    scalar = _finite_float(payload)
    if scalar is not None:
        return ScoreBreakdown(scalar, "scalar_metric")

    named = _collect_named_metrics(payload)
    if not named:
        return None

    primary_key = _normalized_metric_name(primary_name)
    if primary_key in named:
        value = named[primary_key]
        return ScoreBreakdown(value, primary_name, {primary_name: value})

    component_keys = [_normalized_metric_name(name) for name in component_names]
    if component_keys and all(key in named for key in component_keys):
        components = {
            name: named[key] for name, key in zip(component_names, component_keys)
        }
        return ScoreBreakdown(mean(components.values()), "component_mean", components)

    if allow_fallback:
        return ScoreBreakdown(mean(named.values()), "available_metric_mean", named)
    return None


def candidate_fingerprint(
    code: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    feature_schema: Optional[Iterable[str]] = None,
) -> str:
    payload = {
        "code": "\n".join(line.rstrip() for line in code.strip().splitlines()),
        "config": config or {},
        "feature_schema": sorted(feature_schema or []),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validation_metrics_only(
    metrics: Optional[Mapping[str, Optional[float]]],
    *,
    primary_fallback: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """Keep only explicitly named validation ranking metrics.

    Keys containing ``test`` or any unrecognized field are intentionally
    discarded so experience memory cannot become a side channel for test
    feedback.
    """
    allowed = {
        "gauc": "GAUC",
        "validationgauc": "GAUC",
        "ndcg5": "nDCG@5",
        "validationndcg5": "nDCG@5",
        "primary": "primary",
        "validationprimary": "primary",
    }
    result: dict[str, Optional[float]] = {
        "GAUC": None,
        "nDCG@5": None,
        "primary": primary_fallback,
    }
    for raw_name, raw_value in (metrics or {}).items():
        normalized = _normalized_metric_name(raw_name)
        canonical = allowed.get(normalized)
        if canonical is None or "test" in normalized:
            continue
        result[canonical] = _finite_float(raw_value)
    if result["primary"] is None:
        result["primary"] = primary_fallback
    return result


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    candidate_mean: float
    incumbent_score: float
    improvement: float
    seed_wins: int
    required_seed_wins: int
    reason: str


@dataclass(frozen=True)
class PromotionPolicy:
    min_improvement: float = 0.002
    required_seeds: int = 3
    required_seed_wins: int = 2
    maximize: bool = True

    def evaluate(
        self,
        *,
        candidate_score: float,
        incumbent_score: float,
        candidate_seed_scores: Sequence[float],
        incumbent_seed_scores: Optional[Sequence[float]] = None,
    ) -> PromotionDecision:
        valid_candidate_seeds = [
            score for score in candidate_seed_scores if math.isfinite(score)
        ]
        if len(valid_candidate_seeds) < self.required_seeds:
            return PromotionDecision(
                False,
                candidate_score,
                incumbent_score,
                0.0,
                0,
                self.required_seed_wins,
                f"only {len(valid_candidate_seeds)} valid seeds; "
                f"{self.required_seeds} required",
            )

        candidate_mean = mean(valid_candidate_seeds)
        direction = 1.0 if self.maximize else -1.0
        improvement = direction * (candidate_mean - incumbent_score)

        if incumbent_seed_scores and len(incumbent_seed_scores) >= self.required_seeds:
            comparisons = zip(valid_candidate_seeds, incumbent_seed_scores)
            seed_wins = sum(
                direction * (candidate - incumbent) > 0
                for candidate, incumbent in comparisons
            )
        else:
            seed_wins = sum(
                direction * (candidate - incumbent_score) > 0
                for candidate in valid_candidate_seeds
            )

        promoted = (
            improvement >= self.min_improvement
            and seed_wins >= self.required_seed_wins
        )
        if promoted:
            reason = (
                f"mean improved by {improvement:.6f}; "
                f"won {seed_wins}/{len(valid_candidate_seeds)} seeds"
            )
        elif improvement < self.min_improvement:
            reason = (
                f"mean improvement {improvement:.6f} is below "
                f"{self.min_improvement:.6f}"
            )
        else:
            reason = (
                f"won {seed_wins}/{len(valid_candidate_seeds)} seeds; "
                f"{self.required_seed_wins} wins required"
            )

        return PromotionDecision(
            promoted,
            candidate_mean,
            incumbent_score,
            improvement,
            seed_wins,
            self.required_seed_wins,
            reason,
        )


@dataclass
class ExperimentRecord:
    round_number: int
    node_id: str
    fingerprint: str
    score: Optional[float]
    seed_scores: list[float]
    promoted: bool
    reason: str
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    metric_deltas: dict[str, Optional[float]] = field(default_factory=dict)
    seed_metrics: list[dict[str, Optional[float]]] = field(default_factory=list)
    seed_mean: Optional[float] = None
    seed_std: Optional[float] = None
    seed_wins: int = 0
    principal_change: str = "unspecified controlled change"
    components: list[str] = field(default_factory=list)
    stage4_confirmed: Optional[bool] = None
    role: str = ""
    category: str = ""
    improvement: float = 0.0


@dataclass
class AblationEvidence:
    round_number: int
    candidate_id: str
    component: str
    category: str
    full_score: float
    ablated_score: float
    primary_contribution: float
    seed_wins: int
    verdict: str
    gauc_contribution: Optional[float] = None
    ndcg5_contribution: Optional[float] = None
    interaction_with: list[str] = field(default_factory=list)
    synergy: Optional[float] = None


@dataclass
class ExperimentMemory:
    fingerprints: set[str] = field(default_factory=set)
    records: list[ExperimentRecord] = field(default_factory=list)
    ablation_evidence: list[AblationEvidence] = field(default_factory=list)
    failed_hypotheses: list[str] = field(default_factory=list)

    def register_candidate(self, fingerprint: str) -> bool:
        if fingerprint in self.fingerprints:
            return False
        self.fingerprints.add(fingerprint)
        return True

    def record(self, record: ExperimentRecord) -> None:
        self.records.append(record)
        if not record.promoted:
            self.failed_hypotheses.append(record.reason)

    def mark_stage4_confirmation(
        self, node_id: str, *, confirmed: bool, reason: str
    ) -> None:
        for record in reversed(self.records):
            if record.node_id != node_id:
                continue
            record.stage4_confirmed = confirmed
            if not confirmed:
                self.failed_hypotheses.append(
                    f"Stage 4 confirmation rejected node {node_id}: {reason}"
                )
            return

    def records_for_round(self, round_number: int) -> list[ExperimentRecord]:
        return [
            record
            for record in self.records
            if record.round_number == round_number
        ]

    def diverse_near_winners(
        self, *, round_number: Optional[int] = None, limit: int = 3
    ) -> list[ExperimentRecord]:
        """Return strong, diverse, successful candidates that did not win.

        Diversity is enforced by the declared principal change and registered
        component set.  Fingerprints still prevent exact code/config repeats.
        """
        candidates = [
            record
            for record in self.records
            if record.score is not None
            and record.stage4_confirmed is None
            and "duplicate code/config/feature fingerprint" not in record.reason
            and (round_number is None or record.round_number == round_number)
        ]
        candidates.sort(key=lambda record: float(record.score), reverse=True)
        selected = []
        seen_directions = set()
        for record in candidates:
            direction = (
                record.principal_change.strip().lower(),
                tuple(sorted(record.components)),
            )
            if direction in seen_directions:
                continue
            seen_directions.add(direction)
            selected.append(record)
            if len(selected) >= max(0, int(limit)):
                break
        return selected

    def record_ablation(self, evidence: AblationEvidence) -> None:
        self.ablation_evidence.append(evidence)

    def direction_summary(self) -> dict[str, dict[str, float]]:
        """Aggregate validation evidence without letting it replace promotion."""
        summary: dict[str, dict[str, float]] = {}
        for record in self.records:
            if not record.category:
                continue
            item = summary.setdefault(
                record.category,
                {
                    "trials": 0.0,
                    "promotions": 0.0,
                    "mean_gain": 0.0,
                    "ablation_trials": 0.0,
                    "ablation_gain": 0.0,
                },
            )
            item["trials"] += 1.0
            item["promotions"] += float(record.promoted)
            item["mean_gain"] += record.improvement
        for evidence in self.ablation_evidence:
            if evidence.category:
                item = summary.setdefault(
                    evidence.category,
                    {
                        "trials": 0.0,
                        "promotions": 0.0,
                        "mean_gain": 0.0,
                        "ablation_trials": 0.0,
                        "ablation_gain": 0.0,
                    },
                )
                item["ablation_trials"] += 1.0
                item["ablation_gain"] += evidence.primary_contribution
        for item in summary.values():
            if item["trials"]:
                item["mean_gain"] /= item["trials"]
            if item["ablation_trials"]:
                item["ablation_gain"] /= item["ablation_trials"]
        return summary

    def prompt_evidence(self, category: str, *, limit: int = 6) -> str:
        """Return compact environment-grounded feedback for one candidate role."""
        records = [item for item in self.records if item.category == category][-limit:]
        ablations = [
            item for item in self.ablation_evidence if item.category == category
        ][-limit:]
        payload = {
            "recent_validation": [
                {
                    "role": item.role,
                    "improvement": round(item.improvement, 8),
                    "promoted": item.promoted,
                    "reason": item.reason,
                }
                for item in records
            ],
            "recent_ablation": [
                {
                    "component": item.component,
                    "primary_contribution": round(item.primary_contribution, 8),
                    "verdict": item.verdict,
                    "interaction_with": item.interaction_with,
                    "synergy": item.synergy,
                }
                for item in ablations
            ],
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def portfolio_lessons(self, *, limit: int = 6) -> str:
        """Compress cross-branch outcomes without replaying raw logs."""
        ranked_records = sorted(
            self.records,
            key=lambda item: item.improvement,
            reverse=True,
        )
        most_informative = ranked_records[:limit]
        if len(ranked_records) > limit:
            most_informative += ranked_records[-min(2, limit):]
        ranked_ablations = sorted(
            self.ablation_evidence,
            key=lambda item: abs(item.primary_contribution),
            reverse=True,
        )[:limit]
        payload = {
            "direction_summary": self.direction_summary(),
            "cross_branch_validation": [
                {
                    "category": item.category,
                    "role": item.role,
                    "improvement": round(item.improvement, 8),
                    "promoted": item.promoted,
                    "reason": item.reason,
                }
                for item in most_informative
            ],
            "component_and_synergy_lessons": [
                {
                    "category": item.category,
                    "component": item.component,
                    "primary_contribution": round(item.primary_contribution, 8),
                    "verdict": item.verdict,
                    "interaction_with": item.interaction_with,
                    "synergy": item.synergy,
                }
                for item in ranked_ablations
            ],
            "recent_implementation_failures": self.failed_hypotheses[-4:],
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprints": sorted(self.fingerprints),
            "records": [asdict(record) for record in self.records],
            "ablation_evidence": [asdict(item) for item in self.ablation_evidence],
            "direction_summary": self.direction_summary(),
            "failed_hypotheses": list(self.failed_hypotheses),
        }


@dataclass(frozen=True)
class ResearchLoopConfig:
    max_research_rounds: int = 4
    patience: int = 2
    stage1_validation_iterations: int = 2
    baseline_tuning_iterations: int = 24
    candidate_branches: int = 8
    candidate_parallel_workers: int = 3
    stage3_generation_attempts: int = 16
    candidate_tuning_iterations: int = 8
    candidate_refinement_top_k: int = 3
    candidate_refinement_iterations: int = 12
    candidate_finalist_top_k: int = 2
    finalist_top_k: int = 3
    finalist_num_seeds: int = 3
    experience_top_k: int = 3
    stage3_startup_stagger_seconds: float = 0.0
    final_confirmation_num_seeds: int = 5
    ablation_candidate_top_k: int = 2
    ablation_synergy_pairs: int = 3


def research_loop_config_from_mapping(raw: Any) -> ResearchLoopConfig:
    """Build the typed loop config from OmegaConf or a normal mapping."""
    values = {}
    for item in fields(ResearchLoopConfig):
        getter = getattr(raw, "get", None)
        values[item.name] = (
            getter(item.name, item.default)
            if callable(getter)
            else getattr(raw, item.name, item.default)
        )
    return ResearchLoopConfig(**values)


def estimate_max_experiment_runs(
    config: ResearchLoopConfig,
    *,
    stage4_max_components: int,
) -> dict[str, int]:
    """Return the explicit worst-case execution budget for one full run."""
    stage2_seed_runs = config.finalist_top_k * config.finalist_num_seeds
    candidate_seed_runs = config.candidate_finalist_top_k * config.finalist_num_seeds
    candidate_refinement_runs = (
        config.candidate_refinement_top_k * config.candidate_refinement_iterations
    )
    stage4_variants = stage4_max_components + config.ablation_synergy_pairs
    stage4_ablation_runs = config.ablation_candidate_top_k * stage4_variants
    stage4_seed_runs = (
        config.ablation_candidate_top_k
        * (stage4_variants + 1)
        * config.finalist_num_seeds
    )
    final_confirmation_runs = (
        config.ablation_candidate_top_k
        * config.final_confirmation_num_seeds
    )
    per_research_round = (
        config.stage3_generation_attempts
        + config.candidate_branches * config.candidate_tuning_iterations
        + candidate_refinement_runs
        + candidate_seed_runs
        + stage4_ablation_runs
        + stage4_seed_runs
        + final_confirmation_runs
    )
    per_round_search = (
        config.stage3_generation_attempts
        + config.candidate_branches * config.candidate_tuning_iterations
        + candidate_refinement_runs
        + stage4_ablation_runs
    )
    maximum_search_iterations = (
        config.stage1_validation_iterations
        + config.baseline_tuning_iterations
        + config.max_research_rounds * per_round_search
    )
    total = (
        config.stage1_validation_iterations
        + config.baseline_tuning_iterations
        + stage2_seed_runs
        + config.max_research_rounds * per_research_round
    )
    return {
        "stage1_validation": config.stage1_validation_iterations,
        "stage2_tuning": config.baseline_tuning_iterations,
        "stage2_seed_evaluation": stage2_seed_runs,
        "per_research_round": per_research_round,
        "final_confirmation_per_round": final_confirmation_runs,
        "maximum_search_iterations": maximum_search_iterations,
        "maximum_total": total,
    }


@dataclass
class ResearchLoopState:
    config: ResearchLoopConfig = field(default_factory=ResearchLoopConfig)
    policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    memory: ExperimentMemory = field(default_factory=ExperimentMemory)
    current_round: int = 0
    no_improvement_rounds: int = 0
    incumbent_node_id: Optional[str] = None
    incumbent_score: Optional[float] = None
    incumbent_seed_scores: list[float] = field(default_factory=list)
    incumbent_metrics: dict[str, Optional[float]] = field(default_factory=dict)
    pending_candidate_id: Optional[str] = None
    pending_candidate_score: Optional[float] = None
    eligible_candidate_ids: set[str] = field(default_factory=set)
    round_open: bool = False
    round_improved: bool = False

    def set_incumbent(
        self,
        node_id: str,
        score: float,
        seed_scores: Sequence[float] = (),
        metrics: Optional[Mapping[str, Optional[float]]] = None,
    ) -> None:
        self.incumbent_node_id = node_id
        self.incumbent_score = score
        self.incumbent_seed_scores = list(seed_scores)
        self.incumbent_metrics = validation_metrics_only(
            metrics, primary_fallback=score
        )
        self.pending_candidate_id = None
        self.pending_candidate_score = None
        self.eligible_candidate_ids.clear()

    def start_round(self) -> int:
        if self.round_open:
            raise RuntimeError("the current research round is still open")
        if self.should_stop:
            raise RuntimeError("research loop has reached its stopping condition")
        self.current_round += 1
        self.pending_candidate_id = None
        self.pending_candidate_score = None
        self.eligible_candidate_ids.clear()
        self.round_open = True
        self.round_improved = False
        return self.current_round

    def evaluate_candidate(
        self,
        *,
        node_id: str,
        fingerprint: str,
        score: float,
        seed_scores: Sequence[float],
        metrics: Optional[Mapping[str, Optional[float]]] = None,
        seed_metrics: Sequence[Mapping[str, Optional[float]]] = (),
        principal_change: str = "unspecified controlled change",
        components: Sequence[str] = (),
        role: str = "",
        category: str = "",
    ) -> PromotionDecision:
        if not self.round_open:
            raise RuntimeError("start_round must be called before evaluating candidates")
        if self.incumbent_score is None:
            raise RuntimeError("incumbent must be established before research rounds")
        if not self.memory.register_candidate(fingerprint):
            decision = PromotionDecision(
                False,
                score,
                self.incumbent_score,
                0.0,
                0,
                self.policy.required_seed_wins,
                "duplicate code/config/feature fingerprint",
            )
        else:
            decision = self.policy.evaluate(
                candidate_score=score,
                incumbent_score=self.incumbent_score,
                candidate_seed_scores=seed_scores,
                incumbent_seed_scores=self.incumbent_seed_scores,
            )

        candidate_metrics = validation_metrics_only(
            metrics, primary_fallback=score
        )
        metric_deltas = {}
        for metric_name in ("GAUC", "nDCG@5", "primary"):
            candidate_value = candidate_metrics.get(metric_name)
            incumbent_value = self.incumbent_metrics.get(metric_name)
            metric_deltas[metric_name] = (
                float(candidate_value) - float(incumbent_value)
                if candidate_value is not None and incumbent_value is not None
                else None
            )
        finite_seed_scores = [
            float(value) for value in seed_scores if math.isfinite(value)
        ]
        seed_mean = mean(finite_seed_scores) if finite_seed_scores else None
        seed_std = (
            math.sqrt(
                sum((value - seed_mean) ** 2 for value in finite_seed_scores)
                / len(finite_seed_scores)
            )
            if seed_mean is not None
            else None
        )
        self.memory.record(
            ExperimentRecord(
                round_number=self.current_round,
                node_id=node_id,
                fingerprint=fingerprint,
                score=score,
                seed_scores=list(seed_scores),
                promoted=decision.promoted,
                reason=decision.reason,
                metrics=candidate_metrics,
                metric_deltas=metric_deltas,
                seed_metrics=[
                    validation_metrics_only(item) for item in seed_metrics
                ],
                seed_mean=seed_mean,
                seed_std=seed_std,
                seed_wins=decision.seed_wins,
                principal_change=principal_change,
                components=list(components),
                role=role,
                category=category,
                improvement=decision.improvement,
            )
        )
        if decision.promoted and (
            self.pending_candidate_score is None
            or (
                decision.candidate_mean > self.pending_candidate_score
                if self.policy.maximize
                else decision.candidate_mean < self.pending_candidate_score
            )
        ):
            self.pending_candidate_id = node_id
            self.pending_candidate_score = decision.candidate_mean
        if decision.promoted:
            self.eligible_candidate_ids.add(node_id)
        return decision

    def accept_pending_candidate(
        self,
        *,
        node_id: str,
        final_node_id: Optional[str] = None,
        score: float,
        seed_scores: Sequence[float],
        metrics: Optional[Mapping[str, Optional[float]]] = None,
    ) -> PromotionDecision:
        if not self.round_open:
            raise RuntimeError("there is no open research round")
        if node_id not in self.eligible_candidate_ids:
            raise ValueError("only a promoted Stage 3 candidate can replace the incumbent")
        if self.incumbent_score is None:
            raise RuntimeError("incumbent must exist before final confirmation")
        confirmation_policy = PromotionPolicy(
            min_improvement=self.policy.min_improvement,
            required_seeds=self.config.final_confirmation_num_seeds,
            required_seed_wins=math.ceil(
                self.config.final_confirmation_num_seeds / 2
            ),
            maximize=self.policy.maximize,
        )
        decision = confirmation_policy.evaluate(
            candidate_score=score,
            incumbent_score=self.incumbent_score,
            candidate_seed_scores=seed_scores,
            incumbent_seed_scores=self.incumbent_seed_scores,
        )
        if not decision.promoted:
            self.memory.mark_stage4_confirmation(
                node_id, confirmed=False, reason=decision.reason
            )
            self.pending_candidate_id = None
            self.pending_candidate_score = None
            self.eligible_candidate_ids.discard(node_id)
            return decision
        self.memory.mark_stage4_confirmation(
            node_id, confirmed=True, reason=decision.reason
        )
        self.set_incumbent(
            final_node_id or node_id,
            score,
            seed_scores,
            metrics=metrics,
        )
        self.round_improved = True
        return decision

    def round_summary(
        self, round_number: int, *, near_winner_limit: Optional[int] = None
    ) -> dict[str, Any]:
        records = self.memory.records_for_round(round_number)
        limit = (
            self.config.experience_top_k
            if near_winner_limit is None
            else near_winner_limit
        )
        near_winners = self.memory.diverse_near_winners(
            round_number=round_number,
            limit=limit,
        )
        return {
            "schema_version": 1,
            "feedback_scope": "validation_only",
            "test_metrics_used": False,
            "round_number": round_number,
            "incumbent": {
                "node_id": self.incumbent_node_id,
                "score": self.incumbent_score,
                "metrics": dict(self.incumbent_metrics),
                "seed_scores": list(self.incumbent_seed_scores),
            },
            "candidates": [asdict(record) for record in records],
            "near_winners": [asdict(record) for record in near_winners],
            "recent_failed_hypotheses": self.memory.failed_hypotheses[-6:],
            "fingerprint_count": len(self.memory.fingerprints),
        }

    @staticmethod
    def _format_optional(value: Optional[float], *, signed: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.6f}" if signed else f"{value:.6f}"

    def experience_prompt(self, *, before_round: int) -> str:
        """Build a concise validation-only experience brief for Stage 3."""
        previous_round = before_round - 1
        if previous_round < 1:
            return (
                "Structured candidate experience: no previous Stage 3 round; "
                "start from the current incumbent."
            )
        records = self.memory.records_for_round(previous_round)
        near_winners = self.memory.diverse_near_winners(
            round_number=previous_round,
            limit=self.config.experience_top_k,
        )
        lines = [
            "Structured validation-only candidate experience from the previous round:",
            "- Never use test labels, test metrics, or test-derived feedback.",
            f"- Previous round evaluated {len(records)} unique fingerprints.",
            "- Current incumbent metrics: "
            f"GAUC={self._format_optional(self.incumbent_metrics.get('GAUC'))}, "
            f"nDCG@5={self._format_optional(self.incumbent_metrics.get('nDCG@5'))}, "
            f"primary={self._format_optional(self.incumbent_metrics.get('primary'))}.",
        ]
        if near_winners:
            lines.append(
                "- Transferable near-winner lessons; adapt one at a time to the "
                "incumbent, do not copy an entire rejected candidate unchanged:"
            )
            for record in near_winners:
                deltas = record.metric_deltas
                lines.append(
                    "  - "
                    f"{record.principal_change}; components="
                    f"{','.join(record.components) or 'unregistered'}; "
                    f"delta_GAUC={self._format_optional(deltas.get('GAUC'), signed=True)}, "
                    f"delta_nDCG@5={self._format_optional(deltas.get('nDCG@5'), signed=True)}, "
                    f"delta_primary={self._format_optional(deltas.get('primary'), signed=True)}; "
                    f"seed_wins={record.seed_wins}/{len(record.seed_scores)}, "
                    f"seed_std={self._format_optional(record.seed_std)}; "
                    f"decision={record.reason}."
                )
        else:
            lines.append("- No successful non-winning candidate is available for transfer.")
        if self.memory.failed_hypotheses:
            lines.append("- Recent rejected directions; do not repeat unchanged:")
            lines.extend(
                f"  - {reason}"
                for reason in self.memory.failed_hypotheses[-6:]
            )
        lines.append(
            "- Exact code/config/feature fingerprints remain blocked even when "
            "they appear in this summary."
        )
        return "\n".join(lines)

    def finish_round(self) -> None:
        if not self.round_open:
            raise RuntimeError("there is no open research round")
        if self.round_improved:
            self.no_improvement_rounds = 0
        else:
            self.no_improvement_rounds += 1
        self.round_open = False

    @property
    def should_stop(self) -> bool:
        return not self.round_open and (
            self.current_round >= self.config.max_research_rounds
            or self.no_improvement_rounds >= self.config.patience
        )
