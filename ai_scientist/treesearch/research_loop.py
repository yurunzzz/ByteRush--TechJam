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


@dataclass
class ExperimentMemory:
    fingerprints: set[str] = field(default_factory=set)
    records: list[ExperimentRecord] = field(default_factory=list)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprints": sorted(self.fingerprints),
            "records": [asdict(record) for record in self.records],
            "failed_hypotheses": list(self.failed_hypotheses),
        }


@dataclass(frozen=True)
class ResearchLoopConfig:
    max_research_rounds: int = 3
    patience: int = 3
    stage1_validation_iterations: int = 2
    baseline_tuning_iterations: int = 20
    candidate_branches: int = 3
    stage3_generation_attempts: int = 12
    candidate_tuning_iterations: int = 12
    finalist_top_k: int = 3
    finalist_num_seeds: int = 3


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
    candidate_seed_runs = (
        config.candidate_branches
        * config.finalist_top_k
        * config.finalist_num_seeds
    )
    stage4_ablation_runs = stage4_max_components
    stage4_seed_runs = (
        stage4_max_components + 1
    ) * config.finalist_num_seeds
    per_research_round = (
        config.stage3_generation_attempts
        + config.candidate_branches * config.candidate_tuning_iterations
        + candidate_seed_runs
        + stage4_ablation_runs
        + stage4_seed_runs
    )
    per_round_search = (
        config.stage3_generation_attempts
        + config.candidate_branches * config.candidate_tuning_iterations
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
    pending_candidate_id: Optional[str] = None
    pending_candidate_score: Optional[float] = None
    round_open: bool = False
    round_improved: bool = False

    def set_incumbent(
        self,
        node_id: str,
        score: float,
        seed_scores: Sequence[float] = (),
    ) -> None:
        self.incumbent_node_id = node_id
        self.incumbent_score = score
        self.incumbent_seed_scores = list(seed_scores)
        self.pending_candidate_id = None
        self.pending_candidate_score = None

    def start_round(self) -> int:
        if self.round_open:
            raise RuntimeError("the current research round is still open")
        if self.should_stop:
            raise RuntimeError("research loop has reached its stopping condition")
        self.current_round += 1
        self.pending_candidate_id = None
        self.pending_candidate_score = None
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

        self.memory.record(
            ExperimentRecord(
                round_number=self.current_round,
                node_id=node_id,
                fingerprint=fingerprint,
                score=score,
                seed_scores=list(seed_scores),
                promoted=decision.promoted,
                reason=decision.reason,
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
        return decision

    def accept_pending_candidate(
        self,
        *,
        node_id: str,
        final_node_id: Optional[str] = None,
        score: float,
        seed_scores: Sequence[float],
    ) -> PromotionDecision:
        if not self.round_open:
            raise RuntimeError("there is no open research round")
        if node_id != self.pending_candidate_id:
            raise ValueError("only the promoted candidate can replace the incumbent")
        if self.incumbent_score is None:
            raise RuntimeError("incumbent must exist before final confirmation")
        decision = self.policy.evaluate(
            candidate_score=score,
            incumbent_score=self.incumbent_score,
            candidate_seed_scores=seed_scores,
            incumbent_seed_scores=self.incumbent_seed_scores,
        )
        if not decision.promoted:
            self.memory.failed_hypotheses.append(
                f"Stage 4 confirmation rejected node {node_id}: {decision.reason}"
            )
            self.pending_candidate_id = None
            self.pending_candidate_score = None
            return decision
        self.set_incumbent(final_node_id or node_id, score, seed_scores)
        self.round_improved = True
        return decision

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
