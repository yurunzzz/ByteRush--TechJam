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
    max_component_regression: float = 0.001

    def evaluate(
        self,
        *,
        candidate_score: float,
        incumbent_score: float,
        candidate_seed_scores: Sequence[float],
        incumbent_seed_scores: Optional[Sequence[float]] = None,
        candidate_metrics: Optional[Mapping[str, Optional[float]]] = None,
        incumbent_metrics: Optional[Mapping[str, Optional[float]]] = None,
        artifacts_valid: bool = True,
    ) -> PromotionDecision:
        if not artifacts_valid:
            return PromotionDecision(
                False,
                candidate_score,
                incumbent_score,
                0.0,
                0,
                self.required_seed_wins,
                "candidate artifacts are incomplete or cannot be loaded",
            )
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

        component_regressions = []
        clean_candidate_metrics = validation_metrics_only(candidate_metrics)
        clean_incumbent_metrics = validation_metrics_only(incumbent_metrics)
        for metric_name in ("GAUC", "nDCG@5"):
            candidate_value = clean_candidate_metrics.get(metric_name)
            incumbent_value = clean_incumbent_metrics.get(metric_name)
            if candidate_value is None or incumbent_value is None:
                continue
            delta = direction * (float(candidate_value) - float(incumbent_value))
            if delta < -abs(float(self.max_component_regression)):
                component_regressions.append((metric_name, delta))

        promoted = (
            improvement >= self.min_improvement
            and seed_wins >= self.required_seed_wins
            and not component_regressions
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
        elif component_regressions:
            details = ", ".join(
                f"{name}={delta:+.6f}" for name, delta in component_regressions
            )
            reason = (
                "component metric regression exceeds tolerance "
                f"{self.max_component_regression:.6f}: {details}"
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
    factor_ids: list[str] = field(default_factory=list)
    considered_factor_ids: list[str] = field(default_factory=list)
    factor_selection_reason: str = ""
    factor_rejected_reasons: dict[str, str] = field(default_factory=dict)
    factor_cards: list[dict[str, Any]] = field(default_factory=list)
    model_family: str = ""
    research_family: str = ""
    loss_family: str = ""
    parent_node_id: str = ""
    parent_model_family: str = ""
    source_stage: str = ""
    source_phase: str = "stage3"
    status: str = "evaluated"
    code_path: str = ""
    checkpoint_path: str = ""
    validation_split: str = "fixed_train_validation"
    random_seeds: list[int] = field(default_factory=list)
    training_epochs: Optional[int] = None
    artifacts_valid: bool = True
    eligible_for_promotion: bool = False
    provisional: bool = False
    assignment_id: str = ""
    assignment_kind: str = "exploit"
    base_parent_id: str = ""
    donor_candidate_id: str = ""
    transferred_mechanism_ids: list[str] = field(default_factory=list)
    transfer_status: str = "not_applicable"


@dataclass
class CandidateAttemptRecord:
    """One generation/screening attempt, including cheap rejected attempts."""

    round_number: int
    assignment_id: str
    assignment_kind: str
    role: str
    category: str
    parent_node_id: str
    donor_candidate_id: str = ""
    status: str = "generated"
    failure_category: str = ""
    reason: str = ""
    node_id: str = ""
    elapsed_seconds: Optional[float] = None


@dataclass(frozen=True)
class TransferableInsightRecord:
    """Validation-only component evidence that may be moved to an incumbent."""

    donor_node_id: str
    donor_model_family: str
    mechanism_ids: tuple[str, ...]
    components: tuple[str, ...]
    principal_change: str
    metrics: Mapping[str, Optional[float]]
    metric_deltas: Mapping[str, Optional[float]]
    seed_mean: Optional[float]
    seed_std: Optional[float]
    code_path: str
    checkpoint_path: str
    confidence: str


@dataclass
class FrontierRecord:
    node_id: str
    source_stage: str
    round_number: int
    score: float
    metrics: dict[str, Optional[float]]
    model_family: str
    research_family: str
    loss_family: str
    parent_node_id: str
    parent_model_family: str
    fingerprint: str
    semantic_signature: str
    principal_change: str
    role: str


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
    frontier: list[FrontierRecord] = field(default_factory=list)
    attempts: list[CandidateAttemptRecord] = field(default_factory=list)

    def register_candidate(self, fingerprint: str) -> bool:
        if fingerprint in self.fingerprints:
            return False
        self.fingerprints.add(fingerprint)
        return True

    def record(self, record: ExperimentRecord) -> None:
        self.records.append(record)
        if not record.promoted and record.status not in {
            "bootstrap_pending_standardization",
            "reference_control",
        }:
            self.failed_hypotheses.append(record.reason)

    def record_attempt(self, attempt: CandidateAttemptRecord) -> None:
        self.attempts.append(attempt)

    def round_attempts(self, round_number: int) -> list[CandidateAttemptRecord]:
        return [item for item in self.attempts if item.round_number == round_number]

    def round_health(self, round_number: int) -> dict[str, Any]:
        attempts = self.round_attempts(round_number)
        accepted = [item for item in attempts if item.status == "accepted"]
        failures: dict[str, int] = {}
        for item in attempts:
            if item.status == "accepted":
                continue
            key = item.failure_category or item.status
            failures[key] = failures.get(key, 0) + 1
        records = self.records_for_round(round_number)
        gains = [float(item.improvement) for item in records if item.score is not None]
        transfers = [item for item in attempts if item.assignment_kind == "transfer"]
        successful_transfers = [item for item in transfers if item.status == "accepted"]
        return {
            "attempts": len(attempts),
            "valid_candidates": len(accepted),
            "evaluated_candidates": len(records),
            "best_primary_gain": max(gains) if gains else None,
            "failure_categories": failures,
            "transfer_attempts": len(transfers),
            "successful_transfers": len(successful_transfers),
        }

    def transferable_insights(
        self,
        *,
        incumbent_node_id: str = "",
        limit: int = 6,
        max_primary_gap: float = 0.002,
    ) -> list[TransferableInsightRecord]:
        """Return strong reloadable donor evidence without any test feedback."""
        candidates = []
        for record in self.records:
            if (
                record.node_id == incumbent_node_id
                or record.score is None
                or not record.artifacts_valid
                or not record.model_family
                or record.status in {"invalid_artifacts", "bootstrap_pending_standardization"}
            ):
                continue
            primary_delta = record.metric_deltas.get("primary")
            if primary_delta is not None and float(primary_delta) < -abs(max_primary_gap):
                continue
            mechanisms = tuple(record.transferred_mechanism_ids or record.components)
            if not mechanisms:
                mechanisms = (record.research_family or record.role,)
            confidence = (
                "ablation_supported"
                if any(
                    item.candidate_id == record.node_id
                    and item.primary_contribution > 0
                    for item in self.ablation_evidence
                )
                else "multi_seed_positive"
                if record.seed_wins > 0 and record.improvement > 0
                else "validation_hypothesis"
            )
            candidates.append(
                TransferableInsightRecord(
                    donor_node_id=record.node_id,
                    donor_model_family=record.model_family,
                    mechanism_ids=tuple(str(item) for item in mechanisms if item),
                    components=tuple(record.components),
                    principal_change=record.principal_change,
                    metrics=dict(record.metrics),
                    metric_deltas=dict(record.metric_deltas),
                    seed_mean=record.seed_mean,
                    seed_std=record.seed_std,
                    code_path=record.code_path,
                    checkpoint_path=record.checkpoint_path,
                    confidence=confidence,
                )
            )
        candidates.sort(
            key=lambda item: (
                float(item.metric_deltas.get("primary") or 0.0),
                float(item.metrics.get("primary") or float("-inf")),
            ),
            reverse=True,
        )
        selected = []
        seen = set()
        for item in candidates:
            signature = (item.donor_model_family, item.mechanism_ids)
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(item)
            if len(selected) >= max(0, int(limit)):
                break
        return selected

    def mark_verified_incumbent(self, node_id: str) -> None:
        for record in reversed(self.records):
            if record.node_id != node_id:
                continue
            record.provisional = False
            record.status = "verified_incumbent"
            record.eligible_for_promotion = True
            return

    def add_frontier(self, record: FrontierRecord, *, limit: int = 8) -> None:
        """Keep strong validation-successful nodes without collapsing diversity."""
        equivalent = [
            item
            for item in self.frontier
            if item.semantic_signature == record.semantic_signature
        ]
        if equivalent and max(item.score for item in equivalent) >= record.score:
            return
        self.frontier = [
            item
            for item in self.frontier
            if item.node_id != record.node_id
            and item.semantic_signature != record.semantic_signature
        ]
        self.frontier.append(record)
        ranked = sorted(self.frontier, key=lambda item: item.score, reverse=True)
        selected: list[FrontierRecord] = []
        seen_nodes = set()

        def take(key_name: str) -> None:
            seen = set()
            for item in ranked:
                key = getattr(item, key_name)
                if not key or key in seen or item.node_id in seen_nodes:
                    continue
                selected.append(item)
                seen.add(key)
                seen_nodes.add(item.node_id)
                if len(selected) >= max(1, int(limit)):
                    return

        take("model_family")
        if len(selected) < max(1, int(limit)):
            take("research_family")
        for item in ranked:
            if len(selected) >= max(1, int(limit)):
                break
            if item.node_id not in seen_nodes:
                selected.append(item)
                seen_nodes.add(item.node_id)
        self.frontier = selected

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

    def factor_summary(self) -> dict[str, dict[str, Any]]:
        """Summarize where each selected library factor helped or failed."""
        summary: dict[str, dict[str, Any]] = {}
        for record in self.records:
            for factor_id in record.considered_factor_ids:
                item = summary.setdefault(
                    factor_id,
                    {
                        "trials": 0,
                        "considerations": 0,
                        "promotions": 0,
                        "mean_gain": 0.0,
                        "mean_delta_GAUC": 0.0,
                        "mean_delta_nDCG@5": 0.0,
                        "GAUC_observations": 0,
                        "nDCG@5_observations": 0,
                        "model_roles": [],
                        "recent_conditions": [],
                        "recent_rejections": [],
                    },
                )
                item["considerations"] += 1
                if factor_id not in record.factor_ids:
                    rejection = record.factor_rejected_reasons.get(factor_id)
                    if rejection:
                        item["recent_rejections"].append(rejection[:180])
                        item["recent_rejections"] = item["recent_rejections"][-3:]
                    continue
                item["trials"] += 1
                item["promotions"] += int(record.promoted)
                item["mean_gain"] += record.improvement
                for metric_name, count_name in (
                    ("GAUC", "GAUC_observations"),
                    ("nDCG@5", "nDCG@5_observations"),
                ):
                    value = record.metric_deltas.get(metric_name)
                    if value is None:
                        continue
                    item[f"mean_delta_{metric_name}"] += float(value)
                    item[count_name] += 1
                if record.role and record.role not in item["model_roles"]:
                    item["model_roles"].append(record.role)
                item["recent_conditions"].append(
                    {
                        "model_role": record.role,
                        "candidate_context": record.principal_change[:180],
                        "selection_reason": record.factor_selection_reason[:180],
                        "gain": round(record.improvement, 8),
                        "promoted": record.promoted,
                    }
                )
                item["recent_conditions"] = item["recent_conditions"][-3:]
        for item in summary.values():
            trials = int(item["trials"])
            if trials:
                item["mean_gain"] /= trials
            else:
                item["mean_gain"] = None
            for metric_name, count_name in (
                ("GAUC", "GAUC_observations"),
                ("nDCG@5", "nDCG@5_observations"),
            ):
                observations = int(item[count_name])
                if observations:
                    item[f"mean_delta_{metric_name}"] /= observations
                else:
                    item[f"mean_delta_{metric_name}"] = None
        discovered = self.discovered_factor_cards()
        for factor_id, item in summary.items():
            if factor_id in discovered:
                item["self_description"] = discovered[factor_id]
        return summary

    def discovered_factor_cards(self) -> dict[str, dict[str, Any]]:
        cards: dict[str, dict[str, Any]] = {}
        for record in self.records:
            for card in record.factor_cards:
                factor_id = str(card.get("factor_id", ""))
                if factor_id.startswith("custom_"):
                    cards[factor_id] = dict(card)
        return cards

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
        for attempt in self.attempts:
            if not attempt.category:
                continue
            item = summary.setdefault(
                attempt.category,
                {
                    "trials": 0.0,
                    "promotions": 0.0,
                    "mean_gain": 0.0,
                    "ablation_trials": 0.0,
                    "ablation_gain": 0.0,
                },
            )
            item["generation_attempts"] = item.get("generation_attempts", 0.0) + 1.0
            if attempt.status == "accepted":
                item["valid_candidates"] = item.get("valid_candidates", 0.0) + 1.0
        for item in summary.values():
            if item["trials"]:
                item["mean_gain"] /= item["trials"]
            if item["ablation_trials"]:
                item["ablation_gain"] /= item["ablation_trials"]
            attempts = item.get("generation_attempts", 0.0)
            valid = item.get("valid_candidates", 0.0)
            item["generation_success_rate"] = valid / attempts if attempts else 0.0
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
            "factor_library_evidence": self.factor_summary(),
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
            "diverse_frontier": [
                {
                    "node_id": item.node_id,
                    "source_stage": item.source_stage,
                    "model_family": item.model_family,
                    "research_family": item.research_family,
                    "loss_family": item.loss_family,
                    "score": item.score,
                    "principal_change": item.principal_change,
                }
                for item in self.frontier[:limit]
            ],
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprints": sorted(self.fingerprints),
            "records": [asdict(record) for record in self.records],
            "ablation_evidence": [asdict(item) for item in self.ablation_evidence],
            "direction_summary": self.direction_summary(),
            "factor_library_evidence": self.factor_summary(),
            "discovered_factor_cards": self.discovered_factor_cards(),
            "failed_hypotheses": list(self.failed_hypotheses),
            "diverse_frontier": [asdict(item) for item in self.frontier],
            "candidate_attempts": [asdict(item) for item in self.attempts],
            "transferable_insights": [
                asdict(item) for item in self.transferable_insights()
            ],
        }


@dataclass(frozen=True)
class ResearchLoopConfig:
    max_research_rounds: int = 4
    patience: int = 2
    stage1_validation_iterations: int = 2
    stage1b_enabled: bool = False
    stage1b_model_families: tuple[str, ...] = ("mlp", "wide_deep", "dcn")
    stage1b_generation_attempts: int = 6
    stage1b_max_epochs: int = 5
    frontier_max_size: int = 8
    baseline_tuning_iterations: int = 24
    stage2_num_seeds: int = 3
    stage2_root_top_k: int = 4
    max_component_regression: float = 0.001
    candidate_branches: int = 8
    stage3_incumbent_parent_slots: int = 2
    stage3_frontier_parent_slots: int = 2
    stage3_bootstrap_parent_slots: int = 1
    initial_candidate_roles: tuple[str, ...] = ()
    reserved_candidate_role: str = ""
    candidate_parallel_workers: int = 3
    stage3_generation_attempts: int = 16
    candidate_tuning_top_k: int = 2
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
    min_valid_candidates_per_round: int = 3
    target_valid_candidates_per_round: int = 5
    max_candidate_branches: int = 8
    branch_growth_per_positive_round: int = 2
    max_repair_attempts_per_assignment: int = 2
    no_valid_round_patience: int = 2
    low_gain_round_patience: int = 3
    min_round_primary_gain: float = 0.0002
    provisional_min_primary_gain: float = 0.0005
    final_promotion_min_primary_gain: float = 0.001
    transfer_base_ratio: float = 0.30
    transfer_min_ratio: float = 0.15
    transfer_max_ratio: float = 0.50
    donor_max_primary_gap: float = 0.002
    smoke_test_enabled: bool = True
    smoke_test_timeout_seconds: int = 240
    max_wall_clock_seconds: int = 19800
    finalize_reserve_seconds: int = 600
    checkpoint_submission_each_incumbent: bool = True


def research_loop_config_from_mapping(raw: Any) -> ResearchLoopConfig:
    """Build the typed loop config from OmegaConf or a normal mapping."""
    values = {}
    for item in fields(ResearchLoopConfig):
        getter = getattr(raw, "get", None)
        value = (
            getter(item.name, item.default)
            if callable(getter)
            else getattr(raw, item.name, item.default)
        )
        if item.name in {"initial_candidate_roles", "stage1b_model_families"}:
            value = tuple(value or ())
        values[item.name] = value
    return ResearchLoopConfig(**values)


def estimate_max_experiment_runs(
    config: ResearchLoopConfig,
    *,
    stage4_max_components: int,
) -> dict[str, int]:
    """Return the explicit worst-case execution budget for one full run."""
    stage2_root_count = 1
    if config.stage1b_enabled:
        stage2_root_count += min(
            max(0, config.stage2_root_top_k - 1),
            len(config.stage1b_model_families),
        )
    stage2_seed_runs = (
        stage2_root_count * config.finalist_top_k * config.stage2_num_seeds
    )
    stage2_tuning_runs = stage2_root_count * config.baseline_tuning_iterations
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
        + config.candidate_tuning_top_k * config.candidate_tuning_iterations
        + candidate_refinement_runs
        + candidate_seed_runs
        + stage4_ablation_runs
        + stage4_seed_runs
        + final_confirmation_runs
    )
    per_round_search = (
        config.stage3_generation_attempts
        + config.candidate_tuning_top_k * config.candidate_tuning_iterations
        + candidate_refinement_runs
        + stage4_ablation_runs
    )
    maximum_search_iterations = (
        config.stage1_validation_iterations
        + (config.stage1b_generation_attempts if config.stage1b_enabled else 0)
        + stage2_tuning_runs
        + config.max_research_rounds * per_round_search
    )
    total = (
        config.stage1_validation_iterations
        + (config.stage1b_generation_attempts if config.stage1b_enabled else 0)
        + stage2_tuning_runs
        + stage2_seed_runs
        + config.max_research_rounds * per_research_round
    )
    return {
        "stage1_validation": config.stage1_validation_iterations,
        "stage1b_diverse_roots": (
            config.stage1b_generation_attempts if config.stage1b_enabled else 0
        ),
        "stage2_roots": stage2_root_count,
        "stage2_tuning": stage2_tuning_runs,
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
    incumbent_source_stage: str = ""
    verified_incumbent_id: Optional[str] = None
    provisional_candidate_id: Optional[str] = None
    provisional_candidate_score: Optional[float] = None
    pending_candidate_id: Optional[str] = None
    pending_candidate_score: Optional[float] = None
    eligible_candidate_ids: set[str] = field(default_factory=set)
    round_open: bool = False
    round_improved: bool = False
    no_valid_rounds: int = 0
    low_gain_rounds: int = 0
    last_round_valid_candidates: int = 0
    last_round_best_gain: Optional[float] = None
    forced_stop_reason: str = ""

    def set_incumbent(
        self,
        node_id: str,
        score: float,
        seed_scores: Sequence[float] = (),
        metrics: Optional[Mapping[str, Optional[float]]] = None,
        source_stage: str = "",
    ) -> None:
        self.incumbent_node_id = node_id
        self.incumbent_score = score
        self.incumbent_seed_scores = list(seed_scores)
        self.incumbent_metrics = validation_metrics_only(
            metrics, primary_fallback=score
        )
        self.incumbent_source_stage = source_stage
        self.verified_incumbent_id = node_id
        if self.provisional_candidate_id == node_id:
            self.provisional_candidate_id = None
            self.provisional_candidate_score = None
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

    def request_stop(self, reason: str) -> None:
        self.forced_stop_reason = str(reason)

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
        factor_ids: Sequence[str] = (),
        considered_factor_ids: Sequence[str] = (),
        factor_selection_reason: str = "",
        factor_rejected_reasons: Optional[Mapping[str, str]] = None,
        factor_cards: Sequence[Mapping[str, Any]] = (),
        model_family: str = "",
        research_family: str = "",
        loss_family: str = "",
        parent_node_id: str = "",
        parent_model_family: str = "",
        source_stage: str = "",
        source_phase: str = "stage3",
        code_path: str = "",
        checkpoint_path: str = "",
        validation_split: str = "fixed_train_validation",
        random_seeds: Sequence[int] = (),
        training_epochs: Optional[int] = None,
        artifacts_valid: bool = True,
        assignment_id: str = "",
        assignment_kind: str = "exploit",
        base_parent_id: str = "",
        donor_candidate_id: str = "",
        transferred_mechanism_ids: Sequence[str] = (),
        round_number: Optional[int] = None,
        policy_override: Optional[PromotionPolicy] = None,
    ) -> PromotionDecision:
        if not self.round_open and round_number is None:
            raise RuntimeError("start_round must be called before evaluating candidates")
        if self.incumbent_score is None:
            raise RuntimeError("incumbent must be established before research rounds")
        candidate_metrics = validation_metrics_only(
            metrics, primary_fallback=score
        )
        policy = policy_override or self.policy
        if not self.memory.register_candidate(fingerprint):
            decision = PromotionDecision(
                False,
                score,
                self.incumbent_score,
                0.0,
                0,
                policy.required_seed_wins,
                "duplicate code/config/feature fingerprint",
            )
        else:
            decision = policy.evaluate(
                candidate_score=score,
                incumbent_score=self.incumbent_score,
                candidate_seed_scores=seed_scores,
                incumbent_seed_scores=self.incumbent_seed_scores,
                candidate_metrics=candidate_metrics,
                incumbent_metrics=self.incumbent_metrics,
                artifacts_valid=artifacts_valid,
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
                round_number=(
                    self.current_round if round_number is None else int(round_number)
                ),
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
                factor_ids=list(factor_ids),
                considered_factor_ids=list(considered_factor_ids or factor_ids),
                factor_selection_reason=factor_selection_reason,
                factor_rejected_reasons=dict(factor_rejected_reasons or {}),
                factor_cards=[dict(item) for item in factor_cards],
                model_family=model_family,
                research_family=research_family,
                loss_family=loss_family,
                parent_node_id=parent_node_id,
                parent_model_family=parent_model_family,
                source_stage=source_stage,
                source_phase=source_phase,
                status=(
                    "provisional"
                    if decision.promoted and source_phase != "stage3"
                    else "promoted"
                    if decision.promoted
                    else "invalid_artifacts"
                    if not artifacts_valid
                    else "valid_not_promoted"
                ),
                code_path=code_path,
                checkpoint_path=checkpoint_path,
                validation_split=validation_split,
                random_seeds=[int(item) for item in random_seeds],
                training_epochs=training_epochs,
                artifacts_valid=artifacts_valid,
                eligible_for_promotion=decision.promoted,
                provisional=decision.promoted and source_phase != "stage3",
                assignment_id=assignment_id,
                assignment_kind=assignment_kind,
                base_parent_id=base_parent_id or parent_node_id,
                donor_candidate_id=donor_candidate_id,
                transferred_mechanism_ids=list(transferred_mechanism_ids),
                transfer_status=(
                    "positive"
                    if assignment_kind == "transfer" and decision.improvement > 0
                    else "non_positive"
                    if assignment_kind == "transfer"
                    else "not_applicable"
                ),
            )
        )
        if decision.promoted and not self.round_open:
            better = (
                self.provisional_candidate_score is None
                or (
                    decision.candidate_mean > self.provisional_candidate_score
                    if policy.maximize
                    else decision.candidate_mean < self.provisional_candidate_score
                )
            )
            if better:
                self.provisional_candidate_id = node_id
                self.provisional_candidate_score = decision.candidate_mean
        if decision.promoted and self.round_open and (
            self.pending_candidate_score is None
            or (
                decision.candidate_mean > self.pending_candidate_score
                if self.policy.maximize
                else decision.candidate_mean < self.pending_candidate_score
            )
        ):
            self.pending_candidate_id = node_id
            self.pending_candidate_score = decision.candidate_mean
        if decision.promoted and self.round_open:
            self.eligible_candidate_ids.add(node_id)
        return decision

    def admit_provisional_candidate(self, node_id: str, score: float) -> None:
        """Allow a small promising gain to enter Stage 4, not to auto-promote."""
        if not self.round_open:
            raise RuntimeError("there is no open research round")
        self.eligible_candidate_ids.add(node_id)
        for record in reversed(self.memory.records):
            if record.node_id != node_id:
                continue
            record.provisional = True
            record.eligible_for_promotion = True
            record.status = "provisional_stage4"
            break
        if self.pending_candidate_score is None or score > self.pending_candidate_score:
            self.pending_candidate_id = node_id
            self.pending_candidate_score = score

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
            raise ValueError("only a candidate admitted by the promotion gate can replace the incumbent")
        if self.incumbent_score is None:
            raise RuntimeError("incumbent must exist before final confirmation")
        confirmation_policy = PromotionPolicy(
            min_improvement=self.config.final_promotion_min_primary_gain,
            required_seeds=self.config.final_confirmation_num_seeds,
            required_seed_wins=math.ceil(
                self.config.final_confirmation_num_seeds / 2
            ),
            maximize=self.policy.maximize,
            max_component_regression=self.policy.max_component_regression,
        )
        decision = confirmation_policy.evaluate(
            candidate_score=score,
            incumbent_score=self.incumbent_score,
            candidate_seed_scores=seed_scores,
            incumbent_seed_scores=self.incumbent_seed_scores,
            candidate_metrics=metrics,
            incumbent_metrics=self.incumbent_metrics,
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
            source_stage="stage4_final_confirmation",
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
            "schema_version": 2,
            "feedback_scope": "validation_only",
            "test_metrics_used": False,
            "round_number": round_number,
            "incumbent": {
                "node_id": self.incumbent_node_id,
                "score": self.incumbent_score,
                "metrics": dict(self.incumbent_metrics),
                "seed_scores": list(self.incumbent_seed_scores),
                "source_stage": self.incumbent_source_stage,
                "verified": self.verified_incumbent_id == self.incumbent_node_id,
            },
            "provisional_candidate": {
                "node_id": self.provisional_candidate_id,
                "score": self.provisional_candidate_score,
            },
            "candidates": [asdict(record) for record in records],
            "candidate_attempts": [
                asdict(item) for item in self.memory.round_attempts(round_number)
            ],
            "round_health": self.memory.round_health(round_number),
            "near_winners": [asdict(record) for record in near_winners],
            "diverse_frontier": [asdict(record) for record in self.memory.frontier],
            "recent_failed_hypotheses": self.memory.failed_hypotheses[-6:],
            "fingerprint_count": len(self.memory.fingerprints),
            "adaptive_stop_state": {
                "no_valid_rounds": self.no_valid_rounds,
                "low_gain_rounds": self.low_gain_rounds,
                "no_improvement_rounds": self.no_improvement_rounds,
                "forced_stop_reason": self.forced_stop_reason,
            },
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
            prefix = (
                "Structured candidate experience: no previous Stage 3 round; "
                "use the assigned Stage 1B/frontier parent without test feedback."
            )
            if not self.memory.frontier:
                return prefix
            frontier = "; ".join(
                f"{item.model_family}@{item.score:.6f}({item.source_stage})"
                for item in self.memory.frontier[: self.config.experience_top_k]
            )
            return prefix + " Diverse validation frontier: " + frontier
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
        health = self.memory.round_health(self.current_round)
        self.last_round_valid_candidates = int(health["valid_candidates"])
        self.last_round_best_gain = health["best_primary_gain"]
        if self.last_round_valid_candidates <= 0:
            self.no_valid_rounds += 1
        else:
            self.no_valid_rounds = 0
        if (
            self.last_round_best_gain is None
            or self.last_round_best_gain < self.config.min_round_primary_gain
        ):
            self.low_gain_rounds += 1
        else:
            self.low_gain_rounds = 0
        self.round_open = False

    @property
    def should_stop(self) -> bool:
        return not self.round_open and (
            bool(self.forced_stop_reason)
            or self.current_round >= self.config.max_research_rounds
            or self.no_improvement_rounds >= self.config.patience
            or self.no_valid_rounds >= self.config.no_valid_round_patience
            or self.low_gain_rounds >= self.config.low_gain_round_patience
        )
