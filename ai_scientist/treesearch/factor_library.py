"""Small, self-describing factor library for autonomous candidate design."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FactorCard:
    factor_id: str
    semantics: str
    helps_when: tuple[str, ...]
    model_fit: tuple[str, ...]
    avoid_when: tuple[str, ...]
    data_cost: str
    leakage_rule: str


FACTOR_CARDS: tuple[FactorCard, ...] = (
    FactorCard(
        "causal_recent_history",
        "Past-only recent user interests, preserving order and recency before the target exposure.",
        ("active users have weak ranking", "recent intent may differ from long-term identity"),
        ("DIN-style attention", "gated interest encoder", "FM residual feature"),
        ("most users have no usable history", "the candidate already has an equivalent history path"),
        "medium",
        "Use only interactions before the target; freeze outcome-derived state after train.",
    ),
    FactorCard(
        "user_author_affinity",
        "Past-only strength and recency of a user's preference for the target author.",
        ("author preference appears persistent", "history-rich users underperform"),
        ("FM cross", "gated affinity tower", "target attention"),
        ("author IDs are mostly missing or unseen",),
        "low",
        "Build affinities from train history only and never use the current outcome.",
    ),
    FactorCard(
        "user_tag_type_affinity",
        "Past-only preference for video tags or types, useful beyond exact user/video IDs.",
        ("cold items or unseen videos underperform", "semantic preference should transfer across items"),
        ("DeepFM/DCN crosses", "gated affinity tower", "metadata-aware interest model"),
        ("tag/type coverage is too sparse",),
        "medium",
        "Fit vocabularies and outcome-derived affinity statistics on train only.",
    ),
    FactorCard(
        "temporal_recency_context",
        "Prediction-time-known date, hour, weekday, and recency context describing temporal drift.",
        ("validation changes by day or hour", "recent behavior should matter more"),
        ("FM feature", "DCN/DeepFM interaction", "time-aware weighting"),
        ("the split does not expose the required timestamp at prediction time",),
        "low",
        "Use only timestamp fields available at prediction time; keep the official chronological split.",
    ),
    FactorCard(
        "static_user_profile",
        "Train-independent user activity and profile attributes for sparse-history and cold-start behavior.",
        ("low-history users underperform", "ID embeddings overfit frequent users"),
        ("FM feature", "wide crosses", "small profile tower"),
        ("profile missingness dominates and has no explicit bucket",),
        "low",
        "Use static profile columns with explicit unknown and missing buckets.",
    ),
    FactorCard(
        "static_video_metadata",
        "Video type, author, tag, duration bucket, and other prediction-time-known item metadata.",
        ("cold or rare videos underperform", "item IDs do not generalize"),
        ("FM feature", "DeepFM/DCN crosses", "metadata-aware item tower"),
        ("metadata is unavailable for a large share of test items",),
        "low",
        "Do not use post-exposure statistics; rebuild any aggregate from the train window.",
    ),
    FactorCard(
        "user_item_context_cross",
        "Explicit interactions between user state, item metadata, tab, and temporal context.",
        ("main effects are present but conditional behavior is missed",),
        ("FM", "DeepFM", "DCN", "AutoInt"),
        ("the candidate already adds a high-capacity interaction block",),
        "medium",
        "Cross only prediction-time-known fields and regularize sparse combinations.",
    ),
    FactorCard(
        "auxiliary_behavior_signal",
        "Train-window click, like, follow, comment, or watch-time behavior used as supervision, not inference input.",
        ("the primary label is sparse", "related behavior supplies useful representation learning"),
        ("shared-bottom auxiliary head", "MMoE only after task conflict is measured"),
        ("the auxiliary label is missing or merely copies long_view",),
        "medium",
        "Read only the train window and never expose current-row outcomes at inference.",
    ),
)

FACTOR_CARD_BY_ID = {card.factor_id: card for card in FACTOR_CARDS}
FACTOR_IDS = frozenset(FACTOR_CARD_BY_ID)


def factor_library_prompt(
    *,
    role_group: str,
    role_category: str,
    observed_evidence: Mapping[str, Any] | None = None,
    discovered_cards: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Render a compact library; ordering nudges relevance without forcing use."""
    preferred = {
        "history_interest": {
            "causal_recent_history",
            "user_author_affinity",
            "user_tag_type_affinity",
        },
        "context_interaction": {
            "temporal_recency_context",
            "static_user_profile",
            "static_video_metadata",
            "user_item_context_cross",
        },
        "objective_and_training": {"auxiliary_behavior_signal"},
    }.get(role_group, set())
    ranked = sorted(
        FACTOR_CARDS,
        key=lambda card: (card.factor_id not in preferred, card.factor_id),
    )
    payload = {
        "candidate_context": {
            "group": role_group,
            "category": role_category,
        },
        "factor_cards": [asdict(card) for card in ranked],
        "agent_discovered_cards": list((discovered_cards or {}).values()),
        "observed_factor_evidence": dict(observed_evidence or {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)
