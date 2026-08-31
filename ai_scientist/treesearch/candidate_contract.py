"""Static contracts for diverse, executable KuaiRand research candidates."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .factor_library import FACTOR_IDS


MAX_RESEARCH_PROTOTYPE_EPOCHS = 12


@dataclass(frozen=True)
class CandidateRole:
    name: str
    group: str
    category: str
    objective: str
    required_evidence: str
    allowed_model_families: tuple[str, ...] = ()
    allowed_loss_families: tuple[str, ...] = ()
    autonomous: bool = False

    def prompt(
        self,
        index: int,
        total: int,
        *,
        retry_feedback: Sequence[str] = (),
        evidence_memory: str = "",
        factor_library_context: str = "",
        parent_node_id: str = "",
        parent_model_family: str = "fm",
        assignment_id: str = "",
        assignment_kind: str = "exploit",
        donor_context: str = "",
    ) -> str:
        if self.autonomous:
            return autonomous_candidate_prompt(
                self,
                index=index,
                total=total,
                retry_feedback=retry_feedback,
                evidence_memory=evidence_memory,
                factor_library_context=factor_library_context,
                parent_node_id=parent_node_id,
                parent_model_family=parent_model_family,
                assignment_id=assignment_id,
                assignment_kind=assignment_kind,
            )
        evidence_ids = ", ".join(CURATED_EVIDENCE)
        base_name = self.name.split("_alternative_", 1)[0]
        techniques = TECHNIQUE_CATALOG.get(base_name, ())
        role_contract = {
            "ranking_objective": (
                "\nRanking-objective executable contract:\n"
                "- Extract encoded user IDs exactly from the first feature column, for example "
                "user_ids = x_tensor[:, 0]. Build every pairwise or listwise group from those "
                "IDs; a positive and negative from different users is invalid.\n"
            ),
            "auxiliary_objective": (
                "\nAuxiliary-objective executable contract:\n"
                "- The trusted loader tuple has no click/like/play-time label. Read a real auxiliary "
                "field from the train-only log_standard_4_08_to_4_21_pure.csv, align it to train "
                "examples by stable exposure keys, and never read the later log.\n"
                "- Never derive an auxiliary target from train_y, y, long_view, or a copy/alias of "
                "the primary label. There is no long_view fallback for a missing auxiliary field.\n"
            ),
        }.get(base_name, "")
        technique_text = "\n".join(f"  - {item}" for item in techniques)
        model_family_text = ", ".join(self.allowed_model_families) or parent_model_family
        loss_family_text = ", ".join(self.allowed_loss_families) or "any allowed loss family"
        feedback_text = (
            "\nRole-specific feedback from rejected attempts:\n"
            + "\n".join(f"  - {item}" for item in retry_feedback[-3:])
            if retry_feedback
            else ""
        )
        memory_text = (
            f"\nRelevant validation/ablation memory:\n{evidence_memory.strip()}\n"
            if evidence_memory.strip()
            else ""
        )
        factor_text = (
            "\nAutonomous factor library (small metadata cards, not extra training data):\n"
            + factor_library_context.strip()
            + "\nInspect the cards even if the central mechanism is not a feature model. "
            "Select zero, one, or at most two factors only when they plausibly help this "
            "candidate. Selecting none is valid when the rationale is explicit.\n"
            if factor_library_context.strip()
            else ""
        )
        scaffold_text = candidate_contract_scaffold(
            self,
            parent_node_id=parent_node_id,
            parent_model_family=parent_model_family,
        )
        assignment_marker = build_assignment_marker(
            self,
            assignment_id=assignment_id or f"{self.name}:{parent_node_id}",
            assignment_kind=assignment_kind,
            parent_node_id=parent_node_id,
            parent_model_family=parent_model_family,
        )
        donor_text = (
            "\nExplicit cross-parent transfer assignment:\n"
            + donor_context.strip()
            + "\nUse the assigned incumbent as the only base code. Transfer exactly "
            "one named donor mechanism; do not copy the entire donor model, data "
            "pipeline, CONFIG, or checkpoint. Register the transferred component "
            "and keep it independently ablatable.\n"
            if donor_context.strip()
            else ""
        )
        recipe_key = (
            "cross_parent_transfer"
            if assignment_kind == "transfer"
            else base_name
        )
        recipe = ROLE_EXECUTION_RECIPES.get(recipe_key, ROLE_EXECUTION_RECIPES["default"])
        recipe_text = (
            "\nRole execution recipe (all checks are pre-training gates):\n"
            + "\n".join(f"- {item}" for item in recipe)
            + "\n"
        )
        return (
            f"Candidate role {index}/{total}: {self.name}\n"
            f"Major research group: {self.group}\n"
            f"Research category: {self.category}\n"
            f"Required mechanism: {self.objective}\n"
            f"Required evidence in code: {self.required_evidence}\n"
            f"Role-specific technique menu (prior, not a mandate):\n{technique_text}\n"
            "Choose the smallest coherent mechanism that tests the hypothesis. "
            "Do not copy the whole menu into one candidate.\n"
            "This role is exclusive. Do not replace its required target model family "
            "and do not submit a CONFIG-only hyperparameter change. The candidate may "
            "be a coherent bundle: one central mechanism plus compatible supporting "
            "factors, objectives, or training changes. Guard every component separately."
            "\nRequired code output contract:\n"
            "- Define a literal RESEARCH_MANIFEST with candidate_id, role, group, "
            "category, model_family, research_family, loss_family, parent_node_id, "
            "parent_model_family, input_schema_version, hypothesis, mechanism, "
            "mechanism_ids, modified_symbols, expected_metric, "
            "tunable_parameters, ablation_components, combination_compatibility, "
            "change_scope, component_dependencies, and evidence.\n"
            "- Evidence is a list of mappings with source_type, reference, and supports. "
            f"Allowed literature references: {evidence_ids}.\n"
            "- Use these exact literal manifest values (do not rename or omit them): "
            f"'role': {self.name!r}, 'group': {self.group!r}, "
            f"'category': {self.category!r}.\n"
            f"- Use exactly 'parent_node_id': {parent_node_id!r}, "
            f"'parent_model_family': {parent_model_family!r}, and "
            "'input_schema_version': 2.\n"
            f"- Use exactly 'research_family': {research_family_for_role(self)!r}.\n"
            f"- model_family must be one of: {model_family_text}. "
            f"loss_family must be one of: {loss_family_text}.\n"
            "- For every literal True ABLATION_COMPONENTS key, evidence supports "
            "must be a literal non-empty list containing that exact key. Valid forms "
            "are literature references from the catalog, 'validation:<stored-id>', "
            "or 'dependency:<direct-reason>'. Do not use other source_type values.\n"
            "- Example only: if components are 'history_factor' and 'interest_path', "
            "a valid item is {'source_type': 'literature', 'reference': 'din_2018', "
            "'supports': ['history_factor', 'interest_path']}. Replace the example "
            "names with the exact enabled component keys.\n"
            "- mechanism_ids must be a literal list of concise snake_case mechanism "
            "identifiers; use it to distinguish substantive ideas from wording changes.\n"
            "- Every enabled ABLATION_COMPONENTS entry must be called literally through "
            "component_enabled(name) in the code path it controls.\n"
            "- A factor/model role must define literal FEATURE_FACTORS entries with "
            "library_id, name, raw_fields, transform, output_fields, and state_policy; build_features "
            "must create the fields and a guarded model path must consume them.\n"
            "- Every candidate must define a literal FACTOR_SELECTION mapping with "
            "considered_factor_ids, selected_factor_ids, selection_reason, rejected_reasons, "
            "and created_factor_cards. "
            "Consider at least one library card. selected_factor_ids may be empty, but may contain "
            "at most two known IDs and must be a subset of considered_factor_ids. Every selected "
            "ID must have a matching FEATURE_FACTORS library_id and real executable use.\n"
            "- If no existing card describes a useful factor, created_factor_cards may define at "
            "most two new cards. Each needs factor_id beginning with 'custom_', semantics, "
            "helps_when, model_fit, avoid_when, data_cost, and leakage_rule. A custom card is a "
            "small self-description, not permission to add external data.\n"
            "- Every FEATURE_FACTORS output_fields string must also appear literally "
            "inside the build_features function body, for example "
            "features['history_author_ids']; a module constant or manifest-only name "
            "does not prove that the field is created.\n"
            "- State concrete inputs, transformations, tensor shapes, outputs, and why "
            "each change should affect GAUC, nDCG@5, or both.\n"
            "Before answering, privately check syntax, the trusted interfaces, causal "
            "feature state, component guards, checkpoint round-trip, and CUDA tensor "
            "placement. Return only the concise plan and complete code required by the "
            "global response format."
            "\nMachine-readable assignment marker (preserve this assignment in "
            "RESEARCH_MANIFEST; it is parsed before any GPU training):\n"
            + assignment_marker
            +
            "\nThe following role-specific scaffold is syntactically valid. Copy its "
            "literal container shapes and exact identity fields, then make the named "
            "component control the real implementation (do not leave it as metadata only):\n"
            + scaffold_text
            + donor_text
            + recipe_text
            + role_contract
            + factor_text
            + memory_text
            + feedback_text
        )


def autonomous_candidate_prompt(
    role: CandidateRole,
    *,
    index: int,
    total: int,
    retry_feedback: Sequence[str] = (),
    evidence_memory: str = "",
    factor_library_context: str = "",
    parent_node_id: str = "",
    parent_model_family: str = "fm",
    assignment_id: str = "",
    assignment_kind: str = "autonomous",
) -> str:
    """Render one open Stage 3 assignment without choosing its science."""
    feedback_text = (
        "\nConcrete rejection feedback for this slot:\n"
        + "\n".join(f"- {item}" for item in retry_feedback[-3:])
        + "\nAddress the conflict with a substantively different implementation. "
        "No replacement direction is prescribed.\n"
        if retry_feedback
        else ""
    )
    memory_text = (
        "\nValidation-only feedback available to every autonomous slot:\n"
        + evidence_memory.strip()
        + "\n"
        if evidence_memory.strip()
        else ""
    )
    factor_text = (
        "\nOptional factor cards available to every autonomous slot:\n"
        + factor_library_context.strip()
        + "\nThese cards are capabilities, not priorities. Select none, one, or at "
        "most two when they are needed by your hypothesis; another legal mechanism "
        "may be chosen instead.\n"
        if factor_library_context.strip()
        else ""
    )
    assignment_marker = build_assignment_marker(
        role,
        assignment_id=assignment_id or f"autonomous:{index}:{parent_node_id}",
        assignment_kind=assignment_kind,
        parent_node_id=parent_node_id,
        parent_model_family=parent_model_family,
    )
    evidence_ids = ", ".join(sorted(CURATED_EVIDENCE))
    model_families = ", ".join(sorted(ALLOWED_MODEL_FAMILIES))
    research_families = ", ".join(sorted(ALLOWED_RESEARCH_FAMILIES))
    loss_families = ", ".join(sorted(ALLOWED_LOSS_FAMILIES))
    return (
        f"Autonomous candidate slot {index}/{total}\n"
        "No scientific choice has been made for this slot. Inspect the current "
        "incumbent and shared validation feedback, then independently choose one "
        "coherent, evidence-backed hypothesis and implement it completely. You may "
        "change features, architecture, objective, loss/training behavior, or another "
        "legal mechanism when your reasoning supports it. Do not submit a metadata- "
        "or CONFIG-only variation.\n"
        "\nDiversity is checked after generation. Do not imitate another candidate or "
        "rename metadata around the same implementation. A rejected retry reports the "
        "specific conflict but never assigns the replacement idea.\n"
        "\nRequired literal code contract:\n"
        "- Define RESEARCH_MANIFEST with candidate_id, role, group, category, "
        "model_family, research_family, loss_family, parent_node_id, "
        "parent_model_family, input_schema_version, hypothesis, mechanism, "
        "mechanism_ids, modified_symbols, expected_metric, tunable_parameters, "
        "ablation_components, combination_compatibility, change_scope, "
        "component_dependencies, and evidence.\n"
        f"- Use the neutral assignment identity 'role': {role.name!r}, "
        f"'group': {role.group!r}, and 'category': {role.category!r}. These values "
        "identify the slot and do not specify its research direction.\n"
        f"- Use exactly 'parent_node_id': {parent_node_id!r}, "
        f"'parent_model_family': {parent_model_family!r}, and "
        "'input_schema_version': 2.\n"
        "- Self-declare the scientific taxonomy that matches the implementation. "
        f"model_family must be one of [{model_families}]; research_family one of "
        f"[{research_families}]; loss_family one of [{loss_families}]. The taxonomy "
        "is for validation and diversity accounting, not an instruction to choose a "
        "particular family.\n"
        "- hypothesis and mechanism must be concrete. mechanism_ids and "
        "modified_symbols must be non-empty literal snake_case string lists that "
        "describe the real implementation.\n"
        "- Define literal ABLATION_COMPONENTS and guard every enabled component in "
        "its executable path with component_enabled(name). Keep components "
        "independently switchable and list the same names in the manifest.\n"
        "- Evidence must name every enabled component. Use only curated literature "
        f"IDs [{evidence_ids}], stored validation references beginning validation:, "
        "or direct dependency references beginning dependency:.\n"
        "- Define literal FACTOR_SELECTION with considered_factor_ids, "
        "selected_factor_ids, selection_reason, rejected_reasons, and "
        "created_factor_cards. Consider at least one available card. Selecting no "
        "factor is valid with explicit rejection reasons.\n"
        "- If factors are selected, define matching literal FEATURE_FACTORS entries "
        "with library_id, name, raw_fields, transform, output_fields, and state_policy; "
        "build_features must create every output and the guarded prediction path must "
        "consume it. Custom cards are limited to two custom_ snake_case IDs.\n"
        "- Any pairwise/listwise objective must form groups from encoded user IDs in "
        "x_tensor[:, 0] and use only same-user pairs. Any auxiliary target must come "
        "from a real train-window field, never from long_view or later-period logs.\n"
        "- Preserve the trusted split, evaluator, CandidateModel interfaces, checkpoint "
        "round-trip, CUDA placement, and validation-only feedback. Fit all learned "
        "feature state on train only and compute histories causally.\n"
        "- Keep a new prototype at or below 12 epochs and vectorize batch computation. "
        "Return only the concise plan and complete runnable code required by the "
        "global response format.\n"
        + factor_text
        + memory_text
        + feedback_text
        + "\nMachine-readable assignment marker; this must remain the final line of "
        "the assignment:\n"
        + assignment_marker
    )


DEFAULT_CANDIDATE_ROLES: tuple[CandidateRole, ...] = (
    CandidateRole(
        "causal_history_interest",
        "history_interest",
        "factor_model",
        "Create causal history factors and a matching interest encoder that actually consumes their embeddings.",
        "build_features must create real history fields and the guarded model path must consume them.",
        allowed_loss_families=("pointwise_bce", "hybrid_bce_bpr"),
    ),
    CandidateRole(
        "affinity_interest",
        "history_interest",
        "factor_model",
        "Create causal user-author/tag/type affinity factors and a matching gated or attention interest model.",
        "Declare raw-to-factor mappings and prove the guarded model path uses every added field.",
        allowed_loss_families=("pointwise_bce", "hybrid_bce_bpr"),
    ),
    CandidateRole(
        "ranking_objective",
        "objective_and_training",
        "training_objective",
        "Change optimization toward GAUC/nDCG with a controlled hybrid pointwise and pairwise/listwise loss.",
        "The loss path must be guarded as an ablation component and keep long_view as the primary target.",
        allowed_loss_families=("hybrid_bce_bpr", "pairwise_bpr", "listwise_softmax"),
    ),
    CandidateRole(
        "auxiliary_objective",
        "objective_and_training",
        "training_objective",
        "Use legal training-window auxiliary behavior or watch-time targets without exposing validation/test outcomes.",
        "The auxiliary head/loss and its weight must be explicit, guarded, and absent from inference inputs.",
        allowed_loss_families=("multitask", "censored_watch_time"),
    ),
    CandidateRole(
        "context_interaction",
        "context_interaction",
        "factor_model",
        "Create static user/video and crossed context factors with a matching DeepFM/DCN/xDeepFM interaction block.",
        "The feature builder and guarded interaction path must both change and remain independently ablatable.",
        allowed_loss_families=("pointwise_bce", "hybrid_bce_bpr"),
    ),
    CandidateRole(
        "temporal_interaction",
        "context_interaction",
        "factor_model",
        "Create causal time/recency factors with a matching interaction or robust temporal model.",
        "State cutoff behavior and prove the vectorized guarded model path consumes the temporal fields.",
        allowed_loss_families=("pointwise_bce", "hybrid_bce_bpr"),
    ),
    CandidateRole(
        "incumbent_extension",
        "evidence_combination",
        "evidence_synthesis",
        "Extend the incumbent along the strongest positive evidence. If no prior direction is convincingly positive, choose one coherent open theme that addresses the current validation weakness.",
        "Name the inherited and new components, or explain why open exploration is preferable; guard every new component independently.",
        allowed_loss_families=("pointwise_bce", "hybrid_bce_bpr", "pairwise_bpr", "listwise_softmax", "multitask", "censored_watch_time"),
    ),
    CandidateRole(
        "cross_direction_synthesis",
        "evidence_combination",
        "evidence_synthesis",
        "Combine mutually supportive mechanisms backed by prior validation, ablation memory, or established recommendation research.",
        "Name the evidence for the combination, declare dependencies, and guard every component independently.",
        allowed_loss_families=("pointwise_bce", "hybrid_bce_bpr", "pairwise_bpr", "listwise_softmax", "multitask", "censored_watch_time"),
    ),
)


ALLOWED_MODEL_FAMILIES = frozenset(
    {
        "fm",
        "mlp",
        "wide_deep",
        "deepfm",
        "dcn",
        "xdeepfm",
        "ncf",
        "din_lite",
        "multitask_shared_bottom",
        "hybrid",
    }
)
ALLOWED_LOSS_FAMILIES = frozenset(
    {
        "pointwise_bce",
        "hybrid_bce_bpr",
        "pairwise_bpr",
        "listwise_softmax",
        "multitask",
        "censored_watch_time",
    }
)
ALLOWED_RESEARCH_FAMILIES = frozenset(
    {
        "architecture",
        "feature_engineering",
        "history_interest",
        "ranking_objective",
        "auxiliary_objective",
        "context_interaction",
        "temporal_interaction",
        "training_strategy",
        "evidence_synthesis",
    }
)

AUTONOMOUS_STAGE3_ROLE = CandidateRole(
    name="autonomous_stage3",
    group="autonomous_research",
    category="open_choice",
    objective="Independently choose and implement one evidence-backed research hypothesis.",
    required_evidence="The declared hypothesis must match executable code and legal evidence.",
    autonomous=True,
)


def bootstrap_candidate_roles(model_families: Sequence[str]) -> tuple[CandidateRole, ...]:
    """Create Stage 1B roles with an exact, machine-checkable family target."""
    roles = []
    for family in model_families:
        family = str(family)
        if family not in ALLOWED_MODEL_FAMILIES or family == "fm":
            continue
        roles.append(
            CandidateRole(
                name=f"architecture_{family}",
                group="architecture_exploration",
                category="model_architecture",
                objective=(
                    f"Replace the FM scorer with one lightweight {family} model while "
                    "keeping schema v2, validation-only evaluation, and primary unchanged."
                ),
                required_evidence=(
                    "create_model and the model class must materially implement the declared "
                    "family; cap bootstrap training at the prompt-provided epoch budget."
                ),
                allowed_model_families=(family,),
                allowed_loss_families=("pointwise_bce",),
            )
        )
    return tuple(roles)


CURATED_EVIDENCE = {
    "bpr_pairwise_implicit_2009": "BPR: Bayesian Personalized Ranking from Implicit Feedback",
    "wide_deep_2016": "Wide & Deep Learning for Recommender Systems",
    "deepfm_2017": "DeepFM: A Factorization-Machine based Neural Network",
    "dcn_2017": "Deep & Cross Network for Ad Click Predictions",
    "din_2018": "Deep Interest Network for Click-Through Rate Prediction",
    "esmm_2018": "Entire Space Multi-Task Model for Conversion Rate Prediction",
    "mmoe_2018": "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts",
    "dien_2019": "Deep Interest Evolution Network for Click-Through Rate Prediction",
    "autoint_2019": "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks",
}


TECHNIQUE_CATALOG = {
    "causal_history_interest": (
        "First choice: causal recent-20/50/100 author, tag, duration, and recency profiles with a lightweight DIN-style target-conditioned interest encoder.",
        "Escalate only after a working profile/DIN result: DIEN or SIM; avoid a large SASRec/GRU stack as the first prototype.",
        "Use only interactions earlier than the target and freeze all outcome-derived state at the end of train.",
    ),
    "affinity_interest": (
        "Build past-only user-author, user-tag, and user-video-type affinities and consume them through a small gate or attention block.",
        "For unknown IDs, mix ID embeddings with author/tag/duration metadata rather than relying on one UNK vector.",
        "A segmentation or mixture-of-experts gate is a later option only when history length/activity groups show different errors.",
    ),
    "context_interaction": (
        "Combine user/video metadata, explicit user-author/tag/duration crosses, and a small DCN or DeepFM interaction block.",
        "Train-label rates require out-of-fold or leave-one-out encoding for train rows and train-only frozen smoothing for validation/test.",
        "Keep the official five fields as an ablation reference; do not modify protected input/data.py or input/evaluate.py.",
    ),
    "temporal_interaction": (
        "Try causal hour/day/recency factors, 3/7/14-day decay, or train-only distribution weighting with a small matching interaction model.",
        "Prefer time-aware weighting and compact context before adding model depth; keep the official chronological split unchanged.",
        "Embedding/feature dropout and L2 may support temporal robustness but must be separately ablatable.",
    ),
    "ranking_objective": (
        "First compare pointwise BCE with a hybrid BCE+BPR objective, then hard negatives (50% model-hard, 50% random).",
        "Escalate to LambdaLoss, ApproxNDCG, or position-weighted pairwise loss only after the BPR path is valid.",
        "Keep default BCE as an ablation control and never use validation labels for negative mining.",
    ),
    "auxiliary_objective": (
        "Use long_view as the main task and legal train-window click/like/follow/play-time signals as auxiliary targets with small initial weights.",
        "Soft watch-ratio or censored watch-time losses are training targets only, never current-exposure inference features.",
        "Start with a shared-bottom model; use MMoE/PLE only if simple auxiliary learning is positive and task conflict is observed.",
    ),
    "incumbent_extension": (
        "Extend the incumbent with one central evidence-backed mechanism and any factors/objectives it functionally requires.",
        "Prefer components with positive stored ablation evidence; do not re-add a component previously classified harmful unchanged.",
        "When prior evidence is weak or flat, use this slot for one new, well-motivated theme rather than forcing an incumbent extension.",
    ),
    "cross_direction_synthesis": (
        "Combine components only when curated literature, stored validation evidence, or a direct dependency supports the pairing.",
        "Keep every component independently switchable so Stage 4 can estimate conditional contribution and key pair synergy.",
    ),
}


ROLE_EXECUTION_RECIPES = {
    "causal_history_interest": (
        "Define literal FEATURE_FACTORS with a known library_id and concrete output_fields.",
        "Create every output field literally inside build_features using past-only train history.",
        "Pass the new tensor through create_model and consume it inside a component_enabled guard.",
        "Keep unknown/empty history behavior deterministic and checkpoint the fitted history state.",
    ),
    "affinity_interest": (
        "Build train-only smoothed user-author or user-tag affinity state and freeze it for validation.",
        "Declare exact raw_fields/output_fields and consume the resulting tensor in the guarded model path.",
        "Never use the current row label or validation outcomes to build affinity state.",
    ),
    "context_interaction": (
        "Implement one DeepFM/DCN/xDeepFM interaction path in create_model, not metadata-only changes.",
        "Keep the official schema-v2 adapter and make every added factor executable and ablatable.",
    ),
    "ranking_objective": (
        "Read user_ids from x_tensor[:, 0] inside step.",
        "Construct an explicit same_user equality mask before selecting positive-negative pairs.",
        "Combine the ranking term with BCE through one registered loss component and handle empty pairs.",
    ),
    "auxiliary_objective": (
        "Load an allowed auxiliary label only from the training-window log and align by stable exposure keys.",
        "Keep long_view as the primary head and exclude auxiliary labels from inference features.",
    ),
    "cross_parent_transfer": (
        "Start from the incumbent code, not the donor code.",
        "Transfer exactly one donor mechanism and preserve every unrelated incumbent path.",
        "Use one new component_enabled guard and declare the donor/mechanism in the manifest.",
        "Do not transfer donor CONFIG, checkpoint, split, evaluator, or unrelated architecture blocks.",
    ),
    "default": (
        "Change one executable scientific mechanism, not CONFIG or metadata alone.",
        "Register and call every new component through component_enabled.",
        "Preserve validation-only evaluation and checkpoint round-trip interfaces.",
    ),
}


REQUIRED_MANIFEST_FIELDS = (
    "candidate_id",
    "role",
    "group",
    "category",
    "model_family",
    "research_family",
    "loss_family",
    "parent_node_id",
    "parent_model_family",
    "input_schema_version",
    "hypothesis",
    "mechanism",
    "mechanism_ids",
    "modified_symbols",
    "expected_metric",
    "tunable_parameters",
    "ablation_components",
    "combination_compatibility",
    "change_scope",
    "component_dependencies",
    "evidence",
)

ASSIGNMENT_MARKER_PREFIX = "BYTE_RUSH_ASSIGNMENT_JSON="


def build_assignment_marker(
    role: CandidateRole,
    *,
    assignment_id: str,
    assignment_kind: str,
    parent_node_id: str,
    parent_model_family: str,
) -> str:
    payload = {
        "assignment_id": assignment_id,
        "assignment_kind": assignment_kind,
        "autonomous": role.autonomous,
        "role": role.name,
        "group": role.group,
        "category": role.category,
        "objective": role.objective,
        "required_evidence": role.required_evidence,
        "allowed_model_families": list(role.allowed_model_families),
        "allowed_loss_families": list(role.allowed_loss_families),
        "parent_node_id": parent_node_id,
        "parent_model_family": parent_model_family,
    }
    return ASSIGNMENT_MARKER_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ) + "\n"


def extract_assignment_contract(text: str) -> Mapping[str, Any] | None:
    """Read the last assignment marker from an assembled worker prompt."""
    payload = None
    for line in str(text).splitlines():
        if not line.startswith(ASSIGNMENT_MARKER_PREFIX):
            continue
        try:
            candidate = json.loads(line[len(ASSIGNMENT_MARKER_PREFIX) :])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(candidate, Mapping):
            payload = candidate
    return payload


def role_from_assignment(payload: Mapping[str, Any]) -> CandidateRole:
    return CandidateRole(
        name=str(payload.get("role", "")),
        group=str(payload.get("group", "")),
        category=str(payload.get("category", "")),
        objective=str(payload.get("objective", "")),
        required_evidence=str(payload.get("required_evidence", "")),
        allowed_model_families=tuple(
            map(str, payload.get("allowed_model_families", ()) or ())
        ),
        allowed_loss_families=tuple(
            map(str, payload.get("allowed_loss_families", ()) or ())
        ),
        autonomous=payload.get("autonomous") is True,
    )


def classify_contract_failures(reasons: Sequence[str]) -> str:
    joined = " ".join(map(str, reasons)).lower()
    if "duplicate" in joined:
        return "duplicate"
    if "same-user" in joined or "same user" in joined or "ranking objective" in joined:
        return "ranking_group_error"
    if "feature_factors" in joined or "build_features" in joined:
        return "feature_implementation_error"
    if "component_enabled" in joined or "ablation" in joined:
        return "ablation_guard_error"
    if "manifest" in joined or "factor_selection" in joined:
        return "schema_error"
    if "create_model" in joined or "model_family" in joined:
        return "model_implementation_error"
    if "test" in joined or "leak" in joined:
        return "leakage_error"
    return "contract_error"


def rewrite_for_smoke_test(code: str) -> str:
    """Return a one-epoch version without changing the scientific mechanism."""
    tree = _tree(code)
    if tree is None:
        return code
    changed = False
    for statement in tree.body:
        is_config = (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CONFIG"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Dict)
        )
        if not is_config:
            continue
        for key_node, value_node in zip(statement.value.keys, statement.value.values):
            if not isinstance(key_node, ast.Constant):
                continue
            if (
                key_node.value in {"max_epochs", "epochs", "patience"}
                and isinstance(value_node, ast.Constant)
            ):
                value_node.value = 1
                changed = True
        break
    if not changed:
        return code
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def model_family_from_code(code: str, *, default: str = "fm") -> str:
    manifest = literal_assignment(code, "RESEARCH_MANIFEST")
    if isinstance(manifest, Mapping):
        family = str(manifest.get("model_family", ""))
        if family in ALLOWED_MODEL_FAMILIES:
            return family
    return default


def research_family_for_role(role: CandidateRole) -> str:
    if role.group == "architecture_exploration":
        return "architecture"
    if role.name.startswith("ranking_objective"):
        return "ranking_objective"
    if role.name.startswith("auxiliary_objective"):
        return "auxiliary_objective"
    if role.name.startswith("temporal_interaction"):
        return "temporal_interaction"
    if role.group in ALLOWED_RESEARCH_FAMILIES:
        return role.group
    if role.group == "objective_and_training":
        return "ranking_objective"
    if role.group == "context_interaction":
        return "context_interaction"
    return "evidence_synthesis"


def _snake_case(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "_" for character in value.lower()
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "candidate_component"


def _default_factor_id(role: CandidateRole) -> str:
    base_name = role.name.split("_alternative_", 1)[0]
    return {
        "causal_history_interest": "causal_recent_history",
        "affinity_interest": "user_author_affinity",
        "context_interaction": "user_item_context_cross",
        "temporal_interaction": "temporal_recency_context",
        "auxiliary_objective": "auxiliary_behavior_signal",
    }.get(base_name, "static_user_profile")


def candidate_contract_scaffold(
    role: CandidateRole,
    *,
    parent_node_id: str,
    parent_model_family: str,
) -> str:
    """Render a compact literal scaffold that matches the static validator."""
    family = (
        role.allowed_model_families[0]
        if role.allowed_model_families
        else parent_model_family
    )
    loss_family = (
        role.allowed_loss_families[0]
        if role.allowed_loss_families
        else "pointwise_bce"
    )
    component = _snake_case(role.name + "_component")
    factor_id = _default_factor_id(role)
    factor_role = role.category in {"factor_data", "factor_model"}
    selection = {
        "considered_factor_ids": [factor_id],
        "selected_factor_ids": [factor_id] if factor_role else [],
        "selection_reason": (
            "selected factor is required by the assigned factor/model mechanism"
            if factor_role
            else "the assigned mechanism is architecture/objective-only"
        ),
        "rejected_reasons": (
            {}
            if factor_role
            else {factor_id: "not required by this architecture/objective-only candidate"}
        ),
        "created_factor_cards": [],
    }
    manifest = {
        "candidate_id": role.name,
        "role": role.name,
        "group": role.group,
        "category": role.category,
        "model_family": family,
        "research_family": research_family_for_role(role),
        "loss_family": loss_family,
        "parent_node_id": parent_node_id,
        "parent_model_family": parent_model_family,
        "input_schema_version": 2,
        "hypothesis": "the assigned controlled mechanism improves validation ranking",
        "mechanism": role.objective,
        "mechanism_ids": [_snake_case(role.name)],
        "modified_symbols": ["build_features", "create_model", "train_model"],
        "expected_metric": ["GAUC", "nDCG@5", "primary"],
        "tunable_parameters": [],
        "ablation_components": [component],
        "combination_compatibility": "single independently guarded component",
        "change_scope": "one principal research mechanism",
        "component_dependencies": {},
        "evidence": [
            {
                "source_type": "dependency",
                "reference": "dependency:assigned_mechanism_requires_this_code_path",
                "supports": [component],
            }
        ],
    }
    factor_note = (
        "\n# Because this is a factor/model role, also define FEATURE_FACTORS with "
        f"library_id={factor_id!r}; its literal output_fields must be created in "
        "build_features and consumed by the guarded model path."
        if factor_role
        else "\nFEATURE_FACTORS = []"
    )
    return (
        f"RESEARCH_MANIFEST = {manifest!r}\n"
        f"FACTOR_SELECTION = {selection!r}"
        f"{factor_note}\n"
        f"ABLATION_COMPONENTS = {{{component!r}: True}}\n"
        "def component_enabled(name):\n"
        "    return bool(ABLATION_COMPONENTS.get(name, False))\n"
        f"# In the real changed code path: if component_enabled({component!r}): ...\n"
    )


def _replace_or_append_literal_assignment(code: str, name: str, value: Any) -> str:
    """Replace metadata without touching executable model/training statements."""
    tree = _tree(code)
    if tree is None:
        return code
    lines = code.splitlines()
    for statement in tree.body:
        matches = (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            )
        ) or (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        )
        if not matches:
            continue
        start = max(0, statement.lineno - 1)
        end = max(start + 1, getattr(statement, "end_lineno", statement.lineno))
        lines[start:end] = [f"{name} = {value!r}"]
        return "\n".join(lines) + "\n"
    suffix = "" if code.endswith("\n") else "\n"
    return code + suffix + f"\n{name} = {value!r}\n"


def normalize_candidate_metadata(
    candidate_code: str,
    role: CandidateRole,
    *,
    expected_parent_id: str,
    expected_parent_model_family: str,
) -> str:
    """Repair only declarative metadata after a successful execution.

    This deliberately does not invent feature/model/loss code or component
    guards. Scientific and leakage checks still reject candidates whose actual
    implementation does not match the assigned role.
    """
    if _tree(candidate_code) is None:
        return candidate_code
    if role.autonomous:
        return candidate_code
    components = literal_assignment(candidate_code, "ABLATION_COMPONENTS")
    enabled = sorted(
        str(name)
        for name, enabled_value in (components.items() if isinstance(components, Mapping) else ())
        if isinstance(name, str) and enabled_value is True
    )
    if not enabled:
        return candidate_code

    manifest = literal_assignment(candidate_code, "RESEARCH_MANIFEST")
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    declared_model_family = str(manifest.get("model_family", ""))
    model_family = (
        declared_model_family
        if declared_model_family in ALLOWED_MODEL_FAMILIES
        else role.allowed_model_families[0]
        if role.allowed_model_families
        else expected_parent_model_family
    )
    loss_family = str(manifest.get("loss_family", ""))
    if loss_family not in ALLOWED_LOSS_FAMILIES:
        loss_family = (
            role.allowed_loss_families[0]
            if role.allowed_loss_families
            else "pointwise_bce"
        )
    mechanism_ids = manifest.get("mechanism_ids")
    if not isinstance(mechanism_ids, (list, tuple)) or not mechanism_ids:
        mechanism_ids = [_snake_case(role.name)]
    manifest.update(
        {
            "candidate_id": str(manifest.get("candidate_id") or role.name),
            "role": role.name,
            "group": role.group,
            "category": role.category,
            "model_family": model_family,
            "research_family": research_family_for_role(role),
            "loss_family": loss_family,
            "parent_node_id": expected_parent_id,
            "parent_model_family": expected_parent_model_family,
            "input_schema_version": 2,
            "hypothesis": str(
                manifest.get("hypothesis")
                or "the assigned mechanism improves validation ranking"
            ),
            "mechanism": str(manifest.get("mechanism") or role.objective),
            "mechanism_ids": [_snake_case(str(item)) for item in mechanism_ids],
            "modified_symbols": list(
                manifest.get("modified_symbols")
                if isinstance(manifest.get("modified_symbols"), (list, tuple))
                else ["build_features", "create_model", "train_model"]
            ),
            "expected_metric": list(
                manifest.get("expected_metric")
                if isinstance(manifest.get("expected_metric"), (list, tuple))
                else ["GAUC", "nDCG@5", "primary"]
            ),
            "tunable_parameters": list(
                manifest.get("tunable_parameters")
                if isinstance(manifest.get("tunable_parameters"), (list, tuple))
                else []
            ),
            "ablation_components": enabled,
            "combination_compatibility": str(
                manifest.get("combination_compatibility")
                or "components are independently guarded"
            ),
            "change_scope": str(
                manifest.get("change_scope") or "one principal research mechanism"
            ),
            "component_dependencies": (
                dict(manifest.get("component_dependencies"))
                if isinstance(manifest.get("component_dependencies"), Mapping)
                else {}
            ),
            "evidence": [
                {
                    "source_type": "dependency",
                    "reference": "dependency:executed_guarded_component",
                    "supports": enabled,
                }
            ],
        }
    )
    normalized = _replace_or_append_literal_assignment(
        candidate_code, "RESEARCH_MANIFEST", manifest
    )

    raw_factors = literal_assignment(normalized, "FEATURE_FACTORS")
    declared_factor_ids = [
        str(item.get("library_id", ""))
        for item in raw_factors
        if isinstance(raw_factors, (list, tuple)) and isinstance(item, Mapping)
        and str(item.get("library_id", "")) in FACTOR_IDS
    ] if isinstance(raw_factors, (list, tuple)) else []
    selection = literal_assignment(normalized, "FACTOR_SELECTION")
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    selected = [
        str(item)
        for item in selection.get("selected_factor_ids", [])
        if str(item) in FACTOR_IDS and str(item) in declared_factor_ids
    ]
    factor_role = role.category in {"factor_data", "factor_model"}
    if factor_role and not selected and declared_factor_ids:
        selected = declared_factor_ids[:2]
    considered = [
        str(item)
        for item in selection.get("considered_factor_ids", [])
        if str(item) in FACTOR_IDS
    ]
    for factor_id in selected:
        if factor_id not in considered:
            considered.append(factor_id)
    if not considered:
        considered = [_default_factor_id(role)]
    rejected = selection.get("rejected_reasons")
    rejected = dict(rejected) if isinstance(rejected, Mapping) else {}
    for factor_id in considered:
        if factor_id not in selected:
            rejected.setdefault(
                factor_id, "not required by the implemented principal mechanism"
            )
    selection.update(
        {
            "considered_factor_ids": considered,
            "selected_factor_ids": selected,
            "selection_reason": str(
                selection.get("selection_reason")
                or "selected factors match the executed implementation"
            ),
            "rejected_reasons": rejected,
            "created_factor_cards": (
                list(selection.get("created_factor_cards"))
                if isinstance(selection.get("created_factor_cards"), (list, tuple))
                else []
            ),
        }
    )
    return _replace_or_append_literal_assignment(
        normalized, "FACTOR_SELECTION", selection
    )


@dataclass(frozen=True)
class CandidateContractResult:
    valid: bool
    reasons: tuple[str, ...]
    manifest: Mapping[str, Any]
    feature_factors: tuple[Mapping[str, Any], ...]
    factor_selection: Mapping[str, Any]


@dataclass(frozen=True)
class TuningContractResult:
    valid: bool
    reasons: tuple[str, ...]
    config: Mapping[str, Any]


def _tree(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except (SyntaxError, TypeError):
        return None


def literal_assignment(code: str, name: str) -> Any:
    tree = _tree(code)
    if tree is None:
        return None
    for statement in tree.body:
        value = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ):
            value = statement.value
        if value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                return None
    return None


def config_assignment(code: str) -> Mapping[str, Any] | None:
    """Parse CONFIG while preserving trusted dynamic expressions as AST text."""
    tree = _tree(code)
    if tree is None:
        return None
    for statement in tree.body:
        value = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CONFIG"
            for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "CONFIG"
        ):
            value = statement.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            return None
        parsed: dict[str, Any] = {}
        for key_node, value_node in zip(value.keys, value.values):
            if key_node is None:
                return None
            try:
                key = ast.literal_eval(key_node)
            except (ValueError, TypeError, SyntaxError):
                return None
            if not isinstance(key, str) or key in parsed:
                return None
            try:
                parsed[key] = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError):
                parsed[key] = {
                    "__ast_expression__": ast.dump(
                        value_node,
                        include_attributes=False,
                    )
                }
        return parsed
    return None


def _is_config_expression(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"__ast_expression__"}
        and isinstance(value.get("__ast_expression__"), str)
    )


def restore_dynamic_config_fields(base_code: str, candidate_code: str) -> str:
    """Restore runtime-controlled CONFIG expressions before a tuning run."""
    base_tree = _tree(base_code)
    candidate_tree = _tree(candidate_code)
    if base_tree is None or candidate_tree is None:
        return candidate_code

    def config_node(tree: ast.Module) -> ast.Dict | None:
        for statement in tree.body:
            value = None
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CONFIG"
                for target in statement.targets
            ):
                value = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "CONFIG"
            ):
                value = statement.value
            if value is not None:
                return value if isinstance(value, ast.Dict) else None
        return None

    base_node = config_node(base_tree)
    candidate_node = config_node(candidate_tree)
    if base_node is None or candidate_node is None:
        return candidate_code

    def fields(node: ast.Dict) -> dict[str, tuple[int, ast.expr]] | None:
        result = {}
        for index, (key_node, value_node) in enumerate(zip(node.keys, node.values)):
            if key_node is None:
                return None
            try:
                key = ast.literal_eval(key_node)
            except (ValueError, TypeError, SyntaxError):
                return None
            if not isinstance(key, str) or key in result:
                return None
            result[key] = (index, value_node)
        return result

    base_fields = fields(base_node)
    candidate_fields = fields(candidate_node)
    if (
        base_fields is None
        or candidate_fields is None
        or set(base_fields) != set(candidate_fields)
    ):
        return candidate_code

    changed = False
    for key, (_, base_value) in base_fields.items():
        try:
            ast.literal_eval(base_value)
            continue
        except (ValueError, TypeError, SyntaxError):
            pass
        candidate_index, candidate_value = candidate_fields[key]
        if ast.dump(candidate_value, include_attributes=False) != ast.dump(
            base_value,
            include_attributes=False,
        ):
            candidate_node.values[candidate_index] = copy.deepcopy(base_value)
            changed = True

    if not changed:
        return candidate_code
    ast.fix_missing_locations(candidate_tree)
    return ast.unparse(candidate_tree) + "\n"


def component_guard_calls(code: str) -> set[str]:
    tree = _tree(code)
    if tree is None:
        return set()
    calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "component_enabled" or len(node.args) != 1:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            calls.add(argument.value)
    return calls


def function_dump(code: str, function_name: str) -> str | None:
    tree = _tree(code)
    if tree is None:
        return None
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == function_name:
            return ast.dump(statement, include_attributes=False)
    return None


def _function_string_constants(code: str, function_name: str) -> set[str]:
    tree = _tree(code)
    if tree is None:
        return set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == function_name:
            return {
                node.value
                for node in ast.walk(statement)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
    return set()


def _named_function_nodes(code: str, function_name: str) -> tuple[ast.AST, ...]:
    tree = _tree(code)
    if tree is None:
        return ()
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )


def _is_full_slice(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _is_encoded_user_column(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id not in {
        "x",
        "x_tensor",
        "batch_x",
    }:
        return False
    if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
        return False
    rows, column = node.slice.elts
    return (
        _is_full_slice(rows)
        and isinstance(column, ast.Constant)
        and column.value == 0
    )


def _ranking_objective_reasons(candidate_code: str) -> list[str]:
    step_nodes = _named_function_nodes(candidate_code, "step")
    if not step_nodes:
        return ["ranking objective has no step method"]

    user_names: set[str] = set()
    for step in step_nodes:
        for node in ast.walk(step):
            value = None
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                value = node.value
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = [node.target]
            if value is None or not _is_encoded_user_column(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    user_names.add(target.id)

    if not user_names:
        return [
            "ranking objective must extract encoded users from x_tensor[:, 0] "
            "inside step"
        ]

    def contains_user_reference(node: ast.AST) -> bool:
        return any(
            (isinstance(item, ast.Name) and item.id in user_names)
            or _is_encoded_user_column(item)
            for item in ast.walk(node)
        )

    grouped = False
    for step in step_nodes:
        for node in ast.walk(step):
            if isinstance(node, ast.Compare) and any(
                isinstance(operator, (ast.Eq, ast.NotEq))
                for operator in node.ops
            ):
                if contains_user_reference(node):
                    grouped = True
                    break
        if grouped:
            break

    if not grouped:
        return [
            "ranking objective extracts users but does not construct "
            "same-user groups or equality masks"
        ]
    return []


_AUXILIARY_RAW_FIELDS = {
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
}
_PRIMARY_LABEL_NAMES = {
    "y",
    "y_tensor",
    "train_y",
    "valid_y",
    "long_view",
    "long_view_y",
    "long_view_labels",
}


def _expression_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
    }


def _mapping_access_key(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _auxiliary_objective_reasons(candidate_code: str) -> list[str]:
    tree = _tree(candidate_code)
    if tree is None:
        return ["auxiliary objective code is not valid Python"]

    assignments: dict[str, set[str]] = {}
    auxiliary_targets: set[str] = set()
    for node in ast.walk(tree):
        value = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if value is None:
            continue
        dependencies = _expression_names(value)
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            assignments.setdefault(target.id, set()).update(dependencies)
            lowered = target.id.lower()
            if (
                "aux" in lowered
                or "click" in lowered
                or "like" in lowered
                or "follow" in lowered
                or "play_time" in lowered
                or "watch_target" in lowered
            ):
                auxiliary_targets.add(target.id)

    tainted = set(_PRIMARY_LABEL_NAMES)
    changed = True
    while changed:
        changed = False
        for name, dependencies in assignments.items():
            if name not in tainted and dependencies & tainted:
                tainted.add(name)
                changed = True

    copied_primary = set(auxiliary_targets & tainted)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg
                and "aux" in keyword.arg.lower()
                and _expression_names(keyword.value) & tainted
            ):
                copied_primary.add(keyword.arg)
    if copied_primary:
        return [
            "auxiliary targets derive from the primary long_view label: "
            + ", ".join(sorted(copied_primary))
        ]

    executable_functions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    raw_fields = {
        key
        for function in executable_functions
        for item in ast.walk(function)
        if (key := _mapping_access_key(item)) in _AUXILIARY_RAW_FIELDS
    }
    if not raw_fields:
        return [
            "auxiliary objective does not read a real train-window auxiliary field"
        ]

    executable_strings = {
        item.value
        for function in executable_functions
        for item in ast.walk(function)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    if "log_standard_4_08_to_4_21_pure.csv" not in executable_strings:
        return [
            "auxiliary objective must source labels from the train-only log file"
        ]
    if "log_standard_4_22_to_5_08_pure.csv" in executable_strings:
        return [
            "auxiliary objective reads the validation/test-period log"
        ]
    return []


def _substantive_dump(code: str) -> str | None:
    tree = _tree(code)
    if tree is None:
        return None
    ignored = {"CONFIG", "RESEARCH_MANIFEST", "FEATURE_FACTORS", "ABLATION_COMPONENTS"}
    body = []
    for statement in tree.body:
        assigned = set()
        if isinstance(statement, ast.Assign):
            assigned = {
                target.id for target in statement.targets if isinstance(target, ast.Name)
            }
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            assigned = {statement.target.id}
        if assigned & ignored:
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        body.append(statement)
    tree.body = body
    return ast.dump(tree, include_attributes=False)


def is_config_only_change(base_code: str, candidate_code: str) -> bool:
    base = _substantive_dump(base_code)
    candidate = _substantive_dump(candidate_code)
    return base is not None and base == candidate


def validate_tuning_contract(
    base_code: str,
    candidate_code: str,
    *,
    tried_configs: Iterable[Mapping[str, Any]] = (),
) -> TuningContractResult:
    """Require tuning nodes to change one literal CONFIG and no algorithm code."""
    reasons: list[str] = []
    base_config = config_assignment(base_code)
    candidate_config = config_assignment(candidate_code)
    if not isinstance(candidate_config, Mapping):
        candidate_config = {}
        reasons.append("missing explicit CONFIG mapping")
    if not isinstance(base_config, Mapping):
        reasons.append("base code has no explicit CONFIG mapping")
    else:
        if set(candidate_config) != set(base_config):
            reasons.append("CONFIG keys changed")
        changed_literals = []
        for key, base_value in base_config.items():
            if key not in candidate_config:
                continue
            candidate_value = candidate_config[key]
            if _is_config_expression(base_value):
                if candidate_value != base_value:
                    reasons.append(f"dynamic CONFIG expression changed: {key}")
            elif _is_config_expression(candidate_value):
                reasons.append(f"literal CONFIG value became executable: {key}")
            elif candidate_value != base_value:
                changed_literals.append(key)
        if not changed_literals:
            reasons.append("no literal CONFIG value changed")
    if not is_config_only_change(base_code, candidate_code):
        reasons.append("tuning changed code outside CONFIG or metadata")
    if any(dict(candidate_config) == dict(previous) for previous in tried_configs):
        reasons.append("duplicate previously executed CONFIG")
    return TuningContractResult(
        valid=not reasons,
        reasons=tuple(reasons),
        config=candidate_config,
    )


def validate_candidate_contract(
    base_code: str,
    candidate_code: str,
    role: CandidateRole,
    *,
    expected_parent_id: str | None = None,
    expected_parent_model_family: str | None = None,
) -> CandidateContractResult:
    reasons: list[str] = []
    candidate_config = config_assignment(candidate_code) or {}
    epochs = candidate_config.get(
        "max_epochs", candidate_config.get("epochs")
    )
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or not 1 <= epochs <= MAX_RESEARCH_PROTOTYPE_EPOCHS
    ):
        reasons.append(
            "CONFIG must use a literal max_epochs/epochs integer between 1 and "
            f"{MAX_RESEARCH_PROTOTYPE_EPOCHS}"
        )
    model_family = ""
    research_family = ""
    loss_family = ""
    parent_family = str(
        expected_parent_model_family or model_family_from_code(base_code)
    )
    manifest = literal_assignment(candidate_code, "RESEARCH_MANIFEST")
    if not isinstance(manifest, Mapping):
        manifest = {}
        reasons.append("missing literal RESEARCH_MANIFEST")
    else:
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
        if missing:
            reasons.append("RESEARCH_MANIFEST missing: " + ", ".join(missing))
        if manifest.get("role") != role.name:
            reasons.append(
                f"manifest role {manifest.get('role')!r} does not match {role.name!r}"
            )
        if manifest.get("group") != role.group:
            reasons.append(
                f"manifest group {manifest.get('group')!r} does not match {role.group!r}"
            )
        if manifest.get("category") != role.category:
            reasons.append(
                f"manifest category {manifest.get('category')!r} does not match {role.category!r}"
            )
        model_family = str(manifest.get("model_family", ""))
        if model_family not in ALLOWED_MODEL_FAMILIES:
            reasons.append(f"unsupported model_family: {model_family!r}")
        if role.allowed_model_families and model_family not in role.allowed_model_families:
            reasons.append(
                f"model_family {model_family!r} violates role family constraint "
                f"{role.allowed_model_families!r}"
            )
        if (
            not role.autonomous
            and role.group != "architecture_exploration"
            and model_family != parent_family
        ):
            reasons.append(
                "non-architecture candidate changed model family from "
                f"{parent_family!r} to {model_family!r}"
            )
        manifest_parent_family = str(manifest.get("parent_model_family", ""))
        if manifest_parent_family != parent_family:
            reasons.append(
                f"manifest parent_model_family {manifest_parent_family!r} does not "
                f"match actual parent family {parent_family!r}"
            )
        if expected_parent_id is not None and manifest.get("parent_node_id") != expected_parent_id:
            reasons.append(
                f"manifest parent_node_id {manifest.get('parent_node_id')!r} does not "
                f"match actual parent {expected_parent_id!r}"
            )
        if manifest.get("input_schema_version") != 2:
            reasons.append("input_schema_version must be literal integer 2")
        research_family = str(manifest.get("research_family", ""))
        if research_family not in ALLOWED_RESEARCH_FAMILIES:
            reasons.append(f"unsupported research_family: {research_family!r}")
        elif (
            not role.autonomous
            and research_family != research_family_for_role(role)
        ):
            reasons.append(
                f"research_family {research_family!r} does not match role family "
                f"{research_family_for_role(role)!r}"
            )
        loss_family = str(manifest.get("loss_family", ""))
        if loss_family not in ALLOWED_LOSS_FAMILIES:
            reasons.append(f"unsupported loss_family: {loss_family!r}")
        elif role.allowed_loss_families and loss_family not in role.allowed_loss_families:
            reasons.append(
                f"loss_family {loss_family!r} violates role constraint "
                f"{role.allowed_loss_families!r}"
            )
        mechanism_ids = manifest.get("mechanism_ids", [])
        if not isinstance(mechanism_ids, (list, tuple)) or not mechanism_ids:
            reasons.append("manifest mechanism_ids must be a non-empty sequence")
        elif any(
            not isinstance(item, str)
            or not item
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in item)
            for item in mechanism_ids
        ):
            reasons.append("mechanism_ids must contain lowercase snake_case identifiers")
        for field_name in ("hypothesis", "mechanism"):
            value = manifest.get(field_name)
            if not isinstance(value, str) or not value.strip():
                reasons.append(f"manifest {field_name} must be a non-empty string")
        modified_symbols = manifest.get("modified_symbols", [])
        if (
            not isinstance(modified_symbols, (list, tuple))
            or not modified_symbols
            or any(not isinstance(item, str) or not item.strip() for item in modified_symbols)
        ):
            reasons.append("manifest modified_symbols must be a non-empty string sequence")

    components = literal_assignment(candidate_code, "ABLATION_COMPONENTS")
    enabled = (
        {
            name
            for name, value in components.items()
            if isinstance(name, str) and value is True
        }
        if isinstance(components, Mapping)
        else set()
    )
    if not enabled:
        reasons.append("no enabled literal ABLATION_COMPONENTS")
    unguarded = enabled - component_guard_calls(candidate_code)
    if unguarded:
        reasons.append(
            "components not guarded by component_enabled: "
            + ", ".join(sorted(unguarded))
        )

    manifest_components = manifest.get("ablation_components", [])
    if isinstance(manifest_components, Sequence) and not isinstance(
        manifest_components, str
    ):
        undeclared = {str(name) for name in manifest_components} - enabled
        if undeclared:
            reasons.append(
                "manifest ablation components not enabled: "
                + ", ".join(sorted(undeclared))
            )
        omitted = enabled - {str(name) for name in manifest_components}
        if omitted:
            reasons.append(
                "enabled components omitted from manifest: "
                + ", ".join(sorted(omitted))
            )
    else:
        reasons.append("manifest ablation_components must be a sequence")

    dependencies = manifest.get("component_dependencies", {})
    if not isinstance(dependencies, Mapping):
        reasons.append("manifest component_dependencies must be a mapping")
    compatibility = manifest.get("combination_compatibility")
    if len(enabled) > 1 and not compatibility:
        reasons.append(
            "multi-component candidate must explain combination_compatibility"
        )

    evidence = manifest.get("evidence", [])
    supported_components: set[str] = set()
    if not isinstance(evidence, (list, tuple)) or not evidence:
        reasons.append("candidate combination has no verifiable evidence")
    else:
        for index, item in enumerate(evidence, 1):
            if not isinstance(item, Mapping):
                reasons.append(f"evidence[{index}] must be a mapping")
                continue
            source_type = item.get("source_type")
            reference = str(item.get("reference", ""))
            supports = item.get("supports")
            if source_type not in {"literature", "validation", "dependency"}:
                reasons.append(f"evidence[{index}] has unsupported source_type")
            if source_type == "literature" and reference not in CURATED_EVIDENCE:
                reasons.append(f"evidence[{index}] is not in the curated literature catalog")
            if source_type == "validation" and not reference.startswith("validation:"):
                reasons.append(f"evidence[{index}] does not reference stored validation evidence")
            if source_type == "dependency" and not reference.startswith("dependency:"):
                reasons.append(f"evidence[{index}] does not state a direct dependency")
            if not isinstance(supports, (list, tuple)) or not supports:
                reasons.append(f"evidence[{index}] does not name supported components")
            else:
                supported_components.update(map(str, supports))
        unsupported = enabled - supported_components
        if unsupported:
            reasons.append(
                "components without literature, validation, or dependency evidence: "
                + ", ".join(sorted(unsupported))
            )

    if is_config_only_change(base_code, candidate_code):
        reasons.append(
            "candidate changes only CONFIG or metadata; tuning is not a research branch"
        )
    architecture_change = role.group == "architecture_exploration" or (
        role.autonomous
        and (research_family == "architecture" or model_family != parent_family)
    )
    if architecture_change:
        base_model = function_dump(base_code, "create_model")
        candidate_model = function_dump(candidate_code, "create_model")
        if candidate_model is None or candidate_model == base_model:
            reasons.append("architecture candidate does not materially change create_model")

    raw_factors = literal_assignment(candidate_code, "FEATURE_FACTORS")
    factors = (
        tuple(item for item in raw_factors if isinstance(item, Mapping))
        if isinstance(raw_factors, (list, tuple))
        else ()
    )
    factor_selection = literal_assignment(candidate_code, "FACTOR_SELECTION")
    considered_factor_ids: list[str] = []
    selected_factor_ids: list[str] = []
    if not isinstance(factor_selection, Mapping):
        factor_selection = {}
        reasons.append("missing literal FACTOR_SELECTION")
    else:
        missing_selection_fields = {
            "considered_factor_ids",
            "selected_factor_ids",
            "selection_reason",
            "rejected_reasons",
            "created_factor_cards",
        } - set(factor_selection)
        if missing_selection_fields:
            reasons.append(
                "FACTOR_SELECTION missing: "
                + ", ".join(sorted(missing_selection_fields))
            )
        considered = factor_selection.get("considered_factor_ids", [])
        selected = factor_selection.get("selected_factor_ids", [])
        if not isinstance(considered, (list, tuple)) or not considered:
            reasons.append("considered_factor_ids must be a non-empty sequence")
        else:
            considered_factor_ids = [str(item) for item in considered]
        if not isinstance(selected, (list, tuple)):
            reasons.append("selected_factor_ids must be a sequence")
        else:
            selected_factor_ids = [str(item) for item in selected]
        created_cards = factor_selection.get("created_factor_cards", [])
        created_factor_ids: set[str] = set()
        if not isinstance(created_cards, (list, tuple)):
            reasons.append("created_factor_cards must be a sequence")
        elif len(created_cards) > 2:
            reasons.append("created_factor_cards may contain at most two cards")
        else:
            required_card_fields = {
                "factor_id",
                "semantics",
                "helps_when",
                "model_fit",
                "avoid_when",
                "data_cost",
                "leakage_rule",
            }
            for index, card in enumerate(created_cards, 1):
                if not isinstance(card, Mapping):
                    reasons.append(f"created_factor_cards[{index}] must be a mapping")
                    continue
                missing = required_card_fields - set(card)
                if missing:
                    reasons.append(
                        f"created_factor_cards[{index}] missing: "
                        + ", ".join(sorted(missing))
                    )
                factor_id = str(card.get("factor_id", ""))
                if (
                    not factor_id.startswith("custom_")
                    or any(
                        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                        for character in factor_id
                    )
                ):
                    reasons.append(
                        f"created_factor_cards[{index}] factor_id must be custom_ snake_case"
                    )
                elif factor_id in FACTOR_IDS or factor_id in created_factor_ids:
                    reasons.append(
                        f"created_factor_cards[{index}] factor_id is duplicate"
                    )
                else:
                    created_factor_ids.add(factor_id)
        known_factor_ids = FACTOR_IDS | created_factor_ids
        unknown = (set(considered_factor_ids) | set(selected_factor_ids)) - known_factor_ids
        if unknown:
            reasons.append("unknown factor library IDs: " + ", ".join(sorted(unknown)))
        if len(selected_factor_ids) > 2:
            reasons.append("selected_factor_ids may contain at most two factors")
        if not set(selected_factor_ids).issubset(considered_factor_ids):
            reasons.append("selected_factor_ids must be a subset of considered_factor_ids")
        if not str(factor_selection.get("selection_reason", "")).strip():
            reasons.append("FACTOR_SELECTION must explain selection_reason")
        rejected = factor_selection.get("rejected_reasons")
        if not isinstance(rejected, Mapping):
            reasons.append("FACTOR_SELECTION rejected_reasons must be a mapping")
        else:
            missing_rejections = (
                set(considered_factor_ids)
                - set(selected_factor_ids)
                - {str(item) for item in rejected}
            )
            if missing_rejections:
                reasons.append(
                    "considered but unselected factors need rejected_reasons: "
                    + ", ".join(sorted(missing_rejections))
                )

    assigned_factor_role = (
        not role.autonomous and role.category in {"factor_data", "factor_model"}
    )
    needs_factor_code = assigned_factor_role or bool(selected_factor_ids)
    if assigned_factor_role and not selected_factor_ids:
        reasons.append("factor candidate must select at least one library factor")
    if needs_factor_code:
        if not factors:
            reasons.append("factor candidate has no literal FEATURE_FACTORS entries")
        required_factor_fields = {
            "library_id",
            "name",
            "raw_fields",
            "transform",
            "output_fields",
            "state_policy",
        }
        for index, factor in enumerate(factors, 1):
            missing = required_factor_fields - set(factor)
            if missing:
                reasons.append(
                    f"FEATURE_FACTORS[{index}] missing: "
                    + ", ".join(sorted(missing))
                )
        declared_library_ids = {
            str(factor.get("library_id", "")) for factor in factors
        }
        missing_selected = set(selected_factor_ids) - declared_library_ids
        if missing_selected:
            reasons.append(
                "selected factors missing matching FEATURE_FACTORS library_id: "
                + ", ".join(sorted(missing_selected))
            )
        base_builder = function_dump(base_code, "build_features")
        candidate_builder = function_dump(candidate_code, "build_features")
        if candidate_builder is None or candidate_builder == base_builder:
            reasons.append("factor candidate does not materially change build_features")
        builder_fields = _function_string_constants(candidate_code, "build_features")
        declared_outputs = {
            output
            for factor in factors
            for output in factor.get("output_fields", [])
            if isinstance(output, str) and output
        }
        missing_outputs = declared_outputs - builder_fields
        if missing_outputs:
            reasons.append(
                "declared factor outputs not created literally in build_features: "
                + ", ".join(sorted(missing_outputs))
            )
        if not role.autonomous and role.category == "factor_model":
            base_model = function_dump(base_code, "create_model")
            candidate_model = function_dump(candidate_code, "create_model")
            if candidate_model is None or candidate_model == base_model:
                reasons.append(
                    "factor_model candidate does not materially change create_model"
                )

    base_role_name = role.name.split("_alternative_", 1)[0]
    ranking_loss = loss_family in {
        "hybrid_bce_bpr",
        "pairwise_bpr",
        "listwise_softmax",
    }
    auxiliary_loss = loss_family in {"multitask", "censored_watch_time"}
    if base_role_name == "ranking_objective" or (
        role.autonomous
        and (research_family == "ranking_objective" or ranking_loss)
    ):
        reasons.extend(_ranking_objective_reasons(candidate_code))
    if base_role_name == "auxiliary_objective" or (
        role.autonomous
        and (research_family == "auxiliary_objective" or auxiliary_loss)
    ):
        reasons.extend(_auxiliary_objective_reasons(candidate_code))

    return CandidateContractResult(
        valid=not reasons,
        reasons=tuple(reasons),
        manifest=manifest,
        feature_factors=factors,
        factor_selection=factor_selection,
    )


_IMPLEMENTATION_METADATA = {
    "CONFIG",
    "RESEARCH_MANIFEST",
    "FEATURE_FACTORS",
    "FACTOR_SELECTION",
    "ABLATION_COMPONENTS",
}


class _ComponentLabelNormalizer(ast.NodeTransformer):
    """Alpha-normalize guard labels without changing guard topology."""

    def __init__(self) -> None:
        self.labels: dict[str, str] = {}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "component_enabled"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            label = node.args[0].value
            normalized = self.labels.setdefault(label, f"component_{len(self.labels)}")
            node.args[0] = ast.Constant(value=normalized)
        return node


def candidate_implementation_signature(code: str) -> str | None:
    """Hash executable AST while ignoring configuration and declarative labels."""
    tree = _tree(code)
    if tree is None:
        return None
    body = []
    for statement in tree.body:
        assigned: set[str] = set()
        if isinstance(statement, ast.Assign):
            assigned = {
                target.id for target in statement.targets if isinstance(target, ast.Name)
            }
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            assigned = {statement.target.id}
        if assigned & _IMPLEMENTATION_METADATA:
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        body.append(statement)
    tree.body = body
    tree = _ComponentLabelNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    canonical = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_SEMANTIC_STOP_WORDS = {
    "a",
    "an",
    "and",
    "candidate",
    "for",
    "from",
    "improve",
    "improves",
    "mechanism",
    "model",
    "of",
    "or",
    "ranking",
    "the",
    "this",
    "to",
    "validation",
    "with",
}


def _semantic_tokens(values: Iterable[Any]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.findall(r"[a-z0-9]+", str(value).lower()):
            if token not in _SEMANTIC_STOP_WORDS:
                tokens.add(token)
    return tokens


_EXECUTABLE_SURFACES = (
    "build_features",
    "build_research_schema",
    "create_model",
    "forward",
    "predict",
    "run_training",
    "step",
    "train_model",
)


def _semantic_surface_symbols(values: Iterable[Any]) -> set[str]:
    surfaces: set[str] = set()
    for value in values:
        text = str(value).lower()
        matched = {name for name in _EXECUTABLE_SURFACES if name in text}
        surfaces.update(matched or {"helper"})
    return surfaces


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def candidate_semantic_signature(result: CandidateContractResult) -> str:
    """Fingerprint declared science without lineage or surface component labels."""
    payload = {
        "model_family": str(result.manifest.get("model_family", "")),
        "research_family": str(result.manifest.get("research_family", "")),
        "loss_family": str(result.manifest.get("loss_family", "")),
        "mechanism_ids": sorted(map(str, result.manifest.get("mechanism_ids", []))),
        "factor_library_ids": sorted(
            {
                *map(str, result.factor_selection.get("selected_factor_ids", [])),
                *(
                    str(item.get("library_id", ""))
                    for item in result.feature_factors
                    if item.get("library_id")
                ),
            }
        ),
        "modified_symbols": sorted(
            map(str, result.manifest.get("modified_symbols", []))
        ),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def candidate_semantic_similarity(
    left: CandidateContractResult,
    right: CandidateContractResult,
) -> float:
    """Return weighted similarity of two self-declared research mechanisms."""
    left_families = {
        f"model:{left.manifest.get('model_family', '')}",
        f"research:{left.manifest.get('research_family', '')}",
        f"loss:{left.manifest.get('loss_family', '')}",
    }
    right_families = {
        f"model:{right.manifest.get('model_family', '')}",
        f"research:{right.manifest.get('research_family', '')}",
        f"loss:{right.manifest.get('loss_family', '')}",
    }
    left_ids = _semantic_tokens(left.manifest.get("mechanism_ids", []))
    right_ids = _semantic_tokens(right.manifest.get("mechanism_ids", []))
    left_text = _semantic_tokens(
        (
            left.manifest.get("hypothesis", ""),
            left.manifest.get("mechanism", ""),
            left.manifest.get("change_scope", ""),
        )
    )
    right_text = _semantic_tokens(
        (
            right.manifest.get("hypothesis", ""),
            right.manifest.get("mechanism", ""),
            right.manifest.get("change_scope", ""),
        )
    )
    left_factors = _semantic_tokens(
        [*left.factor_selection.get("selected_factor_ids", [])]
        + [
            item.get("library_id", "") or item.get("name", "")
            for item in left.feature_factors
        ]
    )
    right_factors = _semantic_tokens(
        [*right.factor_selection.get("selected_factor_ids", [])]
        + [
            item.get("library_id", "") or item.get("name", "")
            for item in right.feature_factors
        ]
    )
    left_symbols = _semantic_surface_symbols(
        left.manifest.get("modified_symbols", [])
    )
    right_symbols = _semantic_surface_symbols(
        right.manifest.get("modified_symbols", [])
    )
    mechanism_similarity = max(
        _jaccard(left_ids, right_ids),
        _jaccard(left_text, right_text),
    )
    weighted_similarity = min(
        1.0,
        max(
            0.0,
            0.35 * _jaccard(left_families, right_families)
            + 0.40 * mechanism_similarity
            + 0.15 * _jaccard(left_factors, right_factors)
            + 0.10 * _jaccard(left_symbols, right_symbols),
        ),
    )
    same_structural_direction = (
        all(
            left.manifest.get(name)
            and left.manifest.get(name) == right.manifest.get(name)
            for name in ("model_family", "research_family", "loss_family")
        )
        and left_factors == right_factors
        and _jaccard(left_symbols, right_symbols) >= 0.80
    )
    return max(
        weighted_similarity,
        mechanism_similarity,
        1.0 if same_structural_direction else 0.0,
    )


def format_factor_change(result: CandidateContractResult) -> str:
    lines = []
    for factor in result.feature_factors:
        raw_fields = ", ".join(map(str, factor.get("raw_fields", [])))
        output_fields = ", ".join(map(str, factor.get("output_fields", [])))
        lines.append(
            f"{factor.get('name')}: raw ({raw_fields}) -> "
            f"{factor.get('transform')} -> outputs ({output_fields})"
        )
    return "\n".join(lines)


def select_candidate_roles(
    direction_summary: Mapping[str, Mapping[str, float]],
    *,
    round_number: int,
    branch_count: int,
    preferred_role_names: Sequence[str] = (),
    reserved_role_name: str = "",
) -> tuple[CandidateRole, ...]:
    """Allocate fixed tree capacity to evidence-weighted, still-diverse roles."""
    count = max(1, int(branch_count))
    if round_number <= 1 and preferred_role_names:
        roles_by_name = {role.name: role for role in DEFAULT_CANDIDATE_ROLES}
        selected = []
        for name in preferred_role_names:
            role = roles_by_name.get(str(name))
            if role is not None and role not in selected:
                selected.append(role)
            if len(selected) >= count:
                break
        for role in DEFAULT_CANDIDATE_ROLES:
            if len(selected) >= count:
                break
            if role not in selected:
                selected.append(role)
        return tuple(selected)
    if round_number <= 1 or not direction_summary:
        return DEFAULT_CANDIDATE_ROLES[:count]

    by_group: dict[str, list[CandidateRole]] = {}
    for role in DEFAULT_CANDIDATE_ROLES:
        by_group.setdefault(role.group, []).append(role)

    total_trials = 1.0 + sum(
        float(values.get("trials", 0.0)) for values in direction_summary.values()
    )

    def utility(group: str) -> float:
        values = direction_summary.get(group, {})
        trials = float(values.get("trials", 0.0))
        mean_gain = float(values.get("mean_gain", 0.0))
        ablation_gain = float(values.get("ablation_gain", 0.0))
        promotion_rate = float(values.get("promotions", 0.0)) / max(1.0, trials)
        generation_success_rate = float(values.get("generation_success_rate", 0.0))
        generation_attempts = float(values.get("generation_attempts", 0.0))
        exploration = math.sqrt(math.log(total_trials + 1.0) / (trials + 1.0))
        reliability = (
            0.0006 * generation_success_rate
            - 0.0004 * (1.0 - generation_success_rate)
            if generation_attempts >= 2
            else 0.0
        )
        return (
            mean_gain
            + ablation_gain
            + 0.001 * promotion_rate
            + 0.0005 * exploration
            + reliability
        )

    groups = list(by_group)
    ranked = sorted(groups, key=utility, reverse=True)
    least_tried = min(
        groups,
        key=lambda group: float(direction_summary.get(group, {}).get("trials", 0.0)),
    )
    if count < len(groups):
        allocated = ranked[: max(0, count - 1)]
        if least_tried not in allocated:
            allocated.append(least_tried)
        allocated = allocated[:count]
    else:
        # Preserve one branch for every major direction. Spend the remaining
        # capacity on evidence-backed exploitation while retaining exploration.
        allocated = list(groups)
        exploitation_order = [ranked[0], ranked[0], *ranked[1:]]
        while len(allocated) < count:
            allocated.append(
                exploitation_order[(len(allocated) - len(groups)) % len(exploitation_order)]
            )

    selected: list[CandidateRole] = []
    group_uses: dict[str, int] = {}
    for group in allocated:
        use = group_uses.get(group, 0)
        group_uses[group] = use + 1
        bases = by_group[group]
        base = bases[use % len(bases)]
        alternative = use // len(bases)
        if alternative == 0:
            selected.append(base)
        else:
            selected.append(
                CandidateRole(
                    name=f"{base.name}_alternative_{alternative + 1}",
                    group=base.group,
                    category=base.category,
                    objective=(
                        base.objective
                        + " Use a materially different mechanism from every sibling "
                        "candidate in this group."
                    ),
                    required_evidence=base.required_evidence,
                )
            )
    selected = selected[:count]
    reserved = next(
        (role for role in DEFAULT_CANDIDATE_ROLES if role.name == reserved_role_name),
        None,
    )
    if reserved is not None and reserved not in selected:
        if selected:
            selected[-1] = reserved
        else:
            selected.append(reserved)
    return tuple(selected)
