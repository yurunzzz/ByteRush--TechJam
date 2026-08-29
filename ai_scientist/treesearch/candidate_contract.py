"""Static contracts for diverse, executable KuaiRand research candidates."""
from __future__ import annotations

import ast
import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CandidateRole:
    name: str
    group: str
    category: str
    objective: str
    required_evidence: str

    def prompt(
        self,
        index: int,
        total: int,
        *,
        retry_feedback: Sequence[str] = (),
        evidence_memory: str = "",
    ) -> str:
        evidence_ids = ", ".join(CURATED_EVIDENCE)
        base_name = self.name.split("_alternative_", 1)[0]
        techniques = TECHNIQUE_CATALOG.get(base_name, ())
        technique_text = "\n".join(f"  - {item}" for item in techniques)
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
        return (
            f"Candidate role {index}/{total}: {self.name}\n"
            f"Major research group: {self.group}\n"
            f"Research category: {self.category}\n"
            f"Required mechanism: {self.objective}\n"
            f"Required evidence in code: {self.required_evidence}\n"
            f"Role-specific technique menu (prior, not a mandate):\n{technique_text}\n"
            "Choose the smallest coherent mechanism that tests the hypothesis. "
            "Do not copy the whole menu into one candidate.\n"
            "This role is exclusive. Do not replace it with a different model family "
            "and do not submit a CONFIG-only hyperparameter change. The candidate may "
            "be a coherent bundle: one central mechanism plus compatible supporting "
            "factors, objectives, or training changes. Guard every component separately."
            "\nRequired code output contract:\n"
            "- Define a literal RESEARCH_MANIFEST with candidate_id, role, group, "
            "category, hypothesis, mechanism, mechanism_ids, modified_symbols, expected_metric, "
            "tunable_parameters, ablation_components, combination_compatibility, "
            "change_scope, component_dependencies, and evidence.\n"
            "- Evidence is a list of mappings with source_type, reference, and supports. "
            f"Allowed literature references: {evidence_ids}.\n"
            "- Use these exact literal manifest values (do not rename or omit them): "
            f"'role': {self.name!r}, 'group': {self.group!r}, "
            f"'category': {self.category!r}.\n"
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
            "name, raw_fields, transform, output_fields, and state_policy; build_features "
            "must create the fields and a guarded model path must consume them.\n"
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
            + memory_text
            + feedback_text
        )


DEFAULT_CANDIDATE_ROLES: tuple[CandidateRole, ...] = (
    CandidateRole(
        "causal_history_interest",
        "history_interest",
        "factor_model",
        "Create causal history factors and a matching interest encoder that actually consumes their embeddings.",
        "build_features must create real history fields and the guarded model path must consume them.",
    ),
    CandidateRole(
        "affinity_interest",
        "history_interest",
        "factor_model",
        "Create causal user-author/tag/type affinity factors and a matching gated or attention interest model.",
        "Declare raw-to-factor mappings and prove the guarded model path uses every added field.",
    ),
    CandidateRole(
        "ranking_objective",
        "objective_and_training",
        "training_objective",
        "Change optimization toward GAUC/nDCG with a controlled hybrid pointwise and pairwise/listwise loss.",
        "The loss path must be guarded as an ablation component and keep long_view as the primary target.",
    ),
    CandidateRole(
        "auxiliary_objective",
        "objective_and_training",
        "training_objective",
        "Use legal training-window auxiliary behavior or watch-time targets without exposing validation/test outcomes.",
        "The auxiliary head/loss and its weight must be explicit, guarded, and absent from inference inputs.",
    ),
    CandidateRole(
        "context_interaction",
        "context_interaction",
        "factor_model",
        "Create static user/video and crossed context factors with a matching DeepFM/DCN/xDeepFM interaction block.",
        "The feature builder and guarded interaction path must both change and remain independently ablatable.",
    ),
    CandidateRole(
        "temporal_interaction",
        "context_interaction",
        "factor_model",
        "Create causal time/recency factors with a matching interaction or robust temporal model.",
        "State cutoff behavior and prove the vectorized guarded model path consumes the temporal fields.",
    ),
    CandidateRole(
        "incumbent_extension",
        "evidence_combination",
        "evidence_synthesis",
        "Extend the incumbent with a coherent set of mechanisms supported by its validation and ablation evidence.",
        "Name the inherited and new components, justify dependencies, and guard every new component independently.",
    ),
    CandidateRole(
        "cross_direction_synthesis",
        "evidence_combination",
        "evidence_synthesis",
        "Combine mutually supportive mechanisms backed by prior validation, ablation memory, or established recommendation research.",
        "Name the evidence for the combination, declare dependencies, and guard every component independently.",
    ),
)


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
    ),
    "cross_direction_synthesis": (
        "Combine components only when curated literature, stored validation evidence, or a direct dependency supports the pairing.",
        "Keep every component independently switchable so Stage 4 can estimate conditional contribution and key pair synergy.",
    ),
}


REQUIRED_MANIFEST_FIELDS = (
    "candidate_id",
    "role",
    "group",
    "category",
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


@dataclass(frozen=True)
class CandidateContractResult:
    valid: bool
    reasons: tuple[str, ...]
    manifest: Mapping[str, Any]
    feature_factors: tuple[Mapping[str, Any], ...]


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
) -> CandidateContractResult:
    reasons: list[str] = []
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

    raw_factors = literal_assignment(candidate_code, "FEATURE_FACTORS")
    factors = (
        tuple(item for item in raw_factors if isinstance(item, Mapping))
        if isinstance(raw_factors, (list, tuple))
        else ()
    )
    if role.category in {"factor_data", "factor_model"}:
        if not factors:
            reasons.append("factor candidate has no literal FEATURE_FACTORS entries")
        required_factor_fields = {
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
        base_builder = function_dump(base_code, "build_features")
        candidate_builder = function_dump(candidate_code, "build_features")
        if candidate_builder is None or candidate_builder == base_builder:
            reasons.append("factor candidate does not materially change build_features")
        builder_fields = _function_string_constants(candidate_code, "build_features")
        declared_outputs = {
            str(output)
            for factor in factors
            for output in factor.get("output_fields", [])
        }
        missing_outputs = declared_outputs - builder_fields
        if missing_outputs:
            reasons.append(
                "declared factor outputs not created literally in build_features: "
                + ", ".join(sorted(missing_outputs))
            )
        if role.category == "factor_model":
            base_model = function_dump(base_code, "create_model")
            candidate_model = function_dump(candidate_code, "create_model")
            if candidate_model is None or candidate_model == base_model:
                reasons.append(
                    "factor_model candidate does not materially change create_model"
                )

    return CandidateContractResult(
        valid=not reasons,
        reasons=tuple(reasons),
        manifest=manifest,
        feature_factors=factors,
    )


def candidate_semantic_signature(result: CandidateContractResult) -> str:
    """Fingerprint mechanisms rather than surface code or prose wording."""
    payload = {
        "mechanism_ids": sorted(map(str, result.manifest.get("mechanism_ids", []))),
        "factors": sorted(str(item.get("name", "")) for item in result.feature_factors),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


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
) -> tuple[CandidateRole, ...]:
    """Allocate fixed tree capacity to evidence-weighted, still-diverse roles."""
    count = max(1, int(branch_count))
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
        exploration = math.sqrt(math.log(total_trials + 1.0) / (trials + 1.0))
        return mean_gain + ablation_gain + 0.001 * promotion_rate + 0.0005 * exploration

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
    return tuple(selected)
