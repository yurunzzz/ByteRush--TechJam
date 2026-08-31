import unittest

from ai_scientist.treesearch.candidate_contract import (
    AUTONOMOUS_STAGE3_ROLE,
    DEFAULT_CANDIDATE_ROLES,
    bootstrap_candidate_roles,
    build_assignment_marker,
    candidate_implementation_signature,
    candidate_semantic_similarity,
    candidate_semantic_signature,
    config_assignment,
    format_factor_change,
    extract_assignment_contract,
    normalize_candidate_metadata,
    restore_dynamic_config_fields,
    rewrite_for_smoke_test,
    role_from_assignment,
    select_candidate_roles,
    validate_candidate_contract,
    validate_tuning_contract,
)
from ai_scientist.treesearch.factor_library import factor_library_prompt


BASE = """
CONFIG = {'learning_rate': 0.001, 'epochs': 6}
ABLATION_COMPONENTS = {}
def component_enabled(name):
    return False
def build_features(splits, feature_state=None):
    return splits, feature_state
def create_model(feature_dimension, config=None):
    return ('fm', feature_dimension)
"""


def history_candidate(role_name="causal_history_interest", component="history_factor"):
    return f"""
CONFIG = {{'learning_rate': 0.001, 'epochs': 6}}
RESEARCH_MANIFEST = {{
    'candidate_id': 'history_din',
    'role': {role_name!r},
    'group': 'history_interest',
    'category': 'factor_model',
    'model_family': 'fm',
    'research_family': 'history_interest',
    'loss_family': 'pointwise_bce',
    'parent_node_id': '',
    'parent_model_family': 'fm',
    'input_schema_version': 2,
    'hypothesis': 'past author affinity improves user-level ranking',
    'mechanism': 'causal profile consumed by target attention',
    'mechanism_ids': ['causal_history_profile', 'din_target_attention'],
    'modified_symbols': ['build_features', 'create_model'],
    'expected_metric': ['GAUC', 'nDCG@5'],
    'tunable_parameters': ['history_window', 'attention_dim'],
    'ablation_components': [{component!r}, 'din_attention'],
    'combination_compatibility': 'attention consumes the causal history factor',
    'change_scope': 'candidate only',
    'component_dependencies': {{'din_attention': [{component!r}]}},
    'evidence': [
        {{'source_type': 'literature', 'reference': 'din_2018', 'supports': ['din_attention']}},
        {{'source_type': 'dependency', 'reference': 'dependency:din_needs_history', 'supports': [{component!r}]}},
    ],
}}
FACTOR_SELECTION = {{
    'considered_factor_ids': ['causal_recent_history', 'user_author_affinity'],
    'selected_factor_ids': ['causal_recent_history'],
    'selection_reason': 'causal recent history directly supports target attention',
    'rejected_reasons': {{'user_author_affinity': 'narrower than the chosen history profile'}},
    'created_factor_cards': [],
}}
FEATURE_FACTORS = [{{
    'library_id': 'causal_recent_history',
    'name': 'author_history_match',
    'raw_fields': ['user_id', 'author_id', 'time_ms'],
    'transform': 'past-only recent author affinity',
    'output_fields': ['history_author_match'],
    'state_policy': 'train_only_frozen',
}}]
ABLATION_COMPONENTS = {{{component!r}: True, 'din_attention': True}}
def component_enabled(name):
    return ABLATION_COMPONENTS[name]
def build_features(splits, feature_state=None):
    if component_enabled({component!r}):
        output_field = 'history_author_match'
    return splits, feature_state
def create_model(feature_dimension, config=None):
    if component_enabled('din_attention'):
        return ('din', feature_dimension)
    return ('fm', feature_dimension)
    """


def autonomous_history_candidate(component="history_factor"):
    return (
        history_candidate(
            role_name=AUTONOMOUS_STAGE3_ROLE.name,
            component=component,
        )
        .replace("'group': 'history_interest'", "'group': 'autonomous_research'")
        .replace("'category': 'factor_model'", "'category': 'open_choice'")
    )

def objective_candidate(role, step_body="", extra_code=""):
    component = (
        "same_user_ranking_loss"
        if role.name == "ranking_objective"
        else "real_auxiliary_target"
    )
    reference = (
        "bpr_pairwise_implicit_2009"
        if role.name == "ranking_objective"
        else "esmm_2018"
    )
    return f"""
CONFIG = {{'learning_rate': 0.001, 'epochs': 6}}
RESEARCH_MANIFEST = {{
    'candidate_id': 'objective_candidate',
    'role': {role.name!r},
    'group': {role.group!r},
    'category': {role.category!r},
    'model_family': 'fm',
    'research_family': {('ranking_objective' if role.name == 'ranking_objective' else 'auxiliary_objective')!r},
    'loss_family': {('hybrid_bce_bpr' if role.name == 'ranking_objective' else 'multitask')!r},
    'parent_node_id': '',
    'parent_model_family': 'fm',
    'input_schema_version': 2,
    'hypothesis': 'align training with the ranking task',
    'mechanism': 'a guarded objective with valid training inputs',
    'mechanism_ids': [{component!r}],
    'modified_symbols': ['CandidateModel.step', 'create_model'],
    'expected_metric': ['GAUC', 'nDCG@5'],
    'tunable_parameters': ['objective_weight'],
    'ablation_components': [{component!r}],
    'combination_compatibility': 'can support the unchanged FM scorer',
    'change_scope': 'candidate training only',
    'component_dependencies': {{}},
    'evidence': [
        {{'source_type': 'literature', 'reference': {reference!r}, 'supports': [{component!r}]}},
    ],
}}
FACTOR_SELECTION = {{
    'considered_factor_ids': ['auxiliary_behavior_signal', 'temporal_recency_context'],
    'selected_factor_ids': [],
    'selection_reason': 'the controlled objective should isolate loss behavior without new inputs',
    'rejected_reasons': {{
        'auxiliary_behavior_signal': 'would confound the objective comparison',
        'temporal_recency_context': 'not required by this loss hypothesis',
    }},
    'created_factor_cards': [],
}}
ABLATION_COMPONENTS = {{{component!r}: True}}
def component_enabled(name):
    return ABLATION_COMPONENTS[name]
def build_features(splits, feature_state=None):
    return splits, feature_state
class CandidateModel:
    def step(self, x, y):
        if component_enabled({component!r}):
            enabled = True
{step_body}
        return None
def create_model(feature_dimension, config=None):
    return ('objective', feature_dimension)
{extra_code}
"""


class CandidateContractTests(unittest.TestCase):
    def test_assignment_marker_round_trip_preserves_parallel_identity(self):
        role = DEFAULT_CANDIDATE_ROLES[2]
        marker = build_assignment_marker(
            role,
            assignment_id="round2:transfer:1",
            assignment_kind="transfer",
            parent_node_id="incumbent-123",
            parent_model_family="wide_deep",
        )
        payload = extract_assignment_contract("prefix\n" + marker + "suffix")

        self.assertIsNotNone(payload)
        self.assertEqual(payload["assignment_id"], "round2:transfer:1")
        self.assertEqual(payload["assignment_kind"], "transfer")
        self.assertEqual(payload["parent_node_id"], "incumbent-123")
        restored = role_from_assignment(payload)
        self.assertEqual(restored.name, role.name)
        self.assertFalse(restored.autonomous)

    def test_autonomous_assignment_and_prompt_do_not_prescribe_direction(self):
        prompt = AUTONOMOUS_STAGE3_ROLE.prompt(
            2,
            5,
            retry_feedback=["too similar to accepted slot round1:autonomous:1"],
            evidence_memory="validation observations",
            parent_node_id="incumbent-1",
            parent_model_family="fm",
            assignment_id="round1:autonomous:2",
            assignment_kind="autonomous",
        )
        payload = extract_assignment_contract(prompt)
        restored = role_from_assignment(payload)

        self.assertTrue(restored.autonomous)
        self.assertEqual(payload["assignment_id"], "round1:autonomous:2")
        self.assertNotIn("exploit", prompt.lower())
        self.assertNotIn("donor", prompt.lower())
        self.assertNotIn("required mechanism", prompt.lower())
        self.assertIn("No scientific choice has been made", prompt)
        self.assertIn("source_type must be exactly one of", prompt)
        self.assertIn("'literature', 'validation', or 'dependency'", prompt)
        self.assertIn("'supports': ['your_exact_component_key']", prompt)
        self.assertIn("contract, smoke, or execution failures", prompt)
        self.assertIn("duplicate or research-family", prompt)
        self.assertIn("user_ids = x_tensor[:, 0]", prompt)
        self.assertIn("exact string literal inside build_features", prompt)
        self.assertTrue(prompt.rstrip().endswith(build_assignment_marker(
            AUTONOMOUS_STAGE3_ROLE,
            assignment_id="round1:autonomous:2",
            assignment_kind="autonomous",
            parent_node_id="incumbent-1",
            parent_model_family="fm",
        ).rstrip()))

    def test_autonomous_normalizer_preserves_agent_scientific_declarations(self):
        candidate = autonomous_history_candidate()
        normalized = normalize_candidate_metadata(
            candidate,
            AUTONOMOUS_STAGE3_ROLE,
            expected_parent_id="ignored",
            expected_parent_model_family="fm",
        )
        self.assertEqual(normalized, candidate)

    def test_autonomous_factor_candidate_passes_without_assigned_direction(self):
        result = validate_candidate_contract(
            BASE,
            autonomous_history_candidate(),
            AUTONOMOUS_STAGE3_ROLE,
        )
        self.assertTrue(result.valid, result.reasons)

        over_budget = validate_candidate_contract(
            BASE,
            autonomous_history_candidate().replace("'epochs': 6", "'epochs': 13"),
            AUTONOMOUS_STAGE3_ROLE,
        )
        self.assertFalse(over_budget.valid)
        self.assertTrue(any("between 1 and 12" in reason for reason in over_budget.reasons))

    def test_autonomous_family_change_requires_real_model_change(self):
        candidate = autonomous_history_candidate().replace(
            "'model_family': 'fm'", "'model_family': 'dcn'"
        ).replace(
            "    if component_enabled('din_attention'):\n"
            "        return ('din', feature_dimension)\n"
            "    return ('fm', feature_dimension)",
            "    return ('fm', feature_dimension)",
        )
        result = validate_candidate_contract(
            BASE, candidate, AUTONOMOUS_STAGE3_ROLE
        )
        self.assertFalse(result.valid)
        self.assertIn(
            "architecture candidate does not materially change create_model",
            result.reasons,
        )

    def test_autonomous_ranking_loss_cannot_bypass_same_user_guard(self):
        ranking_role = DEFAULT_CANDIDATE_ROLES[2]
        candidate = objective_candidate(
            ranking_role,
            step_body="""
        x_tensor = x
        positive = y > 0
        negative = y <= 0
""",
        )
        candidate = (
            candidate
            .replace(f"'role': {ranking_role.name!r}", f"'role': {AUTONOMOUS_STAGE3_ROLE.name!r}")
            .replace(f"'group': {ranking_role.group!r}", f"'group': {AUTONOMOUS_STAGE3_ROLE.group!r}")
            .replace(f"'category': {ranking_role.category!r}", f"'category': {AUTONOMOUS_STAGE3_ROLE.category!r}")
            .replace("'research_family': 'ranking_objective'", "'research_family': 'training_strategy'")
        )
        result = validate_candidate_contract(
            BASE, candidate, AUTONOMOUS_STAGE3_ROLE
        )
        self.assertFalse(result.valid)
        self.assertTrue(
            any("x_tensor[:, 0]" in reason for reason in result.reasons),
            result.reasons,
        )

    def test_implementation_signature_ignores_metadata_and_component_labels(self):
        first = autonomous_history_candidate(component="history_factor")
        second = (
            first
            .replace("'learning_rate': 0.001", "'learning_rate': 0.002")
            .replace("history_factor", "renamed_factor")
            .replace("'candidate_id': 'history_din'", "'candidate_id': 'renamed'")
        )
        changed = first.replace("return ('din', feature_dimension)", "return ('dcn', feature_dimension)")

        self.assertEqual(
            candidate_implementation_signature(first),
            candidate_implementation_signature(second),
        )
        self.assertNotEqual(
            candidate_implementation_signature(first),
            candidate_implementation_signature(changed),
        )
        self.assertIsNone(candidate_implementation_signature("def broken("))

    def test_semantic_similarity_catches_renamed_same_idea(self):
        first = validate_candidate_contract(
            BASE, autonomous_history_candidate(), AUTONOMOUS_STAGE3_ROLE
        )
        renamed = autonomous_history_candidate().replace(
            "'causal_history_profile', 'din_target_attention'",
            "'recent_profile_v2', 'target_gate_v2'",
        )
        second = validate_candidate_contract(
            BASE, renamed, AUTONOMOUS_STAGE3_ROLE
        )
        relabeled = validate_candidate_contract(
            BASE,
            renamed.replace(
                "'research_family': 'history_interest'",
                "'research_family': 'feature_engineering'",
            ),
            AUTONOMOUS_STAGE3_ROLE,
        )
        synonymous = validate_candidate_contract(
            BASE,
            renamed
            .replace(
                "'past author affinity improves user-level ranking'",
                "'viewer preference over prior creators improves ordering'",
            )
            .replace(
                "'causal profile consumed by target attention'",
                "'target-conditioned pooling reads earlier creator interactions'",
            ),
            AUTONOMOUS_STAGE3_ROLE,
        )
        different = validate_candidate_contract(
            BASE,
            renamed
            .replace("'research_family': 'history_interest'", "'feature_engineering'")
            .replace("'past author affinity improves user-level ranking'", "'static metadata improves unseen item representation'")
            .replace("'causal profile consumed by target attention'", "'regularized static metadata residual'"),
            AUTONOMOUS_STAGE3_ROLE,
        )

        self.assertGreaterEqual(candidate_semantic_similarity(first, second), 0.80)
        self.assertGreaterEqual(candidate_semantic_similarity(first, relabeled), 0.80)
        self.assertGreaterEqual(candidate_semantic_similarity(first, synonymous), 0.80)
        self.assertLess(candidate_semantic_similarity(first, different), 0.80)

    def test_smoke_rewrite_limits_training_without_changing_mechanism(self):
        code = BASE.replace("'epochs': 6", "'epochs': 12, 'patience': 4")
        smoke = rewrite_for_smoke_test(code)

        config = config_assignment(smoke)
        self.assertEqual(config["epochs"], 1)
        self.assertEqual(config["patience"], 1)
        self.assertIn("return ('fm', feature_dimension)", smoke)

    def test_metadata_normalizer_repairs_shapes_without_inventing_model_code(self):
        role = bootstrap_candidate_roles(["wide_deep"])[0]
        candidate = """
CONFIG = {'learning_rate': 0.001, 'epochs': 5}
RESEARCH_MANIFEST = {'model_family': 'wide_deep', 'ablation_components': {'wide_deep_block': True}}
FACTOR_SELECTION = {'considered_factor_ids': ['legacy_fm_fields'], 'selected_factor_ids': [], 'rejected_reasons': []}
ABLATION_COMPONENTS = {'wide_deep_block': True}
def component_enabled(name):
    return ABLATION_COMPONENTS.get(name, False)
def build_features(splits, feature_state=None):
    return splits, feature_state
def create_model(feature_dimension, config=None):
    if component_enabled('wide_deep_block'):
        return ('wide_deep', feature_dimension)
    return ('fm', feature_dimension)
"""

        normalized = normalize_candidate_metadata(
            candidate,
            role,
            expected_parent_id="fm-root",
            expected_parent_model_family="fm",
        )
        result = validate_candidate_contract(
            BASE,
            normalized,
            role,
            expected_parent_id="fm-root",
            expected_parent_model_family="fm",
        )

        self.assertTrue(result.valid, result.reasons)
        self.assertEqual(result.manifest["ablation_components"], ["wide_deep_block"])
        self.assertEqual(
            result.factor_selection["considered_factor_ids"], ["static_user_profile"]
        )
        self.assertEqual(result.factor_selection["selected_factor_ids"], [])

    def test_stage1b_model_family_and_parent_are_hard_constraints(self):
        role = bootstrap_candidate_roles(["wide_deep"])[0]
        candidate = """
CONFIG = {'learning_rate': 0.001, 'epochs': 5}
RESEARCH_MANIFEST = {
    'candidate_id': 'wide_deep_root',
    'role': 'architecture_wide_deep',
    'group': 'architecture_exploration',
    'category': 'model_architecture',
    'model_family': 'wide_deep',
    'research_family': 'architecture',
    'loss_family': 'pointwise_bce',
    'parent_node_id': 'fm-root',
    'parent_model_family': 'fm',
    'input_schema_version': 2,
    'hypothesis': 'wide and deep paths improve ranking',
    'mechanism': 'joint memorization and generalization',
    'mechanism_ids': ['wide_deep_parallel_paths'],
    'modified_symbols': ['CandidateModel', 'create_model'],
    'expected_metric': ['GAUC', 'nDCG@5'],
    'tunable_parameters': ['hidden_dim'],
    'ablation_components': ['wide_deep_block'],
    'combination_compatibility': 'single architecture block',
    'change_scope': 'model only',
    'component_dependencies': {},
    'evidence': [{'source_type': 'literature', 'reference': 'wide_deep_2016', 'supports': ['wide_deep_block']}],
}
FACTOR_SELECTION = {
    'considered_factor_ids': ['user_item_context_cross'],
    'selected_factor_ids': [],
    'selection_reason': 'isolate architecture in the bootstrap',
    'rejected_reasons': {'user_item_context_cross': 'defer factors to Stage 3'},
    'created_factor_cards': [],
}
ABLATION_COMPONENTS = {'wide_deep_block': True}
def component_enabled(name):
    return ABLATION_COMPONENTS[name]
def build_features(splits, feature_state=None):
    return splits, feature_state
def create_model(feature_dimension, config=None):
    if component_enabled('wide_deep_block'):
        return ('wide_deep', feature_dimension)
    return ('fm', feature_dimension)
"""
        valid = validate_candidate_contract(
            BASE,
            candidate,
            role,
            expected_parent_id="fm-root",
            expected_parent_model_family="fm",
        )
        wrong_family = validate_candidate_contract(
            BASE,
            candidate.replace("'model_family': 'wide_deep'", "'model_family': 'dcn'"),
            role,
            expected_parent_id="fm-root",
            expected_parent_model_family="fm",
        )
        wrong_parent = validate_candidate_contract(
            BASE,
            candidate,
            role,
            expected_parent_id="another-root",
            expected_parent_model_family="fm",
        )
        self.assertTrue(valid.valid, valid.reasons)
        self.assertFalse(wrong_family.valid)
        self.assertFalse(wrong_parent.valid)

    def test_factor_and_model_bundle_passes_contract(self):
        role = DEFAULT_CANDIDATE_ROLES[0]
        result = validate_candidate_contract(BASE, history_candidate(), role)
        self.assertTrue(result.valid, result.reasons)

    def test_factor_summary_does_not_use_rich_markup_brackets(self):
        role = DEFAULT_CANDIDATE_ROLES[0]
        result = validate_candidate_contract(BASE, history_candidate(), role)

        summary = format_factor_change(result)

        self.assertIn("raw (user_id, author_id, time_ms)", summary)
        self.assertIn("outputs (history_author_match)", summary)
        self.assertNotIn("[user_id", summary)

    def test_role_prompt_spells_out_machine_checkable_literals(self):
        role = DEFAULT_CANDIDATE_ROLES[0]
        prompt = role.prompt(1, len(DEFAULT_CANDIDATE_ROLES))

        self.assertIn(f"'role': {role.name!r}", prompt)
        self.assertIn(f"'group': {role.group!r}", prompt)
        self.assertIn(f"'category': {role.category!r}", prompt)
        self.assertIn("supports must be a literal non-empty list", prompt)
        self.assertIn("inside the build_features function body", prompt)
        self.assertIn("features['history_author_ids']", prompt)
        self.assertIn("FACTOR_SELECTION", prompt)

    def test_factor_library_is_considered_without_forcing_selection(self):
        role = DEFAULT_CANDIDATE_ROLES[2]
        prompt = role.prompt(
            1,
            3,
            factor_library_context=factor_library_prompt(
                role_group=role.group,
                role_category=role.category,
            ),
        )
        code = objective_candidate(
            role,
            step_body="""
        x_tensor = x
        user_ids = x_tensor[:, 0]
        for user_id in torch.unique(user_ids):
            same_user = user_ids == user_id
""",
        )

        result = validate_candidate_contract(BASE, code, role)

        self.assertIn("auxiliary_behavior_signal", prompt)
        self.assertTrue(result.valid, result.reasons)
        self.assertEqual(result.factor_selection["selected_factor_ids"], [])

    def test_agent_can_create_and_use_a_missing_factor_card(self):
        role = DEFAULT_CANDIDATE_ROLES[0]
        code = history_candidate().replace(
            "['causal_recent_history', 'user_author_affinity']",
            "['custom_history_diversity']",
        ).replace(
            "['causal_recent_history']",
            "['custom_history_diversity']",
        ).replace(
            "'created_factor_cards': []",
            "'created_factor_cards': [{"
            "'factor_id': 'custom_history_diversity', "
            "'semantics': 'past-only diversity of recent interests', "
            "'helps_when': ['recent history is repetitive'], "
            "'model_fit': ['gated interest encoder'], "
            "'avoid_when': ['history is empty'], "
            "'data_cost': 'low', "
            "'leakage_rule': 'use only rows before the target'"
            "}]",
        ).replace(
            "'library_id': 'causal_recent_history'",
            "'library_id': 'custom_history_diversity'",
        )

        result = validate_candidate_contract(BASE, code, role)

        self.assertTrue(result.valid, result.reasons)
        self.assertEqual(
            result.factor_selection["selected_factor_ids"],
            ["custom_history_diversity"],
        )
    def test_objective_prompts_state_data_semantics(self):
        ranking_prompt = DEFAULT_CANDIDATE_ROLES[2].prompt(
            3, len(DEFAULT_CANDIDATE_ROLES)
        )
        auxiliary_prompt = DEFAULT_CANDIDATE_ROLES[3].prompt(
            4, len(DEFAULT_CANDIDATE_ROLES)
        )

        self.assertIn("user_ids = x_tensor[:, 0]", ranking_prompt)
        self.assertIn("different users is invalid", ranking_prompt)
        self.assertIn("log_standard_4_08_to_4_21_pure.csv", auxiliary_prompt)
        self.assertIn("no long_view fallback", auxiliary_prompt)

    def test_cross_user_pairwise_candidate_is_rejected(self):
        role = DEFAULT_CANDIDATE_ROLES[2]
        code = objective_candidate(
            role,
            step_body="""
        x_tensor = x
        positive = y > 0
        negative = y <= 0
        pair_loss = positive.sum() + negative.sum()
""",
        )

        result = validate_candidate_contract(BASE, code, role)

        self.assertFalse(result.valid)
        self.assertTrue(
            any("x_tensor[:, 0]" in reason for reason in result.reasons),
            result.reasons,
        )

    def test_same_user_pairwise_candidate_passes(self):
        role = DEFAULT_CANDIDATE_ROLES[2]
        code = objective_candidate(
            role,
            step_body="""
        x_tensor = x
        user_ids = x_tensor[:, 0]
        for user_id in torch.unique(user_ids):
            same_user = user_ids == user_id
            pair_count = same_user.sum()
""",
        )

        result = validate_candidate_contract(BASE, code, role)

        self.assertTrue(result.valid, result.reasons)

    def test_primary_label_auxiliary_copy_is_rejected(self):
        role = DEFAULT_CANDIDATE_ROLES[3]
        code = objective_candidate(
            role,
            extra_code="""
def run_training():
    train_y = labels
    aux_y = train_y.copy()
    return aux_y
""",
        )

        result = validate_candidate_contract(BASE, code, role)

        self.assertFalse(result.valid)
        self.assertTrue(
            any("primary long_view label" in reason for reason in result.reasons),
            result.reasons,
        )

    def test_train_only_real_auxiliary_field_passes(self):
        role = DEFAULT_CANDIDATE_ROLES[3]
        code = objective_candidate(
            role,
            extra_code="""
def build_auxiliary_targets(input_dir):
    path = (
        input_dir
        / 'KuaiRand-Pure'
        / 'data'
        / 'log_standard_4_08_to_4_21_pure.csv'
    )
    targets = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            targets.append(float(row['is_click']))
    return targets

def run_training():
    aux_y = build_auxiliary_targets(input_dir)
    return aux_y
""",
        )

        result = validate_candidate_contract(BASE, code, role)

        self.assertTrue(result.valid, result.reasons)

    def test_config_only_candidate_is_rejected(self):
        role = DEFAULT_CANDIDATE_ROLES[0]
        code = history_candidate().replace("return ('din', feature_dimension)", "return ('fm', feature_dimension)")
        result = validate_candidate_contract(code, code.replace("0.001", "0.002"), role)
        self.assertFalse(result.valid)
        self.assertTrue(any("CONFIG" in reason for reason in result.reasons))

    def test_semantic_duplicate_ignores_component_wording(self):
        role = DEFAULT_CANDIDATE_ROLES[0]
        first = validate_candidate_contract(BASE, history_candidate(), role)
        second = validate_candidate_contract(
            BASE,
            history_candidate(component="history_profile"),
            role,
        )
        self.assertEqual(
            candidate_semantic_signature(first),
            candidate_semantic_signature(second),
        )

    def test_tuning_changes_only_literal_config(self):
        tuned = BASE.replace("0.001", "0.0005")
        self.assertTrue(validate_tuning_contract(BASE, tuned).valid)
        invalid = tuned.replace("return ('fm', feature_dimension)", "return ('deepfm', feature_dimension)")
        self.assertFalse(validate_tuning_contract(BASE, invalid).valid)

    def test_tuning_preserves_dynamic_seed_expression(self):
        base = BASE.replace(
            "CONFIG = {'learning_rate': 0.001, 'epochs': 6}",
            "CONFIG = {'seed': int(os.environ.get('AI_SCIENTIST_SEED', '0')), "
            "'learning_rate': 0.001, 'epochs': 6}",
        )
        tuned = base.replace("'learning_rate': 0.001", "'learning_rate': 0.0005")

        result = validate_tuning_contract(base, tuned)

        self.assertTrue(result.valid, result.reasons)
        self.assertIn("__ast_expression__", result.config["seed"])
        self.assertEqual(config_assignment(tuned)["learning_rate"], 0.0005)

    def test_tuning_cannot_replace_dynamic_seed_or_config_keys(self):
        base = BASE.replace(
            "CONFIG = {'learning_rate': 0.001, 'epochs': 6}",
            "CONFIG = {'seed': int(os.environ.get('AI_SCIENTIST_SEED', '0')), "
            "'learning_rate': 0.001, 'epochs': 6}",
        )
        changed_seed = base.replace(
            "int(os.environ.get('AI_SCIENTIST_SEED', '0'))",
            "123",
        )
        missing_key = base.replace(", 'epochs': 6", "")

        seed_result = validate_tuning_contract(base, changed_seed)
        key_result = validate_tuning_contract(base, missing_key)

        self.assertFalse(seed_result.valid)
        self.assertTrue(any("dynamic CONFIG" in reason for reason in seed_result.reasons))
        self.assertFalse(key_result.valid)
        self.assertTrue(any("CONFIG keys" in reason for reason in key_result.reasons))

    def test_runtime_seed_is_restored_before_tuning_execution(self):
        base = BASE.replace(
            "CONFIG = {'learning_rate': 0.001, 'epochs': 6}",
            "CONFIG = {'seed': int(os.environ.get('AI_SCIENTIST_SEED', '0')), "
            "'learning_rate': 0.001, 'epochs': 6}",
        )
        proposed = base.replace(
            "int(os.environ.get('AI_SCIENTIST_SEED', '0'))",
            "0",
        ).replace("'epochs': 6", "'epochs': 12")

        restored = restore_dynamic_config_fields(base, proposed)
        result = validate_tuning_contract(base, restored)

        self.assertTrue(result.valid, result.reasons)
        self.assertIn("__ast_expression__", result.config["seed"])
        self.assertEqual(result.config["epochs"], 12)

    def test_dynamic_portfolio_keeps_all_groups_and_eight_slots(self):
        summary = {
            'history_interest': {'trials': 2, 'promotions': 2, 'mean_gain': 0.004, 'ablation_gain': 0.002},
            'objective_and_training': {'trials': 2, 'promotions': 1, 'mean_gain': 0.002, 'ablation_gain': 0.001},
            'context_interaction': {'trials': 2, 'promotions': 0, 'mean_gain': 0.0, 'ablation_gain': 0.0},
            'evidence_combination': {'trials': 0, 'promotions': 0, 'mean_gain': 0.0, 'ablation_gain': 0.0},
        }
        roles = select_candidate_roles(summary, round_number=2, branch_count=8)
        self.assertEqual(len(roles), 8)
        self.assertEqual({role.group for role in roles}, set(summary))
        self.assertEqual(len({role.name for role in roles}), 8)

    def test_dynamic_portfolio_reserves_adaptive_slot(self):
        summary = {
            "history_interest": {"trials": 1, "mean_gain": 0.004},
            "objective_and_training": {"trials": 1, "mean_gain": 0.003},
            "context_interaction": {"trials": 1, "mean_gain": 0.002},
            "evidence_combination": {"trials": 1, "mean_gain": -0.001},
        }

        roles = select_candidate_roles(
            summary,
            round_number=2,
            branch_count=3,
            reserved_role_name="incumbent_extension",
        )

        self.assertIn("incumbent_extension", [role.name for role in roles])


if __name__ == "__main__":
    unittest.main()
