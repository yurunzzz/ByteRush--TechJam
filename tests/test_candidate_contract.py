import unittest

from ai_scientist.treesearch.candidate_contract import (
    DEFAULT_CANDIDATE_ROLES,
    candidate_semantic_signature,
    config_assignment,
    format_factor_change,
    restore_dynamic_config_fields,
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
