import unittest

from ai_scientist.treesearch.candidate_contract import (
    DEFAULT_CANDIDATE_ROLES,
    candidate_semantic_signature,
    format_factor_change,
    select_candidate_roles,
    validate_candidate_contract,
    validate_tuning_contract,
)


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
FEATURE_FACTORS = [{{
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


if __name__ == "__main__":
    unittest.main()
