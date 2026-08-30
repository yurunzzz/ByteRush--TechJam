import unittest

from omegaconf import OmegaConf

from ai_scientist.treesearch.research_loop import (
    AblationEvidence,
    PromotionPolicy,
    ResearchLoopConfig,
    ResearchLoopState,
    candidate_fingerprint,
    estimate_max_experiment_runs,
    extract_primary_score,
    research_loop_config_from_mapping,
    validation_metrics_only,
)
from ai_scientist.treesearch.utils.config import ResearchLoopSettings


class ResearchLoopMetricTests(unittest.TestCase):
    def test_direct_primary_metric_wins(self):
        result = extract_primary_score(
            {"primary": 0.71, "GAUC": 0.70, "nDCG@5": 0.68}
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "primary")
        self.assertAlmostEqual(result.score, 0.71)

    def test_primary_is_built_from_benchmark_components(self):
        metric = {
            "metric_names": [
                {
                    "metric_name": "GAUC",
                    "lower_is_better": False,
                    "data": [
                        {
                            "dataset_name": "validation",
                            "final_value": 0.70,
                            "best_value": 0.72,
                        }
                    ],
                },
                {
                    "metric_name": "nDCG@5",
                    "lower_is_better": False,
                    "data": [
                        {
                            "dataset_name": "validation",
                            "final_value": 0.66,
                            "best_value": 0.68,
                        }
                    ],
                },
            ]
        }
        result = extract_primary_score(metric)
        self.assertIsNotNone(result)
        self.assertEqual(result.source, "component_mean")
        self.assertAlmostEqual(result.score, 0.70)

    def test_experience_metrics_reject_test_or_unknown_fields(self):
        result = validation_metrics_only(
            {
                "validation GAUC": 0.71,
                "validation nDCG@5": 0.63,
                "validation primary": 0.67,
                "test primary": 0.99,
                "private_leaderboard": 1.0,
            }
        )

        self.assertEqual(
            result,
            {"GAUC": 0.71, "nDCG@5": 0.63, "primary": 0.67},
        )


class PromotionPolicyTests(unittest.TestCase):
    def test_candidate_must_pass_mean_and_seed_gates(self):
        policy = PromotionPolicy(min_improvement=0.002)
        decision = policy.evaluate(
            candidate_score=0.704,
            incumbent_score=0.700,
            candidate_seed_scores=[0.703, 0.704, 0.705],
        )
        self.assertTrue(decision.promoted)
        self.assertEqual(decision.seed_wins, 3)

    def test_one_lucky_seed_does_not_promote(self):
        policy = PromotionPolicy(min_improvement=0.002)
        decision = policy.evaluate(
            candidate_score=0.701,
            incumbent_score=0.700,
            candidate_seed_scores=[0.710, 0.697, 0.696],
        )
        self.assertFalse(decision.promoted)


class ResearchLoopStateTests(unittest.TestCase):
    def test_strict_config_schema_accepts_experience_top_k(self):
        schema = OmegaConf.structured(ResearchLoopSettings)
        merged = OmegaConf.merge(schema, {"experience_top_k": 5})

        self.assertEqual(merged.experience_top_k, 5)

    def test_strict_config_schema_accepts_stage3_elimination_and_topics(self):
        schema = OmegaConf.structured(ResearchLoopSettings)
        merged = OmegaConf.merge(
            schema,
            {
                "candidate_tuning_top_k": 2,
                "initial_candidate_roles": [
                    "causal_history_interest",
                    "ranking_objective",
                    "incumbent_extension",
                ],
                "reserved_candidate_role": "incumbent_extension",
            },
        )

        self.assertEqual(merged.candidate_tuning_top_k, 2)
        self.assertEqual(merged.initial_candidate_roles[-1], "incumbent_extension")

    def test_default_patience_matches_official_convergence_rule(self):
        self.assertEqual(ResearchLoopConfig().patience, 2)

    def test_cross_component_synergy_is_available_to_next_round_prompt(self):
        state = ResearchLoopState()
        state.memory.record_ablation(
            AblationEvidence(
                round_number=1,
                candidate_id="candidate-a",
                component="history_factor+ranking_loss",
                category="evidence_combination",
                full_score=0.710,
                ablated_score=0.704,
                primary_contribution=0.006,
                seed_wins=3,
                verdict="positive_synergy",
                interaction_with=["history_factor", "ranking_loss"],
                synergy=0.002,
            )
        )

        prompt_memory = state.memory.portfolio_lessons()

        self.assertIn("history_factor+ranking_loss", prompt_memory)
        self.assertIn("positive_synergy", prompt_memory)
        self.assertIn('"synergy": 0.002', prompt_memory)

    def test_production_budget_is_calculated_from_all_loop_layers(self):
        budget = estimate_max_experiment_runs(
            ResearchLoopConfig(), stage4_max_components=6
        )

        self.assertEqual(budget["stage2_seed_evaluation"], 9)
        self.assertEqual(budget["per_research_round"], 162)
        self.assertEqual(budget["maximum_search_iterations"], 370)
        self.assertEqual(budget["maximum_total"], 683)

    def test_partial_mapping_keeps_production_defaults(self):
        config = research_loop_config_from_mapping(
            {"candidate_tuning_iterations": 4}
        )

        self.assertEqual(config.candidate_tuning_iterations, 4)
        self.assertEqual(config.baseline_tuning_iterations, 24)
        self.assertEqual(config.finalist_num_seeds, 3)

    def test_duplicate_candidates_are_rejected(self):
        state = ResearchLoopState()
        state.set_incumbent("baseline", 0.70, [0.70, 0.70, 0.70])
        state.start_round()
        fingerprint = candidate_fingerprint("print('candidate')")

        first = state.evaluate_candidate(
            node_id="candidate-a",
            fingerprint=fingerprint,
            score=0.704,
            seed_scores=[0.703, 0.704, 0.705],
        )
        second = state.evaluate_candidate(
            node_id="candidate-b",
            fingerprint=fingerprint,
            score=0.706,
            seed_scores=[0.706, 0.706, 0.706],
        )

        self.assertTrue(first.promoted)
        self.assertFalse(second.promoted)
        self.assertIn("duplicate", second.reason)

    def test_round_keeps_only_the_best_promoted_candidate(self):
        state = ResearchLoopState()
        state.set_incumbent("baseline", 0.70)
        state.start_round()

        state.evaluate_candidate(
            node_id="candidate-a",
            fingerprint=candidate_fingerprint("code-a"),
            score=0.704,
            seed_scores=[0.704, 0.704, 0.704],
        )
        state.evaluate_candidate(
            node_id="candidate-b",
            fingerprint=candidate_fingerprint("code-b"),
            score=0.706,
            seed_scores=[0.706, 0.706, 0.706],
        )

        self.assertEqual(state.pending_candidate_id, "candidate-b")

    def test_stage4_regression_cannot_replace_the_incumbent(self):
        state = ResearchLoopState()
        state.set_incumbent("baseline", 0.700, [0.700, 0.700, 0.700])
        state.start_round()
        initial = state.evaluate_candidate(
            node_id="candidate",
            fingerprint=candidate_fingerprint("candidate-code"),
            score=0.704,
            seed_scores=[0.704, 0.704, 0.704],
        )
        self.assertTrue(initial.promoted)

        confirmation = state.accept_pending_candidate(
            node_id="candidate",
            final_node_id="stage4-candidate",
            score=0.699,
            seed_scores=[0.699, 0.699, 0.699],
        )

        self.assertFalse(confirmation.promoted)
        self.assertEqual(state.incumbent_node_id, "baseline")
        self.assertAlmostEqual(state.incumbent_score, 0.700)
        self.assertFalse(state.round_improved)
        self.assertIn("Stage 4 confirmation rejected", state.memory.failed_hypotheses[-1])

    def test_any_promoted_top_candidate_can_win_five_seed_confirmation(self):
        state = ResearchLoopState()
        state.set_incumbent("baseline", 0.700, [0.700, 0.700, 0.700])
        state.start_round()
        for node_id, score in (("candidate-a", 0.706), ("candidate-b", 0.704)):
            decision = state.evaluate_candidate(
                node_id=node_id,
                fingerprint=candidate_fingerprint(node_id),
                score=score,
                seed_scores=[score, score, score],
            )
            self.assertTrue(decision.promoted)

        confirmation = state.accept_pending_candidate(
            node_id="candidate-b",
            final_node_id="candidate-b-final",
            score=0.705,
            seed_scores=[0.704, 0.705, 0.706, 0.705, 0.705],
        )

        self.assertTrue(confirmation.promoted)
        self.assertEqual(state.incumbent_node_id, "candidate-b-final")

    def test_patience_counts_failed_rounds_not_failed_candidates(self):
        state = ResearchLoopState(
            config=ResearchLoopConfig(max_research_rounds=3, patience=2)
        )
        state.set_incumbent("baseline", 0.70)
        state.start_round()

        for candidate_number in range(3):
            state.evaluate_candidate(
                node_id=f"candidate-{candidate_number}",
                fingerprint=candidate_fingerprint(f"code-{candidate_number}"),
                score=0.699,
                seed_scores=[0.699, 0.699, 0.699],
            )

        self.assertEqual(state.no_improvement_rounds, 0)
        state.finish_round()
        self.assertEqual(state.no_improvement_rounds, 1)
        self.assertFalse(state.should_stop)

    def test_patience_stops_repeated_failed_rounds(self):
        state = ResearchLoopState(
            config=ResearchLoopConfig(max_research_rounds=3, patience=2)
        )
        state.set_incumbent("baseline", 0.70)

        for round_number in range(2):
            state.start_round()
            state.evaluate_candidate(
                node_id=f"candidate-{round_number}",
                fingerprint=candidate_fingerprint(f"code-{round_number}"),
                score=0.699,
                seed_scores=[0.699, 0.699, 0.699],
            )
            state.finish_round()

        self.assertTrue(state.should_stop)

    def test_round_summary_keeps_component_deltas_and_seed_stability(self):
        state = ResearchLoopState()
        state.set_incumbent(
            "baseline",
            0.700,
            [0.699, 0.700, 0.701],
            metrics={"GAUC": 0.720, "nDCG@5": 0.680, "primary": 0.700},
        )
        state.start_round()
        state.evaluate_candidate(
            node_id="near-winner",
            fingerprint=candidate_fingerprint("history model"),
            score=0.701,
            seed_scores=[0.700, 0.701, 0.702],
            metrics={
                "GAUC": 0.724,
                "nDCG@5": 0.678,
                "primary": 0.701,
                "test primary": 0.999,
            },
            seed_metrics=[
                {"GAUC": 0.723, "nDCG@5": 0.677, "primary": 0.700},
                {"GAUC": 0.724, "nDCG@5": 0.678, "primary": 0.701},
                {"GAUC": 0.725, "nDCG@5": 0.679, "primary": 0.702},
            ],
            principal_change="causal history attention",
            components=["history_factor"],
        )
        state.finish_round()

        summary = state.round_summary(1)
        record = summary["candidates"][0]
        self.assertFalse(summary["test_metrics_used"])
        self.assertEqual(summary["feedback_scope"], "validation_only")
        self.assertNotIn("test primary", record["metrics"])
        self.assertAlmostEqual(record["metric_deltas"]["GAUC"], 0.004)
        self.assertAlmostEqual(record["metric_deltas"]["nDCG@5"], -0.002)
        self.assertAlmostEqual(record["metric_deltas"]["primary"], 0.001)
        self.assertGreater(record["seed_std"], 0.0)
        self.assertEqual(summary["near_winners"][0]["node_id"], "near-winner")

        prompt = state.experience_prompt(before_round=2)
        self.assertIn("causal history attention", prompt)
        self.assertIn("delta_GAUC=+0.004000", prompt)
        self.assertIn("delta_nDCG@5=-0.002000", prompt)
        self.assertNotIn("0.999", prompt)

    def test_factor_memory_keeps_model_context_and_custom_description(self):
        state = ResearchLoopState()
        state.set_incumbent(
            "baseline",
            0.700,
            metrics={"GAUC": 0.720, "nDCG@5": 0.680, "primary": 0.700},
        )
        state.start_round()
        state.evaluate_candidate(
            node_id="factor-candidate",
            fingerprint=candidate_fingerprint("factor-candidate"),
            score=0.704,
            seed_scores=[0.704, 0.704, 0.704],
            metrics={"GAUC": 0.723, "nDCG@5": 0.685, "primary": 0.704},
            principal_change="gated history diversity tower",
            role="causal_history_interest",
            category="history_interest",
            factor_ids=["custom_history_diversity"],
            factor_selection_reason="active users show repetitive histories",
            factor_cards=[
                {
                    "factor_id": "custom_history_diversity",
                    "semantics": "past-only diversity of recent interests",
                    "helps_when": ["active histories are repetitive"],
                    "model_fit": ["gated history tower"],
                    "avoid_when": ["history is empty"],
                    "data_cost": "low",
                    "leakage_rule": "past rows only",
                }
            ],
        )

        summary = state.memory.factor_summary()["custom_history_diversity"]

        self.assertEqual(summary["model_roles"], ["causal_history_interest"])
        self.assertAlmostEqual(summary["mean_gain"], 0.004)
        self.assertEqual(
            summary["self_description"]["semantics"],
            "past-only diversity of recent interests",
        )

    def test_near_winner_feedback_is_diverse_and_excludes_confirmed_winner(self):
        state = ResearchLoopState(
            config=ResearchLoopConfig(experience_top_k=2)
        )
        state.set_incumbent("baseline", 0.700)
        state.start_round()
        for node_id, score, change, component in (
            ("confirmed", 0.705, "history", "history_factor"),
            ("history-copy", 0.704, "history", "history_factor"),
            ("pairwise", 0.703, "hybrid BPR", "pairwise_loss"),
            ("temporal", 0.702, "time gap", "temporal_factor"),
        ):
            state.evaluate_candidate(
                node_id=node_id,
                fingerprint=candidate_fingerprint(node_id),
                score=score,
                seed_scores=[score, score, score],
                principal_change=change,
                components=[component],
            )
        state.pending_candidate_id = "confirmed"
        state.accept_pending_candidate(
            node_id="confirmed",
            score=0.705,
            seed_scores=[0.705, 0.705, 0.705],
        )
        state.finish_round()

        selected = state.memory.diverse_near_winners(round_number=1, limit=2)
        self.assertEqual(
            [record.node_id for record in selected],
            ["history-copy", "pairwise"],
        )


if __name__ == "__main__":
    unittest.main()
