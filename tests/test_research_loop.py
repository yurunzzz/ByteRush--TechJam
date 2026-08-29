import unittest

from ai_scientist.treesearch.research_loop import (
    PromotionPolicy,
    ResearchLoopConfig,
    ResearchLoopState,
    candidate_fingerprint,
    estimate_max_experiment_runs,
    extract_primary_score,
    research_loop_config_from_mapping,
)


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
    def test_default_patience_matches_official_convergence_rule(self):
        self.assertEqual(ResearchLoopConfig().patience, 3)

    def test_production_budget_is_calculated_from_all_loop_layers(self):
        budget = estimate_max_experiment_runs(
            ResearchLoopConfig(), stage4_max_components=6
        )

        self.assertEqual(budget["stage2_seed_evaluation"], 9)
        self.assertEqual(budget["per_research_round"], 102)
        self.assertEqual(budget["maximum_search_iterations"], 184)
        self.assertEqual(budget["maximum_total"], 337)

    def test_partial_mapping_keeps_production_defaults(self):
        config = research_loop_config_from_mapping(
            {"candidate_tuning_iterations": 4}
        )

        self.assertEqual(config.candidate_tuning_iterations, 4)
        self.assertEqual(config.baseline_tuning_iterations, 20)
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


if __name__ == "__main__":
    unittest.main()
