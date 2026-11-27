"""Regression coverage for zero-turn batch guide binding failures."""
import unittest
from unittest.mock import patch
from playground.research.guide_level_projection import bind_batch_level
from playground.research.batch_experiments import safe_batch_error


class BatchGuideBindingTests(unittest.TestCase):
    def test_successful_level_is_bound_to_batch_scenario(self):
        scenario = {"provenance": {"guide_level": "level1"}}
        def matched(scene, level, category):
            scene["provenance"]["graded_guide_status"] = "matched"
            return True
        with patch('playground.research.guide_level_projection.apply_level', side_effect=matched):
            self.assertTrue(bind_batch_level(scenario, "level1", None))
        provenance = scenario['provenance']
        self.assertEqual(provenance['available_guide_levels'], ['level1'])
        self.assertIn(provenance['guide_level'], provenance['available_guide_levels'])

    def test_unmatched_guide_is_not_falsely_bound(self):
        scenario = {"provenance": {"guide_level": "level1"}}
        with patch('playground.research.guide_level_projection.apply_level', return_value=False):
            self.assertFalse(bind_batch_level(scenario, "level1", "unknown"))
        self.assertNotIn('available_guide_levels', scenario['provenance'])

    def test_diagnostic_retains_stage_and_masks_token(self):
        message = safe_batch_error(ValueError('This level was not bound to the selected scenario'), 'interviewer_preparation')
        self.assertIn('interviewer_preparation', message)
        self.assertIn('This level was not bound', message)
        self.assertNotIn('sk-secret_exampleabcdefghijk', safe_batch_error(ValueError('sk-secret_exampleabcdefghijk')))

    def test_exact_selected_level_not_all_levels(self):
        scenario = {"provenance": {}}
        with patch('playground.research.guide_level_projection.apply_level', return_value=True):
            self.assertTrue(bind_batch_level(scenario, "level2", "category-a"))
        self.assertEqual(scenario['provenance']['available_guide_levels'], ['level2'])
        self.assertEqual(scenario['provenance']['guide_category'], 'category-a')

if __name__ == '__main__':
    unittest.main()
