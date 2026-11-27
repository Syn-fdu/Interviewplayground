"""Functional packaging guard: bundled interviews must be importable, not just named."""
from pathlib import Path
import unittest

from playground.research.data_intake import pack_data_directory, build_catalog
from playground.research.graded_guides import list_datasets


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "resources" / "bundled_data" / "InterviewData_Cumulative_Refined_14" / "social_research"


class BundledDatasetIntegrityTests(unittest.TestCase):
    def test_all_seven_datasets_have_real_eligible_candidates(self):
        # Public GitHub clones intentionally omit private interview records.
        # Skip only when *none* of the seven datasets is provided; if any are
        # present, demand all seven so partial deployments cannot pass silently.
        records = list(DATA.glob("*/processed/interview_records.json"))
        if not records:
            self.skipTest("Private research datasets are excluded from the public source checkout")
        self.assertEqual(len(records), 7, "Incomplete bundled dataset: expected 7 interview record files")
        catalogue = build_catalog(pack_data_directory(DATA))
        self.assertEqual(len(catalogue["datasets"]), 7)
        self.assertEqual(catalogue["total_candidates"], 97)
        self.assertEqual(catalogue["total_scenario_eligible"], 97)
        for dataset in catalogue["datasets"]:
            count = sum(
                row["dataset"] == dataset["id"] and row.get("scenario_eligible")
                for row in catalogue["candidates"]
            )
            self.assertGreater(count, 0, dataset["id"])

    def test_all_grades_kept(self):
        self.assertEqual(len(list_datasets()), 26)


if __name__ == "__main__":
    unittest.main()
