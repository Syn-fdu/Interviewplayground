"""Offline regression for tabbed Batch UI and history ordering."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from playground.research.batch_experiments import BatchManager


class TabbedBatchUITests(unittest.TestCase):
    def test_markup_and_script_share_controls(self):
        root = Path(__file__).resolve().parents[1]
        html = (root / "index.html").read_text(encoding="utf-8")
        js = (root / "assets/batch_experiments.js").read_text(encoding="utf-8")
        batch = html.split('<section class="app-view" id="batch"', 1)[1].split(
            '<section class="app-view" id="benchmark"', 1)[0]
        for marker in (
            'id="batchNewTab"', 'id="batchHistoryTab"',
            'id="batchNewPanel"', 'id="batchHistoryPanel"',
            'id="batchHistoryRows"',
            'id="batchStart"', 'id="batchStop"', 'id="batchResume"',
            'id="batchHistoryResume"', 'id="batchCreate"',
            'id="batchExport"', 'id="batchRows"',
        ):
            self.assertEqual(batch.count(marker), 1, marker)
        self.assertLess(batch.index('id="batchNewTab"'),
                        batch.index('id="batchNewPanel"'))
        self.assertLess(batch.index('id="batchHistoryTab"'),
                        batch.index('id="batchHistoryPanel"'))
        self.assertLess(batch.index('id="batchHistoryPanel"'),
                        batch.index('id="batch-current-title"'))
        from bs4 import BeautifulSoup
        section = BeautifulSoup(batch, "html.parser")
        for cid, heading in (
            ("batchInterviewerTimeout", "采访者参数"),
            ("batchRespondentTimeout", "受访者参数"),
            ("batchRetrievalTimeout", "受访者参数"),
        ):
            control = section.find(id=cid)
            self.assertEqual(control.find_parent("section").find("h3").get_text(strip=True), heading)
        self.assertEqual(len(section.select("#batchHistoryRows")), 1)
        self.assertEqual(len(section.select("#batchDeleteSelection")), 0)
        self.assertIn('setBatchMode(mode)', js)
        self.assertIn("'interviewplayground.batch.activeTab.v1'", js)
        self.assertIn("'/api/batch/create'", js)
        self.assertIn("'/api/batch/launch'", js)
        self.assertIn("'/api/batch/stop'", js)
        self.assertIn("'/api/batch/resume'", js)
        self.assertIn("'/api/batch/export?experiment_id='", js)
        self.assertIn("'/api/batch/delete'", js)

    def test_form_and_controls_are_separate_and_defaults_are_one(self):
        from bs4 import BeautifulSoup
        root = Path(__file__).resolve().parents[1]
        html = (root / "index.html").read_text(encoding="utf-8")
        js = (root / "assets" / "batch_experiments.js").read_text(encoding="utf-8")
        section = BeautifulSoup(html, "html.parser").select_one("#batch")
        config = section.select_one("#batchNewPanel")
        current = section.select_one(".batch-current-panel")
        history = section.select_one("#batchHistoryPanel")
        self.assertIsNotNone(config.select_one("#batchCreate"))
        self.assertIsNone(config.select_one("#batchStart"))
        self.assertTrue(all(current.select_one("#" + key) for key in (
            "batchStart", "batchStop", "batchResume", "batchExport")))
        self.assertIsNotNone(history.select_one("#batchHistoryResume"))
        self.assertIsNotNone(history.select_one("#batchHistoryRows"))
        self.assertIsNone(history.select_one("#batchDeleteSelection"))
        self.assertEqual(len(history.select(".batch-v2-history-table th")), 4)
        self.assertIn('name="batchHistoryPick"', js)
        self.assertNotIn('name="batchDeletePick"', js)
        self.assertIn("selected.length === 1", js)
        self.assertIn("selected.length === 0 || containsActive", js)
        for level, title in (
            ("level0", "浓缩大纲"), ("level1", "基本大纲"), ("level2", "细致大纲")
        ):
            self.assertIn(title, config.select_one(f'#batchGuideLevel option[value="{level}"]').text)
        for cid in ("batchQualityMode", "batchTraitValue-knowledge",
                    "batchTraitValue-disclosure"):
            self.assertEqual(config.select_one(f"#{cid}").find_parent("section").find("h3").text, "受访者参数")
        for name in ("knowledge", "understanding", "reflexivity",
                     "memory", "verbosity", "disclosure"):
            self.assertEqual(config.select_one(f'[data-batch-trait="{name}"]')["value"], "1")
        self.assertTrue(bool(config.select_one("#batchConsent").has_attr("checked")))
        for role in ("Interviewer", "Respondent"):
            options = config.select_one(f"#batch{role}").find_all("option")
            self.assertEqual([o["value"] for o in options], ["shubi", "local"])
        self.assertIn("'/api/batch/create'", js)
        self.assertIn("'/api/batch/launch'", js)
        self.assertIn("'/api/batch/resume'", js)
        self.assertIn("historySelection", js)
        history_load = js.split("$('batchHistoryResume').addEventListener('click'", 1)[1].split(
            "$('batchRestoreConfig')", 1)[0]
        self.assertNotIn("'/api/batch/resume'", history_load)
        self.assertNotIn("'/api/batch/launch'", history_load)

    def test_history_only_exposes_newest_ten_without_deleting_older(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = BatchManager(tmp, lambda *_args: {"status": "completed"})
            data = {}
            for i in range(12):
                job_id = f"job_{11-i:02d}"
                folder = Path(tmp) / job_id
                folder.mkdir()
                (folder / "experiment.json").write_text("{}", encoding="utf-8")
                data[job_id] = {
                    "experiment_id": job_id,
                    "dataset": "test", "state": "completed",
                    "created_at": f"2026-11-{i+1:02d}T12:00:00+00:00",
                    "updated_at": None, "finished_at": None,
                    "options": {}, "progress": {}, "interviews": []}
            with patch.object(manager, "_read", side_effect=lambda job: data[job]):
                found = manager.list()
            self.assertEqual(len(found), 10)
            self.assertEqual(found[0]["created_at"], "2026-11-12T12:00:00+00:00")
            self.assertEqual(found[-1]["created_at"], "2026-11-03T12:00:00+00:00")
            self.assertEqual(len(list(Path(tmp).glob("*/experiment.json"))), 12)


if __name__ == "__main__":
    unittest.main()
