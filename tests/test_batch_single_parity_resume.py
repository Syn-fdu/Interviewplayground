"""Stop/continue parity: an obsolete generation must not delete a resumed run.

No external model calls.  Both generations use the same saved scenario and
participant, just as the UI does when Continue is clicked immediately.
"""
import copy
import threading
import time
import types
import unittest
from unittest.mock import patch


class TestContinueOwnership(unittest.TestCase):
    def test_old_generation_cleanup_cannot_delete_new_generation(self):
        import server
        from playground.research import group_interviewer_bridge as bridge
        store = server.LocalStore()
        store.catalog = {
            "source_fingerprint": "fp",
            "candidates": [{"candidate_id": "one", "dataset": "ds", "scenario_eligible": True}],
            "source_index": {"ds": 0},
            "datasets": [{"id": "ds", "label": "Dataset"}],
        }
        store.source_zip = b"sample"
        scenario = {"scenario_id": "saved_scenario", "construction_mode": "full",
                    "provenance": {"candidate_id": "one", "dataset": "ds",
                                   "source_fingerprint_sha256": "fp", "guide_level": "level1"}}
        entered_old = threading.Event()
        release_old = threading.Event()
        entered_new = threading.Event()
        release_new = threading.Event()
        old_stop = threading.Event()
        new_stop = threading.Event()
        error = []

        def prepare(bundle, timeout_seconds):
            if not entered_old.is_set():
                entered_old.set()
                if not release_old.wait(5):
                    raise RuntimeError("old generation setup timed out")

        def interviewer(req, scene, source, revision, provider=None):
            entered_new.set()
            if not release_new.wait(5):
                raise RuntimeError("new generation timed out")
            return {"finished": True, "action": "conclude", "question": "Thank you."}

        batch = {
            "experiment_id": "job", "dataset": "ds", "source_fingerprint_sha256": "fp",
            "_scenario_snapshot": copy.deepcopy(scenario),
            "options": {"guide_level": "level1", "interviewer_provider": "local",
                        "respondent_provider": "local", "preparation_mode": "fast_hybrid",
                        "max_turns": 2, "language": "zh", "interviewer_timeout_seconds": 60,
                        "respondent_timeout_seconds": 60, "retrieval_timeout_seconds": 60}
        }

        def run(stop):
            try:
                server._run_batch_interview_scoped(copy.deepcopy(batch), "one", [], stop)
            except Exception as exc:
                error.append(exc)

        with patch.object(server, "STORE", store), \
             patch.object(server, "make_research_bundle", return_value={"audit": {"revision": 1}}), \
             patch.object(server, "prepare_evidence_index", side_effect=prepare), \
             patch.object(server, "_run_preparation_request", return_value={"prepared": True}), \
             patch("playground.research.interviewer_preparation.prepare_public_outline",
                   return_value={"ready": True}), \
             patch("playground.common.model_services.get",
                   return_value=types.SimpleNamespace(model="mock")), \
             patch.object(bridge, "run_next", side_effect=interviewer):
            old = threading.Thread(target=run, args=(old_stop,), daemon=True)
            old.start()
            self.assertTrue(entered_old.wait(3))
            old_stop.set()
            new = threading.Thread(target=run, args=(new_stop,), daemon=True)
            new.start()
            # The new run must not be blocked by the obsolete worker holding
            # STORE.lock during long evidence preparation.
            self.assertTrue(entered_new.wait(3), "new generation blocked by old preparation")
            with store.lock:
                active = list(store.scenarios)
                self.assertEqual(len(active), 1)
                self.assertTrue(active[0].startswith("saved_scenario_run_"))
                resumed_sid = active[0]
            release_old.set()
            old.join(3)
            self.assertFalse(old.is_alive())
            with store.lock:
                self.assertIn(resumed_sid, store.scenarios,
                              "obsolete generation deleted resumed scenario")
                self.assertIn(resumed_sid, store.research_bundles)
            release_new.set()
            new.join(3)
            self.assertFalse(new.is_alive())
            self.assertFalse(error, error)


if __name__ == "__main__":
    unittest.main()
