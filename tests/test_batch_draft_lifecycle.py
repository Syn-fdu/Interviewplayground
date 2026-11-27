"""UI contract: configuring is not execution; history loading is not execution."""
import tempfile
import time
import unittest

from playground.research.batch_experiments import BatchManager, BatchError


class BatchDraftLifecycleTests(unittest.TestCase):
    def test_create_draft_then_start(self):
        calls = []
        def worker(batch, cid, turns, stop):
            calls.append((cid, len(turns)))
            turns.append({"question": "Q", "answer": "A"})
            return {"status": "completed"}
        with tempfile.TemporaryDirectory() as directory:
            manager = BatchManager(directory, worker)
            options = {"max_turns": 5}
            candidates = [{"candidate_id": "c1", "scenario_eligible": True}]
            data = manager.start("dataset", "fingerprint", candidates, options,
                                 launch=False)
            job_id = data["experiment_id"]
            self.assertEqual(data["state"], "created")
            self.assertEqual(calls, [])
            self.assertNotIn(job_id, manager.running)
            self.assertEqual(manager.status(job_id)["counts"]["pending"], 1)
            with self.assertRaises(BatchError):
                manager.resume(job_id, "fingerprint")
            self.assertEqual(manager.status(job_id)["state"], "created")
            manager.launch_created(job_id, "fingerprint")
            for _ in range(200):
                if manager.status(job_id)["state"] == "completed":
                    break
                time.sleep(.01)
            self.assertEqual(manager.status(job_id)["state"], "completed")
            self.assertEqual(calls, [("c1", 0)])
            # The completed turn stays in durable result files.
            self.assertEqual(manager._load_partial(job_id, 1)["turns"],
                             [{"question": "Q", "answer": "A"}])

    def test_history_continuation_keeps_completed_round(self):
        def worker(batch, cid, turns, stop):
            if not turns:
                turns.append({"question": "Q1", "answer": "A1"})
                batch["_report_progress"]("turn_saved", turn_count=1,
                                          checkpoint={"step": 1}, last_turn_seconds=1.0)
                return {"status": "incomplete", "error": "interrupted test"}
            turns.append({"question": "Q2", "answer": "A2"})
            batch["_report_progress"]("turn_saved", turn_count=2,
                                      checkpoint={"step": 2}, last_turn_seconds=1.0)
            return {"status": "completed"}
        with tempfile.TemporaryDirectory() as directory:
            manager = BatchManager(directory, worker)
            data = manager.start("dataset", "fingerprint", [
                {"candidate_id": "c1", "scenario_eligible": True}
            ], {"max_turns": 2}, launch=False)
            job = data["experiment_id"]
            manager.launch_created(job, "fingerprint")
            for _ in range(200):
                if manager.status(job)["state"] not in ("running", "stopping"):
                    break
                time.sleep(.01)
            self.assertEqual(manager._load_partial(job, 1)["turns"],
                             [{"question": "Q1", "answer": "A1"}])
            # Status/view is read-only; resume is the explicit execution action.
            manager.status(job)
            self.assertEqual(len(manager._load_partial(job, 1)["turns"]), 1)
            manager.resume(job, "fingerprint")
            for _ in range(200):
                if manager.status(job)["state"] == "completed":
                    break
                time.sleep(.01)
            self.assertEqual(len(manager._load_partial(job, 1)["turns"]), 2)


if __name__ == "__main__":
    unittest.main()
