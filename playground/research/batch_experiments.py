"""Durable, local-only batch orchestration. No HTML, model keys, or evaluation code.

The injected run_one callback owns the existing interview engine. Each participant
gets a durable result and transcript before the next participant begins. A stopped
or crashed task may be resumed at participant boundaries; partial conversations
are *not* misrepresented as completed interviews.
"""
from __future__ import annotations

import copy
import io
import json
import multiprocessing
import signal
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ID = re.compile(r"^[a-f0-9]{32}$")
STATUSES = {"completed", "failed", "incomplete", "skipped", "interrupted"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def transcript_text(turns, ending=None):
    lines = []
    for i, turn in enumerate(turns, 1):
        lines.extend((f"[Turn {i}]", "Interviewer: " + turn["question"],
                      "Respondent: " + turn["answer"], ""))
    if ending:
        lines.extend(("[Closing]", "Interviewer: " + ending, ""))
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def safe_batch_error(exc, stage=None):
    """Include a useful stage and message without leaking common API credentials."""
    code = getattr(exc, "code", None) or getattr(exc, "kind", None) or "execution_error"
    code = re.sub(r"[^a-zA-Z0-9_-]", "", str(code))[:64] or "execution_error"
    raw = str(exc).strip().replace("\r", " ").replace("\n", " ")[:1000]
    raw = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", raw)
    raw = re.sub(r"(?i)(api[_-]?key|authorization|secret|token)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", raw)
    raw = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", raw)
    where = f" [{stage}]" if stage else ""
    return f"{type(exc).__name__} ({code}){where}" + (f": {raw}" if raw else "")


class BatchError(ValueError):
    pass


class BatchManager:
    def __init__(self, root, run_one, *, isolate_calls=False):
        self.root = Path(root)
        self.run_one = run_one
        self.lock = threading.RLock()
        self.running = set()
        self.stop_events = {}
        # Some third-party calls are not interruptible. Track any surviving
        # provider threads for diagnostics and block their late persistence.
        self.inflight = {}
        self.active_calls = {}
        # On POSIX, provider execution can run in a killable subprocess.
        self.isolate_calls = bool(isolate_calls and os.name == "posix")
        self.processes = {}

    def _directory(self, job_id):
        if not isinstance(job_id, str) or not ID.fullmatch(job_id):
            raise BatchError("Invalid batch ID")
        return self.root / job_id

    def _read(self, job_id):
        path = self._directory(job_id) / "experiment.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("experiment_id") != job_id:
                raise BatchError("Invalid batch metadata")
            return data
        except FileNotFoundError as exc:
            raise BatchError("Batch not found") from exc

    @staticmethod
    def summary(data):
        results = data["interviews"]
        counts = {status: sum(r["status"] == status for r in results) for status in STATUSES}
        counts["pending"] = sum(r["status"] == "pending" for r in results)
        counts["running"] = sum(r["status"] == "running" for r in results)
        return {"experiment_id": data["experiment_id"], "dataset": data["dataset"],
                 "state": data["state"], "created_at": data["created_at"],
                "options": copy.deepcopy(data.get("options", {})),  # key-free snapshot
                "finished_at": data.get("finished_at"), "total": len(results),
                "updated_at": data.get("updated_at"), "progress": data.get("progress"),
                "counts": counts, "interviews": [
                    {key: row.get(key) for key in ("candidate_id", "status", "turn_count", "error", "last_turn_seconds", "last_turn_at", "last_interviewer_seconds", "last_respondent_seconds")}
                    for row in results]}

    def list(self):
        with self.lock:
            records = []
            for path in self.root.glob("*/experiment.json"):
                try:
                    records.append(self.summary(self._read(path.parent.name)))
                except (BatchError, ValueError, KeyError, TypeError):
                    continue
            # UUID directory names are random, not chronological. Keep the
            # newest ten visible; do not automatically delete older research data.
            records.sort(key=lambda r: (r.get("created_at") or "", r["experiment_id"]), reverse=True)
            return records[:10]

    def status(self, job_id):
        with self.lock:
            data = self._read(job_id)
            result = self.summary(data)
            if job_id not in self.running and result["state"] in ("running", "stopping"):
                result["state"] = "interrupted"
            self.inflight[job_id] = {t for t in self.inflight.get(job_id, ()) if t.is_alive()}
            result["active_provider_calls"] = len(self.inflight[job_id]) + sum(p.is_alive() for p in self.processes.get(job_id, ()))
            return result

    def start(self, dataset, fingerprint, candidates, options, before_launch=None, launch=True):
        if not isinstance(dataset, str) or not dataset or len(dataset) > 200:
            raise BatchError("Select a valid dataset")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise BatchError("Missing source fingerprint")
        if not candidates or len(candidates) > 2000:
            raise BatchError("Dataset must contain 1–2000 records")
        with self.lock:
            if self.running and launch:
                raise BatchError("Another batch is already running")
            job_id = uuid.uuid4().hex
            rows = []
            for item in candidates:
                cid = item["candidate_id"]
                usable = bool(item.get("scenario_eligible"))
                rows.append({"candidate_id": cid, "status": "pending" if usable else "skipped",
                             "turn_count": 0, "error": None if usable else "Scenario ineligible"})
            data = {"schema_version": "batch_experiment_v1", "experiment_id": job_id,
                    "dataset": dataset, "source_fingerprint_sha256": fingerprint,
                    "options": copy.deepcopy(options), "created_at": utc_now(),
                    "finished_at": None, "state": "running" if launch else "created", "updated_at": utc_now(),
                    "progress": {"phase": "queued", "candidate_id": None, "started_at": None,
                                 "phase_started_at": None, "last_activity_at": utc_now(),
                                 "last_turn_seconds": None, "last_turn_at": None},
                    "interviews": rows}
            if before_launch is not None:
                before_launch(job_id)
            atomic_json(self._directory(job_id) / "experiment.json", data)
            if launch:
                self._launch(job_id)
            return self.summary(data)

    def launch_created(self, job_id, fingerprint):
        """Start a configured draft; unlike resume, no prior turn is replayed."""
        with self.lock:
            if self.running:
                raise BatchError("Another batch is already running")
            data = self._read(job_id)
            if data["state"] != "created":
                raise BatchError("Only a newly created experiment can be started")
            if data["source_fingerprint_sha256"] != fingerprint:
                raise BatchError("Reload the original dataset before starting")
            data["state"] = "running"
            data["progress"]["phase"] = "queued"
            data["updated_at"] = utc_now()
            atomic_json(self._directory(job_id) / "experiment.json", data)
            self._launch(job_id)
            return self.summary(data)

    def _launch(self, job_id):
        event = threading.Event()
        self.running.add(job_id)
        self.stop_events[job_id] = event
        threading.Thread(target=self._worker, args=(job_id, event), daemon=True,
                         name="batch-" + job_id[:8]).start()

    def resume(self, job_id, fingerprint):
        """Continue incomplete participants only from their saved native checkpoint."""
        with self.lock:
            if self.running:
                raise BatchError("Another batch is already running")
            # A stopped generation cannot commit new turns: its stop Event is
            # checked under the manager lock by every progress callback. The
            # new generation uses a separate Event and its own transcript copy.
            # This matches the single-interview AbortController behaviour: do
            # not wait for an uncooperative remote request before resuming.
            self.inflight[job_id] = {t for t in self.inflight.get(job_id, ()) if t.is_alive()}
            data = self._read(job_id)
            if data["source_fingerprint_sha256"] != fingerprint:
                raise BatchError("Reload the exact original dataset before resuming")
            if data["state"] == "created":
                raise BatchError("Experiment has not started; use Start, not Continue")
            if data["state"] == "completed" and all(
                    r["status"] in ("completed", "skipped") for r in data["interviews"]):
                return self.summary(data)
            for index, row in enumerate(data["interviews"], 1):
                if row["status"] in ("running", "failed", "incomplete", "interrupted"):
                    saved = self._load_partial(job_id, index)
                    durable_turns = saved.get("turns", [])
                    if row.get("turn_count", 0) > len(durable_turns):
                        raise BatchError(
                            f"Interview {index} metadata records {row['turn_count']} turns, "
                            f"but only {len(durable_turns)} were durably committed; "
                            "refusing to restart at turn 0. Restore the missing result file.")
                    if durable_turns and not saved.get("checkpoint"):
                        raise BatchError("Saved interview has turns but no native checkpoint; "
                                         "refusing to silently restart at turn 0")
                    row.update(status="pending", error=None, turn_count=len(durable_turns))
                    row["resume_from_turn"] = len(durable_turns)
            data.update(state="running", finished_at=None)
            first_pending = next((r for r in data["interviews"] if r["status"] == "pending"), None)
            resume_turn = first_pending.get("turn_count", 0) if first_pending else 0
            data["progress"] = {"phase": "resume_queued", "candidate_id": first_pending["candidate_id"] if first_pending else None,
                                "resume_from_turn": resume_turn,
                                "started_at": None, "phase_started_at": None,
                                "last_activity_at": utc_now(), "last_turn_seconds": None,
                                "last_turn_at": None}
            atomic_json(self._directory(job_id) / "experiment.json", data)
            self._launch(job_id)
            return self.summary(data)

    def _scenario_path(self, job_id, index):
        return self._directory(job_id) / "stages" / f"scenario_{index:04d}.json"

    def _load_scenario(self, job_id, index, data, cid):
        path = self._scenario_path(job_id, index)
        if not path.exists():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if (document.get("experiment_id") != job_id or
            document.get("candidate_id") != cid or
            document.get("source_fingerprint_sha256") != data["source_fingerprint_sha256"] or
            document.get("guide_level") != data["options"].get("guide_level", "level1") or
            document.get("guide_category") != data["options"].get("guide_category") or
            not isinstance(document.get("scenario"), dict)):
            raise BatchError(f"Saved scenario stage does not match candidate {cid}; refusing to rebuild silently")
        return document["scenario"]

    def _save_scenario(self, job_id, index, data, cid, scenario):
        atomic_json(self._scenario_path(job_id, index), {
            "schema_version": "batch_scenario_stage_v1", "experiment_id": job_id,
            "candidate_id": cid,
            "source_fingerprint_sha256": data["source_fingerprint_sha256"],
            "guide_level": data["options"].get("guide_level", "level1"),
            "guide_category": data["options"].get("guide_category"),
            "scenario": scenario})

    def _question_path(self, job_id, index):
        return self._directory(job_id) / "stages" / f"question_{index:04d}.json"

    def _load_question(self, job_id, index, data, cid, turn_count):
        path = self._question_path(job_id, index)
        if not path.exists():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        if (doc.get("schema_version") != "batch_pending_question_v1" or
                doc.get("experiment_id") != job_id or
                doc.get("candidate_id") != cid or
                doc.get("source_fingerprint_sha256") != data["source_fingerprint_sha256"] or
                not isinstance(doc.get("turn_index"), int) or
                not isinstance(doc.get("question"), str) or not doc["question"].strip() or
                not isinstance(doc.get("interviewer_checkpoint"), dict)):
            raise BatchError(f"Saved interviewer question for {cid} has invalid metadata")
        if doc["turn_index"] < turn_count:
            # The full answered turn is already durable; the pending question is stale.
            path.unlink(missing_ok=True)
            return None
        if doc["turn_index"] != turn_count:
            raise BatchError(f"Saved interviewer question turn does not match transcript for {cid}")
        return {"question": doc["question"],
                "turn_index": doc["turn_index"],
                "interviewer_checkpoint": doc["interviewer_checkpoint"]}

    def _save_question(self, job_id, index, data, cid, artifact):
        if (not isinstance(artifact, dict) or
                not isinstance(artifact.get("question"), str) or
                not artifact["question"].strip() or
                not isinstance(artifact.get("turn_index"), int) or
                not isinstance(artifact.get("interviewer_checkpoint"), dict)):
            raise BatchError("Interviewer question checkpoint is incomplete")
        atomic_json(self._question_path(job_id, index), {
            "schema_version": "batch_pending_question_v1",
            "experiment_id": job_id, "candidate_id": cid,
            "source_fingerprint_sha256": data["source_fingerprint_sha256"],
            **copy.deepcopy(artifact)})

    def _load_partial(self, job_id, index):
        path = self._directory(job_id) / "results" / f"interview_{index:04d}.json"
        if not path.exists():
            return {"turns": []}
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc.get("turns"), list):
            raise BatchError("Invalid saved interview transcript")
        timings = doc.get("turn_timings", [])
        if not isinstance(timings, list) or len(timings) > len(doc["turns"]):
            raise BatchError("Invalid saved interview timing metadata")
        return doc

    def delete(self, job_id):
        """Delete precisely one stopped job, never arbitrary paths or live jobs."""
        with self.lock:
            if job_id in self.running:
                raise BatchError("Cannot delete a running batch; interrupt it first")
            self._read(job_id)
            shutil.rmtree(self._directory(job_id))
            return {"deleted": job_id}

    def prune(self, ids):
        if not isinstance(ids, list) or not ids or len(ids) > 100:
            raise BatchError("Choose 1–100 stopped experiments to remove")
        if len(set(ids)) != len(ids):
            raise BatchError("Duplicate experiment ID")
        with self.lock:
            for job_id in ids:
                if job_id in self.running:
                    raise BatchError("Cannot delete a running batch; interrupt it first")
                self._read(job_id)
            for job_id in ids:
                shutil.rmtree(self._directory(job_id))
        return {"deleted": ids}

    def stop(self, job_id):
        """Commit the cancellation boundary immediately; close sockets outside lock.

        The batch's durable progress must not move after this point. Socket
        shutdown can block in third-party transports, so it must never delay
        the HTTP stop response or hold the manager's persistence lock.
        """
        with self.lock:
            data = self._read(job_id)
            event = self.stop_events.get(job_id)
            if event is not None:
                event.set()
                now = utc_now()
                data["state"] = "interrupted"
                for index, row in enumerate(data["interviews"], 1):
                    if row["status"] == "running":
                        saved = self._load_partial(job_id, index)
                        row.update(status="interrupted", turn_count=len(saved.get("turns", [])),
                                   error="Interrupted by researcher")
                        if saved.get("turns"):
                            saved.update(status="interrupted", error="Interrupted by researcher",
                                         recorded_at=now)
                            self._save_interview(self._directory(job_id), index, saved)
                data["progress"] = {**(data.get("progress") or {}),
                                    "phase": "interrupted", "phase_started_at": now,
                                    "last_activity_at": now}
                data["updated_at"] = now
                data["finished_at"] = now
                atomic_json(self._directory(job_id) / "experiment.json", data)
                self.running.discard(job_id)
                self.stop_events.pop(job_id, None)
            result = self.summary(data)
            result["active_provider_calls"] = (sum(t.is_alive() for t in self.inflight.get(job_id, ()))
                                               + sum(p.is_alive() for p in self.processes.get(job_id, ())))
        if event is not None:
            # Close sockets outside the manager lock to avoid stopping other
            # persistence operations while the network transport is closing.
            from playground.common.interruptible_http import abort_event
            aborted = abort_event(event)
            # Kill the entire isolated provider process group. This also stops
            # descendant preparation threads and subprocesses inside that group.
            for process in tuple(self.processes.get(job_id, ())):
                self._terminate_process(process)
            # Allow cooperatively cancelled requests a tiny scheduling window
            # to finish, without ever waiting indefinitely on a stuck provider.
            for old in tuple(self.inflight.get(job_id, ())):
                if old.is_alive():
                    old.join(timeout=0.05)
            result["active_provider_calls"] = (sum(t.is_alive() for t in self.inflight.get(job_id, ()))
                                               + sum(p.is_alive() for p in self.processes.get(job_id, ())))
            result["progress"]["cancelled_http_sockets"] = aborted
            with self.lock:
                latest = self._read(job_id)
                # Do not overwrite a newer resumed generation.
                if self.stop_events.get(job_id) is None and latest.get("state") == "interrupted":
                    latest["progress"]["cancelled_http_sockets"] = aborted
                    atomic_json(self._directory(job_id) / "experiment.json", latest)
        return result

    @staticmethod
    def _terminate_process(process):
        if process.pid is None:
            return
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            process.join(0.35)
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.join(0.35)
        else:
            process.join(timeout=0)

    def _isolated_invoke(self, job_id, stop, data, cid, turns, checkpoint, scenario_snapshot, pending_question, progress, outcome, completed):
        """Transport complete turns to the parent; no child writes durable files."""
        ctx = multiprocessing.get_context('fork')
        receiver, sender = ctx.Pipe(duplex=True)
        child_stop = ctx.Event()

        def child_main():
            # The child has its own process group, so descendants can be stopped.
            try:
                os.setsid()
            except OSError:
                pass
            receiver.close()
            try:
                runtime_data = dict(data)
                runtime_data['_checkpoint'] = checkpoint
                runtime_data['_scenario_snapshot'] = scenario_snapshot
                runtime_data['_pending_question'] = copy.deepcopy(pending_question)
                def child_progress(phase, turn_count=None, last_turn_seconds=None, checkpoint=None, artifact=None):
                    sender.send(('progress', phase, turn_count, last_turn_seconds, checkpoint,
                                 copy.deepcopy(turns), copy.deepcopy(artifact)))
                    if phase in ('turn_saved', 'scenario_saved', 'interviewer_question_saved'):
                        # The next LLM question must not start until the parent
                        # has atomically written this turn + native checkpoint.
                        # Without this ACK, Stop can kill the child while a
                        # fully answered turn is still queued in the Pipe.
                        reply = sender.recv()
                        if reply != ('committed', phase, turn_count):
                            raise BatchError('Parent did not acknowledge durable turn checkpoint')
                runtime_data['_report_progress'] = child_progress
                from playground.common.interruptible_http import model_call_scope
                with model_call_scope(child_stop):
                    result = self.run_one(runtime_data, cid, turns, child_stop)
                sender.send(('result', result, copy.deepcopy(turns)))
            except BaseException as exc:
                try:
                    sender.send(('error', safe_batch_error(exc)))
                except (BrokenPipeError, OSError):
                    pass
            finally:
                sender.close()

        process = ctx.Process(target=child_main, name='batch-provider-' + job_id[:8], daemon=True)
        with self.lock:
            self.processes.setdefault(job_id, set()).add(process)
        try:
            process.start()
            sender.close()
            while not stop.is_set():
                if receiver.poll(0.025):
                    try:
                        message = receiver.recv()
                    except EOFError:
                        break
                    if message[0] == 'progress':
                        _, phase, count, seconds, cp, snapshot, artifact = message
                        if stop.is_set():
                            break
                        turns[:] = snapshot
                        progress(phase, count, seconds, cp, artifact)
                        if phase in ('turn_saved', 'scenario_saved', 'interviewer_question_saved') and not stop.is_set():
                            # ACK only after _save_interview and experiment.json
                            # have both completed successfully.
                            receiver.send(('committed', phase, count))
                    elif message[0] == 'result':
                        turns[:] = message[2]
                        outcome['result'] = message[1]
                        break
                    elif message[0] == 'error':
                        outcome['exception'] = RuntimeError(message[1])
                        break
                elif not process.is_alive():
                    break
            if stop.is_set():
                child_stop.set()
                self._terminate_process(process)
            elif process.is_alive():
                process.join(0.1)
            if not stop.is_set() and not outcome:
                outcome['exception'] = RuntimeError(
                    'Isolated provider process exited without returning a result')
        except BaseException as exc:
            outcome['exception'] = exc
            if process.pid is not None:
                self._terminate_process(process)
        finally:
            receiver.close()
            completed.set()
            with self.lock:
                self.processes.get(job_id, set()).discard(process)

    def _worker(self, job_id, stop):
        try:
            with self.lock:
                data = self._read(job_id)
            for row in data["interviews"]:
                if stop.is_set():
                    break
                if row["status"] != "pending":
                    continue
                cid = row["candidate_id"]
                with self.lock:
                    row["status"] = "running"
                    now = utc_now()
                    prior = self._load_partial(job_id, next(i for i, item in enumerate(data["interviews"]) if item is row) + 1)
                    starting_at = len(prior.get("turns", []))
                    # A failed/missing result must never silently reset an
                    # acknowledged participant to the beginning.
                    if row.get("turn_count", 0) > starting_at:
                        raise BatchError(
                            f"Refusing to roll back candidate {cid}: "
                            f"metadata={row['turn_count']}, durable={starting_at}")
                    if starting_at and not prior.get("checkpoint"):
                        raise BatchError("Cannot resume saved turns without checkpoint")
                    staged = self._load_scenario(job_id, next(i for i, item in enumerate(data["interviews"]) if item is row) + 1, data, cid)
                    data["progress"] = {"phase": "scenario_restored" if staged is not None else "scenario_setup", "candidate_id": cid,
                                        "resume_from_turn": starting_at,
                                        "started_at": now, "phase_started_at": now,
                                        "last_activity_at": now, "last_turn_seconds": None,
                                        "last_turn_at": None}
                    data["updated_at"] = now
                    atomic_json(self._directory(job_id) / "experiment.json", data)
                index = next(i for i, item in enumerate(data["interviews"]) if item is row) + 1
                saved = self._load_partial(job_id, index)
                turns = copy.deepcopy(saved.get("turns", []))
                # Diagnostics are sidecar metadata, never part of respondent history.
                turn_timings = copy.deepcopy(saved.get("turn_timings", []))
                if not isinstance(turn_timings, list) or len(turn_timings) > len(turns):
                    raise BatchError("Saved timing metadata is inconsistent with completed turns")
                turn_timings.extend([None] * (len(turns) - len(turn_timings)))
                checkpoint = copy.deepcopy(saved.get("checkpoint"))
                if len(turns) != row.get("turn_count", 0):
                    raise BatchError("Resume boundary changed between metadata and result read")
                if turns and checkpoint is None:
                    raise BatchError("Cannot start a new generation at turn 0: checkpoint missing")
                # Immutable expected boundary carried into the child. A later
                # stage cannot silently change the resumed turn index.
                resume_boundary = len(turns)
                outcome = {}
                completed = threading.Event()
                scenario_snapshot = copy.deepcopy(self._load_scenario(job_id, index, data, cid))
                pending_question = copy.deepcopy(self._load_question(job_id, index, data, cid, len(turns)))

                def progress(phase, turn_count=None, last_turn_seconds=None, checkpoint=None, artifact=None):
                    # Do not persist any late callbacks after cancellation.
                    with self.lock:
                        # Never accept a turn committed after Stop; the provider
                        # might have generated it *after* cancellation. The last
                        # previously acknowledged full-turn checkpoint is durable.
                        if completed.is_set() or stop.is_set():
                            return
                        now = utc_now()
                        info = data["progress"]
                        if phase != info.get("phase"):
                            info["phase_started_at"] = now
                        info.update(phase=phase, candidate_id=cid, last_activity_at=now,
                                    resume_from_turn=resume_boundary)
                        if turn_count is not None:
                            if turn_count < resume_boundary:
                                raise BatchError(
                                    f"Resume rollback: expected at least {resume_boundary} "
                                    f"turns, provider reported {turn_count}")
                            # A count greater than the durable boundary is
                            # provisional until an atomic turn_saved commit.
                            row["turn_count"] = max(len(self._load_partial(job_id, index).get("turns", [])),
                                                    resume_boundary)
                        if last_turn_seconds is not None:
                            info["last_turn_seconds"] = round(last_turn_seconds, 3)
                            info["last_turn_at"] = now
                            row["last_turn_seconds"] = round(last_turn_seconds, 3)
                            row["last_turn_at"] = now
                        data["updated_at"] = now
                        base = self._directory(job_id)
                        if phase == "scenario_saved":
                            if not isinstance(artifact, dict):
                                raise BatchError("Scenario stage requires a serializable scenario document")
                            self._save_scenario(job_id, index, data, cid, copy.deepcopy(artifact))
                        if phase == 'interviewer_question_saved':
                            if artifact.get('turn_index') != len(self._load_partial(job_id, index).get('turns', [])):
                                raise BatchError('Pending question boundary has changed')
                            self._save_question(job_id, index, data, cid, artifact)
                        if phase == "turn_saved":
                            if not isinstance(artifact, dict) or not isinstance(artifact.get('timing'), dict):
                                # Legacy/custom run_one providers may not emit timing metadata.
                                artifact = {'timing': None}
                            if checkpoint is None:
                                raise BatchError("Native interviewer checkpoint missing after saved turn")
                            frozen = copy.deepcopy(turns)
                            previous = self._load_partial(job_id, index)
                            if len(frozen) < len(previous.get("turns", [])):
                                raise BatchError("Turn commit would roll back durable interview history")
                            if turn_count != len(frozen):
                                raise BatchError("Turn count and transcript differ during checkpoint commit")
                            if len(frozen) != len(turn_timings) + 1:
                                raise BatchError("Timing metadata does not match next committed round")
                            timing = copy.deepcopy(artifact['timing'])
                            turn_timings.append(timing)
                            document = {"schema_version": "batch_interview_v1", "experiment_id": job_id,
                                        "dataset": data["dataset"], "candidate_id": cid,
                                        "status": "running", "turn_count": len(frozen), "turns": frozen,
                                        "turn_timings": copy.deepcopy(turn_timings),
                                        "checkpoint": copy.deepcopy(checkpoint), "closing_message": None,
                                        "error": None, "options": data["options"], "recorded_at": now}
                            if timing is not None:
                                row["last_interviewer_seconds"] = timing.get("interviewer_seconds")
                                row["last_respondent_seconds"] = timing.get("respondent_seconds")
                                info["last_interviewer_seconds"] = timing.get("interviewer_seconds")
                                info["last_respondent_seconds"] = timing.get("respondent_seconds")
                            self._save_interview(base, index, document)
                            self._question_path(job_id, index).unlink(missing_ok=True)
                            row["turn_count"] = len(frozen)
                            outcome["checkpoint"] = copy.deepcopy(checkpoint)
                        atomic_json(base / "experiment.json", data)

                def invoke():
                    try:
                        runtime_data = dict(data)
                        runtime_data["_report_progress"] = progress
                        runtime_data["_scenario_snapshot"] = scenario_snapshot
                        runtime_data["_pending_question"] = pending_question
                        runtime_data["_checkpoint"] = checkpoint
                        from playground.common.interruptible_http import model_call_scope
                        with model_call_scope(stop):
                            outcome["result"] = self.run_one(runtime_data, cid, turns, stop)
                    except Exception as exc:
                        outcome["exception"] = exc
                    finally:
                        completed.set()
                        with self.lock:
                            self.inflight.get(job_id, set()).discard(threading.current_thread())

                # The manager owns persistence, not the provider thread. Abort
                # the wait exactly as the single-interview frontend aborts fetch.
                # An uncooperative call may remain in flight, but its captured
                # cancelled Event prevents all later progress/turn commits.
                if self.isolate_calls:
                    target = lambda: self._isolated_invoke(job_id, stop, data, cid, turns,
                                                           checkpoint, scenario_snapshot, pending_question, progress, outcome, completed)
                else:
                    target = invoke
                thread = threading.Thread(target=target, daemon=True,
                                          name="batch-call-" + job_id[:8])
                with self.lock:
                    self.inflight.setdefault(job_id, set()).add(thread)
                thread.start()
                while not completed.wait(0.025):
                    if stop.is_set():
                        break
                with self.lock:
                    if not thread.is_alive():
                        self.inflight.get(job_id, set()).discard(thread)
                # If Continue has started, this obsolete worker must not
                # overwrite the new generation's metadata or transcript.
                with self.lock:
                    if self.stop_events.get(job_id) is not stop:
                        return
                interrupted = stop.is_set()
                if interrupted:
                    # Freeze only durable completed rounds with a matching native checkpoint.
                    # A model response might finish while stop is being set, before its
                    # checkpoint has been committed. Such a turn must not be advertised.
                    durable = self._load_partial(job_id, index)
                    frozen_turns = copy.deepcopy(durable.get("turns", []))
                    outcome["checkpoint"] = durable.get("checkpoint")
                    state, ending, error = "interrupted", None, "Interrupted by researcher"
                elif "exception" in outcome:
                    exc = outcome["exception"]
                    state, ending, error = "failed", None, safe_batch_error(
                        exc, data.get("progress", {}).get("phase"))
                    # The last acknowledged round, not an uncommitted mutable
                    # turns list, is the durable resume boundary.
                    durable = self._load_partial(job_id, index)
                    frozen_turns = copy.deepcopy(durable.get("turns", []))
                    outcome["checkpoint"] = durable.get("checkpoint")
                else:
                    result = outcome["result"]
                    state = result.get("status")
                    ending = result.get("closing_message")
                    error = result.get("error")
                    if state not in ("completed", "incomplete"):
                        state, ending, error = "failed", None, "Invalid interview result state"
                    frozen_turns = copy.deepcopy(turns)
                durable_timings = self._load_partial(job_id, index).get("turn_timings", [])
                document = {"schema_version": "batch_interview_v1",
                            "experiment_id": job_id, "dataset": data["dataset"],
                            "candidate_id": cid, "status": state,
                            "turn_count": len(frozen_turns), "turns": frozen_turns,
                            "turn_timings": copy.deepcopy(durable_timings),
                            "closing_message": ending, "error": error,
                            "options": data["options"], "recorded_at": utc_now()}
                base = self._directory(job_id)
                document["checkpoint"] = copy.deepcopy(outcome.get("checkpoint") or checkpoint)
                stem = f"interview_{index:04d}"
                with self.lock:
                    if self.stop_events.get(job_id) is not stop:
                        return
                    self._save_interview(base, index, document)
                    row.update(status=state, error=error, turn_count=len(frozen_turns),
                               transcript_path=f"transcripts/{stem}.txt",
                               result_path=f"results/{stem}.json")
                    data["updated_at"] = utc_now()
                    data["progress"]["phase"] = "interrupted" if interrupted else "saved"
                    data["progress"]["last_activity_at"] = utc_now()
                    atomic_json(base / "experiment.json", data)
                if interrupted:
                    break
            with self.lock:
                if self.stop_events.get(job_id) is not stop:
                    return
                data["state"] = "interrupted" if stop.is_set() else (
                    "completed" if all(r["status"] in ("completed", "skipped") for r in data["interviews"])
                    else "finished_with_errors")
                data["finished_at"] = utc_now()
                data["updated_at"] = utc_now()
                data["progress"]["phase"] = "interrupted" if stop.is_set() else "finished"
                atomic_json(self._directory(job_id) / "experiment.json", data)
        except Exception as exc:
            # A corrupt stage/checkpoint or an unexpected provider failure can
            # happen before the normal per-participant error path is installed.
            # Never strand the experiment as 'running' with no worker alive.
            with self.lock:
                if self.stop_events.get(job_id) is stop:
                    try:
                        current = self._read(job_id)
                        now = utc_now()
                        phase = (current.get("progress") or {}).get("phase")
                        error = safe_batch_error(exc, phase)
                        current["state"] = "interrupted" if stop.is_set() else "finished_with_errors"
                        rows = current.get("interviews", [])
                        active = [i for i, row in enumerate(rows, 1) if row.get("status") == "running"]
                        if not active:
                            # The worker can fail before it persists row=running
                            # (for example while validating a staged scenario).
                            # In that case the first pending participant is the
                            # one the worker was attempting to start.
                            active = [next((i for i, row in enumerate(rows, 1)
                                            if row.get("status") == "pending"), None)]
                        for index in active:
                            if index is None:
                                continue
                            row = rows[index - 1]
                            saved = self._load_partial(job_id, index)
                            row["status"] = "interrupted" if stop.is_set() else "failed"
                            row["error"] = error
                            row["turn_count"] = len(saved.get("turns", []))
                        current["updated_at"] = now
                        current["finished_at"] = now
                        current["progress"] = {**(current.get("progress") or {}),
                                               "phase": "interrupted" if stop.is_set() else "failed",
                                               "last_activity_at": now, "error": error}
                        atomic_json(self._directory(job_id) / "experiment.json", current)
                    except Exception:
                        # Keep the original worker exception in logs when even
                        # diagnostic persistence cannot be completed.
                        import traceback
                        traceback.print_exc()
        finally:
            with self.lock:
                if self.stop_events.get(job_id) is stop:
                    self.running.discard(job_id)
                    self.stop_events.pop(job_id, None)

    @staticmethod
    def _save_interview(base, index, document):
        (base / "results").mkdir(parents=True, exist_ok=True)
        (base / "transcripts").mkdir(parents=True, exist_ok=True)
        stem = f"interview_{index:04d}"
        atomic_json(base / "results" / (stem + ".json"), document)
        dest = base / "transcripts" / (stem + ".txt")
        temp = dest.with_name(dest.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temp.write_text(transcript_text(document["turns"], document.get("closing_message")), encoding="utf-8")
            temp.replace(dest)
        finally:
            temp.unlink(missing_ok=True)

    def export(self, job_id):
        with self.lock:
            data = self._read(job_id)
            base = self._directory(job_id)
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in [base / "experiment.json", *sorted((base / "results").glob("*.json")),
                             *sorted((base / "transcripts").glob("*.txt"))]:
                    if path.is_file():
                        archive.write(path, path.relative_to(base).as_posix())
            return output.getvalue()
