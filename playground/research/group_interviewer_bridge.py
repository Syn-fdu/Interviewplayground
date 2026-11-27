"""Native group Interviewer adapter. One session owns one Interviewer and one Outline.
No private respondent memory is passed to the interviewer.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, fields, is_dataclass
from typing import get_type_hints, get_origin, get_args, Union
import types
from dataclasses import replace
import io
import json
import os
import re
import threading
import zipfile
from playground.interview_agent.interviewer import Interviewer
from playground.interview_agent.outline.models import Outline
from playground.interview_agent.llm.openai_compat import OpenAICompatClient
from playground.research.data_intake import candidate_id

_LOCK = threading.RLock()
_SESSIONS = {}
_OUTLINES = {}
_ROLES = ("planner","reflection","questioning","followup","response","disruptive")

class GroupBridgeError(ValueError):
    pass


def _decode_dataclass(typ, value):
    """Restore validated native dataclass trees from our own JSON checkpoint."""
    origin = get_origin(typ)
    args = get_args(typ)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        return _decode_dataclass(next(t for t in args if t is not type(None)), value)
    if origin is list:
        if not isinstance(value, list):
            raise GroupBridgeError("Invalid checkpoint list")
        return [_decode_dataclass(args[0], x) for x in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise GroupBridgeError("Invalid checkpoint mapping")
        return {k: _decode_dataclass(args[1], v) for k, v in value.items()}
    if is_dataclass(typ):
        if not isinstance(value, dict):
            raise GroupBridgeError("Invalid checkpoint dataclass")
        hints = get_type_hints(typ)
        return typ(**{f.name:_decode_dataclass(hints[f.name], value[f.name]) for f in fields(typ)
                      if f.name in value})
    if typ in (str, int, float, bool) and not isinstance(value, typ):
        raise GroupBridgeError("Checkpoint field has unexpected type")
    return value


def export_checkpoint(provider, session_id, scenario_id, fingerprint, revision):
    """Capture bridge + native interviewer state immediately after one respondent answer."""
    key = (provider, session_id, scenario_id, fingerprint, revision)
    with _LOCK:
        state = _SESSIONS.get(key)
    if state is None:
        raise GroupBridgeError("Cannot checkpoint a missing interviewer session")
    with state["lock"]:
        agent = state["agent"].state
        return {"format": "native_interviewer_checkpoint_v1", "provider": provider,
                "fingerprint": fingerprint, "revision": revision,
                "language": state["language"], "history": copy.deepcopy(state["history"]),
                "history_length": state["history_length"], "last_result": copy.deepcopy(state["last_result"]),
                "native": {"outline": agent.outline.to_dict(),
                           "questions": {k:asdict(v) for k,v in agent.qstate.items()},
                           "turns": [asdict(x) for x in agent.turns],
                           "evidence_bank": [asdict(x) for x in agent.evidence_bank],
                           "disruption": asdict(agent.disruption),
                           "active_qid": agent.active_qid, "finished": agent.finished}}


def _restore_checkpoint(bridge, checkpoint, history, provider, fingerprint, revision, lang):
    from playground.interview_agent.state import (QuestionState, Turn, Disruption)
    from playground.interview_agent.slots import EvidenceItem
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != "native_interviewer_checkpoint_v1"
            or checkpoint.get("provider") != provider or checkpoint.get("fingerprint") != fingerprint
            or checkpoint.get("revision") != revision or checkpoint.get("language") != lang):
        raise GroupBridgeError("Native checkpoint metadata does not match current interview")
    prior = checkpoint.get("history_length")
    past = checkpoint.get("history")
    if not isinstance(prior, int) or prior < 0 or not isinstance(past, list):
        raise GroupBridgeError("Invalid checkpoint history")
    if len(history) != prior + 1 or history[:-1] != past:
        raise GroupBridgeError("Saved transcript does not match interviewer checkpoint")
    native = checkpoint.get("native") or {}
    previous_turns = native.get("turns", [])
    if len(previous_turns) != len(history):
        raise GroupBridgeError("Native turn count does not match saved transcript")
    for row, saved in zip(previous_turns, history):
        if row.get("utterance") != saved.get("question"):
            raise GroupBridgeError("Question mismatch in saved native state")
    agent = bridge["agent"].state
    if set(native.get("questions", {})) != set(agent.qstate):
        raise GroupBridgeError("Outline question set differs from saved checkpoint")
    agent.qstate = {k:_decode_dataclass(QuestionState,v) for k,v in native["questions"].items()}
    agent.turns = [_decode_dataclass(Turn,v) for v in previous_turns]
    agent.evidence_bank = [_decode_dataclass(EvidenceItem,v) for v in native.get("evidence_bank", [])]
    agent.disruption = _decode_dataclass(Disruption,native["disruption"])
    agent.active_qid = native.get("active_qid")
    agent.finished = native.get("finished", False)
    if agent.finished:
        raise GroupBridgeError("Cannot continue an already finished interviewer")
    bridge["history"] = copy.deepcopy(past)
    bridge["history_length"] = prior
    bridge["last_result"] = copy.deepcopy(checkpoint.get("last_result"))
    bridge["finished"] = False


def _find_outline(source_bytes, dataset, candidate, source_fp, revision, level="level1"):
    key=(source_fp,dataset,candidate,revision,level)
    with _LOCK:
        if key in _OUTLINES:
            return copy.deepcopy(_OUTLINES[key])
    # The original Sidecar is deliberately not used. Preparation is mandatory.
    raise GroupBridgeError("采访者大纲尚未预处理，请先执行采访者准备阶段。")

def remote_configuration(provider="shubi",role="interviewer"):
    from playground.common.model_services import get
    try:return get(provider,role)
    except ValueError as exc:raise GroupBridgeError(str(exc)) from exc

def remote_status():
    from playground.common.model_services import status
    return status()["interviewer"]

def _clients_for(provider):
    if provider not in ("maas","shubi","local"):
        raise GroupBridgeError("Unknown interviewer provider.")
    cfg=remote_configuration(provider)
    llms={role:OpenAICompatClient(cfg,retries=3) for role in _ROLES}
    llms["default"]=llms["planner"]
    return llms

# The native Interviewer exposes `state.finished`, but some providers emit a
# standalone farewell before the planner marks the outline completed.
# A conservative fallback handles such closings without rerouting native decisions.
_CLOSING_CUES = (
    "感谢您今天的", "感谢您的参与", "感谢您接受", "感谢您的分享",
    "感谢您抽出时间", "感谢今天的分享", "访谈到此结束", "采访到此结束",
    "本次访谈就到这里", "我们今天就聊到这里",
)
_FAREWELL_CUES = ("再见", "祝您", "祝你", "保重", "下次再见", "goodbye", "take care", "have a great day")
_ENGLISH_CLOSINGS = (
    "thank you for your time", "thank you for sharing", "thank you for participating",
    "that concludes our interview", "that concludes the interview", "end of the interview",
)

def _explicit_closing_utterance(utterance: str) -> bool:
    """Only recognize a complete interviewer sign-off, never questions or thanks within a probe."""
    text=re.sub(r"\\s+", " ", utterance or "").strip()
    if not text or len(text)>550:
        return False
    # If the interviewer asks a substantive question, do not infer a terminal action.
    if any(mark in text for mark in ("？", "?", "请问", "能否", "能不能", "可以再", "could you", "would you")):
        return False
    low=text.casefold()
    return (
        any(cue in text for cue in _CLOSING_CUES) and any(cue in low for cue in _FAREWELL_CUES)
    ) or (
        any(cue in low for cue in _ENGLISH_CLOSINGS)
        and any(cue in low for cue in ("goodbye", "take care", "wish you", "all the best", "concludes"))
    )

def run_next(req, registered, source_bytes, research_revision, provider="local"):
    """Run one native Interviewer.step() with protocol native_interviewer_v1."""
    if not isinstance(req,dict):
        raise GroupBridgeError("Expected JSON object.")
    if req.get("protocol")!="native_interviewer_v1":
        raise GroupBridgeError("Expected protocol native_interviewer_v1; old interviewer request schemas are removed.")
    allowed={"protocol","provider","scenario_id","session_id","history","language",
             "source_fingerprint_sha256","research_revision","interviewer_timeout_seconds","checkpoint"}
    unknown=set(req)-allowed
    if unknown:
        raise GroupBridgeError("Legacy or unsupported interviewer fields: "+", ".join(sorted(unknown)[:8]))
    if req.get("provider")!=provider:
        raise GroupBridgeError("Interviewer provider mismatch.")
    scenario=req.get("scenario_id")
    session=req.get("session_id")
    history=req.get("history")
    if not isinstance(session,str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}",session):
        raise GroupBridgeError("Invalid or missing session_id.")
    if not isinstance(history,list) or len(history)>10000:
        raise GroupBridgeError("Invalid history.")
    for turn in history:
        if not isinstance(turn,dict) or set(turn)!={"question","answer"}:
            raise GroupBridgeError("History must contain only question and answer.")
        if (not isinstance(turn["question"],str) or not isinstance(turn["answer"],str)
                or not turn["question"].strip() or not turn["answer"].strip()):
            raise GroupBridgeError("History contains an empty question or answer.")
        if len(turn["question"])>12000 or len(turn["answer"])>40000:
            raise GroupBridgeError("History turn exceeds per-turn text limit.")
    if req.get("research_revision")!=research_revision:
        raise GroupBridgeError("Research revision changed; start a new session.")
    provenance=registered.get("provenance") or {}
    fingerprint=provenance.get("source_fingerprint_sha256")
    if req.get("source_fingerprint_sha256")!=fingerprint:
        raise GroupBridgeError("Source fingerprint mismatch.")
    dataset=provenance.get("dataset")
    candidate=provenance.get("candidate_id")
    if not isinstance(dataset,str) or not isinstance(candidate,str):
        raise GroupBridgeError("Missing candidate source metadata.")
    if provider not in ("maas","shubi","local"):
        raise GroupBridgeError("Unknown interviewer provider.")
    key=(provider,session,scenario,fingerprint,research_revision)
    with _LOCK:
        state=_SESSIONS.get(key)
        if state is None:
            # A resumed batch has a durable native checkpoint but a new in-memory
            # session_id. Restore below instead of rejecting any nonempty history.
            restoring = bool(history)
            if restoring and req.get("checkpoint") is None:
                raise GroupBridgeError(
                    "Group interviewer session missing: transcript contains "
                    f"{len(history)} completed turns but no native checkpoint "
                    "was supplied (phase=session_restore). Resume from the "
                    "saved batch result or restart this interview.")
            outline=Outline.from_dict(_find_outline(source_bytes,dataset,candidate,fingerprint,research_revision,provenance.get("guide_level","level1")))
            lang=req.get("language","zh")
            if lang not in ("zh","en","bi"):
                raise GroupBridgeError("Unsupported interview language.")
            # Presentation-language preference only: no changes to the native
            # Planner, Reflection, Behavior Agent skills, policies, or routing.
            language_note={
                "zh":"用中文向受访者提问，保持原生访谈目标与追问策略。",
                "en":"Ask the participant in English, keeping the original interview objectives and follow-up policy.",
                "bi":"可使用中文和英语自然访谈，按照对话情境选择语言，不需要逐句翻译。"
            }[lang]
            if not history:
                outline.notes=(outline.notes or "")+"\nInterview language preference: "+language_note
            llms=_clients_for(provider)
            interviewer=Interviewer(outline,llms,max_turns=1_000_000)
            state={"agent":interviewer,"llms":llms,"history_length":-1,"last_result":None,
                   "language":lang,"history":[],"lock":threading.RLock()}
            if restoring:
                try:
                    _restore_checkpoint(state, req["checkpoint"], history, provider,
                                        fingerprint, research_revision, lang)
                except (GroupBridgeError, KeyError, TypeError, ValueError) as exc:
                    raise GroupBridgeError(
                        f"Native interviewer checkpoint restore failed "
                        f"(phase=session_restore, turns={len(history)}, "
                        f"checkpoint_format={req['checkpoint'].get('format') if isinstance(req['checkpoint'], dict) else 'invalid'}): {exc}") from exc
            elif req.get("checkpoint") is not None:
                raise GroupBridgeError("Cannot restore a checkpoint without transcript")
            _SESSIONS[key]=state
    # One lock per session. A slow remote API call must not block unrelated
    # interviews or other sessions from getting their own Group Agent.
    with state["lock"]:
        if req.get("language","zh")!=state["language"]:
            raise GroupBridgeError("Language cannot change mid-session. Start a new interview.")
        prior=state["history_length"]
        if state.get("finished") and state.get("last_result") is not None:
            # Terminal calls never invoke the native model again, even when the
            # browser accidentally submits another request after a closing.
            return copy.deepcopy(state["last_result"])
        if len(history)==prior and state["last_result"] is not None:
            if history!=state["history"]:
                raise GroupBridgeError("Retry transcript does not match the recorded history.")
            # Browser retry for the previous request: no second inference.
            return copy.deepcopy(state["last_result"])
        if prior==-1:
            if history:raise GroupBridgeError("Initial call must have empty history.")
            reply=None
        elif len(history)==prior+1:
            if history[:-1]!=state["history"]:
                raise GroupBridgeError("Prior transcript changed; start a new session.")
            reply=history[-1].get("answer")
            if not isinstance(reply,str) or not reply.strip():
                raise GroupBridgeError("Missing previous respondent answer.")
        else:
            raise GroupBridgeError("Out-of-sequence interview call; restart session.")
        # Apply timeout to the existing clients in place. Never recreate Interviewer
        # or clear history when the researcher changes the time limit.
        per_call_timeout=req.get("interviewer_timeout_seconds",105)
        try:
            per_call_timeout=int(per_call_timeout)
        except (ValueError,TypeError):
            raise GroupBridgeError("Interviewer timeout must be an integer number of seconds.")
        if not 30<=per_call_timeout<=600:
            raise GroupBridgeError("Interviewer timeout must be between 30 and 600 seconds.")
        seen=set()
        for client in state["llms"].values():
            if id(client) in seen:continue
            seen.add(id(client))
            if hasattr(client,"cfg") and hasattr(client.cfg,"timeout"):
                client.cfg=replace(client.cfg,timeout=float(per_call_timeout))
        checkpoint=copy.deepcopy(state["agent"].state)
        try:
            question=state["agent"].step(reply)
        except Exception:
            # An LLM can fail between Reflection, Planner and Behavior. Retry
            # the same turn from a clean native state, not a half-updated state.
            state["agent"].state=checkpoint
            raise
        if not isinstance(question,str) or not question.strip():
            state["agent"].state=checkpoint
            raise GroupBridgeError("Native interviewer generated no question.")
        agent_state=state["agent"].state
        decision=agent_state.last_turn().decision
        finished_by_native=bool(agent_state.finished)
        finished_by_farewell=(not finished_by_native and
                             _explicit_closing_utterance(question))
        finished=finished_by_native or finished_by_farewell
        if finished_by_farewell:
            # Mirror the completion in the bridge-owned native state, without
            # changing group member Planner/Reflection/Behavior source code.
            agent_state.finish()
        result={
            "protocol":"native_interviewer_v1",
            "question":question,
            "action":"conclude" if finished else str(decision.action),
            "native_action":str(decision.action),
            "topic_id":getattr(decision,"question_id",None),
            "target_key":getattr(decision,"target_slot",None),
            "model_id":state["llms"]["planner"].cfg.model
                if hasattr(state["llms"]["planner"],"cfg") else None,
            "group_agent_state":{"turns":len(agent_state.turns),"finished":finished,
                                 "finish_source":"native_state" if finished_by_native
                                   else "explicit_farewell" if finished_by_farewell else None},
            "finished":finished,
            "closing_message":question if finished else None,
            "source":"native_Interviewer.step",
            "interviewer_provider":provider
        }
        state["history_length"]=len(history)
        state["history"]=copy.deepcopy(history)
        state["last_result"]=copy.deepcopy(result)
        state["finished"]=finished
        return result
