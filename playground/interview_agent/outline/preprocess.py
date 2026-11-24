"""PDF interview guide -> structured Outline.

Step 1: extract text (PyMuPDF, falling back to pypdf).
Step 2: structure it with the LLM (topics / questions / goals / probes). If no LLM is given
        or the LLM output is unusable, a heuristic parser is used.
The resulting Outline is cached as JSON so the planner reads a clean, ID-addressable view.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..llm.base import LLMClient
from .models import Outline

_STRUCTURE_PROMPT = """You convert an interview guide (raw text extracted from a PDF) into a clean JSON structure.
Rules:
- Keep the guide's own wording of main questions; do not invent new main questions. The "text" must be ONLY
  the question the interviewer asks: move a "目的/Purpose" sentence into "intent" and drop numbering/labels such
  as "A1. 日常接触" from "text".
- Group questions into topics following the guide's sections (use one topic if there are none).
- Drop page numbers, headers/footers and layout noise. Put participant background, consent or tone
  instructions for the interviewer into "notes".
- For every question add "goal": one sentence on what information it should elicit (infer if absent),
  "intent": why the guide asks it, and "probes": suggested follow-up angles (only those in the guide, or at
  most 2 short inferred ones).
- Split each question's information needs into slots, derived from THIS question's wording (never generic):
  "required_slots": the minimum information that makes the question answered (usually 1-3; one per thing the
  question literally asks for). "optional_deepening_slots": useful deeper information (e.g. concrete episode,
  reasons, effects, personal judgement) with "priority" low|medium|high. Never put deepening items in required.
  Each slot: {"slot_id": short_snake_case_english_id, "description": str}.
- If the guide has skip/branch instructions for a question, add "branch_rules": [{"condition": the guide's wording,
  "skip": [question ids], "goto": question id}]. Question ids are "T<topic number>Q<question number>" counted in
  order within the guide (e.g. T1Q2). Omit the key when there are none; never invent rules.
- Write goal/intent/probes/descriptions in the same language as the guide.
Return ONLY JSON:
{"title": str, "objective": str, "notes": str,
 "topics": [{"title": str, "questions": [{"text": str, "goal": str, "intent": str, "probes": [str],
   "required_slots": [{"slot_id": str, "description": str}],
   "optional_deepening_slots": [{"slot_id": str, "description": str, "priority": str}],
   "branch_rules": [{"condition": str, "skip": [str], "goto": str}]}]}]}"""

_SLOT_PROMPT = """You are given main questions from an interview guide. For EACH question derive its information slots
from that question's own wording (never generic). Return ONLY JSON:
{"questions": [{"id": str, "goal": str, "intent": str,
  "required_slots": [{"slot_id": snake_case_english, "description": str, "priority": "high"}],
  "optional_deepening_slots": [{"slot_id": snake_case_english, "description": str, "priority": "low|medium|high"}]}]}
Rules:
- required_slots: the minimum information that makes the question answered, one slot per thing the question
  literally asks for (e.g. "which tools do you use and for what?" -> tools_used, primary_uses). Usually 1-3.
  Never a single slot that merely repeats the whole question when it asks for several things.
- optional_deepening_slots: useful deeper information (frequency, a concrete episode, reasons, effects,
  personal judgement ...) relevant to THIS question; 2-4 items. They are never mandatory.
- goal: one sentence; intent: why the guide asks it. Write descriptions in the same language as the question.
- Keep the given ids unchanged and return every question."""

_PURPOSE = re.compile(r"^\s*目的\s*[:：]\s*(.+?[。.；;])\s*")
_LABEL = re.compile(r"^\s*[A-Za-z]{1,2}\d+[\.．、:：]?\s*[^\s？?。.]{1,12}\s+(?=\S)")


def clean_question(text: str) -> dict:
    """Split guide noise off a raw question: a leading '目的：…。' becomes intent and a label such as
    'A1. 日常接触' is dropped, leaving the sentence the interviewer actually asks."""
    intent = ""
    m = _PURPOSE.match(text)
    if m:
        intent, text = m.group(1).strip().rstrip("。.；;"), text[m.end():]
    text = _LABEL.sub("", text, count=1).strip()
    return {"text": text, "intent": intent, "goal": intent}


def _needs_slots(q) -> bool:
    """Required fallback means not yet semantically prepared.

    Optional slots alone are not evidence that the required question coverage
    has been derived. The user's native InterviewAgent PDF defines these as
    separate information-collection targets.
    """
    return not q.required_slots or all(s.source == "fallback" for s in q.required_slots)


def enrich_slots(outline: Outline, llm: LLMClient, batch: int = 8) -> int:
    """Native Slot Agent: batch derivation, validate per question, recover omissions.

    The source question text and native IDs remain unchanged. Incomplete model
    outputs never become a successful prepared outline. Authentication,
    permissions and provider configuration errors are fatal: asking smaller
    batches cannot fix HTTP 401.
    """
    if batch < 1:
        raise ValueError("Slot preprocessing batch must be positive.")
    todo = [q.id for q in outline.questions() if _needs_slots(q)]
    done = 0

    def request(group):
        payload=[{"id":q.id,"text":q.text,"section":q.section,"intent":q.intent}
                 for q in group]
        data=llm.chat_json([
            {"role":"system","content":_SLOT_PROMPT},
            {"role":"user","content":json.dumps({"questions":payload},ensure_ascii=False)},
        ],temperature=0.0)
        if not isinstance(data,dict) or not isinstance(data.get("questions"),list):
            raise ValueError("Preprocess model omitted the questions array")
        found={}
        for item in data["questions"]:
            if isinstance(item,dict) and isinstance(item.get("id"),str):
                found[item["id"]]=item
        return found

    def merge(q, entry):
        if not isinstance(entry,dict):
            return False
        required=entry.get("required_slots")
        if not isinstance(required,list) or not required:
            return False
        if any(not isinstance(item,dict) or
               not isinstance(item.get("description"),str) or
               not item["description"].strip() for item in required):
            return False
        optional=entry.get("optional_deepening_slots")
        if optional is None:
            optional=[{"id":item.id,"description":item.description,
                       "priority":item.priority,"source":item.source}
                      for item in q.optional_deepening_slots]
        original_rules=q.branch_rules
        original_probes=q.probes
        candidate=Outline._question({
            "text":q.text,
            "goal":entry.get("goal") or q.goal,
            "intent":entry.get("intent") or q.intent,
            "probes":original_probes,
            "required_slots":required,
            "optional_deepening_slots":optional
        },q.id,q.topic_id,q.section)
        if _needs_slots(candidate):
            return False
        candidate.branch_rules=original_rules
        for topic in outline.topics:
            if topic.id==q.topic_id:
                for index,current in enumerate(topic.questions):
                    if current.id==q.id:
                        topic.questions[index]=candidate
                        return True
        return False

    def fatal(exc):
        # Explicitly fail fast on permissions/configuration errors, without
        # treating network/API issues as a reason to fabricate Slots.
        from playground.interview_agent.llm.openai_compat import RemoteAPIError
        return isinstance(exc,RemoteAPIError) and (
            exc.status in (400,401,403,404,422) or
            exc.kind in ("authentication_or_permission",
                         "model_or_request_configuration"))

    # Use batches for throughput; valid items from partial responses are saved.
    for pos in range(0,len(todo),batch):
        group=[outline.get(qid) for qid in todo[pos:pos+batch]]
        group=[q for q in group if q is not None]
        try:
            by_id=request(group)
        except Exception as exc:
            if fatal(exc):
                raise
            by_id={}
            print("[preprocess] batch response unavailable; retrying missing questions individually ("
                  +type(exc).__name__+")")
        for q in group:
            if merge(q,by_id.get(q.id)):
                done+=1

    # Repair only unresolved/omitted questions, once each. A persistent
    # transient error is reported by the final strict coverage check.
    for qid in todo:
        q=outline.get(qid)
        if q is None or not _needs_slots(q):
            continue
        try:
            by_id=request([q])
        except Exception as exc:
            if fatal(exc):
                raise
            print("[preprocess] unresolved question "+qid+" ("+type(exc).__name__+")")
            continue
        if merge(q,by_id.get(q.id)):
            done+=1
    return done


def extract_pdf_text(path: str | Path) -> str:
    path = str(path)
    try:
        import pymupdf as fitz  # PyMuPDF
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("Install PyMuPDF or pypdf to read PDFs") from e
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


_Q_MARK = re.compile(r"[?？]\s*$")
_NUM = re.compile(r"^\s*(?:Q?\d+[\.\)、:：]|[（(]\d+[)）]|[-•·*])\s*")
_HEADING = re.compile(r"^\s*(?:(?:part|section|topic|theme)\s*\d+|第[一二三四五六七八九十\d]+[部分章节]|[一二三四五六七八九十]+[、.．]|\d+[\.．]\d*\s)", re.I)


def heuristic_outline(text: str) -> Outline:
    """Fallback: headings start topics; lines (or wrapped line groups) ending in ?/？ are questions."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else "Interview"
    topics, current, buf = [], None, ""

    def new_topic(name):
        t = {"title": name, "questions": []}
        topics.append(t)
        return t

    for line in lines[1:]:
        if _HEADING.match(line) and not _Q_MARK.search(line):
            current, buf = new_topic(line), ""
            continue
        buf = (buf + " " + _NUM.sub("", line)).strip() if buf else _NUM.sub("", line)
        if _Q_MARK.search(line):
            if current is None:
                current = new_topic("Main topics")
            current["questions"].append(clean_question(buf))
            buf = ""
        elif _NUM.match(line) is None and len(buf) > 200:
            buf = ""
    return Outline.from_dict({"title": title, "topics": topics})


def build_outline(pdf_path: str | Path, llm: Optional[LLMClient] = None,
                  cache_path: Optional[str | Path] = None) -> Outline:
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        raise ValueError("No extractable text in PDF (scanned PDFs need OCR first).")
    outline: Optional[Outline] = None
    if llm is not None:
        try:
            data = llm.chat_json([
                {"role": "system", "content": _STRUCTURE_PROMPT},
                {"role": "user", "content": text[:30000]},
            ], temperature=0.0)
            outline = Outline.from_dict(data)
        except Exception as e:  # noqa: BLE001 - fall back on any model/parse failure
            print(f"[preprocess] LLM structuring failed ({e}); using heuristic parser.")
    if not outline or not outline.questions():
        outline = heuristic_outline(text)
    if not outline.questions():
        raise ValueError("Could not find any interview questions in the PDF.")
    if llm is not None:
        enriched = enrich_slots(outline, llm)
        if enriched:
            print(f"[preprocess] derived slots with the LLM for {enriched} question(s).")
    generic = [q.id for q in outline.questions() if _needs_slots(q)]
    if generic:
        print(f"[preprocess] WARNING: {len(generic)} question(s) only have a generic 'main_answer' slot "
              f"({', '.join(generic[:8])}{'...' if len(generic) > 8 else ''}); follow-ups cannot target specific information.")
    if cache_path:
        Path(cache_path).write_text(json.dumps(outline.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return outline
