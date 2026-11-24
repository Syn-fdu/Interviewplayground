"""Create in-memory, evidence-bound *restricted local previews* from InterviewData V2.

A scenario is NOT a privacy/license approval, not a trained persona, and not an
independent benchmark. Its answers are short source excerpts, with turn-level
provenance. No raw data is written to the project or distributed with it.
"""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from playground.research.data_intake import IntakeError, candidate_id, discover_sources

PREVIEW_DATASETS = ()  # Deprecated: supported datasets are discovered from the uploaded package
STOPWORDS = set("""about above after again against almost also among and another any are because been being between both but can could did does doing each even every first for from had has have having her here him his how into its just know like little many more most much need not now one only other our out over really said say she should since some still such than that the their them there these they thing think this those through time too under very want was were what when where which while who why will with would you your yours tell talk about please could would thank thanks""".split())



def scrub(text, name=None, max_length=650):
    """Best-effort redaction only; NOT a guarantee of anonymization."""
    text = str(text or "")
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"https?://\S+", "[LINK]", text)
    text = re.sub(r"(?<!\d)(?:\+?1[.\- ]?)?\(?\d{3}\)?[.\- ]?\d{3}[.\- ]?\d{4}(?!\d)", "[PHONE]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ID]", text)
    if isinstance(name, str) and 3 <= len(name.strip()) <= 80 and name.lower() not in ("unknown", "none", "anonymous", "not specified"):
        text = re.sub(re.escape(name.strip()), "[PERSON]", text, flags=re.IGNORECASE)
    return text.strip()[:max_length]


def tokens(text):
    words = re.findall(r"[a-z]{3,}", str(text).lower())
    result = [w for w in words if w not in STOPWORDS]
    # Native Chinese and non-Latin scripts: retain adjacent CJK bigrams.
    for seq in re.findall(r"[\u3400-\u9fff]{2,}", str(text)):
        result.extend(seq[n:n+2] for n in range(len(seq) - 1))
    return result


def _load_rows(z, dataset, kind, dataset_index=None):
    """Read files from the validated discovery index, independent of dataset name."""
    info = (dataset_index or discover_sources(z.namelist()).get(dataset))
    if not info or kind not in info.get("files", {}):
        raise IntakeError("该数据集未提供 %s.json。" % kind)
    path = info["files"][kind]
    d = json.loads(z.read(path))
    if not isinstance(d, dict) or not isinstance(d.get("records"), list):
        raise IntakeError("V2 记录结构不是预期的 records 数组。")
    return d["records"]

def _get_by_id(rows, source_id):
    for row in rows:
        if isinstance(row, dict) and str(row.get("dialogue_id")) == source_id:
            return row
    raise IntakeError("关联的三类 JSON 无法通过 dialogue_id 找到该记录。")


def _question_answer_pairs(record, name):
    turns = record.get("turns", [])
    matches = []
    last_question = None
    last_question_idx = None
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        role, text = turn.get("role"), str(turn.get("text") or "").strip()
        if role == "interviewer" and len(text) >= 15:
            last_question = text
            last_question_idx = int(turn.get("turn_index", idx))
        elif role == "respondent" and last_question and 65 <= len(text) <= 6000:
            a = scrub(text, name, 550)
            q = scrub(last_question, name, 420)
            if len(tokens(q)) >= 2 and len(a) >= 65:
                matches.append({
                    "original_question": q,
                    "answer": a,
                    "original_answer_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "interviewer_turn_index": last_question_idx,
                    "respondent_turn_index": int(turn.get("turn_index", idx)),
                })
            last_question = None
            last_question_idx = None
    # A second respondent segment without an intervening interviewer isn't a new Q/A pair.
    return matches


def _best_evidence(pairs, max_cards=8):
    results, fingerprints = [], set()
    for pair in pairs:
        akey = pair["original_answer_sha256"]
        if akey in fingerprints:
            continue
        fingerprints.add(akey)
        # Keep evidence in chronological order; source wording is not reinterpreted.
        results.append(pair)
        if max_cards is not None and len(results) >= max_cards:
            break
    return results


def _top_topics(evidence, topic):
    counts = Counter(w for p in evidence for w in set(tokens(p["original_question"]))
                     if len(w) >= 4 and not w.startswith("["))
    top = [k for k, _ in counts.most_common(5)]
    if topic and topic not in top:
        top.insert(0, topic)
    return top[:6]


def build_from_v2(zip_source, candidate, fingerprint, dataset_index=None, dataset_label=None):
    """Build a single in-memory scenario from a *verified* catalog candidate."""
    if not candidate or not candidate.get("structural_ok"):
        raise IntakeError("该记录没有通过结构初筛，不能创建 Scenario 1。")
    if candidate.get("build_mode") == "guide_only":
        return build_guide_only(zip_source, candidate, fingerprint, dataset_index, dataset_label)
    ds = candidate["dataset"]
    if (not candidate.get("scenario_eligible", candidate.get("structural_ok")) or
            any(f in candidate["flags"] for f in ("cross_dataset_duplicate", "empty_turns", "missing_verified_roles", "unsupported_record_unit", "multiple_respondents"))):
        raise IntakeError("此数据集或记录不适合自动构建真实场景，请先完成角色/重复/敏感内容审核。")
    with zipfile.ZipFile(io.BytesIO(zip_source) if isinstance(zip_source, (bytes, bytearray)) else zip_source) as z:
        record = None
        for row in _load_rows(z, ds, "interview_records", dataset_index):
            if isinstance(row, dict) and candidate_id(ds, row.get("dialogue_id")) == candidate["candidate_id"]:
                record = row
                break
        if record is None:
            raise IntakeError("V2 中找不到所选候选对应的原始访谈记录。")
        id_ = str(record["dialogue_id"])
        info = dataset_index or discover_sources(z.namelist()).get(ds)
        profile = next((x for x in _load_rows(z, ds, "respondent_profiles", info)
                        if str(x.get("dialogue_id")) == id_), None) if "respondent_profiles" in info["files"] else None
        outline = next((x for x in _load_rows(z, ds, "interview_outlines", info)
                        if str(x.get("dialogue_id")) == id_), None) if "interview_outlines" in info["files"] else None
        # The selected prebuilt bank is the exclusive respondent evidence input.
        from playground.research.prebuilt_memorybank import read_selected
        prebuilt = read_selected(z, info, ds, record)
    if profile and record.get("profile_id") and profile.get("profile_id") != record["profile_id"]:
        raise IntakeError("人物画像 profile_id 与所选访谈不匹配；拒绝借用其他人物画像。")
    if outline and record.get("outline_id") and outline.get("outline_id") != record["outline_id"]:
        raise IntakeError("采访大纲 outline_id 与所选访谈不匹配。")
    has_profile = bool(profile and candidate.get("build_mode") == "full_profile")
    profile = profile if has_profile else {}
    outline = outline or {}
    if len(profile.get("respondent_speakers") or []) > 1:
        raise IntakeError("News 访谈涉及多位受访者，未经角色复核不能自动模拟为单个人物。")
    identity = profile.get("identity") or {}
    name = identity.get("name_candidate") if isinstance(identity, dict) else None
    if prebuilt is not None:
        from playground.research.prebuilt_memorybank import materialize
        memories, memory_stats = materialize(prebuilt, record, name)
        pairs = [{"original_question":m["source_question"],"answer":m["text"],
                  "original_answer_sha256":m["source"]["original_answer_sha256"],
                  "interviewer_turn_index":m["source"]["interviewer_turn_index"],
                  "respondent_turn_index":m["source"]["respondent_turn_index"]}
                 for m in memories]
    else:
        pairs = _best_evidence(_question_answer_pairs(record, name), None)
    if len(pairs) < 4:
        raise IntakeError("该访谈可追溯的有效问答对少于 4 组，无法建立 Scenario 1。")
    language = record.get("language", "en")
    episode = record.get("episode")
    episode_title = episode.get("title") if isinstance(episode, dict) else (episode if isinstance(episode, str) else None)
    topic = (outline.get("source_topic_label") or
             next((str(t.get("title")) for t in (outline.get("topics") or [])
                   if isinstance(t,dict) and t.get("title")),None)
             or episode_title or "Open interview")
    topic = scrub(topic, name, 90)
    topics = _top_topics(pairs, topic)
    if language == "en":
        goal = f"Explore the participant's experience and views about {topic}; ask open questions and seek concrete details."
        background = "You are interviewing an anonymized participant. No personal identity or hidden answers are available before the interview."
        opening = "I'm happy to talk about my experience. What would you like to know?"
        fallback = "I can't answer that from the available interview record. Could you ask about a more specific aspect of this topic?"
        repeated = "I can repeat what I said earlier: "
    else:
        goal = f"围绕「{topic}」了解受访者的经历和观点，并通过非诱导性追问获得具体内容。"
        background = "受访者身份已隐藏。访谈员只能看到主题和研究任务，不能提前看到原始回答。"
        opening = "我愿意介绍自己的经历，你想先了解哪一方面？"
        fallback = "原始访谈资料没有能够支持这个问题的回答，能否问一个更具体的问题？"
        repeated = "此前已谈到："
    facts = []
    for n, item in enumerate(pairs, start=1):
        facts.append({
            "fact_id": f"E{n:02d}",
            "label": f"Source question {n}: {item['original_question'][:90]}",
            "source_question": item["original_question"],
            "answer": item["answer"],
            "triggers": list(dict.fromkeys(tokens(item["original_question"])))[:30],
            "source": {
                "interviewer_turn_index": item["interviewer_turn_index"],
                "respondent_turn_index": item["respondent_turn_index"],
                "original_answer_sha256": item["original_answer_sha256"],
            },
        })
    # v0.7: Source-grounded guide / entire-available-respondent memory / persona.
    # Loaded lazily to keep the V2 readers independent of the research builders.
    from playground.research.research_scenario import (make_interview_guide, build_memory_bank,
                                   build_persona, build_evaluation_reference)
    guide = make_interview_guide(outline, record, topic, name)
    source_questions = guide.pop("source_questions_for_local_refinement", [])
    # Interviewer public prompts must come from the selected normalized Outline,
    # not from the raw Record used to construct respondent memory.
    public_questions=[str(q.get("text") if isinstance(q,dict) else q).strip()
                      for t in outline.get("topics") or [] if isinstance(t,dict)
                      for q in t.get("questions") or []]
    public_questions=[q for q in dict.fromkeys(public_questions) if q][:180]
    if prebuilt is None:
        memories, memory_stats = build_memory_bank(record, name)
        memory_stats["memory_source"]="legacy_records"
    if len(memories) < 4:
        raise IntakeError("该访谈可使用的受访者记忆少于四条，无法建立可信场景。")
    persona = build_persona(profile, memories, name)
    # Role-specific, source-labelled projections: the interviewer sees only a
    # minimal pre-interview briefing, never a respondent-private biography.
    from playground.research.participant_profile import build_profile_views
    interviewer_briefing, respondent_context, profile_digest = build_profile_views(profile)
    persona.update(respondent_context)
    evaluation_reference = build_evaluation_reference(outline, memories, guide)
    cid = candidate["candidate_id"]
    sid = "V2_SCENARIO_1_" + cid[-12:]
    return {
        "construction_mode": "full_profile" if has_profile else "record_only",
        "allowed_respondent_modes": ["human", "local_llm", "rule"],
        "scenario_id": sid, "title": "Scenario 1 · " + (dataset_label or ds.replace("_", " ").title())[:70] + " · " + topic[:46],
        "domain": str(topic), "source_type": "v2_restricted_local", "language": language,
        "usage": {
            "privacy_review": "pending", "license_review": "pending", "reviewed": False,
            "local_restricted_preview": True, "local_preview_confirmed": True,
            "external_model_allowed": False, "redistribution_allowed": False,
            "note": "真实 V2 数据的本机受限预览；不是隐私脱敏/授权通过，也不允许云端调用。",
        },
        "provenance": {
            "candidate_id": cid, "dataset": ds, "source_fingerprint_sha256": fingerprint,
            "dialogue_id_sha256": hashlib.sha256(id_.encode("utf-8")).hexdigest(),
            "profile_join": has_profile, "outline_join": bool(outline), "source_type": "prebuilt_memorybank" if prebuilt is not None else "V2 record; interviewer-question/next-respondent-turn pairs",
            "evidence_count": len(facts), "planner_question_count": len(pairs), "automatic_redaction_is_incomplete": True,
            "original_outline_provided": bool(outline.get("original_interview_outline_provided")),
            "guide_source_kind": guide["source_kind"],
            "guide_review_status": guide["review_status"],
            "memory_stats": memory_stats,
            "scenario_schema_version": "2.0",
            "profile_projection_version": "participant_profile.v0.9.3",
            "profile_projection_sha256": profile_digest,
            "plan_warning": "Preprocessed interviewer uses normalized Outline only; reconstructed sources require leakage review.",
        },
        "public": {
            "goal": guide["goal"] or goal, "background": background,
            "topics": [t["title"] for t in guide["topics"][:6]] or topics,
            "opening": public_questions[0] if public_questions else "",
            "plan_questions": public_questions,
            "interview_guide": guide,
            "participant_briefing": interviewer_briefing,
        },
        "private": {
            "opening_statement": opening,
            "generic_response": fallback,
            "fallback_response": fallback,
            "repeated_prefix": repeated,
            "facts": facts,
            "memory_bank": memories,  # never sent to interviewer or browser in V2 restricted mode
            "respondent_persona": persona,
            "evaluation_reference": evaluation_reference,  # server-only; never to interviewer
            "source_interviewer_questions": source_questions,  # local-only optional guide refinement
        },
    }



def build_guide_only(zip_source, candidate, fingerprint, dataset_index=None, dataset_label=None):
    """Guide-only -> human interview. No respondent model, memory or invented transcript."""
    from playground.research.guide_schema import with_validation_snapshot, validate_guide
    ds = candidate["dataset"]
    with zipfile.ZipFile(io.BytesIO(zip_source) if isinstance(zip_source, (bytes, bytearray)) else zip_source) as z:
        info = dataset_index or discover_sources(z.namelist()).get(ds)
        outlines = _load_rows(z, ds, "interview_outlines", info)
        outline = next((o for o in outlines if candidate_id(ds, o.get("dialogue_id") or o.get("outline_id")) ==
                        candidate["candidate_id"]), None)
    if outline is None:
        raise IntakeError("没有找到对应来源大纲，不会生成虚构大纲。")
    topic = scrub(next((str(t.get("title")) for t in outline.get("topics") or []
                         if isinstance(t,dict) and t.get("title")),None)
                  or ds.replace("_", " "), max_length=130)
    raw_goals = outline.get("goal") or outline.get("interview_goal_original") or []
    if isinstance(raw_goals, str):
        raw_goals = [raw_goals]
    targets = outline.get("target_information_points_original") or []
    if isinstance(targets, str):
        targets = [targets]
    previews = outline.get("original_guide_question_preview") or []
    questions = []
    for t in outline.get("topics") or []:
        if isinstance(t,dict):
            for q in t.get("questions") or []:
                text=q.get("text") if isinstance(q,dict) else q
                if isinstance(text,str) and text.strip():
                    questions.append(scrub(text,max_length=340))
    for q in (previews if not questions else []):
        raw = q.get("source_text") if isinstance(q, dict) else q
        if isinstance(raw, str) and raw.strip():
            questions.append(scrub(raw, max_length=340))
    for q in targets:
        if isinstance(q, str) and q.strip() and not questions:
            questions.append(scrub(q, max_length=340))
    if not questions:
        steps = outline.get("reconstructed_steps") or []
        if isinstance(steps, list):
            for step in steps:
                q = (step.get("verbatim_interviewer_text") or step.get("question") or step.get("text")) if isinstance(step, dict) else step
                if isinstance(q, str) and q.strip():
                    questions.append(scrub(q, max_length=340))
                    if len(questions) >= 12:
                        break
    if not questions:
        doc = outline.get("original_guide_document") or {}
        source = doc.get("verbatim_text", "") if isinstance(doc, dict) else ""
        questions = [scrub(s, max_length=340) for s in str(source).splitlines()
                     if len(s.strip()) >= 15 and ("?" in s or "？" in s)][:12]
    questions = [q for q in questions if q][:12]
    if not questions:
        raise IntakeError("大纲没有可追溯的具体研究问题，无法创建真人访谈。")
    goal = scrub(raw_goals[0], max_length=400) if raw_goals and isinstance(raw_goals[0],str) else topic
    topics = []
    for i, q in enumerate(questions[:8], 1):
        topics.append({
            "topic_id": f"T{i:02d}", "title": q[:90],
            "source_kind": "original_guide_provided" if outline.get("original_interview_outline_provided") else "source_guide_unverified",
            "subtopics": [{
                "subtopic_id": f"T{i:02d}-S01",
                "label": q[:110],
                "suggested_probes": [q, "能否举一个具体例子？", "还有哪些相关经历值得补充？"],
                "source_interviewer_turn_indices": [],
            }],
        })
    guide = with_validation_snapshot({
        "goal": goal,
        "research_questions": [{"id":"RQ1", "text":goal, "review_status":"requires_human_review"}],
        "topics": topics,
        "source_kind":"original_guide_provided" if outline.get("original_interview_outline_provided") else "source_guide_unverified",
        "source_label":"来源大纲（提取结果待人工复核）",
        "review_status":"needs_review",
        "provenance_note":"只提供采访任务；未有历史真人问答，禁止模拟受访者。",
        "phases":[],
    })
    validation = validate_guide(guide, normalize=False)
    if not validation["ok"]:
        raise IntakeError("来源大纲未通过结构校验：" + validation["errors"][0]["message"])
    sid = "V2_GUIDE_" + candidate["candidate_id"][-12:]
    lang = str(outline.get("language") or "")
    if lang not in ("zh", "en"):
        chinese = len(re.findall(r"[\u3400-\u9fff]", " ".join(questions[:3])))
        lang = "zh" if chinese >= 5 else "en"
    return {
        "scenario_id":sid, "construction_mode":"guide_only", "allowed_respondent_modes":["human"],
        "title":"真人采访 · " + (dataset_label or ds)[:70],
        "domain":topic, "source_type":"v2_restricted_local", "language":lang,
        "usage":{"privacy_review":"pending","license_review":"pending","reviewed":False,
                 "local_restricted_preview":True,"local_preview_confirmed":True,
                 "external_model_allowed":False,"redistribution_allowed":False},
        "provenance":{"candidate_id":candidate["candidate_id"],"dataset":ds,
                      "source_fingerprint_sha256":fingerprint,"profile_join":False,
                      "outline_join":True,"record_available":False,
                      "scenario_schema_version":"2.1","guide_source_kind":guide["source_kind"],
                      "guide_review_status":"needs_review"},
        "public":{"goal":goal,"background":"真人受访者，访前画像未知。请根据受访者实际回答追问。",
                  "topics":[t["title"] for t in topics],
                  "opening":questions[0],"plan_questions":questions,
                  "interview_guide":guide,
                  "participant_briefing":{"schema_version":"participant_briefing.v1",
                                          "visibility":"pre_interview_non_sensitive",
                                          "review_status":"not_provided","attributes":[]}},
        "private":{"facts":[],"memory_bank":[],"respondent_persona":{},
                   "evaluation_reference":{},"source_interviewer_questions":[]},
    }


def choose_for_scenario1(catalog, selected=(), dataset=None):
    """Choose a verified-shape source from any discovered dataset."""
    available = {d["id"] for d in catalog.get("datasets", []) if d.get("scenario_eligible_count", 0) > 0}
    if not available:
        raise IntakeError("当前数据包没有足够问答证据可构建 Scenario 的访谈。")
    if dataset is None:
        dataset = sorted(available)[0]
    if dataset not in available:
        raise IntakeError("当前数据集中没有可构建的访谈，请查看数据质量与候选记录。")
    possible = [x for x in catalog["candidates"] if x["scenario_eligible"] and x["dataset"] == dataset]
    by_id = {x["candidate_id"]: x for x in possible}
    if selected:
        preferred = [by_id[x] for x in selected if x in by_id]
        if preferred:
            possible = preferred
    possible.sort(key=lambda c: (abs(c["turn_count"] - 20), -c["respondent_turns"], c["candidate_id"]))
    if not possible:
        raise IntakeError("没有可用访谈记录。")
    return possible[:30]
