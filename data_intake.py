#!/usr/bin/env python3
"""Schema-driven, metadata-only InterviewData V2 discovery (no dataset allowlist).

Accepts ZIP layouts:
  InterviewData/datasets/<dataset_id>/processed/{manifest,interview_records,...}.json
  datasets/<dataset_id>/processed/...
  <dataset_id>/processed/...
A single enclosing project directory is also accepted. The browser sees ONLY
summary metadata, never transcript, identity, profile, guide, or source paths.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile

MAX_ZIP_BYTES = 400 * 1024 * 1024
MAX_TOTAL_BYTES = 1600 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024 * 1024  # enhanced v3 memory-bearing records can exceed 220 MiB
MAX_FILES = 20_000
DATASET_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
CORE_TYPES = ("interview_records", "respondent_profiles", "interview_outlines")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
KNOWN_AUDIT_FIELDS = ("sensitivity", "sensitive_content", "requires_sensitive_review")
# Only schema/field names and legacy prefixes are supported; dataset names are never hardcoded.

class IntakeError(ValueError):
    """Invalid or unsupported InterviewData package."""


def candidate_id(dataset, source_id):
    digest = sha256((dataset + "::" + str(source_id)).encode("utf-8")).hexdigest()
    return "CAND_" + digest[:20]


def _zip_source(source):
    if isinstance(source, (bytes, bytearray)):
        if len(source) > MAX_ZIP_BYTES:
            raise IntakeError("ZIP 超过 400 MiB 导入限制。")
        return io.BytesIO(source), sha256(source).hexdigest()
    source = Path(source)
    if not source.is_file() or source.stat().st_size > MAX_ZIP_BYTES:
        raise IntakeError("文件不存在，或 ZIP 超过 400 MiB 限制。")
    digest = sha256()
    with source.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(buf)
    return source, digest.hexdigest()


def _validated_members(z):
    found = {}
    total = 0
    infos = z.infolist()
    if len(infos) > MAX_FILES:
        raise IntakeError("ZIP 文件数量过多。")
    for info in infos:
        path = info.filename
        parts = path.split("/")
        if (path.startswith("/") or ".." in parts or "\\" in path or
                info.flag_bits & 1 or re.match(r"^[A-Za-z]:", path)):
            raise IntakeError("ZIP 含非法路径或加密文件。")
        if info.file_size > MAX_MEMBER_BYTES:
            raise IntakeError("ZIP 包含超过单文件上限的文件。")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise IntakeError("ZIP 解压总体积超限。")
        if path in found:
            raise IntakeError("ZIP 内有重复文件路径。")
        found[path] = info
    return found


def _dataset_path(path):
    """Resolve known flat and categorized InterviewData archive layouts.

    Supports uploading a category folder as a standalone ZIP, e.g.
      social_research/<dataset>/processed/manifest.json
      entertainment/<dataset>/processed/manifest.json
    as well as the complete canonical layout:
      InterviewData/datasets/<category>/<dataset>/processed/manifest.json
    and legacy flat datasets/<dataset>/processed/manifest.json.
    An optional single enclosing directory is allowed; unrelated arbitrary
    prefixes remain rejected so two datasets cannot silently be conflated.
    """
    parts=path.split("/")
    if len(parts)<3 or parts[-2]!="processed":
        return None
    dataset=parts[-3]
    if not DATASET_ID_RE.fullmatch(dataset):
        return None
    prefix=parts[:-3]
    categories={"social_research","entertainment","entertainment_interviews"}
    # Explicitly known family of dataset-root prefixes.
    roots=[[],["datasets"],["InterviewData","datasets"]]
    categorized=[]
    for cat in categories:
        categorized.extend([
            [cat],["datasets",cat],["InterviewData","datasets",cat],
            ["InterviewData",cat],
        ])
    allowed_prefixes=roots+categorized
    allowed=prefix in allowed_prefixes
    # A ZIP may have one outer wrapper (user-named extracted directory).
    if not allowed and len(prefix)>1 and DATASET_ID_RE.fullmatch(prefix[0]):
        allowed=prefix[1:] in allowed_prefixes
    # Retain historical multi-level taxonomy under InterviewData/datasets.
    if not allowed and len(prefix) in (4,5) and prefix[:2]==["InterviewData","datasets"]:
        allowed=(prefix[2] in categories and
                 all(DATASET_ID_RE.fullmatch(x) for x in prefix[3:]))
    # Bundled release ZIP has a fixed, versioned two-level container wrapper.
    # Do not accept arbitrary unrelated directory trees.
    if not allowed and len(prefix)==4 and prefix[0:2]==["InterviewPlayground_MVP","bundled_data"]:
        allowed=(prefix[2].startswith("InterviewData_") and prefix[3] in categories)
    if not allowed:
        return None
    return dataset,"/".join(parts[:-1])+"/"

def discover_sources(paths):
    """Discover dataset roots using ZIP member names, independent of their IDs.

    This is only a directory scan; content validation happens in build_catalog.
    Reject ambiguous same-ID roots rather than silently mixing data.
    """
    candidates = defaultdict(set)
    for path in paths:
        hit = _dataset_path(path)
        if hit:
            candidates[hit[0]].add(hit[1])
    if not candidates:
        raise IntakeError("未发现数据集目录。支持 social_research/<id>/processed/manifest.json、entertainment/<id>/processed/manifest.json，以及 InterviewData/datasets/<category>/<id>/processed/manifest.json 和旧版扁平结构。")
    result = {}
    for dataset, prefixes in sorted(candidates.items()):
        if len(prefixes) != 1:
            raise IntakeError("数据集 %s 出现在多个目录中，无法唯一识别。" % dataset)
        root = next(iter(prefixes))
        result[dataset] = {
            "root": root,
            "manifest": root + "manifest.json",
            "files": {},
        }
        for kind in CORE_TYPES + ("memorybank",):
            alternatives = [root + kind + ".json", root + "tv_" + kind + ".json"]
            existing = [p for p in alternatives if p in paths]
            if len(existing) > 1:
                raise IntakeError("数据集 %s 同时存在两个 %s 文件。" % (dataset, kind))
            if existing:
                result[dataset]["files"][kind] = existing[0]
    return result


def _read_json(z, path):
    try:
        data = json.loads(z.read(path).decode("utf-8-sig"))
    except (KeyError, ValueError, UnicodeError, RuntimeError, zipfile.BadZipFile) as exc:
        raise IntakeError("JSON 无法解析：" + path.rsplit("/", 1)[-1]) from exc
    if not isinstance(data, dict):
        raise IntakeError("JSON 顶层必须是对象：" + path.rsplit("/", 1)[-1])
    return data


def _rows(z, path):
    rows = _read_json(z, path).get("records")
    if not isinstance(rows, list):
        raise IntakeError("JSON 缺少 records 数组：" + path.rsplit("/", 1)[-1])
    return rows


def _by_dialogue(rows):
    result = {}
    duplicate = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = row.get("dialogue_id")
        if rid is None or not str(rid).strip():
            continue
        key = str(rid)
        if key in result:
            duplicate = True
        result[key] = row
    return result, duplicate


def _basic_pair_and_memory_count(row):
    """Conservative preview of builder evidence thresholds; no private data returned."""
    question = None
    answer_digests = set()
    pair_digests = set()
    count = 0
    for turn in row.get("turns", []):
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("text") or "").strip()
        if turn.get("role") == "interviewer" and len(text) >= 15:
            question = text
        elif turn.get("role") == "respondent":
            if len(text) >= 8:
                answer_digests.add(sha256(text.encode("utf-8")).digest())
            if question and 65 <= len(text) <= 6000:
                q_words = len(re.findall(r"[a-z]{3,}", question.lower()))
                q_cjk = len(re.findall(r"[\u3400-\u9fff]", question))
                if q_words >= 2 or q_cjk >= 3:
                    digest = sha256(text.encode("utf-8")).digest()
                    if digest not in pair_digests:
                        pair_digests.add(digest)
                        count += 1
            question = None
    return count, len(answer_digests)


def _record_metadata(ds, row, pos, profiles, outlines, integrity_ok, sensitive):
    rid = row.get("dialogue_id")
    rid = str(rid) if rid is not None and str(rid).strip() else "__missing_%d" % pos
    turns = row.get("turns")
    turns = turns if isinstance(turns, list) else []
    role_counts = Counter(str(t.get("role", "unknown")) for t in turns if isinstance(t, dict))
    flags = []
    profile = profiles.get(rid)
    outline = outlines.get(rid)
    if not profile:
        flags.append("missing_profile_join")
    if not outline:
        flags.append("missing_outline_join")
    if (profile and row.get("profile_id") and profile.get("profile_id") != row.get("profile_id")):
        flags.append("mismatched_profile_id")
    if (outline and row.get("outline_id") and outline.get("outline_id") != row.get("outline_id")):
        flags.append("mismatched_outline_id")
    if not role_counts["interviewer"] or not role_counts["respondent"]:
        flags.append("missing_verified_roles")
    if len(turns) < 4:
        flags.append("short_or_empty_transcript")
    provided_indices = [t.get("turn_index") for t in turns if isinstance(t, dict) and "turn_index" in t]
    if provided_indices and (len(provided_indices) != len(turns) or
                             any(type(n) is not int or n < 0 for n in provided_indices) or
                             provided_indices != sorted(set(provided_indices))):
        flags.append("invalid_turn_indices")
    if any(not isinstance(t, dict) or not str(t.get("text") or "").strip() for t in turns):
        flags.append("empty_turns")
    unit = str(row.get("record_unit") or "interview")
    if unit not in ("interview", "semi_structured_interview", "structured_interview"):
        flags.append("unsupported_record_unit")
    provenance = row.get("source_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    review_flags = provenance.get("data_review_flags")
    review_flags = review_flags if isinstance(review_flags, list) else []
    if sensitive or any("sensitive" in str(f).lower() for f in review_flags):
        flags.append("sensitive_dataset_review")
    if not integrity_ok:
        flags.append("file_integrity_failed")
    if profile and isinstance(profile.get("respondent_speakers"), list) and len(profile["respondent_speakers"]) > 1:
        flags.append("multiple_respondents")
    block = {
        "mismatched_profile_id", "mismatched_outline_id",
        "missing_verified_roles", "short_or_empty_transcript", "empty_turns", "unsupported_record_unit",
        "file_integrity_failed", "multiple_respondents", "invalid_turn_indices"
    }
    structural_ok = not any(f in block for f in flags)
    # Use minimal counts only; no transcript is carried into metadata.
    qa_count, memory_count = _basic_pair_and_memory_count(row)
    if qa_count < 4:
        flags.append("insufficient_qa_evidence")
    if memory_count < 4:
        flags.append("insufficient_memory")
    scenario_eligible = structural_ok and qa_count >= 4 and memory_count >= 4
    # Participant code is an identifier, not an independently sourced profile.
    has_persona = bool(profile and isinstance(profile.get("attributes"), dict)
                       and profile.get("attributes"))
    build_mode = "full_profile" if has_persona else "record_only"
    return {
        "candidate_id": candidate_id(ds, rid), "dataset": ds, "record_unit": unit,
        "build_mode": build_mode, "allowed_respondent_modes": ["human", "local_llm", "rule"],
        "turn_count": len(turns), "interviewer_turns": role_counts["interviewer"],
        "respondent_turns": role_counts["respondent"],
        "join_complete": bool(profile and outline), "structural_ok": structural_ok,
        "scenario_eligible": scenario_eligible,
        "evidence_pair_count": qa_count, "memory_candidate_count": memory_count,
        "review_status": "pending", "playground_ready": False, "flags": flags,
    }


def _transcript_digest(row):
    turns = row.get("turns") or []
    if not turns:
        return None
    h = sha256()
    for t in turns:
        if not isinstance(t, dict):
            continue
        h.update(str(t.get("role") or "").encode("utf-8"))
        h.update(b"\0")
        h.update(str(t.get("text") or "").strip().encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def build_catalog(source):
    """Validate arbitrary named datasets; only metadata can leave this module."""
    src, fingerprint = _zip_source(source)
    try:
        z = zipfile.ZipFile(src)
    except (zipfile.BadZipFile, OSError) as exc:
        raise IntakeError("不是有效的 ZIP 文件。") from exc
    with z:
        paths = _validated_members(z)
        index = discover_sources(paths)
        datasets, candidates, validation = [], [], []
        transcript_seen = {}
        for ds, info in index.items():
            manifest_path = info["manifest"]
            if manifest_path not in paths:
                datasets.append({"id": ds, "label": ds.replace("_", " ").title(),
                                 "count": 0, "structurally_usable": 0, "scenario_eligible_count": 0,
                                 "metadata_count": 0, "integrity_ok": False,
                                 "core_three_json_complete": False, "status": "incomplete_or_failed",
                                 "issue": "缺少 manifest.json", "caveat": "资料不完整", "approvals": {}})
                validation.append({"dataset": ds, "code": "manifest_missing"})
                continue
            manifest = _read_json(z, manifest_path)
            id_valid = manifest.get("dataset_id") in (None, ds)
            release = str(manifest.get("dataset_release_version") or "")
            version_ok = ((str(manifest.get("manifest_version")) == "2.0" and
                          re.fullmatch(r"v2\.(?:0|[1-9][0-9]*)(?:[_-][A-Za-z0-9_-]+)?", release) is not None)
                       or (str(manifest.get("manifest_version")) in ("3.3","3.4") and
                           release.startswith("v3.")))
            outputs = manifest.get("output_sha256")
            mismatches = []
            if not isinstance(outputs, dict) or not outputs:
                outputs = {}
                mismatches.append("missing_output_hashes")
            for name, expected in outputs.items():
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.json", name):
                    mismatches.append("invalid_output_reference")
                    continue
                member = info["root"] + name
                if member not in paths:
                    mismatches.append(name + "_missing")
                elif not isinstance(expected, str) or not SHA256_RE.fullmatch(expected) or sha256(z.read(member)).hexdigest().lower() != expected.lower():
                    mismatches.append(name + "_sha256_mismatch")
            for kind, member in info["files"].items():
                if member.rsplit("/", 1)[-1] not in outputs:
                    mismatches.append(kind + "_hash_missing")
            if not id_valid:
                validation.append({"dataset": ds, "code": "dataset_id_mismatch"})
            if not version_ok:
                validation.append({"dataset": ds, "code": "version_mismatch"})
            if mismatches:
                validation.append({"dataset": ds, "code": "integrity_failed", "files": mismatches[:12]})
            integrity_ok = version_ok and id_valid and not mismatches
            # Any real transcript can be a scene source; a standalone guide can
            # launch a new human interview. The persona file is optional.
            source_complete = ("interview_records" in info["files"] or
                               "interview_outlines" in info["files"])
            full_three_files_present = all(kind in info["files"] for kind in CORE_TYPES)
            if not source_complete:
                validation.append({"dataset": ds, "code": "core_three_json_missing"})
            ds_candidates = []
            metadata_count = 0
            if source_complete:
                records = (_rows(z, info["files"]["interview_records"])
                           if "interview_records" in info["files"] else [])
                profiles, p_dup = _by_dialogue(_rows(z, info["files"]["respondent_profiles"])
                           if "respondent_profiles" in info["files"] else [])
                outlines, o_dup = _by_dialogue(_rows(z, info["files"]["interview_outlines"])
                           if "interview_outlines" in info["files"] else [])
                if p_dup or o_dup:
                    validation.append({"dataset": ds, "code": "duplicate_dialogue_join"})
                    integrity_ok = False
                sensitive = (bool(manifest.get("sensitive_content")) or bool(manifest.get("requires_sensitive_review"))
                             or str(manifest.get("sensitivity") or "").lower() in ("high", "sensitive", "restricted_sensitive"))
                for pos, row in enumerate(records):
                    if not isinstance(row, dict):
                        validation.append({"dataset": ds, "code": "invalid_record_row", "position": pos})
                        continue
                    item = _record_metadata(ds, row, pos, profiles, outlines, integrity_ok, sensitive)
                    if "memorybank" in info["files"]:
                        item["memory_source"]="memorybank.json"
                        item["scenario_eligible"]=bool(item["structural_ok"] and item["respondent_turns"]>=4)
                        item["flags"]=[x for x in item["flags"] if x not in ("insufficient_qa_evidence","insufficient_memory")]
                    ds_candidates.append(item)
                    digest = _transcript_digest(row)
                    if digest and item["structural_ok"]:
                        transcript_seen.setdefault(digest, []).append(item)
                if p_dup or o_dup:
                    for c in ds_candidates:
                        c["structural_ok"] = c["scenario_eligible"] = False
                        if "duplicate_dialogue_join" not in c["flags"]:
                            c["flags"].append("duplicate_dialogue_join")
                # A guide without any source transcript is still usable to
                # interview a NEW human; no invented respondent or transcript.
                if not records:
                    for pos, outline in enumerate(outlines.values()):
                        source_id = str(outline.get("dialogue_id") or outline.get("outline_id") or pos)
                        usable_guide = bool(
                            any(t.get("questions") for t in (outline.get("topics") or []) if isinstance(t,dict)) or
                            outline.get("original_guide_question_preview") or
                            outline.get("target_information_points_original") or
                            outline.get("reconstructed_steps") or
                            outline.get("interview_goal_original"))
                        if not usable_guide:
                            continue
                        ds_candidates.append({
                            "candidate_id": candidate_id(ds, source_id),
                            "dataset": ds, "record_unit": "guide_only",
                            "build_mode": "guide_only",
                            "allowed_respondent_modes": ["human"],
                            "turn_count": 0, "interviewer_turns": 0, "respondent_turns": 0,
                            "join_complete": False, "structural_ok": bool(integrity_ok),
                            "scenario_eligible": bool(integrity_ok),
                            "evidence_pair_count": 0, "memory_candidate_count": 0,
                            "review_status": "pending", "playground_ready": False,
                            "flags": ["guide_only_human_interview"],
                        })
                candidates.extend(ds_candidates)
            else:
                # Metadata-only data can still be inventoried; never silently treated as Q/A.
                for p in paths:
                    if p.startswith(info["root"]) and p.endswith("conversation_metadata.json"):
                        metadata_count = len(_read_json(z, p).get("records", []))
                        break
            lbl = str(manifest.get("dataset_label") or manifest.get("title") or ds.replace("_", " ").title()).strip()
            # Do not allow unbounded text or markup in an untrusted manifest label.
            lbl = lbl[:90] or ds
            usable = sum(c["structural_ok"] for c in ds_candidates)
            eligible = sum(c["scenario_eligible"] for c in ds_candidates)
            datasets.append({
                "id": ds, "label": lbl, "count": len(ds_candidates),
                "structurally_usable": usable, "scenario_eligible_count": eligible,
                "metadata_count": metadata_count,
                "record_count": len(records) if source_complete else 0,
                "guide_only_count": sum(c.get("build_mode")=="guide_only" for c in ds_candidates),
                "construction_modes": sorted(set(c.get("build_mode", "record_only") for c in ds_candidates)),
                "integrity_ok": integrity_ok, "core_three_json_complete": full_three_files_present,
                "scene_source_present": source_complete,
                "memorybank_present": "memorybank" in info["files"],
                "memorybank_semantic_review_complete": manifest.get("memorybank_semantic_review_complete") is True,
                "status": "indexed_pending_review" if source_complete and integrity_ok else "incomplete_or_failed",
                "issue": "本机预览尚未完成人工审核" if source_complete and integrity_ok else "三类文件或哈希校验未通过",
                "caveat": "可构建 %d / %d 份" % (eligible, len(ds_candidates)) if source_complete else "缺少完整问答",
                "approvals": {
                    "manual_review": manifest.get("manual_review_completed") is True,
                    "privacy_review": manifest.get("privacy_review_completed") is True,
                    "license_and_privacy_review": manifest.get("privacy_and_license_review_completed") is True,
                    "playground_export": manifest.get("approved_for_playground_exports") is True,
                },
                "v2_release": manifest.get("dataset_release_version"),
                "requires_sensitive_review": any("sensitive_dataset_review" in c["flags"] for c in ds_candidates),
            })
        # Block cross-dataset identical transcripts, without dataset-specific matching rules.
        for matches in transcript_seen.values():
            if len({m["dataset"] for m in matches}) > 1:
                for m in matches:
                    m["structural_ok"] = m["scenario_eligible"] = False
                    if "cross_dataset_duplicate" not in m["flags"]:
                        m["flags"].append("cross_dataset_duplicate")
        for d in datasets:
            members = [c for c in candidates if c["dataset"] == d["id"]]
            d["structurally_usable"] = sum(c["structural_ok"] for c in members)
            d["scenario_eligible_count"] = sum(c["scenario_eligible"] for c in members)
            if d["core_three_json_complete"]:
                d["caveat"] = ("敏感资料待复核 · " if d.get("requires_sensitive_review") else "") + ("可构建 %d / %d 份" % (d["scenario_eligible_count"], d["count"]))
        ids = [c["candidate_id"] for c in candidates]
        if len(ids) != len(set(ids)):
            raise IntakeError("重复候选 ID 或 dialogue_id；请检查源数据。")
        return {
            "source_fingerprint": fingerprint, "source_type": "InterviewData_v2_zip",
            "datasets": datasets, "candidates": candidates, "validation": validation,
            "total_candidates": len(candidates),
            "total_metadata_only": sum(d["metadata_count"] for d in datasets),
            "total_structurally_usable": sum(d["structurally_usable"] for d in datasets),
            "total_scenario_eligible": sum(d["scenario_eligible_count"] for d in datasets),
            "privacy_mode": "metadata_only_pending_review",
            "source_index": index,  # SERVER-ONLY; never included in catalog_summary or export
        }


def catalog_summary(catalog, selected=()):
    if not catalog:
        return {"loaded": False, "datasets": [], "selected_count": 0,
                "total_scenario_eligible": 0, "message": "尚未导入 InterviewData；请上传 ZIP。"}
    return {k: catalog[k] for k in (
        "source_fingerprint", "datasets", "validation", "total_candidates",
        "total_metadata_only", "total_structurally_usable", "total_scenario_eligible", "privacy_mode"
    )} | {"loaded": True, "selected_count": len(selected),
            "message": "已自动发现数据集；完整性通过不代表隐私或许可审核通过。"}


def auto_pick(catalog, selected, limit=10, datasets=None, replace=False):
    if not 1 <= limit <= 200:
        raise IntakeError("自动选择数量必须在 1 到 200 之间。")
    all_names = {d["id"] for d in catalog["datasets"]}
    choices = set(datasets) if datasets is not None else {d["id"] for d in catalog["datasets"] if d["scenario_eligible_count"]}
    if not choices <= all_names:
        raise IntakeError("数据集不存在于当前导入的 ZIP。")
    already = set() if replace else set(selected)
    ranked = [c for c in catalog["candidates"] if c["dataset"] in choices
              and c["structural_ok"] and "sensitive_dataset_review" not in c["flags"]
              and c["candidate_id"] not in already]
    pools = defaultdict(list)
    for item in ranked:
        pools[item["dataset"]].append(item)
    for pool in pools.values():
        pool.sort(key=lambda item: (not item["scenario_eligible"], -min(item["turn_count"], 50), item["candidate_id"]))
    chosen, order = [], sorted(pools)
    while len(chosen) < limit and any(pools[d] for d in order):
        for ds in order:
            if pools[ds] and len(chosen) < limit:
                chosen.append(pools[ds].pop(0)["candidate_id"])
    return ([] if replace else list(selected)) + chosen, chosen


def selection_manifest(catalog, selected):
    idmap = {x["candidate_id"]: x for x in catalog["candidates"]}
    rows = [idmap[x] for x in selected if x in idmap]
    return {
        "format": "interview_playground_v2_intake_selection", "format_version": "0.3",
        "source_fingerprint_sha256": catalog["source_fingerprint"],
        "status": "candidate_selection_only_not_approved_for_experiment",
        "record_count": len(rows), "records": rows, "validation": catalog["validation"],
        "privacy_release": "no_transcript_or_profile_content_included",
    }


def pack_data_directory(source_dir):
    """Package processed JSON from local datasets directory into an in-memory ZIP.

    Supports --data-dir <.../datasets>, <.../InterviewData>, <.../single_dataset>,
    or a containing folder with InterviewData/datasets/. Ignores raw and media files.
    Does not persist extracted data, does not follow symlinks.
    """
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise IntakeError("--data-dir 必须指向本机的文件夹。")
    candidates = [source_dir / "InterviewData" / "datasets", source_dir / "datasets", source_dir]
    root = next((p for p in candidates if p.is_dir() and any(q.is_dir() and (q / "processed").is_dir() for q in p.iterdir())), None)
    if root is None and (source_dir / "processed").is_dir():
        root = source_dir.parent
        only_name = source_dir.name
    else:
        only_name = None
    if root is None:
        raise IntakeError("找不到 <dataset_id>/processed/ 结构。")
    buf = io.BytesIO()
    total, count = 0, 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for ds in sorted(p for p in root.iterdir() if p.is_dir() and (p / "processed").is_dir()):
            if only_name and ds.name != only_name:
                continue
            if not DATASET_ID_RE.fullmatch(ds.name):
                raise IntakeError("文件夹名必须为小写英文字母、数字或下划线，以字母开头。")
            pdir = ds / "processed"
            for file in sorted(pdir.glob("*.json")):
                if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(source_dir):
                    raise IntakeError("发现非法链接或路径。")
                size = file.stat().st_size
                if size > MAX_MEMBER_BYTES:
                    raise IntakeError("单个 JSON 文件过大。")
                total += size
                count += 1
                if total > MAX_TOTAL_BYTES or count > MAX_FILES:
                    raise IntakeError("本机目录过大。")
                z.write(file, arcname=f"datasets/{ds.name}/processed/{file.name}")
    if not count or buf.tell() > MAX_ZIP_BYTES:
        raise IntakeError("找不到可导入的 JSON，或规范化 ZIP 超过 400 MiB。")
    return buf.getvalue()
