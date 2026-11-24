"""Participant profile projection for Interview Playground v0.9.3.

Interviewers receive only a short, non-sensitive pre-interview briefing;
respondents receive attributed source-linked profile cues and explicitly
human-verified identity attributes. Nothing here infers personality from
age/gender/occupation or promotes raw profile fields to verified facts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata

PUBLIC_FIELDS = (
    "occupation", "role", "professional_experience", "industry",
    "ecosystem", "framework", "organization_type",
)
RESPONDENT_FIELDS = PUBLIC_FIELDS + ("broad_region", "language_preference", "education")
FIELD_ALIASES = {
    "occupation": {"occupation", "job title", "job_title", "job role", "profession", "职业", "岗位", "职务"},
    "role": {"role", "role type", "role_type", "role description", "role_description", "participant role", "职业角色", "角色", "研究角色"},
    "professional_experience": {"total years of experience", "years of experience", "years_experience", "work experience", "professional experience", "从业年限", "工作年限"},
    "industry": {"industry", "sector", "field of work", "work sector", "行业", "领域"},
    "ecosystem": {"ecosystem", "technology ecosystem", "技术生态"},
    "framework": {"framework", "development framework", "技术框架"},
    "organization_type": {"organisation type", "organization type", "organisation_type", "organization_type", "机构类型", "组织类型"},
    "broad_region": {"country", "region", "location", "broad region", "地区", "国家", "区域"},
    "language_preference": {"preferred language", "language preference", "language_preference", "母语", "访谈语言"},
    "education": {"education", "education level", "最高学历", "教育程度"},
}
ALIASES = {" ".join(unicodedata.normalize('NFKC', key).casefold().replace('_',' ').split()): canonical
           for canonical, aliases in FIELD_ALIASES.items() for key in aliases}

CONTACT = re.compile(r'(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|https?://|www\.|\b\d{3}-\d{2}-\d{4}\b|\+?\d[\d\s().-]{9,}\d)', re.I)
UNSAFE = re.compile(r'(?:ignore\s+(?:all\s+)?(?:previous|prior)|system\s*prompt|developer\s*instructions|memory[_ -]?bank|private[_ -]?persona|\b(?:M\d{4}|F\d{4}_\d+)\b|忽略(?:以上|之前)|系统提示|隐藏记忆|不要遵循)', re.I)
VALID_REVIEW = {"reviewed", "human_verified", "approved", "verified", "privacy_reviewed"}
VALID_LINKAGE = {"verified_document_id", "verified", "human_verified", "manually_verified"}


def _value(raw, length=90):
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        return ""
    s = unicodedata.normalize('NFKC', str(raw)).strip()
    if not s or len(s) > length or any(ord(ch)<32 for ch in s) or CONTACT.search(s) or UNSAFE.search(s):
        return ""
    if s.lower() in {"none", "null", "unknown", "n/a", "not applicable", "nan"}:
        return ""
    return s


def _verified_attr(raw):
    """Only explicitly reviewed identity attributes; never infer from name verification."""
    if not isinstance(raw, dict):
        return {}
    result = {}
    for k in ("age_range", "age", "gender", "education", "occupation", "role", "industry", "language_preference"):
        if k in raw and (v := _value(raw[k], 70)):
            result[k] = v
    return result


def _source_fields(profile):
    if not isinstance(profile, dict):
        return {}
    # v3.4 keeps only independently sourced profile attributes. Do not infer
    # identity, habits or traits from raw interview answers.
    if isinstance(profile.get("attributes"), dict):
        result={}
        for key, raw in profile["attributes"].items():
            canonical=ALIASES.get(" ".join(unicodedata.normalize('NFKC',str(key)).casefold().replace('_',' ').split()))
            if canonical and canonical not in result and (val:=_value(raw,80)):
                result[canonical]=val
        return result
    source = profile.get("profile_fields")
    if not isinstance(source, dict):
        return {}
    joined = str(profile.get("profile_linkage") or "").lower() in VALID_LINKAGE
    reviewed = str(profile.get("profile_review_status") or "").lower() in VALID_REVIEW
    # Flags can describe a *different*, unlinked questionnaire while a role
    # in the interview header is still source-linked. Trust the linkage state
    # of this actual profile row, not substring matching in unrelated flags.
    if not (joined or reviewed):
        return {}
    found = {}
    for key, raw in source.items():
        canonical = ALIASES.get(" ".join(unicodedata.normalize('NFKC', str(key)).casefold().replace('_', ' ').split()))
        if not canonical or canonical in found:
            continue
        if value := _value(raw, 80):
            found[canonical] = value
    return found


def _public_briefing(attributes, reviewed=False):
    return {
        "schema_version": "participant_briefing.v1",
        "visibility": "pre_interview_non_sensitive",
        "review_status": "human_reviewed" if reviewed else "source_linked_unreviewed",
        "attributes": [{"field": f, "value": attributes[f],
                        "evidence_status": "human_reviewed" if reviewed else "source_linked_unreviewed"}
                       for f in PUBLIC_FIELDS if f in attributes][:7],
    }


def build_profile_views(profile):
    profile = profile if isinstance(profile, dict) else {}
    source_fields = _source_fields(profile)
    reviewed = str(profile.get("profile_review_status") or "").lower() in VALID_REVIEW
    # Explicit reviewed attributes may be provided by a researcher; raw 'identity'
    # does not count as verified just because a name happens to be verified.
    verified = _verified_attr(profile.get("verified_identity_attributes"))
    if reviewed:
        verified.update(_verified_attr(profile.get("identity")))
    # Verified, explicitly reviewed professional details can also be part of
    # the non-sensitive pre-interview briefing when raw profile_fields is empty.
    for key in PUBLIC_FIELDS:
        if key in verified and key not in source_fields:
            source_fields[key] = verified[key]
    respondent_fields = [{"field": f, "value": source_fields[f],
                          "evidence_status": "human_reviewed" if reviewed else "source_linked_unreviewed"}
                         for f in RESPONDENT_FIELDS if f in source_fields]
    public = _public_briefing(source_fields, reviewed)
    respondent = {
        "verified_identity_attributes": verified,
        "source_linked_profile_attributes": respondent_fields[:10],
        "profile_review_status": "human_reviewed" if reviewed else "requires_human_review",
        "profile_construction": "source_fields_only_no_inferred_personality",
    }
    digest = hashlib.sha256(json.dumps({"public":public,"respondent":respondent},
                                   ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
    return public, respondent, digest


def sanitize_public_briefing(raw):
    """Hard allowlist for researcher-edited public role briefing.

    Private memory, full biography, identity, free-form context and arbitrary
    attribute names can never cross into the interviewer's model prompt.
    """
    if not isinstance(raw, dict):
        return _public_briefing({})
    reviewed = raw.get("review_status") == "human_reviewed"
    fields = {}
    items = raw.get("attributes")
    if isinstance(items, list):
        for row in items[:20]:
            if not isinstance(row, dict):
                continue
            key = row.get("field")
            if key in PUBLIC_FIELDS and key not in fields and (val := _value(row.get("value"), 80)):
                fields[key] = val
    return _public_briefing(fields, reviewed)


def interviewer_public_view(briefing):
    """Minimized view; never use unbounded requester-supplied profile objects."""
    b = sanitize_public_briefing(briefing)
    return {"availability": "limited_source_profile" if b["attributes"] else "no_pre_interview_profile",
            "review_status": b["review_status"], "attributes": copy.deepcopy(b["attributes"])}


def respondent_prompt_view(persona):
    """Compact full respondent *permitted* context; no hidden memory bank."""
    if not isinstance(persona, dict):
        return {"verified_identity_attributes":{}, "source_linked_profile_attributes":[]}
    verified = _verified_attr(persona.get("verified_identity_attributes"))
    source = []
    items = persona.get("source_linked_profile_attributes")
    if isinstance(items, list):
        for row in items[:15]:
            if not isinstance(row, dict):
                continue
            field = row.get("field")
            if field in RESPONDENT_FIELDS and (value := _value(row.get("value"), 80)):
                source.append({"field":field, "value":value,
                               "evidence_status":"human_reviewed" if row.get("evidence_status")=="human_reviewed"
                               else "source_linked_unreviewed"})
    return {"verified_identity_attributes":verified, "source_linked_profile_attributes":source[:10],
            "review_status":"human_reviewed" if persona.get("review_status")=="human_reviewed"
            else "requires_human_review"}
