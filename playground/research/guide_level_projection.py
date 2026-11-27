"""Apply an explicitly selected, dataset/candidate-matched graded guide.

Only public interviewer questions are replaced. The respondent's evidence and
private memory remain untouched. Original guide provenance is not upgraded.
"""
import copy
from playground.research.graded_guides import match, question_view
from playground.research.guide_schema import with_validation_snapshot, validate_guide
from playground.research.outline_levels import has_refined_guide, build_native_outline
from playground.interview_agent.outline.models import Outline


def apply_level(scenario, level, category=None):
    dataset = (scenario.get("provenance") or {}).get("dataset")
    candidate = (scenario.get("provenance") or {}).get("candidate_id")
    row = match(dataset, candidate, category)
    if row:
        topics = question_view(row, level)
        question_items = [(q["text"], []) for t in topics for q in t["questions"]]
        title = topics[0]["title"]
        origin = "graded_guides"
    elif has_refined_guide(dataset):
        native = Outline.from_dict(build_native_outline(dataset, level))
        question_items = [(q.text, list(q.probes)) for q in native.questions()]
        title = native.title
        origin = "outline_levels"
    else:
        scenario["provenance"]["graded_guide_status"] = "unavailable_original_retained"
        return False
    questions = [text for text, _ in question_items]
    original = scenario["public"]["interview_guide"]
    result = copy.deepcopy(original)
    topic = copy.deepcopy(original["topics"][0])
    template = copy.deepcopy(topic["subtopics"][0])
    topic["topic_id"] = "T01"
    topic["title"] = title[:200]
    topic["subtopics"] = []
    for i, (question, probes) in enumerate(question_items[:30], 1):
        sub = copy.deepcopy(template)
        sub["subtopic_id"] = "T01-S%02d" % i
        sub["label"] = question[:300]
        sub["suggested_probes"] = [question, *probes]
        sub["source_interviewer_turn_indices"] = []
        sub["review_status"] = "requires_human_review"
        topic["subtopics"].append(sub)
    result["topics"] = [topic]
    result["review_status"] = "needs_review"
    result["graded_guide_source"] = {
        "dataset": dataset, "guide_id": row.get("guide_id") if row else None,
        "origin": origin,
        "level": level, "category": row.get("category") if row else None,
        "review_required": True
    }
    result = with_validation_snapshot(result)
    quality = validate_guide(result, normalize=False)
    if not quality["ok"]:
        raise ValueError("Graded guide projection failed: " + quality["errors"][0]["message"])
    scenario["public"]["interview_guide"] = result
    scenario["public"]["plan_questions"] = questions[:30]
    scenario["public"]["topics"] = [topic["title"]]
    scenario["provenance"]["graded_guide_status"] = "matched"
    return True


def attach_all_levels(scenario, category=None):
    """Bind three guide variants to ONE scene, without repeating source/evidence preparation.

    Missing or ambiguous matches are reported (not replaced with another
    interview's guide). Variants are private server-side scenario metadata.
    """
    from playground.research.graded_guides import LEVELS
    stored={}
    for level in LEVELS:
        candidate=copy.deepcopy(scenario)
        if apply_level(candidate, level, category):
            stored[level]={
                "interview_guide":candidate["public"]["interview_guide"],
                "plan_questions":candidate["public"]["plan_questions"],
                "topics":candidate["public"]["topics"],
            }
    scenario["_attached_levels"]=stored
    scenario["provenance"]["available_guide_levels"]=list(stored)
    scenario["provenance"]["guide_level"]="level1" if "level1" in stored else None
    scenario["provenance"]["guide_category"]=category
    if "level1" in stored:
        select_attached_level(scenario,"level1")
    return stored


def select_attached_level(scenario, level):
    """Select a prebound variant in constant time; no source rebuild or LLM call."""
    if level not in ("level0","level1","level2"):
        raise ValueError("Invalid outline level")
    variants=scenario.get("_attached_levels") or {}
    if level not in variants:
        raise ValueError("此场景未匹配该等级大纲，无法选择；请检查数据集和分类大纲")
    projected=variants[level]
    for key,value in projected.items():
        scenario["public"][key]=copy.deepcopy(value)
    scenario["provenance"]["guide_level"]=level
    scenario["provenance"]["graded_guide_status"]="matched"
    return scenario


def bind_batch_level(scenario, level, category=None):
    """Bind the chosen guide and its availability for a fresh batch scenario.

    Unlike an interactive scenario, a batch interview needs only one level.
    Do not claim availability when the guide has no exact match: the native
    preparation layer will retain its existing no-cross-interview fallback.
    """
    matched = apply_level(scenario, level, category)
    if matched:
        provenance = scenario["provenance"]
        provenance["available_guide_levels"] = [level]
        provenance["guide_level"] = level
        provenance["guide_category"] = category
    return matched
