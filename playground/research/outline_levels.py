"""Dataset-local, lazily selected three-level interview guides.

Each bundled dataset owns exactly one outline_levels.json with Level 0/1/2.
Unbundled draft guides are not installed or silently substituted.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import copy
import json
from playground.interview_agent.outline.models import Outline

LEVELS = ("level0", "level1", "level2")
BUNDLED_ROOT = Path(__file__).resolve().parents[2] / "resources" / "bundled_data"


def normalize_level(level):
    if level not in LEVELS:
        raise ValueError("大纲等级必须为 level0、level1 或 level2")
    return level


def _paths():
    """Find installed dataset-local guide paths, without reading other guides."""
    paths = {}
    for path in sorted(BUNDLED_ROOT.glob("*/social_research/*/processed/outline_levels.json")):
        dataset = path.parent.parent.name
        if dataset in paths:
            raise ValueError(f"数据集 {dataset} 的三级大纲有多个候选文件")
        paths[dataset] = path
    return paths


@lru_cache(maxsize=32)
def _load_one(dataset, path_string, mtime_ns):
    # Cache identity includes filesystem modification time, so updates refresh.
    document = json.loads(Path(path_string).read_text(encoding="utf-8"))
    if document.get("dataset_id") != dataset:
        raise ValueError(f"数据集 {dataset} 的大纲标识不匹配")
    levels = document.get("levels")
    if not isinstance(levels, dict) or set(levels) != set(LEVELS):
        raise ValueError(f"数据集 {dataset} 必须恰好包含三个等级")
    foundation_ids = {q["id"] for q in levels["level1"]}
    if not 1 <= len(levels["level0"]) <= 6 or not 5 <= len(levels["level1"]) <= 15 or not 5 <= len(levels["level2"]) <= 10:
        raise ValueError(f"数据集 {dataset} 的三级问题数量不符合要求")
    if any(q.get("parent_id") not in foundation_ids for q in levels["level2"]):
        raise ValueError(f"数据集 {dataset} 有深入问题未关联基础问题")
    if any(q.get("id") not in foundation_ids for item in levels["level0"] for q in
           ({"id": ref} for ref in item.get("covers_level1", []))):
        raise ValueError(f"数据集 {dataset} 的粗略问题引用无效")
    covered = {ref for item in levels["level0"] for ref in item.get("covers_level1", [])}
    if covered != foundation_ids:
        raise ValueError(f"数据集 {dataset} 的粗略大纲未覆盖全部基础问题")
    return document


def get_guide(dataset):
    path = _paths().get(dataset)
    if not path:
        raise ValueError(f"数据集 {dataset} 没有内置三级精修大纲")
    return copy.deepcopy(_load_one(dataset, str(path), path.stat().st_mtime_ns))


def catalog():
    """Compatibility interface. Do not use it in the interview execution path."""
    return {dataset: get_guide(dataset)["levels"] for dataset in _paths()}


def has_refined_guide(dataset):
    return dataset in _paths()


def preview(dataset, level):
    level = normalize_level(level)
    document = get_guide(dataset)
    levels = document["levels"]
    return {"ok":True, "dataset":dataset, "level":level,
            "research_focus":document.get("research_focus"),
            "review_status":document.get("review_status"),
            "level_counts":{k:len(v) for k,v in levels.items()},
            "questions":copy.deepcopy(levels[level]),
            "effective_question_count":len(levels["level1"]) if level=="level2" else len(levels[level])}


def build_native_outline(dataset, level):
    level = normalize_level(level)
    document = get_guide(dataset)
    levels = document["levels"]
    if level == "level0":
        topics = [{"id":"L0", "title":"研究目标与核心议题", "questions":[
            {"id":q["id"], "text":q["question_zh"], "goal":q["question_zh"]}
            for q in levels["level0"]]}]
    else:
        probes = {}
        if level == "level2":
            for q in levels["level2"]:
                probes.setdefault(q["parent_id"], []).append(q["question_zh"])
        topics = [{"id":"L1", "title":"研究目标与基础问题", "questions":[
            {"id":q["id"], "text":q["question_zh"], "goal":q["question_zh"],
             "probes":probes.get(q["id"], [])} for q in levels["level1"]]}]
    outline = Outline.from_dict({"title":dataset.replace("_"," ").title(),
        "objective":document.get("research_focus") or dataset,
        "notes":f"Reconstructed candidate guide; bundled dataset: {dataset}; specificity: {level}",
        "topics":topics})
    if not outline.questions():
        raise ValueError("精修大纲没有可用问题")
    return outline.to_dict()


def summary():
    return {"levels":list(LEVELS),"datasets":sorted(_paths()),
            "storage":"dataset_local_bundled_data", "review_status":"research_candidate_not_original"}
