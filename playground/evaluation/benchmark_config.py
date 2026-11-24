"""Validated, declarative Benchmark rule sets for Interview Playground.

The JSON is authoritative for *which* built-in measures run and which supported
parameters they use.  Arbitrary Python/JS formulas and model judges are never
executed from a user-supplied file.  Legacy v0.1 rubric files are accepted by
mapping their documented measure IDs onto the registered implementations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

SCHEMA = "interview_benchmark_config.v1"
CONFIG_VERSION = "1.0"
MAX_METRICS = 23

class BenchmarkConfigError(ValueError):
    pass

# Registered algorithms correspond exactly to functions within
# benchmark_foundation.evaluate.  Textual "formula_or_rule" in the old rubric is
# descriptive and is NOT interpreted as executable code.
REGISTRY = {
    "turns_recorded": ("objective", "auto", "完成轮数"),
    "interview_completion_ratio": ("objective", "auto", "访谈完成比例"),
    "interviewer_question_characters": ("objective", "auto", "采访问题字符数"),
    "participant_response_characters": ("objective", "auto", "受访者回答字符数"),
    "exact_duplicate_question_count": ("objective", "auto", "精确重复提问次数"),
    "exact_duplicate_question_rate": ("objective", "auto", "精确重复提问比例"),
    "distinct_model_reference_ids": ("objective", "auto", "不同引用编号数（模型自报）"),
    "model_reference_mentions": ("objective", "auto", "引用编号出现次数（模型自报）"),
    "interruptions": ("objective", "auto", "中断事件数"),
    "transient_retries": ("objective", "auto", "临时重试次数"),
    "spoken_truth": ("objective", "auto", "Spoken Truth · 来源问题信息引出代理（语义审核）"),
    "guide_coverage": ("rubric", "manual", "回答覆盖的子主题比例"),
    "relevant_response_volume_characters": ("rubric", "manual", "相关回答加权字符量"),
    "novel_response_count": ("rubric", "manual", "大纲外相关回答次数"),
    "leading_question_count": ("rubric", "manual", "诱导性提问次数"),
    "unclear_question_count": ("rubric", "manual", "不清晰提问次数"),
    "support_rapport_count": ("rubric", "manual", "支持/共情表达次数"),
    "redundant_question_count_manual": ("rubric", "manual", "无效语义重复提问次数（人工）"),
    "redundant_question_rate_manual": ("rubric", "manual", "无效语义重复提问比例（人工）"),
    "coherence_manual": ("rubric", "manual", "整体连贯性（人工）"),
    "adaptiveness_manual": ("rubric", "manual", "整体适应性（人工）"),
    "overall_experience_self_report": ("rubric", "human_self_report", "真人报告的整体体验"),
    "comfort_self_report": ("rubric", "human_self_report", "真人报告的舒适度"),
}

MANUAL_RUBRICS = {
    "research_relevance": {"0": "无关", "1": "部分相关", "2": "直接相关"},
    "coherence": {"1": "明显不连贯", "2": "承接较弱", "3": "基本连贯", "4": "逻辑与过渡良好"},
    "adaptiveness": {"1": "无视回答", "2": "较少基于回答追问", "3": "有有效追问", "4": "多次有效调整策略"},
    "binary": {"leading": "诱导性", "unclear": "模糊不清", "support_rapport": "支持与关系建立"},
}


def _validate_manual_rubrics(raw):
    if not isinstance(raw, dict) or set(raw) != set(MANUAL_RUBRICS):
        raise BenchmarkConfigError("manual_rubrics 必须包含 research_relevance、coherence、adaptiveness、binary")
    out={}
    for name,default in MANUAL_RUBRICS.items():
        values=raw.get(name)
        if not isinstance(values,dict) or set(values)!=set(default):
            raise BenchmarkConfigError(f"manual_rubrics.{name} 的评分等级不完整或被更改")
        if any(not isinstance(v,str) or not v.strip() or len(v)>160 for v in values.values()):
            raise BenchmarkConfigError(f"manual_rubrics.{name} 的各等级说明必须为 1–160 字")
        out[name]={k:values[k].strip() for k in default}
    return out

PARAMS = {
    "exact_duplicate_question_count": {"normalization": "nfkc_strip_punctuation"},
    "exact_duplicate_question_rate": {"normalization": "nfkc_strip_punctuation"},
    "relevant_response_volume_characters": {"relevance_weights": [0, 1, 2]},
    "novel_response_count": {"minimum_relevance": 2},
}

# Legacy format's text is retained only as fixed documentation. If a user
# edits this field expecting a new algorithm, reject rather than silently ignore.
LEGACY_FORMULAS = {
    "turns_recorded": "count(turns)",
    "interview_completion_ratio": "min(turns_recorded / turn_limit,1)",
    "exact_duplicate_question_rate": "normalization_equal_prior_questions / all_questions",
    "distinct_model_reference_ids": "count(unique(used_fact_ids or evidence_ids))",
    "model_reference_mentions": "sum(reference_ids_per_turn)",
    "guide_coverage": "unique(covered_subtopic_ids_after_review)/all_guide_subtopics",
    "relevant_response_volume_characters": "sum(len(answer.strip())*relevance_0_to_2)",
    "novel_response_count": "count(relevance==2 and reviewed_subtopics==[])",
    "leading_question_count": "count(leading==true)",
    "unclear_question_count": "count(unclear==true)",
    "support_rapport_count": "count(support_rapport==true)",
    "coherence_manual": "one_reviewed_1_to_4_rating",
    "adaptiveness_manual": "one_reviewed_1_to_4_rating",
    "overall_experience_self_report": "participant_1_to_5",
    "comfort_self_report": "participant_1_to_4",
}


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def default_config():
    return {
        "schema": SCHEMA,
        "version": CONFIG_VERSION,
        "id": "benchmark_foundation_default",
        "name": "Interview Benchmark · Foundation",
        "description": "只计算已具备证据的指标；其余显示 N/A；不计算未经验证的总分。",
        "metrics": [
            {"id": id_, "enabled": True, "method": id_, "label": label,
             "evaluator": evaluator, "group": group,
             "parameters": deepcopy(PARAMS.get(id_, {}))}
            for id_, (group, evaluator, label) in REGISTRY.items()
        ],
        "scoring": {"composite_score": None},
        "manual_rubrics": deepcopy(MANUAL_RUBRICS),
    }


def _check_parameters(id_, parameters):
    if not isinstance(parameters, dict):
        raise BenchmarkConfigError(f"{id_}: parameters 必须为对象")
    default = PARAMS.get(id_, {})
    if set(parameters) - set(default):
        raise BenchmarkConfigError(f"{id_}: 不支持参数 {', '.join(sorted(set(parameters)-set(default)))}")
    result = deepcopy(default)
    result.update(parameters)
    if "normalization" in result and result["normalization"] not in (
        "nfkc_strip_punctuation", "exact_trim"):
        raise BenchmarkConfigError(f"{id_}: normalization 只能为 nfkc_strip_punctuation 或 exact_trim")
    if "minimum_relevance" in result and (
        type(result["minimum_relevance"]) is not int or result["minimum_relevance"] not in (1, 2)):
        raise BenchmarkConfigError(f"{id_}: minimum_relevance 只能为 1 或 2")
    if "relevance_weights" in result:
        weights = result["relevance_weights"]
        if not isinstance(weights, list) or len(weights) != 3 or any(
            type(w) not in (int, float) or not math.isfinite(w) or w < 0 or w > 4 for w in weights):
            raise BenchmarkConfigError(f"{id_}: relevance_weights 必须为三个 0–4 的有限数值")
        if weights != sorted(weights):
            raise BenchmarkConfigError("relevance_weights 必须随相关度不下降")
    return result


def _from_legacy(raw):
    # Both v0.1 rubric and the accidentally delivered wrapper are accepted.
    if raw.get("benchmark_document") == "interview_benchmark.v0.1":
        raw = raw.get("rubric")
        if not isinstance(raw, dict):
            raise BenchmarkConfigError("旧版汇总 JSON 缺少 rubric 对象")
    if raw.get("schema") != "interview_benchmark_rubric.v0.1":
        raise BenchmarkConfigError("不支持的 Benchmark JSON 类型；请导入 Config v1 或 v0.1 Rubric")
    if raw.get("version") != "0.1":
        raise BenchmarkConfigError("旧版 Rubric 版本不受支持")
    rows = raw.get("metrics")
    if not isinstance(rows, list):
        raise BenchmarkConfigError("旧版 Rubric 缺少 metrics 列表")
    translated = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise BenchmarkConfigError("旧版 Rubric 存在无效指标")
        id_ = row["id"]
        if id_ not in REGISTRY:
            raise BenchmarkConfigError(f"旧版 Rubric 指标 {id_} 没有可运行的评分算法")
        expected_formula = LEGACY_FORMULAS.get(id_)
        if row.get("formula_or_rule") != expected_formula:
            raise BenchmarkConfigError(f"旧版 {id_} 的公式说明已更改，无法执行。请改用 Config v1 的受支持参数")
        if row.get("evaluator") != REGISTRY[id_][1]:
            raise BenchmarkConfigError(f"旧版 {id_} 的评价者类型与受支持算法不一致")
        translated.append({"id": id_, "enabled": True, "method": id_,
                           "label": row.get("name_zh", REGISTRY[id_][2]),
                           "evaluator": REGISTRY[id_][1], "group": REGISTRY[id_][0],
                           "parameters": deepcopy(PARAMS.get(id_, {}))})
    score_def = raw.get("scores", {})
    if not isinstance(score_def, dict) or score_def.get("composite_score") is not None:
        raise BenchmarkConfigError("不支持未经校准的综合分；scores.composite_score 必须为 null")
    human_rubric = raw.get("manual_scoring", MANUAL_RUBRICS)
    verified_rubric = _validate_manual_rubrics(human_rubric)
    return {
        "schema": SCHEMA, "version": CONFIG_VERSION,
        "id": "migrated_foundation_v01", "name": "Interview Benchmark v0.1 · 已转换",
        "description": "从旧版 Rubric 转换；formula_or_rule 为说明文字，实际计算由内置指标及参数执行。",
        "metrics": translated,
        "scoring": {"composite_score": None},
        "manual_rubrics": verified_rubric,
    }


def validate_config(raw=None):
    """Validate or import a JSON rule set and return a canonical executable config.

    All algorithm identities and parameters are explicitly allowlisted.  This
    prevents imported files from running arbitrary Python, JavaScript or network
    code, and prevents user labels from masquerading as new calculations.
    """
    if raw is None:
        raw = default_config()
    if not isinstance(raw, dict):
        raise BenchmarkConfigError("Benchmark Config 必须是 JSON 对象")
    raw = deepcopy(raw)
    if raw.get("schema") != SCHEMA:
        raw = _from_legacy(raw)
    if raw.get("version") != CONFIG_VERSION:
        raise BenchmarkConfigError("Benchmark Config 版本不受支持（需要 1.0）")
    if raw.get("schema") != SCHEMA:
        raise BenchmarkConfigError("Benchmark Config schema 不正确")
    extra = set(raw) - {"schema", "version", "id", "name", "description", "metrics", "scoring", "manual_rubrics"}
    if extra:
        raise BenchmarkConfigError("Benchmark Config 含尚未实现的字段：" + ", ".join(sorted(extra)) +
            "；请使用支持的参数，不会静默忽略评分规则")
    for k in ("id", "name", "description"):
        value = raw.get(k, "")
        if not isinstance(value, str) or len(value) > (120 if k != "description" else 500):
            raise BenchmarkConfigError(f"Benchmark Config {k} 不合法")
    if not raw.get("name"):
        raise BenchmarkConfigError("Benchmark Config 缺少 name")
    scoring = raw.get("scoring", {"composite_score": None})
    if not isinstance(scoring, dict) or set(scoring) - {"composite_score"} or scoring.get("composite_score") is not None:
        raise BenchmarkConfigError("当前不支持加权总分；scoring.composite_score 必须为 null")
    rows = raw.get("metrics")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_METRICS:
        raise BenchmarkConfigError("metrics 必须包含 1–23 条指标")
    normalized, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise BenchmarkConfigError("metric 必须是对象")
        if set(row) - {"id", "enabled", "method", "label", "evaluator", "group", "parameters"}:
            raise BenchmarkConfigError(f"{row.get('id', 'metric')}: 存在不支持的规则字段")
        id_ = row.get("id")
        if not isinstance(id_, str) or id_ not in REGISTRY:
            raise BenchmarkConfigError(f"未知指标 {id_}；不执行来自 JSON 的自定义代码或公式")
        if id_ in seen:
            raise BenchmarkConfigError("重复指标 ID：" + id_)
        seen.add(id_)
        if type(row.get("enabled")) is not bool:
            raise BenchmarkConfigError(f"{id_}: enabled 必须为 true/false")
        if row.get("method", id_) != id_:
            raise BenchmarkConfigError(f"{id_}: method 与受支持的内置计算器不一致")
        group, evaluator, default_label = REGISTRY[id_]
        if row.get("group", group) != group or row.get("evaluator", evaluator) != evaluator:
            raise BenchmarkConfigError(f"{id_}: 不能把人工评价改写为自动评价")
        label = row.get("label", default_label)
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise BenchmarkConfigError(f"{id_}: label 必须为 1–80 字")
        normalized.append({"id": id_, "enabled": row["enabled"], "method": id_,
                           "label": label.strip(), "group": group, "evaluator": evaluator,
                           "parameters": _check_parameters(id_, row.get("parameters", {}))})
    if not any(m["enabled"] for m in normalized):
        raise BenchmarkConfigError("至少启用一项指标")
    repeat_modes = {m["parameters"]["normalization"] for m in normalized
                    if m["enabled"] and m["id"] in (
                      "exact_duplicate_question_count", "exact_duplicate_question_rate")}
    if len(repeat_modes) > 1:
        raise BenchmarkConfigError("重复计数与重复率的 normalization 必须一致")
    return {
        "schema": SCHEMA, "version": CONFIG_VERSION,
        "id": raw.get("id", ""), "name": raw["name"],
        "description": raw.get("description", ""), "metrics": normalized,
        "manual_rubrics": _validate_manual_rubrics(raw.get("manual_rubrics", MANUAL_RUBRICS)),
        "scoring": {"composite_score": None}
    }


def config_summary(config):
    config = validate_config(config)
    active = [m for m in config["metrics"] if m["enabled"]]
    return {"name": config["name"], "id": config["id"], "version": config["version"],
            "schema": config["schema"], "sha256": fingerprint(config),
            "enabled_metric_count": len(active),
            "objective_count": sum(m["group"] == "objective" for m in active),
            "rubric_count": sum(m["group"] == "rubric" for m in active)}
