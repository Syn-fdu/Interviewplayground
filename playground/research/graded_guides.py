"""Read-only, source-preserving graded guides. Never infer unrelated interview mappings."""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
from playground.research.data_intake import candidate_id

ROOT = Path(__file__).resolve().parents[2] / "resources" / "bundled_data" / "graded_guides"
LEVELS = ("level0", "level1", "level2")

@lru_cache(maxsize=32)
def _dataset(dataset):
    path = ROOT / dataset / "three_levels.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def list_datasets():
    return sorted(p.parent.name for p in ROOT.glob("*/three_levels.json"))

def variants(dataset):
    data=_dataset(dataset)
    if data is None:return []
    if isinstance(data, dict):return [data]
    if isinstance(data, list):return data
    raise ValueError("Invalid graded guide type")

def match(dataset, candidate, category=None):
    """Resolve exact interview IDs first; never substitute another interview's guide.

    A shared dataset guide applies to its whole dataset. Multiple category guides
    require an explicit category, except that one unambiguous category is allowed.
    """
    rows=variants(dataset)
    if not rows:return None
    if len(rows)==1 and rows[0].get("guide_kind") in (None,"dataset","category","unified"):
        row=rows[0]
        if row.get("guide_kind")=="category" and category and row.get("category")!=category:
            return None
        return row
    exact=[]
    for row in rows:
        if not isinstance(row,dict):continue
        if row.get("guide_kind")=="per_interview":
            gid=str(row.get("guide_id", ""))
            source_id=gid.split("::",1)[-1]
            if candidate_id(dataset,source_id)==candidate:
                exact.append(row)
    if len(exact)>1:raise ValueError("Ambiguous interview guide")
    if exact:return exact[0]
    if len(rows)==1 and rows[0].get("guide_kind")=="unified":return rows[0]
    categories=[r for r in rows if r.get("guide_kind")=="category"]
    if category:
        chosen=[r for r in categories if r.get("category")==category]
        if len(chosen)==1:return chosen[0]
    return None

def question_view(row, level):
    if level not in LEVELS:raise ValueError("Invalid guide level")
    items=row.get(level)
    if not isinstance(items,list):raise ValueError("Guide level is not a list")
    questions=[]
    for item in items:
        if isinstance(item,str):q=item; identifier=None
        elif isinstance(item,dict):q=item.get("question");identifier=item.get("id")
        else:continue
        if isinstance(q,str) and q.strip():
            questions.append({"text":q.strip(), "id":identifier or "Q"+str(len(questions)+1)})
    if not questions:raise ValueError("Selected graded guide is empty")
    return [{"id":"G1","title":str(row.get("category") or row.get("dataset") or "Graded guide"),"questions":questions}]

def available_levels(dataset,candidate,category=None):
    row=match(dataset,candidate,category)
    return [k for k in LEVELS if isinstance(row.get(k),list) and any(
        isinstance(x,str) and x.strip() or isinstance(x,dict) and isinstance(x.get("question"),str) and x["question"].strip()
        for x in row[k])] if row else []
