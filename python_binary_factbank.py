"""MemoryBank/V2 source-only two-class temporal labels. No model calls.
The authoritative facts are *eligible source records*, never agent-generated summaries.
Classification is metadata for post-retrieval judgement only.
"""
from python_factbank import classify as native_v15
VERSION="python_binary_factbank_v15.integrated.1"

def classify_source(source):
    row={"original_text":str(source.get("text") or ""),
         "source_question":str(source.get("source_question") or ""),
         "semantic_status":source.get("semantic_status") or "",
         "source_verified":True,"question_verified":bool(source.get("source_question"))}
    result=native_v15(row)
    category=result.get("category")
    uncertain=category not in ("memory","reality")
    if uncertain:
        # This is *not* a claim that a non-fact is a current state. It is an
        # explicitly flagged provisional label until downstream source judgement.
        category="reality"
    return {"fact_category":category,"review_required":bool(result.get("review_required") or uncertain),
            "confidence":0.0 if uncertain else float(result.get("confidence") or 0.0),
            "classification_reason":("binary_provisional_"+str(result.get("classification_reason"))
                if uncertain else str(result.get("classification_reason") or "python_v15")),
            "classification_origin":VERSION}
