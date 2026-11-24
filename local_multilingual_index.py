"""v1.5.8 optional offline multilingual index; source-derived retrieval hints only.

No model is downloaded. Set INTERVIEW_MULTILINGUAL_MODEL_DIR to an existing local
SentenceTransformer directory to enable this feature. An unavailable model is
reported, not silently treated as a multilingual index.
"""
from __future__ import annotations
import hashlib
import math
import os
from pathlib import Path

INDEX_VERSION = 'offline-multilingual-v158'
_MODEL = {}


def _model_path():
    raw = os.getenv('INTERVIEW_MULTILINGUAL_MODEL_DIR', '').strip()
    if not raw:
        return None, 'not_configured'
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        return None, 'model_directory_missing'
    return path, None


def _get_model(path):
    key = str(path)
    if key not in _MODEL:
        from sentence_transformers import SentenceTransformer
        # An explicit local path prevents accidental network model downloads.
        _MODEL[key] = SentenceTransformer(str(path), device='cpu')
    return _MODEL[key]


def _normalize(vec):
    floats = [float(x) for x in vec]
    length = math.sqrt(sum(x*x for x in floats))
    return [round(v / length, 7) for v in floats] if length > 0 else None


def build_index(records, seconds_left=None):
    path, error = _model_path()
    if error:
        return None, error
    if seconds_left is not None and seconds_left < 45:
        return None, 'insufficient_time_budget'
    try:
        model = _get_model(path)
        texts = [str(r.get('source_question') or '') + '\n' + str(r.get('text') or '') for r in records]
        vectors = model.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
        normalized = [_normalize(v) for v in vectors]
        if len(normalized) != len(records) or not all(normalized):
            return None, 'invalid_vector_output'
        return {'version':INDEX_VERSION, 'model_fingerprint':hashlib.sha256(str(path).encode()).hexdigest()[:16],
                'vectors':{str(r['id']):v for r,v in zip(records,normalized)}}, None
    except (ImportError, OSError, ValueError, RuntimeError, TypeError):
        return None, 'local_model_unavailable'


def rank(question, groups, index):
    """Only rank authorized group IDs; never construct or disclose source facts."""
    if not isinstance(index, dict) or index.get('version') != INDEX_VERSION:
        return []
    path, error = _model_path()
    if error or hashlib.sha256(str(path).encode()).hexdigest()[:16] != index.get('model_fingerprint'):
        return []
    try:
        encoded = _get_model(path).encode([str(question)], show_progress_bar=False, convert_to_numpy=True)
        query = _normalize(encoded[0])
        if not query:
            return []
        vectors = index.get('vectors') or {}
        ranked = []
        for group in groups:
            scores = []
            for source in group.get('facts') or []:
                vector = vectors.get(str(source.get('id')))
                if isinstance(vector, list) and len(vector)==len(query):
                    scores.append(sum(a*b for a,b in zip(query,vector)))
            if scores:
                ranked.append((max(scores), str(group.get('id'))))
        return sorted(ranked, key=lambda pair:(-pair[0],pair[1]))
    except (ImportError, OSError, ValueError, RuntimeError, TypeError):
        return []
