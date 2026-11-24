"""Original v1.4.15 deterministic full-sentence-prefix compaction."""
import re

def compact_original(text, limit=240):
    value=re.sub(r'\s+',' ',str(text or '')).strip()
    if len(value)<=limit:return value
    prefix=value[:limit]
    breaks=[m.end() for m in re.finditer(r'[。！？.!?；;]',prefix)]
    cut=max(breaks) if breaks and max(breaks)>=max(60,limit//2) else limit
    return prefix[:cut].rstrip()+'…'
