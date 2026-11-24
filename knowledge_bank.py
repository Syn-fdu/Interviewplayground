"""Local domain knowledge preparation; distinct from source-linked personal memories.

Never derive autobiographical facts from these public editorial teaching cues.
Classification is metadata-only; no respondent transcripts are read here.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

BASE=Path(__file__).resolve().parent / 'knowledge_banks'
VERSION='knowledge_preparation.v1'

@lru_cache(maxsize=1)
def _lookup():
    path=BASE/'candidate_domains.json'
    if not path.is_file():return {}
    d=json.loads(path.read_text(encoding='utf-8'))
    return d.get('candidates',{})

@lru_cache(maxsize=32)
def _bank(domain):
    if not re.fullmatch(r'[a-z_]{3,50}',domain):return None
    file=BASE/(domain+'.json')
    if not file.is_file():return None
    data=json.loads(file.read_text(encoding='utf-8'))
    entries=data.get('entries',[])
    if not 100<=len(entries)<=200 or len({v['knowledge_id'] for v in entries})!=len(entries):
        raise ValueError('Invalid KnowledgeBank for '+domain)
    return data

def resolve_domain(bundle):
    scenario=bundle.get('source_scenario') or {}
    prov=scenario.get('provenance') or {}
    cid=prov.get('candidate_id','')
    row=_lookup().get(cid)
    if not row:return None
    if row['dataset_id']!=prov.get('dataset'):return None
    return row

def signature(bundle, knowledge_value=None):
    traits=bundle.get('respondent',{}).get('behavior_parameters') or {}
    knowledge=traits.get('knowledge',0.5) if knowledge_value is None else knowledge_value
    if isinstance(knowledge,bool) or not isinstance(knowledge,(int,float)) or not math.isfinite(knowledge) or not 0<=knowledge<=1:
        raise ValueError('knowledge must be a finite 0..1 number')
    row=resolve_domain(bundle)
    if row is None:
        return {'status':'unclassified','knowledge':knowledge,'domain':None,'bank_digest':None}
    bank=_bank(row['domain'])
    if bank is None:return {'status':'bank_missing','knowledge':knowledge,'domain':row['domain'],'bank_digest':None}
    digest=hashlib.sha256(json.dumps(bank['entries'],sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    return {'status':'available','knowledge':float(knowledge),'domain':row['domain'],
            'subdomain':row.get('subdomain'), 'bank_digest':digest}

def prepare(bundle,knowledge_value=None):
    sig=signature(bundle,knowledge_value)
    if sig['status']!='available':
        return {'status':sig['status'],'domain':sig.get('domain'), 'selected_count':0,
                'available_count':0,'entries':[], 'signature':sig}
    bank=_bank(sig['domain'])
    full=bank['entries']
    # Groups of five facets are ranked by stable source-scoped hash, so
    # increasing knowledge never removes previously selected concepts.
    source=(bundle.get('source_scenario') or {}).get('provenance') or {}
    cid=str(source.get('candidate_id') or source.get('source_fingerprint_sha256') or 'unbound')
    concepts={}
    for entry in full: concepts.setdefault(entry['concept_id'],[]).append(entry)
    ranked=sorted(concepts, key=lambda c:hashlib.sha256((cid+'|'+sig['domain']+'|'+c).encode('utf8')).hexdigest())
    ordered=[]
    for c in ranked:
        ordered.extend(sorted(concepts[c],key=lambda e:e['knowledge_id']))
    # 0 -> 0, .5 -> 50, 1 -> all 100; one-sided monotonic step selection.
    amount=min(len(ordered),max(0,int(math.floor(sig['knowledge']*len(ordered)+1e-9))))
    selected=[{'knowledge_id':e['knowledge_id'],'domain':e['domain'],
               'topic':e['topic'],'keywords_en':e.get('keywords_en',[]),'text':e['text'],
               'source_kind':e['source_kind'],'verification_status':e['verification_status']}
              for e in ordered[:amount]]
    return {'status':'prepared','domain':sig['domain'],'subdomain':sig.get('subdomain'),
        'knowledge_level':sig['knowledge'], 'selected_count':len(selected),
        'available_count':len(full),'entries':selected,'signature':sig,
        'provenance':'editorial_general_domain_knowledge_not_personal',
        'human_review_required':True}

def retrieve(prepared, question, *, max_items=4, max_chars=650):
    """Retrieve only question-relevant general knowledge from the prepared subset."""
    entries=prepared.get('entries') or []
    text=str(question or '').lower()
    if not text.strip():return []
    # Bigram overlap of domain terms lets this work for Chinese, with a
    # specific-term preference. Do not assume every question needs general knowledge.
    def grams(v):
        letters=re.findall(r'[a-z0-9]{3,}|[\u3400-\u9fff]+',v.lower())
        return {w for token in letters for w in ([token] if len(token)<=2 else [token]+[token[i:i+2] for i in range(len(token)-1)])}
    qg=grams(text)
    ranked=[]
    for entry in entries:
        term=entry['topic'].lower()
        english_matches=[w for w in entry.get('keywords_en',[]) if w and w.lower() in text]
        if term in text or english_matches:
            score=100
        else:
            term_g=grams(term)
            overlap=len(term_g&qg)
            score=overlap/max(1,len(term_g))
        if score>0.35:
            ranked.append((score,entry))
    ranked.sort(key=lambda x:(-x[0],x[1]['knowledge_id']))
    out=[];nchars=0
    # Avoid near-identical repeated facets from one concept/topic.
    topics=set()
    for _,e in ranked:
        if e['topic'] in topics:continue
        txt=e['text']
        if nchars+len(txt)>max_chars:continue
        out.append({'knowledge_id':e['knowledge_id'],'topic':e['topic'],'text':txt})
        topics.add(e['topic']);nchars+=len(txt)
        if len(out)>=max_items:break
    return out
