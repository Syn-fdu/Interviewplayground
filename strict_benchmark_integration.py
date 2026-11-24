"""Isolated strict benchmark adapter: no calls from the interview execution path.

Original Fast-Grounded and SpokenTruth engines are kept under
standalone_evaluation/ and invoked only after explicit researcher action.
No synthetic memory citations, fake model judgments, or transcript truncation.
"""
from __future__ import annotations
import base64, copy, hashlib, io, json, os, re, pathlib, shutil, subprocess, sys, tempfile, threading, time, uuid, zipfile
from datetime import datetime, timezone
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from benchmark_diagnostics import diagnose, record as record_failure, read_job as read_error_records

ROOT=pathlib.Path(__file__).resolve().parent
FAST=ROOT/'standalone_evaluation'/'fast'/'fast_benchmark.py'
SPOKEN=ROOT/'standalone_evaluation'/'spoken'/'strict_full.py'
JOB_HOME=ROOT/'prepared_memory'/'strict_benchmark_jobs'
LOCK=threading.RLock()
JOBS={}
MAX_JOBS=2
MAX_INPUT_BYTES=12*1024*1024

class StrictContractError(ValueError): pass

def dump(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def _validate_identity(run,outline,memory):
    if not all(isinstance(x,dict) for x in (run,outline,memory)):
        raise StrictContractError('Transcript、Outline、Memory 必须是 JSON 对象')
    if run.get('schema_version')!='interview_transcript_v1' or run.get('report_type')!='benchmark_transcript':
        raise StrictContractError('Transcript 必须使用 benchmark_transcript / interview_transcript_v1')
    turns=run.get('turns')
    if not isinstance(turns,list) or not turns:
        raise StrictContractError('必须提供至少一轮完整问答')
    sid=run.get('scenario_id')
    if not isinstance(sid,str) or not sid:
        raise StrictContractError('Transcript 缺少 scenario_id')
    meta=outline.get('metadata')
    if not isinstance(meta,dict):
        raise StrictContractError('Outline.metadata 必须是对象')
    agent_outline=outline.get('agent_outline')
    if not isinstance(agent_outline,dict):
        raise StrictContractError('Outline.agent_outline 必须是对象')
    if memory.get('scenario_id')!=sid or meta.get('scenario_id')!=sid:
        raise StrictContractError('Transcript / Outline / Memory 场景 ID 不一致')
    rev=run.get('research_revision')
    if rev is None or memory.get('source_revision')!=rev or meta.get('revision')!=rev:
        raise StrictContractError('研究修订版本不一致')
    expected={
        'source_fingerprint_sha256':run.get('source_fingerprint_sha256'),
        'source_dataset':run.get('source_dataset'),
        'source_candidate_id':run.get('source_candidate_id'),
    }
    provenance_verified=True
    for field,value in expected.items():
        for origin,record in [('Outline',meta),('Memory',memory)]:
            actual=record.get(field)
            if actual not in (None,'') and actual!=value:
                raise StrictContractError(
                    origin+' 导出的 '+field+' 与 Transcript 不一致。'
                    '请使用同一来源、同一候选人的三个导出文件；不能混合不同来源包。')
            if actual in (None,''):
                provenance_verified=False  # Legacy exports stay readable, but their hashes cannot be checked.
    source=memory.get('source_facts')
    if not isinstance(source,list) or not source:
        raise StrictContractError('Memory.source_facts 不能为空')
    ids=[]
    for i,item in enumerate(source):
        if not isinstance(item,dict) or not isinstance(item.get('text'),str) or not item['text'].strip():
            raise StrictContractError('来源记忆 %d 为空或结构错误'%i)
        dataset_source=item.get('dataset_source')
        if dataset_source is not None and not isinstance(dataset_source,dict):
            raise StrictContractError('来源记忆 %d 的 dataset_source 必须是对象'%i)
        mid=item.get('dataset_memory_id') or (dataset_source or {}).get('dataset_memory_id')
        if not isinstance(mid,str) or not mid:
            raise StrictContractError('来源记忆 %d 缺少真实 dataset_memory_id，不允许伪造'%i)
        if not isinstance(item.get('id'),str) or not item['id']:
            raise StrictContractError('来源记忆 %d 缺少 source_fact ID'%i)
        ids.append(mid)
    if len(set(ids))!=len(ids):
        raise StrictContractError('源 Memory ID 重复，拒绝静默去重')
    midset=set(ids)
    checked=[]
    for i,turn in enumerate(turns,1):
        if not isinstance(turn,dict) or turn.get('round')!=i or turn.get('record_type')=='closing':
            raise StrictContractError('Transcript 轮次必须从1连续编号，不能混入结束语')
        if (turn.get('scenario_id')!=sid or turn.get('session_id')!=run.get('session_id')
              or turn.get('research_revision')!=rev):
            raise StrictContractError('第 %d 轮场景、Session 或版本不一致'%i)
        for key in ('interviewer','respondent'):
            if not isinstance(turn.get(key),str) or not turn[key].strip():
                raise StrictContractError('第 %d 轮缺少真实 %s 正文'%(i,key))
        att=turn.get('memory_attribution')
        if not isinstance(att,dict) or att.get('schema')!='memory_citation.v1':
            raise StrictContractError(
                '第 %d 轮上游 Transcript 未导出 memory_citation.v1；'
                '请从同一场景的受访者回答记录核查，不能默认补造空引用。'%i)
        if att.get('complete') is not True:
            raise StrictContractError(
                '第 %d 轮上游回答标记 Citation 不完整（method=%s）。'
                '这不是 JSONL 解析失败；原版 SpokenTruth 不允许将未知引用计为零。'
                '请检查该轮 Respondent 的 grounding/引用生成日志。'
                %(i,att.get('citation_method','unknown')))
        a=att.get('available_memory_ids')
        c=att.get('cited_memory_ids')
        if not isinstance(a,list) or not isinstance(c,list) or any(not isinstance(x,str) for x in a+c):
            raise StrictContractError('第 %d 轮 Memory ID 数组无效'%i)
        if len(a)!=len(set(a)) or len(c)!=len(set(c)) or set(c)-set(a) or set(a)-midset:
            raise StrictContractError('第 %d 轮 Memory ID 重复、来源错误或不属于可用集合'%i)
        checked.append(turn)
    topics=agent_outline.get('topics')
    if not isinstance(topics,list) or not topics:
        raise StrictContractError('缺少已准备的原生 Interviewer Outline')
    required=[]
    question_ids=[]
    for idx,topic in enumerate(topics):
        if not isinstance(topic,dict):
            raise StrictContractError('Outline.topics[%d] 必须是对象'%idx)
        questions=topic.get('questions')
        if not isinstance(questions,list) or not questions:
            raise StrictContractError('Outline.topics[%d].questions 必须是非空数组'%idx)
        for q in questions:
            if not isinstance(q,dict) or not isinstance(q.get('id'),str) or not q['id']:
                raise StrictContractError('Outline 问题缺少唯一有效 ID')
            question_ids.append(q['id'])
            slots=q.get('required_slots')
            if not isinstance(slots,list) or not slots:
                raise StrictContractError('问题 '+q['id']+' 缺少有效 required_slots；不允许默认填充')
            for slot in slots:
                if not isinstance(slot,dict) or not isinstance(slot.get('id'),str) or not slot['id'] or not isinstance(slot.get('description'),str) or not slot['description'].strip():
                    raise StrictContractError('Outline Slot ID / 语义描述缺失')
                # Native Interviewer fallback is part of its actual protocol.
                # Preserve its exact ID/source; do not invent LLM-derived slots.
                required.append(str(q['id'])+':'+str(slot['id']))
    if not question_ids or len(question_ids)!=len(set(question_ids)):
        raise StrictContractError('原始 Outline 问题 ID 缺失或重复')
    if len(required)!=len(set(required)):
        raise StrictContractError('必需 Slot ID 重复')
    native_fallback_count=sum(
        1 for topic in topics for q in topic.get('questions',[])
        if all(slot.get('source')=='fallback' for slot in q.get('required_slots',[])))
    return {'rounds':len(checked),'source_memories':len(source),'required_slots':len(required),
            'source_provenance_verified':provenance_verified,
            'native_fallback_questions':native_fallback_count,
            'slot_semantic_quality':('native_fallback_not_llm_enriched' if native_fallback_count
                                     else 'has_non_fallback_slots'),
            'fast_eligible':True,'full_spoken_eligible':True}

def make_bundle(run,outline,memory):
    stats=_validate_identity(run,outline,memory)
    manifest={'schema':'playground_strict_benchmark_bundle.v1',
              'created_at':datetime.now(timezone.utc).isoformat(),
              'scenario_id':run['scenario_id'],'session_id':run['session_id'],
              'research_revision':run['research_revision'],
              'source_dataset':run.get('source_dataset'),
              'source_candidate_id':run.get('source_candidate_id'),
              'source_fingerprint_sha256':run.get('source_fingerprint_sha256'),
              'stats':stats,
              'privacy':'PRIVATE: contains full interview answers and source memory',
              'engines':{'fast':'FastGrounded Standalone fixed-key v1.2',
                         'spoken':'SpokenTruth AllMemory v3.3/v3.4 strict_full.py'}}
    payload={'transcript.json':run,'outline.json':outline,'memory.json':memory}
    manifest['sha256']={k:hashlib.sha256(dump(v)).hexdigest() for k,v in payload.items()}
    with io.BytesIO() as sink:
        with zipfile.ZipFile(sink,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,value in payload.items():
                z.writestr(name,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
            z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        return sink.getvalue(),manifest

def convert_playground_jsonl(text):
    """Lossless envelope conversion for the native Playground JSONL export.

    Round records are copied without filtering or inventing attribution data.
    This only validates transcript-level consistency. The independent full
    strict contract is enforced later with the ORIGINAL outline and memory.
    """
    if not isinstance(text,str) or not text.strip():
        raise StrictContractError('请选择有真实逐轮问答的 .jsonl 文件')
    if len(text.encode('utf-8'))>8*1024*1024:
        raise StrictContractError('JSONL 超过8MB限制')
    rows=[]
    for number,line in enumerate(text.lstrip("\ufeff").splitlines(),1):
        if not line.strip():
            continue
        try:
            item=json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrictContractError('JSONL 第%d行不是有效 JSON：%s'%(number,exc.msg))
        if not isinstance(item,dict):
            raise StrictContractError('JSONL 第%d行必须是逐轮问答对象'%number)
        rows.append(item)
    if not rows:
        raise StrictContractError('JSONL 没有可用的问答行')
    # A completed Playground interview appends a genuine closing event.
    # Preserve that record separately rather than mistaking it for a Q/A turn.
    closing=None
    if rows[-1].get('record_type')=='closing':
        closing=rows.pop()
    if not rows or any(r.get('record_type')=='closing' for r in rows):
        raise StrictContractError('JSONL 没有有效问答，或结束语出现在问答中间')
    first=rows[0]
    required_identity=('scenario_id','session_id','research_revision',
                       'source_dataset','source_candidate_id',
                       'source_fingerprint_sha256')
    for key in required_identity:
        if first.get(key) is None or first.get(key)=='':
            raise StrictContractError('JSONL 首轮缺少真实来源字段：'+key)
    if first.get('schema_version')!='interview_transcript_v1':
        raise StrictContractError('该 JSONL 不是原生 interview_transcript_v1')
    citation_issues=[]
    for i,row in enumerate(rows,1):
        if row.get('round')!=i:
            raise StrictContractError('JSONL 轮次必须从1开始连续编号，第%d轮错误'%i)
        for key in required_identity:
            if row.get(key)!=first[key]:
                raise StrictContractError('JSONL 第%d轮 %s 与首轮不一致'%(i,key))
        if row.get('schema_version')!=first['schema_version']:
            raise StrictContractError('JSONL 不同轮次的 schema_version 不一致')
        for key in ('interviewer','respondent'):
            if not isinstance(row.get(key),str) or not row[key].strip():
                raise StrictContractError('JSONL 第%d轮缺少 %s 正文'%(i,key))
        att=row.get('memory_attribution')
        if not isinstance(att,dict) or att.get('schema')!='memory_citation.v1':
            citation_issues.append({'round':i,'problem':'missing_memory_attribution'})
            continue
        if att.get('complete') is not True:
            citation_issues.append({'round':i,'problem':'citation_incomplete',
                                    'method':att.get('citation_method')})
            continue
        available=att.get('available_memory_ids')
        cited=att.get('cited_memory_ids')
        if not isinstance(available,list) or not isinstance(cited,list) or any(
                not isinstance(x,str) for x in available+cited):
            citation_issues.append({'round':i,'problem':'citation_ids_missing_or_invalid'})
        elif len(available)!=len(set(available)) or len(cited)!=len(set(cited)) or set(cited)-set(available):
            citation_issues.append({'round':i,'problem':'citation_ids_duplicate_or_unavailable'})
    if closing is not None:
        for key in required_identity:
            if closing.get(key)!=first[key]:
                raise StrictContractError('结束记录 %s 与访谈来源不一致'%key)
    result={
        'report_type':'benchmark_transcript',
        'schema_version':'interview_transcript_v1',
        'scenario_id':first['scenario_id'],
        'session_id':first['session_id'],
        'source_dataset':first['source_dataset'],
        'source_candidate_id':first['source_candidate_id'],
        'source_fingerprint_sha256':first['source_fingerprint_sha256'],
        'research_revision':first['research_revision'],
        'interviewer_policy':first.get('interviewer_provider'),
        'turns':rows,
    }
    if closing is not None:
        result['closing_record']=closing
    # The converter is intentionally lossless. Missing/unfinished citations
    # remain unknown, not fabricated as empty but complete.
    return result, {
        'rounds':len(rows),
        'scenario_id':first['scenario_id'],
        'research_revision':first['research_revision'],
        'session_id':first['session_id'],
        'citation_ready':not citation_issues,
        'citation_issues':citation_issues,
        'closing_record_preserved':closing is not None,
        'note':('原始访谈已读取，但 Citation 不完整；独立原版评分会拒绝严格机评。'
                if citation_issues else
                'Transcript 引用结构齐全；还须与同场景 Outline 和 Memory 逐项校验。')
    }


def import_external_archive(encoded):
    """Read a standalone strict-input ZIP without weakening source contracts.

    Only the three expected JSON names and optional manifest are accepted.
    The upload is never extracted to disk; no source IDs are manufactured.
    """
    import binascii
    if not isinstance(encoded, str) or not encoded:
        raise StrictContractError("请选择包含三个严格 JSON 的 ZIP 文件")
    if len(encoded)>16*1024*1024:
        raise StrictContractError("压缩包上传超过大小限制（Base64 16MB）")
    try:
        raw=base64.b64decode(encoded,validate=True)
    except (binascii.Error,ValueError):
        raise StrictContractError("ZIP 文件编码无效")
    if len(raw)>12*1024*1024:
        raise StrictContractError("ZIP 压缩文件不能超过12MB")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members=z.infolist()
            names=[info.filename for info in members]
            required={'transcript.json','outline.json','memory.json'}
            allowed=required|{'manifest.json'}
            if len(names)!=len(set(names)) or not required.issubset(names) or not set(names).issubset(allowed):
                raise StrictContractError(
                    "ZIP 必须仅包含根目录的 transcript.json / outline.json / memory.json"
                    "（可额外含 manifest.json）；不允许重复路径或额外文件")
            if any(info.is_dir() or info.file_size>8*1024*1024 or
                   info.compress_size>12*1024*1024 for info in members):
                raise StrictContractError("ZIP 内有目录或过大的文件")
            if sum(info.file_size for info in members)>18*1024*1024:
                raise StrictContractError("ZIP 解压后的数据超过18MB，拒绝处理")
            obj={}
            for kind in ('transcript','outline','memory'):
                try:
                    obj[kind]=json.loads(z.read(kind+'.json').decode('utf-8-sig'))
                except (UnicodeError,json.JSONDecodeError,ValueError):
                    raise StrictContractError(kind+'.json 不是有效 UTF-8 JSON')
            # Refuse a modified package with stale original checksums.
            if 'manifest.json' in names:
                try:
                    manifest=json.loads(z.read('manifest.json').decode('utf-8'))
                except (UnicodeError,json.JSONDecodeError):
                    raise StrictContractError("manifest.json 内容无效")
                expected=manifest.get('sha256') if isinstance(manifest,dict) else None
                if isinstance(expected,dict):
                    for kind in ('transcript','outline','memory'):
                        stored=expected.get(kind+'.json')
                        if stored and stored!=hashlib.sha256(dump(obj[kind])).hexdigest():
                            raise StrictContractError(kind+'.json 与 manifest SHA256 不一致')
    except (zipfile.BadZipFile,EOFError,RuntimeError):
        raise StrictContractError("文件不是完整有效的 ZIP 包")
    _,_,_,stats=_inputs(obj)
    return obj,stats


def _inputs(payload):
    if not isinstance(payload,dict):
        raise StrictContractError('输入必须是对象')
    memory=payload.get('memory');outline=payload.get('outline');run=payload.get('transcript')
    stats=_validate_identity(run,outline,memory)
    return run,outline,memory,stats

def _execute_check(job_dir, include_fast=True, include_spoken=True):
    calls=[
      [sys.executable,str(FAST),'check','--transcript',str(job_dir/'transcript.json'),
       '--outline',str(job_dir/'outline.json'),
       '--output-dir',str(job_dir/'fast_check')],
      [sys.executable,str(SPOKEN),'check-input','--memory',str(job_dir/'memory.json'),
       '--transcript',str(job_dir/'transcript.json')]
    ]
    for cmd in ([calls[0]] if include_fast else []) + ([calls[1]] if include_spoken else []):
        completed=subprocess.run(cmd,cwd=job_dir,capture_output=True,text=True,timeout=40)
        if completed.returncode:
            raise StrictContractError((completed.stderr or completed.stdout)[-1600:].strip())

def _remote_model_environment(base_url):
    """Provide the standalone engine with a key scoped to the exact judge URL.

    A key configured for another API must never be copied to this process.
    `strict_full.py` expects OPENAI_API_KEY for its remote provider.
    """
    from model_services import get, normalize_base_url
    url=normalize_base_url(base_url)
    if not url.startswith("https://"):
        raise StrictContractError("远程 SpokenTruth 只允许 HTTPS API 地址")
    for provider in ("shubi","maas"):
        cfg=get(provider,"interviewer",False)
        if cfg.base_url.rstrip("/")==url.rstrip("/") and cfg.api_key:
            env=os.environ.copy()
            env["OPENAI_API_KEY"]=cfg.api_key
            return env
    # Standalone project supports its own explicit OPENAI_API_KEY.
    explicit=os.environ.get("OPENAI_API_KEY")
    if explicit:
        env=os.environ.copy()
        return env
    raise StrictContractError(
        "远程 SpokenTruth API 无匹配密钥。请先配置与 Base URL 一致的采访员提供商，"
        "或单独设置 OPENAI_API_KEY；禁止跨域复用其他服务商 Key。")


def _benchmark_failure_detail(folder, mode, completed=None, exc=None):
    """Return the actual failure stage and safe error summary, not a generic failed badge.

    The raw model prompts, response bodies and API credentials remain inaccessible.
    """
    result_dir=folder/('fast_result' if mode=='fast' else 'spoken_result')
    reason=''
    stage='unknown'
    error_file=result_dir/'error.json'
    if error_file.is_file():
        try:
            record=json.loads(error_file.read_text(encoding='utf-8'))
            reason=str(record.get('reason') or '')
        except (OSError,ValueError,TypeError):
            pass
    if mode=='spoken':
        # SpokenTruth writes the actual engine failure here. Earlier versions
        # ignored this file, dropping the useful exception and its stage.
        failure=result_dir/'last_failure.json'
        if failure.is_file():
            try:
                info=json.loads(failure.read_text(encoding='utf-8'))
                if isinstance(info,dict):
                    reason=str(info.get('error') or reason)
                    stage=str(info.get('stage') or 'spoken_model_review')
            except (OSError,ValueError,TypeError):
                pass
        elif (result_dir/'complete_review.json').is_file():
            stage='spoken_postprocessing'
        elif (result_dir/'identity.json').is_file():
            stage='spoken_model_review'
        else:
            stage='spoken_initialization'
    if mode=='fast':
        events=result_dir/'events.jsonl'
        if events.is_file():
            try:
                for line in events.read_text(encoding='utf-8').splitlines()[-30:]:
                    event=json.loads(line)
                    if event.get('event') in ('model_request_failed','remote_request_failed'):
                        stage=str(event.get('stage') or 'grounding')
                    elif stage=='unknown' and event.get('event')=='model_request_started':
                        stage=str(event.get('stage') or 'unknown')
            except (OSError,ValueError,TypeError):
                pass
    if not reason and completed is not None:
        # The engine's own ERROR line is typically more useful than the Python traceback.
        messages=(completed.stderr or '').splitlines()
        lines=[x for x in messages if x.strip().startswith('ERROR:')]
        reason=(lines[-1] if lines else (completed.stderr or completed.stdout or '').strip().splitlines()[-1:][0] if (completed.stderr or completed.stdout or '').strip() else '')
    if not reason and exc is not None:
        reason=str(exc)
    reason=reason.strip() or '评分进程异常结束，未返回错误原因'
    # Never accidentally show a configured bearer key in the UI.
    import re
    reason=re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+',r'\1[REDACTED]',reason)
    reason=re.sub(r'(?i)(api[_ -]?key\s*[:=]\s*)\S+',r'\1[REDACTED]',reason)
    return {'error':reason[:1000], 'failure_stage':stage}

def _signal_process_group(proc):
    """Send Ctrl+C equivalent immediately to scorer's isolated process group."""
    import signal
    if proc.poll() is not None:return
    pid=getattr(proc,'pid',None)
    if not isinstance(pid,int):
        # Synthetic test doubles may not have an OS PID; never signal our own
        # hosting process accidentally.
        return
    try:
        if os.name=='nt':
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(pid,signal.SIGINT)
    except (OSError,ValueError,ProcessLookupError,PermissionError):
        try:proc.terminate()
        except (OSError,ProcessLookupError,AttributeError):pass

def _stop_process_tree(proc, *, already_signaled=False):
    """A short bounded OS-level stop, not a cooperative model wait.

    SIGINT/CTRL_BREAK to scorer process group first; then SIGKILL/taskkill on
    a strict deadline. wait() instead of communicate() to avoid racing the
    worker's own pipe reader. The API returns only after the subprocess exits.
    """
    import signal
    if not already_signaled:_signal_process_group(proc)
    try:
        proc.wait(timeout=0.18)
        return ('','')
    except (subprocess.TimeoutExpired,AttributeError):
        pass
    if proc.poll() is None:
        try:
            if os.name=='nt':
                subprocess.run(['taskkill','/F','/T','/PID',str(proc.pid)],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                    timeout=1.4,check=False)
            else:
                os.killpg(proc.pid,signal.SIGKILL)
        except (OSError,subprocess.TimeoutExpired,ProcessLookupError,
                PermissionError,AttributeError):
            try:proc.kill()
            except (OSError,ProcessLookupError,AttributeError):pass
    try:proc.wait(timeout=0.55)
    except (subprocess.TimeoutExpired,AttributeError):pass
    return ('','')


def _fast_report_complete(report):
    """Accept only completed reports under their matching scoring protocol.

    v8 basic/deep coverage was previously rejected simply because the
    coordinator only knew the v7 success status. This is a compatibility
    bridge, NOT a relaxation of per-Slot evidence verification.
    """
    if not isinstance(report,dict) or report.get('schema')!='fast_grounded_benchmark.v1':
        return False
    if report.get('scoring_protocol')=='fast_one_shot_dual_slot_coverage.v8':
        v8=report.get('v8_scores')
        if report.get('status')!='complete_dual_slot_review' or not isinstance(v8,dict):
            return False
        count=v8.get('total_slot_count')
        assessed=v8.get('assessed_slot_count')
        basic=v8.get('basic_covered_slot_count')
        deep=v8.get('deep_covered_slot_count')
        return (v8.get('evidence_review_complete') is True
                and type(count) is int and count>0
                and type(assessed) is int and assessed==count
                and type(basic) is int and type(deep) is int
                and 0<=deep<=basic<=count
                and isinstance(report.get('slot_assessments'),list)
                and len(report['slot_assessments'])==count)
    if report.get('scoring_protocol')=='fast_one_shot_required_optional_slot_coverage.v9':
        v9=report.get('v9_scores')
        if report.get('status')!='complete_classified_slot_review' or not isinstance(v9,dict):
            return False
        req_total=v9.get('required_total_slot_count')
        opt_total=v9.get('optional_deepening_total_slot_count')
        req_covered=v9.get('required_covered_slot_count')
        opt_covered=v9.get('optional_deepening_covered_slot_count')
        actual=v9.get('assessed_slot_count')
        total=v9.get('total_slot_count')
        entries=report.get('slot_assessments')
        return (
            v9.get('evidence_review_complete') is True and
            all(type(n) is int for n in
                (req_total,opt_total,req_covered,opt_covered,actual,total)) and
            req_total>0 and opt_total>=0 and total==req_total+opt_total and
            actual==total and 0<=req_covered<=req_total and
            0<=opt_covered<=opt_total and isinstance(entries,list) and
            len(entries)==total and
            sum(x.get('category')=='required' for x in entries if isinstance(x,dict))==req_total and
            sum(x.get('category')=='optional_deepening' for x in entries if isinstance(x,dict))==opt_total)
    # Backward-compatibility for verified earlier strict reports.
    return report.get('status')=='complete_provisional_not_deep_equivalent'


def _worker(job_id,mode,kwargs):
    with LOCK:
        job=JOBS[job_id]
        generation=job.get('generation',1)
    folder=pathlib.Path(job['directory'])
    try:
        child_env=None
        round_count=len(json.loads((folder/'transcript.json').read_text(encoding='utf-8')).get('turns', []))
        # Model calls are bounded individually. A full 30- or 100-round
        # ReportCard is a sequence of potentially many requests; a fixed
        # whole-job deadline incorrectly killed otherwise healthy runs.
        # Progress/checkpoints survive interruptions in the same job directory.
        fast_request_timeout=240  # Fast v7 single evidence-first request, below typical 300s gateway
        # SpokenTruth's standalone call remains bounded by its 300-second
        # request timeout. Do not impose an additional total job wall clock
        # that can terminate otherwise healthy long evaluations.
        if mode=='fast':
            cmd=[sys.executable,str(FAST),'run','--transcript',str(folder/'transcript.json'),
                 '--outline',str(folder/'outline.json'),
                 '--output-dir',str(folder/'fast_result'),
                 '--model',kwargs.get('model','gemma3:4b'),
                 '--budget-seconds','0','--request-timeout',str(fast_request_timeout)]
            if kwargs.get('provider')=='remote':
                if kwargs.get('remote_consent') is not True:
                    raise StrictContractError('远程双引擎调用需要明确授权')
                cmd.extend(['--provider','remote','--base-url',kwargs['base_url'],
                            '--authorize-external-transfer'])
                child_env=_remote_model_environment(kwargs['base_url'])
            timeout=None  # no global deadline; per API request still limited
            result_name='fast_reportcard.json'
            output=folder/'fast_result'/result_name
        elif mode=='spoken':
            cmd=[sys.executable,str(SPOKEN),'run','--memory',str(folder/'memory.json'),
                 '--transcript',str(folder/'transcript.json'),
                 '--output',str(folder/'spoken_result'),
                 '--provider',kwargs.get('provider','ollama'),
                 '--model',kwargs.get('model','qwen3.5:4b'),
                 '--base-url',kwargs.get('base_url','http://127.0.0.1:11434'),
                 '--request-timeout','300']
            if kwargs.get('provider')=='remote':
                if kwargs.get('remote_consent') is not True:
                    raise StrictContractError('必须单独同意向第三方发送全部来源记忆和访谈问题')
                cmd.append('--authorize-external-transfer')
                child_env=_remote_model_environment(kwargs.get('base_url',''))
            timeout=None  # each model request has its own 300s bound
            result_name='spoken_truth_proxy_report.json'
            output=folder/'spoken_result'/result_name
        else:raise StrictContractError('Unknown evaluation mode')
        # Never log API secrets and never send scoring text to the interview API.
        # Cooperative task cancellation: stop the current subprocess, keeping
        # already checkpointed batches and previously completed sibling reports.
        with LOCK:
            if job.get('pause_requested'):
                job['status']='paused'
                return
        proc=subprocess.Popen(cmd,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,env=child_env,
                              **({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP}
                                 if os.name=='nt' else {'start_new_session':True}))
        with LOCK:
            stale=(job.get('generation') != generation)
            paused_on_launch=bool(job.get('pause_requested'))
            if not stale and not paused_on_launch:
                job['process']=proc
        if stale or paused_on_launch:
            # Never terminate a process while holding the global job lock.
            # Also publish 'paused' on this early-exit path: previously the
            # worker returned with status='running' forever.
            _stop_process_tree(proc)
            with LOCK:
                if not stale and job.get('generation')==generation:
                    job['status']='paused'
            return
        deadline=(time.monotonic()+timeout) if timeout is not None else None
        try:
            while True:
                with LOCK:paused=bool(job.get('pause_requested'))
                if paused:
                    _stop_process_tree(proc)
                    with LOCK:
                        if job.get('generation')==generation:job['status']='paused'
                    return
                try:
                    wait_time=0.5 if deadline is None else min(0.5,max(0.01,deadline-time.monotonic()))
                    stdout,stderr=proc.communicate(timeout=wait_time)
                    break
                except subprocess.TimeoutExpired:
                    if deadline is not None and time.monotonic()>=deadline:
                        _stop_process_tree(proc)
                        raise subprocess.TimeoutExpired(cmd,timeout)
            result=subprocess.CompletedProcess(cmd,proc.returncode,stdout,stderr)
        finally:
            with LOCK:
                if job.get('generation')==generation:
                    job.pop('process',None)
        with LOCK:
            if job.get('generation')!=generation:return
            if job.get('pause_requested'):
                job['status']='paused'
                return
        if result.returncode:
            failure=_benchmark_failure_detail(folder,mode,completed=result)
            raise StrictContractError(failure['error'])
        if not output.is_file():
            raise StrictContractError('评分器未产出经过全量校验的 '+result_name)
        report=json.loads(output.read_text(encoding='utf-8'))
        if mode=='fast' and not _fast_report_complete(report):
            raise StrictContractError(
                'Fast-Grounded 结果未满足对应评分协议的完整性要求；'
                '请检查报告的 scoring_protocol、status 与 Slot 证据核验状态')
        if mode=='spoken' and (report.get('completed') is not True or
             report.get('schema') not in ('spoken_truth_all_memory_opportunity.v3',
                                          'spoken_truth_all_memory_opportunity.v3_1')):
            raise StrictContractError('SpokenTruth 未通过全量完成与 Schema 验证')
        with LOCK:
            if job.get('generation')!=generation:return
            if job.get('pause_requested'):
                job['status']='paused'
                return
            job.update({'status':'succeeded','report':report,'pause_requested':False})
    except Exception as exc:
        detail=_benchmark_failure_detail(folder,mode,exc=exc)
        diagnostic=record_failure(job_id,mode,detail['error'],detail.get('failure_stage','unknown'))
        with LOCK:
            if job.get('generation')==generation and not job.get('pause_requested'):
                job.update({'status':'failed',**detail,'diagnostic':diagnostic})
        # Keep failed request outputs in the researcher-only job directory until
        # explicitly exported. Never expose these sensitive responses as static files.

def begin(payload, initial_pause=False):
    run,outline,memory,stats=_inputs(payload)
    mode=payload.get('mode')
    if mode not in ('fast','spoken'):
        raise StrictContractError('请选择 fast 或 spoken')
    if payload.get('local_consent') is not True:
        raise StrictContractError('请明确授权本机评分')
    if payload.get('provider')=='remote':
        if payload.get('remote_consent') is not True:
            raise StrictContractError('远程传输全部记忆和问题必须单独同意')
        _remote_model_environment(payload.get('base_url',''))
    with LOCK:
        running=sum(j['status']=='running' for j in JOBS.values())
        if running>=MAX_JOBS:raise StrictContractError('同时最多两个独立评分任务')
        job_id=uuid.uuid4().hex
        folder=JOB_HOME/job_id
        folder.mkdir(parents=True,exist_ok=False)
        for key,obj in [('transcript',run),('outline',outline),('memory',memory)]:
            (folder/(key+'.json')).write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
        JOBS[job_id]={'status':'running','generation':1,'directory':str(folder),'mode':mode,'stats':stats,
                      'run_settings':{k:payload.get(k) for k in ('provider','remote_consent','base_url','model')},
                      'pause_requested':bool(initial_pause)}
    # A pause arriving during dual-job creation must not launch another API call.
    safe={k:payload.get(k) for k in ('provider','remote_consent','base_url','model')}
    threading.Thread(target=_worker,args=(job_id,mode,safe),daemon=True).start()
    return {'job_id':job_id,'status':'running','stats':stats}

def status(job_id,include_report=False):
    with LOCK:
        job=JOBS.get(job_id)
        if job is None:raise StrictContractError('评分任务不存在，或服务已经重启')
        result={k:v for k,v in job.items() if k in ('status','mode','stats','error','failure_stage','diagnostic')}
        if include_report and result['status']=='succeeded':
            result['report']=copy.deepcopy(job['report'])
        return result

def rejected_responses_archive(dual_id):
    """Export both engines' raw model responses and failure diagnostics in one ZIP.

    Each scoring engine has an independent directory. The archive may be
    exported after both engines stop, including failed evaluations. Never
    include submitted inputs, prompts containing API credentials, identity
    files, or files from the HTTP static directory.
    """
    checked=dual_status(dual_id,include_report=False)
    if checked['status'] in ('running','stopping'):
        raise StrictContractError('两个评分任务仍在运行，请结束后导出原始响应')
    with DUAL_LOCK:
        children=dict(DUAL_JOBS[dual_id]['jobs'])
    manifest={'schema':'dual_raw_model_responses.v1',
              'dual_job_id':dual_id,
              'dual_status':checked['status'],
              'engines':{}}
    max_file_bytes=8*1024*1024
    max_zip_bytes=24*1024*1024

    def safe_name(name):
        # Only known scoring artifacts; no arbitrary output filenames.
        return (name in {'raw_model_response.txt','correction_raw_model_response.txt',
                         'last_failure.json','error.json','events.jsonl',
                         'partial_fast_reportcard.json',
                         'stage1_raw_model_output.json',
                         'reportcard_evidence_audit.json',
                         'field_normalization_audit.json','correction_audit.json'}
                or name.endswith('_raw_model_response.txt')
                or name.endswith('_raw_response.txt'))
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
            for engine in ('fast','spoken'):
                status_info=checked['engines'][engine]
                record={'status':status_info['status'],
                        'failure_stage':status_info.get('failure_stage'),
                        'exported_files':[],
                        'raw_response_files':[]}
                manifest['engines'][engine]=record
                job_id=children.get(engine)
                if not job_id:
                    record['note']='评分子任务未能创建，因此没有模型原始响应'
                    if status_info.get('error'):
                        record['error']=status_info['error']
                    continue
                with LOCK:
                    job=JOBS.get(job_id)
                    folder=pathlib.Path(job['directory']) if job else None
                result_dir=folder/(engine+'_result') if folder is not None else None
                if result_dir is None or not result_dir.is_dir():
                    record['note']='评分目录不存在，可能已经导出审计并清理，或任务在启动前失败'
                    if status_info.get('error'):
                        record['error']=status_info['error']
                    continue
                for item in sorted(result_dir.iterdir(),key=lambda p:p.name):
                    if not item.is_file() or item.is_symlink() or not safe_name(item.name):
                        continue
                    if item.stat().st_size>max_file_bytes:
                        record.setdefault('skipped_oversize',[]).append(item.name)
                        continue
                    archive.write(item,engine+'/'+item.name)
                    record['exported_files'].append(item.name)
                    if item.name.endswith('_raw_model_response.txt') or item.name.endswith('_raw_response.txt'):
                        record['raw_response_files'].append(item.name)
                if not record['raw_response_files']:
                    record['note']='本次任务没有保存可读取的模型原始响应；可能在请求开始前失败或请求超时'
                if status_info.get('error'):
                    record['error']=status_info['error']
            archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            archive.writestr('READ_ME_FIRST.txt',
                '双引擎模型原始响应（研究者诊断用途）\\n'
                'fast/：Fast-Grounded 提问与回答评分的模型原始响应及事件日志。\\n'
                'spoken/：SpokenTruth 原始结果及校正响应、失败诊断（如有）。\\n'
                'manifest.json：每个引擎的运行状态、已导出文件及缺失原因。\\n'
                '某个目录为空或没有 raw_response 不表示模型返回了空 JSON，'
                '可能是请求失败前未能取得模型响应。\\n'
                '原始内容可能包含敏感访谈信息。仅用于研究者私下调试。\\n')
        payload=buf.getvalue()
    if len(payload)>max_zip_bytes:
        raise StrictContractError('原始响应 ZIP 超过24MB，请从本机评分工作目录提取')
    return payload

def full_audit_archive(job_id):
    """Return a reproducible audit archive without destroying task evidence.

    Cleanup must be a separate, explicit action; HTTP downloads are not proof
    that the user received the bytes successfully.
    """
    with LOCK:
        job=JOBS.get(job_id)
        if job is None or job['status']!='succeeded':
            raise StrictContractError('仅可下载已完成且严格校验通过的评分审计')
        folder=pathlib.Path(job['directory'])
        if not folder.is_dir():
            raise StrictContractError('原始审计文件目录不存在，无法生成完整审计')
        with io.BytesIO() as sink:
            with zipfile.ZipFile(sink,'w',zipfile.ZIP_DEFLATED) as z:
                for file in sorted(folder.rglob('*')):
                    if not file.is_file():
                        continue
                    rel=file.relative_to(folder)
                    if any(part in ('__pycache__','.venv') for part in rel.parts):
                        continue
                    if file.stat().st_size>20*1024*1024:
                        raise StrictContractError('审计文件过大，请手动从本地运行目录提取')
                    z.write(file,rel.as_posix())
            payload=sink.getvalue()
        if len(payload)>24*1024*1024:
            raise StrictContractError('审计结果 ZIP 超过24MB，请从本机运行目录取回')
        # Download is read-only and idempotent. The browser may fail after
        # receiving headers; deleting source evidence here makes recovery
        # impossible and breaks the second engine's audit export.
        return payload

# Dual evaluation coordinator. Each scoring algorithm remains the untouched
# standalone executable; this coordinator owns scheduling and reporting only.
DUAL_LOCK=threading.RLock()
DUAL_JOBS={}

def ollama_available_models():
    """Check only the fixed loopback Ollama inventory; no model generation.

    This check prevents a dual job being created when both chosen models
    are missing, the exact failure reproduced in the user's diagnostic ZIP.
    """
    endpoint='http://127.0.0.1:11434/api/tags'
    try:
        with urlopen(endpoint,timeout=3) as response:
            obj=json.load(response)
    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as exc:
        raise StrictContractError(
            '本机 Ollama 未连接或无法查询模型列表。请先运行 ollama serve，'
            '再检查 http://127.0.0.1:11434/api/tags；未启动评分任务。'
        ) from exc
    models=obj.get('models') if isinstance(obj,dict) else None
    if not isinstance(models,list):
        raise StrictContractError('Ollama /api/tags 返回格式不正确，未启动评分任务。')
    return sorted({entry['name'] for entry in models if isinstance(entry,dict)
                   and isinstance(entry.get('name'),str) and entry['name'].strip()})


def preflight_dual_models(payload):
    """Do not start a local scoring job against missing Ollama model tags."""
    if payload.get('evaluation_provider',payload.get('spoken_provider','ollama'))!='ollama':
        return None
    available=ollama_available_models()
    chosen={'Fast-Grounded':str(payload.get('fast_model') or 'gemma3:4b').strip(),
            'SpokenTruth':str(payload.get('spoken_model') or 'qwen3.5:4b').strip()}
    missing={engine:tag for engine,tag in chosen.items() if tag not in available}
    if missing:
        details='；'.join(f'{name}: {tag}' for name,tag in missing.items())
        installed='、'.join(available[:18]) if available else '没有检测到任何模型'
        raise StrictContractError(
            f'本机 Ollama 模型不存在（HTTP 404 原因）：{details}。'
            f'当前已安装：{installed}。'
            '请在评分页面选一个已安装的完整模型名称，'
            '或执行 ollama pull <模型名> 安装所需模型。'
            '未发送任何评分请求。'
        )
    return {'provider':'ollama','models':chosen,'available_count':len(available)}


def begin_dual(payload):
    run,outline,memory,stats=_inputs(payload)
    if payload.get('local_consent') is not True:
        raise StrictContractError('请授权在本机评价完整访谈与来源资料')
    provider=payload.get('evaluation_provider',payload.get('spoken_provider','ollama'))
    if provider not in ('ollama','remote'):
        raise StrictContractError('SpokenTruth 模型服务无效')
    if provider=='ollama':
        preflight_dual_models(payload)
    if provider=='remote':
        if payload.get('remote_consent') is not True:
            raise StrictContractError('远程双引擎评分必须明确同意传输完整采访问题与 Memory')
        _remote_model_environment(payload.get('spoken_base_url',''))
    # Coordinator['status'] is a launch-time marker, not an authoritative
    # terminal state: previously completed jobs remain 'running' in that dict.
    # Use live child states. Otherwise *every second* evaluation in a server
    # lifetime is falsely rejected, even after both engines succeeded.
    with DUAL_LOCK:
        active=[job_id for job_id in DUAL_JOBS
                if dual_status(job_id)['status'] in ('running','stopping')]
        if active:
            raise StrictContractError('已有一场双项评分尚未结束，请等待完成或中断后再开始新任务')
    # Input parsing already completed in _inputs; launch the actual scorers directly.
    dual_id=uuid.uuid4().hex
    with DUAL_LOCK:
        DUAL_JOBS[dual_id]={'status':'running','jobs':{},'stats':stats,
                            'started_at':time.time(),'error':None,'pause_requested':False}
    # Exact independent tasks run in parallel. Do not combine their findings.
    def launch():
        child={}
        errors={}
        specs={
          'fast':{'mode':'fast','provider':provider,
                  'model':payload.get('fast_model') or (
                      'gpt-6.1-sol' if provider=='remote' else 'gemma3:4b'),
                  'base_url':payload.get('spoken_base_url') if provider=='remote' else 'http://127.0.0.1:11434',
                  'remote_consent':payload.get('remote_consent') is True},
          'spoken':{'mode':'spoken','provider':provider,
                    'model':payload.get('spoken_model') or (
                      'qwen3.5-flash' if provider=='remote' else 'qwen3.5:4b'),
                    'base_url':(payload.get('spoken_base_url') if provider=='remote'
                                else 'http://127.0.0.1:11434'),
                    'remote_consent':payload.get('remote_consent') is True}
        }
        for engine,settings in specs.items():
            try:
                with DUAL_LOCK:
                    initial_pause=DUAL_JOBS[dual_id].get('pause_requested',False)
                started=begin({**{'transcript':run,'outline':outline,'memory':memory},
                               'local_consent':True,**settings},initial_pause=initial_pause)
                child[engine]=started['job_id']
                with DUAL_LOCK:
                    DUAL_JOBS[dual_id]['jobs'][engine]=started['job_id']
                    should_pause=DUAL_JOBS[dual_id].get('pause_requested',False)
                    # Synchronize the pause flag before releasing coordinator lock:
                    # a pause during begin() must not get lost.
                    with LOCK:
                        if should_pause:
                            JOBS[started['job_id']]['pause_requested']=True
            except Exception as exc:
                errors[engine]=str(exc)[:900]
                record_failure(dual_id,engine,errors[engine],'launch')
                with DUAL_LOCK:DUAL_JOBS[dual_id]['error']=errors
        with DUAL_LOCK:
            DUAL_JOBS[dual_id]['launch_complete']=True
    threading.Thread(target=launch,daemon=True).start()
    return {'status':'running','job_id':dual_id,'stats':stats,
            'engines':['fast','spoken']}

def pause_dual(dual_id):
    """Publish PAUSED synchronously, send Ctrl+C to each local scorer now.

    This operation does not wait for upstream network calls or polling loops.
    A paused worker can no longer commit a late result; resume increments
    its generation and launches new child scorer processes using the saved
    input/checkpoints.
    """
    with DUAL_LOCK:
        dual=DUAL_JOBS.get(dual_id)
        if dual is None:raise StrictContractError('机评任务不存在或服务已重启')
        current=dual_status(dual_id)
        if current['status'] in ('succeeded','failed','paused'):
            return {'job_id':dual_id,'status':current['status'],
                    'message':'任务已结束或中断'}
        dual['pause_requested']=True
        dual['status']='paused'
        ids=list(dual['jobs'].values())
        processes=[]
        with LOCK:
            for child_id in ids:
                job=JOBS.get(child_id)
                if job and job['status']=='running':
                    job['pause_requested']=True
                    job['status']='paused'
                    proc=job.get('process')
                    if proc is not None and proc.poll() is None:
                        processes.append(proc)
    # Signal both concurrently first. Then wait only a bounded local stop
    # window, never for the network request (which is in a killed process).
    for proc in processes:_signal_process_group(proc)
    for proc in processes:_stop_process_tree(proc,already_signaled=True)
    return {'job_id':dual_id,'status':'paused',
            'message':'机评已中断：已发送 Ctrl+C 等效信号；历史完整结果与检查点保留，可继续机评。'}


def resume_dual(dual_id):
    """Resume an interrupted dual run with all-or-nothing preflight.

    The prior implementation started Fast before checking that SpokenTruth could
    resume. If SpokenTruth was invalid, the API raised an error after Fast had
    already restarted, leaving the UI and dual state inconsistent.
    """
    with DUAL_LOCK:
        dual=DUAL_JOBS.get(dual_id)
        if dual is None:
            raise StrictContractError('续评任务不存在：服务可能已重启，或原任务索引已丢失')
        if not dual.get('launch_complete'):
            raise StrictContractError('评分任务仍在启动中')
        # Do not reject solely because pause_requested is still true:
        # that flag persists until the resumption transaction. Check actual
        # child statuses below; only fully stopped children can resume.
        children=dict(dual['jobs'])
        # Lock order is DUAL_LOCK then LOCK everywhere in this transition.
        with LOCK:
            pending=[]
            for engine in ('fast','spoken'):
                child_id=children.get(engine)
                if not child_id:
                    raise StrictContractError('原评分子任务不存在，无法恢复：'+engine)
                child=JOBS.get(child_id)
                if child is None:
                    raise StrictContractError('评分任务索引已丢失：'+engine)
                if child['status']=='succeeded':
                    continue
                active_proc=child.get('process')
                if active_proc is not None and active_proc.poll() is None:
                    raise StrictContractError('本地评分进程正在终止，请稍后点击继续机评：'+engine)
                if child['status']=='running':
                    raise StrictContractError('该评分器仍在运行，不能重复启动：'+engine)
                if child['status'] not in ('failed','paused'):
                    raise StrictContractError('该评分器状态不允许恢复：'+engine)
                folder=pathlib.Path(child['directory'])
                if not folder.is_dir() or not all(
                    (folder/(name+'.json')).is_file()
                    for name in ('transcript','outline','memory')):
                    raise StrictContractError('原评分输入或检查点目录缺失，无法继续：'+engine)
                settings=copy.deepcopy(child['run_settings'])
                pending.append((child_id,engine,settings))
            # All validations complete: commit both engines before launching any.
            for child_id,engine,settings in pending:
                JOBS[child_id]['generation']=JOBS[child_id].get('generation',1)+1
                JOBS[child_id].update(status='running',error=None,
                    failure_stage=None,diagnostic=None,pause_requested=False)
            dual['pause_requested']=False
            dual['status']='running' if pending else 'succeeded'
    # Threads launch after the state transition, never while it is half-complete.
    for child_id,engine,settings in pending:
        threading.Thread(target=_worker,args=(child_id,engine,settings),daemon=True).start()
    return {'job_id':dual_id,'status':dual['status'],
            'resumed_engines':[engine for _,engine,_ in pending],
            'message':'未修改已完成评分；待恢复子任务已在启动前统一检查，Fast 检查点将尝试复用'}

def dual_status(dual_id,include_report=False):
    # Use one ordered snapshot of coordinator and child state. The previous
    # implementation released DUAL_LOCK while collecting children and then
    # wrote a stale 'running'/'paused' status over a newer resume transition.
    with DUAL_LOCK:
        job=DUAL_JOBS.get(dual_id)
        if job is None:raise StrictContractError('双项评分任务不存在，可能已经重启')
        children=dict(job['jobs'])
        launch_done=job.get('launch_complete',False)
        launch_errors=copy.deepcopy(job.get('error'))
        stats=copy.deepcopy(job['stats'])
        pause_requested=bool(job.get('pause_requested'))
        reports={}
        stages={}
        fast_events_path=None
        with LOCK:
            for engine in ('fast','spoken'):
                if engine not in children:
                    stages[engine]={'status':'failed' if launch_done else 'pending',
                                    'error':(launch_errors or {}).get(engine)}
                else:
                    child=JOBS.get(children[engine])
                    if child is None:
                        stages[engine]={'status':'failed','error':'子任务索引丢失'}
                    else:
                        stages[engine]={k:copy.deepcopy(child[k]) for k in
                                        ('status','error','failure_stage','diagnostic') if k in child}
                        if engine=='fast':
                            fast_events_path=pathlib.Path(child['directory'])/'fast_result'/'events.jsonl'
                        if child['status']=='succeeded' and include_report:
                            reports[engine]=copy.deepcopy(child.get('report'))
    # Read only numeric progress metadata; never expose prompts or responses.
    if fast_events_path is not None and fast_events_path.is_file():
        try:
            known=set()
            with fast_events_path.open('r',encoding='utf-8') as stream:
                for line in stream:
                    if 'batch_scores_validated' not in line:continue
                    try:record=json.loads(line)
                    except ValueError:continue  # concurrent writer's partial line
                    if record.get('event')!='batch_scores_validated':continue
                    for n in record.get('round_ids',[]):
                        if type(n) is int and 1<=n<=stats['rounds']:known.add(n)
            stages['fast']['progress']={
                'scored_rounds':len(known),'total_rounds':stats['rounds'],
                'percent':round(100*len(known)/max(1,stats['rounds']),1)}
        except (OSError,ValueError,TypeError,KeyError):
            pass
    all_terminal=launch_done and all(stages[x]['status'] in ('succeeded','failed','paused') for x in stages)
    finished=all_terminal and all(stages[x]['status']=='succeeded' for x in stages)
    phase=('succeeded' if finished else
           'paused' if all_terminal and any(v['status']=='paused' for v in stages.values()) else
           'failed' if all_terminal else 'stopping' if pause_requested else 'running')
    result={'job_id':dual_id,'status':phase,'stats':stats,'engines':stages,
            'complete_both':finished,'model_fact_judgment':{
             'fast':'Interviewer-only ReportCard: questions, observed respondent answers and outline; no memory',
             'spoken':'LLM per-memory question opportunity DIRECT/ADJACENT/NONE'}}
    if include_report:
        result['reports']=reports
    # This computed response is a snapshot. Never publish it as the controlling
    # state: a concurrent pause/resume could have happened after the snapshot.
    return result

def dual_audit_archive(dual_id):
    checked=dual_status(dual_id,include_report=False)
    if checked['status']!='succeeded':
        raise StrictContractError('双项评分未全部成功，不允许宣称完整双项审计')
    with DUAL_LOCK:children=dict(DUAL_JOBS[dual_id]['jobs'])
    # Produce both independently validated audit packages; never overwrite one
    # engine's raw evidence with the other's output.
    audits={}
    for engine in ('fast','spoken'):
        audits[engine]=full_audit_archive(children[engine])
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            for engine,payload in audits.items():
                z.writestr(engine+'_complete_audit.zip',payload)
            z.writestr('dual_manifest.json',json.dumps({
                'schema':'playground_dual_benchmark_audit.v1',
                'dual_job_id':dual_id,
                'algorithms':['Fast-Grounded original','SpokenTruth original'],
                'both_completed':True,
                'child_job_ids':children},ensure_ascii=False,indent=2))
        return buf.getvalue()



def dual_partial_fast_report(dual_id):
    """Allow researcher-only inspection of a failed/paused Fast report.

    Never guess a filesystem path from a browser-supplied identifier.
    Resolve strictly through the private job registry, with a size cap.
    """
    if not isinstance(dual_id,str) or not re.fullmatch(r'[0-9a-f]{32}',dual_id):
        raise StrictContractError('Invalid dual evaluation ID')
    with DUAL_LOCK:
        dual=DUAL_JOBS.get(dual_id)
        if not dual:
            raise StrictContractError('Dual evaluation does not exist')
        fast_id=(dual.get('jobs') or {}).get('fast')
        with LOCK:
            job=JOBS.get(fast_id)
            if not job:
                return {'ok':True,'available':False}
            if job['status']=='running':
                return {'ok':True,'available':False,'state':'running'}
            directory=pathlib.Path(job['directory'])
    file=directory/'fast_result'/'partial_fast_reportcard.json'
    if not file.is_file() or file.stat().st_size>8*1024*1024:
        return {'ok':True,'available':False}
    obj=json.loads(file.read_text('utf8'))
    if obj.get('schema')!='fast_grounded_benchmark.v1':
        raise StrictContractError('Invalid stored Fast partial report')
    return {'ok':True,'available':True,'report':obj}

def dual_error_records(dual_id):
    """Only records from this dual run; excludes raw prompts and secrets."""
    with DUAL_LOCK:
        dual=DUAL_JOBS.get(dual_id)
        if not dual:raise StrictContractError('任务不存在或服务已经重启')
        child_ids=list(dual['jobs'].values())
    return {'schema':'playground_benchmark_errors.v1','job_id':dual_id,
            'records':read_error_records([dual_id,*child_ids])}
