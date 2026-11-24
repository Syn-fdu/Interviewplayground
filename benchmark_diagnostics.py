"""Privacy-conscious, unified benchmark error diagnostics. No source text or secrets."""
from __future__ import annotations
import datetime, json, os, pathlib, re, threading, uuid

LOCK=threading.RLock()
LOG_DIR=pathlib.Path(__file__).resolve().parent/'prepared_memory'/'benchmark_error_logs'

def sanitize(message):
    text=str(message or '')
    text=re.sub(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+',r'\1[REDACTED]',text)
    text=re.sub(r'(?i)((?:api[_ -]?key|access[_ -]?token|password|secret)\s*[:=]\s*)\S+',r'\1[REDACTED]',text)
    text=re.sub(r'(?i)(https?://)[^\s/]+/[^\s]*',r'\1[API_ENDPOINT_REDACTED]',text)
    # Do not persist prompts, responses, tracebacks or arbitrary long exception bodies.
    return text.replace('\r',' ').replace('\n',' ')[:320]

def diagnose(message, stage='unknown', exception=None):
    s=str(message or '')
    low=s.lower()
    http=re.search(r'(?:http[^0-9]{0,12}|upstream_http_)(\d{3})',low)
    code=int(http.group(1)) if http else None
    if ('content_filter' in low or 'content management policy' in low):
        kind='content_policy';title='请求被内容审核拦截';why='服务商拒绝了当前请求内容，不是网络掉线。';how='查看服务商的合规要求，定位被拒绝的材料；不要盲目重试或绕过过滤。';retry=False
    elif (code==429 or 'rate limit' in low or 'too many requests' in low):
        kind='rate_limit';title='API 请求频率受限';why='当前服务商限制调用速度或配额。';how='检查服务商配额、并发数与限流提示，降低请求频率后再继续。';retry=False
    elif (code==404 and ('ollama' in low or 'model' in low or 'not found' in low)):
        kind='model_unavailable';title='本机评分模型未安装或模型接口不存在（HTTP 404）'
        why='Ollama 无法找到所请求的模型名或 API 路径；模型推理并未开始。'
        how='先在评分页面选择已安装的模型；使用 ollama list 查看完整模型名称，必要时执行 ollama pull <模型名>。'
        retry=False
    elif (code in (401,403) or 'invalid api key' in low or 'unauthorized' in low):
        kind='authorization';title='API 授权失败';why='密钥、权限或对应的接口地址不正确。';how='核对密钥与服务商地址，确认账户可以使用该模型。';retry=False
    elif any(x in low for x in ('time budget exhausted','subprocess timed out','timed out after','评分超时','时间预算耗尽')):
        kind='execution_timeout';title='评分执行时间不足';why='任务已超过配置的机评运行时限，不一定是网络故障。';how='保留已完成的批次检查点，核对模型响应速度和时间配置后继续。';retry=False
    elif (code in (524,504,408) or any(x in low for x in ('gateway timeout','upstream timeout'))):
        kind='upstream_gateway_timeout';title='模型服务商网关超时（HTTP 524/504）'
        why='上游网关先于客户端等待期限结束请求。通常涉及模型推理、排队或网关时间限制；不能说明电脑断网，也不能靠增加本地超时解决。'
        how='Fast 会将本次失败批次自动拆小重试，保留此前成功检查点；若单轮仍失败，请核对服务商网关限制与模型处理时间。'
        retry=True
    elif (code==408 or any(x in low for x in ('timed out','timeout','winerror 10060','read timeout','response timeout'))):
        kind='request_timeout';title='模型/API 响应超时（不是断网证明）';why='评分请求在设定的等待时间内未收到完整响应。可能是模型推理过慢、请求排队、服务端负载或传输超时，不能仅凭这个错误认定电脑网络中断。';how='核对失败批次、单次请求耗时和模型供应商的超时日志，必要时延长单请求等待并减少并发；继续机评会优先复用已保存的批次。';retry=False
    elif any(x in low for x in (
            'remote end closed connection without response','remotedisconnected',
            'connection prematurely closed',
            'connection reset','connection aborted','broken pipe','winerror 10054')):
        kind='remote_disconnect';title='远程模型连接被中途关闭'
        why='上游服务在返回 HTTP 响应之前关闭了连接。不能据此认定本机断网；短时间内可由代理连接复用、服务端重启或网关负载引起。'
        how='Fast 将对该批次执行一次有限重试，持续失败则缩小到两轮或单轮；已完成的批次保留。'
        retry=True
    elif (code in (502,503,504) or any(x in low for x in ('winerror 10054','connection reset','connection refused','connection aborted','network is unreachable','网络不稳定','failed to fetch'))):
        kind='connection_or_upstream';title='连接或上游服务异常';why='请求连接或上游服务返回失败；不等于本机互联网连接中断。';how='查看服务商状态、代理日志与具体 HTTP 状态，排除上游拥塞/重启后再续评。';retry=True
    elif any(x in low for x in ('model output truncated','finish_reason=length','finish_reason=max_tokens',
                                 'context length','context window','maximum context','token limit',
                                 'too many tokens','num_ctx','exceeds the context')):
        kind='model_capacity';title='模型上下文或输出长度不足';why='SpokenTruth 一次提交全部来源记忆和访谈问题，模型输入/输出容量可能不足；这不是本地网络断线。';how='核对实际 token 限制、输出截断提示与模型服务端日志；保留原始响应，避免误当完整评分。';retry=False
    elif 'fast v7: slot evidence incomplete or suspicious' in low:
        kind='slot_evidence_review';title='Fast Slot 证据核验未通过'
        why='模型对部分必需 Slot 未逐项作答、给出了无法核实的引文，或出现可疑的机械式覆盖判断。'
        how='已将详情保存到 partial_fast_reportcard.json；查看逐 Slot 状态与原始响应，不要将未核实结果当作完整覆盖率。'
        retry=False
    elif any(x in low for x in ('review must','results count','wrong number of results',
                                 'question_index','invalid outcome','memory index',
                                 'schema validation','must contain',
                                 'cannot targeted-repair','full result array is missing or incomplete',
                                 'spokentruth incomplete after targeted repairs')):
        kind='spoken_output';title='SpokenTruth 模型评分输出未通过结构校验';why='模型返回的逐记忆评价可能缺项、数量不符或字段异常。';how='检查 last_failure.json 与 raw_model_response.txt，优先恢复已有响应而非重复整批计费。';retry=False
    elif any(x in low for x in ('no such model','model not found','model unavailable','model does not exist','模型不存在','未安装模型','ollama 未就绪')):
        kind='model_unavailable';title='评分模型暂不可用';why='选定的模型未安装、未启动，或服务商没有开放该模型。';how='检查模型名称、Ollama 服务和远程模型权限；确认模型可正常调用后继续。';retry=False
    elif any(x in low for x in ('insufficient_quota','quota exceeded','billing','余额不足','配额不足')):
        kind='quota';title='API 额度不足';why='账户额度或计费配置无法支持本次请求。';how='检查服务商余额、配额或计费设置，恢复后再继续。';retry=False
    elif any(x in low for x in ('scenario_id mismatch','revision mismatch','fingerprint','sha256 不一致','场景 id 不一致','修订版本不一致','同一来源','来源不一致')):
        kind='source_mismatch';title='评分材料来源不一致';why='Transcript、Outline 或 Memory 的场景、版本、来源校验未通过。';how='重新从同一场访谈导出三份文件；不要拼接不同场景的数据。';retry=False
    elif any(x in low for x in ('bad zip','zip file','zip 包','压缩文件','unsupported file','invalid jsonl','文件不是完整','jsonl')):
        kind='file_import';title='导入文件无法读取';why='文件格式不正确、包不完整，或导入内容不符合预期。';how='核对 ZIP/JSONL 格式，重新导出完整文件，不要手动删减原文。';retry=False
    elif any(x in low for x in ('missing report','未产出','缺少评分结果','未获得可用分数','model did not return visualizable')):
        kind='missing_output';title='评分器没有产生可用结果';why='模型已调用，但返回内容不足以形成可信分数。';how='下载原始响应与错误记录，检查模型输出是否完整；不能补造缺失的评分。';retry=False
    elif any(x in low for x in ('schema','rationale','json decode','not valid json','invalid json','结构化','字段缺失','结构错误')):
        kind='schema';title='模型输出格式不符合要求';why='模型返回的 JSON 或字段不符合评分器所需格式。';how='检查原始响应与 Schema，确认模型支持结构化输出；不能填补虚构分数。';retry=False
    elif any(x in low for x in ('exceeded','too large','过大','轮次不连续','缺少','不一致','input','文件不是')):
        kind='input';title='评分输入未通过校验';why='导入数据缺项、版本不一致，或超过评分器限制。';how='核对 Transcript、Outline、Memory 是否来自同一场景，按错误字段修复输入。';retry=False
    else:
        kind='internal';title='评分任务未能完成';why='程序遇到未归类的异常，当前不能可靠判断是哪一种故障。';how='下载诊断记录核对失败阶段；不要删除任务目录或伪造分数。';retry=False
    return {'code':kind,'title':title,'explanation':why,'action':how,'network_retry_allowed':retry,'stage':str(stage or 'unknown')[:60]}

def record(job_id,engine,message,stage='unknown', exception=None):
    data=diagnose(message,stage,exception)
    # Store a short safe signature only, not untrusted model response / personal source.
    # Strict allowlist: never write arbitrary exception text, which may include
    # private interview content echoed by a provider.
    msg=str(message or '')
    indicators=re.findall(r'(?i)(?:WinError\s*\d+|HTTP\s*\d{3}|upstream_http_\d{3}|content_filter|schema mismatch|connection timed out|timeout|rate limit|remote end closed connection without response|connection reset by peer|broken pipe)',msg)
    raw='; '.join(indicators[:6])
    if not raw and exception is not None:
        raw=type(exception).__name__
    if not raw:
        raw='unavailable'
    event={'id':uuid.uuid4().hex,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'job_id':str(job_id),'engine':str(engine),'stage':data['stage'],
           **data,'technical_summary':raw}
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    with LOCK:
        with (LOG_DIR/'errors.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,ensure_ascii=False)+'\n')
        target=LOG_DIR/(str(job_id)+'.jsonl')
        with target.open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,ensure_ascii=False)+'\n')
    return event

def read_job(job_ids):
    result=[]
    with LOCK:
        for job_id in job_ids:
            if not re.fullmatch('[0-9a-f]{32}',str(job_id)):continue
            source=LOG_DIR/(job_id+'.jsonl')
            if source.is_file():
                for line in source.read_text(encoding='utf-8').splitlines():
                    try:result.append(json.loads(line))
                    except ValueError:continue
    return sorted(result,key=lambda x:x.get('time_utc',''))
