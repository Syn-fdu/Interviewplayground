/* Batch experiments UI: presentation layer only; all execution is server-owned. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const root = $('batch');
  if (!root) return;

  const status = $('batchStatus');
  const dataset = $('batchDataset');
  const providerSelect = role => $(role === 'interviewer' ? 'batchInterviewer' : 'batchRespondent');
  const prefix = role => role === 'interviewer' ? 'batchInterviewer' : 'batchRespondent';
  const singleHistoryIds = ['batchHistory'];
  const traitNames = ['knowledge', 'understanding', 'reflexivity', 'memory', 'verbosity', 'disclosure'];
  const phaseNames = {
    queued: '等待安排',
    resume_queued: '准备恢复已保存轮次',
    scenario_setup: '构建受访者场景',
    evidence_preparation: '准备证据索引',
    interviewer_preparation: '采访者预处理',
    respondent_preparation: '受访者记忆准备',
    interviewer_question: '采访者生成问题',
    respondent_answer: '受访者生成回答',
    turn_saved: '本轮已完成',
    saved: '访谈结果已保存',
    interrupt_requested: '实验已中断',
    interrupted: '实验已中断',
    finished: '批次执行结束'
  };

  let current = '';
  let historySelection = '';
  let lastSnapshot = null;
  let stopPending = false;
  let datasetNames = {};
  let experimentHistory = [];
  let batchDefaults = null;
  let defaultsLoaded = false;

  const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]
  ));
  const localTime = iso => iso ? new Date(iso).toLocaleString() : '—';
  const seconds = n => n != null && Number.isFinite(Number(n)) ? `${Number(n).toFixed(1)} 秒` : '—';

  async function api(path, payload) {
    const options = payload === undefined ? {cache: 'no-store'} : {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Playground-Local': 'intake-v2'},
      body: JSON.stringify({...payload, local_consent: true})
    };
    const resp = await fetch(path, options);
    const value = await resp.json();
    if (!resp.ok) throw Error(value.error || `HTTP ${resp.status}`);
    return value;
  }

  function message(text, bad=false) {
    status.textContent = text;
    status.style.color = bad ? '#9c4235' : '#245b4b';
  }

  function readInteger(id, min, max) {
    const node = $(id);
    const value = Number(node.value);
    if (!node.value.trim() || !Number.isInteger(value) || value < min || value > max) {
      node.focus();
      throw Error(`${node.closest('label')?.textContent?.trim() || id} 必须是 ${min}–${max} 的整数`);
    }
    return value;
  }

  function batchTraits() {
    return Object.fromEntries(
      traitNames.map(name => [name, Number(root.querySelector(`[data-batch-trait="${name}"]`).value)])
    );
  }

  function updateTraitLabels() {
    const values = batchTraits();
    for (const key of traitNames) {
      $('batchTraitValue-' + key).textContent = values[key].toFixed(1);
    }
  }
  root.querySelectorAll('[data-batch-trait]').forEach(node => node.addEventListener('input', updateTraitLabels));

  const batchDrafts = {interviewer: {}, respondent: {}};

  function readBatchForm(role) {
    const id = prefix(role);
    const provider = providerSelect(role).value;
    const previous = batchDrafts[role][provider] || {};
    const model = $(id + 'Model').value.trim();
    const base_url = $(id + 'Url').value.trim();
    const api_key = $(id + 'Key').value.trim() || (previous.base_url === base_url ? (previous.api_key || '') : '');
    return {provider, model, base_url, api_key};
  }

  function storeBatchForm(role) {
    const profile = readBatchForm(role);
    batchDrafts[role][profile.provider] = profile;
    $(prefix(role) + 'ConfigStatus').textContent = '已保存到当前浏览器会话；不会写入实验文件';
    return profile;
  }

  function showBatchProfile(role) {
    const id = prefix(role);
    const provider = providerSelect(role).value;
    const profile = batchDrafts[role][provider] || batchDefaults?.[role]?.[provider] || {};
    $(id + 'Model').value = profile.model || '';
    $(id + 'Url').value = profile.base_url || '';
    $(id + 'Key').value = '';
    $(id + 'Url').disabled = provider === 'local';
    $(id + 'Key').disabled = provider === 'local';
    $(id + 'ConfigStatus').textContent = provider === 'local'
      ? '仅访问本机 Ollama；批量模型与单次访谈分开配置'
      : '批量专属配置 · Key 不会进入浏览器存储或实验文件';
  }

  async function loadBatchDefaults() {
    if (defaultsLoaded) return;
    const data = await api('/api/batch/model-defaults');
    batchDefaults = data.profiles;
    defaultsLoaded = true;
    for (const role of ['interviewer', 'respondent']) showBatchProfile(role);
  }

  function allBatchProfiles() {
    return Object.fromEntries(['interviewer', 'respondent'].map(role => [role, readBatchForm(role)]));
  }

  function saveBatchPreferences() {
    try {
      const values = {
        batchDataset: $('batchDataset')?.value || '',
        batchScenario: $('batchScenario')?.value || '',
        batchGuideLevel: $('batchGuideLevel')?.value || '',
        batchGuideCategory: $('batchGuideCategory')?.value || '',
        batchInterviewer: $('batchInterviewer')?.value || '',
        batchRespondent: $('batchRespondent')?.value || '',
        batchLanguage: $('batchLanguage')?.value || '',
        batchPreparation: $('batchPreparation')?.value || '',
        batchMaxTurns: $('batchMaxTurns')?.value || '',
        batchInterviewerTimeout: $('batchInterviewerTimeout')?.value || '',
        batchRespondentTimeout: $('batchRespondentTimeout')?.value || '',
        batchRetrievalTimeout: $('batchRetrievalTimeout')?.value || '',
        batchQualityMode: $('batchQualityMode')?.checked ?? true,
        batchConsent: $('batchConsent')?.checked ?? false,
        traits: batchTraits()
      };
      sessionStorage.setItem('interviewplayground.batch.parameters.v4', JSON.stringify(values));
    } catch (_err) {}
  }

  function restoreBatchPreferences() {
    try {
      const saved = JSON.parse(sessionStorage.getItem('interviewplayground.batch.parameters.v4') || 'null');
      if (!saved) return;
      const boolKeys = new Set(['batchQualityMode', 'batchConsent']);
      for (const [key, value] of Object.entries(saved)) {
        if (key === 'traits') continue;
        const node = $(key);
        if (!node) continue;
        if (boolKeys.has(key)) {
          node.checked = !!value;
          continue;
        }
        if (node.options && [...node.options].length && ![...node.options].some(option => option.value === value)) continue;
        node.value = value;
      }
      for (const [trait, value] of Object.entries(saved.traits || {})) {
        const node = root.querySelector(`[data-batch-trait="${trait}"]`);
        if (node && Number.isFinite(value)) node.value = value;
      }
      updateTraitLabels();
    } catch (_err) {}
  }

  root.addEventListener('change', event => {
    if (!event.target.matches('input[type="password"], input[type="url"], #batchInterviewerModel, #batchRespondentModel')) {
      saveBatchPreferences();
    }
  });

  for (const role of ['interviewer', 'respondent']) {
    providerSelect(role).addEventListener('change', () => {
      showBatchProfile(role);
      saveBatchPreferences();
    });
    const id = prefix(role);
    $(id + 'Apply').addEventListener('click', () => {
      try {
        storeBatchForm(role);
        message(`已保存${role === 'interviewer' ? '采访者' : '受访者'}配置到当前浏览器会话`);
      } catch (err) {
        message(err.message, true);
      }
    });
    $(id + 'Test').addEventListener('click', async () => {
      try {
        storeBatchForm(role);
        message(`正在测试${role === 'interviewer' ? '采访者' : '受访者'}连接（仅发送虚构内容）…`);
        const response = await api('/api/batch/model-test', {role, model_profiles: allBatchProfiles()});
        $(id + 'ConfigStatus').textContent = `连接成功 · ${response.model} · ${response.preview}`;
        message('批量连接测试成功');
      } catch (err) {
        $(id + 'ConfigStatus').textContent = err.message;
        message(err.message, true);
      }
    });
  }

  function clearCurrentPanel() {
    lastSnapshot = null;
    $('batchProgress').value = 0;
    $('batchProgressText').textContent = '0/0 · 0%';
    $('batchCounts').textContent = '完成 0 · 失败 0 · 未完成 0 · 中断 0 · 不可执行 0 · 等待 0 · 运行中 0';
    $('batchHistoryConfigStatus').textContent = experimentHistory.length
      ? '请选择一个实验以查看当前情况。'
      : '暂无历史实验；请先开始一次新实验。';
    $('batchPhase').textContent = '—';
    $('batchPhaseDetail').textContent = '—';
    $('batchCurrentCandidate').textContent = '—';
    $('batchPhaseElapsed').textContent = '—';
    $('batchLastTurn').textContent = '—';
    $('batchLastTurnAt').textContent = '—';
    $('batchLastInterviewer').textContent = '—';
    $('batchLastRespondent').textContent = '—';
    $('batchRows').innerHTML = '';
    $('batchExport').disabled = true;
    $('batchStop').disabled = true;
    $('batchResume').disabled = true;
    $('batchStart').disabled = true;
    $('batchHistoryResume').disabled = !historySelection;
  }

  function currentSummary() {
    return experimentHistory.find(row => row.experiment_id === current) || null;
  }

  function canResume(summary) {
    if (!summary) return false;
    if (summary.state === 'running' || summary.state === 'stopping' || summary.state === 'created') return false;
    const counts = summary.counts || {};
    return summary.state !== 'completed' || !!counts.failed || !!counts.incomplete || !!counts.interrupted;
  }

  function syncSingleHistorySelection(value, sourceId='') {
    const changingExperiment = current !== (value || '');
    current = value || '';
    if (changingExperiment) {
      $('batchHistoryInterviewerKey').value = '';
      $('batchHistoryRespondentKey').value = '';
    }
    for (const id of singleHistoryIds) {
      const node = $(id);
      if (!node) continue;
      if ([...node.options].some(option => option.value === current)) node.value = current;
      else if (!current) node.value = '';
    }
    const selected = currentSummary();
    $('batchExport').disabled = !selected;
    $('batchHistoryExport').disabled = !historySelection;
    $('batchRestoreConfig').disabled = !historySelection;
    const resumable = canResume(selected);
    $('batchResume').disabled = !resumable;
    $('batchHistoryResume').disabled = !historySelection;
    if (selected) {
      const datasetLabel = datasetNames[selected.dataset] || selected.dataset;
      const options = selected.options || {};
      $('batchHistoryConfigStatus').textContent =
        `数据集：${datasetLabel} · 创建于 ${localTime(selected.created_at)} · 状态 ${selected.state} · 访谈语言 ${options.language || '—'} · 等级 ${(options.guide_level || 'level1')} · 最多 ${options.max_turns ?? '—'} 轮`;
    } else {
      $('batchHistoryConfigStatus').textContent = experimentHistory.length
        ? '请选择一个实验以查看当前情况。'
        : '暂无历史实验；请先开始一次新实验。';
    }
    if (!current) clearCurrentPanel();
    else if (sourceId !== 'render') refresh();
  }

  function selectedHistoryIds() {
    return [...root.querySelectorAll('input[name="batchHistoryPick"]:checked')].map(node => node.value);
  }
  function syncHistorySelection() {
    const selected = selectedHistoryIds();
    historySelection = selected.length === 1 ? selected[0] : '';
    const exactlyOne = selected.length === 1;
    const containsActive = selected.some(id => {
      const row = experimentHistory.find(record => record.experiment_id === id);
      return row && ['running','stopping'].includes(row.state);
    });
    $('batchHistoryResume').disabled = !exactlyOne;
    $('batchHistoryExport').disabled = !exactlyOne;
    $('batchRestoreConfig').disabled = !exactlyOne;
    $('batchDelete').disabled = selected.length === 0 || containsActive;
    for (const row of root.querySelectorAll('#batchHistoryRows tr[data-job]')) {
      row.classList.toggle('batch-v2-selected', selected.includes(row.dataset.job));
    }
  }
  function renderHistoryLists() {
    const selectedBefore = current && experimentHistory.some(row => row.experiment_id === current)
      ? current : '';
    const previousSelected = new Set(selectedHistoryIds());
    const currentSelect = $('batchHistory');
    currentSelect.replaceChildren();
    currentSelect.add(new Option(experimentHistory.length ? '选择当前实验' : '暂无历史实验', ''));
    for (const row of experimentHistory) {
      const name = datasetNames[row.dataset] || row.dataset;
      currentSelect.add(new Option(`${name} · ${new Date(row.created_at).toLocaleString()} · ${row.state}`, row.experiment_id));
    }
    $('batchHistoryRows').innerHTML = experimentHistory.map(row => {
      const id = escape(row.experiment_id);
      const label = escape(datasetNames[row.dataset] || row.dataset);
      const date = escape(localTime(row.created_at));
      const done = Number(row.counts?.completed || 0);
      const checked = previousSelected.has(row.experiment_id) ? 'checked' : '';
      return `<tr data-job="${id}">
        <td><input type="checkbox" name="batchHistoryPick" value="${id}" aria-label="选择 ${label}" ${checked}></td>
        <td><div class="batch-history-name">${label}</div><div class="batch-history-date">${date} · ${id.slice(0,8)}</div></td>
        <td>${escape(row.state)}</td><td>${done} / ${Number(row.total || 0)}</td>
      </tr>`;
    }).join('') || '<tr><td colspan="4">暂无历史实验</td></tr>';
    syncHistorySelection();
    syncSingleHistorySelection(selectedBefore, 'render');
    if (current && selectedBefore) refresh();
  }

  function updateElapsed() {
    const progress = lastSnapshot?.progress || {};
    const phaseStart = progress.phase_started_at ? new Date(progress.phase_started_at).getTime() : NaN;
    $('batchPhaseElapsed').textContent = Number.isFinite(phaseStart) && lastSnapshot &&
      ['running', 'stopping'].includes(lastSnapshot.state)
      ? seconds(Math.max(0, (Date.now() - phaseStart) / 1000))
      : '—';
  }

  function render(value) {
    if (!value) return;
    if (stopPending && value.state === 'running') return;
    current = value.experiment_id;
    lastSnapshot = value;
    if (value.state !== 'running') stopPending = false;

    const counts = value.counts || {};
    const done = (counts.completed || 0) + (counts.failed || 0) + (counts.skipped || 0) + (counts.incomplete || 0) + (counts.interrupted || 0);
    const total = value.total || 0;
    const percent = total ? Math.round(100 * done / total) : 0;
    const progress = value.progress || {};

    $('batchProgress').value = percent;
    $('batchProgressText').textContent = `${done}/${total} · ${percent}%`;
    $('batchCounts').textContent =
      `完成 ${counts.completed || 0} · 失败 ${counts.failed || 0} · 未完成 ${counts.incomplete || 0} · 中断 ${counts.interrupted || 0} · 不可执行 ${counts.skipped || 0} · 等待 ${counts.pending || 0} · 运行中 ${counts.running || 0}`;
    $('batchPhase').textContent = value.state === 'created'
      ? '等待开始' : (phaseNames[progress.phase] || value.state || '—');
    if (!stopPending) {
      status.textContent = ({
        created: '待开始', running: '运行中', stopping: '中断中',
        interrupted: '已中断', completed: '已完成',
        finished_with_errors: '存在错误'
      })[value.state] || value.state || '—';
    }
    $('batchPhaseDetail').textContent = progress.phase
      ? `阶段：${progress.phase}${progress.resume_from_turn ? ` · 从第 ${progress.resume_from_turn} 轮继续` : ''}`
      : '—';
    $('batchCurrentCandidate').textContent = progress.candidate_id || '—';
    $('batchLastTurn').textContent = seconds(progress.last_turn_seconds);
    $('batchLastTurnAt').textContent = progress.last_turn_at ? `记录时间：${localTime(progress.last_turn_at)}` : '尚无上一轮记录';
    $('batchLastInterviewer').textContent = seconds(progress.last_interviewer_seconds);
    $('batchLastRespondent').textContent = seconds(progress.last_respondent_seconds);
    updateElapsed();

    const rows = (value.interviews || []).map(row => `
      <tr>
        <td>${escape(row.candidate_id)}</td>
        <td>${escape(row.status || '')}</td>
        <td>${Number(row.turn_count || 0)}</td>
        <td>${row.last_turn_seconds == null ? '—' : seconds(row.last_turn_seconds)}</td>
        <td>${row.last_interviewer_seconds == null ? '—' : seconds(row.last_interviewer_seconds)}</td>
        <td>${row.last_respondent_seconds == null ? '—' : seconds(row.last_respondent_seconds)}</td>
        <td>${escape(row.error || '')}</td>
      </tr>`).join('');
    $('batchRows').innerHTML = rows || '<tr><td colspan="7">暂无轮次数据</td></tr>';

    const running = value.state === 'running' || value.state === 'stopping';
    $('batchStop').disabled = !running || value.state === 'stopping';
    $('batchStart').disabled = value.state !== 'created';
    $('batchResume').disabled = !canResume(value);
    $('batchExport').disabled = !current;

    const hint = progress.phase === 'resume_queued'
      ? `正在准备恢复 ${progress.candidate_id || '当前受访者'} 的已保存轮次；继续前会重建场景和模型状态，但不会清空历史轮次。`
      : progress.phase === 'interviewer_preparation'
        ? '采访者正在根据研究大纲做预处理；这个阶段可能较久。'
        : progress.phase === 'respondent_preparation'
          ? '正在准备受访者事实与记忆索引；如果慢，请观察当前步骤等待时间。'
          : progress.phase === 'interviewer_question'
            ? '当前瓶颈更可能出现在采访者问题生成。'
            : progress.phase === 'respondent_answer'
              ? '受访者正在执行检索、回答生成及质量改写。'
              : '新实验与历史实验共用这个当前视图；列表切换不会改动实验数据。';
    $('batchHint').textContent = hint;

    const datasetLabel = datasetNames[value.dataset] || value.dataset;
    const options = value.options || {};
    $('batchHistoryConfigStatus').textContent =
      `数据集：${datasetLabel} · 创建于 ${localTime(value.created_at)} · 状态 ${value.state} · 语言 ${options.language || '—'} · 记忆准备 ${options.preparation_mode || '—'} · Level ${options.guide_level || '1'} · 最多 ${options.max_turns ?? '—'} 轮`;

    syncSingleHistorySelection(value.experiment_id, 'render');
  }

  async function refreshDatasets() {
    const response = await api('/api/batch/datasets');
    const previous = dataset.value;
    datasetNames = Object.fromEntries((response.datasets || []).map(row => [row.id, row.label]));
    dataset.replaceChildren(new Option('请选择数据集', ''));
    for (const row of response.datasets || []) {
      dataset.add(new Option(`${row.label}（总计 ${row.total}，可执行 ${row.eligible}）`, row.id));
    }
    if ([...dataset.options].some(option => option.value === previous)) dataset.value = previous;
    else if (!dataset.value && response.datasets?.length === 1) dataset.value = response.datasets[0].id;
  }

  async function refreshScenarioOptions() {
    const select = $('batchScenario');
    const wanted = select.value;
    select.replaceChildren(new Option('全部可执行访谈（整个数据集）', ''));
    if (!dataset.value) return;
    const response = await api('/api/batch/candidates?dataset=' + encodeURIComponent(dataset.value));
    for (const row of response.candidates || []) {
      if (row.eligible) select.add(new Option(row.label, row.id));
    }
    if ([...select.options].some(option => option.value === wanted)) select.value = wanted;
  }

  async function refreshCategories() {
    const select = $('batchGuideCategory');
    const wanted = select.value;
    select.replaceChildren(new Option('自动（仅唯一大纲适用）', ''));
    if (!dataset.value) return;
    const response = await api('/api/graded-guides/status?dataset=' + encodeURIComponent(dataset.value));
    for (const category of response.categories || []) select.add(new Option(category, category));
    if ([...select.options].some(option => option.value === wanted)) select.value = wanted;
  }

  dataset.addEventListener('change', () => Promise.all([refreshCategories(), refreshScenarioOptions()])
    .then(saveBatchPreferences)
    .catch(err => message(err.message, true)));

  async function refresh() {
    if (!current) return;
    try {
      render(await api('/api/batch/status?experiment_id=' + encodeURIComponent(current)));
    } catch (err) {
      message(err.message, true);
    }
  }

  async function history() {
    const result = await api('/api/batch/experiments');
    experimentHistory = result.experiments || [];
    renderHistoryLists();
    if (current) await refresh();
    else clearCurrentPanel();
  }

  async function work(action) {
    try {
      const result = await action();
      if (result?.experiment_id) render(result);
      await history();
    } catch (err) {
      message(err.message, true);
    }
  }

  function selectedHistoryExperiment() {
    return currentSummary();
  }
  function chosenHistory() {
    const selected = selectedHistoryIds();
    if (selected.length !== 1) return null;
    return experimentHistory.find(row => row.experiment_id === selected[0]) || null;
  }

  function applyHistoryOptions(summary) {
    if (!summary) throw Error('请先在历史实验列表中选择一个实验');
    const options = summary.options || {};
    const mapping = {
      batchDataset: summary.dataset || '',
      batchScenario: options.candidate_id || '',
      batchGuideLevel: options.guide_level || 'level1',
      batchGuideCategory: options.guide_category || '',
      batchInterviewer: options.interviewer_provider || 'shubi',
      batchRespondent: options.respondent_provider || 'shubi',
      batchLanguage: options.language || 'zh',
      batchPreparation: options.preparation_mode || 'python_binary',
      batchMaxTurns: String(options.max_turns ?? 30),
      batchInterviewerTimeout: String(options.interviewer_timeout_seconds ?? 300),
      batchRespondentTimeout: String(options.respondent_timeout_seconds ?? 300),
      batchRetrievalTimeout: String(options.retrieval_timeout_seconds ?? 240)
    };
    for (const [id, value] of Object.entries(mapping)) {
      const node = $(id);
      if (!node) continue;
      if (node.options && [...node.options].length && ![...node.options].some(option => option.value === value)) continue;
      node.value = value;
    }
    $('batchQualityMode').checked = options.quality_mode !== false;
    if (options.behavioral_traits) {
      for (const [trait, value] of Object.entries(options.behavioral_traits)) {
        const node = root.querySelector(`[data-batch-trait="${trait}"]`);
        if (node) node.value = value;
      }
      updateTraitLabels();
    }
    for (const role of ['interviewer', 'respondent']) {
      const savedProfile = options.model_profiles?.[role];
      if (!savedProfile) continue;
      const previous = batchDrafts[role][savedProfile.provider] || {};
      batchDrafts[role][savedProfile.provider] = {
        ...savedProfile,
        api_key: previous.model === savedProfile.model &&
          previous.base_url === savedProfile.base_url ? (previous.api_key || '') : ''
      };
      // New-experiment UI is binary (Remote API / Local). Existing legacy
      // MaaS jobs still resume with their exact saved provider on the backend.
      providerSelect(role).value = savedProfile.provider === 'local' ? 'local' : 'shubi';
    }
    showBatchProfile('interviewer');
    showBatchProfile('respondent');
    saveBatchPreferences();
    message('已载入参数；历史续访需使用原模型和 URL，密钥请在「新实验」中重新填写。');
  }

  $('batchReload').addEventListener('click', () => work(async () => {
    await refreshDatasets();
    await Promise.all([refreshCategories(), refreshScenarioOptions()]);
    return null;
  }));

  $('batchCreate').addEventListener('click', () => work(async () => {
    if (!$('batchConsent').checked) throw Error('请先勾选数据处理许可');
    if (!dataset.value) throw Error('请选择数据集');
    // Hold credentials only in this page's live memory for the separate Start.
    // They never enter localStorage, sessionStorage, or experiment JSON.
    for (const role of ['interviewer','respondent']) {
      const profile = readBatchForm(role);
      batchDrafts[role][profile.provider] = profile;
    }
    return api('/api/batch/create', {
      dataset:dataset.value,
      candidate_id:$('batchScenario').value||null,
      model_profiles:allBatchProfiles(),
      guide_level:$('batchGuideLevel').value,
      guide_category:$('batchGuideCategory').value||null,
      interviewer_provider:$('batchInterviewer').value,
      respondent_provider:$('batchRespondent').value,
      language:$('batchLanguage').value,
      preparation_mode:$('batchPreparation').value,
      quality_mode:$('batchQualityMode').checked,
      max_turns:readInteger('batchMaxTurns',0,1000000),
      interviewer_timeout_seconds:readInteger('batchInterviewerTimeout',30,600),
      respondent_timeout_seconds:readInteger('batchRespondentTimeout',30,900),
      retrieval_timeout_seconds:readInteger('batchRetrievalTimeout',30,900),
      behavioral_traits:batchTraits(),
      external_processing_consent:$('batchConsent').checked
    });
  }));

  async function startCreated() {
    const selected = selectedHistoryExperiment();
    if (!selected || selected.state !== 'created') throw Error('请先创建或选择一个尚未启动的实验');
    return api('/api/batch/launch', {
      experiment_id: selected.experiment_id, model_profiles: profilesForSavedJob(selected)
    });
  }
  $('batchStart').addEventListener('click', () => work(startCreated));

  function profilesForSavedJob(selected) {
    const saved = selected.options?.model_profiles;
    if (!saved?.interviewer || !saved?.respondent) throw Error('缺少实验保存的模型配置');
    const profiles = {};
    for (const role of ['interviewer','respondent']) {
      const profile = saved[role];
      const draft = batchDrafts[role]?.[profile.provider] || {};
      const matches = draft.model === profile.model && draft.base_url === profile.base_url;
      const supplied = $(role === 'interviewer' ? 'batchHistoryInterviewerKey' : 'batchHistoryRespondentKey').value.trim();
      profiles[role] = {provider:profile.provider, model:profile.model, base_url:profile.base_url,
                        api_key:supplied || (matches ? (draft.api_key || '') : '')};
    }
    return profiles;
  }

  async function resumeSelected() {
    const selected = selectedHistoryExperiment();
    if (!selected) throw Error('请先选择一个历史实验');
    if (!canResume(selected)) throw Error('当前实验没有可继续的部分');
    return api('/api/batch/resume', {
      experiment_id: selected.experiment_id,
      model_profiles: profilesForSavedJob(selected)
    });
  }

  $('batchStop').addEventListener('click', () => work(async () => {
    const selected = selectedHistoryExperiment();
    if (!selected) throw Error('没有可中断的当前实验');
    stopPending = true;
    message('中断中…');
    const result = await api('/api/batch/stop', {experiment_id: selected.experiment_id});
    if (selected.experiment_id === current && lastSnapshot) {
      lastSnapshot = {...lastSnapshot, state: 'stopping'};
      render(lastSnapshot);
    }
    return result;
  }));

  $('batchResume').addEventListener('click', () => work(resumeSelected));
  $('batchHistoryResume').addEventListener('click', () => {
    const chosen = chosenHistory();
    if (!chosen) return message('请先选择历史实验', true);
    syncSingleHistorySelection(chosen.experiment_id, 'history');
    message('已载入 · 可继续');
    $('batch-current-title').scrollIntoView({behavior:'smooth', block:'start'});
  });

  $('batchRestoreConfig').addEventListener('click', () => {
    try {
      applyHistoryOptions(chosenHistory());
    } catch (err) {
      message(err.message, true);
    }
  });

  $('batchHistoryExport').addEventListener('click', () => {
    const selected = chosenHistory();
    if (!selected) return message('请先选择一个历史实验', true);
    window.location.href = '/api/batch/export?experiment_id=' + encodeURIComponent(selected.experiment_id);
  });
  $('batchExport').addEventListener('click', () => {
    const selected = selectedHistoryExperiment();
    if (!selected) return message('请先选择一个实验', true);
    window.location.href = '/api/batch/export?experiment_id=' + encodeURIComponent(selected.experiment_id);
  });

  $('batchSelectFinished').addEventListener('click', () => {
    for (const box of root.querySelectorAll('input[name="batchHistoryPick"]')) {
      const selected = experimentHistory.find(row => row.experiment_id === box.value);
      box.checked = Boolean(selected && !['running','stopping'].includes(selected.state));
    }
    syncHistorySelection();
  });
  $('batchClearSelection').addEventListener('click', () => {
    for (const box of root.querySelectorAll('input[name="batchHistoryPick"]')) box.checked = false;
    syncHistorySelection();
  });
  $('batchHistoryRows').addEventListener('change', event => {
    if (event.target.matches('input[name="batchHistoryPick"]')) syncHistorySelection();
  });
  $('batchDelete').addEventListener('click', () => work(async () => {
    const chosen = selectedHistoryIds();
    if (!chosen.length) throw Error('请先选择要删除的历史实验');
    if (chosen.some(id => ['running','stopping'].includes(
      experimentHistory.find(row => row.experiment_id === id)?.state
    ))) throw Error('运行中的实验不能删除');
    if (!window.confirm(`确认永久删除 ${chosen.length} 个历史实验及全部记录？`)) return null;
    const result = await api('/api/batch/delete', {experiment_ids: chosen});
    if (chosen.includes(current)) current = '';
    historySelection = '';
    message(`已删除 ${chosen.length} 个历史实验`);
    return result;
  }));
  $('batchHistory').addEventListener('change', event => syncSingleHistorySelection(event.target.value, 'batchHistory'));

  window.addEventListener('playground:view', event => {
    if (event.detail.view !== 'batch') return;
    work(async () => {
      await loadBatchDefaults();
      await refreshDatasets();
      restoreBatchPreferences();
      await Promise.all([refreshCategories(), refreshScenarioOptions()]);
      showBatchProfile('interviewer');
      showBatchProfile('respondent');
      await history();
      return null;
    });
  });

  setInterval(() => {
    if (root.classList.contains('is-active') && current) refresh();
  }, 1500);

  setInterval(() => {
    if (root.classList.contains('is-active')) updateElapsed();
  }, 250);

  if (location.hash === '#batch') {
    work(async () => {
      await loadBatchDefaults();
      await refreshDatasets();
      restoreBatchPreferences();
      await Promise.all([refreshCategories(), refreshScenarioOptions()]);
      showBatchProfile('interviewer');
      showBatchProfile('respondent');
      await history();
      return null;
    });
  }


  function setBatchMode(mode) {
    const historyMode = mode === 'history';
    const newTab = $('batchNewTab'), oldTab = $('batchHistoryTab');
    newTab.setAttribute('aria-selected', String(!historyMode));
    oldTab.setAttribute('aria-selected', String(historyMode));
    newTab.tabIndex = historyMode ? -1 : 0;
    oldTab.tabIndex = historyMode ? 0 : -1;
    $('batchNewPanel').hidden = historyMode;
    $('batchHistoryPanel').hidden = !historyMode;
    try { sessionStorage.setItem('interviewplayground.batch.activeTab.v1', historyMode ? 'history' : 'new'); } catch(_e){}
  }
  for (const [tab,mode] of [['batchNewTab','new'],['batchHistoryTab','history']]) {
    $(tab).addEventListener('click', () => setBatchMode(mode));
    $(tab).addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const target = mode === 'new' ? 'history' : 'new';
      setBatchMode(event.key === 'Home' ? 'new' : event.key === 'End' ? 'history' : target);
      $(event.key === 'Home' ? 'batchNewTab' : event.key === 'End' ? 'batchHistoryTab' : (target === 'history' ? 'batchHistoryTab' : 'batchNewTab')).focus();
    });
  }
  try { setBatchMode(sessionStorage.getItem('interviewplayground.batch.activeTab.v1') || 'new'); }
  catch(_e) { setBatchMode('new'); }

  updateTraitLabels();
  clearCurrentPanel();
})();
