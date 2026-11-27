# InterviewPlayground

**Interview & Evaluation Workbench | 访谈与评测一体化研究工作台**

InterviewPlayground 的核心不是一套单独的评分器，而是**让研究者准备访谈、构建受访者情境、与模拟受访者开展多轮研究访谈，并在访谈完成后评估采访质量和信息获得情况**的本地研究工具。评测是访谈工作的后续复盘环节，不能替代访谈本身。

**当前版本：v1.9.4。** 单一浏览器工作台（`index.html`），本地 Python HTTP 后端，独立评测子进程；默认只监听 `127.0.0.1`。

## 从研究准备到访谈，再到评测

```text
导入或选择有权限使用的访谈数据
          ↓
研究准备：场景、受访者画像、来源 MemoryBank、研究 Outline
          ↓
运行 Interviewer Agent，与模拟受访者逐轮访谈
          ↓
查看真实问答轮次、受访者原文与引用来源
          ↓
结束访谈 / 导出会话
          ↓
可选：Fast-Grounded + SpokenTruth 双引擎独立评测
          ↓
检查信息点覆盖、追问质量、Memory 机会及证据，导出结果
```

### 一、访谈是主功能（Interview）

- **研究准备**：导入数据、选择研究场景、设定研究目标与访谈大纲（Outline），准备来源可追溯的受访者画像和 MemoryBank。
- **采访者 Agent**：根据大纲中的问题、必需信息点与深化信息点，结合上轮访谈内容规划主问题和追问。
- **受访者模拟（Respondent Studio）**：由独立受访者代理在所选材料允许的事实边界内回答；采访者不能直接读取受访者的私有 MemoryBank。
- **多轮对话与来源追溯**：保留完整轮次、受访者原文和可追溯的来源信息，不把采访者的问题误当成受访者事实。
- **数据安全与模型使用**：用户自行配置本地或远程模型；向外部模型发送访谈内容前必须按研究资料的权限状态确认。

### 二、评测是访谈后的复盘功能（Evaluation）

**Fast-Grounded** 使用访谈全文与采访大纲评测采访者：

| 评价项 | 含义 |
|---|---|
| 必需信息点覆盖率 | `covered required_slots / all required_slots` |
| 深化信息点覆盖率 | `covered optional_deepening_slots / all optional_deepening_slots` |
| 问题相关性 | 是否契合研究目标 |
| 追问承接 | 是否接续受访者回答 |
| 提问清晰度 | 是否清楚易懂 |
| 非引导性 | 是否避免暗示、诱导 |
| 沟通融洽度 | 是否自然且尊重 |

两种 Slot 覆盖率**使用独立分母**。判为覆盖的 Slot 必须有受访者轮次和逐字证据支持；其他五项是简略的全程访谈质量判断。

**SpokenTruth** 独立检查完整受访者 Memory 与采访者真实问题之间的表达机会，必要时对遗漏的 Memory 定向补评。它不与 Fast 的结果互相代替。

两引擎是独立评分进程。运行中可点击**「中断机评」**终止本地评分进程，再通过「继续机评」恢复未完成工作；远程服务器已接收的请求不保证取消，也可能产生费用。

评测既可以关联工作台内的当前访谈，也可以导入经过身份校验的外部 Transcript / Outline / Memory ZIP。

## 完整发行包的数据集说明

本发行包**包含 7 组本地已处理访谈数据、97 位可执行受访者**，也保留全部 26 个分级大纲集合及历史实验的本地检查点。

直接运行 `python server.py` 或启动脚本会自动载入这些内置数据；如果传入 `--data-dir` 或 `--data-zip`，则优先使用明确指定的数据源。需要空工作台时使用 `--no-bundled-data`。

**研究数据可能包含受访者原话、个人信息和受授权限制的第三方材料。** 本地运行包不等于公开发布授权。当前 `.gitignore` 默认阻止提交访谈记录及本地实验缓存，避免误将敏感数据推送到公共仓库。详情见 `GITHUB_DATA_SAFETY.md`。

## 安装与启动

要求 Python 3.10+；如有需要先安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

Windows：双击 `start_windows.bat`，或者在 PowerShell 运行：

```powershell
python server.py --open-browser
```

macOS / Linux：

```bash
sh start_mac_linux.sh
```

工作台一般位于 `http://127.0.0.1:8765`。首次使用应检查模型 API 地址、实际支持的模型 ID、密钥与传输权限。原有默认评分模型名：Fast `gpt-6.1-sol`，SpokenTruth `qwen3.5-flash`；此名称不保证每家服务商都支持。

## 项目目录

```text
InterviewPlayground/
├── server.py                      # 唯一 Python Web 入口
├── index.html                     # 研究、访谈、评测共用的唯一网页
├── playground/
│   ├── research/                  # 数据导入、研究准备、场景与 Outline
│   ├── interview/                 # 受访者会话与模型行为
│   ├── interviewer_agent/         # 采访者 Agent、状态、规划与技能
│   ├── memory/                    # 来源证据、MemoryBank、检索
│   ├── evaluation/                # 评测调度、任务管理和诊断
│   └── common/                    # 共享模型服务和运行时性能
├── standalone_evaluation/        # 隔离运行的 Fast / SpokenTruth 评分器
├── resources/
│   ├── benchmark/                 # 在线使用的评分配置 JSON
│   ├── knowledge_banks/           # 领域知识
│   ├── memory_pools/              # 一般情境提示，非受访者事实
│   └── prompts/                   # 受访者 Prompt 资源
├── assets/                        # 网页样式
├── tests/                         # 精简的活跃自动测试
├── docs/                          # 架构、来源及发布安全说明
├── .github/workflows/             # GitHub CI
└── start_windows.bat / start_mac_linux.sh / Start-InterviewPlayground.ps1
```

内部业务模块使用 `playground.<area>.<module>` 形式的明确导入；独立评分器的隔离导入维持独立环境。根目录不再堆放数十个业务 `.py` 文件。

## 数据集来源、研究伦理和授权

**访谈数据与程序代码是两回事。GitHub 公开源码包不包含七组原始访谈记录。** 根据新提供的 `InterviewData.zip` 中的源文件记录与公开发布页面，已找到以下三个高度对应的研究出版来源：

- **AI 编程工作流程**：与 IT University of Copenhagen 的 [Understanding AI Assistance in Professional Software Engineering — Replication Package](https://zenodo.org/records/22893731) 高度对应；13 份 `Interview_P01.md`–`P13.md`、`Interview_guide.pdf` 和问卷文件名匹配。DOI：`10.5281/zenodo.22893731`。
- **开源依赖安全**：与 Anna Makarudze 的 [Vulnerabilities in Open-Source Dependencies: A Developer's Perspective](https://zenodo.org/records/20576332) 高度对应；访谈 PDF、参与者资料表与主题分析文件名吻合。DOI：`10.5281/zenodo.20576332`，发布页标示版权归属，**不等于允许任意再分发**。
- **SecuRoPS 专家访谈**：与 [Supplemental Materials to SecuRoPS Expert Interview](https://data.mendeley.com/datasets/692hjff5jz/1) 高度对应；公开页列有 21 名专家访谈原始资料和研究指南。DOI：`10.17632/692hjff5jz.1`，页面标示 CC BY 4.0。

**谨慎说明：** 以上依据公开标题、参与者规模、原始文件名与附件加工清单进行交叉比对；因为新附件中的这七组 `RAW_SOURCE_INVENTORY` 均标示原始 raw 文件未保存在该归档中，**尚未通过原文件哈希或全文逐项比对证明它们与加工数据一一对应**。对另外四组（AI 开发者访谈、天文学开放数据、国际学生障碍、教师家长参与）尚未定位可充分核实的原始发布者、DOI 和授权信息。

七组材料在本地处理清单中均有人工审核、隐私审批或公开发布授权尚未完成的标记。不要因此将它们标为“完全开放数据”；不要在未审核情况下公开上传 Full Local 中的访谈文本。

详细的**七组数据来源核查表、原始文件线索、发布页引用与尚未完成的验证事项**见 [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)；公开发布前请阅读 [`docs/PUBLICATION_REVIEW.md`](docs/PUBLICATION_REVIEW.md)。领域知识库与默认 MemoryPool 的再分发权利也需要另外审核。

## 测试

无需商业模型密钥的快速验收：

```powershell
python -m unittest discover -s tests -v
```

完整历史回归、真实 Chromium 浏览器以及本地 HTTPS 按实际 Prompt 生成模型响应的双引擎模拟验证，分别在测试副本运行；不等于真实第三方模型质量验收。源码中包含 GitHub Actions 的轻量回归配置。

## 开源与贡献

仓库尚未自动附加 LICENSE；只有在确定自己有权授权全部相关源码时，才选择合适的开源许可证。对访谈数据、第三方研究材料和外部导入资料，应分别核对其作者、许可证、参与者隐私及再分发权限。

版本历史见 [`CHANGELOG.md`](CHANGELOG.md)，模块职责与边界见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## v1.9.4 · Batch Interview Experiments

The single-page workbench now includes **批量访谈实验** (`#batch`). Import or explicitly load a dataset first, configure interviewer/respondent providers and the max turns, review the source rights and remote processing consent, then click **开始批量实验**. The task runs on the Python server even if you navigate away from the batch page. Keep the local server running.

- One candidate is attempted at a time, with a distinct native interviewer session and respondent session. Ineligible records are recorded as `skipped` rather than silently omitted.
- Results are written locally to `.runtime_cache/batch_experiments/<experiment_id>/`. Download through the Batch view to obtain `experiment.json`, `results/interview_XXXX.json`, and `transcripts/interview_XXXX.txt`.
- Status is `completed` only when the native interviewer signals a conclusion. Failures, source-ineligible records, manual stopping, and a reached turn limit are reported separately. Transcript files contain only real, returned turns and any actual closing message.
- After a crash/restart, reload the exact input dataset and use **恢复失败/未完成记录**. Already completed candidates are preserved. An incomplete participant is restarted from turn 1 because the in-memory native agent state is not restored mid-conversation. This is **participant-level recovery**, not exact token-level replay.
- This batch mode does **not** invoke Fast-Grounded, SpokenTruth, or any automatic evaluation.
- Batch experiments may contain private source-derived text and generated interviews. `.runtime_cache/` and `resources/bundled_data/` are ignored by Git. Do not distribute exported results or local data archives without an independent rights/privacy review.
- Batch interview execution uses the same native interviewer and respondent paths as single-interview mode, but live provider configuration, network availability, and model behavior determine whether a given task can finish. The offline test suite does not claim live model inference has been verified.

### API (loopback only)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/batch/datasets` | List imported dataset counts |
| POST | `/api/batch/start` | Start a batch with explicit processing consent |
| GET | `/api/batch/experiments` | List saved local batches |
| GET | `/api/batch/status?experiment_id=...` | Poll progress |
| POST | `/api/batch/stop` | Stop after the current model step/participant |
| POST | `/api/batch/resume` | Retry failed/incomplete/pending candidates for same source fingerprint |
| GET | `/api/batch/export?experiment_id=...` | Download local ZIP |

### Recommended GitHub branch workflow

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/batch-interview-experiments
# Copy the updated project files into this checkout; preserve .git/.
git status --short
git add server.py index.html assets/batch_experiments.js playground/research/batch_experiments.py \
  tests/test_batch_experiments.py tests/test_runtime_contracts.py tests/test_single_page_navigation.py \
  .gitignore README.md CHANGELOG.md docs/ARCHITECTURE.md start_windows.bat
git diff --cached --stat
python -m unittest discover -s tests -q
git commit -m "feat: add durable dataset-wide batch interviews"
git push -u origin feature/batch-interview-experiments
```

Open a Pull Request into `main`, review its diff and checks, then merge. **Never `git init` again inside an existing clone**, and avoid `git add -f resources/bundled_data` unless every record has been approved for publication.


### 批量实验：即时中断和阶段进度（v1.9.4 hotfix）

- 点击「中断实验」会立即在界面显示中断请求；后端设置停止标记并快速结束批次控制线程，不等待缓慢的模型响应。远程服务已开始的计算可能仍会执行到超时或完成，无法保证撤回已发起的请求。
- 完整的问答对会作为 partial Transcript 和 JSON 保存；迟到的问答不会追加到中断结果。批次和受访者分别标记为 `interrupted`。
- 支持对中断记录重新开始当前受访者；已完成的受访者不会重复执行。当前版本恢复粒度仍是**受访者级**，不会从半轮中途继续。
- 展示采访者预处理、证据准备、受访者记忆准备、问题生成、回答生成等阶段，以及上一轮耗时、最近活动时间与当前等待时间。
- 诊断字段 `progress` 与每条受访者记录的 `last_turn_seconds`、`last_turn_at` 会保存在 `experiment.json`；实际写入完成的问答才计入轮次。

> 注意：中断只保证在本地停止推进并忽略迟到结果，不能保证第三方模型服务停止计费。更新前请备份 `.runtime_cache` 的实验数据。

### 批量实验参数

批量实验现在支持分别配置采访者单次 API 调用时限（30–600 秒）、受访者回答预算和检索预算（各 30–900 秒），以及与单次访谈一致的六项受访者行为参数。可通过“读取单次访谈当前设置”复制当前页面数值。模型提供商、模型 ID、API 地址及 Key 沿用模拟访谈的统一后端设置，避免产生不同来源的密钥配置。每个批次保存固定参数快照，恢复运行沿用原批次参数。

### 批量实验：中断后继续与记录清理

在批量实验页面，点击「中断实验」会保留已经完整提交的问答轮次；随后在「历史实验」中选择同一批次，点击「继续访谈（保留轮次）」，从下一轮接续运行。每轮保存 Transcript、JSON 和原生采访者/受访者内部检查点（仅保存完整问答）。正在运行时也可以导出已完成轮次。

**兼容性**：旧版本中断后的部分访谈若没有原生内部状态检查点，无法保证真正无损接续；系统会明确拒绝继续，原始文件不被覆盖。新版本开始的实验才支持可靠的轮次级继续。原始来源数据集必须保持一致。

「清除当前实验记录」要求确认，并永久删除所选的已停止批次（包括本地 Transcript、实验 JSON、内部续访状态）。运行或正在停止的批次不能删除。删除不可撤销，请先导出 ZIP 备份。


### Three-level research outlines (v1.9.4 extension)

The `resources/outlines/refined_three_levels.json` overlay contains 15 research-candidate guides.
The native Interviewer receives the selected **Level 0** (compressed research themes), **Level 1** (basic questions), or **Level 2** (same basic questions with concrete follow-up probes). The respondent records, profiles, and memorybank remain unchanged. A dataset without a reconstructed guide retains its existing imported outline and is explicitly treated as not refined.

The single interview and batch settings both expose a level selector. A session pins its level and cannot change the level mid-conversation. Native interviewer outline preprocessing and its cache are isolated by level. Offline tests cover conversion, source isolation and level routing; live remote provider inference still requires local credential-based testing.

## Git 上传说明（本交付包）

这是可直接初始化为 Git 仓库的项目源码，解压后在项目目录中执行：

```bash
git init
git add .
git status
git commit -m "Import InterviewPlayground Git-ready source"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

**隐私与数据**：本交付包刻意不包含 `.git` 历史、`.runtime_cache`、
`prepared_memory`、原始受访者 records / profile / memorybank 等本机研究数据。
已保留经过分级处理的 `graded_guides` 和 `outline_levels.json`；发布前请复核它们的分发许可。
原始研究数据需要由有授权的操作者在本地导入，不能将 Git 仓库视为这些数据的备份。
模型 API Key 不得提交到 Git；模型配置需要在本地重新填写。
纯离线运行仍需依照原始 README 安装 Python 依赖、选择并准备模型服务。

**本次界面改动**：单次访谈按角色折叠参数，增加三级大纲切换入口；批次历史提供多选清理及
可辨认的实验名称；批次最多轮数支持 `0`（无轮数上限，应人工监控费用并主动停止）。
分级大纲严格按数据集/候选访谈匹配，缺失时保留原大纲而非跨受访者借用。

**本轮修复（场景三级大纲 + 单次/批量参数隔离）**

- 场景构建只做一次：按当前数据集与候选访谈严格匹配 Level 0、Level 1、Level 2，保存为同一个 Scene 的已绑定大纲快照。若某等级缺失，页面会禁用该等级，不借用其他受访者的数据。
- **模拟访谈**页面中选择当前大纲等级并应用即可；后端只切换已绑定快照，保留源访谈证据与私人记忆，不再重新导入或重建场景。切换前需停止/导出当前已进行的访谈。
- 单次模拟与批量实验具有相互独立的参数控制和模型配置；批量支持数据集内单个场景或全数据集执行、三级大纲、0=不限轮次、语言、采访者与受访者（本地 Ollama 或远程兼容 API）、三个阶段时限、分拣方式和六项行为参数。
- 批量保存一份**不含 API Key**的独立实验参数快照，运行线程使用隔离的模型/URL/Key，不读取单次模拟的后端覆盖配置。
- 批量历史可载入之前的模型、URL、时限、轮次、大纲和行为参数。Key 始终只在运行进程内存；**重启服务后，续跑需要重新输入原批次的 API Key**，且模型与 URL 必须与原实验保持一致。
- 对本地 Ollama 模式，运行机器必须启动 Ollama 并安装所选模型。远程模式需要对应提供商有效的 URL、模型与 Key；测试连接发送的内容仅为虚构测试文本。

**本地运行**：依照本 README 的 Python 启动方式运行；需要真实访谈证据时在本机导入你有使用权的数据包。
