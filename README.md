# InterviewPlayground

**Interview & Evaluation Workbench | 访谈与评测一体化研究工作台**

InterviewPlayground 的核心不是一套单独的评分器，而是**让研究者准备访谈、构建受访者情境、与模拟受访者开展多轮研究访谈，并在访谈完成后评估采访质量和信息获得情况**的本地研究工具。评测是访谈工作的后续复盘环节，不能替代访谈本身。

**当前版本：v1.9.3。** 单一浏览器工作台（`index.html`），本地 Python HTTP 后端，独立评测子进程；默认只监听 `127.0.0.1`。

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
