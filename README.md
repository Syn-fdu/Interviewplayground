# InterviewPlayground

**Version 1.9.2 — source-cleaned research interview workstation**

本地优先的研究访谈工作台：准备研究场景、加载受访者 MemoryBank、运行访谈，最后让独立的 Fast-Grounded 与 SpokenTruth 评分。`index.html`、`server.py` 与两个独立评分子进程共同构成程序；浏览器与后端使用同一个发行版本号。

## Quick start

Windows: run `start_windows.bat` from the project directory.  
macOS/Linux: run `sh start_mac_linux.sh`.  
Manual: `python server.py --open-browser`.

建议 Python 3.10+，首次启动可运行 `python -m pip install -r requirements.txt`。本机服务默认绑定 `127.0.0.1`；不要直接暴露到公网。评分需要正确的 API 地址、可用模型 ID、密钥和明确的外部数据传输授权。

**默认评审模型：** Fast-Grounded `gpt-6.1-sol`；SpokenTruth `qwen3.5-flash`。模型名原样保留；不能据此保证所选供应商支持这些名称。

## 功能与计算口径

**Fast-Grounded** 使用完整访谈和对应 Outline，在正常路径下一次模型请求完成所有分类 Slot 评分及五个简化质量指标：

| 指标 | 分母或含义 |
|---|---|
| 必需信息点覆盖率 | `required_slots` 已覆盖数 / 全部必需数 |
| 深化信息点覆盖率 | `optional_deepening_slots` 已覆盖数 / 全部深化数 |
| 问题相关性 | 研究目标关联程度 |
| 追问承接 | 是否接续受访者回答 |
| 提问清晰度 | 采访者表达是否清楚 |
| 非引导性 | 是否避免诱导 |
| 沟通融洽度 | 是否自然尊重 |

两类 Slot **是 Outline 的两种类别，不是同一 Slot 的基本/深度等级**。每个正向 Slot 必须提供可在真实受访者回答中核对的原文引文与轮次。五项质量指标为独立的粗粒度模型判断；缺失时显示「未评价」，不虚构分数。

**SpokenTruth** 对完整 Memory 列表与实际采访者问题分别判断事实是否获得表达机会，并可对遗漏的 Memory 逐项定向补评。它不直接替代 Fast 的 Slot 评价。

**评分中止与恢复：**「中断机评」对本地评分进程组发送等效 Ctrl+C 信号，必要时终止整个组，前端显示「已中断」「继续机评」。恢复会保留完整评分与可用检查点；当前单次 Fast 请求被打断后可能需要重新发起远端请求。第三方服务器的推理不保证随本地中断同时停止，也可能计费。服务重启不能自动接管以前内存中的 Job ID。

## 项目结构

| 路径 | 职责 |
|---|---|
| `server.py`、`index.html`、`assets/` | 本机 HTTP API、前端与样式 |
| `data_intake.py`、`scenario_builder.py`、`research_*.py` | 数据导入、场景与研究准备 |
| `interview_agent/`、`respondent_*.py`、`integrated_memory.py` | 访谈 Agent、受访者行为与 Memory |
| `source_recall.py`、`factbank.py`、`python_factbank.py` | 证据检索、来源 FactBank、统一的时间事实规则 |
| `strict_benchmark_integration.py` | 两个评分器的任务调度、协议、状态、中断与恢复 |
| `standalone_evaluation/fast/fast_benchmark.py` | Fast-Grounded 独立评分 |
| `standalone_evaluation/spoken/strict_full.py` | SpokenTruth 独立评分 |
| `standalone_evaluation/fast/core/` | **独立评分器的隔离依赖**，与主工作台同名模块刻意运行在不同导入上下文，不能机械去重 |
| `interview_agent/skills/` | Agent 运行时的四种行为规则（必须保留） |
| `knowledge_banks/`、`memory_pools/` | 运行时领域知识与默认公共 MemoryPool |
| `bundled_data/` | 额外的示例访谈数据集；**公开仓库默认忽略，需单独核实数据公开权利** |
| `tests/` | 不含旧 Iteration 测试的精简活跃回归 |
| `benchmark/benchmark.json` | 当前研究评估接口运行时读取的配置，并非独立网页 |

`python_factbank.py` 已整合九层历史事实分类规则，但它们的处理顺序、来源状态保护与最终 v15 分类逻辑保持原样。`python_binary_factbank.py` 是不同职责的“二分类输出适配器”，不能与基础分类器混为一谈。

## 数据导入与可复现性

使用网页选择数据，或在启动时指定 `--data-zip /path/to/authorized-dataset.zip` / `--data-dir /path/to/authorized-folder`。在「关联当前访谈」和「外部资料」两种机评入口，Transcript/Outline/Memory 应来自同一场景与研究版本。程序默认不会把外部访谈文件写入项目源目录。

为了让旧版已有的预置示例行为保持一致，**完整本地发行包仍随附原有 `bundled_data/`**；但其内容包含访谈记录，不能在未经授权的情况下直接公开到 GitHub。`.gitignore` 默认阻止 `git add .` 提交这部分数据。公共 GitHub clone 如果没有这些数据，手动上传真实或授权的示例 ZIP 仍可使用主程序；内置「refined」示例则不可用。这是一个**数据版权/隐私的发布安全边界**，不是功能回归。参见 `PUBLICATION_REVIEW.md`。

运行时 `prepared_memory/bundled_archive_cache` 是根据 `bundled_data` **自动生成**的 ZIP 缓存，不提交到仓库。旧版缓存与其对应原始文件已逐条比对内容一致。

## Tests

标准库快速验收（不会调用付费模型）：

```bash
python -m unittest discover -s tests -v
```

历史完整回归、模拟 HTTPS 模型服务和 Chromium 浏览器验收已在隔离的 QA 副本执行；不向公开仓库堆放几十个旧版本测试文件。文件级清理过程记录在发行版外部的重构报告；业务运行依赖仍保留。
旧 `/strict_benchmark.html` 和 `/research_studio.html` 链接不再对应独立 HTML 文件，会直接由服务端重定向到统一主页的 Benchmark / Studio 区域。

## GitHub 发布须知

1. **先检查 `bundled_data/` 的使用授权及其中的个人信息**；`.gitignore` 虽默认排除，但用 GitHub 网页拖拽文件上传时不一定生效。
2. 不要发布 `.env`、API 密钥、研究参与者原始资料、任务日志和缓存。
3. **本包尚未附加开源许可证**；请在公开前明确选择并加入符合你意图的 LICENSE。没有许可证并不自动授权他人复制和修改。
4. 不要将本机服务端口对外公开，评分外发要获得研究者明确同意。

项目修改记录见 `CHANGELOG.md`。安全及公开数据清单见 `PUBLICATION_REVIEW.md`。
