# Data provenance — source evidence and redistribution limits

此文件只追踪本程序**当前完整本地版附带的七组访谈数据**。用户提供的 `InterviewData.zip` 是更广泛的 27 个源语料/28 个目录归档；不是所有语料都进入此 Playground 的默认本地示例。原始研究记录、整理后的 `processed` JSON 和用于访谈模拟的派生 MemoryBank 是不同层级的数据，不能混为一谈。

## 核查方法与证据等级

- 检查用户提供的 `InterviewData.zip`，其中 `datasets/social_research/<dataset_id>/processed/manifest.json` 和 `quality/RAW_SOURCE_INVENTORY.json`；核对 7 个数据集 ID、原始文件名、记录数及使用限制。
- 对可能原始出版者进行公开查询，优先使用 Zenodo、Mendeley Data 和高校研究项目的正式发布页；只在标题、内容、规模及原始文件名多维重合时列出**候选原始来源**。
- 对七组数据集，新附件中的 `RAW_SOURCE_INVENTORY.json` 均标示 `raw_not_present_in_supplied_archives`。因此在当前材料内**无法通过原始文件哈希/逐字全文对照确认相同版本**。随包 `manifest.json` 记录的历史 SHA256 是加工链线索，不等于本次校验原始文件。
- 所有七组均不是自动批准的公开数据：附带 `approved_for_public_release:false`，且外部模型传输审核也未通过。论文发表和下载页面存在不等于取得个人资料的再分发权利。

## 七组内置数据的审查

| 内部 ID | 处理后访谈数 | 可追溯的候选出版出处及匹配依据 | 核查状态 |
|---|---:|---|---|
| `ai_coding_workflows` | 13 | [Understanding AI Assistance in Professional Software Engineering - Replication Package](https://zenodo.org/records/22893731)；DOI `10.5281/zenodo.22893731`；高度吻合（13 份 Markdown 访谈及同名访谈指南） | **公开出版物可定位；与内部整理版尚未逐文件核对** |
| `ai_developer_interviews` | 24 | 附件含经过加工的访谈记录与处理元数据，但没有可核实的原始出版机构、稳定网址、DOI 或许可证 | **原始外部来源未确定** |
| `astronomy_open_data` | 11 | 附件含经过加工的访谈记录与处理元数据，但没有可核实的原始出版机构、稳定网址、DOI 或许可证 | **原始外部来源未确定** |
| `international_student_barriers` | 10 | 附件含经过加工的访谈记录与处理元数据，但没有可核实的原始出版机构、稳定网址、DOI 或许可证 | **原始外部来源未确定** |
| `oss_dependency_security` | 17 | [Vulnerabilities in Open-Source Dependencies: A Developer’s Perspective](https://zenodo.org/records/20576332)；DOI `10.5281/zenodo.20576332`；高度吻合（20 份访谈 PDF + 参与者表格及主题分析） | **公开出版物可定位；与内部整理版尚未逐文件核对** |
| `parental_engagement_teachers` | 9 | 附件含经过加工的访谈记录与处理元数据，但没有可核实的原始出版机构、稳定网址、DOI 或许可证 | **原始外部来源未确定** |
| `securops_experts` | 13 | [Supplemental Materials to SecuRoPS Expert Interview](https://data.mendeley.com/datasets/692hjff5jz/1)；DOI `10.17632/692hjff5jz.1`；关联高度可信（SecuRoPS 21 名参与者原始访谈项目；未核对逐文件哈希） | **公开出版物可定位；与内部整理版尚未逐文件核对** |

### 已定位发布页面详情

**`ai_coding_workflows`**：[Understanding AI Assistance in Professional Software Engineering - Replication Package](https://zenodo.org/records/22893731)（DOI: `10.5281/zenodo.22893731`）。来源发布者或作者：Hristova, Gala Scaramuzza, Popovici et al., IT University of Copenhagen。匹配理由：高度吻合（13 份 Markdown 访谈及同名访谈指南）。
加工包的 `manifest.json` 原始文件名线索：`Interview_P01.md`、`Interview_P02.md`、`Interview_P03.md`、`Interview_P04.md`、`Interview_P05.md`。**来源链接不能证明加工后所有文本与发布版本相同。**

**`oss_dependency_security`**：[Vulnerabilities in Open-Source Dependencies: A Developer’s Perspective](https://zenodo.org/records/20576332)（DOI: `10.5281/zenodo.20576332`）。来源发布者或作者：Anna Makarudze, Blekinge Institute of Technology。匹配理由：高度吻合（20 份访谈 PDF + 参与者表格及主题分析）。
加工包的 `manifest.json` 原始文件名线索：`Interview 1.pdf`、`Interview 10.pdf`、`Interview 11.pdf`、`Interview 12.pdf`、`Interview 13.pdf`。**来源链接不能证明加工后所有文本与发布版本相同。**

**`securops_experts`**：[Supplemental Materials to SecuRoPS Expert Interview](https://data.mendeley.com/datasets/692hjff5jz/1)（DOI: `10.17632/692hjff5jz.1`）。来源发布者或作者：Samson Oruma, Ricardo Colomo-Palacios, Mary Sánchez-Gordón, Vasileios Gkioulos。匹配理由：关联高度可信（SecuRoPS 21 名参与者原始访谈项目；未核对逐文件哈希）。
加工包的 `manifest.json` 原始文件名线索：原始文件清单未保存在本次处理包清单中。**来源链接不能证明加工后所有文本与发布版本相同。**

### 原始文件不可得与可使用范围

所有七组处理清单的 `RAW_SOURCE_INVENTORY.json` 都明确写着 `raw_not_present_in_supplied_archives`；即使 manifest 中含历史原始文件名 / 哈希，本次上传材料也不能据此恢复原始文件。完整本地版使用的是另行处理后的 `InterviewData_Cumulative_Refined_14`。

受访者画像、访谈大纲与 MemoryBank 均可能包含重建、语义整理和自动处理内容。来源清单里 `manual_review_completed:false`、`approved_for_public_release:false`、`approved_for_external_model_api:false`，表示这些资料**不应在没有后续许可、伦理和隐私审核的情况下直接公开或传给外部服务商**。

请分别审核两类其他资源：`resources/knowledge_banks/` 领域知识，以及 `resources/memory_pools/default_public_memory_pool.json` 一般情境线索。默认 MemoryPool 不是研究参与者的事实陈述。

## 发布前需要补齐的元数据

对每个有意公开的数据集，须确定原始作者、正式题名、出版版本、永久 DOI、原始文件 SHA256、访问/许可状态、署名要求、研究对象知情同意和去标识化情况，再决定能否向公众发布加工版。尤其需要把公开数据授权和云端模型推理许可分开审核。

GitHub Source 发行包不包含上述访谈数据；Full Local 版保留数据以维持本地示例功能，但不能自动视为可公开再分发的仓库版本。

## 附件内部定位

新上传归档：`InterviewData.zip` → `InterviewData/datasets/social_research/<dataset_id>/processed/manifest.json` 和 `InterviewData/datasets/social_research/<dataset_id>/quality/RAW_SOURCE_INVENTORY.json`。
当前 Full Local 版：`resources/bundled_data/InterviewData_Cumulative_Refined_14/social_research/<dataset_id>/processed/manifest.json`。
同时可在源附件 `InterviewData/standard/provenance/README.md` 查看整个导入集合的 provenance 管理规则；它明确禁止凭 processed JSON 伪造缺失的原始 raw 文件。