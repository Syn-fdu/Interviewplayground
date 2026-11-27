# Changelog

## Core workflow correction: bound guide levels and isolated batch configs
- Bind all three level-specific interviewer guide variants to the scene in one build.
- Choose an already-bound guide in single interview through select-guide API; do not reconstruct the original source.
- Preserve selected grade in the interviewer preparation cache and protect pending sessions.
- Separate interviewer/respondent API credentials by role, URL, and thread-local experiment scope.
- Add batch-only per-role URL, model, API Key, provider, and synthetic connection probes.
- Persist non-secret configuration per batch and require original parameters when resuming.
- Add local Ollama interviewer support, single respondent API test, and per-candidate batch scope.
- Verify no private credentials enter experiment.json, and add regression tests.

## V3 — 批量访谈界面重构

- 高级选取退出可见界面，保留旧脚本的兼容节点。
- 精简重复的配置说明和提示。
- 批量实验参数重组为访谈任务、模型配置、记忆与行为、执行控制。
- 保留三级大纲和独立配置机制。
