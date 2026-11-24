# Architecture — interview-first workstation

InterviewPlayground serves **one browser interface** from `index.html` and a local-only HTTP API from `server.py`.

The primary flow is:

1. **Research preparation** (`playground/research`): import evidence-backed research data, select a respondent and scenario, refine the interviewer Outline
2. **Conduct interview** (`playground/interviewer_agent` + `playground/interview`): preserve interviewer/respondent role boundaries, ask and answer follow-up questions, record complete turns
3. **Source-grounded memory** (`playground/memory`): provide only authorized respondent memories and evidence; do not expose private MemoryBank to interviewer
4. **Optional post-interview evaluation** (`playground/evaluation` + `standalone_evaluation`): Fast-Grounded Outline Slot coverage and interviewer-quality measures, SpokenTruth Memory opportunity analysis

First-party application modules use canonical `playground.<area>.<module>` imports, avoiding implicit modification of `sys.path` during web startup. The independent scoring engines remain under `standalone_evaluation/` intentionally, rather than directly importing active interviewer modules.

`resources/benchmark/benchmark.json` is an **active** default scoring configuration for `/api/benchmark/config/validate`, not an alternative webpage. `resources/knowledge_banks/`, `resources/memory_pools/`, and `resources/prompts/` are active inputs.

`standalone_evaluation/fast/core/` maintains isolated dependencies for independent scoring; byte-identical helper names in this directory can differ in how they resolve imports and record diagnostics, so do not delete them without full process-level regression.

Runtime job files, logs, caches and API keys belong outside the tracked source history. In public GitHub distributions the original interview datasets are excluded; they remain available only in the offline full local package pending rights and privacy review.
