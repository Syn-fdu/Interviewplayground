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

## Batch experiment extension (v1.9.4)

Batch runs use `playground/research/batch_experiments.py` (persistence, queue,
resume, export); `server.py::_run_batch_interview` (adapter to existing
`build_from_v2`, `make_research_bundle`, `prepare_public_outline`, native
`run_next`, and `respondent_v08`); and `assets/batch_experiments.js` (UI only).
Model configuration and secrets stay in the existing model services module.
No evaluator is called by the batch adapter. No additional root-level HTML or
model engine fork was introduced. Output under `.runtime_cache/batch_experiments/`
is intentionally excluded from Git. The experiment manifest retains status and
one line of error per participant; private turns stay inside individual JSON/TXT.

`stop` is cooperative and does not abort an already-issued model HTTP request.
Resumption is at participant boundaries, not within the lost in-memory native
interviewer state. Existing fully completed outputs are not regenerated.
