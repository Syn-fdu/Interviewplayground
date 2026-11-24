# Architecture & dependency boundaries

InterviewPlayground is a **single-page** local app. `index.html` includes Research Studio, the interview interface and the dual-engine Benchmark. No additional standalone HTML pages are required. The old `/research_studio.html` and `/strict_benchmark.html` URLs redirect to sections on the main page.

The active `benchmark/benchmark.json` file is **not an HTML page or archived test**: `server.py` reads this default evaluation configuration through `/api/benchmark/config/validate`. Deleting it would break that endpoint.

- `server.py`: local HTTP API, data authorization, scenario and interview sessions
- `research_*`, `data_intake.py`, `scenario_builder.py`: research preparation
- `interview_agent/`: active interviewer planner, turn state, skills and adapters
- `integrated_memory.py`, `source_recall.py`, `python_factbank.py`: source-grounded respondent MemoryBank
- `strict_benchmark_integration.py`: Fast and Spoken subprocess coordination; interrupt/resume
- `standalone_evaluation/fast/fast_benchmark.py`: per-required and per-optional-deepening Slot coverage plus five interviewer-quality scores
- `standalone_evaluation/spoken/strict_full.py`: Memory question opportunity assessment
- `standalone_evaluation/fast/core/`: a few deliberately sandboxed imports for standalone scoring (separate diagnostic/runtime context)
- `knowledge_banks/`, `memory_pools/`: active default assets
- `tests/`: small active tests; external comprehensive QA tests are maintained separately

The Windows batch file, PowerShell launch helper, and POSIX shell script each serve a different launch environment. The old `start_server_mode.ps1` was removed as an overlapping entry.

The public GitHub package intentionally excludes `bundled_data/` interview records until redistribution permissions and privacy have been reviewed. The local parity build includes those records. Shared business code is byte-identical between the two builds.

Legacy-named Python compatibility adapters remain where their functionality is actually referenced by the active application. Renaming or deleting such modules solely because they contain `legacy` would risk behavioral regressions.
