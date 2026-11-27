# Source cleanup and GitHub data-safety notes

## What is preserved
This package is a **functional local distribution**, not a stripped-down public
source release. It preserves the exact seven research datasets and their
interview records, outlines, MemoryBank JSON, participant profiles and
manifests, along with 26 graded-guide collections, runtime batch checkpoints
and prepared-memory caches.

Bundled-data verification:
- 7 datasets / 97 candidates / 97 eligible interviews
- `ai_coding_workflows`: 13 candidates, all eligible
- Graded-guide registry: 26 datasets
- `GET /api/batch/datasets` and `/api/batch/candidates?dataset=ai_coding_workflows`
  return nonzero real values when the server is started without explicit inputs.
- The server now automatically loads bundled data if it exists. Pass
  `--data-dir` or `--data-zip` to override; pass `--no-bundled-data` to opt out.

## What was deleted
Only `.git` embedded metadata (not source files), bytecode/pytest caches,
22 obsolete BATCH_*_README.md patch notices, 2 old patch-install instructions,
and one stale test that imported `safe_failure_reason`, which does not exist
in this uploaded application's current BatchManager implementation.
No `resources/`, `.runtime_cache/`, `prepared_memory/`, interviewer skills,
or production source file was removed.

## Why a GitHub push can still omit the dataset
The existing `.gitignore` deliberately excludes private processed interviews,
local memory caches and runtime experiment transcripts. Those paths are
**present in this ZIP** but will not be uploaded by ordinary `git add -A`.

**Do not upload real interview data to a public GitHub repository without
explicit participant/privacy and redistribution authorization.** Even if the
repository is private, confirm collaborators and organization policy permit
storing these research records on GitHub.

If the repository is private, permission is confirmed and you WANT all seven
processed datasets versioned, you may explicitly add only processed JSON
using the following Git pathspec after placing this package in a fresh clone:

    git add -A
    git add -f "resources/bundled_data/InterviewData_Cumulative_Refined_14/social_research/*/processed/*.json"

Do not force-add `.runtime_cache/` or `prepared_memory/` as they contain
local private model state and checkpoints. Keep those in a separate encrypted
backup. If you need to continue local batch histories, preserve that cache.

## New feature branch (in the cloned repository)
    git switch -c feature/batch-experiments-graded-outlines
    git status --short
    python -m pytest -q tests
    git add -A
    git diff --cached --stat
    git commit -m "feat: preserve bundled interviews and batch graded outlines"
    git push -u origin feature/batch-experiments-graded-outlines

Do not run the commit if the required dataset files were unintentionally
ignored; read the force-add instructions above and confirm privacy first.
