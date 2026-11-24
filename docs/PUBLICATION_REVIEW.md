# Before publishing to GitHub

## Dataset provenance status (verified from bundled manifests)

The seven internal dataset IDs are `ai_coding_workflows`, `ai_developer_interviews`,
`astronomy_open_data`, `international_student_barriers`, `oss_dependency_security`,
`parental_engagement_teachers`, and `securops_experts` (97 interview records in
their processed manifests altogether).

**These are internal names, not verified external publication citations.**
No original data collection institution, published dataset URL/DOI, or
redistribution license was established by the provided archive. Every checked
manifest has `privacy_review_completed: false` and
`approved_for_public_release: false`. Some processing/review status fields
refer only to derived interview outlines and memory records, not public rights.

Before sharing data externally, obtain origin and permissions per dataset.
`knowledge_banks/` and `memory_pools/` also require their own source and
rights review. See `DATA_PROVENANCE.md` and the README dataset provenance section for the latest source candidates, verification limitations and full table.

## Mandatory review

- [ ] Confirm the legal basis/licensing of the **seven bundled interview datasets** in `resources/bundled_data/`.
- [ ] Check for personally identifiable information (names, emails, phone numbers, participant profiles and free-form interview transcripts), and redact or obtain consent before public distribution.
- [ ] The GitHub source ZIP intentionally omits `resources/bundled_data/`. The full offline parity ZIP retains it only for local evaluation.
- [ ] Review the knowledge-bank and default memory-pool datasets for redistribution rights.
- [ ] Choose and add a LICENSE; no license has been presumed on your behalf.
- [ ] Add your own `.env` locally only; never commit credentials.
- [ ] Review repository contents after `git add` with `git status` and `git diff --cached --stat` before pushing.

## Security exclusions

The repository ignores: Python bytecode, runtime errors/JSONL logs, generated model responses, temporary files, local env files, certificates, model cache, benchmark jobs/checkpoints, and bundled interview recordings.

**Important:** `.gitignore` applies to git staging; it does not protect files uploaded manually by dragging a directory into the GitHub web UI. For a public repository, use the separate **GitHub source ZIP** rather than the complete local-usage ZIP.

## Functional parity caveat

The complete offline ZIP contains authorized-or-not-yet-reviewed preloaded interview data and reproduces the previous v1.9 sample dataset availability. The public source ZIP omits those records and therefore cannot show the preloaded sample until the user provides appropriately licensed data. The local HTTP app, import/selection logic, interview paths, scoring algorithms, stop/resume mechanisms, and browser controls are the same source files.
