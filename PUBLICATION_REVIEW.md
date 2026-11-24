# Before publishing to GitHub

## Mandatory review

- [ ] Confirm the legal basis/licensing of the **seven bundled interview datasets** in `bundled_data/`.
- [ ] Check for personally identifiable information (names, emails, phone numbers, participant profiles and free-form interview transcripts), and redact or obtain consent before public distribution.
- [ ] The GitHub source ZIP intentionally omits `bundled_data/`. The full offline parity ZIP retains it only for local evaluation.
- [ ] Review the knowledge-bank and default memory-pool datasets for redistribution rights.
- [ ] Choose and add a LICENSE; no license has been presumed on your behalf.
- [ ] Add your own `.env` locally only; never commit credentials.
- [ ] Review repository contents after `git add` with `git status` and `git diff --cached --stat` before pushing.

## Security exclusions

The repository ignores: Python bytecode, runtime errors/JSONL logs, generated model responses, temporary files, local env files, certificates, model cache, benchmark jobs/checkpoints, and bundled interview recordings.

**Important:** `.gitignore` applies to git staging; it does not protect files uploaded manually by dragging a directory into the GitHub web UI. For a public repository, use the separate **GitHub source ZIP** rather than the complete local-usage ZIP.

## Functional parity caveat

The complete offline ZIP contains authorized-or-not-yet-reviewed preloaded interview data and reproduces the previous v1.9 sample dataset availability. The public source ZIP omits those records and therefore cannot show the preloaded sample until the user provides appropriately licensed data. The local HTTP app, import/selection logic, interview paths, scoring algorithms, stop/resume mechanisms, and browser controls are the same source files.
