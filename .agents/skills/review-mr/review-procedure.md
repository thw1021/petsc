### 3. Read and review the diff
- Review the file named by `DIFF_FILE`.
- Report any file the diff shows as binary (`Binary files ... differ` or `GIT binary patch`) as not covered.
- Review through to line `LINES`, in parts if needed. Do **not** re-run the diff per file.
- Focus on:
  - Bugs and correctness issues
  - Performance implications
  - Code quality, style, and documentation: check against conventions in @doc/developers/style.md
  - Unchecked or swallowed error returns (e.g. calls missing `PetscCall()`)
- Never review `.out` file *contents*; do flag mismatches between a code change and its reference output (missing update, unjustified regeneration, orphan file).
- PETSc error model: treat `PetscCall()`, `PetscCheck()`, `SETERRQ` as terminal — don't report leaks/un-restored arrays on fatal paths. Do report bugs on non-error paths or before the error fires.
- Classify each finding: CRITICAL / HIGH / MEDIUM / Style / LOW. Don't pad the report with praise.
- Report every occurrence of each confirmed issue with file:line — never a representative example or "and similar elsewhere". Scan every changed hunk before reporting. This is completeness per issue, not a quota: a clean diff producing few or no findings is a valid outcome.

Severity weights for PETSc:
- **CRITICAL / HIGH / MEDIUM** — correctness, performance, real bugs.
- **Style** — important. PETSc convention violations (clang-format, naming, idioms, AGENTS.md anti-patterns) are real review blockers. Treat at par with MEDIUM.
- **LOW** — count, do not list. End the report with `(N LOW findings suppressed; ask to show them.)` when `N > 0`. List individual LOW items only if asked.

Every finding at MEDIUM or above must state its **trigger** (the concrete input or event that produces the failure) and its **cost** (what goes wrong, in absolute terms — never only a ratio). If the honest trigger requires violating the code's documented use, or the honest cost is a worse message, a lost convenience, or one manual rerun, the finding is LOW. A design alternative is not a finding.

Calibration examples:
- MEDIUM: "a failed POST is counted as posted" — trigger: glab exits nonzero with empty stderr; cost: the tool reports success for a comment that never landed.
- LOW: "no retry on a transient API timeout" — trigger: observed once; cost: one rerun of an allow_failure CI job. The fix (a retry loop) costs more than that.

### 4. Verify each finding before reporting
Treat every finding at Style or above as tentative: reopen the cited code in the current working tree and confirm the finding matches it, is real — not a misread or speculation — and is actionable. For MEDIUM and above, confirm the trigger and cost justify the severity and the fix neither breaks a documented use nor adds more mechanism than the cost warrants; downgrade to LOW when not. Report only findings that survive.

### 5. Compose report
- Per finding: severity, file:line, description, trigger and cost (MEDIUM and above), suggested fix. Order CRITICAL → HIGH → MEDIUM → Style. If nothing at or above Style is found, say so explicitly.
- State any path Section 3 reported as not covered.

### 6. Write report
- Always write the report (with a title) to ai-review.html! Add a footnote with claude version and model used, date, time, MR_IID, CI_PIPELINE_ID, CI_JOB_ID, when available.
