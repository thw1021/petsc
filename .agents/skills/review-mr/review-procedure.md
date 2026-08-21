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
- Classify each finding: CRITICAL / HIGH / MEDIUM / Style / LOW.
- A clean diff producing few or no findings is a valid outcome.

Severity weights for PETSc:
- **CRITICAL / HIGH / MEDIUM** — correctness, performance, real bugs.
- **Style** — important. PETSc convention violations (clang-format, naming, idioms, AGENTS.md anti-patterns) and factually wrong documentation or error text are real review blockers. Treat at par with MEDIUM.
- **LOW** — count, do not list; list individual LOW items only if asked.

Every MEDIUM+ finding must state its **trigger** (concrete input or event; an MPI rank/partition condition counts) and **impact** (concrete consequence: wrong result, crash, hang, leak; a slowdown must name the affected operation). Demote to LOW when the trigger violates documented use, or the impact is only a clumsy-but-accurate message, a lost convenience, or one cheap rerun (a test or CI job, not a production solve). Factually wrong user-facing text (docstring, error message) is never LOW — Style at minimum, higher if following it causes damage. A design alternative is not a finding.

Calibration:
- MEDIUM: `VecRestoreArray()` missing on an early-return path — trigger: the `n == 0` fast path; impact: vector left locked, next `VecGetArray()` errors.
- Style: error message says "matrix" where it means "vector" — untrue published text, never LOW.
- LOW: error message awkward but accurate.

### 4. Verify each finding before reporting
Treat every finding at Style or above as tentative: reopen the cited code in the current working tree and confirm the finding matches it, is real — not a misread or speculation — and is actionable. For MEDIUM and above, confirm the trigger and impact justify the severity and the fix neither breaks a documented use nor adds more mechanism than the impact warrants; downgrade to LOW when not. Report only findings that survive.

### 5. Compose report
- Per finding: severity, file:line, description, trigger and impact (MEDIUM and above), suggested fix. List every occurrence of a confirmed issue with file:line — never a representative example or "and similar elsewhere". Order CRITICAL → HIGH → MEDIUM → Style.
- State the coverage: files and lines reviewed, and any path Section 3 reported as not covered. End with `(N LOW findings suppressed; ask to show them.)` when `N > 0`.
- The report contains the findings, the coverage, and the LOW count — nothing else. If there are no findings at or above Style, say exactly that.

### 6. Write report
- Always write the report (with a title) to ai-review.html! Add a footnote with claude version and model used, date, time, MR_IID, CI_PIPELINE_ID, CI_JOB_ID, when available.
