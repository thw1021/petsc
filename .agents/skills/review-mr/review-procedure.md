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

Severity weights for PETSc — every finding must carry the evidence its kind of claim requires:
- **CRITICAL / HIGH / MEDIUM** — behavioral claims: correctness, performance, real bugs. State the **trigger** (concrete input or event; an MPI rank/partition condition counts) and **impact** (concrete consequence: wrong result, crash, hang, leak; a slowdown must name the affected operation). Demote to LOW when the trigger violates documented use, or the impact is only a clumsy-but-accurate message, a lost convenience, or one cheap rerun (a test or CI job, not a production solve).
- **Style** — at par with MEDIUM; a real review blocker. Two kinds:
  - Convention violation: name the broken rule — the AGENTS.md clause, `doc/developers/style.md` section, or linter check. No trigger/impact; do not invent one ("someone reads the code" carries no information). Cannot name the rule — LOW at best.
  - Factually wrong user-facing text (docstring, error message, docs): quote the text and state what the code actually does. Never LOW; higher than Style if following the text causes damage.
- **LOW** — count, do not list; list individual LOW items only if asked.

A design alternative is not a finding.

Calibration:
- MEDIUM: `VecRestoreArray()` missing on an early-return path — trigger: the `n == 0` fast path; impact: vector left locked, next `VecGetArray()` errors.
- Style: braces around a single-statement `if` — rule: AGENTS.md anti-pattern "Braces on single-statement if/else".
- Style: error message says "matrix" where it means "vector" — untrue published text, never LOW.
- LOW: error message awkward but accurate.

### 4. Verify each finding before reporting
Treat every finding at Style or above as tentative. Verify it against `DIFF_FILE` and, when more
context is needed, the file at `SRC_SHA` or `MR_HEAD_SHA`. Do not use the working tree unless it
matches that revision and the file has no uncommitted changes. If the exact revision is unavailable
and the diff lacks enough context, do not report the finding. Confirm that each surviving finding
is real and actionable, its evidence supports its severity, and its fix does not break documented
use or add more mechanism than the impact warrants. Downgrade it to LOW when those conditions fail.

### 5. Compose report
- Per finding: severity, file:line, description, its required evidence (Section 3), suggested fix.
  List each occurrence in the reviewed diff with file:line; do not use a representative example or
  "and similar elsewhere". Order CRITICAL → HIGH → MEDIUM → Style.
- State the coverage: files and lines reviewed, and any path Section 3 reported as not covered. End with `(N LOW findings suppressed; ask to show them.)` when `N > 0`.
- The report contains the findings, the coverage, the LOW count, and what other sections explicitly say to state — nothing else: no praise, no MR summary, no design commentary. If there are no findings at or above Style, say exactly that.

### 6. Write the merge-request report
For `review-mr` and `review-mr-post`, write the titled report to `ai-review.html`. Add a footnote
with the Claude version and model, the effort level from `$CLAUDE_EFFORT`, the current date and
time, `MR_IID`, `CI_PIPELINE_ID`, and `CI_JOB_ID`. Omit unavailable values.
