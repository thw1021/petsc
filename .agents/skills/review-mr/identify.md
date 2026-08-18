Run each command in this file as a separate shell command with no shell metacharacters (no `$(...)`, pipes, `;`, `&&`/`||`, here-docs).

### 1. Fetch the merge request
Reject any `<MR_IID>` that isn't `^[0-9]+$`.

Run `python3 lib/petsc/bin/maint/ai_review_fetch.py mr <MR_IID>`, omitting `<MR_IID>` to use the open merge request whose source branch is the current branch.

- Exit status 3: it cannot pick the merge request on its own. Ask which IID, then rerun with it.
- Any other non-zero exit status: stop and report.

It writes `mr-<MR_IID>-meta.json` and `mr-<MR_IID>-diff.txt`, and prints `MR_IID`, `SOURCE_BRANCH`, `MR_HEAD_SHA`, `BASE_SHA`, `HEAD_SHA`, `START_SHA`, `META_FILE`, `DIFF_FILE`, `FILES`, `LINES`, `DRIFT`, one `BINARY_FILE` line per binary file, and a `WARNING:` line per non-fatal condition. Use those values; do not re-derive them. The diff is reviewed in @review-procedure.md Section 4.

### 2. Drift check
If it prints `DRIFT=yes`, warn that the local branch and the MR head diverge and recommend `/review-branch`.

### 3. Warnings
Repeat every `WARNING:` line it printed in the report.
