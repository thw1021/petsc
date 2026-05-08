Review the code changes in GitLab MR $ARGUMENTS and report findings to stdout.

This command reviews the **remote MR state**, not local `HEAD`. Section 3 is a drift check that warns the user when the local branch HEAD differs from the MR head.

Throughout this file, `<NAME>` placeholders (e.g. `<MR_IID>`, `<source_branch>`) are substituted literally with the value resolved before each use — never passed through as the placeholder text itself. `<MR_IID>` is resolved in Section 1; `<source_branch>` is resolved in Section 1 except for the MR-number-only case, where Section 3 resolves it on demand.

Adhere to @CLAUDE.md while reviewing.

## Steps

### 1. Identify the MR
Pick the case below that applies. In every case, run each command as a separate Bash call. Each call must be a single static command — no `$(...)`, pipes, `;`, `&&`/`||`, redirections, here-docs, or multi-line scripts — otherwise the static analyzer will trigger a permission prompt.

- If an MR number is given (e.g. `8786`), use it directly.
- If a diff file path is given, read it, then ask the user for the source branch and run `glab mr list --source-branch <branch>`.
- If nothing is given:
    1.  Get the current branch:
        ```
        git branch --show-current
        ```
    2.  If the output is empty (detached `HEAD`), stop and ask the user for the branch name.
    3.  Otherwise run, substituting the literal branch name:
        ```
        glab mr list --source-branch <branch-from-step-1>
        ```

### 2. Get MR metadata
Run each command below as a separate Bash call (same single-static-command rule as Section 1).

1.  Get base/head/start SHAs from the latest version (use index 0 of the returned array — versions are returned newest-first):
    ```
    glab api "projects/:id/merge_requests/<MR_IID>/versions"
    ```
2.  Get the changed file list and per-file diffs:
    ```
    glab api "projects/:id/merge_requests/<MR_IID>/changes"
    ```

### 3. Drift check (local vs. MR head)
If `<source_branch>` was not resolved in Section 1 (MR-number-only case), fetch it as a separate Bash call and read the `source_branch` field from the response:
```
glab api "projects/:id/merge_requests/<MR_IID>"
```

Then check whether the source branch exists locally:
```
git show-ref --verify --quiet refs/heads/<source_branch>
```
If exit is non-zero, the user has no local copy — skip the rest of this section.

Otherwise get the local branch HEAD SHA:
```
git rev-parse <source_branch>
```
Compare it to the MR head SHA from Section 2 step 1. If they differ, warn the user that the local branch HEAD differs from the MR head — they may have unpushed local commits, or their local branch may be behind the MR. Recommend `/review-branch` if they want the local state reviewed instead.

### 4. Read and review the diff
- Use one of: `git diff` output (branch flow), the local diff file when the user handed you one, or the `changes` payload from Section 2.
- Act as a senior software engineer. Focus on:
  - Bugs and correctness issues
  - Performance implications
  - Code quality, style, and documentation: check against conventions in @doc/developers/style.md
  - Missing error handling
- PETSc-specific error-model rule: treat `PetscCall()`, `PetscCheck()`, `SETERRQ`, and related PETSc error macros/functions as terminal. Do not report issues that matter only after such a fatal error path is taken, such as unreleased resources, un-restored arrays, or partially updated state. Still report bugs that affect behavior on non-error paths or before the fatal error is raised.
- Classify each finding by severity: CRITICAL, HIGH, MEDIUM, LOW, or Style/Nit.
- Do not praise or compliment the merge request. For example, do not state the merge request is well organized or well conceived.

### 5. Report findings
For each finding, print to stdout:
- **Severity** (CRITICAL/HIGH/MEDIUM/LOW/Style)
- **File and line number**
- **Description** of the issue
- **Suggested fix** (if applicable)
