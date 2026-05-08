Review a local branch's changes against its target branch. Adhere to @CLAUDE.md while reviewing.

Throughout this file, `<NAME>` placeholders (e.g. `<SRC>`, `<BASE_MAIN>`, `<DEST>`) are substituted literally with the resolved value before each use — never passed through as the placeholder text itself.

## Resolve the source branch
`SRC` is `$ARGUMENTS` if given, otherwise `HEAD`. Must be a single ref name — if `$ARGUMENTS` contains whitespace or multiple tokens, stop and ask the user to disambiguate.

## Resolve the target branch
PETSc branches target either `main` or `release`. Determine which by running each command below as a separate Bash call. Each call must be a single static command — no `$(...)`, pipes, `;`, `&&`/`||`, redirections, here-docs, or multi-line scripts — otherwise the static analyzer will trigger a permission prompt. Assumes `origin` points at the upstream `petsc/petsc` repo (which has both branches); if the fetch in step 1 fails because `origin` is a fork, stop and ask the user which remote to use.

1.  Fetch refs:
    ```
    git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main
    ```
2.  Get `BASE_MAIN`:
    ```
    git merge-base origin/main <SRC>
    ```
3.  Test reachability:
    ```
    git merge-base --is-ancestor <BASE_MAIN> origin/release
    ```
    Read the exit code from the Bash tool's result: 0 → `DEST=origin/release`; 1 → `DEST=origin/main`. Any other non-zero exit (e.g., unknown commit) is an error — stop and report it instead of defaulting.

This works even when `release` has already been merged into `main` since `SRC` was created.

State the resolved `DEST` in your output before reviewing.

## Diff and review
Run `git diff <DEST>...<SRC>` (three-dot: changes on `SRC` since its merge-base with `DEST`), then follow the "Read and review the diff" and "Report findings" sections from @.claude/commands/review-mr.md.
