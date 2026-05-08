Review a local branch's changes against its target branch. Adhere to @CLAUDE.md while reviewing.

## Resolve the source branch
`SRC` is `$ARGUMENTS` if given, otherwise `HEAD`. Must be a single ref name.

## Resolve the target branch
PETSc branches target either `main` or `release`. Determine which by running each command below as a separate Bash call (do **not** combine into one multi-line script — each must be statically analyzable to run without a permission prompt):

1.  Fetch refs:
    ```
    git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main
    ```
2.  Get `BASE_MAIN`:
    ```
    git merge-base origin/main <SRC>
    ```
3.  Test reachability (substitute the SHA from step 2):
    ```
    git merge-base --is-ancestor <BASE_MAIN> origin/release
    ```
    Exit 0 → `DEST=origin/release`; non-zero → `DEST=origin/main`.

This works even when `release` has already been merged into `main` since `SRC` was created.

State the resolved `DEST` in your output before reviewing.

## Diff and review
Run `git diff <DEST>...<SRC>` (three-dot: changes on `SRC` since its merge-base with `DEST`) and follow steps 3–4 from @.claude/commands/review-mr.md.
