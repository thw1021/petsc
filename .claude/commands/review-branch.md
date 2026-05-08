Review a local branch's changes against its target branch. Adhere to @CLAUDE.md while reviewing.

## Resolve the source branch
Let `SRC` be `$ARGUMENTS` if given, otherwise `HEAD`.

## Resolve the target branch
PETSc branches target either `main` or `release`. If you are not sure which one `SRC` targets, run the fork-point check below:
```
git fetch -q --no-tags origin +release:remotes/origin/release +main:remotes/origin/main
base_release=$(git merge-base --octopus origin/release origin/main SRC)
base_main=$(git merge-base origin/main SRC)
if [ "$base_release" = "$base_main" ]; then
  DEST=origin/release
else
  DEST=origin/main
fi
```
When `SRC`'s merge-base with `main` matches the common ancestor of `release`/`main`/`SRC`, `SRC` diverged before `release` was branched off `main` — so it targets `release`. Otherwise `main`.

State the resolved `DEST` in your output before reviewing.

## Diff and review
Run `git diff DEST...SRC` and follow steps 3–4 from @.claude/commands/review-mr.md.
