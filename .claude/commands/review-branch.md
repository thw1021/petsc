Review a local branch's changes against its target branch. Adhere to @CLAUDE.md while reviewing.

## Resolve the source branch
Let `SRC` be `$ARGUMENTS` if given, otherwise `HEAD`.

## Resolve the target branch
PETSc branches target either `main` or `release`. If you are not sure which one `SRC` targets, run the fork-point check below:
```bash
SRC="${1:-HEAD}"

git fetch -q --no-tags origin release:refs/remotes/origin/release main:refs/remotes/origin/main
split=$(git merge-base origin/release origin/main)
base_main=$(git merge-base origin/main "$SRC")
if [ "$base_main" = "$split" ]; then
  DEST=origin/release
else
  DEST=origin/main
fi
```
When `SRC`'s merge-base with `main` matches the split point between `release` and `main`, `SRC` targets `release`. Otherwise it targets `main`.

State the resolved `DEST` in your output before reviewing.

## Diff and review
Run `git diff "$DEST...$SRC"` and follow steps 3–4 from @.claude/commands/review-mr.md. The three-dot form is intentional: it reviews changes on `SRC` since the merge-base with `DEST`.
