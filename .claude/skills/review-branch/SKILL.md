---
name: review-branch
description: Review a local PETSc branch's changes against its target branch (origin/main or origin/release). Use when the user asks to "review this branch", "review my changes", "check what I've done before pushing", or runs `make branch-review`.
argument-hint: <branch | commit-ref | empty for HEAD>
---

Adhere to @CLAUDE.md.

`SRC` is `$ARGUMENTS` if given, else `HEAD`. Reject anything that isn't a single ref matching `^[A-Za-z0-9._/@][A-Za-z0-9._/@~^-]*$`.

Resolve `DEST`:

```
MB=$(git merge-base origin/main <SRC>) && git merge-base --is-ancestor "$MB" origin/release && echo origin/release || echo origin/main
```

If `origin/main` doesn't resolve (`git rev-parse --verify -q origin/main` exits non-zero), first run `git fetch -q --no-tags origin +release:refs/remotes/origin/release +main:refs/remotes/origin/main`, then retry. Any other failure: abort and report — do not guess `DEST`.

State `DEST`, then size with `git diff --shortstat <DEST>...<SRC> -- ':(exclude)*.out'`. Capture to a file: `git diff --name-status <DEST>...<SRC> > branch-review.diff` (index of every changed file, `.out` included, for flagging reference-output mismatches), then `git diff <DEST>...<SRC> -- ':(exclude)*.out' >> branch-review.diff` (full diffs, `.out` bodies omitted). If `branch-review.diff` is empty, stop and report — no changes, or check `DEST`/`SRC`. Read it in full (in parts if large). Do **not** re-run `git diff` per file. Options (e.g. `--shortstat`) must precede the revision args, not follow the pathspec.

Then follow @../review-mr/review-procedure.md (Sections 4–6) to classify, verify, and report findings. Report to stdout only — skip Section 7; do not write `ai-review.html`.
