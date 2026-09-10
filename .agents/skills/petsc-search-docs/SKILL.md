---
name: petsc-search-docs
description: Look up PETSc documentation (manual pages, users manual, FAQ, install guide) from the locally built docs and answer with a citation. Use when the user asks what a PETSc function, type, or option does, how to use a PETSc feature, or where something is documented, or runs `/petsc-search-docs question`. Do not use for editing code.
argument-hint: <question or PETSc identifier>
---

Answer from the locally built docs, not from memory and not from the source tree. The
question is `$ARGUMENTS`, or the user's last question if empty.

## Decide how to answer

**One exact identifier** (a single `CamelCase` name such as `KSPGMRESSetRestart`, `PCFIELDSPLIT`,
`MatAssemblyType`, and nothing conceptual around it): answer inline. One trimmed manual page is
about 2 KB, cheaper than dispatching an agent.

```bash
cd "$PETSC_DIR"
DOC=""; for d in ${PETSC_ARCH}-doc $(ls -dt arch-*-doc); do [ -d "$d/manualpages" ] || continue; DOC=${DOC:-$d}; [ -d "${d%-doc}/tantivy/index/docs" ] && { DOC=$d; break; }; done
[ -n "$DOC" ] || { echo "no arch-*-doc build found; run make docs"; exit 0; }
for f in $DOC/manualpages/*/NAME.md; do echo "== $f"; sed '/^## Location/,$d' "$f"; done
```

If the identifier is a setter and its page lists no options key, check the owning type page
before saying there is none (`PCFIELDSPLIT.md` holds the key for `PCFieldSplitSetBlockSize()`):

```bash
for f in $(grep -l NAME $DOC/manualpages/*/*.md | grep '/[A-Z0-9_]*\.md$'); do echo "== $f"; sed -n '/^## Options Database/,/^## /p' "$f" | grep -i KEYWORD; done
```

**Anything else** (a concept, a how-to, "which option controls...", several identifiers, or the
inline page did not answer): dispatch the `petsc-search-docs` subagent with the question verbatim and
any needed context in one self-contained prompt. Relay its answer and its `Sources:` list,
including the `Doc build:` line. Do not re-read the pages it cited.

If the `petsc-search-docs` subagent type is not registered, run a general-purpose subagent instead and
tell it to read and follow `.claude/agents/petsc-search-docs.md` as its role.

## Reply

Lead with the answer, then `Sources:` with the public URL for each page used
(`manualpages/KSP/KSPSolve.md` is https://petsc.org/main/manualpages/KSP/KSPSolve/,
`manual/ksp.md` is https://petsc.org/main/manual/ksp/). Name the doc build when it is not
`${PETSC_ARCH}-doc`, since it may come from another branch. Say plainly when the docs do not
answer the question.
