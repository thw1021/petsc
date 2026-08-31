---
name: petsc-dev-workflow
description: >-
  Configure, build, test, and lint a PETSc checkout, including petsc4py. Use when selecting or
  changing PETSC_ARCH, compiling PETSc or its bindings, discovering or running tests, updating
  expected output, or choosing repository validation commands. Do not use for code navigation or
  a review that does not run development commands.
---

# PETSc development workflow

Adhere to @AGENTS.md.

Use one configured `PETSC_ARCH` throughout the task unless the user explicitly asks to work with
multiple configurations. If the active configuration does not cover a required package, scalar
type, index size, language, or backend, report the limitation and ask before changing the
configuration.

## Route to the relevant procedure

Read only the references needed for the current work:

- Before creating, selecting, or changing an architecture, read
  [references/configure.md](references/configure.md).
- Before building PETSc or petsc4py, regenerating Fortran bindings, or diagnosing build failures,
  read [references/build.md](references/build.md).
- Before discovering, running, debugging, or updating tests, read
  [references/test.md](references/test.md).
- Before formatting or checking C/C++ or `config/` Python, read
  [references/lint.md](references/lint.md).
- For any work under `src/binding/petsc4py/`, additionally read
  [references/petsc4py.md](references/petsc4py.md).

## Shared completion rules

- Inspect `git status --short` before commands that can rewrite files and preserve awareness of
  pre-existing developer changes.
- During iteration, build and run only the tests relevant to the change. Do not start `make test`
  without a selector or `make alltests` unless the user explicitly requests the full suite; either
  command can run thousands of tests.
- For C/C++ changes, complete the final source checks described in `references/lint.md`.
- Report the architecture used, important configure limitations, commands run, failures or skipped
  checks, and any pre-existing developer files processed by formatting.
