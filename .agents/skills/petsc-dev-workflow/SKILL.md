---
name: petsc-dev-workflow
description: >-
  Configure, build, test, and lint PETSc and petsc4py, and build PETSc documentation. Use when
  selecting or changing PETSC_ARCH, compiling, discovering or running tests, updating expected
  output, choosing repository validation commands, or running an explicitly authorized
  documentation build. Do not use for code navigation or reviews that do not run development
  commands.
---

# PETSc development workflow

Adhere to @AGENTS.md.

Keep one configured `PETSC_ARCH` throughout the task unless the user requests multiple
configurations. If it lacks a required capability, report the limitation and ask before changing
it. Reference commands use `arch-name` as a placeholder for the selected architecture.

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
- Before checking, running, or diagnosing PETSc documentation, read
  [references/docs.md](references/docs.md).
- For any work under `src/binding/petsc4py/`, additionally read
  [references/petsc4py.md](references/petsc4py.md).

## Shared completion rules

- During iteration, build and test only what is relevant to the change. Do not run `make test`
  without a selector or `make alltests` unless the user requests the full suite; either command can
  start thousands of tests.
- For C/C++ and petsc4py changes, complete the final source checks described in `references/lint.md`.
- Report the architecture used, important configure limitations, commands run, failures or skipped
  checks
