---
name: petsc-lint
description: >-
  Format PETSc C/C++ and handwritten Fortran, and run checks for C docstrings, Python configure
  code, and petsc4py. Also explain CI checks for manual-page metadata, shell scripts, and
  petsclinter. Use for source formatting or check failures. Documentation builds are covered
  by petsc-docs.
---

# Lint and format PETSc and petsc4py

Read and follow `AGENTS.md` at the PETSc repository root. Run commands from that root except
where a different directory is stated. `arch-name` is a placeholder for the architecture supplied
by the user. Apply only the checks relevant to the changed files.

The repository-root `make lint` runs PETSc's C/C++ source and docstring linter. The `make lint`
target in `src/binding/petsc4py/` runs the bindings' Cython and Ruff checks. Select the directory
and target according to the source being checked.

## C and C++

Formatting is controlled by the repository's `.clang-format` file. Run `make clangformat` for
C/C++ changes:

```console
$ make clangformat
```

PETSc hardcodes the required clang-format major version. If the default executable has the wrong
version, select the correct executable through the make variable PETSCCLANGFORMAT:

```console
$ make PETSCCLANGFORMAT=/path/to/clang-format clangformat
```

Inspect the `checkclangformatversion` rule in `lib/petsc/conf/rules_util.mk` to determine the
required version. If a compatible executable is unavailable, report the required version and
that formatting could not be completed.

`make clangformat` processes all tracked C/C++ files in its formatting set. Compare the result
with the working tree before formatting to preserve existing edits and keep the task's patch
focused. Inspect the formatter output as well as the diff: the recipe ignores formatter errors,
so a successful `make` exit alone does not establish that formatting completed.

## Handwritten Fortran

Run `make fprettify` for changes to handwritten `.h90` or `.F90` files. It requires `fprettify`
and reformats all tracked files with those suffixes. Inspect the diff and preserve unrelated
edits.

## C docstrings

The repository-root `make lint` invokes `petsclinter`. It is an optional check for C docstrings,
not a mandatory completion step for every source change. When requested or useful for a specific
change, restrict it to the affected directory, for example:

```console
$ make PETSC_ARCH=arch-name lint DIRECTORY=src/ksp/ksp/interface
```

It requires the Python `clang` package and a compatible `libclang`. Inspect `help-lint` and the
requirements under `lib/petsc/bin/maint/petsclinter/` when dependency or invocation details are
needed. If the dependencies are unavailable, report the limitation and inspect the docstrings
against the conventions in `AGENTS.md`.

For Markdown and Sphinx documentation work, follow [petsc-docs](../petsc-docs/SKILL.md).

## Python configure code

For Python under `config/`, run `make PETSC_ARCH=arch-name vermin`. PETSc does not impose other
general formatter or linter rules there. If `vermin` is unavailable, report that instead of
substituting an unrelated policy.

## CI checks for reference

These descriptions explain CI behavior; they do not add local verification requirements. Run
these CI checks only when the user specifically requests them, whether through these targets
or equivalent commands.

- `make checkfprettifyformat`: The `checksource` job requires a clean tracked working tree,
  runs `make fprettify`, and fails if formatting changes it.
- `make checkbadSource`: The `checksource` job checks source conventions defined in
  `lib/petsc/conf/rules_util.mk` and reports violations.
- `make checkbadManualPages`: The `checksource` job invokes `lib/petsc/bin/getAPI.py` to check
  manual-page metadata, including `.seealso:` formatting. It does not build documentation.
- `make checkshellcheck`: The `checksource` job applies ShellCheck's suggested patches to the
  scripts selected by the root makefile, then fails if the tracked working tree differs from
  `HEAD`. This target modifies files.
- `make test-lint`: The `linux-analyzer` job runs Vermin, MyPy, package consistency checks, and
  petsclinter's regression tests.

## petsc4py

Before concluding a petsc4py code change, run from `src/binding/petsc4py/`:

```console
$ make PETSC_ARCH=arch-name lint
```

This target recreates `petsc4py-lint-env`, installs the lint requirements, and runs Cython and
Ruff checks. Inspect failures before applying fixes; report unavailable tools or dependencies.
All new petsc4py code must be documented. A documentation build is a separate task covered by
[petsc-docs](../petsc-docs/SKILL.md).
