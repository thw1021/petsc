# Lint and format PETSc

## C and C++

Run both repository checks for C/C++ changes:

```console
$ make checkbadSource
$ make clangformat
```

PETSc hardcodes the required clang-format major version. If the default executable has the wrong
version, select the correct executable through the make variable PETSCCLANGFORMAT:

```console
$ make PETSCCLANGFORMAT=/path/to/clang-format clangformat
```

Inspect the makefile rule `checkclangformatversion` in `lib/petsc/conf/rules_util.mk` to infer the
needed version.

`make clangformat` processes all tracked C/C++ files in its formatting set, including files with
pre-existing changes. Record changed files before running it, then report which pre-existing files
were processed and may have changed. Never revert those changes.

## Python configure code

For Python under `config/`, run `make PETSC_ARCH=arch-name vermin`. PETSc does not impose other
general formatter or linter rules there. If `vermin` is unavailable, report that instead of
substituting an unrelated policy.

## Other files

There is no additional mandatory repository-wide lint command for C docstrings, handwritten
Fortran, general documentation, or petsclinter itself. Use justified task-specific checks, but do
not present repository-root `make lint`, a documentation build, or another generic command as PETSc
maintainer policy.
