# Lint and format PETSc

## C and C++

Run `make checkbadSource` for C/C++ changes. Run `make clangformat` only when the working-tree
guard below permits it:

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

`make clangformat` processes all tracked C/C++ files in its formatting set. If tracked C/C++ files
outside the current task have changes, do not run it; report that it was skipped.

## Python configure code

For Python under `config/`, run `make PETSC_ARCH=arch-name vermin`. PETSc does not impose other
general formatter or linter rules there. If `vermin` is unavailable, report that instead of
substituting an unrelated policy.

## Other files

There is no additional mandatory repository-wide lint command for C docstrings, handwritten
Fortran, general documentation, or petsclinter itself. Use justified task-specific checks, but do
not present repository-root `make lint`, a documentation build, or another generic command as PETSc
maintainer policy.
