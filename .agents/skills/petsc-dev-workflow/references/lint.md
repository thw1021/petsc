# Lint and format PETSc

## C and C++

Before concluding any C/C++ change, run both repository checks:

```console
$ make checkbadSource
$ make clangformat
```

PETSc hardcodes the required clang-format major version. If the default executable has the wrong
version, select the correct executable through the make variable:

```console
$ make PETSCCLANGFORMAT=/path/to/clang-format-22 clangformat
```

`make clangformat` processes the repository's tracked C/C++ formatting set, including pre-existing
developer changes. Record the modified tracked files before running it, run it even when unrelated
developer changes exist, and report that those pre-existing files were also processed and may have
been modified. Never revert those changes.

## Python configure code

For Python under `config/`, run `make vermin`; PETSc does not impose other general formatter or
linter rules there. If `vermin` is unavailable, report that the check could not be run rather than
substituting an unrelated policy.

## Currently unspecified categories

This skill does not prescribe an additional mandatory lint command for C docstrings, handwritten
Fortran, general documentation, or changes to petsclinter itself. In particular, do not add a
repository-root `make lint`, a documentation build, or another generic validation command as a
mandatory check for these categories. C/C++ files still receive the checks above. Recommend a
task-specific check when directly justified, or use a command requested by the user or CI, but
identify it as task-specific rather than PETSc maintainer policy.
