# Test PETSc

## Select tests

Test definitions are generated automatically when the `test` target is invoked. Use the harness
help when command details are needed:

```console
$ make -f gmakefile.test help
```

The `s`/`search` selector accepts a source path, directory, or target glob. It can be passed to
`print-test` to inspect matching targets or directly to `test` when the relevant selection is
already known:

```console
$ make -f gmakefile print-test s='src/package/path/file.c'
$ make test s='src/package/path/file.c'
```

Use `searchin` (or its alias `i`) to filter the results of an `s`/`search` selection when a source
path or glob covers more tests than are relevant:

```console
$ make -f gmakefile print-test s='src/package/path/' searchin='*specific_test*'
$ make test s='src/package/path/' searchin='*specific_test*'
```

Tests can also be selected by fields from their `/*TEST*/` definitions. For example, discover
tests requiring an external package with:

```console
$ make -f gmakefile print-test query='requires' queryval='external_package_name'
```

Choose the selection method appropriate to the change. `print-test` is useful for unfamiliar or
potentially broad selections, but it need not precede a known targeted test.

Use existing `/*TEST*/` definitions, generated scripts, and nearby tests as evidence for meaningful
argument combinations. A registered harness test is useful when it covers the changed behavior
because it also applies the test metadata and normal expected-output comparison. Direct executable
invocations remain appropriate for focused diagnosis or a scenario not represented by a registered
test; state what they validate and recognize that they do not perform the harness's expected-output
comparison. Do not invent a broader matrix of options merely to increase coverage. If the existing
tests do not exercise the changed behavior, report that gap and decide from the task whether a
focused direct run or a test-definition change is appropriate.

Do not run `make test` without a selector or `make alltests` unless the user explicitly requests the
full suite; both can start thousands of tests. If the active configuration skips a required case,
report the missing configuration feature and ask before reconfiguring.

Useful targeted controls include `NO_RM=1` to retain executables, `PRINTONLY=1` to print a command,
`NP=` to override process count, `EXTRA_OPTIONS=` to append options, `DEBUG=1`, `VALGRIND=1`, and
`test-fail=1` to rerun the previous failures. Consult `doc/developers/testing.md` for the generated
test-script workflow and more complex queries.

## Diagnose failures

Use `make -f gmakefile print-test test-fail=1` to list failures from the previous run. Inspect the
generated scripts and logs under `$PETSC_ARCH/tests/`; the `test*err.log` files provide the
aggregate failure details. Reproduce the narrowest failing test rather than rerunning a broad
selection.

## Update expected output

Use `REPLACE=1` only when the new output is known to be correct. After replacement:

1. Inspect every changed `output/*.out` file.
2. Rerun the selected test without `REPLACE=1`.
3. Require a clean result under the test's normal `petscdiff` invocation.

Normal `petscdiff` use does not compare floating-point numbers unless `-j` or
`DIFF_NUMBERS=1` is requested; do not impose numeric-token equality on an ordinary test run.

Do not use `ALT=1` or generated-script `-M` to create or replace alternative output files. PETSc
avoids new alternate outputs; leave any exceptional update to the user.
