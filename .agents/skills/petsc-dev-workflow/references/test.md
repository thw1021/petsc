# Test PETSc

## Define tests

PETSc harness tests are described in `/*TEST ... TEST*/` blocks, normally at the bottom of source
files. Common test-block keys are `test`, `testset`, `suffix`, `nsize`, `args`, `requires`,
`output_file`, `filter`, `filter_output`, `localrunfiles`, `temporaries`, `timeoutfactor`, and `env`.
Use `requires:` for runtime requirements such as packages, precision, `!complex`, or
`datafilespath`. Expected output normally lives in `output/<testname>.out` relative to the source
file.

Reuse an existing test source when possible, adding a `/*TEST*/` variant with arguments drawn from
nearby tests or generated scripts. Create a source only if none can naturally exercise the
behavior or the user explicitly requests a new one. Harness tests apply metadata and expected-output
comparison; for focused diagnosis or an unregistered scenario, run the executable directly and
state what it validates. Use `testset` inheritance or loops for variants sharing setup and output;
use separate suffixes when variants need separate outputs. Do not invent a broad option matrix
merely to increase coverage.

## Select tests

Test definitions are generated as needed. Use `print-test` to inspect every selection before using
the same selectors with `make test` to run it.

```console
# Select by source path, directory, or target glob
$ make -f gmakefile PETSC_ARCH=arch-name print-test s='src/ksp/ksp/tests/ex1.c'

# Narrow that selection by target name
$ make -f gmakefile PETSC_ARCH=arch-name print-test s='src/ksp/ksp/tests/ex1.c' i='*ex1_1'

# Find tests exercising an external package
$ make -f gmakefile PETSC_ARCH=arch-name print-test query='requires' queryval='hdf5'

# Find all source definitions requiring GPU-aware MPI, including unavailable tests
$ ./config/query_tests.py --use-source --petsc-dir="$PETSC_DIR" requires '*GPU_AWARE*'

# Query /*TEST*/ args (leading option dashes are omitted)
$ make -f gmakefile PETSC_ARCH=arch-name print-test query='args' queryval='*pc_type*bddc*'

# Show all harness options
$ make -f gmakefile.test PETSC_ARCH=arch-name help
```

`query` names a test-level `/*TEST*/` keyword and `queryval` matches its value; neither inspects
source or runtime types. `query='requires'` searches runtime requirements, not `build: requires:`.
`make` queries cover only tests generated for the active `PETSC_ARCH`;
`config/query_tests.py --use-source` includes tests the configuration cannot run. Argument queries
omit leading dashes and ignore numbers, loop values, and input files. Use a target glob such as
`s='*bddc*'` for broader name-based candidates; verify runtime use in source when needed.

Targeted controls: `NO_RM=1` retains executables; `PRINTONLY=1` prints without running (only the
first command for loops); `V=1` shows loop commands; `EXTRA_OPTIONS=` appends options; `DEBUG=1`
and `VALGRIND=1` instrument runs; and `test-fail=1` reruns previous failures. See
`doc/developers/testing.md` for generated scripts and complex queries; inspect
`config/query_tests.py` only when selector implementation details are needed.

## Diagnose failures

Use `make -f gmakefile PETSC_ARCH=arch-name print-test test-fail=1` to list failures from the
previous run. Inspect the generated scripts and logs under `$PETSC_ARCH/tests/`; the `test*err.log`
files provide the aggregate failure details. Reproduce the narrowest failing test rather than
rerunning a broad selection.

## Update expected output

Use `REPLACE=1` only when the new output is known to be correct. After replacement:

1. Inspect every changed `output/*.out` file.
2. Rerun the selected test without `REPLACE=1`.
3. Require a clean result under the test's specific `petscdiff` invocation.

Normal `petscdiff` use does not compare floating-point numbers unless the test has `-j` in
`diff_args` or `DIFF_NUMBERS=1` is requested.

Do not use `ALT=1` or generated-script `-M` to create or replace alternative output files. PETSc
tries to avoid new alternate outputs; leave any exceptional update to the user.
