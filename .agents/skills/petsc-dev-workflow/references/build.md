# Build PETSc

## Decide whether to run a final full build

Before the first build that will actually be run, establish whether the user wants one final
`make -jN all` after all edits, generation, and formatting. If the user has not already said, ask
once. Store the answer for the current task, do not ask again, and do not change the choice unless
the user asks. In a command-only or validation-plan response, describe the final full build as
conditional on that choice rather than including it as a required step.

## Iterative build

Use the library target while editing:

```console
$ make -jN libs
```

Choose a reasonable positive `N` for the host. The `libs` target is normally sufficient unless the
task explicitly includes an external package built after PETSc.

PETSc compilation is warning-free. Inspect the compiler output and treat warnings emitted while
compiling PETSc sources as issues to investigate and report.

When a public C header or API change adds, removes, or changes an interface represented in the
generated Fortran bindings, regenerate them before building:

```console
$ make fortranbindings
$ make -jN libs
```

Inspect the generated diff rather than assuming regeneration is harmless.

## Optional final full build

If the user chose a final full build, run it after all edits, generation, and formatting:

```console
$ make -jN all
```

Some packages under
`config/BuildSystem/config/packages/` set `self.builtafterpetsc = 1`; the final `all` build checks
that PETSc API changes have not broken those configured dependents. Avoid repeatedly using `all`
during the edit/build loop unless the task is also changing such a package. If the user declined
the final build, do not run it and report that it was omitted by agreement.

On any build failure, inspect the compiler diagnostics produced by the command. A `make all`
invocation also writes `$PETSC_ARCH/lib/petsc/conf/make.log`; inspect that log for `make all`
failures. Report the architecture, failed command, and relevant diagnostics. Do not clean or change
configurations merely to see whether the failure disappears.
