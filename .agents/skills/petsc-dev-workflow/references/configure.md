# Configure PETSc

## Select the architecture

Use the configure help when command details are needed, do not make wild suggesions

```console
$ ./configure --help
```

Unless the user requests multiple architectures, keep using the `PETSC_ARCH` already selected for
the task. Do not switch configurations merely because another existing architecture looks more
convenient.

When the user asks only for a configure command or plan, do not treat the request as authorization
or an intention to execute it. State reasonable assumptions for unresolved choices, such as
creating a new architecture or downloading rather than using a system package, provide the
requested command, and do not ask whether to run it. Do not append build commands unless the user
also asks how to build PETSc.

Before executing configuration when `PETSC_ARCH` is not set, ask the user whether to use an
available configured architecture or create a new one. Before creating one, ask whether the task
needs special configure flags or external packages. Add external packages to an executed
configuration only after the user has specified or approved them.

When proposing or creating a new architecture, use a name supplied by the user; otherwise generate
a concise name from the configuration's purpose and distinguishing options which is not already
present locally. For example, use `arch-debug` for an ordinary debugging build or `arch-opt-hdf5-int64`
for an optimized build with HDF5 and 64-bit indices. Do not include an agent, model, or vendor name
unless it is an internal configuration that is created to test the configure scripts.
Unless the user explicitly asks otherwise, include debugging and strict `PetscErrorCode` checking.
Show configure commands in a `console` code block:

```console
$ ./configure PETSC_ARCH=arch-debug --with-debugging=1 --with-strict-petscerrorcode [user-requested options]
```

When proposing configure commands to users or executing them, do not export variables like `PETSC_ARCH`, `CC`,
and related compilation/linking commands, but explicitly use them as configure arguments.

## Reconfigure

Use `make reconfigure` only when the configure options remain unchanged but something in the
environment, external packages, or configuration scripts under `config/` has changed. It invokes
the generated reconfigure script for the current architecture and first cleans its build products.

When configure options need to change, invoke `./configure` with the complete updated options
instead of using `make reconfigure`. Before configuring into the existing `PETSC_ARCH`, ask whether
the user wants its directory removed first; a clean directory is sometimes necessary to avoid
library conflicts. Do not remove it without explicit authorization. Follow either configure path
with the normal build procedure.

Files under `config/examples/*.py` primarily define CI configurations. Read them for examples and
package-specific configure knowledge; use one directly when reproducing that configuration or
when the user requests it, not as the default way to update a local architecture.

If the current architecture lacks a required capability, report exactly what is missing and ask
whether to modify the configuration. Do not silently add packages or change scalar type, precision,
index size, language support, MPI implementation, or accelerator backend.

On failure, inspect the current architecture's `lib/petsc/conf/configure.log` and report the
relevant error together with the configure command. Do not retry by successively adding speculative
options.
