# Configure PETSc

## Select the architecture

Consult configure help when command details are needed instead of guessing options:

```console
$ ./configure --help
```

Keep using the `PETSC_ARCH` selected for the task. Do not switch merely because another existing
architecture looks more convenient.

When the user asks only for a configure command or plan, state any assumptions and provide it
without executing it or appending build commands.

Before executing configuration when no architecture is selected, ask whether to reuse an existing
one or create a new one. If the requirements for a new architecture are unspecified, ask about
special configure flags and external packages. Add external packages only after the user specifies
or approves them.

Use a user-supplied architecture name. Otherwise, derive a concise unused name from the purpose and
distinguishing options, such as `arch-debug` or `arch-opt-hdf5-int64`. Do not identify the agent or
model in the name.
Unless the user explicitly asks otherwise, include debugging and strict `PetscErrorCode` checking:

```console
$ ./configure PETSC_ARCH=arch-debug --with-debugging=1 --with-strict-petscerrorcode=1 [user-requested options]
```

Fortran bindings are enabled by default if a suitable compiler is found.

Pass configuration values directly as arguments instead of relying on exported environment
variables, which configure sanitizes during compiler and tool detection. This includes

- Compilers: `CC`, `CXX`, `FC`, `F77`, `F90`
- Compiler and preprocessor settings: `CFLAGS`, `CXXFLAGS`, `FCFLAGS`, `FFLAGS`, `F90FLAGS`,
  `CPP`, `CPPFLAGS`, `CXXPP`, `CXXPPFLAGS`
- Link settings: `LDFLAGS`, `LIBS`
- MPI, make, and archive tools: `MPI_DIR`, `RM`, `MAKEFLAGS`, `AR`, `RANLIB`

```console
$ ./configure PETSC_ARCH=arch-name CC=mpicc CFLAGS='-g -O0' [other options]
```

On failure, inspect the current architecture's `lib/petsc/conf/configure.log` and report the
relevant error together with the configure command. Do not retry by successively adding speculative
options.

## Reconfigure

Use `make reconfigure` only when the configure options remain unchanged but the environment,
external packages, or configuration scripts under `config/` changed. It cleans the current build
products and invokes the generated reconfigure script.

```console
$ make PETSC_ARCH=arch-name reconfigure
```

When configure options need to change, invoke `./configure` with the complete updated options
instead of using `make reconfigure`. A clean architecture directory is sometimes necessary to
avoid library conflicts; ask before removing an existing one.

Files under `config/examples/*.py` primarily define CI configurations. Consult them for
package-specific knowledge, but use one directly only to reproduce that configuration or when the
user requests it.

If the current architecture lacks a required capability, report exactly what is missing and ask
whether to modify the configuration. Do not silently add packages or change scalar type, precision,
index size, language support, MPI implementation, or accelerator backend.
