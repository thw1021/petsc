# Configure PETSc

## Configure the supplied architecture

Consult configure help when command details are needed instead of guessing options:

```console
$ ./configure --help
```

When the user asks only for a configure command or plan, state any assumptions and provide it
without executing it or appending build commands.

Before creating a new configuration, ask about special configure flags and external packages when
the user has not specified them. Add external packages only after the user specifies or approves
them.

Unless the user explicitly asks otherwise, include debugging and strict `PetscErrorCode` checking:

```console
$ ./configure PETSC_ARCH=arch-name --with-debugging=1 --with-strict-petscerrorcode=1 [user-requested options]
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

If the supplied architecture lacks a required capability, report exactly what is missing. Adding
packages or changing scalar type, precision, index size, language support, MPI implementation, or
accelerator backend changes the configuration.
