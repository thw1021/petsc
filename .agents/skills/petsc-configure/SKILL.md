---
name: petsc-configure
description: >-
  Configure or reconfigure PETSc, including the PETSc configuration used by petsc4py.
  Use when selecting configure options, changing a user-supplied PETSC_ARCH, or diagnosing
  configure failures. Also use for configure command proposals that should not be executed.
---

# Configure PETSc

Read and follow `AGENTS.md` at the PETSc repository root. Run commands from that root;
`arch-name` is a placeholder for the architecture supplied by the user.

## Configure the supplied architecture

Consult configure help when command details are needed instead of guessing options:

```console
$ ./configure --help
```

When the user asks only for a configure command or plan, state any assumptions and provide it
without executing it or appending build commands. An architecture placeholder is sufficient for
a proposal; obtain the user's architecture before executing configure.

Use the configure flags and external packages already specified or approved for the task. Ask
only when an unresolved choice, such as the compiler, MPI implementation, or accelerator backend,
materially affects the requested configuration. Otherwise, use the development defaults below
and configure's defaults for a new configuration. Preserve existing settings when reconfiguring.
Do not add unrequested external packages.

For a new development configuration, include debugging and strict `PetscErrorCode` checking
unless the user requests different settings:

```console
$ ./configure PETSC_ARCH=arch-name --with-debugging=1 --with-strict-petscerrorcode=1 [user-requested options]
```

Fortran bindings are enabled by default if a suitable compiler is found.

Pass configuration values through supported command-line options instead of relying on exported
compiler and tool variables, which configure normally sanitizes. For example, use `CC`, `CXX`,
`FC`, `CFLAGS`, `CXXFLAGS`, `FFLAGS`, `LDFLAGS`, or `LIBS` as arguments. Use `--with-mpi-dir`
to select an MPI installation and `--with-make-np` to set build parallelism; do not assume that
every sanitized environment variable is also a supported configure option.

```console
$ ./configure PETSC_ARCH=arch-name CC=mpicc CFLAGS='-g -O0' [other options]
```

On failure, inspect `arch-name/lib/petsc/conf/configure.log` and report the relevant error
together with the configure command. If configure fails before that log is created or updated,
inspect the command output and the repository-root `configure.log`, if produced by this run.
Do not retry by successively adding speculative options.

## Reconfigure

When configure options need to change, invoke `./configure` with the complete updated options
instead of using `make reconfigure`. Recover the previous options from the generated reconfigure
script under `arch-name/lib/petsc/conf/`, when available, and preserve settings outside the
requested change. A clean architecture directory is sometimes necessary to avoid library
conflicts; ask before removing an existing one.

Files under `config/examples/*.py` primarily define CI configurations. Consult them for
package-specific knowledge, but use one directly only to reproduce that configuration or when the
user requests it.

If the supplied architecture lacks a required capability, report exactly what is missing. Adding
packages or changing scalar type, precision, index size, language support, MPI implementation, or
accelerator backend changes the configuration.
