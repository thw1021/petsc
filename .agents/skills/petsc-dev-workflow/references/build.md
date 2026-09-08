# Build PETSc

## Iterative build

Use the library target while editing:

```console
$ make -f makefile PETSC_ARCH=arch-name libs
```

The `libs` target is sufficient unless the task includes an external package built after PETSc.

## Optional fortran binding build

When a public C header or API change adds, removes, or changes an interface represented in the
generated Fortran bindings, regenerate them before building only if the current configuration
enables them:

```console
$ make PETSC_ARCH=arch-name fortranbindings
```

## Optional full build

The `all` target also builds configured packages whose definitions under
`config/BuildSystem/config/packages/` set `self.builtafterpetsc = 1`:

```console
$ make PETSC_ARCH=arch-name all
```

If the selected configuration includes such a package and the user has not already decided, ask
whether to run one final `all` build after editing, generation, and formatting. Do not use `all`
repeatedly during iteration unless the task changes that package.
