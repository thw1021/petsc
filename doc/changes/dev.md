# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

```{rubric} Configure/Build:
```

- Remove the `--with-serialize-functions` configure option and the associated function-pointer serialization feature (`PETSC_SERIALIZE_FUNCTIONS`)
- Remove the `PetscPreLoad*` functionality and code that used it

```{rubric} Sys:
```

- Remove `petscoptions.h` from `petscsys.h`; code that calls the options database routines, such as `PetscOptionsGetInt()` and `PetscOptionsBegin()`, must now include `petscoptions.h` explicitly
- Remove `petscviewer.h` from other commonly used include files; code that calls the viewer routines must now include `petscviewer.h` explicitly
- Reduce the include dependencies of `petscviewer.h`, `petscdraw.h`, `petscoptions.h`, `petscbt.h`, `petsclog.h`, which no longer force the inclusion of `petscsys.h`;
  code that relied on these headers to pull in `petscsys.h` must include `petscsys.h`, for example, explicitly
- Move the PETSc memory-management API (`PetscMalloc()`, `PetscFree()`, and related routines) into the new header `petscmem.h`; the API remains available through `petscsys.h`

```{rubric} Event Logging:
```

```{rubric} PetscViewer:
```

```{rubric} PetscDraw:
```

```{rubric} AO:
```

```{rubric} IS:
```

```{rubric} VecScatter / PetscSF:
```

```{rubric} PF:
```

```{rubric} Vec:
```

- Add a device implementation of `VecSqrtAbs()` for `VECKOKKOS`; previously it copied the vector to the host

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

- Add device implementations of `MatNorm()` with `NORM_1`, `NORM_FROBENIUS`, and `NORM_INFINITY` for `MATAIJKOKKOS`; previously all norms copied the matrix values to the host

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

```{rubric} KSP:
```

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

```{rubric} TaoTerm:
```

```{rubric} PetscRegressor:
```

```{rubric} PetscDA:
```

```{rubric} DM:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

```{rubric} FE/FV:
```

```{rubric} DMNetwork:
```

```{rubric} DMStag:
```

```{rubric} DT:
```

```{rubric} Fortran:
```
