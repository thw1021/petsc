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

```{rubric} Sys:
```

- Remove `petscoptions.h` from `petscsys.h`; code that calls the options database routines, such as `PetscOptionsGetInt()` and `PetscOptionsBegin()`, must now include `petscoptions.h` explicitly
- Remove `petscviewer.h` from `petscsys.h`; code that calls the viewer routines must now include `petscviewer.h` explicitly
- Reduce the include dependencies of `petscviewer.h`, `petscdraw.h`, `petscoptions.h`, `petscbt.h`, `petsclog.h`; code that relied on these headers
  to pull in `petscsys.h` must include those headers directly

- Move the PETSc memory-management API (`PetscMalloc()`, `PetscFree()`, and related routines) into the new header `petscmem.h`; it remains available through `petscsys.h`

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

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

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
