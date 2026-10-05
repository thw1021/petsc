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

```{rubric} Sys:
```

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

- Change `PCMG` applied with `PCMatApply()` to keep the `MatProduct` of each block residual, restriction, and interpolation between applications, so that their symbolic phase, which for `MATMPIAIJ` times `MATMPIDENSE` allocates work matrices and a `PetscSF`, is not redone on every application; `PCMGMatResidualDefault()` and `PCMGMatResidualTransposeDefault()` keep the product attached to the residual block
- Fix `PCMatApply()` with `PCMG` erroring when called again with blocks of a different leading dimension

```{rubric} KSP:
```

- Fix `KSPMatSolve()` with `KSPRICHARDSON` erroring when called again with a block of solutions of a different leading dimension

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
