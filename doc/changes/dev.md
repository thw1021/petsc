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

- Add `VecNestGetSubVecsRead()` and `VecNestRestoreSubVecsRead()` for read-only access to subvectors
- Add `VecPointwiseSign()` and `VecSignMode`

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

- Add ``TaoBRGNSetRegularizationType()``, ``TaoBRGNGetRegularizationType()``
- Add `TaoGetInequalityConstraintsRoutine()`, `TaoGetEqualityConstraintsRoutine()`, `TaoGetJacobianInequalityRoutine()` and `TaoGetJacobianEqualityRoutine()`
- Add new `TaoTerm` object to manipulate objective function terms with many methods

```{rubric} TaoTerm:
```

- Add `TAOTERMTAOCALLBACKS` implementation of `TaoTerm` for constructing a term from the callbacks passed to a `Tao` object
- Add `TAOTERMBRGNREGULARIZER` implementation of `TaoTerm` for constructing a term from the callbacks passed to a `TaoBRGNSetReguarizerObjectiveAndGradientRoutine()`
- Add `TAOTERMADMMREGULARIZER` implementation of `TaoTerm` for constructing a term from the callbacks passed to a `TaoADMMSetReguarizerObjectiveAndGradientRoutine()`
- Add `TAOTERMADMMISFIT` implementation of `TaoTerm` for constructing a term from the callbacks passed to a `TaoADMMSetMisfitObjectiveAndGradientRoutine()`
- Add `TAOTERMSHELL` implementation of `TaoTerm` for user-defined callbacks

```{rubric} PetscRegressor:
```

```{rubric} DM/DA:
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
