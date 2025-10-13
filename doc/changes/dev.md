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
- Add `TaoGetTerm()`, `TaoSetTerm()`, and `TaoAddTerm()` for manipulating the objective, gradient, and Hessian evaluation of a `Tao` using `TaoTerm`
- Add `TaoBRGNGetRegularizationType()`, `TaoBRGNSetReguarizationType()`, `TaoBRGNGetRegularizerTerm()` and `TaoBRGNSetRegularizerTerm()` for finer control of `TAOBRGN`
- Remove `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()` and `TaoBRGNSetRegularizerHessianRoutine()`, use `TaoBRGNSetRegulizerTerm()` instead
- Remove `setBRGNRegularizerObjectiveGradient()`, and `setBRGNRegularizerHessian()` Python routines. Use `setBRGNRegularizerTerm()` instead
- Remove many ADMM related operations as it has been reimplemented using `TaoTerm`: `TaoADMMRegularizerType` enum (including `TAO_ADMM_REGULARIZER_USER`, `TAO_ADMM_REGULARIZER_SOFT_THRESH` values), `TaoGetADMMParentTao()`, `TaoADMMSetConstraintVectorRHS()`, `TaoADMMSetRegularizerCoefficient()`, `TaoADMMGetRegularizerCoefficient()`, `TaoADMMSetMisfitConstraintJacobian()`, `TaoADMMSetRegularizerConstraintJacobian()`, `TaoADMMSetRegularizerHessianRoutine()`, `TaoADMMSetRegularizerObectiveAndGradientRoutine()`, `TaoADMMSetMisfitHessianRoutine()`, `TaoADMMSetMisfitObjectiveAndGradientRoutine()`, `TaoADMMSetMisfitHessianChangeStatus()`, `TaoADMMSetRegHessianChangeStatus()`, `TaoADMMSetRegularizerType()`, `TaoADMMGetRegularizerType()`
- Add `TaoADMMSetTermGroups()` and `TaoADMMGetTermGroups()`

```{rubric} TaoTerm:
```

- Add `TAOTERMCALLBACKS` implementation of `TaoTerm` constructed from the callbacks passed to a `Tao` object
- Add `TAOTERMSHELL` implementation of `TaoTerm` for user-defined callbacks
- Add `TAOTERMSUM` implementation of `TaoTerm` for scaled, mapped sums of terms
- Add `TAOTERMHALFL2SQUARED` implementation of `TaoTerm` for a typical squared-norm penalty function
- Add `TAOTERML1` implementation of `TaoTerm` for a typical 1-norm penalty function
- Add `TAOTERMQUADRATIC` implementation of `TaoTerm` for a quadratic penalty function
- Add Python routines for `TaoTerm`

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
