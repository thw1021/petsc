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

- Make `SYCL` a `devicePackage`, i.e., builds `--with-sycl` now have `PETSC_HAVE_DEVICE` defined
- Add the option `--with-devicelanguage` to compile `PetscDevice` code using either a C or C++ compiler

```{rubric} Sys:
```

```{rubric} Event Logging:
```

- Add two approaches for GPU energy monitoring:  `-log_view_gpu_energy` and `-log_view_gpu_energy_meter`
- Add API `PetscLogGpuEnergy()`, `PetscLogGpuEnergyMeter()`, `PetscLogGpuEnergyMeterBegin()` and `PetscLogGpuEnergyMeterEnd()` for GPU energy monitoring

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

- Add multi-precision support for MUMPS. One could use `-pc_precision <single, double>` to set the precision to be used by MUMPS, which can be different from `PetscScalar`'s precision

```{rubric} KSP:
```

- Remove `KSPHPDDMPrecision` in favor of `PetscPrecision`

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
- Add `TAOTERMSUM` implementation of `TaoTerm` for scaled, mapped sums of terms
- Add `TAOTERMHALFL2SQUARED` implementation of `TaoTerm` for a typical squared-norm penalty function

```{rubric} PetscRegressor:
```

```{rubric} DM/DA:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

- Add `DMPlexVecGetClosureAtDepth()`

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
