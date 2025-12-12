(ch_da)=

# PetscDA: Ensemble Data Assimilation

PETSc's `PetscDA` object coordinates ensemble-based data assimilation (DA) workflows. It centralizes ensemble storage, observational metadata, and user-defined forecast/analysis operators so that algorithms can run independently of the MPI layout or the vector/matrix backends. The PetscDA layer currently focuses on ensemble transform Kalman filter (ETKF)-style updates but is extensible to other assimilation techniques that follow the same interfaces.

(sec_da_lifecycle)=

## Lifecycle overview

A typical assimilation cycle alternates between forecast propagation and statistical analysis:

1. Initialize a `PetscDA` context and configure ensemble sizes and data structures.
2. Populate the ensemble state vectors and optional observation-error descriptions.
3. Advance each ensemble member with a model operator supplied by the application.
4. Combine forecasts with observations through `PetscDAAnalysis()` to produce the posterior ensemble.
5. Repeat until the desired simulation horizon is complete, optionally extracting diagnostics after each phase.

Throughout this loop the `PetscDA` object abstracts the global vectors, scatters, and reductions needed to compute ensemble means, anomalies, and square-root transforms.

(sec_da_creating)=

## Creating a PetscDA context

Create, configure, and destroy a `PetscDA` object with the standard PETSc object lifecycle:

```c
PetscDA da;
PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
PetscCall(PetscDASetType(da, PETSCDAETKF));
PetscCall(PetscDASetSizes(da, ensemble_size, local_state_size, observation_size));
PetscCall(PetscDASetFromOptions(da));
PetscCall(PetscDASetUp(da));

/* ... data assimilation loop ... */

PetscCall(PetscDADestroy(&da));
```

`PetscDASetSizes()` records (1) the number of ensemble members, (2) the local state dimension per MPI rank, and (3) the number of observations that will be processed simultaneously. After `PetscDASetUp()` the object owns ensemble storage and is ready to hand out views on members.

(sec_da_ensemble)=

## Managing ensembles

`PetscDA` stores ensemble members as PETSc `Vec` objects and exposes convenience helpers to access them safely:

- `PetscDAGetEnsembleMember()` / `PetscDARestoreEnsembleMember()` map ensemble indices to `Vec` handles that participate in PETSc's reference counting.
- `PetscDASetEnsembleMember()` lets applications inject externally created vectors into specific slots, which is useful when importing state snapshots from disk or another solver component.
- `PetscDAComputeMean()` forms the sample mean across all members.
- `PetscDAComputeAnomalies()` returns a tall-and-skinny `Mat` whose columns are the mean-subtracted ensemble anomalies. Many square-root filters use this matrix to construct low-rank covariance factorizations.

(sec_da_analysis)=

## Analysis step

Observation-error variances (or more general descriptions) are supplied through `PetscDASetObsErrorVariance()`. The associated vector is assumed to follow the global observation ordering; `PetscDAGetObsErrorVariance()` returns the stored object for later inspection or reuse.

`PetscDAAnalysis()` performs the assimilation step by calling a user-provided observation operator:

```c
/* Prototype for observation operator H(x) */
PetscErrorCode ObservationOperator(Vec state, Vec prediction, void *ctx) {
  /* Map model state -> observation space */
  return PETSC_SUCCESS;
}

PetscCall(PetscDAAnalysis(da, observation_vec, ObservationOperator, user_ctx));
```

The callback must fill `prediction` with the modelled observations corresponding to `state`, respecting the layout dictated by `PetscDASetSizes()`. `PetscDAAnalysis()` handles all ensemble reductions, gain computations, and posterior updates.

(sec_da_model)=

## Forecast step

`PetscDAApplyModel()` wraps the forecast step. The user supplies a function that advances a single ensemble member:

```c
/* Prototype for model forecast M(x) */
PetscErrorCode ModelOperator(Vec x_in, Vec x_out, void *ctx) {
  /* Advance x_in by dt to produce x_out */
  /* (e.g., step a TS object) */
  return PETSC_SUCCESS;
}

PetscCall(PetscDAApplyModel(da, ModelOperator, ts_ctx));
```

The operator can call into PETSc time integrators ({any}`ch_ts`), nonlinear solvers ({any}`ch_snes`), or bespoke device kernels. The PetscDA layer orchestrates calls across the entire ensemble, issuing them in rank-local loops while ensuring that ownership and recycling semantics remain correct.

(sec_da_impls)=

## Implementations

The default implementation is the ensemble transform Kalman filter indicated by the type string `PETSCDAETKF` (which resolves to `"etkf"`). Alternative PetscDA types can be registered with `PetscDARegister()` and selected at runtime:

- `-petscda_type etkf` chooses the built-in square-root ETKF.
- `-da_etkf_sqrt_type {cholesky,eigen}` (or, programmatically, `PetscDAETKFSetSqrtType()`) toggles between Cholesky and eigenvalue-based square-root updates. The default is `cholesky`, which is computationally more efficient (O(n³/3) vs O(n³)) and preferred when the reduced-space matrix is known to be positive definite. The `eigen` method is more robust for semi-definite matrices as it handles small negative eigenvalues arising from numerical round-off.

Custom PetscDA types should implement the `PetscDASetType()` registration hook, populate virtual methods for analysis and forecast orchestration, and take advantage of the anomaly computations provided by the base class.

(sec_da_options)=

## Command-line options

The `PetscDA` object obeys standard PETSc options parsing. Commonly used switches include:

- `-petscda_type <name>` – select a registered PetscDA implementation.
- `-petscda_view` / `-petscda_view ::ascii_info_detail` – inspect ensemble metadata and internal sizes.

Because `PetscDA` participates in the PETSc object registry, any prefix applied with `PetscDASetOptionsPrefix()` scopes these options.

(sec_da_viewing)=

## Viewing and monitoring

`PetscDAView()` and `PetscDAViewFromOptions()` expose ensemble sizing, observation dimensions, and implementation-specific diagnostics. Views can be directed to ASCII, HDF5, or custom `PetscViewer` targets, enabling lightweight instrumentation of assimilation experiments. For advanced profiling, the PetscDA package registers with PETSc's logging infrastructure via `PetscDAInitializePackage()`/`PetscDAFinalizePackage()`, so standard `-log_view` outputs will include assimilation breakdown.

(sec_da_compat)=

## Compatibility

The PetscDA package provides backward-compatibility headers and shims where feasible, but as a new component, users are encouraged to adopt the `PetscDA` naming convention.

These thin wrappers keep existing applications functional while encouraging new developments to migrate to the canonical `PetscDA*()` routines.

(sec_da_further_reading)=

## Further reading

- {any}`ch_ts` discusses PETSc time integrators that can supply the forecast operator passed to `PetscDAApplyModel()`.
- {any}`ch_vectors` documents vector assembly and parallel data management for the state and observation spaces.
- {any}`ch_snes` outlines nonlinear solvers that often participate in observation or model operators.
- {any}`ch_dmbase` provides background on distributed mesh infrastructure that can coexist with PetscDA-managed ensembles.
