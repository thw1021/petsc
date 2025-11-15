(ch_dataassimilator)=

# PetscDataAssimilator: Ensemble Data Assimilation

PETSc's `PetscDataAssimilator` object coordinates ensemble-based data assimilation (DA) workflows. It centralizes ensemble storage, observational metadata, and user-defined forecast/analysis operators so that algorithms can run independently of the MPI layout or the vector/matrix backends. The PetscDataAssimilator layer currently focuses on ensemble transform Kalman filter (ETKF)-style updates but is extensible to other assimilation techniques that follow the same interfaces.

(sec_da_lifecycle)=

## Lifecycle overview

A typical assimilation cycle alternates between forecast propagation and statistical analysis:

1. Initialize a `PetscDataAssimilator` context and configure ensemble sizes and data structures.
2. Populate the ensemble state vectors and optional observation-error descriptions.
3. Advance each ensemble member with a model operator supplied by the application.
4. Combine forecasts with observations through `PetscDataAssimilatorAnalysis()` to produce the posterior ensemble.
5. Repeat until the desired simulation horizon is complete, optionally extracting diagnostics after each phase.

Throughout this loop the `PetscDataAssimilator` object abstracts the global vectors, scatters, and reductions needed to compute ensemble means, anomalies, and square-root transforms.

(sec_da_creating)=

## Creating a PetscDataAssimilator context

Create, configure, and destroy a `PetscDataAssimilator` object with the standard PETSc object lifecycle:

```c
PetscDataAssimilator da;
PetscCall(PetscDataAssimilatorCreate(PETSC_COMM_WORLD, &da));
PetscCall(PetscDataAssimilatorSetType(da, PETSCDAETKF));
PetscCall(PetscDataAssimilatorSetSizes(da, ensemble_size, local_state_size, observation_size));
PetscCall(PetscDataAssimilatorSetFromOptions(da));
PetscCall(PetscDataAssimilatorSetUp(da));

/* ... data assimilation loop ... */

PetscCall(PetscDataAssimilatorDestroy(&da));
```

`PetscDataAssimilatorSetSizes()` records (1) the number of ensemble members, (2) the local state dimension per MPI rank, and (3) the number of observations that will be processed simultaneously. After `PetscDataAssimilatorSetUp()` the object owns ensemble storage and is ready to hand out views on members.

(sec_da_ensemble)=

## Managing ensemble structure

`PetscDataAssimilator` stores ensemble members as PETSc `Vec` objects and exposes convenience helpers to access them safely:

- `PetscDataAssimilatorGetEnsembleMember()` / `PetscDataAssimilatorRestoreEnsembleMember()` map ensemble indices to `Vec` handles that participate in PETSc's reference counting.
- `PetscDataAssimilatorSetEnsembleMember()` lets applications inject externally created vectors into specific slots, which is useful when importing state snapshots from disk or another solver component.
- `PetscDataAssimilatorComputeMean()` forms the sample mean across all members.
- `PetscDataAssimilatorComputeAnomalies()` returns a tall-and-skinny `Mat` whose columns are the mean-subtracted ensemble anomalies. Many square-root filters use this matrix to construct low-rank covariance factorizations.

When ensemble perturbations are needed, `VecSetRandomGaussian()` can be used to draw samples consistent with a desired mean and variance by combining PETSc's random number generators with user-provided scaling parameters.

(sec_da_observation)=

## Observation handling and analysis

Observation-error variances (or more general descriptions) are supplied through `PetscDataAssimilatorSetObsErrorVariance()`. The associated vector is assumed to follow the global observation ordering; `PetscDataAssimilatorGetObsErrorVariance()` returns the stored object for later inspection or reuse.

`PetscDataAssimilatorAnalysis()` performs the assimilation step by calling a user-provided observation operator:

```c
static PetscErrorCode ObservationOperator(Vec state, Vec prediction, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MyForwardModel(state, prediction, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscCall(PetscDataAssimilatorAnalysis(da, observation_vec, ObservationOperator, user_ctx));
```

The callback must fill `prediction` with the modelled observations corresponding to `state`, respecting the layout dictated by `PetscDataAssimilatorSetSizes()`. `PetscDataAssimilatorAnalysis()` handles all ensemble reductions, gain computations, and posterior updates.

### Stochastic perturbations

Filters requiring stochastic observation perturbations can combine `VecSetRandomGaussian()` with the observation-error variance vector:

```c
PetscCall(VecSetRandomGaussian(noise, rnd, 0.0, 1.0));
PetscCall(VecPointwiseMult(noise, noise, obs_variance));
PetscCall(VecAXPY(observation_vec, 1.0, noise));
```

This pattern produces perturbed observations consistent with the diagonal variance model.

(sec_da_forecast)=

## Forecast propagation

`PetscDataAssimilatorApplyModel()` wraps the forecast step. The user supplies a function that advances a single ensemble member:

```c
static PetscErrorCode ModelOperator(Vec ensemble_in, Vec ensemble_out, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(ensemble_in, ensemble_out));
  PetscCall(TSSolve((TS)ctx, ensemble_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscCall(PetscDataAssimilatorApplyModel(da, ModelOperator, ts_ctx));
```

The operator can call into PETSc time integrators ({any}`ch_ts`), nonlinear solvers ({any}`ch_snes`), or bespoke device kernels. The PetscDataAssimilator layer orchestrates calls across the entire ensemble, issuing them in rank-local loops while ensuring that ownership and recycling semantics remain correct.

Applications that maintain long-lived solver contexts commonly preload Jacobians, preconditioners, or MPI layouts outside of the callback and pass them through `ctx` for efficiency.

(sec_da_implementations)=

## Choosing implementations

The default implementation is the ensemble transform Kalman filter indicated by the type string `PETSCDAETKF` (which resolves to `"etkf"`). Alternative PetscDataAssimilator types can be registered with `PetscDataAssimilatorRegister()` and selected at runtime:

- `-petscdataassimilator_type etkf` chooses the built-in square-root ETKF.
- `-dataassimilator_etkf_sqrt_type {cholesky,eigen}` (or, programmatically, `PetscDataAssimilatorETKFSetSqrtType()`) toggles between Cholesky and eigenvalue-based square-root updates. The default is `cholesky`, which is computationally more efficient (O(n³/3) vs O(n³)) and preferred when the reduced-space matrix is known to be positive definite. The `eigen` method is more robust for semi-definite matrices as it handles small negative eigenvalues arising from numerical round-off.

Custom PetscDataAssimilator types should implement the `PetscDataAssimilatorSetType()` registration hook, populate virtual methods for analysis and forecast orchestration, and take advantage of the anomaly computations provided by the base class.

(sec_da_options)=

## Options database

The `PetscDataAssimilator` object obeys standard PETSc options parsing. Commonly used switches include:

- `-petscdataassimilator_type <name>` – select a registered PetscDataAssimilator implementation.
- `-petscdataassimilator_view` / `-petscdataassimilator_view ::ascii_info_detail` – inspect ensemble metadata and internal sizes.
- `-etkf_sqrt_type <cholesky,eigen>` – set the square-root solver used by the ETKF backend.
- `-petscda_monitor` – enable runtime logging, when supported by the selected implementation.

Because `PetscDataAssimilator` participates in the PETSc object registry, any prefix applied with `PetscDataAssimilatorSetOptionsPrefix()` scopes these options.

(sec_da_diagnostics)=

## Diagnostics and viewing

`PetscDataAssimilatorView()` and `PetscDataAssimilatorViewFromOptions()` expose ensemble sizing, observation dimensions, and implementation-specific diagnostics. Views can be directed to ASCII, HDF5, or custom `PetscViewer` targets, enabling lightweight instrumentation of assimilation experiments. For advanced profiling, the PetscDataAssimilator package registers with PETSc's logging infrastructure via `PetscDataAssimilatorInitializePackage()`/`PetscDataAssimilatorFinalizePackage()`, so standard `-log_view` runs capture time spent in forecast and analysis kernels.


These thin wrappers keep existing applications functional while encouraging new developments to migrate to the canonical `PetscDataAssimilator*()` routines.

(sec_da_related)=

## Related reading

- {any}`ch_ts` discusses PETSc time integrators that can supply the forecast operator passed to `PetscDataAssimilatorApplyModel()`.
- {any}`ch_vectors` documents vector assembly and parallel data management for the state and observation spaces.
- {any}`ch_snes` outlines nonlinear solvers that often participate in observation or model operators.
- {any}`ch_dmbase` provides background on distributed mesh infrastructure that can coexist with PetscDataAssimilator-managed ensembles.