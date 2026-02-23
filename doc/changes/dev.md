# Changes: Development

<!---
% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence
--->

## General


## Configure/Build


## Sys


## Event Logging


## PetscViewer


## PetscDraw


## AO


## IS


## VecScatter / PetscSF


## PF


## Vec


## PetscSection


## PetscPartitioner


## Mat

- Add `MATPRODUCT_PtAP` support for `MATDIAGONAL`

## MatCoarsen


## PC

- Add `TSPseudoComputeFunction()` to get nonlinear residual while avoiding recalculation if possible
- Remove unused `TSPseudoVerifyTimeStepDefault()`
- Remove `TSPseudoComputeTimeStep()` and `TSPseudoVerifyTimeStep()`
- Change the `destroy()` function argument of `TSTrajectorySetTransform()` to type `PetscCtxDestroyFn *`. This means the destroy function must dereference the argument before operating on it
- Correct option `-ts_max_reject` to `-ts_max_step_rejections`
- Correct option `-ts_dt` to `-ts_time_step`
- Change `TSAdaptCheckStage()` to call function set by `TSAdaptSetCheckStage()` before other checks
- Fix `-ts_ssp_nstages` to `-ts_ssp_num_stages`
- Add TSType ``TSRKS`` which implements Runge-Kutta Super-time-steppers.
- Within the ``TSRKS`` framework there are 4 subtypes: ``RKS_RKC1``, ``RKS_RKC2``, ``RKS_RKL1``, ``RKS_RKL2``. These represent first- and second-order Runge-Kutta-Chebyshev and -Legendre methods.

## KSP


## SNES


## SNESLineSearch


## TS


## TAO

- Add `TaoGetDM()` and `TaoSetDM()`

## TaoTerm


## PetscRegressor


## PetscDA


## DM


## DMSwarm

- Add `DMSwarmProjectFields()` and `DMSwarmProjectGradientFields()`
- Add `DMSwarmSort` class
- Add `DMSwarmSortDestroy()` and `DMSwarmSortView()`
- Allow `DMSwarmCellDMSetSort()` to take in `NULL` and clear the sort
- Add `DMSwarmPreallocateMassMatrix()` and `DMSwarmFillMassMatrix()`

## DMPlex

- Add `DMPlexSetClosurePermutationLexicographic()`
- Add `DMPlexDrawCell()`
- Add `DMPlexLabelCompleteStar()`
- Add `DMPlexVecGetClosureAtDepth()`
- Add an extra communicator argument to `DMPlexFilter()` to allow extracting local meshes
- Add `DMPlexCopyFlags()`
- Add `DMPlexRebalanceSharedLabelPoints()`
- Add `DMPlexCheckLabel()` and `DMPlexReconcileLabel()`
- Change CGNS viewer to use multi-component read/write interface for better performance

## FE/FV


## DMNetwork


## DMStag


## DT


## Fortran
