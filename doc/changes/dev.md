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

- Add support for writing CGNS descriptors on the base node: `PetscViewerCGNSGetDescriptors()`, `PetscViewerCGNSRestoreDescriptors()`, `PetscViewerCGNSSetDescriptor()`

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

- Add native batched GMRES Krylov solver to `PCBJKOKKOS` (`-pc_bjkokkos_ksp_type gmres`), with both Jacobi and AMG preconditioning variants
- Add native batched AMG (algebraic multigrid) preconditioner to `PCBJKOKKOS` (`-pc_bjkokkos_batch_pc amg`) using classical Ruge-Stüben coarsening with l1-Jacobi smoothing; supports BICG, TFQMR, and GMRES Krylov solvers
- Change `PCBJKOKKOS` `-pc_bjkokkos_ksp_batch_target -1` to mean "print all batches" instead of raising an error; previously only non-negative values were accepted
- Fix `PCBJKOKKOS` batch solvers reporting `KSP_CONVERGED_ITS` (converged) when max iterations exhausted without convergence; now correctly reports `KSP_DIVERGED_ITS`
- Fix `PCBJKOKKOS` batch solvers using `KSP_CONVERGED_RTOL_NORMAL_EQUATIONS` for plain relative-residual convergence; now correctly uses `KSP_CONVERGED_RTOL`

## KSP


## SNES


## SNESLineSearch


## TS

- Add `DMTSSetIFunctionPre()`
- Add `TSDiscGradSetImplicitFormulation()`
- Expose `TSDiscGradGetX0AndXdot()` and `TSDiscGradRestoreX0AndXdot()`

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
