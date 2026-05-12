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

- Add `PetscGetConfiguration()`

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

- Add `MATPRODUCT_PtAP` support for `MATDIAGONAL` and `MATCONSTANTDIAGONAL`
- Add `MatSeqAIJGetKokkosView()`, `MatSeqAIJRestoreKokkosView()`, `MatSeqAIJGetKokkosViewWrite()` and `MatSeqAIJRestoreKokkosViewWrite()` to the public API

## MatCoarsen


## PC


## KSP


## SNES


## SNESLineSearch


## TS

- Add `DMTSSetIFunctionPre()`
- Add `TSDiscGradSetImplicitFormulation()`
- Expose `TSDiscGradGetX0AndXdot()` and `TSDiscGradRestoreX0AndXdot()`
- Add `TSIsImplicit()` that indicates if the `TSType` is implicit and uses `SNES` or `KSP`

## TAO

- Add `TaoGetDM()` and `TaoSetDM()`
- Allow `TaoAddTerm()` on the bounded solvers `TAOBLMVM`, `TAOBNK`, `TAOBNLS`, `TAOBNTL`, `TAOBNTR`, and `TAOTRON`; every summand must define an assembled Hessian (`TAOBQNK` remains unsupported)
- Refactor `TAOBRGN` onto `TaoTerm` composition: the subsolver now holds two summands, a `TAOTERMGAUSSNEWTON` for $\tfrac{1}{2}\|R(x)\|_2^2$ and a regularizer term scaled by $\lambda$. Add `TaoBRGNSetRegularizerTerm()` and `TaoBRGNGetRegularizerTerm()` so users can install an arbitrary `TaoTerm` regularizer; the legacy `TaoBRGNSetRegularizerObjectiveAndGradientRoutine()` and `TaoBRGNSetRegularizerHessianRoutine()` setters become callback shims around a `TAOTERMSHELL`. Remove `-tao_brgn_mat_explicit`; control the Gauss-Newton Hessian assembly via `-<prefix>brgn_gauss_newton_tao_term_hessian_mat_type` instead. The new path requires the residual Jacobian to support `MatProduct` AtB, so `TAOBRGN` no longer accepts unassembleable Jacobians (e.g. `MATCOMPOSITE`); workflows that previously relied on the matrix-free Gauss-Newton Hessian must either materialize their Jacobian or wait for shell-mode support to land

## TaoTerm

- Add `TAOTERMGAUSSNEWTON`: a `TaoTerm` that wraps the residual machinery of a `Tao` (set with `TaoSetResidualRoutine()` and `TaoSetJacobianResidualRoutine()`) as the smooth least-squares term $\tfrac{1}{2}\|R(x)\|_2^2$ with Gauss-Newton Hessian $J^T J$. The residual Jacobian is cached by `(x_id, x_state)` so successive evaluations at the same iterate do not redundantly invoke the user's Jacobian. New API: `TaoTermCreateGaussNewton()`, `TaoTermGaussNewtonSetTao()`, `TaoTermGaussNewtonGetTao()`, `TaoTermGaussNewtonGetJacobian()`


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
