#pragma once

/* SUBMANSEC = KSP */

/*S
   KSP - Abstract PETSc object that manages the linear solves in PETSc (even those such as direct factorization-based solvers that
         do not use Krylov accelerators).

   Level: beginner

   Notes:
   When a direct solver is used, but no Krylov solver is used, the `KSP` object is still used but with a
   `KSPType` of `KSPPREONLY` (or equivalently `KSPNONE`), meaning that only application of the preconditioner is used as the linear solver.

   Use `KSPSetType()` or the options database key `-ksp_type` to set the specific Krylov solver algorithm to use with a given `KSP` object

   The `PC` object is used to control preconditioners in PETSc.

  `KSP` can also be used to solve some least squares problems (over or under-determined linear systems), using, for example, `KSPLSQR`, see `PETSCREGRESSORLINEAR`
  for additional methods that can be used to solve least squares problems and other linear regressions).

.seealso: [](doc_linsolve), [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `SNES`, `TS`, `PC`, `KSP`, `KSPDestroy()`, `KSPCG`, `KSPGMRES`
S*/
typedef struct _p_KSP *KSP;
