/*
     The KSP orthogonalization routines, used in GMRES and other solvers.
*/
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

/*@
  KSPSetOrthogonalization - Sets the orthogonalization routine used by `KSPGMRES` and other solvers.

  Logically Collective

  Input Parameters:
+ ksp    - the Krylov space solver context
- orthog - orthogonalization function; see `KSPOrthogonalizationFn` for the calling sequence

  Options Database Keys:
+ -ksp_classicalgramschmidt - Activates `KSPClassicalGramSchmidtOrthogonalization()` (default)
- -ksp_modifiedgramschmidt  - Activates `KSPModifiedGramSchmidtOrthogonalization()`

  Level: intermediate

  Notes:
  Two orthogonalization routines are predefined, `KSPModifiedGramSchmidtOrthogonalization()` and the default
  `KSPClassicalGramSchmidtOrthogonalization()`.

  Use `KSPSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](ch_ksp), `KSPOrthogonalizationFn`, `KSPGetOrthogonalization()`, `KSPSetCGSRefinementType()`, `KSPClassicalGramSchmidtOrthogonalization()`, `KSPModifiedGramSchmidtOrthogonalization()`
@*/
PetscErrorCode KSPSetOrthogonalization(KSP ksp, KSPOrthogonalizationFn *orthog)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ksp->orthog = orthog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetOrthogonalization - Gets the orthogonalization routine used by `KSPGMRES` and other solvers.

  Not Collective

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameter:
. orthog - orthogonalization function; see `KSPOrthogonalizationFn` for the calling sequence

  Level: intermediate

  Notes:
  Two orthogonalization routines are predefined, `KSPModifiedGramSchmidtOrthogonalization()` and the default
  `KSPClassicalGramSchmidtOrthogonalization()`.

  Use `KSPSetCGSRefinementType()` to determine if iterative refinement is used to increase stability.

.seealso: [](ch_ksp), `KSPOrthogonalizationFn`, `KSPSetCGSRefinementType()`, `KSPClassicalGramSchmidtOrthogonalization()`, `KSPModifiedGramSchmidtOrthogonalization()`
@*/
PetscErrorCode KSPGetOrthogonalization(KSP ksp, KSPOrthogonalizationFn **orthog)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(orthog, 2);
  *orthog = ksp->orthog;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetCGSRefinementType - Sets the type of iterative refinement to use in the classical Gram-Schmidt
  orthogonalization used by `KSPGMRES` and other solvers.

  Logically Collective

  Input Parameters:
+ ksp  - the Krylov space solver context
- type - the type of refinement

  Options Database Key:
. -ksp_cgs_refinement_type (refine_never|refine_ifneeded|refine_always) - refinement type

  Level: intermediate

  Notes:
  The default is `KSP_CGS_REFINE_NEVER`.

  For a very small set of problems, not using refinement, that is `KSP_CGS_REFINE_NEVER`, may be unstable, thus causing `KSPSolve()`
  to not converge.

.seealso: [](ch_ksp), `KSPSetOrthogonalization()`, `KSPCGSRefinementType`, `KSPClassicalGramSchmidtOrthogonalization()`, `KSPGetCGSRefinementType()`,
          `KSPGetOrthogonalization()`
@*/
PetscErrorCode KSPSetCGSRefinementType(KSP ksp, KSPCGSRefinementType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, type, 2);
  ksp->cgstype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetCGSRefinementType - Gets the type of iterative refinement to use in the classical Gram-Schmidt
  orthogonalization used by `KSPGMRES` and other solvers.

  Not Collective

  Input Parameter:
. ksp - the Krylov space solver context

  Output Parameter:
. type - the type of refinement

  Level: intermediate

.seealso: [](ch_ksp), `KSPSetOrthogonalization()`, `KSPCGSRefinementType`, `KSPClassicalGramSchmidtOrthogonalization()`, `KSPSetCGSRefinementType()`,
          `KSPGetOrthogonalization()`
@*/
PetscErrorCode KSPGetCGSRefinementType(KSP ksp, KSPCGSRefinementType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ksp->cgstype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPModifiedGramSchmidtOrthogonalization -  This is the basic orthogonalization routine
  using modified Gram-Schmidt.

  Collective, No Fortran Support

  Input Parameters:
+ ksp - the Krylov space solver context
. V   - array of previously orthogonalized vectors
. n   - number of vectors
. x   - vector to be orthogonalized (may be `NULL`)
- h   - computed orthogonalization coefficients

  Options Database Key:
. -ksp_modifiedgramschmidt - Activates `KSPModifiedGramSchmidtOrthogonalization()`

  Level: intermediate

  Notes:
  If no `x` is given, then the vector to be orthogonalized is assumed to be located at `V[n]`.

  In general this is much slower than `KSPClassicalGramSchmidtOrthogonalization()` but has better stability properties.

.seealso: [](ch_ksp), `KSPSetOrthogonalization()`, `KSPClassicalGramSchmidtOrthogonalization()`, `KSPGetOrthogonalization()`
@*/
PetscErrorCode KSPModifiedGramSchmidtOrthogonalization(KSP ksp, Vec V[], PetscInt n, Vec x, PetscScalar h[])
{
  PetscInt     j;
  PetscScalar *hh = h;
  Vec          z  = x;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(KSP_Orthogonalization, ksp, 0, 0, 0));
  if (!z) z = V[n];
  for (j = 0; j < n; j++) {
    /* (z, v(j)) */
    PetscCall(VecDot(z, V[j], hh));
    KSPCheckDot(ksp, *hh);
    if (ksp->reason) break;
    /* z <- z - hh[j] v(j) */
    PetscCall(VecAXPY(z, -(*hh++), V[j]));
  }
  PetscCall(PetscLogEventEnd(KSP_Orthogonalization, ksp, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPClassicalGramSchmidtOrthogonalization -  This is the basic orthogonalization routine
  using classical Gram-Schmidt with possible iterative refinement to improve the stability.

  Collective, No Fortran Support

  Input Parameters:
+ ksp - the Krylov space solver context
. V   - array of previously orthogonalized vectors
. n   - number of vectors
. x   - vector to be orthogonalized (may be `NULL`)
- h   - computed orthogonalization coefficients

  Options Database Keys:
+ -ksp_classicalgramschmidt (true|false)                                - Activates `KSPClassicalGramSchmidtOrthogonalization()`
- -ksp_cgs_refinement_type (refine_never|refine_ifneeded|refine_always) - determine if iterative refinement is
                                                                          used to increase the stability of the classical Gram-Schmidt orthogonalization.

  Level: intermediate

  Note:
  Use `KSPSetCGSRefinementType()` to determine if iterative refinement is to be used.
  This is much faster than `KSPModifiedGramSchmidtOrthogonalization()` but has the small possibility of stability issues
  that can usually be handled by using a single step of iterative refinement with `KSPSetCGSRefinementType()`.

.seealso: [](ch_ksp), `KSPCGSRefinementType`, `KSPSetOrthogonalization()`, `KSPSetCGSRefinementType()`,
           `KSPGetCGSRefinementType()`, `KSPGetOrthogonalization()`, `KSPModifiedGramSchmidtOrthogonalization()`
@*/
PetscErrorCode KSPClassicalGramSchmidtOrthogonalization(KSP ksp, Vec V[], PetscInt n, Vec x, PetscScalar h[])
{
  PetscInt     j;
  PetscScalar *hh = h, *lhh;
  Vec          z  = x;
  PetscReal    hnrm, wnrm;
  PetscBool    refine = (PetscBool)(ksp->cgstype == KSP_CGS_REFINE_ALWAYS);

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(KSP_Orthogonalization, ksp, 0, 0, 0));
  if (!z) z = V[n];
  if (ksp->lorthogwork < n) {
    PetscCall(PetscFree(ksp->orthogwork));
    ksp->lorthogwork = PetscMax(30, PetscMax(2 * ksp->lorthogwork, n));
    PetscCall(PetscMalloc1(ksp->lorthogwork, &ksp->orthogwork));
  }
  lhh = ksp->orthogwork;

  /* Clear hh since we will accumulate values into them */
  for (j = 0; j < n; j++) hh[j] = 0.0;

  /*
     This is really a matrix-vector product, with the matrix stored
     as pointer to rows
  */
  PetscCall(VecMDot(z, n, V, lhh)); /* <v,z> */
  for (j = 0; j < n; j++) {
    KSPCheckDot(ksp, lhh[j]);
    if (ksp->reason) goto done;
    lhh[j] = -lhh[j];
  }

  /*
         This is really a matrix-vector product:
         [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from z.
  */
  PetscCall(VecMAXPY(z, n, lhh, V));
  /* note lhh[j] is -<v,z> , hence the subtraction */
  for (j = 0; j < n; j++) {
    hh[j] -= lhh[j]; /* hh += <v,z> */
  }

  /*
     the second step classical Gram-Schmidt is only necessary
     when a simple test criteria is not passed
  */
  if (ksp->cgstype == KSP_CGS_REFINE_IFNEEDED) {
    hnrm = 0.0;
    for (j = 0; j < n; j++) hnrm += PetscRealPart(lhh[j] * PetscConj(lhh[j]));

    hnrm = PetscSqrtReal(hnrm);
    PetscCall(VecNorm(z, NORM_2, &wnrm));
    KSPCheckNorm(ksp, wnrm);
    if (ksp->reason) goto done;
    if (wnrm < hnrm) {
      refine = PETSC_TRUE;
      PetscCall(PetscInfo(ksp, "Performing iterative refinement wnorm %g hnorm %g\n", (double)wnrm, (double)hnrm));
    }
  }

  if (refine) {
    PetscCall(VecMDot(z, n, V, lhh)); /* <v,z> */
    for (j = 0; j < n; j++) {
      KSPCheckDot(ksp, lhh[j]);
      if (ksp->reason) goto done;
      lhh[j] = -lhh[j];
    }
    PetscCall(VecMAXPY(z, n, lhh, V));
    /* note lhh[j] is -<v,z> , hence the subtraction */
    for (j = 0; j < n; j++) {
      hh[j] -= lhh[j]; /* hh += <v,z> */
    }
  }
done:
  PetscCall(PetscLogEventEnd(KSP_Orthogonalization, ksp, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
