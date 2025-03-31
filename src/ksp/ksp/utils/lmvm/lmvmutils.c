#include <petscdevice.h>
#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/
#include <petsc/private/deviceimpl.h>
#include <petscblaslapack.h>

/*@
  MatLMVMUpdate - Adds (X-Xprev) and (F-Fprev) updates to an `MATLMVM` matrix.
  The first time the function is called for an `MATLMVM` matrix, no update is
  applied, but the given X and F vectors are stored for use as Xprev and
  Fprev in the next update.

  If the user has provided another `MATLMVM` matrix in place of J0, the J0
  matrix is also updated recursively.

  Input Parameters:
+ B - A `MATLMVM` matrix
. X - Solution vector
- F - Function vector

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMReset()`, `MatLMVMAllocate()`
@*/
PetscErrorCode MatLMVMUpdate(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(MATLMVM_Update, NULL, NULL, NULL, NULL));
  lmvm = (Mat_LMVM *)B->data;
  if (!lmvm->allocated) {
    PetscCall(MatLMVMAllocate(B, X, F));
  } else {
    VecCheckMatCompatible(B, X, 2, F, 3);
  }
  PetscCall(MatLMVMUpdate(lmvm->J0, X, F));
  PetscCall((*lmvm->ops->update)(B, X, F));
  PetscCall(PetscLogEventEnd(MATLMVM_Update, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCreateJ0(Mat B, Mat *J0)
{
  PetscLayout rmap, cmap;
  VecType     vec_type;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), J0));
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscCall(MatSetLayouts(*J0, rmap, cmap));
  PetscCall(MatGetVecType(B, &vec_type));
  PetscCall(MatSetVecType(*J0, vec_type));
  PetscCall(MatGetOptionsPrefix(B, &prefix));
  PetscCall(MatSetOptionsPrefix(*J0, prefix));
  PetscCall(MatAppendOptionsPrefix(*J0, "mat_lmvm_J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCreateJ0KSP(Mat B, KSP *ksp)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)B), ksp));
  PetscCall(KSPSetOperators(*ksp, lmvm->J0, lmvm->J0));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)B, (PetscObject)*ksp, 1));
  PetscCall(MatGetOptionsPrefix(B, &prefix));
  PetscCall(KSPSetOptionsPrefix(*ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(*ksp, "mat_lmvm_J0_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMCreateJ0KSP_ExactInverse(Mat B, KSP *ksp)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PC        pc;

  PetscFunctionBegin;
  PetscCall(MatLMVMCreateJ0KSP(B, ksp));
  PetscCall(KSPSetType(*ksp, KSPPREONLY));
  PetscCall(KSPGetPC(*ksp, &pc));
  PetscCall(PCSetType(pc, PCMAT));
  PetscCall(PCMatSetApplyOperation(pc, MATOP_SOLVE));
  lmvm->disable_ksp_viewers = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMClearJ0 - Removes all definitions of J0 and reverts to
  an identity matrix (scale = 1.0).

  Input Parameter:
. B - A `MATLMVM` matrix

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMClearJ0(Mat B)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCall(MatDestroy(&lmvm->J0));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  PetscCall(MatLMVMCreateJ0(B, &lmvm->J0));
  PetscCall(MatSetType(lmvm->J0, MATCONSTANTDIAGONAL));
  PetscCall(MatZeroEntries(lmvm->J0));
  PetscCall(MatShift(lmvm->J0, 1.0));
  PetscCall(MatLMVMCreateJ0KSP_ExactInverse(B, &lmvm->J0ksp));
  lmvm->created_J0    = PETSC_TRUE;
  lmvm->created_J0ksp = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0Scale - Allows the user to define a scalar value
  mu such that J0 = mu*I.

  Input Parameters:
+ B     - A `MATLMVM` matrix
- scale - Scalar value mu that defines the initial Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetDiagScale()`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMSetJ0Scale(Mat B, PetscReal scale)
{
  Mat_LMVM *lmvm;
  PetscBool same;
  PetscBool isconstant;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCheck(lmvm->square, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "Scaling is available only for square LMVM matrices");
  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATCONSTANTDIAGONAL, &isconstant));
  if (!isconstant) PetscCall(MatLMVMClearJ0(B));
  PetscCall(MatZeroEntries(lmvm->J0));
  PetscCall(MatShift(lmvm->J0, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0Diag - Allows the user to define a vector
  V such that J0 = diag(V).

  Input Parameters:
+ B - An LMVM-type matrix
- V - Vector that defines the diagonal of the initial Jacobian: values are copied, V is not referenced

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetScale()`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMSetJ0Diag(Mat B, Vec V)
{
  Mat_LMVM *lmvm;
  PetscBool same;
  PetscBool isvdiag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(V, VEC_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCheck(lmvm->allocated, PetscObjectComm((PetscObject)B), PETSC_ERR_ORDER, "Matrix must be allocated before setting diagonal scaling");
  PetscCheck(lmvm->square, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "Diagonal scaling is available only for square LMVM matrices");
  VecCheckSameSize(V, 2, lmvm->Fprev, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATDIAGONAL, &isvdiag));
  if (!isvdiag) {
    PetscCall(MatLMVMClearJ0(B));
    PetscCall(MatSetType(lmvm->J0, MATDIAGONAL));
  }
  PetscCall(MatDiagonalSet(lmvm->J0, V, INSERT_VALUES));
  lmvm->created_J0 = PETSC_FALSE; // Because V was not created
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetJ0InvDiag(Mat B, Vec *V)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool isvdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATDIAGONAL, &isvdiag));
  if (!isvdiag) {
    PetscCall(MatLMVMClearJ0(B));
    PetscCall(MatSetType(lmvm->J0, MATDIAGONAL));
    PetscCall(MatZeroEntries(lmvm->J0));
    PetscCall(MatShift(lmvm->J0, 1.0));
  }
  PetscCall(MatDiagonalGetInverseDiagonal(lmvm->J0, V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMRestoreJ0InvDiag(Mat B, Vec *V)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatDiagonalRestoreInverseDiagonal(lmvm->J0, V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMJ0HasSolve(Mat B, PetscBool *has_solve)
{
  Mat_LMVM *lmvm;

  PetscFunctionBegin;
  *has_solve = PETSC_FALSE;
  lmvm       = (Mat_LMVM *)B->data;
  if (lmvm->square) { PetscCall(PetscObjectTypeCompareAny((PetscObject)lmvm->J0, has_solve, MATCONSTANTDIAGONAL, MATDIAGONAL, MATLMVMDIAGBROYDEN, "")); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0 - Allows the user to define the initial
  Jacobian matrix from which the LMVM-type approximation is
  built up.

  Input Parameters:
+ B  - An LMVM-type matrix
- J0 - The initial Jacobian matrix, will be referenced by B.

  Level: advanced

  Note:
  A KSP is created for inverting J0 with prefix `-mat_lmvm_J0_` and J0
  is set to both operators in `KSPSetOperators()`.  If you want
  to use a separate preconditioning matrix, use `MatLMVMSetKSP()` directly.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0PC()`, `MatLMVMSetJ0KSP()`
@*/
PetscErrorCode MatLMVMSetJ0(Mat B, Mat J0)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0, MAT_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  if (J0 != lmvm->J0) {
    PetscCall(PetscObjectReference((PetscObject)J0));
    PetscCall(MatDestroy(&lmvm->J0));
    lmvm->J0         = J0;
    lmvm->created_J0 = PETSC_FALSE;
    if (lmvm->square) {
      PetscBool J0_has_solve;

      PetscCall(MatLMVMJ0HasSolve(B, &J0_has_solve));
      if (J0_has_solve) {
        PetscCall(KSPDestroy(&lmvm->J0ksp));
        PetscCall(MatLMVMCreateJ0KSP_ExactInverse(B, &lmvm->J0ksp));
        lmvm->created_J0ksp = PETSC_TRUE;
      } else {
        PetscCall(KSPSetOperators(lmvm->J0ksp, J0, J0));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0PC - Allows the user to define a `PC` object that
  acts as the initial inverse-Jacobian matrix.

  Input Parameters:
+ B    - A `MATLMVM` matrix
- J0pc - `PC` object where `PCApply()` defines an inverse application for J0

  Level: advanced

  Note:
  `J0pc` should already contain all the operators necessary for its application.
  The `MATLMVM` matrix only calls `PCApply()` without changing any other
  options.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetJ0PC()`
@*/
PetscErrorCode MatLMVMSetJ0PC(Mat B, PC J0pc)
{
  Mat_LMVM *lmvm;
  PetscBool same;
  Mat       J0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0pc, PC_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCheck(lmvm->square, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  PetscCall(PCGetOperators(J0pc, &J0, NULL));
  if (J0 != lmvm->J0) {
    PetscCall(PetscObjectReference((PetscObject)J0));
    PetscCall(MatDestroy(&lmvm->J0));
    lmvm->J0         = J0;
    lmvm->created_J0 = PETSC_FALSE;
  }
  PetscCall(PetscObjectReference((PetscObject)J0pc));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  PetscCall(MatLMVMCreateJ0KSP(B, &lmvm->J0ksp));
  lmvm->created_J0ksp = PETSC_TRUE;
  PetscCall(KSPSetType(lmvm->J0ksp, KSPPREONLY));
  PetscCall(KSPSetPC(lmvm->J0ksp, J0pc));
  PetscCall(PCDestroy(&J0pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetJ0KSP - Allows the user to provide a pre-configured
  KSP solver for the initial inverse-Jacobian approximation.

  Input Parameters:
+ B     - A `MATLMVM` matrix
- J0ksp - `KSP` solver for the initial inverse-Jacobian application

  Level: advanced

  Note:
  The `KSP` solver should already contain all the operators
  necessary to perform the inversion. The `MATLMVM` matrix only
  calls `KSPSolve()` without changing any other options.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetJ0KSP()`
@*/
PetscErrorCode MatLMVMSetJ0KSP(Mat B, KSP J0ksp)
{
  Mat_LMVM *lmvm;
  PetscBool same;
  Mat       J0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(J0ksp, KSP_CLASSID, 2);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCheck(lmvm->square, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "Inverse J0 can be defined only for square LMVM matrices");
  if (J0ksp != lmvm->J0ksp) {
    lmvm->created_J0ksp       = PETSC_FALSE;
    lmvm->disable_ksp_viewers = PETSC_FALSE; // If the user supplies a more complicated KSP, don't turn off viewers
  }
  PetscCall(PetscObjectReference((PetscObject)J0ksp));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  lmvm->J0ksp = J0ksp;
  PetscCall(KSPGetOperators(J0ksp, &J0, NULL));
  if (J0 != lmvm->J0) {
    PetscCall(PetscObjectReference((PetscObject)J0));
    PetscCall(MatDestroy(&lmvm->J0));
    lmvm->J0         = J0;
    lmvm->created_J0 = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetJ0 - Returns a pointer to the internal `J0` matrix.

  Input Parameter:
. B - A `MATLMVM` matrix

  Output Parameter:
. J0 - `Mat` object for defining the initial Jacobian

  Level: advanced

  Note:

  If `J0` was created by `B` it will have the options prefix `-mat_lmvm_J0_`.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`
@*/
PetscErrorCode MatLMVMGetJ0(Mat B, Mat *J0)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm = (Mat_LMVM *)B->data;
  *J0  = lmvm->J0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetJ0PC - Returns a pointer to the internal `PC` object
  associated with the initial Jacobian.

  Input Parameter:
. B - A `MATLMVM` matrix

  Output Parameter:
. J0pc - `PC` object for defining the initial inverse-Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0PC()`
@*/
PetscErrorCode MatLMVMGetJ0PC(Mat B, PC *J0pc)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm = (Mat_LMVM *)B->data;
  PetscCall(KSPGetPC(lmvm->J0ksp, J0pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetJ0KSP - Returns a pointer to the internal `KSP` solver
  associated with the initial Jacobian.

  Input Parameter:
. B - A `MATLMVM` matrix

  Output Parameter:
. J0ksp - `KSP` solver for defining the initial inverse-Jacobian

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0KSP()`
@*/
PetscErrorCode MatLMVMGetJ0KSP(Mat B, KSP *J0ksp)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm   = (Mat_LMVM *)B->data;
  *J0ksp = lmvm->J0ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMApplyJ0Fwd - Applies an approximation of the forward
  matrix-vector product with the initial Jacobian.

  Input Parameters:
+ B - A `MATLMVM` matrix
- X - vector to multiply with J0

  Output Parameter:
. Y - resulting vector for the operation

  Level: advanced

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`, `MatLMVMSetJ0Scale()`, `MatLMVMSetJ0ScaleDiag()`,
          `MatLMVMSetJ0PC()`, `MatLMVMSetJ0KSP()`, `MatLMVMApplyJ0Inv()`
@*/
PetscErrorCode MatLMVMApplyJ0Fwd(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMult(lmvm->J0, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0HermitianTranspose(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMultHermitianTranspose(lmvm->J0, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMApplyJ0Inv - Applies some estimation of the initial Jacobian
  inverse to the given vector.

  Input Parameters:
+ B - A `MATLMVM` matrix
- X - vector to "multiply" with J0^{-1}

  Output Parameter:
. Y - resulting vector for the operation

  Level: advanced

  Note:
  The specific form of the application
  depends on whether the user provided a scaling factor, a J0 matrix,
  a J0 `PC`, or a J0 `KSP` object. If no form of the initial Jacobian is
  provided, the function simply does an identity matrix application
  (vector copy).

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMSetJ0()`, `MatLMVMSetJ0Scale()`, `MatLMVMSetJ0ScaleDiag()`,
          `MatLMVMSetJ0PC()`, `MatLMVMSetJ0KSP()`, `MatLMVMApplyJ0Fwd()`
@*/
PetscErrorCode MatLMVMApplyJ0Inv(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPushCreateViewerOff(PETSC_TRUE));
  PetscCall(KSPSolve(lmvm->J0ksp, X, Y));
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPopCreateViewerOff());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0InvTranspose(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPushCreateViewerOff(PETSC_TRUE));
  PetscCall(KSPSolveTranspose(lmvm->J0ksp, X, Y));
  if (lmvm->disable_ksp_viewers) PetscCall(PetscOptionsPopCreateViewerOff());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMApplyJ0InvHermitianTranspose(Mat B, Vec X, Vec Y)
{
  PetscFunctionBegin;
  if (!PetscDefined(USE_COMPLEX)) {
    PetscCall(MatLMVMApplyJ0InvTranspose(B, X, Y));
  } else {
    Vec X_conj;

    PetscCall(VecDuplicate(X, &X_conj));
    PetscCall(VecCopy(X, X_conj));
    PetscCall(VecConjugate(X_conj));
    PetscCall(MatLMVMApplyJ0InvTranspose(B, X_conj, Y));
    PetscCall(VecConjugate(Y));
    PetscCall(VecDestroy(&X_conj));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMIsAllocated - Returns a boolean flag that shows whether
  the necessary data structures for the underlying matrix is allocated.

  Input Parameter:
. B - A `MATLMVM` matrix

  Output Parameter:
. flg - `PETSC_TRUE` if allocated, `PETSC_FALSE` otherwise

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMAllocate()`, `MatLMVMReset()`
@*/
PetscErrorCode MatLMVMIsAllocated(Mat B, PetscBool *flg)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  *flg = PETSC_FALSE;
  lmvm = (Mat_LMVM *)B->data;
  if (lmvm->allocated && B->preallocated && B->assembled) *flg = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMAllocate - Produces all necessary common memory for
  LMVM approximations based on the solution and function vectors
  provided.

  Input Parameters:
+ B - A `MATLMVM` matrix
. X - Solution vector
- F - Function vector

  Level: intermediate

  Note:
  If `MatSetSizes()` and `MatSetUp()` have not been called
  before `MatLMVMAllocate()`, the allocation will read sizes from
  the provided vectors and update the matrix.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMReset()`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMAllocate(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCall((*lmvm->ops->allocate)(B, X, F));
  PetscCall(MatLMVMAllocate(lmvm->J0, X, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMResetShift - Zero the shift factor for a `MATLMVM`.

  Input Parameter:
. B - A `MATLMVM` matrix

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMAllocate()`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMResetShift(Mat B)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm        = (Mat_LMVM *)B->data;
  lmvm->shift = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMReset - Flushes all of the accumulated updates out of
  the `MATLMVM` approximation.

  Input Parameters:
+ B           - A `MATLMVM` matrix
- destructive - flag for enabling destruction of data structures

  Level: intermediate

  Note:
  In practice, this will not actually
  destroy the data associated with the updates. It simply resets
  counters, which leads to existing data being overwritten, and
  `MatSolve()` being applied as if there are no updates. A boolean
  flag is available to force destruction of the update vectors.

  If the user has provided another LMVM matrix as J0, the J0
  matrix is also reset in this function.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMAllocate()`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMReset(Mat B, PetscBool destructive)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  lmvm = (Mat_LMVM *)B->data;
  PetscCall(PetscInfo(B, "Reseting %s after %" PetscInt_FMT " iterations\n", ((PetscObject)B)->type_name, lmvm->k));
  PetscCall(MatLMVMReset(lmvm->J0, destructive));
  PetscCall((*lmvm->ops->reset)(B, destructive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetHistorySize - Set the number of past iterates to be
  stored for the construction of the limited-memory quasi-Newton update.

  Input Parameters:
+ B         - A `MATLMVM` matrix
- hist_size - number of past iterates (default 5)

  Options Database Key:
. -mat_lmvm_hist_size <m> - set number of past iterates

  Level: beginner

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetUpdateCount()`
@*/
PetscErrorCode MatLMVMSetHistorySize(Mat B, PetscInt hist_size)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(hist_size >= 0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "QN history size must be a non-negative integer.");
  if (lmvm->allocated && lmvm->m != hist_size) {
    Vec X, F;

    PetscCall(VecDuplicate(lmvm->Xprev, &X));
    PetscCall(VecDuplicate(lmvm->Fprev, &F));
    PetscCall(MatLMVMReset(B, PETSC_TRUE));
    lmvm->m = hist_size;
    PetscCall(MatLMVMAllocate(B, X, F));
    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&F));
  }
  lmvm->m = hist_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLMVMGetHistorySize(Mat B, PetscInt *hist_size)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  *hist_size = lmvm->m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetUpdateCount - Returns the number of accepted updates.

  Input Parameter:
. B - A `MATLMVM` matrix

  Output Parameter:
. nupdates - number of accepted updates

  Level: intermediate

  Note:
  This number may be greater than the total number of update vectors
  stored in the matrix. The counters are reset when `MatLMVMReset()`
  is called.

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMGetRejectCount()`, `MatLMVMReset()`
@*/
PetscErrorCode MatLMVMGetUpdateCount(Mat B, PetscInt *nupdates)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm      = (Mat_LMVM *)B->data;
  *nupdates = lmvm->nupdates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetRejectCount - Returns the number of rejected updates.
  The counters are reset when `MatLMVMReset()` is called.

  Input Parameter:
. B - A `MATLMVM` matrix

  Output Parameter:
. nrejects - number of rejected updates

  Level: intermediate

.seealso: [](ch_ksp), [LMVM Matrices](sec_matlmvm), `MATLMVM`, `MatLMVMReset()`
@*/
PetscErrorCode MatLMVMGetRejectCount(Mat B, PetscInt *nrejects)
{
  Mat_LMVM *lmvm;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  lmvm      = (Mat_LMVM *)B->data;
  *nrejects = lmvm->nrejects;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetJ0Scalar(Mat B, PetscBool *is_scalar, PetscScalar *scale)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATCONSTANTDIAGONAL, is_scalar));
  if (lmvm->square && *is_scalar) PetscCall(MatConstantDiagonalGetConstant(lmvm->J0, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMUpdateOpVecs(Mat B, LMBasis X, LMBasis OpX, PetscErrorCode (*op)(Mat, Vec, Vec))
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  PetscObjectId    J0_id;
  PetscObjectState J0_state;
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)lmvm->J0, &J0_id));
  PetscCall(PetscObjectStateGet((PetscObject)lmvm->J0, &J0_state));
  PetscCall(LMBasisGetRange(X, &oldest, &next));
  if (OpX->operator_id != J0_id || OpX->operator_state != J0_state) {
    // invalidate OpX
    OpX->k              = oldest;
    OpX->operator_id    = J0_id;
    OpX->operator_state = J0_state;
    PetscCall(LMBasisSetCachedProduct(OpX, NULL, NULL));
  }
  OpX->k = PetscMax(OpX->k, oldest);
  for (PetscInt i = OpX->k; i < next; i++) {
    Vec x_i, op_x_i;

    PetscCall(LMBasisGetVecRead(X, i, &x_i));
    PetscCall(LMBasisGetNextVec(OpX, &op_x_i));
    PetscCall(op(B, x_i, op_x_i));
    PetscCall(LMBasisRestoreNextVec(OpX, &op_x_i));
    PetscCall(LMBasisRestoreVecRead(X, i, &x_i));
  }
  PetscAssert(OpX->k == X->k && OpX->operator_id == J0_id && OpX->operator_state == J0_state, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Invalid state for operator-updated LMBasis");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMUpdateOpDiffVecs(Mat B, LMBasis Y, PetscScalar alpha, LMBasis OpX, LMBasis YmalphaOpX)
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  PetscInt         start;
  PetscObjectId    J0_id;
  PetscObjectState J0_state;
  PetscInt         oldest, next;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)lmvm->J0, &J0_id));
  PetscCall(PetscObjectStateGet((PetscObject)lmvm->J0, &J0_state));
  PetscAssert(Y->m == OpX->m, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Incompatible Y and OpX in MatLMVMUpdateOpDiffVecs()");
  PetscAssert(Y->k == OpX->k, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Stale OpX in MatLMVMUpdateOpDiffVecs()");
  PetscCall(LMBasisGetRange(Y, &oldest, &next));
  if (YmalphaOpX->operator_id != J0_id || YmalphaOpX->operator_state != J0_state) {
    // invalidate OpX
    YmalphaOpX->k              = oldest;
    YmalphaOpX->operator_id    = J0_id;
    YmalphaOpX->operator_state = J0_state;
    PetscCall(LMBasisSetCachedProduct(YmalphaOpX, NULL, NULL));
  }
  YmalphaOpX->k = PetscMax(YmalphaOpX->k, oldest);
  start         = YmalphaOpX->k;
  if (next - start == Y->m) { // full matrix AXPY
    PetscCall(MatCopy(Y->vecs, YmalphaOpX->vecs, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(YmalphaOpX->vecs, -alpha, OpX->vecs, SAME_NONZERO_PATTERN));
    YmalphaOpX->k = Y->k;
  } else {
    for (PetscInt i = start; i < next; i++) {
      Vec y_i, op_x_i, y_m_op_x_i;

      PetscCall(LMBasisGetVecRead(Y, i, &y_i));
      PetscCall(LMBasisGetVecRead(OpX, i, &op_x_i));
      PetscCall(LMBasisGetNextVec(YmalphaOpX, &y_m_op_x_i));
      PetscCall(VecAXPBYPCZ(y_m_op_x_i, 1.0, -alpha, 0.0, y_i, op_x_i));
      PetscCall(LMBasisRestoreNextVec(YmalphaOpX, &y_m_op_x_i));
      PetscCall(LMBasisRestoreVecRead(OpX, i, &op_x_i));
      PetscCall(LMBasisRestoreVecRead(Y, i, &y_i));
    }
  }
  PetscAssert(YmalphaOpX->k == Y->k && YmalphaOpX->operator_id == J0_id && YmalphaOpX->operator_state == J0_state, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Invalid state for operator-updated LMBasis");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedBasis(Mat B, MatLMVMBasisType type, LMBasis *basis_p, MatLMVMBasisType *returned_type, PetscScalar *scale)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  LMBasis     basis;
  PetscBool   is_scalar;
  PetscScalar _scale;

  PetscFunctionBegin;
  switch (type) {
  case LMBASIS_S:
  case LMBASIS_Y:
    *basis_p = lmvm->basis[type];
    if (returned_type) *returned_type = type;
    if (scale) *scale = 1.0;
    break;
  case LMBASIS_B0S:
  case LMBASIS_H0Y:
    // if B_0 = gamma * I, do not actually compute these bases
    PetscAssertPointer(returned_type, 4);
    PetscAssertPointer(scale, 5);
    PetscCall(MatLMVMGetJ0Scalar(B, &is_scalar, &_scale));
    if (is_scalar) {
      *basis_p       = lmvm->basis[type == LMBASIS_B0S ? LMBASIS_S : LMBASIS_Y];
      *returned_type = (type == LMBASIS_B0S) ? LMBASIS_S : LMBASIS_Y;
      *scale         = (type == LMBASIS_B0S) ? _scale : (1.0 / _scale);
    } else {
      LMBasis orig_basis = (type == LMBASIS_B0S) ? lmvm->basis[LMBASIS_S] : lmvm->basis[LMBASIS_Y];

      *returned_type = type;
      *scale         = 1.0;
      if (!lmvm->basis[type]) PetscCall(LMBasisCreate(MatLMVMBasisSizeOf(type) == LMBASIS_S ? lmvm->Xprev : lmvm->Fprev, lmvm->m, &lmvm->basis[type]));
      basis = lmvm->basis[type];
      PetscCall(MatLMVMUpdateOpVecs(B, orig_basis, basis, (type == LMBASIS_B0S) ? MatLMVMApplyJ0Fwd : MatLMVMApplyJ0Inv));
      *basis_p = basis;
    }
    break;
  case LMBASIS_S_MINUS_H0Y:
  case LMBASIS_Y_MINUS_B0S: {
    MatLMVMBasisType op_basis_t = (type == LMBASIS_S_MINUS_H0Y) ? LMBASIS_H0Y : LMBASIS_B0S;
    LMBasis          op_basis;

    if (returned_type) *returned_type = type;
    if (scale) *scale = 1.0;
    if (!lmvm->basis[type]) PetscCall(LMBasisCreate(MatLMVMBasisSizeOf(type) == LMBASIS_S ? lmvm->Xprev : lmvm->Fprev, lmvm->m, &lmvm->basis[type]));
    basis = lmvm->basis[type];
    PetscCall(MatLMVMGetUpdatedBasis(B, op_basis_t, &op_basis, &op_basis_t, &_scale));
    PetscCall(MatLMVMUpdateOpDiffVecs(B, lmvm->basis[MatLMVMBasisSizeOf(type)], _scale, op_basis, basis));
    *basis_p = basis;
  } break;
  default:
    PetscUnreachable();
  }
  basis = *basis_p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisGetVecRead(Mat B, MatLMVMBasisType type, PetscInt i, Vec *y, PetscScalar *scale)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  LMBasis     basis;
  PetscBool   is_scalar;
  PetscScalar _scale;

  PetscFunctionBegin;
  switch (type) {
  case LMBASIS_S:
  case LMBASIS_Y:
    *scale = 1.0;
    PetscCall(LMBasisGetVecRead(lmvm->basis[type], i, y));
    break;
  case LMBASIS_B0S:
  case LMBASIS_H0Y:
    // if B_0 = gamma * I, do not actually compute these bases
    PetscCall(MatLMVMGetJ0Scalar(B, &is_scalar, &_scale));
    if (is_scalar) {
      *scale = (type == LMBASIS_B0S) ? _scale : (1.0 / _scale);
      PetscCall(LMBasisGetVecRead(lmvm->basis[type == LMBASIS_B0S ? LMBASIS_S : LMBASIS_Y], i, y));
    } else if (lmvm->do_not_cache_J0_products) {
      Vec     tmp;
      Vec     w;
      LMBasis orig_basis = (type == LMBASIS_B0S) ? lmvm->basis[LMBASIS_S] : lmvm->basis[LMBASIS_Y];
      LMBasis size_basis = lmvm->basis[MatLMVMBasisSizeOf(type)];

      PetscCall(LMBasisGetVecRead(orig_basis, i, &w));
      PetscCall(LMBasisGetWorkVec(size_basis, &tmp));
      if (type == LMBASIS_B0S) {
        PetscCall(MatLMVMApplyJ0Fwd(B, w, tmp));
      } else {
        PetscCall(MatLMVMApplyJ0Inv(B, w, tmp));
      }
      PetscCall(LMBasisRestoreVecRead(orig_basis, i, &w));
      *scale = 1.0;
      *y     = tmp;
    } else {
      LMBasis     basis;
      PetscScalar dummy;

      PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis, &type, &dummy));
      PetscCall(LMBasisGetVecRead(basis, i, y));
      *scale = 1.0;
    }
    break;
  case LMBASIS_S_MINUS_H0Y:
  case LMBASIS_Y_MINUS_B0S: {
    if (lmvm->do_not_cache_J0_products) {
      MatLMVMBasisType op_basis_t  = (type == LMBASIS_S_MINUS_H0Y) ? LMBASIS_H0Y : LMBASIS_B0S;
      MatLMVMBasisType antibasis_t = (type == LMBASIS_S_MINUS_H0Y) ? LMBASIS_S : LMBASIS_Y;
      LMBasis          antibasis   = lmvm->basis[antibasis_t];
      Vec              tmp;
      Vec              w, z;
      PetscScalar      _scale;

      PetscCall(LMBasisGetWorkVec(antibasis, &tmp));
      PetscCall(MatLMVMBasisGetVecRead(B, op_basis_t, i, &z, &_scale));
      PetscCall(LMBasisGetVecRead(antibasis, i, &w));
      PetscCall(VecWAXPY(tmp, -_scale, z, w));
      PetscCall(LMBasisRestoreVecRead(antibasis, i, &w));
      PetscCall(MatLMVMBasisRestoreVecRead(B, op_basis_t, i, &z, &_scale));
      *y = tmp;
    } else {
      PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis, NULL, NULL));
      PetscCall(LMBasisGetVecRead(basis, i, y));
    }
    *scale = 1.0;
  } break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisRestoreVecRead(Mat B, MatLMVMBasisType type, PetscInt i, Vec *y, PetscScalar *scale)
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  PetscBool   is_scalar;
  PetscScalar _scale;

  PetscFunctionBegin;
  switch (type) {
  case LMBASIS_S:
  case LMBASIS_Y:
    PetscCall(LMBasisRestoreVecRead(lmvm->basis[type], i, y));
    break;
  case LMBASIS_B0S:
  case LMBASIS_H0Y:
    // if B_0 = gamma * I, do not actually compute these bases
    PetscCall(MatLMVMGetJ0Scalar(B, &is_scalar, &_scale));
    if (is_scalar) {
      PetscCall(LMBasisRestoreVecRead(lmvm->basis[type == LMBASIS_B0S ? LMBASIS_S : LMBASIS_Y], i, y));
    } else if (lmvm->do_not_cache_J0_products) {
      LMBasis size_basis = lmvm->basis[MatLMVMBasisSizeOf(type)];

      PetscCall(LMBasisRestoreWorkVec(size_basis, y));
    } else {
      PetscCall(LMBasisRestoreVecRead(lmvm->basis[type], i, y));
    }
    break;
  case LMBASIS_S_MINUS_H0Y:
  case LMBASIS_Y_MINUS_B0S: {
    if (lmvm->do_not_cache_J0_products) {
      MatLMVMBasisType antibasis_t = (type == LMBASIS_S_MINUS_H0Y) ? LMBASIS_S : LMBASIS_Y;
      LMBasis          antibasis   = lmvm->basis[antibasis_t];
      PetscCall(LMBasisRestoreWorkVec(antibasis, y));
    } else {
      PetscCall(LMBasisRestoreVecRead(lmvm->basis[type], i, y));
    }
  } break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetRange(Mat B, PetscInt *oldest, PetscInt *next)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(lmvm->basis[LMBASIS_S], oldest, next));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetWorkRow(Mat B, Vec *array_p)
{
  LMBasis basis;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &basis, NULL, NULL));
  PetscCall(LMBasisGetWorkRow(basis, array_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMRestoreWorkRow(Mat B, Vec *array_p)
{
  LMBasis basis;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, LMBASIS_Y, &basis, NULL, NULL));
  PetscCall(LMBasisRestoreWorkRow(basis, array_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMApplyOpThenVecs(PetscScalar alpha, Mat B, PetscInt oldest, PetscInt next, MatLMVMBasisType type_S, PetscErrorCode (*op)(Mat, Vec, Vec), Vec x, PetscScalar beta, Vec y)
{
  LMBasis S;
  Vec     B0x;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, type_S, &S, NULL, NULL));
  PetscCall(LMBasisGetWorkVec(S, &B0x));
  PetscCall(op(B, x, B0x));
  PetscCall(LMBasisGEMVH(S, oldest, next, alpha, B0x, beta, y));
  PetscCall(LMBasisRestoreWorkVec(S, &B0x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMApplyVecsThenOp(PetscScalar alpha, Mat B, PetscInt oldest, PetscInt next, MatLMVMBasisType type_S, MatLMVMBasisType type_Y, PetscErrorCode (*op)(Mat, Vec, Vec), Vec x, PetscScalar beta, Vec y)
{
  LMBasis S, Y;
  Vec     S_x;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, type_S, &S, NULL, NULL));
  PetscCall(MatLMVMGetUpdatedBasis(B, type_Y, &Y, NULL, NULL));
  PetscCall(LMBasisGetWorkVec(S, &S_x));
  PetscCall(LMBasisGEMV(S, oldest, next, alpha, x, 0.0, S_x));
  if (beta == 0.0) {
    PetscCall(op(B, S_x, y));
  } else {
    Vec B0S_x;
    PetscCall(LMBasisGetWorkVec(Y, &B0S_x));
    PetscCall(op(B, S_x, B0S_x));
    PetscCall(VecAYPX(y, beta, B0S_x));
    PetscCall(LMBasisRestoreWorkVec(Y, &B0S_x));
  }
  PetscCall(LMBasisRestoreWorkVec(S, &S_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisGEMVH(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec v, PetscScalar beta, Vec array)
{
  Mat_LMVM        *lmvm              = (Mat_LMVM *)B->data;
  PetscBool        cache_J0_products = lmvm->do_not_cache_J0_products ? PETSC_FALSE : PETSC_TRUE;
  LMBasis          basis;
  MatLMVMBasisType basis_t;
  PetscScalar      gamma;

  PetscFunctionBegin;
  if (cache_J0_products || type == LMBASIS_S || type == LMBASIS_Y) {
    PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis, &basis_t, &gamma));
    PetscCall(LMBasisGEMVH(basis, oldest, next, alpha * gamma, v, beta, array));
  } else {
    switch (type) {
    case LMBASIS_B0S:
      PetscCall(MatLMVMApplyOpThenVecs(alpha, B, oldest, next, LMBASIS_S, MatLMVMApplyJ0HermitianTranspose, v, beta, array));
      break;
    case LMBASIS_H0Y:
      PetscCall(MatLMVMApplyOpThenVecs(alpha, B, oldest, next, LMBASIS_Y, MatLMVMApplyJ0InvHermitianTranspose, v, beta, array));
      break;
    case LMBASIS_Y_MINUS_B0S:
      PetscCall(MatLMVMApplyOpThenVecs(-alpha, B, oldest, next, LMBASIS_S, MatLMVMApplyJ0HermitianTranspose, v, beta, array));
      PetscCall(LMBasisGEMVH(lmvm->basis[LMBASIS_Y], oldest, next, alpha, v, 1.0, array));
      break;
    case LMBASIS_S_MINUS_H0Y:
      PetscCall(MatLMVMApplyOpThenVecs(-alpha, B, oldest, next, LMBASIS_Y, MatLMVMApplyJ0InvHermitianTranspose, v, beta, array));
      PetscCall(LMBasisGEMVH(lmvm->basis[LMBASIS_S], oldest, next, alpha, v, 1.0, array));
      break;
    default:
      PetscUnreachable();
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisMultHermitianTranspose(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, Vec v, Vec array)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMBasisGEMVH(B, type, oldest, next, 1.0, v, 0.0, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// x must come from MatLMVMGetRowWork()
PETSC_INTERN PetscErrorCode MatLMVMBasisGEMV(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, PetscScalar alpha, Vec x, PetscScalar beta, Vec y)
{
  Mat_LMVM *lmvm              = (Mat_LMVM *)B->data;
  PetscBool cache_J0_products = lmvm->do_not_cache_J0_products ? PETSC_FALSE : PETSC_TRUE;
  LMBasis   basis;

  PetscFunctionBegin;
  if (cache_J0_products || type == LMBASIS_S || type == LMBASIS_Y) {
    PetscScalar      gamma;
    MatLMVMBasisType base_type;

    PetscCall(MatLMVMGetUpdatedBasis(B, type, &basis, &base_type, &gamma));
    PetscCall(LMBasisGEMV(basis, oldest, next, alpha * gamma, x, beta, y));
  } else {
    switch (type) {
    case LMBASIS_B0S:
      PetscCall(MatLMVMApplyVecsThenOp(alpha, B, oldest, next, LMBASIS_S, LMBASIS_Y, MatLMVMApplyJ0Fwd, x, beta, y));
      break;
    case LMBASIS_H0Y:
      PetscCall(MatLMVMApplyVecsThenOp(alpha, B, oldest, next, LMBASIS_Y, LMBASIS_S, MatLMVMApplyJ0Inv, x, beta, y));
      break;
    case LMBASIS_Y_MINUS_B0S:
      PetscCall(LMBasisGEMV(lmvm->basis[LMBASIS_Y], oldest, next, alpha, x, beta, y));
      PetscCall(MatLMVMApplyVecsThenOp(-alpha, B, oldest, next, LMBASIS_S, LMBASIS_Y, MatLMVMApplyJ0Fwd, x, 1.0, y));
      break;
    case LMBASIS_S_MINUS_H0Y:
      PetscCall(LMBasisGEMV(lmvm->basis[LMBASIS_S], oldest, next, alpha, x, beta, y));
      PetscCall(MatLMVMApplyVecsThenOp(-alpha, B, oldest, next, LMBASIS_Y, LMBASIS_S, MatLMVMApplyJ0Inv, x, 1.0, y));
      break;
    default:
      PetscUnreachable();
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMBasisMultAdd(Mat B, MatLMVMBasisType type, PetscInt oldest, PetscInt next, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMBasisGEMV(B, type, oldest, next, 1.0, x, 1.0, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMCreateProducts(Mat B, LMBlockType block_type, LMProducts *products)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(LMProductsCreate(lmvm->basis[LMBASIS_S], block_type, products));
  (*products)->debug = lmvm->debug;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMProductsUpdate(Mat B, MatLMVMBasisType type_X, MatLMVMBasisType type_Y, LMBlockType block_type)
{
  Mat_LMVM        *lmvm = (Mat_LMVM *)B->data;
  LMBasis          X, Y;
  MatLMVMBasisType true_type_X, true_type_Y;
  PetscScalar      alpha_X, alpha_Y;
  PetscInt         oldest, next;
  LMProducts       G;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedBasis(B, type_X, &X, &true_type_X, &alpha_X));
  PetscCall(MatLMVMGetUpdatedBasis(B, type_Y, &Y, &true_type_Y, &alpha_Y));
  if (!lmvm->products[block_type][true_type_X][true_type_Y]) PetscCall(MatLMVMCreateProducts(B, block_type, &lmvm->products[block_type][true_type_X][true_type_Y]));
  PetscCall(LMProductsUpdate(lmvm->products[block_type][true_type_X][true_type_Y], X, Y));
  if (true_type_X == type_X && true_type_Y == type_Y) PetscFunctionReturn(PETSC_SUCCESS);
  if (!lmvm->products[block_type][type_X][type_Y]) PetscCall(MatLMVMCreateProducts(B, block_type, &lmvm->products[block_type][type_X][type_Y]));
  G = lmvm->products[block_type][type_X][type_Y];
  PetscCall(MatLMVMGetRange(B, &oldest, &next));
  PetscCall(LMProductsPrepare(G, lmvm->J0, oldest, next));
  if (G->k < lmvm->k) {
    PetscCall(LMProductsCopy(lmvm->products[block_type][true_type_X][true_type_Y], lmvm->products[block_type][type_X][type_Y]));
    if (alpha_X * alpha_Y != 1.0) PetscCall(LMProductsScale(lmvm->products[block_type][type_X][type_Y], alpha_X * alpha_Y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMGetUpdatedProducts(Mat B, MatLMVMBasisType type_X, MatLMVMBasisType type_Y, LMBlockType block_type, LMProducts *lmwd)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatLMVMProductsUpdate(B, type_X, type_Y, block_type));
  *lmwd = lmvm->products[block_type][type_X][type_Y];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMProductsInsertDiagonalValue(Mat B, MatLMVMBasisType type_X, MatLMVMBasisType type_Y, PetscInt idx, PetscScalar v)
{
  Mat_LMVM  *lmvm = (Mat_LMVM *)B->data;
  LMProducts products;

  PetscFunctionBegin;
  if (!lmvm->products[LMBLOCK_DIAGONAL][type_X][type_Y]) PetscCall(MatLMVMCreateProducts(B, LMBLOCK_DIAGONAL, &lmvm->products[LMBLOCK_DIAGONAL][type_X][type_Y]));
  products = lmvm->products[LMBLOCK_DIAGONAL][type_X][type_Y];
  PetscCall(LMProductsInsertNextDiagonalValue(products, idx, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMProductsGetDiagonalValue(Mat B, MatLMVMBasisType type_X, MatLMVMBasisType type_Y, PetscInt idx, PetscScalar *v)
{
  LMProducts products = NULL;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetUpdatedProducts(B, type_X, type_Y, LMBLOCK_DIAGONAL, &products));
  PetscCall(LMProductsGetDiagonalValue(products, idx, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
