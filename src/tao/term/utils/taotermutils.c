#include <petsc/private/taoimpl.h>

PETSC_INTERN PetscErrorCode VecIfNotCongruentGetSameLayoutVec(Vec a, Vec *b)
{
  PetscFunctionBegin;
  if (!(*b)) {
    PetscCall(VecDuplicate(a, b));
  } else {
    PetscLayout layout_a, layout_b;
    PetscBool   is_same;

    PetscCall(VecGetLayout(a, &layout_a));
    PetscCall(VecGetLayout(*b, &layout_b));
    PetscCall(PetscLayoutCompare(layout_a, layout_b, &is_same));
    if (!is_same) {
      PetscCall(VecDestroy(b));
      PetscCall(VecDuplicate(a, b));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateHessianMatricesDefault_H_Internal(TaoTerm term, Mat *H, Mat *Hpre, PetscBool Hpre_is_H, MatType H_mattype)
{
  Mat       _H;
  PetscBool is_shell = PETSC_FALSE;
  PetscBool is_mffd  = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(H_mattype, MATSHELL, &is_shell));
  if (is_shell) PetscCall(PetscInfo(term, "TaoTerm currently does not support MATSHELL for Hessian matrices. Using default MatCreate routines.\n"));
  PetscCall(PetscStrcmp(H_mattype, MATMFFD, &is_mffd));
  if (is_mffd) {
    PetscCall(TaoTermCreateHessianMFFD(term, &_H));
  } else {
    PetscLayout sol_layout;
    VecType     sol_vec_type;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &_H));
    PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
    PetscCall(MatSetLayouts(_H, sol_layout, sol_layout));
    PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
    if (H_mattype) PetscCall(MatSetType(_H, H_mattype));
    else PetscCall(MatSetVecType(_H, sol_vec_type));
    PetscCall(MatSetOption(_H, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(_H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  }

  if (H) {
    PetscCall(PetscObjectReference((PetscObject)_H));
    *H = _H;
  }
  if (Hpre && Hpre_is_H) {
    PetscCall(PetscObjectReference((PetscObject)_H));
    *Hpre = _H;
  }
  PetscCall(MatDestroy(&_H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateHessianMatricesDefault_Hpre_Internal(TaoTerm term, Mat *H, Mat *Hpre, PetscBool Hpre_is_H, MatType Hpre_mattype)
{
  Mat       _Hpre;
  PetscBool is_shell = PETSC_FALSE;
  PetscBool is_mffd  = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(Hpre_mattype, MATSHELL, &is_shell));
  if (is_shell) PetscCall(PetscInfo(term, "TaoTerm currently does not support MATSHELL for Hessian matrices. Using default MatCreate routines.\n"));
  PetscCall(PetscStrcmp(Hpre_mattype, MATMFFD, &is_mffd));
  if (is_mffd) {
    PetscCall(TaoTermCreateHessianMFFD(term, &_Hpre));
  } else {
    PetscLayout sol_layout;
    VecType     sol_vec_type;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &_Hpre));
    PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
    PetscCall(MatSetLayouts(_Hpre, sol_layout, sol_layout));
    PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
    if (Hpre_mattype) PetscCall(MatSetType(_Hpre, Hpre_mattype));
    else PetscCall(MatSetVecType(_Hpre, sol_vec_type));
    PetscCall(MatSetOption(_Hpre, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(_Hpre, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  }
  *Hpre = _Hpre;
  PetscFunctionReturn(PETSC_SUCCESS);
}
