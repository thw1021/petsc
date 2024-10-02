#include <petsc/private/taoimpl.h>
#include <petsc/private/vecimpl.h>

/* Cases:
 *
 * Case 1: alpha==beta==0                              -> (XX00) NO_OP
 * Case 2: beta != 0, alpha == 0, q == NULL            -> (0X10) ZERO. x \gets zero
 * Case 3: beta != 0, alpha == 0, q != NULL            -> (1X10) Q, x \gets q
 * Case 4: beta == 0, alpha != 0, p == NULL            -> (X001) SOLVE, x \gets argmin_y f(y)
 * Case 5: beta == 0, alpha != 0, p != NULL            -> (X101) SOLVE_PARAM, x \gets argmin_y f(y+p)
 * Case 6: beta != 0, alpha != 0, p == q == NULL       -> (0011) SOLVE_COMPOSITE, x \gets argmin_y a*f(y) + b*g(y). Tikhonov Regularization
 * Case 7: beta != 0, alpha != 0, p != NULL, q == NULL -> (0111) SOLVE_COMPOSITE_TRANS, x \gets argmin_y a*f(y;p) + b*g(y). Tikhonov Regularization on parameterized function
 * Case 8: beta != 0, alpha != 0, p == NULL, q != NULL -> (1011) PROX, x \gets prox_{alpha\beta, f}(q)
 * Case 9: beta != 0, alpha != 0, p != NULL, q != NULL -> (1111) PROX_TRANS, x \gets prox with translation vec p at q
 *                                                                                                            */
PETSC_INTERN PetscErrorCode TaoTermProxL2FindOps_Internal(Vec q, Vec p, PetscReal beta, PetscReal alpha, TaoTermProxMapL2Op *l2ops)
{
  PetscBool a_zb, za_b, a_b;

  PetscFunctionBegin;
  a_zb = (alpha != 0 && beta == 0);
  za_b = (alpha == 0 && beta != 0);
  a_b  = (alpha != 0 && beta != 0);

  if (alpha == 0 && beta == 0) {
    *l2ops = TAOTERM_PROX_NO_OP;
  } else if (a_zb && !q) {
    *l2ops = TAOTERM_PROX_ZERO;
  } else if (a_zb && q) {
    *l2ops = TAOTERM_PROX_Q;
  } else if (za_b && !p) {
    *l2ops = TAOTERM_PROX_SOLVE;
  } else if (za_b && p) {
    *l2ops = TAOTERM_PROX_SOLVE_PARAM;
  } else if (a_b && !p && !q) {
    *l2ops = TAOTERM_PROX_SOLVE_COMPOSITE;
  } else if (a_b && p && !q) {
    *l2ops = TAOTERM_PROX_SOLVE_COMPOSITE_TRANS;
  } else if (a_b && q && !p) {
    *l2ops = TAOTERM_PROX_PROX;
  } else if (a_b && q && p) {
    *l2ops = TAOTERM_PROX_PROX_TRANS;
  } else PetscUnreachable();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Checkes whether two vectors are of compatible comm and size
// TODO does this function beloing here?
PETSC_INTERN PetscErrorCode TaoTermWorkvecTestCompatibility(Vec vec1, Vec vec2, PetscBool *flg)
{
  PetscBool   typeflg;
  PetscMPIInt mpiflg;
  PetscInt    low1, low2, high1, high2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec1, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(vec2, VEC_CLASSID, 2);
  PetscAssertPointer(flg, 3);
  PetscCall(PetscObjectObjectTypeCompare((PetscObject)vec1, (PetscObject)vec2, &typeflg));
  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)(vec1)), PetscObjectComm((PetscObject)vec2), &mpiflg));
  PetscCall(VecGetOwnershipRange(vec1, &low1, &high1));
  PetscCall(VecGetOwnershipRange(vec2, &low2, &high2));
  if (typeflg && mpiflg && (low1 == low2) && (high1 == high2)) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
