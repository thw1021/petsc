#include <petsc/private/taoimpl.h>
#include <../src/tao/term/impls/sum/taotermsum.h> // TaoTermSumVecNestGetSubVecsRead(), TaoTermSumVecSetRestoreSubVecsRead()
#include "softthreshold.h"

/*@
  TaoSoftThreshold - Calculates soft thresholding routine with input vector
  and given lower and upper bound and returns it to output vector.

  Input Parameters:
+ in - input vector to be thresholded
. lb - lower bound
- ub - upper bound

  Output Parameter:
. out - Soft thresholded output vector

  Notes:
  Soft thresholding is defined as
  \[ S(input,lb,ub) =
  \begin{cases}
  input - ub  \text{input > ub} \\
  0           \text{lb =< input <= ub} \\
  input - lb  \text{input < lb} \\
  \]

  Level: developer

.seealso: `Tao`, `Vec`
@*/
PetscErrorCode TaoSoftThreshold(Vec in, PetscReal lb, PetscReal ub, Vec out)
{
  PetscInt     i, nlocal, mlocal;
  PetscScalar *inarray, *outarray;

  PetscFunctionBegin;
  PetscCall(VecGetArrayPair(in, out, &inarray, &outarray));
  PetscCall(VecGetLocalSize(in, &nlocal));
  PetscCall(VecGetLocalSize(out, &mlocal));

  PetscCheck(nlocal == mlocal, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Input and output vectors need to be of same size.");
  if (lb == ub) {
    PetscCall(VecRestoreArrayPair(in, out, &inarray, &outarray));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(lb <= ub, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Lower bound needs to be lower than upper bound.");

  for (i = 0; i < nlocal; i++) outarray[i] = PetscMax(0, PetscRealPart(inarray[i]) - ub) + PetscMin(0, PetscRealPart(inarray[i]) - lb);

  PetscCall(VecRestoreArrayPair(in, out, &inarray, &outarray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Does tao's objective term have the form alpha * || x - a ||_l1 + beta/2 * || x - b ||_l2 ?
static PetscErrorCode TaoIsSoftThreshold_Internal(Tao tao, PetscBool *is_softthreshold, PetscReal *l1_scale, PetscReal *l2_scale, Vec *l1_shift, Vec *l2_shift)
{
#if defined(PETSC_USE_COMPLEX)
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  TaoTerm   objective;
  PetscBool is_sum;
  Vec       params;
  Mat       map;
  PetscInt  num_terms;
  PetscBool is_l1[2];
  PetscBool is_l2[2];
  PetscReal scale[2];

  PetscFunctionBegin;
  *is_softthreshold = PETSC_FALSE;
  if (l1_scale) *l1_scale = 0.;
  if (l2_scale) *l2_scale = 0.;
  if (l1_shift) *l1_shift = NULL;
  if (l2_shift) *l2_shift = NULL;
  PetscCall(TaoGetTerm(tao, NULL, &objective, &params, &map));
  PetscCall(PetscObjectTypeCompare((PetscObject)objective, TAOTERMSUM, &is_sum));
  if (!is_sum || map) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoTermSumGetNumSubterms(objective, &num_terms));
  if (num_terms != 2) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = 0; i < 2; i++) {
    TaoTerm subterm;
    Mat     submap;

    PetscCall(TaoTermSumGetSubterm(objective, i, NULL, &scale[i], &subterm, &submap));
    if (submap) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(PetscObjectTypeCompare((PetscObject)subterm, TAOTERML1, &is_l1[i]));
    if (is_l1[i]) {
      PetscReal epsilon;

      PetscCall(TaoTermL1GetEpsilon(subterm, &epsilon));
      if (epsilon != 0.) is_l1[i] = PETSC_FALSE;
    }
    PetscCall(PetscObjectTypeCompare((PetscObject)subterm, TAOTERMHALFL2SQUARED, &is_l2[i]));
  }
  if (!(is_l1[0] && is_l2[1]) && !(is_l1[1] && is_l2[0])) PetscFunctionReturn(PETSC_SUCCESS);
  *is_softthreshold = PETSC_TRUE;
  if (l1_scale) *l1_scale = is_l1[0] ? scale[0] : scale[1];
  if (l2_scale) *l2_scale = is_l2[0] ? scale[0] : scale[1];
  if (params && (l1_shift || l2_shift)) {
    Vec       *sub_params = NULL;
    PetscBool *is_dummy   = NULL;

    PetscCall(TaoTermSumVecNestGetSubVecsRead(params, NULL, &sub_params, &is_dummy));
    for (PetscInt i = 0; i < 2; i++) {
      Vec sub_param = TaoTermSumGetSubVec(params, sub_params, is_dummy, i);

      if (l1_shift && is_l1[i]) {
        PetscCall(PetscObjectReference((PetscObject)sub_param));
        *l1_shift = sub_param;
      }
      if (l2_shift && is_l2[i]) {
        PetscCall(PetscObjectReference((PetscObject)sub_param));
        *l2_shift = sub_param;
      }
    }
    PetscCall(TaoTermSumVecNestRestoreSubVecsRead(params, NULL, &sub_params, &is_dummy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
#endif
}

PetscErrorCode TaoIsSoftThreshold(Tao tao, PetscBool *is_softthreshold)
{
  PetscFunctionBegin;
  PetscCall(TaoIsSoftThreshold_Internal(tao, is_softthreshold, NULL, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* If the objective function has the form

   alpha * || x - a ||_l1 + beta/2 * || x - b ||_l2

   which can also be written

   alpha * || x - a ||_l1 + beta/2 * || (x - a) - (b - a) ||_l2

   and we let y = x - a and c = b - a, then we are trying to find

   min_y (alpha / beta) * || y ||_l1 + 1/2 * || y - c ||_l2

   which has the solution

          / c_i - (alpha / beta),  c_i  >  (alpha / beta),
   y_i = <  0                   , |c_i| <= (alpha / beta),
          \ c_i + (alpha / beta),  c_i  < -(alpha / beta)
*/
PetscErrorCode TaoSolve_SoftThreshold(Tao tao)
{
  PetscBool is_softthreshold;
  PetscReal l1_scale, l2_scale, scale, obj;
  Vec       l1_shift, l2_shift, sol;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(TaoIsSoftThreshold_Internal(tao, &is_softthreshold, &l1_scale, &l2_scale, &l1_shift, &l2_shift));
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCheck(is_softthreshold, comm, PETSC_ERR_PLIB, "Tao is not soft threshold problem");
  PetscCheck(l1_scale >= 0. && l2_scale >= 0., comm, PETSC_ERR_SUP, "Both l1 and l2 scales must be nonnegative");
  PetscCall(TaoGetSolution(tao, &sol));
  if (l1_scale == 0. && l2_scale == 0.) { // objective function is everywhere zero, choose the origin as a solution
    PetscCall(VecZeroEntries(sol));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (l1_scale == 0.) { // l1 term is zero, l2_shift is the solution
    if (l2_shift == NULL) {
      PetscCall(VecZeroEntries(sol));
    } else {
      PetscCall(VecCopy(l2_shift, sol));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (l2_scale == 0.) { // l2 term is zero, l1_shift is the solution
    if (l1_shift == NULL) {
      PetscCall(VecZeroEntries(sol));
    } else {
      PetscCall(VecCopy(l1_shift, sol));
    }
  }
  if (l1_shift == NULL && l2_shift == NULL) { // neither norm is shifted, zero is the solution
    PetscCall(VecZeroEntries(sol));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  scale = (l1_scale / l2_scale);
  if (l1_shift == NULL) {
    PetscCall(VecCopy(l2_shift, sol));
  } else if (l2_shift == NULL) {
    PetscCall(VecCopy(l1_shift, sol));
    PetscCall(VecScale(sol, -1.));
  } else {
    PetscCall(VecWAXPY(sol, -1., l1_shift, l2_shift));
  }
  PetscCall(TaoSoftThreshold(sol, -scale, scale, sol));
  if (l1_shift) PetscCall(VecAXPY(sol, 1.0, l1_shift));
  PetscCall(VecDestroy(&l1_shift));
  PetscCall(VecDestroy(&l2_shift));
  PetscCall(TaoComputeObjective(tao, sol, &obj));
  if (tao->gradient) PetscCall(VecZeroEntries(tao->gradient));
  PetscCall(TaoSetIterationNumber(tao, 0));
  PetscCall(TaoMonitor(tao, 0, obj, 0., 0., 0.));
  PetscCall(TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
