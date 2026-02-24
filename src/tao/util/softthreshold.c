#include <petsc/private/taoimpl.h>
#include <../src/tao/term/impls/sum/taotermsum.h> // TaoTermSumVecNestGetSubVecsRead(), TaoTermSumVecSetRestoreSubVecsRead()

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
  input - ub  & \text{if } input > ub \\
  0           & \text{if } lb \leq input \leq ub \\
  input - lb  & \text{if } input < lb
  \end{cases}
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
