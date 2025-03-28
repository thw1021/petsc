#include <petscmat.h>
#include <../src/mat/utils/veccreatematdense.h>

/*MC
  MATDENSEFROMVECTYPE - MATDENSEFROMVECTYPE = "densefromvectype" - A constructor type that will return a dense matrix
  that matches the `VecType` that has already been set with `MatSetVecType()`.

  Options Database Key:
. -mat_type densefromvectype - sets the matrix type to `MATSEQDENSE` during a call to `MatSetFromOptions()`

  Level: advanced

  Notes:
  This constructor type is for constructing dense matrices from the `PetscOptions` database.  If you already know at
  compile time that you want a dense matrix, you should just use `MatDenseCreateFromVecType()`.

.seealso: [](ch_matrices), `Mat`, `MatDenseCreateFromVecType()`
M*/
PETSC_INTERN PetscErrorCode MatCreate_DenseFromVecType(Mat B)
{
  MPI_Comm  comm;
  VecType   vec_type, root_type;
  PetscBool is_cuda, is_hip;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)B, &comm));
  PetscCall(MatGetVecType(B, &vec_type));
  root_type = vec_type;
  PetscCall(VecTypeGetRootTypeForMatDense_Private(comm, vec_type, &root_type, &is_cuda, &is_hip));
  PetscCall(MatSetVecType(B, root_type));
  PetscCall(MatCreate_DenseFromVecType_Private(B, is_cuda, is_hip, PETSC_DECIDE, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
