#include <petscmat.h> /*I    "petscmat.h"   I*/
#include "veccreatematdense.h"

PETSC_INTERN PetscErrorCode VecTypeGetRootTypeForMatDense_Private(MPI_Comm comm, VecType vec_type, VecType *root_type, PetscBool *is_cuda, PetscBool *is_hip)
{
  PetscBool is_std, is_kokkos;

  PetscFunctionBegin;
  *root_type = VECSTANDARD;
  PetscCall(PetscStrcmpAny(vec_type, &is_std, VECSTANDARD, VECMPI, VECSEQ, ""));
  PetscCall(PetscStrcmpAny(vec_type, is_cuda, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));
  PetscCall(PetscStrcmpAny(vec_type, is_hip, VECHIP, VECMPIHIP, VECSEQHIP, ""));
  PetscCall(PetscStrcmpAny(vec_type, &is_kokkos, VECKOKKOS, VECMPIKOKKOS, VECSEQKOKKOS, ""));
  PetscCheck(is_std || *is_cuda || *is_hip || is_kokkos, comm, PETSC_ERR_SUP, "Not for type %s", vec_type);
  if (*is_cuda) *root_type = VECCUDA;
  else if (*is_hip) *root_type = VECHIP;
  else if (is_kokkos) {
    /* We support only one type of kokkos device */
    PetscCheck(!PetscDefined(HAVE_MACRO_KOKKOS_ENABLE_SYCL), comm, PETSC_ERR_SUP, "Not for sycl backend");
    if (PetscDefined(HAVE_MACRO_KOKKOS_ENABLE_CUDA)) *is_cuda = PETSC_TRUE;
    else if (PetscDefined(HAVE_MACRO_KOKKOS_ENABLE_HIP)) *is_hip = PETSC_TRUE;
    else is_std = PETSC_TRUE;
    *root_type = VECKOKKOS;
  }
  // early exit errors before matrix is created
  if (*is_cuda) PetscCheck(PetscDefined(HAVE_CUDA), comm, PETSC_ERR_SUP, "PETSc not compiled with CUDA support");
  if (*is_hip) PetscCheck(PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "PETSc not compiled with HIP support");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatCreate_DenseFromVecType_Private(Mat B, PetscBool is_cuda, PetscBool is_hip, PetscInt lda, PetscScalar *data)
{
  PetscFunctionBegin;
  if (is_cuda) {
#if defined(PETSC_HAVE_CUDA)
    PetscCall(MatSetType(B, MATDENSECUDA));
    if (lda > 0) PetscCall(MatDenseSetLDA(B, lda));
    PetscCall(MatDenseCUDASetPreallocation(B, data));
#endif
  } else if (is_hip) {
#if defined(PETSC_HAVE_HIP)
    PetscCall(MatSetType(B, MATDENSEHIP));
    if (lda > 0) PetscCall(MatDenseSetLDA(B, lda));
    PetscCall(MatDenseHIPSetPreallocation(B, data));
#endif
  } else {
    PetscCall(MatSetType(B, MATDENSE));
    if (lda > 0) PetscCall(MatDenseSetLDA(B, lda));
    PetscCall(MatSeqDenseSetPreallocation(B, data));
    PetscCall(MatMPIDenseSetPreallocation(B, data));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateDenseFromVecType - Create a matrix that matches the type of a Vec.

  Collective

  Input Parameters:
+ comm  - the communicator
. vtype - the vector type
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n     - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M     - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
. lda   - optional leading dimension. Pass any non-positive number to use the default.
- data  - optional location of matrix data, which should have the same memory type as the vector. Pass `NULL` to have PETSc take care of matrix memory allocation.

  Output Parameter:
. A - the dense matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateDense()`, `MatCreateDenseCUDA()`, `MatCreateDenseHIP()`, `PetscMemType`
@*/
PetscErrorCode MatCreateDenseFromVecType(MPI_Comm comm, VecType vtype, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt lda, PetscScalar *data, Mat *A)
{
  VecType   root_type = VECSTANDARD;
  PetscBool iscuda, iship;

  PetscFunctionBegin;
  PetscCall(VecTypeGetRootTypeForMatDense_Private(comm, vtype, &root_type, &iscuda, &iship));
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCall(MatSetVecType(*A, root_type));
  PetscCall(MatCreate_DenseFromVecType_Private(*A, iscuda, iship, lda, data));
  PetscFunctionReturn(PETSC_SUCCESS);
}
