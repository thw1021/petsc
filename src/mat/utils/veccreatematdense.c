#include <petscmat.h>              /*I "petscmat.h" I*/
#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

/*@
  VecCreateMatDense - Create a matrix that matches the type of a Vec.

  Collective

  Input Parameters:
+ X    - the vector
. m    - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n    - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M    - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N    - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
- data - optional location of matrix data, which should have the same memory type as the vector. Pass `NULL` to have PETSc to control matrix memory allocation.

  Output Parameter:
. A - the matrix. `A` will have the same communicator as `X` and the same `PetscMemType`.

  Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MatCreateDense()', `MatCreateDenseCUDA()`, `MatCreateDenseHIP()`, `PetscMemType`
@*/
PetscErrorCode VecCreateMatDense(Vec X, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar *data, Mat *A)
{
  VecType   root_type;
  PetscBool isstd, iscuda, iship;
  MPI_Comm  comm;

  PetscFunctionBegin;
  PetscCall(VecGetRootType_Private(X, &root_type));
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCall(PetscStrcmp(root_type, VECSTANDARD, &isstd));
  PetscCall(PetscStrcmp(root_type, VECCUDA, &iscuda));
  PetscCall(PetscStrcmp(root_type, VECHIP, &iship));

  /* For performance-portable types (Kokkos, SYCL, ...) that dispatch to */
  if (!(isstd || iscuda || iship)) {
    const PetscScalar *array;
    PetscMemType       memtype;

    PetscCall(VecGetArrayReadAndMemType(X, &array, &memtype));
    PetscCall(VecRestoreArrayReadAndMemType(X, &array));
    switch (memtype) {
    case PETSC_MEMTYPE_HOST:
      isstd = PETSC_TRUE;
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
      iscuda = PETSC_TRUE;
      break;
    case PETSC_MEMTYPE_HIP:
      iship = PETSC_TRUE;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)X), PETSC_ERR_SUP, "Cannot figure out memory type of vector type %s", root_type);
    }
  }

  if (isstd) {
    PetscCall(MatCreateDense(comm, m, n, M, N, data, A));
  }
#if defined(PETSC_HAVE_CUDA)
  else if (iscuda) {
    PetscCall(MatCreateDenseCUDA(comm, m, n, M, N, data, A));
  }
#endif
#if defined(PETSC_HAVE_HIP)
  else if (iship) {
    PetscCall(MatCreateDenseHIP(comm, m, n, M, N, data, A));
  }
#endif
  PetscCall(MatSetVecType(*A, root_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateDenseForVecType - Create a matrix that matches the type of a Vec.

  Collective

  Input Parameters:
+ comm  - the communicator
. vtype - the vector type
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n     - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M     - number of global rows (or `PETSC_DECIDE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DECIDE` to have calculated if `n` is given)
. lda   - optional leading dimension. Pass any non-positive number to use the default.
- data  - optional location of matrix data, which should have the same memory type as the vector. Pass `NULL` to have PETSc to matrix memory allocation.

  Output Parameter:
. A - the matrix.

  Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MatCreateDense()', `MatCreateDenseCUDA()`, `MatCreateDenseHIP()`, `PetscMemType`
@*/
PetscErrorCode MatCreateDenseForVecType(MPI_Comm comm, VecType vtype, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt lda, PetscScalar *data, Mat *A)
{
  VecType   root_type = VECSTANDARD;
  PetscBool isstd, iscuda, iship, iskokkos;

  PetscFunctionBegin;
  PetscCall(PetscStrcmpAny(vtype, &isstd, VECSTANDARD, VECMPI, VECSEQ, ""));
  PetscCall(PetscStrcmpAny(vtype, &iscuda, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));
  PetscCall(PetscStrcmpAny(vtype, &iship, VECHIP, VECMPIHIP, VECSEQHIP, ""));
  PetscCall(PetscStrcmpAny(vtype, &iskokkos, VECKOKKOS, VECMPIKOKKOS, VECSEQKOKKOS, ""));
  PetscCheck(isstd || iscuda || iship || iskokkos, comm, PETSC_ERR_SUP, "Not for type %s\n", vtype);
  if (iskokkos) {
    PetscCheck(!PetscDefined(HAVE_KOKKOS_ENABLE_SYCL), comm, PETSC_ERR_SUP, "Not for sycl backend");
    if (PetscDefined(HAVE_KOKKOS_ENABLE_CUDA)) iscuda = PETSC_TRUE;
    else if (PetscDefined(HAVE_KOKKOS_ENABLE_HIP)) iship = PETSC_TRUE;
    else isstd = PETSC_TRUE;
    root_type = VECKOKKOS;
  }
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  if (lda > 0) PetscCall(MatDenseSetLDA(*A, lda));
  if (isstd) {
    PetscCall(MatSetType(*A, MATDENSE));
    PetscCall(MatSeqDenseSetPreallocation(*A, data));
    PetscCall(MatMPIDenseSetPreallocation(*A, data));
  } else if (iscuda) {
    PetscCheck(PetscDefined(HAVE_CUDA), comm, PETSC_ERR_SUP, "PETSc not compiled with CUDA support");
#if defined(PETSC_HAVE_CUDA)
    PetscCall(MatSetType(*A, MATDENSECUDA));
    PetscCall(MatDenseCUDASetPreallocation(*A, data));
#endif
    root_type = VECCUDA;
  } else if (iship) {
    PetscCheck(PetscDefined(HAVE_HIP), comm, PETSC_ERR_SUP, "PETSc not compiled with HIP support");
#if defined(PETSC_HAVE_HIP)
    PetscCall(MatSetType(*A, MATDENSEHIP));
    PetscCall(MatDenseHIPSetPreallocation(*A, data));
#endif
    root_type = VECHIP;
  }
  PetscCall(MatSetVecType(*A, root_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}
