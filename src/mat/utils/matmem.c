#include <petsc/private/matimpl.h>

PETSC_INTERN PetscErrorCode MatGetCurrentMemType(Mat A, PetscMemType *m)
{
  PetscBool bound, ishypre = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(m, 2);
  *m = PETSC_MEMTYPE_HOST;
  PetscCall(MatBoundToCPU(A, &bound));
  if (!bound) {
    MatType rtype;
    char   *iscuda = NULL, *iship = NULL, *iskok = NULL;

    PetscCall(MatGetRootType_Private(A, &rtype));
    PetscCall(PetscStrstr(rtype, "cusparse", &iscuda));
    if (!iscuda) PetscCall(PetscStrstr(rtype, "cuda", &iscuda));
    PetscCall(PetscStrstr(rtype, "hip", &iship));
    PetscCall(PetscStrstr(rtype, "kokkos", &iskok));
    if (iscuda) *m = PETSC_MEMTYPE_CUDA;
    else if (iship) *m = PETSC_MEMTYPE_HIP;
    else if (iskok) *m = PETSC_MEMTYPE_KOKKOS;
    else {
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHYPRE, &ishypre));
      /* If it's not bound to the CPU, then we default it to the device as hypre would */
      if (ishypre) *m = PETSC_MEMTYPE_DEVICE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
