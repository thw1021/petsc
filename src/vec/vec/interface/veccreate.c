
#include <petsc/private/vecimpl.h>           /*I  "petscvec.h"   I*/

PetscErrorCode VecCreateAsync(MPI_Comm comm, PetscDeviceContext PETSC_UNUSED dctx, Vec *vec)
{
  PetscFunctionBegin;
  PetscValidPointer(vec,3);
  PetscCall(VecInitializePackage());
  *vec = NULL;

  PetscCall(PetscHeaderCreate(*vec,VEC_CLASSID,"Vec","Vector","Vec",comm,VecDestroy,VecView));
  PetscCall(PetscLayoutCreate(comm,&(*vec)->map));
  (*vec)->array_gotten = PETSC_FALSE;
  (*vec)->petscnative  = PETSC_FALSE;
  (*vec)->offloadmask  = PETSC_OFFLOAD_UNALLOCATED;
#if PetscDefined(HAVE_DEVICE)
  (*vec)->minimum_bytes_pinned_memory = 0;
  (*vec)->pinned_memory               = PETSC_FALSE;
  (*vec)->boundtocpu                  = PETSC_TRUE;
#endif
  PetscCall(PetscStrallocpy(PETSCRANDER48,&(*vec)->defaultrandtype));
  PetscFunctionReturn(0);
}

/*@
  VecCreate - Creates an empty vector object. The type can then be set with VecSetType(),
  or VecSetFromOptions().

   If you never  call VecSetType() or VecSetFromOptions() it will generate an
   error when you try to use the vector.

  Collective

  Input Parameter:
. comm - The communicator for the vector object

  Output Parameter:
. vec  - The vector object

  Level: beginner

.seealso: VecSetType(), VecSetSizes(), VecCreateMPIWithArray(), VecCreateMPI(), VecDuplicate(),
          VecDuplicateVecs(), VecCreateGhost(), VecCreateSeq(), VecPlaceArray()
@*/
PetscErrorCode  VecCreate(MPI_Comm comm, Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(VecCreateAsync(comm,NULL,vec));
  PetscFunctionReturn(0);
}
