
#include <petsc/private/vecimpl.h>           /*I  "petscvec.h"   I*/

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
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(vec,2);
  *vec = NULL;
  ierr = VecInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(v, VEC_CLASSID, "Vec", "Vector", "Vec", comm, VecDestroy, VecView);CHKERRQ(ierr);

  ierr            = PetscLayoutCreate(comm,&v->map);CHKERRQ(ierr);
  v->array_gotten = PETSC_FALSE;
  v->petscnative  = PETSC_FALSE;
  v->offloadmask  = PETSC_OFFLOAD_UNALLOCATED;
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  v->minimum_bytes_pinned_memory = 0;
  v->pinned_memory = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_DEVICE)
  v->boundtocpu = PETSC_TRUE;
#endif
  ierr = PetscStrallocpy(PETSCRANDER48,&v->defaultrandtype);CHKERRQ(ierr);
  *vec = v;
  PetscFunctionReturn(0);
}

/*@
   VecSetGhost - sets that a vector will be ghosted vector

   Collective

   Input Parameters:
+  vec - the vectore
.  nghost - number of local ghost points
.  ghosts - global indices of ghost points, these do not need to be in increasing order (sorted)
-  nextra - extra locations at the end of the local vector

   Notes:
     Do not free the ghosts array until after the vector type has been set and the vector fully created

     Use VecGhostGetLocalForm() to access the local, ghosted representation
     of the vector.

     This also automatically sets the ISLocalToGlobalMapping() for this vector.

   Level: advanced

.seealso: VecCreateSeq(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateMPI(),
          VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecCreateMPIWithArray(), VecGhostUpdateEnd(),
          VecCreateGhostBlock(), VecCreateGhostBlockWithArray(), VecSetGhost()

@*/
PetscErrorCode  VecSetGhost(Vec vec,PetscInt nghost,const PetscInt ghosts[],PetscInt nextra)
{
  PetscFunctionBegin;
  if (((PetscObject)vec)->type_name) SETERRQ(PetscObjectComm((PetscObject)vec),PETSC_ERR_ARG_WRONGSTATE,"Vector type has already been set");
  vec->nghost  = nghost;
  vec->ghosts  = (PetscInt*) ghosts;
  vec->nextra  = nextra;
  vec->isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}
