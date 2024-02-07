#if !defined(PETSCDMBF_3D_AMR_H)
  #define PETSCDMBF_3D_AMR_H

  #if defined(PETSCDMBF_2D_AMR_H)
    #error "The include files bf_2d_amr.h and bf_3d_amr.h cannot be combined"
  #endif

  #include <petscdmbf.h> /*I "petscdmbf.h" I*/

  /* convert to p8est for 3D domains */
  #if defined(PETSC_HAVE_MPIUNI)
    #undef MPI_SUCCESS
  #endif
  #if defined(PETSC_HAVE_P4EST)
    #include <p4est_to_p8est.h>
  #endif
  #if defined(PETSC_HAVE_MPIUNI)
    #define MPI_SUCCESS 0
  #endif

  #include "bf_xd.h"

PETSC_EXTERN PetscErrorCode DMBF_3D_AmrCoarsenUniformly(p4est_t *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_3D_AmrRefineUniformly(p4est_t *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_3D_AmrAdapt(p4est_t *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_3D_AmrAdaptData(p4est_t *, p4est_t *, DM, DM_BF_AmrOps *);
PETSC_EXTERN PetscErrorCode DMBF_3D_AmrPartition(p4est_t *);

  /* rename generic functions that are the same for 2D and 3D */
  #define DMBF_XD_AmrCoarsenUniformly DMBF_3D_AmrCoarsenUniformly
  #define DMBF_XD_AmrRefineUniformly  DMBF_3D_AmrRefineUniformly
  #define DMBF_XD_AmrAdapt            DMBF_3D_AmrAdapt
  #define DMBF_XD_AmrAdaptData        DMBF_3D_AmrAdaptData
  #define DMBF_XD_AmrPartition        DMBF_3D_AmrPartition

#endif /* defined(PETSCDMBF_3D_AMR_H) */
