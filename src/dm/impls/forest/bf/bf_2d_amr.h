#if !defined(PETSCDMBF_2D_AMR_H)
  #define PETSCDMBF_2D_AMR_H

  #if defined(PETSCDMBF_3D_AMR_H)
    #error "The include files bf_2d_amr.h and bf_3d_amr.h cannot be combined"
  #endif

  #include <petscdmbf.h> /*I "petscdmbf.h" I*/
  #include "bf_xd.h"

PETSC_EXTERN PetscErrorCode DMBF_2D_AmrCoarsenUniformly(p4est_t *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_2D_AmrRefineUniformly(p4est_t *, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_2D_AmrAdapt(p4est_t *, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMBF_2D_AmrAdaptData(p4est_t *, p4est_t *, DM, DM_BF_AmrOps *);
PETSC_EXTERN PetscErrorCode DMBF_2D_AmrPartition(p4est_t *);

  /* rename generic functions that are the same for 2D and 3D */
  #define DMBF_XD_AmrCoarsenUniformly DMBF_2D_AmrCoarsenUniformly
  #define DMBF_XD_AmrRefineUniformly  DMBF_2D_AmrRefineUniformly
  #define DMBF_XD_AmrAdapt            DMBF_2D_AmrAdapt
  #define DMBF_XD_AmrAdaptData        DMBF_2D_AmrAdaptData
  #define DMBF_XD_AmrPartition        DMBF_2D_AmrPartition

#endif /* defined(PETSCDMBF_2D_AMR_H) */
