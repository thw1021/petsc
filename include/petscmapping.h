#if !defined(PETSCMAPPING_H)
#define PETSCMAPPING_H

#include <petscsys.h>

typedef struct _p_PetscMapping *PetscMapping;

PETSC_EXTERN PetscClassId PETSC_MAPPING_CLASSID;

typedef enum {
  NONE_VALID = -1,
  KEY_VALID = 0,
  MAPS_VALID,
  ALL_VALID,
  INDICES_VALID
} PetscMappingState;

PETSC_EXTERN PetscErrorCode PetscMappingInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscMappingFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscMappingCreate(MPI_Comm,PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingDestroy(PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingView(PetscMapping,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscMappingGetSize(PetscMapping,PetscInt*,PetscInt*);

#endif
