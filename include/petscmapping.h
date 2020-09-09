#if !defined(PETSCMAPPING_H)
#define PETSCMAPPING_H

#include <petscsys.h>

typedef struct _p_PetscMapping *PetscMapping;

PETSC_EXTERN PetscClassId PETSC_MAPPING_CLASSID;

PETSC_EXTERN PetscErrorCode PetscMappingInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscMappingFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscMappingCreate(MPI_Comm,PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingDestroy(PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingView(PetscMapping,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscMappingGetSize(PetscMapping,PetscInt*);

#endif
