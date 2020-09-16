#if !defined(PETSCMAPPING_H)
#define PETSCMAPPING_H

#include <petscsys.h>
#include <petscmappingtypes.h>

PETSC_EXTERN PetscClassId      PETSC_MAPPING_CLASSID;
PETSC_EXTERN PetscFunctionList PetscMappingList;

typedef const char* PetscMappingType;
#define PETSCMAPPINGTRIVIAL "trivial"

PETSC_EXTERN PetscErrorCode PetscMappingInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscMappingFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscMappingCreate(MPI_Comm,PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingDestroy(PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingRegister(const char[],PetscErrorCode(*)(PetscMapping));
PETSC_EXTERN PetscErrorCode PetscMappingSetType(PetscMapping,PetscMappingType);
PETSC_EXTERN PetscErrorCode PetscMappingGetType(PetscMapping,PetscMappingType*);
PETSC_EXTERN PetscErrorCode PetscMappingView(PetscMapping,PetscViewer);

#endif
