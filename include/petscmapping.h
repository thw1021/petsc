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
PETSC_EXTERN PetscErrorCode PetscMappingRegister(const char[],PetscErrorCode(*)(PetscMapping));
PETSC_EXTERN PetscErrorCode PetscMappingRegiserAll(void);
PETSC_EXTERN PetscErrorCode PetscMappingCreate(MPI_Comm,PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingDestroy(PetscMapping*);
PETSC_EXTERN PetscErrorCode PetscMappingSetType(PetscMapping,PetscMappingType);
PETSC_EXTERN PetscErrorCode PetscMappingGetType(PetscMapping,PetscMappingType*);
PETSC_EXTERN PetscErrorCode PetscMappingView(PetscMapping,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscMappingSetUp(PetscMapping);
PETSC_EXTERN PetscErrorCode PetscMappingSetFromOptions(PetscMapping);

PETSC_EXTERN PetscErrorCode PetscMappingGetKeyState(PetscMapping,PetscMappingState*);

PETSC_EXTERN PetscErrorCode PetscMappingSetKeysContiguous(PetscMapping,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PetscMappingGetKeysContiguous(PetscMapping,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode PetscMappingSetKeysDiscontiguous(PetscMapping,PetscInt,const PetscInt[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode PetscMappingGetKeysDiscontiguous(PetscMapping,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode PetscMappingRestoreKeysDiscontiguous(PetscMapping,const PetscInt*[]);
#endif
