#if !defined(PETSCIM_H)
#define PETSCIM_H

#include <petscsys.h>
#include <petscimtypes.h>

PETSC_EXTERN PetscClassId      IM_CLASSID;
PETSC_EXTERN PetscFunctionList IMList;

typedef const char* IMType;
#define IMTRIVIAL "trivial"

PETSC_EXTERN PetscErrorCode IMInitializePackage(void);
PETSC_EXTERN PetscErrorCode IMFinalizePackage(void);
PETSC_EXTERN PetscErrorCode IMRegister(const char[],PetscErrorCode(*)(IM));
PETSC_EXTERN PetscErrorCode IMRegiserAll(void);
PETSC_EXTERN PetscErrorCode IMCreate(MPI_Comm,IM*);
PETSC_EXTERN PetscErrorCode IMDestroy(IM*);
PETSC_EXTERN PetscErrorCode IMSetType(IM,IMType);
PETSC_EXTERN PetscErrorCode IMGetType(IM,IMType*);
PETSC_EXTERN PetscErrorCode IMView(IM,PetscViewer);
PETSC_EXTERN PetscErrorCode IMSetUp(IM);
PETSC_EXTERN PetscErrorCode IMSetFromOptions(IM);

PETSC_EXTERN PetscErrorCode IMGetKeyState(IM,IMState*);

PETSC_EXTERN PetscErrorCode IMSetKeysContiguous(IM,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode IMGetKeysContiguous(IM,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode IMSetKeysDiscontiguous(IM,PetscInt,const PetscInt[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode IMGetKeysDiscontiguous(IM,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode IMRestoreKeysDiscontiguous(IM,const PetscInt*[]);
#endif
