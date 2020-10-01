#if !defined(PETSCIM_H)
#define PETSCIM_H

#include <petscviewer.h>
#include <petscimtypes.h>

PETSC_EXTERN PetscClassId      IM_CLASSID;
PETSC_EXTERN PetscFunctionList IMList;

typedef const char* IMType;
#define IMMAP   "map"
#define IMBASIC "basic"

PETSC_EXTERN const char* const IMStates[];

/* GENERAL */
PETSC_EXTERN PetscErrorCode IMInitializePackage(void);
PETSC_EXTERN PetscErrorCode IMFinalizePackage(void);
PETSC_EXTERN PetscErrorCode IMRegister(const char[],PetscErrorCode(*)(IM));

PETSC_EXTERN PetscErrorCode IMCreate(MPI_Comm,IM*);
PETSC_EXTERN PetscErrorCode IMDestroy(IM*);
PETSC_EXTERN PetscErrorCode IMSetType(IM,IMType);
PETSC_EXTERN PetscErrorCode IMGetType(IM,IMType*);
PETSC_EXTERN PetscErrorCode IMView(IM,PetscViewer);
PETSC_EXTERN PetscErrorCode IMViewFromOptions(IM,PetscObject,const char[]);
PETSC_EXTERN PetscErrorCode IMSetUp(IM);
PETSC_EXTERN PetscErrorCode IMSetFromOptions(IM);

PETSC_EXTERN PetscErrorCode IMSetKeyStateAndSizes(IM,IMState,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode IMGetKeyState(IM,IMState*);
PETSC_EXTERN PetscErrorCode IMGetNumKeys(IM,IMOpMode,PetscInt*);

PETSC_EXTERN PetscErrorCode IMContiguousSetKeyInterval(IM,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode IMContiguousGetKeyInterval(IM,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode IMArraySetKeyArray(IM,PetscInt,const PetscInt[],PetscBool,PetscCopyMode);
PETSC_EXTERN PetscErrorCode IMArrayGetKeyArray(IM,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode IMArrayRestoreKeyArray(IM,const PetscInt*[]);

/* IM_BASIC */
PETSC_EXTERN PetscErrorCode IMBasicCreateFromSizes(MPI_Comm,IMState,PetscInt,PetscInt,IM*);
#endif
