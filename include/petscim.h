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

PETSC_EXTERN PetscErrorCode IMGenerateDefault(MPI_Comm,IMType,IMState,PetscInt,PetscInt,IM*);
PETSC_EXTERN PetscErrorCode IMGetKeyState(IM,IMState*);
PETSC_EXTERN PetscErrorCode IMGetNumKeys(IM,IMOpMode,PetscInt*);

PETSC_EXTERN PetscErrorCode IMSetKeyInterval(IM,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode IMGetKeyInterval(IM,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode IMSetKeyArray(IM,PetscInt,const PetscInt[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode IMGetKeyArray(IM,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode IMRestoreKeyArray(IM,const PetscInt*[]);

PETSC_EXTERN PetscErrorCode IMConvertKeyState(IM,IMState);
PETSC_EXTERN PetscErrorCode IMSort(IM,IMOpMode);
PETSC_EXTERN PetscErrorCode IMPermute(IM,IM);
PETSC_EXTERN PetscErrorCode IMSetSorted(IM,IMOpMode,PetscBool);
PETSC_EXTERN PetscErrorCode IMSorted(IM,IMOpMode,PetscBool*);

/* IM_BASIC */
PETSC_EXTERN PetscErrorCode IMBasicCreateFromSizes(MPI_Comm,IMState,PetscInt,PetscInt,IM*);
PETSC_EXTERN PetscErrorCode IMBasicCreateFromRanges(MPI_Comm,IMState,const PetscInt[],PetscCopyMode,IM*);
#endif /* PETSCIM_H */
