#if !defined(PETSCIM_H)
#define PETSCIM_H

#include <petscviewer.h>
#include <petscimtypes.h>

PETSC_EXTERN PetscClassId      IM_CLASSID;
PETSC_EXTERN PetscFunctionList IMList;

typedef const char* IMType;
#define IMLAYOUT "layout"
#define IMMAP    "map"
#define IMBASIC  "basic"

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

PETSC_EXTERN PetscErrorCode IMSetIndices(IM,PetscInt,const PetscInt*[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode IMGetIndices(IM,const PetscInt*[]);
PETSC_EXTERN PetscErrorCode IMGetSizes(IM,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode IMGetLayout(IM,IM*);
PETSC_EXTERN PetscErrorCode IMRestoreLayout(IM,IM*);
PETSC_EXTERN PetscErrorCode IMRestoreIndices(IM,const PetscInt*[]);

/* IMLAYOUT */
PETSC_EXTERN PetscErrorCode IMLayoutCreate(MPI_Comm,IM*);
PETSC_EXTERN PetscErrorCode IMLayoutSetFromMapping(IM,PetscInt,const PetscInt*[],PetscCopyMode);
PETSC_EXTERN PetscErrorCode IMLayoutSetFromSizes(IM,PetscInt,PetscInt);
#endif /* PETSCIM_H */
