#if !defined(PETSCSYSSAWS_H)
  #define PETSCSYSSAWS_H

  #include <petscsystypes.h>

  #if defined(PETSC_HAVE_SAWS)
PETSC_EXTERN PetscErrorCode PetscSAWsBlock(void);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsViewOff(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsSetBlock(PetscObject, PetscBool);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsBlock(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsGrantAccess(PetscObject);
PETSC_EXTERN PetscErrorCode PetscObjectSAWsTakeAccess(PetscObject);
PETSC_EXTERN void           PetscStackSAWsGrantAccess(void);
PETSC_EXTERN void           PetscStackSAWsTakeAccess(void);
PETSC_EXTERN PetscErrorCode PetscStackViewSAWs(void);
PETSC_EXTERN PetscErrorCode PetscStackSAWsViewOff(void);

  #else
    #define PetscSAWsBlock()                  PETSC_SUCCESS
    #define PetscObjectSAWsViewOff(obj)       PETSC_SUCCESS
    #define PetscObjectSAWsSetBlock(obj, flg) PETSC_SUCCESS
    #define PetscObjectSAWsBlock(obj)         PETSC_SUCCESS
    #define PetscObjectSAWsGrantAccess(obj)   PETSC_SUCCESS
    #define PetscObjectSAWsTakeAccess(obj)    PETSC_SUCCESS
    #define PetscStackViewSAWs()              PETSC_SUCCESS
    #define PetscStackSAWsViewOff()           PETSC_SUCCESS
    #define PetscStackSAWsTakeAccess()
    #define PetscStackSAWsGrantAccess()

  #endif

#endif // #define PETSCSYSSAWS_H
