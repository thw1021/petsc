#ifndef MMGCOMMON_H
#define MMGCOMMON_H

#include <petsc/private/dmpleximpl.h>

#define CHKERRMMG(func,...) do {                        \
    PetscStackPush(PetscStringize(func));               \
    PetscErrorCode mmg_ierr_ = func(__VA_ARGS__);       \
    PetscStackPop;                                      \
    PetscCheck(mmg_ierr_ == 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in %s(): error code %d",PetscStringize(func),mmg_ierr_); \
} while (0)

#endif // MMGCOMMON_H
