#ifndef TESTHEADER_H
#define TESTHEADER_H

#include <petscsystypes.h>

PetscErrorCode testExplicitSynopsis(PetscInt, PetscReal, void *);

extern void ExternHeaderFunctionShouldNotGetStatic(void);

// clang-format off
PETSC_EXTERN       void         testBadFormatting                                (    void    )   ;
// clang-format on

#endif // TESTHEADER_H
