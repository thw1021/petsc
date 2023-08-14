#ifndef TESTHEADER_H
#define TESTHEADER_H

#include <petscsystypes.h>

PetscErrorCode testExplicitSynopsis(PetscInt, PetscReal, void *);

extern void ExternHeaderFunctionShouldNotGetStatic(void);

#endif // TESTHEADER_H
