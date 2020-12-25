#if !defined(PETSCSTREAM_H)
#define PETSCSTREAM_H

#include <petscdevice.h>

typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING = -1,
  PETSC_STREAM_DEFAULT_BLOCKING = 0,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 1
} PetscStreamMode;

typedef struct _p_PetscStream* PetscStream;

PETSC_EXTERN PetscErrorCode PetscStreamCreate(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamDestroy(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamSetMode(PetscStream,PetscStreamMode);
PETSC_EXTERN PetscErrorCode PetscStreamGetMode(PetscStream,PetscStreamMode*);
PETSC_EXTERN PetscErrorCode PetscStreamSynchronize(PetscStream);
#endif
