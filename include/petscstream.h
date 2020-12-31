#if !defined(PETSCSTREAM_H)
#define PETSCSTREAM_H

#include <petscdevice.h>

typedef enum {
  PETSC_STREAM_CUDA,
  PETSC_STREAM_HIP
} PetscStreamType;

typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING = -1,
  PETSC_STREAM_DEFAULT_BLOCKING = 0,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 1
} PetscStreamMode;

typedef struct _n_PetscStream* PetscStream;

PETSC_EXTERN PetscErrorCode PetscStreamCreate(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamDestroy(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamSetup(PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamSetMode(PetscStream,PetscStreamMode);
PETSC_EXTERN PetscErrorCode PetscStreamGetMode(PetscStream,PetscStreamMode*);
PETSC_EXTERN PetscErrorCode PetscStreamGetStream(PetscStream,PetscStreamType,void*);
PETSC_EXTERN PetscErrorCode PetscStreamRestoreStream(PetscStream,PetscStreamType,void*);
PETSC_EXTERN PetscErrorCode PetscStreamSynchronize(PetscStream);
#endif
