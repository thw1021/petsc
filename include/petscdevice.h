#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscdevicetypes.h>

/* Cannot use the device context api without C++ */
#if PetscDefined(HAVE_CXX)
PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);

/* PetscDevice */
PETSC_EXTERN PetscErrorCode PetscDeviceInitialize(PetscDeviceType);
PETSC_EXTERN PetscBool      PetscDeviceInitialized(PetscDeviceType);
PETSC_EXTERN PetscErrorCode PetscDeviceCreate(PetscDeviceType,PetscInt,PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceConfigure(PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceView(PetscDevice,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceGetType(PetscDevice,PetscDeviceType*);
PETSC_EXTERN PetscErrorCode PetscDeviceGetDeviceId(PetscDevice,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscDeviceDestroy(PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceGetDeviceId(PetscDevice,PetscInt*);

/* PetscDeviceContext */
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext,PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext,PetscDevice*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext,PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext,PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextFork(PetscDeviceContext,PetscInt,PetscDeviceContext**);
PETSC_EXTERN PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext,PetscInt,PetscDeviceContextJoinMode,PetscDeviceContext**);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext*);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm,const char[],PetscDeviceContext);

/* memory */
PETSC_EXTERN PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext,void*PETSC_RESTRICT,const void*PETSC_RESTRICT,size_t,PetscDeviceCopyMode);

#define PetscDeviceArrayCopy(dctx,dest,src,n,mode) ((!PetscDefined(HAVE_DEVICE) || (mode == PETSC_DEVICE_COPY_HTOH)) ? PetscArraycpy(dest,src,n) : PetscDeviceMemcpy(dctx,dest,src,(size_t)(n)*sizeof(*(src)),mode))
#endif /* PETSC_HAVE_CXX */

PETSC_EXTERN PetscErrorCode PetscGetMemType(const void*,PetscMemType*);
#endif /* PETSCDEVICE_H */
