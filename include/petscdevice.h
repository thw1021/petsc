#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscsys.h>

#if defined(PETSC_HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#define WaitForCUDA() PetscCUDASynchronize ? cudaDeviceSynchronize() : cudaSuccess;

/* CUDART_VERSION = 1000 x major + 10 x minor version */

/* Could not find exactly which CUDART_VERSION introduced cudaGetErrorName. At least it was in CUDA 8.0 (Sep. 2016) */
#if (CUDART_VERSION >= 8000) /* CUDA 8.0 */
#define CHKERRCUDA(cerr) \
do { \
   if (PetscUnlikely(cerr)) { \
      const char *name  = cudaGetErrorName(cerr); \
      const char *descr = cudaGetErrorString(cerr); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s",(int)cerr,name,descr); \
   } \
} while (0)
#else
#define CHKERRCUDA(cerr) do {if (PetscUnlikely(cerr)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d",(int)cerr);} while (0)
#endif /* CUDART_VERSION >= 8000 */

#define CHKERRCUBLAS(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name = PetscCUBLASGetErrorName(stat); \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuBLAS error %d (%s)",(int)stat,name); \
   } \
} while (0)

#endif /* PETSC_HAVE_CUDA */

#if defined(PETSC_HAVE_HIP)
#include <hip/hip_runtime.h>

#define WaitForHIP() PetscHIPSynchronize ? hipDeviceSynchronize() : hipSuccess;

#define CHKERRHIP(cerr) \
do { \
   if (PetscUnlikely(cerr)) { \
      const char *name  = hipGetErrorName(cerr); \
      const char *descr = hipGetErrorString(cerr); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"hip error %d (%s) : %s",(int)cerr,name,descr); \
   } \
} while (0)

#endif /* PETSC_HAVE_HIP */

typedef struct _n_PetscEvent* PetscEvent;

PETSC_EXTERN PetscErrorCode PetscEventCreate(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventDestroy(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventSetFlags(PetscEvent,unsigned int,unsigned int);
PETSC_EXTERN PetscErrorCode PetscEventGetFlags(PetscEvent,unsigned int*,unsigned int*);
PETSC_EXTERN PetscErrorCode PetscEventSetup(PetscEvent);

typedef enum {
  PETSC_STREAM_CUDA,
  PETSC_STREAM_HIP
} PetscStreamType;

typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING = 0,
  PETSC_STREAM_DEFAULT_BLOCKING = 1,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 2
} PetscStreamMode;

typedef struct _n_PetscStream* PetscStream;

PETSC_EXTERN PetscErrorCode PetscStreamCreate(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamDestroy(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamSetMode(PetscStream,PetscStreamMode);
PETSC_EXTERN PetscErrorCode PetscStreamGetMode(PetscStream,PetscStreamMode*);
PETSC_EXTERN PetscErrorCode PetscStreamGetStream(PetscStream,PetscStreamType,void*);
PETSC_EXTERN PetscErrorCode PetscStreamRestoreStream(PetscStream,PetscStreamType,void*,PetscBool);
PETSC_EXTERN PetscErrorCode PetscStreamSplitBegin(PetscStream,PetscStreamType,void*);
PETSC_EXTERN PetscErrorCode PetscStreamSplitEnd(PetscStream,PetscStreamType,void*,PetscBool);
PETSC_EXTERN PetscErrorCode PetscStreamRecordEvent(PetscStream,PetscStreamType,PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamWaitEvent(PetscStream,PetscEvent,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamSynchronize(PetscStream,PetscStreamType);

#endif /* PETSCDEVICE_H */
