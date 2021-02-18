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

/*E
  PetscMemType - Memory type of a pointer

  Level: beginner

  Developer Note:
   Encoding of the bitmask in binary: xx0=HOST, xx1=DEVICE, x01 for CUDA, x11 for HIP.

.seealso: VecGetArrayAndMemType(), PetscSFBcastAndOpWithMemTypeBegin(), PetscSFReduceWithMemTypeBegin()
E*/
typedef enum {PETSC_MEMTYPE_HOST=0, PETSC_MEMTYPE_DEVICE=1, PETSC_MEMTYPE_CUDA=1, PETSC_MEMTYPE_HIP=3} PetscMemType;

#define PetscMemTypeHost(m)   (((m) & 0x1) == PETSC_MEMTYPE_HOST)
#define PetscMemTypeDevice(m) (((m) & 0x1) == PETSC_MEMTYPE_DEVICE)

typedef struct _n_PetscEvent* PetscEvent;

typedef enum {
  PETSC_STREAM_INVALID = 0,
  PETSC_STREAM_CUDA = 1,
  PETSC_STREAM_HIP = 2
} PetscStreamType;

PETSC_EXTERN PetscErrorCode PetscEventCreate(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventDestroy(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventSetType(PetscEvent,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscEventGetType(PetscEvent,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscEventSetFlags(PetscEvent,unsigned int,unsigned int);
PETSC_EXTERN PetscErrorCode PetscEventGetFlags(PetscEvent,unsigned int*,unsigned int*);
PETSC_EXTERN PetscErrorCode PetscEventSetUp(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventSynchronize(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventQuery(PetscEvent,PetscBool*);

typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING = 0,
  PETSC_STREAM_DEFAULT_BLOCKING = 1,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 2
} PetscStreamMode;

typedef struct _n_PetscStream* PetscStream;

PETSC_EXTERN PetscErrorCode PetscStreamCreate(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamDestroy(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamSetType(PetscStream,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamGetType(PetscStream,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscStreamSetMode(PetscStream,PetscStreamMode);
PETSC_EXTERN PetscErrorCode PetscStreamGetMode(PetscStream,PetscStreamMode*);
PETSC_EXTERN PetscErrorCode PetscStreamSetUp(PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamGetStream(PetscStream,void*);
PETSC_EXTERN PetscErrorCode PetscStreamRestoreStream(PetscStream,void*);
PETSC_EXTERN PetscErrorCode PetscStreamRecordEvent(PetscStream,PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamWaitEvent(PetscStream,PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamSynchronize(PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamQuery(PetscStream,PetscBool*);

typedef enum {
  STREAM_OP_SUM,
  STREAM_OP_SUB,
  STREAM_OP_DIV,
  STREAM_OP_MULT,
  STREAM_OP_EQUAL
} PetscStreamComputeOp;

typedef struct _n_PetscStreamScalar* PetscStreamScalar;

PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate(PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarDestroy(PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetType(PetscStreamScalar,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetType(PetscStreamScalar,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetUp(PetscStreamScalar,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetValue(PetscStreamScalar,const PetscScalar*,PetscMemType,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetHostRead(PetscStreamScalar,const PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetHostWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRestoreHostWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar,const PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetDeviceWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRestoreDeviceWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAccumulateOp(PetscStreamScalar,PetscInt,PetscStreamScalar[],PetscStreamComputeOp,PetscStreamComputeOp,PetscStream);

/*E
    PetscOffloadMask - indicates which memory (CPU, GPU, or none) contains valid data

   PETSC_OFFLOAD_UNALLOCATED  - no memory contains valid matrix entries; NEVER used for vectors
   PETSC_OFFLOAD_GPU - GPU has valid vector/matrix entries
   PETSC_OFFLOAD_CPU - CPU has valid vector/matrix entries
   PETSC_OFFLOAD_BOTH - Both GPU and CPU have valid vector/matrix entries and they match
   PETSC_OFFLOAD_VECKOKKOS - Reserved for Vec_Kokkos. The offload is managed by Kokkos, thus this flag is not used in Vec_Kokkos.

   Level: developer
E*/
typedef enum {PETSC_OFFLOAD_UNALLOCATED=0x0,PETSC_OFFLOAD_CPU=0x1,PETSC_OFFLOAD_GPU=0x2,PETSC_OFFLOAD_BOTH=0x3,PETSC_OFFLOAD_VECKOKKOS=0x100} PetscOffloadMask;
#endif /* PETSCDEVICE_H */
