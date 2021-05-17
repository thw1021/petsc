#if !defined(PETSCDEVICE_H)
#define PETSCDEVICE_H

#include <petscsys.h>

#if PetscDefined(HAVE_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

PETSC_EXTERN cudaEvent_t petsc_gputimer_begin;
PETSC_EXTERN cudaEvent_t petsc_gputimer_end;

/* cuBLAS does not have cublasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscCUBLASGetErrorName(cublasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRCUBLAS macro */

#define WaitForCUDA() PetscCUDASynchronize ? cudaDeviceSynchronize() : cudaSuccess;

/* CUDART_VERSION = 1000 x major + 10 x minor version */

/* Could not find exactly which CUDART_VERSION introduced cudaGetErrorName. At least it was in CUDA 8.0 (Sep. 2016) */
#if (CUDART_VERSION >= 8000) /* CUDA 8.0 */
#define CHKERRCUDA(cerr)                                                \
  do {                                                                  \
    if (PetscUnlikely(cerr)) {                                          \
      const char *name  = cudaGetErrorName(cerr);                       \
      const char *descr = cudaGetErrorString(cerr);                     \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s",(int)cerr,name,descr); \
    }                                                                   \
  } while (0)
#else
#define CHKERRCUDA(cerr) do {if (PetscUnlikely(cerr)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d",(int)cerr);} while (0)
#endif /* CUDART_VERSION >= 8000 */

#define CHKERRCUBLAS(stat)                                              \
  do {                                                                  \
    if (PetscUnlikely(stat)) {                                          \
      const char *name = PetscCUBLASGetErrorName(stat);                 \
      if (((stat == CUBLAS_STATUS_NOT_INITIALIZED) || (stat == CUBLAS_STATUS_ALLOC_FAILED)) && PetscCUDAInitialized) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"cuBLAS error %d (%s). Reports not initialized or alloc failed; this indicates the GPU has run out resources",(int)stat,name); \
      else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuBLAS error %d (%s)",(int)stat,name); \
    }                                                                   \
  } while (0)

PETSC_EXTERN cudaStream_t   PetscDefaultCudaStream; /* The default stream used by PETSc */
PETSC_INTERN PetscErrorCode PetscCUBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscCUSOLVERDnInitializeHandle(void);

PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t*);
#endif /* PETSC_HAVE_CUDA */

#if PetscDefined(HAVE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas.h>
#if defined(__HIP_PLATFORM_NVCC__)
#include <cusolverDn.h>
#else /* __HIP_PLATFORM_NVCC__ */
#include <rocsolver.h>
#endif /* __HIP_PLATFORM_NVCC__ */

#define WaitForHIP() PetscHIPSynchronize ? hipDeviceSynchronize() : hipSuccess;

PETSC_EXTERN hipEvent_t petsc_gputimer_begin;
PETSC_EXTERN hipEvent_t petsc_gputimer_end;

/* hipBLAS does not have hipblasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscHIPBLASGetErrorName(hipblasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRHIPBLAS macro */

#define CHKERRHIP(cerr) \
do { \
   if (PetscUnlikely(cerr)) { \
      const char *name  = hipGetErrorName(cerr); \
      const char *descr = hipGetErrorString(cerr); \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_LIB,"hip error %d (%s) : %s",(int)cerr,name,descr); \
   } \
} while (0)

#define CHKERRHIPBLAS(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name = PetscHIPBLASGetErrorName(stat); \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"hipBLAS error %d (%s)",(int)stat,name); \
   } \
} while (0)

/* hipSolver does not exist yet so we work around it
   rocSOLVER users rocBLAS for the handle
 * */
#if defined(__HIP_PLATFORM_NVCC__)
typedef cusolverDnHandle_t hipsolverHandle_t;
typedef cusolverStatus_t hipsolverStatus_t;

/* Alias hipsolverDestroy to cusolverDnDestroy*/
PETSC_STATIC_INLINE cusolverStatus_t hipsolverDestroy(hipsolverHandle_t *hipsolverhandle)
{
  return cusolverDnDestroy(hipsolverhandle)
}

/* Alias hipsolverCreate to cusolverDnCreate*/
PETSC_STATIC_INLINE cusolverStatus_t hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return cusolverDnCreate(hipsolverhandle)
}
#else /* __HIP_PLATFORM_NVCC__ */
typedef rocblas_handle hipsolverHandle_t;
typedef rocblas_status hipsolverStatus_t;

/* Alias hipsolverDestroy to rocblas_destroy_handle*/
PETSC_STATIC_INLINE rocblas_status hipsolverDestroy(rocblas_handle hipsolverhandle)
{
  return rocblas_destroy_handle(hipsolverhandle);
}

/* Alias hipsolverCreate to rocblas_destroy_handle*/
PETSC_STATIC_INLINE rocblas_status hipsolverCreate(hipsolverHandle_t *hipsolverhandle)
{
  return rocblas_create_handle(hipsolverhandle);
}
#endif /* __HIP_PLATFORM_NVCC__ */

PETSC_EXTERN hipStream_t    PetscDefaultHipStream; /* The default stream used by PETSc */
PETSC_INTERN PetscErrorCode PetscHIPBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscHIPSOLVERInitializeHandle(void);


PETSC_EXTERN PetscErrorCode PetscHIPBLASGetHandle(hipblasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscHIPSOLVERGetHandle(hipsolverHandle_t*);
#endif /* PETSC_HAVE_HIP */

/*E
  PetscMemType - Memory type of a pointer

  Level: beginner

  Developer Note:
   Encoding of the bitmask in binary: xxxxyyyz
   z = 0:                Host memory
   z = 1:                Device memory
   yyy = 000:            CUDA-related memory
   yyy = 001:            HIP-related memory
   xxxxyyy1 = 0000,0001: CUDA memory
   xxxxyyy1 = 0001,0001: CUDA NVSHMEM memory
   xxxxyyy1 = 0000,0011: HIP memory

  Other types of memory, e.g., CUDA managed memory, can be added when needed.

.seealso: VecGetArrayAndMemType(), PetscSFBcastWithMemTypeBegin(), PetscSFReduceWithMemTypeBegin()
E*/
typedef enum {PETSC_MEMTYPE_HOST=0, PETSC_MEMTYPE_DEVICE=0x01, PETSC_MEMTYPE_CUDA=0x01, PETSC_MEMTYPE_NVSHMEM=0x11,PETSC_MEMTYPE_HIP=0x03} PetscMemType;

#define PetscMemTypeHost(m)    (((m) & 0x1) == PETSC_MEMTYPE_HOST)
#define PetscMemTypeDevice(m)  (((m) & 0x1) == PETSC_MEMTYPE_DEVICE)
#define PetscMemTypeCUDA(m)    (((m) & 0xF) == PETSC_MEMTYPE_CUDA)
#define PetscMemTypeHIP(m)     (((m) & 0xF) == PETSC_MEMTYPE_HIP)
#define PetscMemTypeNVSHMEM(m) ((m) == PETSC_MEMTYPE_NVSHMEM)

/*J
  PetscStreamType - Stream type

  Level: beginner

  Developer Notes:
  Any changes here must also be made in src/sys/f90-mod/petscsys.h

.seealso: PetscStreamSetType(), PetscEventSetType(), PetscStreamScalarSetType()
J*/
typedef const char* PetscStreamType;
#define PETSCSTREAMCUDA "cuda"
#define PETSCSTREAMHIP  "hip"

/*S
  PetscEvent - Container for efficient management of device stream events.

  As opposed to MPI streams are entirely decentralized objects, meaning that there exists no "super context" or manager
  which might facilitate synchronization or communication between distinct streams (such as an MPI communicator). Any
  coordination between streams is instead done via events. For two streams to interact, the first stream must record an
  event which the other must wait on.

  Level: beginner

.seealso: PetscEventCreate(), PetscStreamType, PetscEventSetType(), PetscEventDestroy(), PetscStreamRecordEvent(), PetscStreamWaitEvent()
S*/
typedef struct _n_PetscEvent* PetscEvent;

PETSC_EXTERN PetscErrorCode PetscEventInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscEventRegister(const char[],PetscErrorCode(*)(PetscEvent));
PETSC_EXTERN PetscErrorCode PetscEventCreate(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventDestroy(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventSetType(PetscEvent,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscEventGetType(PetscEvent,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscEventSetFlags(PetscEvent,unsigned int,unsigned int);
PETSC_EXTERN PetscErrorCode PetscEventGetFlags(PetscEvent,unsigned int*,unsigned int*);
PETSC_EXTERN PetscErrorCode PetscEventSetUp(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventSetFromOptions(MPI_Comm,const char[],PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventSynchronize(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventQuery(PetscEvent,PetscBool*);

/*E
  PetscStreamMode - Stream blocking mode, indicates how a strea implementation will interact with the default "NULL"
  stream, which is usually blocking.

$ PETSC_STREAM_GLOBAL_BLOCKING - Alias for NULL stream. Any stream of this type will block the hostfor all other streams to finish work before starting its operations.
$ PETSC_STREAM_DEFAULT_BLOCKING - Stream will act independent of other streams, but will still be blocked by actions on the NULL stream.
$ PETSC_STREAM_GLOBAL_NONBLOCKING - Stream is truly asynchronous, and is blocked by nothing, not even the NULL stream.
$ PETSC_STREAM_MAX_MODE - Always 1 greater than the largest PetscStreamMode

  Level: intermediate

  Developer Notes:
  Any changes here must also be made in src/sys/f90-mod/petscsys.h

.seealso: PetscStreamSetMode(), PetscStreamGetMode()
E*/
typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING = 0,
  PETSC_STREAM_DEFAULT_BLOCKING = 1,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 2,
  PETSC_STREAM_MAX_MODE = 3
} PetscStreamMode;
PETSC_EXTERN const char *const PetscStreamModes[];

/*S
  PetscStream - Container for efficient management of a device stream.

  level: beginner

.seealso: PetscStreamCreate(), PetscStreamType, PetscStreamSetType(), PetscStreamDestroy()
S*/
typedef struct _n_PetscStream* PetscStream;

PETSC_EXTERN PetscErrorCode PetscStreamInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscStreamRegister(const char[],PetscErrorCode(*)(PetscStream));
PETSC_EXTERN PetscErrorCode PetscStreamCreate(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamDestroy(PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamSetType(PetscStream,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamGetType(PetscStream,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscStreamSetMode(PetscStream,PetscStreamMode);
PETSC_EXTERN PetscErrorCode PetscStreamGetMode(PetscStream,PetscStreamMode*);
PETSC_EXTERN PetscErrorCode PetscStreamSetUp(PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamSetFromOptions(MPI_Comm,const char[],PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamDuplicate(PetscStream,PetscStream*);
PETSC_EXTERN PetscErrorCode PetscStreamGetStream(PetscStream,void*);
PETSC_EXTERN PetscErrorCode PetscStreamRestoreStream(PetscStream,void*);
PETSC_EXTERN PetscErrorCode PetscStreamRecordEvent(PetscStream,PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamWaitEvent(PetscStream,PetscEvent);
PETSC_EXTERN PetscErrorCode PetscStreamSynchronize(PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamQuery(PetscStream,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscStreamWaitForStream(PetscStream,PetscStream);

/*E
  PSSCacheType - PetscStreamScalar cache identifier

$ PSS_ZERO - Is the PetscStreamScalar = 0
$ PSS_ONE  - Is the PetscStreamScalar = 1
$ PSS_INF  - Is the PetscStreamScalar = inf (such that PetscIsInfScalar() returns PETSC_TRUE)
$ PSS_NAN  - Is the PetscStreamScalar = nan (such that PetscIsNanScalar() returns PETSC_TRUE)
$ PSSCACHE_MAX - Always the maximum cache value

  Level: intermediate

  Developer Notes:
  Any changes here must also be made in src/sys/f90-mod/petscsys.h

.seealso: PetscStreamScalarGetInfo(), PetscStreamScalarSetInfo()
E*/
typedef enum {
  PSS_ZERO = 0,
  PSS_ONE = 1,
  PSS_INF = 2,
  PSS_NAN = 3,
  PSS_CACHE_MAX = 4
} PSSCacheType;
PETSC_EXTERN const char *const PSSCacheTypes[];

/*S
  PetscStreamScalar - A stream-aware container for a PetscScalar.

  This object allows for pipelining of scalar inputs or results between asynchronous functions. In addition, some basic
  arithmetic operations are also exposed in order to ennsure the PetscScalar need never be transfered to the host,
  however this is unavoidable for more complex interactions.

  level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamType, PetscStreamScalarSetType(), PetscStreamScalarDestroy()
S*/
typedef struct _n_PetscStreamScalar* PetscStreamScalar;

PETSC_EXTERN PetscErrorCode PetscStreamScalarInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRegister(const char[],PetscErrorCode(*)(PetscStreamScalar));
PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate(PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarDestroy(PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetType(PetscStreamScalar,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetType(PetscStreamScalar,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetUp(PetscStreamScalar);
PETSC_EXTERN PetscErrorCode PetscStreamScalarDuplicate(PetscStreamScalar,PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetValue(PetscStreamScalar,const PetscScalar*,PetscMemType,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAwait(PetscStreamScalar,PetscScalar*,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar,const PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetDeviceWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRestoreDeviceWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRestoreDeviceRead(PetscStreamScalar,const PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetInfo(PetscStreamScalar,PSSCacheType,PetscBool,PetscBool*,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetInfo(PetscStreamScalar,PSSCacheType,PetscBool);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRealPart(PetscStreamScalar,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAXTY(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAYDX(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);

/*E
  PetscGraphAssemblyType - Indicates if a (possibly) existing graph should be updated, or instantiated anew.

$ PETSC_GRAPH_INIT_ASSEMBLY - Instantiate a new executable graph from a captured graph structure.
$ PETSC_GRAPH_UPDATE_ASSEMBLY - Update an existing graph inplace.

  Level: beginner

.seealso: PetscStreamGraphAssemble()
E*/
typedef enum {
  PETSC_GRAPH_INIT_ASSEMBLY,
  PETSC_GRAPH_UPDATE_ASSEMBLY
} PetscGraphAssemblyType;

/*S
  PetscStreamGraph - Container for device stream graph

  Allows asynchronous operations to be expressed as a DAG instead of single operations. This effectively allows a host
  to launch multiple device operations within a singlle call, amortizing kernel call overhead.

  level: beginner

.seealso: PetscStreamGraphCreate(), PetscStreamType, PetscStreamGraphSetType(), PetscStreamGraphDestroy()
S*/
typedef struct _n_PetscStreamGraph* PetscStreamGraph;

PETSC_EXTERN PetscErrorCode PetscStreamGraphInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscStreamGraphRegister(const char[],PetscErrorCode(*)(PetscStreamGraph));
PETSC_EXTERN PetscErrorCode PetscStreamGraphCreate(PetscStreamGraph*);
PETSC_EXTERN PetscErrorCode PetscStreamGraphDestroy(PetscStreamGraph*);
PETSC_EXTERN PetscErrorCode PetscStreamGraphSetType(PetscStreamGraph,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamGraphGetType(PetscStreamGraph,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscStreamGraphSetUp(PetscStreamGraph);
PETSC_EXTERN PetscErrorCode PetscStreamGraphAssemble(PetscStreamGraph,PetscGraphAssemblyType);
PETSC_EXTERN PetscErrorCode PetscStreamGraphExecute(PetscStreamGraph,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamGraphDuplicate(PetscStreamGraph,PetscStreamGraph*);
PETSC_EXTERN PetscErrorCode PetscStreamGraphGetGraph(PetscStreamGraph,void*);
PETSC_EXTERN PetscErrorCode PetscStreamGraphRestoreGraph(PetscStreamGraph,void*);

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
