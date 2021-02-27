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

/*E
  PetscStreamType - Stream type

  Level: beginner

.seealso: PetscStreamSetType(), PetscEventSetType(), PetscStreamScalarSetType()
E*/
typedef enum {
  PETSC_STREAM_INVALID = 0,
  PETSC_STREAM_CUDA = 1,
  PETSC_STREAM_HIP = 2
} PetscStreamType;

typedef struct _n_PetscEvent* PetscEvent;

PETSC_EXTERN PetscErrorCode PetscEventCreate(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventDestroy(PetscEvent*);
PETSC_EXTERN PetscErrorCode PetscEventSetType(PetscEvent,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscEventGetType(PetscEvent,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscEventSetFlags(PetscEvent,unsigned int,unsigned int);
PETSC_EXTERN PetscErrorCode PetscEventGetFlags(PetscEvent,unsigned int*,unsigned int*);
PETSC_EXTERN PetscErrorCode PetscEventSetUp(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventSynchronize(PetscEvent);
PETSC_EXTERN PetscErrorCode PetscEventQuery(PetscEvent,PetscBool*);

/*E
  PetscStreamMode - Stream blocking mode, indicates how a strea implementation will interact with the default "NULL"
  stream, which is usually blocking.

$ PETSC_STREAM_GLOBAL_BLOCKING - Alias for NULL stream. Any stream of this type will block the hostfor all other streams to finish work before starting its operations.
$ PETSC_STREAM_DEFAULT_BLOCKING - Stream will act independent of other streams, but will still be blocked by actions on the NULL stream.
$ PETSC_STREAM_GLOBAL_NONBLOCKING - Stream is truly asynchronous, and is blocked by nothing, not even the NULL stream.

  Level: intermediate

.seealso: PetscStreamSetMode(), PetscStreamGetMode()
E*/
typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING = 0,
  PETSC_STREAM_DEFAULT_BLOCKING = 1,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 2
} PetscStreamMode;

/*S
  PetscStream - Container for efficient management of a device stream.

  level: beginner

.seealso: PetscStreamCreate(), PetscStreamType, PetscStreamSetType(), PetscStreamDestroy()
S*/
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

typedef enum {
  PSS_ZERO = 0,
  PSS_ONE,
  PSS_INF,
  PSS_NAN,
  PSSCACHE_MAX
} PSSCacheType;

/*S
  PetscStreamScalar - A stream-aware container for a PetscScalar.

  This object allows for pipelining of scalar inputs or results between asynchronous functions. In addition, some basic
  arithmetic operations are also exposed in order to ennsure the PetscScalar need never be transfered to the host,
  however this is unavoidable for more complex interactions.

  level: beginner

.seealso: PetscStreamScalarCreate(), PetscStreamType, PetscStreamScalarSetType(), PetscStreamScalarDestroy()
S*/
typedef struct _n_PetscStreamScalar* PetscStreamScalar;

PETSC_EXTERN PetscErrorCode PetscStreamScalarCreate(PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarDestroy(PetscStreamScalar*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetType(PetscStreamScalar,PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetType(PetscStreamScalar,PetscStreamType*);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetUp(PetscStreamScalar);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetValue(PetscStreamScalar,const PetscScalar*,PetscMemType,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAwait(PetscStreamScalar,PetscScalar*,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetDeviceRead(PetscStreamScalar,const PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetDeviceWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarRestoreDeviceWrite(PetscStreamScalar,PetscScalar**,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarGetInfo(PetscStreamScalar,PSSCacheType,PetscBool,PetscBool*,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarSetInfo(PetscStreamScalar,PSSCacheType,PetscBool);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAXTY(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAYDX(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
PETSC_EXTERN PetscErrorCode PetscStreamScalarAccumulateOp(PetscStreamScalar,PetscInt,PetscStreamScalar[],PetscStreamComputeOp,PetscStreamComputeOp,PetscStream);

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
