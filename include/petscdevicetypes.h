#ifndef PETSCDEVICETYPES_H
#define PETSCDEVICETYPES_H

#include <petscmacros.h> /*I <petscdevicetypes.h> I*/
#include <petscsystypes.h>
#include <petscerror.h>

/*E
  PetscMemType - Memory type of a pointer

  Developer Note:
  Encoding of the bitmask in binary: xxxxyyyz

$ z = 0                - Host memory
$ z = 1                - Device memory
$ yyy = 000            - CUDA-related memory
$ yyy = 001            - HIP-related memory
$ yyy = 010            - SYCL-related memory
$ xxxxyyy1 = 0000,0001 - CUDA memory
$ xxxxyyy1 = 0001,0001 - CUDA NVSHMEM memory
$ xxxxyyy1 = 0000,0011 - HIP memory
$ xxxxyyy1 = 0000,0101 - SYCL memory

  Other types of memory, e.g., CUDA managed memory, can be added when needed.

  Level: beginner

  Notes:
  PETSC_MEMTYPE_KOKKOS depends on the KOKKOS backend configuration

.seealso: `VecGetArrayAndMemType()`, `PetscSFBcastWithMemTypeBegin()`, `PetscSFReduceWithMemTypeBegin()`
E*/
typedef enum {
  PETSC_MEMTYPE_HOST    = 0,
  PETSC_MEMTYPE_DEVICE  = 0x01,
  PETSC_MEMTYPE_CUDA    = 0x01,
  PETSC_MEMTYPE_NVSHMEM = 0x11,
  PETSC_MEMTYPE_HIP     = 0x03,
  PETSC_MEMTYPE_SYCL    = 0x05,
} PetscMemType;
#if PetscDefined(HAVE_CUDA)
# define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_CUDA
#elif PetscDefined(HAVE_HIP)
# define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_HIP
#elif PetscDefined(HAVE_SYCL)
# define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_SYCL
#else
# define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_HOST
#endif

#define PetscMemTypeHost(m)    (((m) & 0x1) == PETSC_MEMTYPE_HOST)
#define PetscMemTypeDevice(m)  (((m) & 0x1) == PETSC_MEMTYPE_DEVICE)
#define PetscMemTypeCUDA(m)    (((m) & 0xF) == PETSC_MEMTYPE_CUDA)
#define PetscMemTypeHIP(m)     (((m) & 0xF) == PETSC_MEMTYPE_HIP)
#define PetscMemTypeSYCL(m)    (((m) & 0xF) == PETSC_MEMTYPE_SYCL)
#define PetscMemTypeNVSHMEM(m) ((m) == PETSC_MEMTYPE_NVSHMEM)

#define PETSC_OFFLOAD_VECKOKKOS_DEPRECATED PETSC_OFFLOAD_VECKOKKOS PETSC_DEPRECATED_ENUM("Use PETSC_OFFLOAD_KOKKOS (since version 3.17.0)")

/*E
  PetscOffloadMask - indicates which memory (CPU, GPU, or none) contains valid data

$ PETSC_OFFLOAD_UNALLOCATED - no memory contains valid matrix entries; NEVER used for vectors
$ PETSC_OFFLOAD_GPU         - GPU has valid vector/matrix entries
$ PETSC_OFFLOAD_CPU         - CPU has valid vector/matrix entries
$ PETSC_OFFLOAD_BOTH        - Both GPU and CPU have valid vector/matrix entries and they match
$ PETSC_OFFLOAD_KOKKOS      - Reserved for Kokkos matrix and vector. It means the offload is managed by Kokkos, thus this flag itself cannot tell you where the valid data is.

  Level: developer
E*/
typedef enum {
  PETSC_OFFLOAD_UNALLOCATED = 0x0,
  PETSC_OFFLOAD_CPU         = 0x1,
  PETSC_OFFLOAD_GPU         = 0x2,
  PETSC_OFFLOAD_BOTH        = 0x3,
  PETSC_OFFLOAD_VECKOKKOS_DEPRECATED = 0x100,
  PETSC_OFFLOAD_KOKKOS      = 0x100
} PetscOffloadMask;

#define PetscOffloadUnallocated(m) ((m)                       == PETSC_OFFLOAD_UNALLOCATED)
#define PetscOffloadHost(m)        (((m) & PETSC_OFFLOAD_CPU) == PETSC_OFFLOAD_CPU)
#define PetscOffloadDevice(m)      (((m) & PETSC_OFFLOAD_GPU) == PETSC_OFFLOAD_GPU)

/*E
  PetscDeviceInitType - Initialization strategy for PetscDevice

$ PETSC_DEVICE_INIT_NONE  - PetscDevice is never initialized
$ PETSC_DEVICE_INIT_LAZY  - PetscDevice is initialized on demand
$ PETSC_DEVICE_INIT_EAGER - PetscDevice is initialized as soon as possible

  Notes:
  PETSC_DEVICE_INIT_NONE implies that any initialization of PetscDevice is disallowed and
  doing so results in an error. Useful to ensure that no accelerator is used in a program.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceType`, `PetscDeviceInitialize()`,
`PetscDeviceInitialized()`, `PetscDeviceCreate()`
E*/
typedef enum {
  PETSC_DEVICE_INIT_NONE,
  PETSC_DEVICE_INIT_LAZY,
  PETSC_DEVICE_INIT_EAGER
} PetscDeviceInitType;
PETSC_EXTERN const char *const PetscDeviceInitTypes[];

/*E
  PetscDeviceType - Kind of accelerator device backend

$ PETSC_DEVICE_HOST    - Host, no accelerator backend found
$ PETSC_DEVICE_CUDA    - CUDA enabled GPU
$ PETSC_DEVICE_HIP     - ROCM/HIP enabled GPU
$ PETSC_DEVICE_SYCL    - SYCL enabled device
$ PETSC_DEVICE_DEFAULT - Automatically select backend based on availability
$ PETSC_DEVICE_MAX     - Always 1 greater than the largest valid PetscDeviceType, invalid type, do not use

  Notes:
  PETSC_DEVICE_DEFAULT is selected in the following order: PETSC_DEVICE_HIP, PETSC_DEVICE_CUDA,
  PETSC_DEVICE_SYCL, PETSC_DEVICE_HOST.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceInitType`, `PetscDeviceCreate()`
E*/
typedef enum {
  PETSC_DEVICE_HOST,
  PETSC_DEVICE_CUDA,
  PETSC_DEVICE_HIP,
  PETSC_DEVICE_SYCL,
  PETSC_DEVICE_MAX
} PetscDeviceType;
PETSC_EXTERN const char *const PetscDeviceTypes[];
#if defined(PETSC_HAVE_HIP)
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_HIP
#elif defined(PETSC_HAVE_CUDA)
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_CUDA
#elif PetscDefined(HAVE_SYCL)
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_SYCL
#else
#  define PETSC_DEVICE_DEFAULT PETSC_DEVICE_HOST
#endif

/*S
  PetscDevice - Handle to an accelerator "device" (usually a GPU)

  Notes:
  This object is used to house configuration and state of a device, but does not offer any
  ability to interact with or drive device computation. This functionality is facilitated
  instead by the PetscDeviceContext object.

  Level: beginner

.seealso: `PetscDeviceType`, `PetscDeviceInitType`, `PetscDeviceCreate()`,
`PetscDeviceConfigure()`, `PetscDeviceDestroy()`, `PetscDeviceContext`,
`PetscDeviceContextSetDevice()`, `PetscDeviceContextGetDevice()`
S*/
typedef struct _n_PetscDevice *PetscDevice;

/*E
  PetscStreamType - Stream blocking mode, indicates how a stream implementation will interact
  with the default "NULL" stream, which is usually blocking.

$ PETSC_STREAM_GLOBAL_BLOCKING    - Alias for NULL stream. Any stream of this type will block the host for all other streams to finish work before starting its operations.
$ PETSC_STREAM_DEFAULT_BLOCKING   - Stream will act independent of other streams, but will still be blocked by actions on the NULL stream.
$ PETSC_STREAM_GLOBAL_NONBLOCKING - Stream is truly asynchronous, and is blocked by nothing, not even the NULL stream.
$ PETSC_STREAM_MAX                - Always 1 greater than the largest PetscStreamType, do not use

  Level: intermediate

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextGetStreamType()`
E*/
typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING,
  PETSC_STREAM_DEFAULT_BLOCKING,
  PETSC_STREAM_GLOBAL_NONBLOCKING,
  PETSC_STREAM_MAX
} PetscStreamType;
PETSC_EXTERN const char *const PetscStreamTypes[];

/*E
  PetscDeviceContextJoinMode - Describes the type of join operation to perform in
  PetscDeviceContextJoin()

$ PETSC_DEVICE_CONTEXT_JOIN_DESTROY - Destroy all incoming sub-contexts after join.
$ PETSC_DEVICE_CONTEXT_JOIN_SYNC    - Synchronize incoming sub-contexts after join.
$ PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC - Do not synchronize incoming sub-contexts after join.

  Level: beginner

.seealso: `PetscDeviceContext`, `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`
E*/
typedef enum {
  PETSC_DEVICE_CONTEXT_JOIN_DESTROY,
  PETSC_DEVICE_CONTEXT_JOIN_SYNC,
  PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC
} PetscDeviceContextJoinMode;
PETSC_EXTERN const char *const PetscDeviceContextJoinModes[];

/*S
  PetscDeviceContext - Container to manage stream dependencies and the various solver handles
  for asynchronous device compute.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextDestroy()`,`PetscDeviceContextFork()`, `PetscDeviceContextJoin()`
S*/
typedef struct _n_PetscDeviceContext *PetscDeviceContext;

typedef enum {
  PETSC_DEVICE_COPY_HTOH,
  PETSC_DEVICE_COPY_DTOH,
  PETSC_DEVICE_COPY_HTOD,
  PETSC_DEVICE_COPY_DTOD,
  PETSC_DEVICE_COPY_AUTO,
} PetscDeviceCopyMode;

typedef enum {
  PETSC_MEMORY_ACCESS_READ       = 0x1,
  PETSC_MEMORY_ACCESS_WRITE      = 0x2,
  PETSC_MEMORY_ACCESS_READ_WRITE = 0x3,
} PetscMemoryAccessMode;
#define PetscMemoryAccessRead(m)  (((m) & PETSC_MEMORY_ACCESS_READ)  == PETSC_MEMORY_ACCESS_READ)
#define PetscMemoryAccessWrite(m) (((m) & PETSC_MEMORY_ACCESS_WRITE) == PETSC_MEMORY_ACCESS_WRITE)

#if defined(__cplusplus)
static_assert(PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_READ),"");
static_assert(PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_READ_WRITE),"");
static_assert(!PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_WRITE),"");
static_assert(PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_WRITE),"");
static_assert(PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_READ_WRITE),"");
static_assert(!PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_READ),"");
#endif

typedef PetscErrorCode(*const PetscHostFunction)(PetscDeviceContext,void*);

typedef enum {
  PETSC_OPERATOR_PLUS,
  PETSC_OPERATOR_MINUS,
  PETSC_OPERATOR_MULTIPLY,
  PETSC_OPERATOR_DIVIDE,
  PETSC_OPERATOR_EQUAL
} PetscOperatorType;

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 PetscMemType PetscOffloadMaskToMemType(PetscOffloadMask mask)
{
  switch (mask) {
  case PETSC_OFFLOAD_UNALLOCATED:
  case PETSC_OFFLOAD_CPU:
    return PETSC_MEMTYPE_HOST;
  case PETSC_OFFLOAD_GPU:
  case PETSC_OFFLOAD_BOTH:
    return PETSC_MEMTYPE_DEVICE;
  case PETSC_OFFLOAD_KOKKOS:
    return PETSC_MEMTYPE_KOKKOS;
  }
  PetscUnreachable();
  return PETSC_MEMTYPE_HOST;
}

PETSC_NODISCARD static inline PetscErrorCode PetscOffloadMaskToDeviceCopyMode(PetscOffloadMask dest, PetscOffloadMask src, PetscDeviceCopyMode *mode)
{
  PetscFunctionBegin;
  *mode = PETSC_DEVICE_COPY_AUTO;
  PetscAssert(dest != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot copy to unallocated");
  PetscAssert(src != PETSC_OFFLOAD_UNALLOCATED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot copy from unallocated");

  if (PetscOffloadDevice(dest)) {
    *mode = PetscOffloadHost(src) ? PETSC_DEVICE_COPY_HTOD : PETSC_DEVICE_COPY_DTOD;
  } else {
    *mode = PetscOffloadHost(src) ? PETSC_DEVICE_COPY_HTOH : PETSC_DEVICE_COPY_DTOH;
  }
  PetscFunctionReturn(0);
}

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 PetscDeviceCopyMode PetscMemTypeToDeviceCopyMode(PetscMemType dest, PetscMemType src)
{
  if (PetscMemTypeHost(dest)) {
    return PetscMemTypeHost(src)   ? PETSC_DEVICE_COPY_HTOH : PETSC_DEVICE_COPY_DTOH;
  } else {
    return PetscMemTypeDevice(src) ? PETSC_DEVICE_COPY_DTOD : PETSC_DEVICE_COPY_HTOD;
  }
}

typedef struct
{
  unsigned int locked  : 1; // 1 = locked,  0 = unlocked
  unsigned int tainted : 1; // 1 = tainted, 0 = untainted
} PetscManagedTypeState;

#define PetscTypeSuffix Scalar
#include "petscmanagedtype.inl"
#define PetscTypeSuffix Real
#include "petscmanagedtype.inl"
#define PetscTypeSuffix Int
#include "petscmanagedtype.inl"
#endif /* PETSCDEVICETYPES_H */
