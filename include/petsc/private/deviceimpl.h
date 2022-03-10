#ifndef PETSCDEVICEIMPL_H
#define PETSCDEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

#if defined(__NVCC__) || defined(__CUDACC__)
#  define PETSC_USING_NVCC 1
#endif

#if defined(__HCC__) || (defined(__clang__) && defined(__HIP__))
#  define PETSC_USING_HCC 1
#endif

#if PetscDefined(USING_HCC) && PetscDefined(USING_NVCC)
#  error using both nvcc and hipcc at the same time?
#endif

/* type cast macros for some additional type-safety in C++ land */
#if defined(__cplusplus)
#  define PetscStreamTypeCast(...) static_cast<PetscStreamType>(__VA_ARGS__)
#  define PetscDeviceTypeCast(...) static_cast<PetscDeviceType>(__VA_ARGS__)
#else
#  define PetscStreamTypeCast(...) ((PetscStreamType)(__VA_ARGS__))
#  define PetscDeviceTypeCast(...) ((PetscDeviceType)(__VA_ARGS__))
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
template <typename T> void PetscValidDeviceType(T,int);
template <typename T> void PetscCheckCompatibleDeviceTypes(T,int,T,int);
template <typename T> void PetscValidDevice(T,int);
template <typename T> void PetscCheckCompatibleDevices(T,int,T,int);
template <typename T> void PetscValidStreamType(T,int);
template <typename T> void PetscValidDeviceContext(T,int);
template <typename T> void PetscCheckCompatibleDeviceContexts(T,int,T,int);
#elif PetscDefined(USE_DEBUG) || PetscDefined(DEVICE_KEEP_ERROR_CHECKING_MACROS)
#define PetscValidDeviceType(dtype,argno)                            do {                      \
    PetscDeviceType pvdt_dtype_ = PetscDeviceTypeCast(dtype);                                  \
    int             pvdt_argno_ = (int)(argno);                                                \
    PetscCheck(                                                                                \
      (pvdt_dtype_ >= PETSC_DEVICE_HOST) && (pvdt_dtype_ <= PETSC_DEVICE_MAX),                 \
      PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscDeviceType '%d': Argument #%d", \
      pvdt_dtype_,pvdt_argno_                                                                  \
    );                                                                                         \
    if (PetscUnlikely(!PetscDeviceConfiguredFor_Internal(pvdt_dtype_))) {                      \
      PetscCheck(                                                                              \
        pvdt_dtype_ != PETSC_DEVICE_MAX,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                  \
        "Invalid PetscDeviceType '%s': Argument #%d",PetscDeviceTypes[pvdt_dtype_],pvdt_argno_ \
      );                                                                                       \
      else SETERRQ(                                                                            \
        PETSC_COMM_SELF,PETSC_ERR_SUP,"Not configured for PetscDeviceType '%s': Argument #%d;" \
        " run configure --help %s for available options",PetscDeviceTypes[pvdt_dtype_],        \
        pvdt_argno_,PetscDeviceTypes[pvdt_dtype_]                                              \
      );                                                                                       \
    }                                                                                          \
  } while (0)

#define PetscCheckCompatibleDeviceTypes(dtype1,argno1,dtype2,argno2) do {                      \
    PetscDeviceType pccdt_dtype1_ = PetscDeviceTypeCast(dtype1);                               \
    PetscDeviceType pccdt_dtype2_ = PetscDeviceTypeCast(dtype2);                               \
    PetscValidDeviceType(pccdt_dtype1_,1);                                                     \
    PetscValidDeviceType(pccdt_dtype2_,2);                                                     \
    PetscCheck(pccdt_dtype1_ == pccdt_dtype2_,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"PetscDeviceTypes are incompatible: Arguments #%d and #%d. Expected PetscDeviceType '%s' but has '%s' instead",argno1,argno2,PetscDeviceTypes[pccdt_dtype1_],PetscDeviceTypes[pccdt_dtype2_]); \
} while (0)

#define PetscValidDevice(dev,argno)                                  do {                      \
    PetscDevice pvd_dev_   = dev;                                                              \
    int         pvd_argno_ = (int)(argno);                                                     \
    PetscValidPointer(pvd_dev_,pvd_argno_);                                                    \
    PetscValidDeviceType(pvd_dev_->type,pvd_argno_);                                           \
    PetscCheck(                                                                                \
      pvd_dev_->id >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,                                        \
      "Invalid PetscDevice: Argument #%d; id %" PetscInt_FMT " < 0",pvd_argno_,pvd_dev_->id    \
    );                                                                                         \
    PetscCheck(                                                                                \
      pvd_dev_->refcnt >= 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,                                    \
      "Invalid PetscDevice: Argument #%d; negative reference count %" PetscInt_FMT,            \
      pvd_argno_,pvd_dev_->refcnt                                                              \
    );                                                                                         \
  } while (0)

/*
  for now just checks strict equality, but this can be changed as some devices (i.e. kokkos and
  any cupm should be compatible once implemented)
*/
#define PetscCheckCompatibleDevices(dev1,argno1,dev2,argno2)          do {                     \
    PetscDevice pccd_dev1_   = (dev1),        pccd_dev2_   = (dev2);                           \
    int         pccd_argno1_ = (int)(argno1), pccd_argno2_ = (int)(argno2);                    \
    PetscValidDevice(pccd_dev1_,pccd_argno1_);                                                 \
    PetscValidDevice(pccd_dev2_,pccd_argno2_);                                                 \
    PetscCheckCompatibleDeviceTypes(                                                           \
      pccd_dev1_->type,pccd_argno1_,pccd_dev2_->type,pccd_argno2_                              \
    );                                                                                         \
  } while (0)

#define PetscValidStreamType(stype,argno)                             do {                     \
    PetscStreamType pvst_stype_ = PetscStreamTypeCast(stype);                                  \
    int             pvst_argno_ = (int)(argno);                                                \
    PetscCheck(                                                                                \
      (pvst_stype_ >= 0) && (pvst_stype_ <= PETSC_STREAM_MAX),                                 \
      PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscStreamType '%d': Argument #%d", \
      pvst_stype_,pvst_argno_                                                                  \
    );                                                                                         \
    PetscCheck(                                                                                \
      pvst_stype_ != PETSC_STREAM_MAX,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,                    \
      "Invalid PetscStreamType '%s': Argument #%d",PetscStreamTypes[pvst_stype_],pvst_argno_   \
    );                                                                                         \
  } while (0)

#define PetscValidDeviceContext(dctx,argno)                           do {                     \
    PetscDeviceContext pvdc_dctx_  = dctx;                                                     \
    int                pvdc_argno_ = (int)(argno);                                             \
    PetscValidPointer(pvdc_dctx_,pvdc_argno_);                                                 \
    PetscValidStreamType(pvdc_dctx_->streamType,pvdc_argno_);                                  \
    if (pvdc_dctx_->device) PetscValidDevice(pvdc_dctx_->device,pvdc_argno_);                  \
    else PetscCheck(                                                                           \
      !pvdc_dctx_->setup,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,                             \
      "Invalid PetscDeviceContext: Argument #%d; "                                             \
      "PetscDeviceContext is setup but has no PetscDevice",pvdc_argno_                         \
    );                                                                                         \
    PetscCheck(                                                                                \
      pvdc_dctx_->id >= 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,                                      \
      "Invalid PetscDeviceContext: Argument #%d; id %" PetscInt_FMT " < 1",                    \
      pvdc_argno_,pvdc_dctx_->id                                                               \
    );                                                                                         \
    PetscCheck(                                                                                \
      pvdc_dctx_->numChildren <= pvdc_dctx_->maxNumChildren,PETSC_COMM_SELF,                   \
      PETSC_ERR_ARG_CORRUPT,"Invalid PetscDeviceContext: Argument #%d; number of children %"   \
      PetscInt_FMT " > max number of children %" PetscInt_FMT,                                 \
      pvdc_argno_,pvdc_dctx_->numChildren,pvdc_dctx_->maxNumChildren                           \
    );                                                                                         \
  } while (0)

#define PetscCheckCompatibleDeviceContexts(dctx1,argno1,dctx2,argno2) do {                     \
    PetscDeviceContext pccdc_dctx1_  = (dctx1),       pccdc_dctx2_  = (dctx2);                 \
    int                pccdc_argno1_ = (int)(argno1), pccdc_argno2_ = (int)(argno2);           \
    PetscValidDeviceContext(pccdc_dctx1_,pccdc_argno1_);                                       \
    PetscValidDeviceContext(pccdc_dctx2_,pccdc_argno2_);                                       \
    if (pccdc_dctx1_->device && pccdc_dctx2_->device) PetscCheckCompatibleDevices(pccdc_dctx1_->device,pccdc_argno1_,pccdc_dctx2_->device,pccdc_argno2_); \
  } while (0)

#else /* PetscDefined(USE_DEBUG) */
#define PetscValidDeviceType(dtype,argno)
#define PetscCheckCompatibleDeviceTypes(dtype1,argno1,dtype2,argno2)
#define PetscValidDevice(dev,argno)
#define PetscCheckCompatibleDevices(dev1,argno1,dev2,argno2)
#define PetscValidStreamType(stype,argno)
#define PetscValidDeviceContext(dctx,argno)
#define PetscCheckCompatibleDeviceContexts(dctx1,argno1,dctx2,argno2)
#endif /* PetscDefined(USE_DEBUG) */

/* if someone is ready to rock with more than 128 GPUs on hand then we're in real trouble */
#define PETSC_DEVICE_MAX_DEVICES 128

typedef struct _DeviceOps *DeviceOps;
struct _DeviceOps {
  /* the creation routine for the corresponding PetscDeviceContext, this is NOT intended
   * to be called by the PetscDevice itself */
  PetscErrorCode (*createcontext)(PetscDeviceContext);
  PetscErrorCode (*configure)(PetscDevice);
  PetscErrorCode (*view)(PetscDevice,PetscViewer);
};

struct _n_PetscDevice {
  struct _DeviceOps ops[1];
  PetscInt          refcnt;   /* reference count for the device */
  PetscInt          id;       /* unique id per created PetscDevice */
  PetscInt          deviceId; /* the id of the underlying device, i.e. the return of
                               * cudaGetDevice() for example */
  PetscDeviceType   type;     /* type of device */
  void             *data;     /* placeholder */
};

typedef struct _DeviceContextOps *DeviceContextOps;
struct _DeviceContextOps {
  PetscErrorCode (*destroy)(PetscDeviceContext);
  PetscErrorCode (*changestreamtype)(PetscDeviceContext,PetscStreamType);
  PetscErrorCode (*setup)(PetscDeviceContext);
  PetscErrorCode (*query)(PetscDeviceContext,PetscBool*);
  PetscErrorCode (*waitforcontext)(PetscDeviceContext,PetscDeviceContext);
  PetscErrorCode (*synchronize)(PetscDeviceContext);
  PetscErrorCode (*getblashandle)(PetscDeviceContext,void*);
  PetscErrorCode (*getsolverhandle)(PetscDeviceContext,void*);
  PetscErrorCode (*getstreamhandle)(PetscDeviceContext,void*);
  PetscErrorCode (*begintimer)(PetscDeviceContext);
  PetscErrorCode (*endtimer)(PetscDeviceContext,PetscLogDouble*);
};

struct _n_PetscDeviceContext {
  struct _DeviceContextOps  ops[1];
  PetscDevice               device;         /* the device this context stems from */
  void                     *data;           /* solver contexts, event, stream */
  PetscInt                  id;             /* unique id per created context */
  PetscInt                 *childIDs;       /* array containing ids of contexts currently forked from this one */
  PetscInt                  numChildren;    /* how many children does this context expect to destroy */
  PetscInt                  maxNumChildren; /* how many children can this context have room for without realloc'ing */
  PetscStreamType           streamType;     /* how should this contexts stream behave around other streams? */
  PetscBool                 setup;
};

/* PetscDevice Internal Functions */
#if PetscDefined(HAVE_CXX)
PETSC_INTERN PetscErrorCode PetscDeviceInitializeFromOptions_Internal(MPI_Comm);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceInitializeDefaultDevice_Internal(PetscDeviceType,PetscInt);
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscDeviceGetDefaultForType_Internal(PetscDeviceType,PetscDevice*);

static inline PETSC_CONSTEXPR_14 PetscBool PetscDeviceConfiguredFor_Internal(PetscDeviceType type)
{
  switch(type) {
  case PETSC_DEVICE_HOST: return PETSC_TRUE;
    /* casts are needed in C++ */
  case PETSC_DEVICE_CUDA: return (PetscBool)PetscDefined(HAVE_CUDA);
  case PETSC_DEVICE_HIP:  return (PetscBool)PetscDefined(HAVE_HIP);
  case PETSC_DEVICE_SYCL: return (PetscBool)PetscDefined(HAVE_SYCL);
  case PETSC_DEVICE_MAX:  return PETSC_FALSE;
    /* Do not add default case! Will make compiler warn on new additions to PetscDeviceType! */
  }
  PetscUnreachable();
  return PETSC_FALSE;
}

/* More general form of PetscDeviceDefaultType_Internal(), as it calls the former using
 * the automatically selected default PetscDeviceType */
#define PetscDeviceGetDefault_Internal(device) PetscDeviceGetDefaultForType_Internal(PETSC_DEVICE_DEFAULT,device)

static inline PetscErrorCode PetscDeviceCheckDeviceCount_Internal(PetscInt count)
{
  PetscFunctionBegin;
  PetscAssert(count < PETSC_DEVICE_MAX_DEVICES,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Detected %" PetscInt_FMT " devices, which is larger than maximum supported number of devices %d",count,PETSC_DEVICE_MAX_DEVICES);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceReference_Internal(PetscDevice device)
{
  PetscFunctionBegin;
  ++(device->refcnt);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceDereference_Internal(PetscDevice device)
{
  PetscFunctionBegin;
  --(device->refcnt);
  PetscAssert(device->refcnt >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"PetscDevice has negative reference count %" PetscInt_FMT,device->refcnt);
  PetscFunctionReturn(0);
}
#else  /* PETSC_HAVE_CXX for PetscDevice Internal Functions */
#define PetscDeviceInitializeFromOptions_Internal(comm)      0
#define PetscDeviceInitializeDefaultDevice_Internal(type,id) 0
#define PetscDeviceConfiguredFor_Internal(type)              PETSC_FALSE
#define PetscDeviceGetDefaultForType_Internal(Type,device)   0
#define PetscDeviceGetDefault_Internal(device)               0
#define PetscDeviceCheckDeviceCount_Internal(count)          0
#define PetscDeviceReference_Internal(device)                0
#define PetscDeviceDereference_Internal(device)              0
#endif /* PETSC_HAVE_CXX for PetscDevice Internal Functions */

/* PetscDeviceContext Internal Functions */
#if PetscDefined(HAVE_CXX)
PETSC_INTERN PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType);
PETSC_INTERN PetscErrorCode PetscDeviceContextSetRootStreamType_Internal(PetscStreamType);
PETSC_INTERN PetscErrorCode PetscDeviceContextGetNullContextForDevice_Internal(PetscDevice,PetscDeviceContext*);

static inline PetscErrorCode PetscDeviceContextSetDefaultDeviceForType_Internal(PetscDeviceContext dctx, PetscDeviceType type)
{
  PetscDevice device;

  PetscFunctionBegin;
  PetscCall(PetscDeviceGetDefaultForType_Internal(type,&device));
  PetscCall(PetscDeviceContextSetDevice(dctx,device));
  PetscFunctionReturn(0);
}

#define PetscDeviceContextSetDefaultDevice_Internal(dctx)                       \
  PetscDeviceContextSetDefaultDeviceForType_Internal(dctx,PETSC_DEVICE_DEFAULT)

static inline PetscErrorCode PetscDeviceContextGetNullContext_Internal(PetscDeviceContext *dctx)
{
  PetscDeviceContext gctx;
  PetscDevice        gdev;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  PetscCall(PetscDeviceContextGetCurrentContext(&gctx));
  PetscCall(PetscDeviceContextGetDevice(gctx,&gdev));
  PetscCall(PetscDeviceContextGetNullContextForDevice_Internal(gdev,dctx));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetOptionalNullContext_Internal(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  if (!*dctx) PetscCall(PetscDeviceContextGetNullContext_Internal(dctx));
  PetscValidDeviceContext(*dctx,1);
  PetscFunctionReturn(0);
}

/* note, only does assertion checking in debug mode */
static inline PetscErrorCode PetscDeviceContextGetCurrentContextAssertType_Internal(PetscDeviceContext *dctx, PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  PetscValidDeviceType(type,2);
  PetscCall(PetscDeviceContextGetCurrentContext(dctx));
  PetscCheckCompatibleDeviceTypes(type,1,(*dctx)->device->type,2);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetBLASHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(handle,2);
  PetscCall((*dctx->ops->getblashandle)(dctx,handle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetSOLVERHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(handle,2);
  PetscCall((*dctx->ops->getsolverhandle)(dctx,handle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextGetStreamHandle_Internal(PetscDeviceContext dctx, void *handle)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(handle,2);
  PetscCall((*dctx->ops->getstreamhandle)(dctx,handle));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextBeginTimer_Internal(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscCall((*dctx->ops->begintimer)(dctx));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscDeviceContextEndTimer_Internal(PetscDeviceContext dctx, PetscLogDouble *elapsed)
{
  PetscFunctionBegin;
  /* we do error checking here as this routine is an entry-point */
  PetscValidDeviceContext(dctx,1);
  PetscValidRealPointer(elapsed,2);
  PetscCall((*dctx->ops->endtimer)(dctx,elapsed));
  PetscFunctionReturn(0);
}
#else /* PETSC_HAVE_CXX for PetscDeviceContext Internal Functions */
#define PetscDeviceContextSetRootDeviceType_Internal(type)                0
#define PetscDeviceContextSetRootStreamType_Internal(type)                0
#define PetscDeviceContextSetDefaultDeviceForType_Internal(dctx,type)     0
#define PetscDeviceContextSetDefaultDevice_Internal(dctx)                 0
#define PetscDeviceContextGetCurrentContextAssertType_Internal(dctx,type) 0
#define PetscDeviceContextGetNullContext_Internal(dctx)                   0
#define PetscDeviceContextGetOptionalNullContext_Internal(dctx)           0
#define PetscDeviceContextGetBLASHandle_Internal(dctx,handle)             0
#define PetscDeviceContextGetSOLVERHandle_Internal(dctx,handle)           0
#define PetscDeviceContextBeginTimer_Internal(dctx)                       0
#define PetscDeviceContextEndTimer_Internal(dctx,elapsed)                 0
#endif /* PETSC_HAVE_CXX for PetscDeviceContext Internal Functions */

PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HOST(PetscDeviceContext);

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext);
#endif
#if PetscDefined(HAVE_SYCL)
PETSC_INTERN PetscErrorCode PetscDeviceContextCreate_SYCL(PetscDeviceContext);
#endif

static inline PetscErrorCode PetscGetMemType(const void *ptr,PetscMemType *type)
{
  PetscFunctionBegin;
  *type = PETSC_MEMTYPE_HOST;
#if defined(PETSC_HAVE_CUDA)
  if (PetscDeviceInitialized(PETSC_DEVICE_CUDA) && ptr) {
    cudaError_t                  cerr;
    struct cudaPointerAttributes attr;
    enum cudaMemoryType          mtype;
    cerr = cudaPointerGetAttributes(&attr,ptr); /* Do not check error since before CUDA 11.0, passing a host pointer returns cudaErrorInvalidValue */
    if (cerr) cerr = cudaGetLastError(); /* If there was an error, return it and then reset it */
    #if (CUDART_VERSION < 10000)
      mtype = attr.memoryType;
    #else
      mtype = attr.type;
    #endif
    if (cerr == cudaSuccess && mtype == cudaMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif

#if defined(PETSC_HAVE_HIP)
  if (PetscDeviceInitialized(PETSC_DEVICE_HIP) && ptr) {
    hipError_t                   cerr;
    struct hipPointerAttribute_t attr;
    enum hipMemoryType           mtype;
    cerr = hipPointerGetAttributes(&attr,ptr);
    if (cerr) cerr = hipGetLastError();
    mtype = attr.memoryType;
    if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif
  PetscFunctionReturn(0);
}
#endif /* PETSCDEVICEIMPL_H */
