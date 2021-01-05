#if !defined(DEVICEIMPL_H)
#define DEVICEIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscdevice.h>

struct _n_PetscStream {
  PetscStreamMode mode;
#if PetscDefined(HAVE_CUDA)
  cudaStream_t    cstream;
#endif /* PETSC_HAVE_CUDA */
#if PetscDefined(HAVE_HIP)
  hipStream_t     hstream;
#endif /* PETSC_HAVE_HIP */
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleHIP_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
#if PetscDefined(HAVE_HIP)
    hipError_t herr;
#endif
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
#if PetscDefined(HAVE_HIP)
    if (!(strm->hstream)) {herr = hipStreamCreate(&strm->hstream);CHKERRHIP(herr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_HIP)
    if (!(strm->hstream)) {
      herr = hipStreamCreateWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleCUDA_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t cerr;
#endif
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream)) {cerr = cudaStreamCreate(&strm->cstream);CHKERRCUDA(cerr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream)) {
      cerr = cudaStreamCreateWithFlags(&strm->cstream, cudaStreamNonBlocking);CHKERRCUDA(cerr);
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamDisassembleHIP_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_HIP)
    if (strm->hstream) {
      hipError_t herr;

      herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
      strm->hstream = NULL;
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamDisassembleCUDA_Internal(PetscStream strm)
{
  PetscFunctionBeginHot;
  switch (strm->mode) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (strm->cstream) {
      cudaError_t cerr;

      cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
      strm->cstream = NULL;
    }
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

struct _n_PetscEvent {
  PetscBool    setup;
#if PetscDefined(HAVE_CUDA)
  cudaEvent_t  cevent;
#endif /* PETSC_HAVE_CUDA */
#if PetscDefined(HAVE_HIP)
  hipEvent_t   hevent;
#endif /* PETSC_HAVE_HIP */
  unsigned int eventFlags, waitFlags;
};

#endif /* DEVICEIMPL_H */
