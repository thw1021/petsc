#if !defined(STREAMIMPL_H)
#define STREAMIMPL_H

#include <petsc/private/petscimpl.h>
#include <petscstream.h>

#define PETSC_MAX_STREAMS 4

struct _n_PetscStream {
#if PetscDefined(HAVE_CUDA)
  cudaStream_t cstream[PETSC_MAX_STREAMS];
#if PetscDefined(USE_DEBUG)
  PetscBool    gotHIP[PETSC_MAX_STREAMS];
#endif /* PETSC_USE_DEBUG */
  cudaEvent_t  cevent;
#endif /* PETSC_HAVE_CUDA */
#if PetscDefined(HAVE_HIP)
  hipStream_t  hstream[PETSC_MAX_STREAMS];
#if PetscDefined(USE_DEBUG)
  PetscBool    gotHIP[PETSC_MAX_STREAMS];
#endif /* PETSC_USE_DEBUG */
  hipEvent_t   hevent;
#endif /* PETSC_HAVE_HIP */
  PetscStreamMode mode;
};

PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleHIP_Private(PetscStream strm)
{
  PetscFunctionBegin;
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
    if (!(strm->hevent))  {herr = hipEventCreate(&strm->hevent);CHKERRHIP(herr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_HIP)
    if (!(strm->hstream)) {
      herr = hipStreamCreateWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
    }
    if (!(strm->hevent))  {herr = hipEventCreate(&strm->hevent);CHKERRHIP(herr);}
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

#if 0
PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleCUDA_Private(PetscStream strm)
{
  PetscFunctionBegin;
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
    if (!(strm->cevent))  {cerr = cudaEventCreate(&strm->cevent);CHKERRCUDA(cerr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream)) {
      cerr = cudaStreamCreateWithFlags(&strm->cstream, cudaStreamNonBlocking);CHKERRCUDA(cerr);
    }
    if (!(strm->cevent))  {cerr = cudaEventCreate(&strm->cevent);CHKERRCUDA(cerr);}
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}
#endif

PETSC_STATIC_INLINE PetscErrorCode PetscStreamAssembleCUDA_Private(PetscStream strm, PetscInt stri)
{
  PetscFunctionBegin;
  switch (strm->mode) {
#if PetscDefined(HAVE_CUDA)
    cudaError_t cerr;
#endif
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* NULL stream always blocks */
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream[stri])) {cerr = cudaStreamCreate(&strm->cstream[stri]);CHKERRCUDA(cerr);}
    if (!(strm->cevent))  {cerr = cudaEventCreate(&strm->cevent);CHKERRCUDA(cerr);}
#endif
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
#if PetscDefined(HAVE_CUDA)
    if (!(strm->cstream[stri])) {
      cerr = cudaStreamCreateWithFlags(&strm->cstream[stri], cudaStreamNonBlocking);CHKERRCUDA(cerr);
    }
    if (!(strm->cevent))  {cerr = cudaEventCreate(&strm->cevent);CHKERRCUDA(cerr);}
#endif
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unknown/invalid PetscStreamMode\n");
    break;
  }
  PetscFunctionReturn(0);
}

#endif
