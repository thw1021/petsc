#include <petsc/private/streamimpl.h>

PetscErrorCode PetscStreamCreate(PetscStream *strm)
{
  PetscStream    s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(strm,1);
  ierr = PetscNew(&s);CHKERRQ(ierr);
  s->mode = PETSC_STREAM_DEFAULT_BLOCKING;
#if defined(PETSC_HAVE_CUDA)
  {
    cudaError_t cerr;

    cerr = cudaStreamCreate(&s->cstream);CHKERRCUDA(cerr);
  }
#endif /* PETSC_HAVE_CUDA */
#if defined(PETSC_HAVE_HIP)
  {
    hipError_t herr;

    herr = hipStreamCreate(&s->hstream);CHKERRHIP(ierr);
  }
#endif /* PETSC_HAVE_HIP */
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamDestroy(PetscStream *strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*strm) PetscFunctionReturn(0);
  PetscValidPointer(strm,1);
#if defined(PETSC_HAVE_CUDA)
  {
    cudaError_t cerr;

    cerr = cudaStreamDestroy((*strm)->cstream);CHKERRCUDA(cerr);
  }
#endif /* PETSC_HAVE_CUDA */
#if defined(PETSC_HAVE_HIP)
  {
    hipError_t herr;

    herr = hipStreamDestroy((*strm)->hstream);CHKERRHIP(ierr);
  }
#endif /* PETSC_HAVE_HIP */
  ierr = PetscFree(strm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSetMode(PetscStream strm, PetscStreamMode mode)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_DEVICE)
  if (mode == PETSC_STREAM_GLOBAL_NONBLOCKING && strm->mode != mode) {
#if defined(PETSC_HAVE_CUDA)
    {
      cudaError_t cerr;

      /* I don't think there is a way to change flags on existing streams */
      cerr = cudaStreamDestroy(strm->cstream);CHKERRCUDA(cerr);
      cerr = cudaStreamCreateWithFlags(&strm->cstream, cudaStreamNonBlocking);CHKERRCUDA(cerr);
    }
#endif
#if defined(PETSC_HAVE_HIP)
    {
      hipError_t herr;

      herr = hipStreamDestroy(strm->hstream);CHKERRHIP(herr);
      herr = hipStreamCreaetWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
    }
#endif
  }
#endif
  strm->mode = mode;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGetMode(PetscStream strm, PetscStreamMode *mode)
{
  PetscFunctionBegin;
  PetscValidPointer(mode,2);
  *mode = strm->mode;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSynchronize(PetscStream strm)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  {
    cudaError_t cerr;

    switch(strm->mode) {
    case PETSC_STREAM_GLOBAL_BLOCKING:
      cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
      break;
    case PETSC_STREAM_DEFAULT_BLOCKING:
    case PETSC_STREAM_GLOBAL_NONBLOCKING:
      cerr = cudaStreamSynchronize(strm->cstream);CHKERRCUDA(cerr);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Stream has incompatible mode\n");
      break;
    }
  }
#endif
#if defined(PETSC_HAVE_HIP)
  {
    hipError_t herr;

    switch(strm->mode) {
    case PETSC_STREAM_GLOBAL_BLOCKING:
      herr = hipDeviceSynchronize();CHKERRHIP(herr);
      break;
    case PETSC_STREAM_DEFAULT_BLOCKING:
    case PETSC_STREAM_GLOBAL_NONBLOCKING:
      herr = hipStreamSynchronize(strm->hstream);CHKERRHIP(herr);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Stream has incompatible mode\n");
      break;
    }
  }
#endif
  PetscFunctionReturn(0);
}
