#include <petsc/private/streamimpl.h>

static PetscInt createCount = 0, destroyCount = 0;

PetscErrorCode PetscStreamCreate(PetscStream *strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ++createCount;
  PetscValidPointer(strm,1);
  {
    size_t      freemem, totalmem;
    PetscBool   isCudaHost;
    cudaError_t cerr;

    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
    ierr = PetscMallocIsCUDAHost(&isCudaHost);CHKERRQ(ierr);
    cerr = cudaMemGetInfo(&freemem, &totalmem);CHKERRCUDA(cerr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "===============================\nBefore creating %D: %4.2fMB/ %4.2fMB (Free/Total) on %s\n===============================\n", createCount, freemem/(1024*1024.0), totalmem/(1024*1024.0), isCudaHost ? "DEVICE" : "HOST");CHKERRQ(ierr);
    ierr = PetscStackView(PETSC_STDOUT);CHKERRQ(ierr);
  }
  /* Setting to null taken from VecCreate(), why though? */
  ierr = PetscNew(strm);CHKERRQ(ierr);
  (*strm)->mode = PETSC_STREAM_DEFAULT_BLOCKING;
#if defined(PETSC_HAVE_CUDA)
  {
    cudaError_t cerr;

    cerr = cudaStreamCreateWithFlags(&((*strm)->cstream), cudaStreamDefault);CHKERRCUDA(cerr);
  }
#endif /* PETSC_HAVE_CUDA */
#if defined(PETSC_HAVE_HIP)
  {
    hipError_t herr;

    herr = hipStreamCreate(&((*strm)->hstream));CHKERRHIP(herr);
  }
#endif /* PETSC_HAVE_HIP */
  {
    size_t      freemem, totalmem;
    PetscBool   isCudaHost;
    cudaError_t cerr;

    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
    ierr = PetscMallocIsCUDAHost(&isCudaHost);CHKERRQ(ierr);
    cerr = cudaMemGetInfo(&freemem, &totalmem);CHKERRCUDA(cerr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "===============================\nAfter creating %D: %4.2fMB/ %4.2fMB (Free/Total) on %s\n===============================\n", createCount, freemem/(1024*1024.0), totalmem/(1024*1024.0), isCudaHost ? "DEVICE" : "HOST");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamDestroy(PetscStream *strm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ++destroyCount;
  //if (!*strm) PetscFunctionReturn(0);
  PetscValidPointer(strm,1);
  {
    size_t      freemem, totalmem;
    PetscBool   isCudaHost;
    cudaError_t cerr;

    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
    ierr = PetscMallocIsCUDAHost(&isCudaHost);CHKERRQ(ierr);
    cerr = cudaMemGetInfo(&freemem, &totalmem);CHKERRCUDA(cerr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "===============================\nBefore Free %D: %4.2fMB/ %4.2fMB (Free/Total) on %s\n===============================\n", destroyCount, freemem/(1024*1024.0), totalmem/(1024*1024.0), isCudaHost ? "DEVICE" : "HOST");CHKERRQ(ierr);
  }
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
  ierr = PetscFree(*strm);CHKERRQ(ierr);
  {
    size_t      freemem, totalmem;
    PetscBool   isCudaHost;
    cudaError_t cerr;

    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
    ierr = PetscMallocIsCUDAHost(&isCudaHost);CHKERRQ(ierr);
    cerr = cudaMemGetInfo(&freemem, &totalmem);CHKERRCUDA(cerr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "===============================\nAfter Free %D: %4.2fMB/ %4.2fMB (Free/Total) on %s\n===============================\n", destroyCount, freemem/(1024*1024.0), totalmem/(1024*1024.0), isCudaHost ? "DEVICE" : "HOST");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamSetMode(PetscStream strm, PetscStreamMode mode)
{
  PetscFunctionBegin;
  {
    PetscErrorCode ierr;

    ierr = PetscStreamSynchronize(strm);CHKERRQ(ierr);
  }
  /* For whatever reason PETSC_HAVE_DEVICE doesn't work? */
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  if ((mode == PETSC_STREAM_GLOBAL_NONBLOCKING) && (strm->mode != mode)) {
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
      herr = hipStreamCreateWithFlags(&strm->hstream, hipStreamNonBlocking);CHKERRHIP(herr);
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
