#include "contextcupm.hpp" /*I "petscdevice.h" I*/

using namespace Petsc;

template <PetscDeviceContextBackends T>
PetscErrorCode cupmContext<T>::destroy(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  cupmError_t              cerr;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (dci->stream) {cerr = cupmStreamDestroy(dci->stream);CHKERRCUPM(cerr);}
  if (dci->event)  {cerr = cupmEventDestroy(dci->event);CHKERRCUPM(cerr);}
  ierr = PetscFree(dctx->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackends T>
PetscErrorCode cupmContext<T>::setUp(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  PetscErrorCode           ierr;
  cupmError_t              cerr;

  PetscFunctionBegin;
  switch (dctx->streamType) {
  case PETSC_STREAM_GLOBAL_BLOCKING:
    /* don't create a stream for global blocking */
    dci->stream = NULL;
    break;
  case PETSC_STREAM_DEFAULT_BLOCKING:
    cerr = cupmStreamCreate(&dci->stream);CHKERRCUPM(cerr);
    break;
  case PETSC_STREAM_GLOBAL_NONBLOCKING:
    cerr = cupmStreamCreateWithFlags(&dci->stream,cupmStreamNonBlocking);CHKERRCUPM(cerr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Invalid PetscStreamType %D",(PetscInt)dctx->streamType);
    break;
  }
  cerr = cupmEventCreate(&dci->event);CHKERRCUPM(cerr);
  ierr = GetHandles(dci);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackends T>
PetscErrorCode cupmContext<T>::query(PetscDeviceContext dctx, PetscBool *idle) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;

  PetscFunctionBegin;
  *idle = cupmStreamQuery(dci->stream) == cupmErrorNotReady ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackends T>
PetscErrorCode cupmContext<T>::waitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dcia = (PetscDeviceContext_IMPLS *)dctxa->data;
  PetscDeviceContext_IMPLS *dcib = (PetscDeviceContext_IMPLS *)dctxb->data;
  cupmError_t               cerr;

  PetscFunctionBegin;
  cerr = cupmEventRecord(dcib->event,dcib->stream);CHKERRCUPM(cerr);
  cerr = cupmStreamWaitEvent(dcia->stream,dcib->event,0);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

template <PetscDeviceContextBackends T>
PetscErrorCode cupmContext<T>::synchronize(PetscDeviceContext dctx) PETSC_NOEXCEPT
{
  PetscDeviceContext_IMPLS *dci = (PetscDeviceContext_IMPLS *)dctx->data;
  cupmError_t               cerr;

  PetscFunctionBegin;
  /* in case anything was queued on the event */
  cerr = cupmStreamWaitEvent(dci->stream,dci->event,0);CHKERRCUPM(cerr);
  cerr = cupmStreamSynchronize(dci->stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
static const cupmContextCuda contextCuda(PetscDeviceContextCreate_CUDAM);
#endif

PetscErrorCode PetscDeviceContextCreate_CUDAM(PetscDeviceContext dctx)
{
  PetscDeviceContext_(CUDA) *dci;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_HIP)
static const cupmContextHip contextHip(PetscDeviceContextCreate_HIPM);
#endif

PetscErrorCode PetscDeviceContextCreate_HIPM(PetscDeviceContext dctx)
{
  PetscDeviceContext_(HIP) *dci;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = (void *)dci;
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
