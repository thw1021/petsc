#include "streamcuda.h"

#if PetscDefined(HAVE_CUDA)
PETSC_STATIC_INLINE PetscErrorCode PetscStreamGraphDestroy_CUDA(PetscStreamGraph sgraph)
{
  PetscStreamGraph_CUDA *psgc = (PetscStreamGraph_CUDA *)sgraph->data;
  PetscErrorCode        ierr;
  cudaError_t           cerr;

  PetscFunctionBegin;
  if (psgc->cexec) {cerr = cudaGraphExecDestroy(psgc->cexec);CHKERRCUDA(cerr);}
  if (psgc->cgraph) {cerr = cudaGraphDestroy(psgc->cgraph);CHKERRCUDA(cerr);}
  ierr = PetscFree(sgraph->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamGraphAssemble_CUDA(PetscStreamGraph sgraph, PetscGraphAssemblyType type)
{
  PetscStreamGraph_CUDA *psgc = (PetscStreamGraph_CUDA *)sgraph->data;
  cudaError_t           cerr;

  PetscFunctionBegin;
  switch (type) {
  case PETSC_GRAPH_UPDATE_ASSEMBLY:
    if (psgc->cexec) {
      cudaGraphUpdateResult update;

      cerr = cudaGraphExecUpdate(psgc->cexec, psgc->cgraph, NULL, &update);
      if (PetscUnlikely(cerr != cudaSuccess)) {
        /* "But this is code duplication!" you might say. And you'd be right. But NVIDIA did not come in peace. In their
       infinite malice they make this function return an opaque "something's gone wrong" errorcode but put the real
       errorcode in the "update" parameter. And just to make sure that no user is left only slightly inconvenienced,
       they provide no way to translate this esoteric new errorcode to text. */
        const char *name  = cudaGetErrorName(cerr);
        const char *descr = cudaGetErrorString(cerr);
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s, cudaGaph error %d, see cudaGraphExecUpdate() documentation for more information",(int)cerr,name,descr,(int)update);
      }
      break;
    }
  case PETSC_GRAPH_INIT_ASSEMBLY:
  {
    char[PETSC_MAX_PATH_LEN] ebuff = {0};

    if (psgc->cexec) {cerr = cudaGraphExecDestroy(psgc->cexec);CHKERRCUDA(cerr);}
    cerr = cudaGraphInstantiate(&psgc->cexec, psgc->cgraph, NULL, ebuff, PETSC_MAX_PATH_LEN);
    if (PetscUnlikely(cerr != cudaSucess)) {
      const char *name  = cudaGetErrorName(cerr);
      const char *descr = cudaGetErrorString(cerr);
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuda error %d (%s) : %s, cudaGaph error: %s",(int)cerr,name,descr,ebuff);
    }
  }
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamGraphExec_CUDA(PetscStreamGraph sgraph, PetscStream pstream)
{
  PetscStreamGraph_CUDA    *psgc = (PetscStreamGraph_CUDA *)sgraph->data;
  PetscStream_CUDA         *psc = (PetscStream_CUDA *)pstream->data;
  cudaError_t              cerr;

  PetscFunctionBegin;
  cerr = cudaGraphLaunc(psgc->cexec, psc->cstream);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamGetGraph_CUDA(PetscStreamGraph sgraph, void *gptr)
{
  PetscStreamGraph_CUDA *psgc = (PetscStreamGraph_CUDA *)sgraph->data;

  PetscFunctionBegin;
  *((cudaGraph_t*) gptr) = psgc->cgraph;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscStreamRestoreGraph_CUDA(PetscStreamGraph sgraph, void *gptr)
{
  PetscStreamGraph_CUDA *psgc = (PetscStreamGraph_CUDA *)sgraph->data;

  PetscFunctionBegin;
  if (PetscUnlikelyDebug(*((cudaGraph_t*) gptr) != psgc->cgraph)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"CUDA graph is not the same as the one that was checked out");
  }
  PetscFunctionReturn(0);
}

static const struct _GraphOps gcuops = {
  PetscStreamGraphCreate_CUDA,
  PetscStreamGraphDestroy_CUDA,
  NULL,
  PetscStreamGraphAssemble_CUDA,
  PetscStreamGraphExec_CUDA,
  PetscStreamGraphGetGraph_CUDA,
  PetscStreamGraphRestoreGraph_CUDA
};
#endif

PetscErrorCode PetscStreamGraphCreate_CUDA(PetscStreamGraph sgraph)
{
  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  {
    PetscStreamGraph_CUDA *psgc;
    PetscErrorCode        ierr;

    ierr = PetscNew(&psgc);CHKERRQ(ierr);
    psgc->cgraph = NULL;
    psgc->cexec  = NULL;
    sgraph->data = (void *)psgc;
    ierr = PetscMemcpy(sgraph->ops, &gcuops, sizeof(gcuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with CUDA support");
#endif
  PetscFunctionReturn(0);
}
