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

PETSC_STATIC_INLINE PetscErrorCode PetscStreamGraphAssemble_CUDA(PetscStreamGraph sgraph)
{
  PetscStreamGraph_CUDA    *psgc = (PetscStreamGraph_CUDA *)sgraph->data;
  cudaError_t              cerr;
  char[PETSC_MAX_PATH_LEN] ebuff = {0};
  /* No idea what this does, but apparently it returns error codes */
  cudaGraphNode_t          pError;

  PetscFunctionBegin;
  if (psgc->cexec) {cerr = cudaGraphExecDestroy(psgc->cexec);CHKERRCUDA(cerr);}
  cerr = cudaGraphInstantiate(&psgc->cexec, psgc->cgraph, &pError, ebuff, PETSC_MAX_PATH_LEN);CHKERRCUDA(cerr);
  if (PetscUnlikely(ebuff[0])) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"CUDA Graph error: %s",ebuff);
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
    sgraph->data = (void *)psgc;
    ierr = PetscMemcpy(sgraph->ops, &gcuops, sizeof(gcuops));CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PETSc is not configured with CUDA support");
#endif
  PetscFunctionReturn(0);
}
