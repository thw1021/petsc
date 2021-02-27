#if !defined(__STREAMCUDA_H)
#define __STREAMCUDA_H

#include <petsc/private/deviceimpl.h>

#if PetscDefined(HAVE_CUDA)
typedef struct {
  cudaStream_t cstream;
} PetscStream_CUDA;

PETSC_INTERN PetscErrorCode PetscStreamDestroy_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamSetUp_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamGetStream_CUDA(PetscStream,void*);
PETSC_INTERN PetscErrorCode PetscStreamRestoreStream_CUDA(PetscStream,void*);
PETSC_INTERN PetscErrorCode PetscStreamRecordEvent_CUDA(PetscStream,PetscEvent);
PETSC_INTERN PetscErrorCode PetscStreamWaitEvent_CUDA(PetscStream,PetscEvent);
PETSC_INTERN PetscErrorCode PetscStreamSynchronize_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamQuery_CUDA(PetscStream,PetscBool*);
PETSC_INTERN PetscErrorCode PetscStreamCaptureBegin_CUDA(PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamCaptureEnd_CUDA(PetscStream,PetscStreamGraph);

typedef struct {
  cudaEvent_t  cevent;
} PetscEvent_CUDA;

PETSC_INTERN PetscErrorCode PetscEventDestroy_CUDA(PetscEvent);
PETSC_INTERN PetscErrorCode PetscEventSetup_CUDA(PetscEvent);
PETSC_INTERN PetscErrorCode PetscEventSynchronize_CUDA(PetscEvent);
PETSC_INTERN PetscErrorCode PetscEventQuery_CUDA(PetscEvent,PetscBool*);

PETSC_INTERN PetscErrorCode PetscStreamScalarDestroy_CUDA(PetscStreamScalar);
PETSC_INTERN PetscErrorCode PetscStreamScalarSetup_CUDA(PetscStreamScalar);
PETSC_INTERN PetscErrorCode PetscStreamScalarSetValue_CUDA(PetscStreamScalar,const PetscScalar*,PetscMemType,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarAwait_CUDA(PetscStreamScalar,PetscScalar*,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarUpdateDevice_CUDA_Internal(PetscStreamScalar,PetscBool,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarGetDevice_CUDA(PetscStreamScalar,PetscScalar**,PetscBool,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarAXTY_CUDA(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarAYDX_CUDA(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarFinalize_CUDA(void);

typedef struct {
  cudaGraph_t     cgraph;
  cudaGraphExec_t cexec;
} PetscStreamGraph_CUDA;

PETSC_INTERN PetscErrorCode PetscStreamGraphDestroy_CUDA(PetscStreamGraph);
PETSC_INTERN PetscErrorCode PetscStreamGraphAssemble_CUDA(PetscStreamGraph,PetscGraphAssemblyType);
PETSC_INTERN PetscErrorCode PetscStreamGraphExec_CUDA(PetscStreamGraph,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamGraphDuplicate_CUDA(PetscStreamGraph,PetscStreamGraph);
PETSC_INTERN PetscErrorCode PetscStreamGraphGetGraph_CUDA(PetscStreamGraph,void*);
PETSC_INTERN PetscErrorCode PetscStreamGraphRestoreGraph_CUDA(PetscStreamGraph,void*);

PETSC_INTERN PetscErrorCode PetscStreamScalarAccumOpDispatch_Internal(PetscStreamScalar,PetscInt,PetscStreamScalar[],PetscStreamComputeOp,PetscStreamComputeOp,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarAXTY_CUDA_Kernel(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
PETSC_INTERN PetscErrorCode PetscStreamScalarAYDX_CUDA_Kernel(PetscScalar,PetscStreamScalar,PetscStreamScalar,PetscStream);
#endif /* HAVE_CUDA */
#endif
