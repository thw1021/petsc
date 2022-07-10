
#if !defined(PETSCBENCH_H)
#define PETSCBENCH_H

#include <petscsys.h>

/* SUBMANSEC = Sys */

PETSC_EXTERN PetscClassId PETSC_BENCH_CLASSID;

/*S
     PetscBench - PETSc object that manages a benchmark

   Level: intermediate

.seealso: `PetscBenchCompute()`, `PetscBenchView()`, `PetscBenchDestroy()`, `PetscBenchSetFromOptions()`
S*/
typedef struct _p_PetscBench* PetscBench;

PETSC_EXTERN PetscErrorCode PetscBenchDestroy(PetscBench*);
PETSC_EXTERN PetscErrorCode PetscBenchRun(PetscBench);
PETSC_EXTERN PetscErrorCode PetscBenchSetFromOptions(PetscBench);
PETSC_EXTERN PetscErrorCode PetscBenchView(PetscBench,PetscViewer);

PETSC_EXTERN PetscErrorCode PetscBenchVecStreamsCreate(MPI_Comm,PetscBench*);

#endif
