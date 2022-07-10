#if !defined(BENCHIMPL)
#define BENCHIMPL
#include <petsc/private/petscimpl.h>
#include <petscbench.h>

typedef struct _PetscBenchOps *PetscBenchOps;

struct _PetscBenchOps {
  PetscErrorCode (*setup)(PetscBench);
  PetscErrorCode (*run)(PetscBench);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscBench);
  PetscErrorCode (*destroy)(PetscBench);
  PetscErrorCode (*view)(PetscBench,PetscViewer);
  PetscErrorCode (*reset)(PetscBench);
};

struct _p_PetscBench {
  PETSCHEADER(struct _PetscBenchOps);
  void *data;
};

PETSC_EXTERN PetscErrorCode PetscBenchCreate(MPI_Comm,PetscBench*);

#endif
