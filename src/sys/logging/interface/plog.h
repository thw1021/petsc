#if !defined(PLOG_H)
#define PLOG_H
#include <petscsys.h>

typedef struct _n_PetscLogGlobalNames *PetscLogGlobalNames;
struct _n_PetscLogGlobalNames {
  MPI_Comm     comm;
  PetscInt     count;
  const char **names;
  PetscInt    *global_to_local;
  PetscInt    *local_to_global;
};

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesCreate(MPI_Comm, PetscInt, const char **, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesDestroy(PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscStageLogCreateGlobalStageNames(MPI_Comm, PetscStageLog, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscStageLogCreateGlobalEventNames(MPI_Comm, PetscStageLog, PetscLogGlobalNames *);

PETSC_INTERN PetscErrorCode PetscStageLogDuplicate(PetscStageLog, PetscStageLog *);

#endif // define PLOG_H
