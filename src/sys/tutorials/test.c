
#include <petscvec.h>

PETSC_EXTERN PetscErrorCode __petsc_single_PetscInitialize(int *, char ***, const char [], const char []);
PETSC_EXTERN PetscErrorCode __petsc_single_PetscFinalize(void);
PETSC_EXTERN MPI_Comm __petsc_single_PETSC_COMM_WORLD;

typedef struct __petsc_single_p_Vec * __petsc_single_Vec;
PETSC_EXTERN PetscErrorCode __petsc_single_VecCreateMPI(MPI_Comm, PetscInt, PetscInt, __petsc_single_Vec *);
PETSC_EXTERN PetscErrorCode __petsc_single_VecDestroy(__petsc_single_Vec *);
PETSC_EXTERN PetscErrorCode __petsc_single_VecGetArrayWrite(__petsc_single_Vec, float **);
PETSC_EXTERN PetscErrorCode __petsc_single_VecRestoreArrayWrite(__petsc_single_Vec, float **);
PETSC_EXTERN PetscErrorCode __petsc_single_VecGetArrayRead(__petsc_single_Vec, const float **);
PETSC_EXTERN PetscErrorCode __petsc_single_VecRestoreArrayRead(__petsc_single_Vec, const float **);

typedef struct __petsc_single_p_PetscViewer * __petsc_single_PetscViewer;
PETSC_EXTERN __petsc_single_PetscViewer __petsc_single_PETSC_VIEWER_STDOUT_(MPI_Comm comm);

PETSC_EXTERN PetscErrorCode __petsc_single_VecView(__petsc_single_Vec, __petsc_single_PetscViewer);
PETSC_EXTERN PetscErrorCode __petsc_single_VecSqrtAbs(__petsc_single_Vec);
PETSC_EXTERN PetscErrorCode __petsc_single_VecExp(__petsc_single_Vec);
PETSC_EXTERN PetscErrorCode __petsc_single_VecLog(__petsc_single_Vec);


int main(int argc, char **argv)
{
  PetscRandom rand;
  Vec         x_double, diff;
  __petsc_single_Vec x_single;
  MPI_Comm    comm_double, comm_single;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(__petsc_single_PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscRegisterFinalize(__petsc_single_PetscFinalize));

  comm_double = PETSC_COMM_WORLD;
  comm_single = __petsc_single_PETSC_COMM_WORLD;

  PetscCall(PetscRandomCreate(comm_double, &rand));

  PetscCall(VecCreateMPI(comm_double, PETSC_DECIDE, 10, &x_double));
  PetscCall(__petsc_single_VecCreateMPI(comm_single, PETSC_DECIDE, 10, &x_single));
  PetscCall(VecSetRandom(x_double, rand));

  {
    const PetscScalar *_x_double;
    float *_x_single;
    PetscInt n_local;

    PetscCall(VecGetLocalSize(x_double, &n_local));
    PetscCall(VecGetArrayRead(x_double, &_x_double));
    PetscCall(__petsc_single_VecGetArrayWrite(x_single, &_x_single));
    for (PetscInt i = 0; i < n_local; i++) _x_single[i] = _x_double[i];
    PetscCall(__petsc_single_VecRestoreArrayWrite(x_single, &_x_single));
    PetscCall(VecRestoreArrayRead(x_double, &_x_double));
  }

  PetscCall(VecExp(x_double));
  PetscCall(__petsc_single_VecExp(x_single));
  PetscCall(VecLog(x_double));
  PetscCall(__petsc_single_VecLog(x_single));

  PetscCall(VecDuplicate(x_double, &diff));
  PetscCall(VecCopy(x_double, diff));
  {
    PetscScalar *_x_double;
    const float *_x_single;
    PetscInt           n_local;

    PetscCall(VecGetLocalSize(x_double, &n_local));
    PetscCall(VecGetArray(diff, &_x_double));
    PetscCall(__petsc_single_VecGetArrayRead(x_single, &_x_single));
    for (PetscInt i = 0; i < n_local; i++) _x_double[i] -= _x_single[i];
    PetscCall(__petsc_single_VecRestoreArrayRead(x_single, &_x_single));
    PetscCall(VecRestoreArray(diff, &_x_double));
  }

  PetscCall(VecView(x_double, PETSC_VIEWER_STDOUT_(comm_double)));
  PetscCall(__petsc_single_VecView(x_single, __petsc_single_PETSC_VIEWER_STDOUT_(comm_single)));

  PetscCall(VecView(diff, PETSC_VIEWER_STDOUT_(comm_double)));

  PetscCall(VecDestroy(&diff));
  PetscCall(__petsc_single_VecDestroy(&x_single));
  PetscCall(VecDestroy(&x_double));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}
