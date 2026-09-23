static char help[] = "Tests VecCUDAGetArray() and friends along with OpenMP target offload\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  const PetscInt     n = 4;
  PetscScalar       *a;
  const PetscScalar *ar;
  Vec                x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetType(x, VECCUDA));

  // Write access hands back uninitialized device memory; fill it on the GPU.
  PetscCall(VecCUDAGetArrayWrite(x, &a));
#pragma omp target teams distribute parallel for is_device_ptr(a)
  for (PetscInt i = 0; i < n; i++) a[i] = 2.0;
  PetscCall(VecCUDARestoreArrayWrite(x, &a));

  // Read-write access: update the buffer in place on the GPU.
  PetscCall(VecCUDAGetArray(x, &a));
#pragma omp target teams distribute parallel for is_device_ptr(a)
  for (PetscInt i = 0; i < n; i++) a[i] = a[i] + 1.0;
  PetscCall(VecCUDARestoreArray(x, &a));

  // Read-only access round-trip.
  PetscCall(VecCUDAGetArrayRead(x, &ar));
  PetscCall(VecCUDARestoreArrayRead(x, &ar));

  PetscCall(PetscObjectSetName((PetscObject)x, "x"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: cuda defined(PETSC_HAVE_OPENMP_TARGET_OFFLOAD_CC)

TEST*/
