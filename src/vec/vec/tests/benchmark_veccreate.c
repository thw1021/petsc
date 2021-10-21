
static char help[] = "Benchmark VecCreate() for GPU vectors.\n\
  -n <length> : vector length\n\n";

#include <petscvec.h>
#include <petsctime.h>
#include <petscdevice.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 5, iter = 10;
  PetscReal      value,rvalue = 0.0;
  Vec            x;
  PetscScalar    one = 1.0;
  PetscLogDouble v0,v1,v2,total_time = 0.0,create_time = 0.0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-iter",&iter,NULL);CHKERRQ(ierr);

  for (i=0; i<iter; i++) {
    ierr = PetscTime(&v0);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);
    ierr = WaitForCUDA();CHKERRQ(ierr);
    ierr = PetscTime(&v1);CHKERRQ(ierr);

    ierr = VecSet(x,one);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,&value);CHKERRQ(ierr);
    ierr = WaitForCUDA();CHKERRQ(ierr);
    ierr = PetscTime(&v2);CHKERRQ(ierr);
    create_time += v1-v0;
    total_time += v2-v0;
    rvalue += value;
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vec Norm = %g\nCreate Time = %g s\nTotal Time= %g s\n",(double)(rvalue/iter),(double)(create_time/iter),(double)(total_time/iter));CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      diff_args: -j
      filter: grep -v type
      output_file: output/ex21_1.out

      test:
         suffix: 1

      test:
         requires: cuda
         suffix: 1_cuda
         args: -vec_type cuda

      test:
         requires: kokkos_kernels
         suffix: 1_kokkos
         args: -vec_type kokkos

      test:
         requires: hip
         suffix: 1_hip
         args: -vec_type hip

   testset:
      diff_args: -j
      filter: grep -v type
      output_file: output/ex21_2.out
      nsize: 2

      test:
         suffix: 2

      test:
         requires: cuda
         suffix: 2_cuda
         args: -vec_type cuda

      test:
         requires: kokkos_kernels
         suffix: 2_kokkos
         args: -vec_type kokkos

      test:
         requires: hip
         suffix: 2_hip
         args: -vec_type hip

TEST*/
