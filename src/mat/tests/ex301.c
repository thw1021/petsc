
static char help[] = "Tests for bugs in A->offloadmask consistency for GPU matrices\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j,rstart,rend,m = 3;
  PetscScalar    one = 1.0,zero = 0.0,negativeone = -1.0;
  PetscReal      norm;
  Vec            x,y;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  for (i=0; i<2; i++) {
    /* Create the matrix and set it to contain explicit zero entries on the diagonal. */
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*m,m*m);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = VecSet(x,one);CHKERRQ(ierr);
    ierr = VecSet(y,zero);CHKERRQ(ierr);
    ierr = MatDiagonalSet(A,y,INSERT_VALUES);CHKERRQ(ierr);

    /* Now set A to be the identity using various approaches.
     * Note that there may be other approaches that should be added here. */
    switch (i) {
    case 0:
      ierr = MatDiagonalSet(A,x,INSERT_VALUES);CHKERRQ(ierr);
      break;
    case 1:
      for (j=rstart; j<rend; j++) {
        ierr = MatSetValue(A,j,j,one,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      break;
    case 2:
      for (j=rstart; j<rend; j++) {
        ierr = MatSetValuesRow(A,j,&one);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    default:
      break;
    }

    /* Compute y <- A*x and verify that the difference between y and x is negligible, as it should be since A is the identity. */
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,negativeone,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
    if (norm > PETSC_SQRT_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Test %d: Norm of error is %g, but should be near 0.\n",i,(double)norm);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   testset:
     suffix: 1
     nsize: 1
     output_file:  output/ex301_1.out
     test:
       suffix: aijviennacl
       requires: viennacl
       args: -mat_type aijviennacl
     test:
       suffix: aijcusparse
       requires: cuda
       args: -mat_type aijcusparse
     test:
       suffix: aijhipsparse
       requires: hip
       args: -mat_type aijhipsparse

   testset:
     suffix: 2
     nsize: 2
     output_file:  output/ex301_2.out
     test:
       suffix: aijviennacl
       requires: viennacl
       args: -mat_type aijviennacl
     test:
       suffix: aijcusparse
       requires: cuda
       args: -mat_type aijcusparse
     test:
       suffix: aijhipsparse
       requires: hip
       args: -mat_type aijhipsparse

TEST*/
