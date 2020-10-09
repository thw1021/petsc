
static char help[] = "Tests MatInvertVariableBlockDiagonalMat()\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,C,D;
  PetscErrorCode ierr;
  PetscInt       i,rows[2];
  PetscScalar    values[2][2];

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,6,18);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  values[0][0] = 2; values[0][1] = 1;
  values[1][0] = 1; values[1][1] = 2;
  for (i=0;i<3;i++){
    rows[0] = 2*i; rows[1] = 2*i + 1;
    ierr = MatSetValues(C,2,rows,2,rows,(PetscScalar*)values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DETERMINE,&A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&D);CHKERRQ(ierr);
  ierr = MatSetType(D,MATAIJ);CHKERRQ(ierr);
  ierr = MatInvertVariableBlockDiagonalMat(A,D);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
