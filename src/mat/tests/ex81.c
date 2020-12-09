
static char help[] = "Tests MatOption().\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B;
  Vec            diag;
  PetscInt       i,n = 10,col[3];
  PetscErrorCode ierr;
  PetscScalar    v[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* Create A which has empty 0-th column */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  v[0] = -1.; v[1] = 2.; v[2] = -1.;
  for (i=2; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr    = MatSetValues(A,1,&i,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  i    = 0; col[0] = 1; v[0] = -1;
  ierr = MatSetValues(A,1,&i,1,col,v,INSERT_VALUES);CHKERRQ(ierr);
  i    = 1; col[0] = 1; col[1] = 2;
  ierr = MatSetValues(A,1,&i,2,col,v+1,INSERT_VALUES);CHKERRQ(ierr);
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Compute B = A^t*A, B misses 0-th diagonal */
  //ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
  ierr = MatProductCreate(A,A,NULL,&B);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,"AtA_");CHKERRQ(ierr);
  ierr = MatProductSetType(B,MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(B,MATPRODUCTALGORITHM_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFill(B,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(B);CHKERRQ(ierr);

  ierr = MatProductSymbolic(B);CHKERRQ(ierr);
  ierr = MatProductNumeric(B);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  //ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  //ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);CHKERRQ(ierr);

  ierr = MatCreateVecs(B,NULL,&diag);CHKERRQ(ierr);
  ierr = MatGetDiagonal(B,diag);CHKERRQ(ierr);
  //ierr = VecView(diag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecSetValue(diag,0,100.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(diag);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diag);CHKERRQ(ierr);
  //ierr = VecView(diag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD," call MatDiagonalSet...\n");CHKERRQ(ierr);
  ierr = MatDiagonalSet(B,diag,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
