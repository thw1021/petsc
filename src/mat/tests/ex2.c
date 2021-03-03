
static char help[] = "Tests MatTranspose(), MatNorm(), MatAXPY() and MatAYPX().\n\n";

#include <petscmat.h>

static PetscErrorCode TransposeAXPY(Mat C,PetscScalar alpha,Mat mat,PetscErrorCode (*f)(Mat,Mat*))
{
  Mat            D,E,F,G;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (f == MatCreateTranspose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^T)^T = (C^T)^T + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^H)^H = (C^H)^H + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  }
  ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  ierr  = f(C,&D);CHKERRQ(ierr);
  ierr  = f(D,&E);CHKERRQ(ierr);
  ierr  = MatAXPY(E,alpha,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr  = MatDestroy(&E);CHKERRQ(ierr);
  ierr  = MatDestroy(&D);CHKERRQ(ierr);
  ierr  = MatDestroy(&C);CHKERRQ(ierr);
  if (f == MatCreateTranspose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  }
  ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  /* MATTRANSPOSE should have a MatTranspose_Transpose or MatTranspose_HT implementation */
  if (f == MatCreateTranspose) {
    ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  } else {
    ierr = MatHermitianTranspose(mat,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  }
  ierr  = f(D,&E);CHKERRQ(ierr);
  ierr  = MatAXPY(C,alpha,E,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr  = MatDestroy(&E);CHKERRQ(ierr);
  ierr  = MatDestroy(&D);CHKERRQ(ierr);
  ierr  = MatDestroy(&C);CHKERRQ(ierr);
  if (f == MatCreateTranspose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^T)^T = (C^T)^T + alpha * (A^T)^T, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  (C^H)^H = (C^H)^H + alpha * (A^H)^H, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
  }
  ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  ierr  = f(C,&D);CHKERRQ(ierr);
  ierr  = f(D,&E);CHKERRQ(ierr);
  ierr  = f(mat,&F);CHKERRQ(ierr);
  ierr  = f(F,&G);CHKERRQ(ierr);
  ierr  = MatAXPY(E,alpha,G,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr  = MatDestroy(&G);CHKERRQ(ierr);
  ierr  = MatDestroy(&F);CHKERRQ(ierr);
  ierr  = MatDestroy(&E);CHKERRQ(ierr);
  ierr  = MatDestroy(&D);CHKERRQ(ierr);
  ierr  = MatDestroy(&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            mat,tmat = 0;
  PetscInt       m = 7,n,i,j,rstart,rend,rect = 0;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscBool      flg;
  PetscScalar    v, alpha;
  PetscReal      normf,normi,norm1;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = m;
  ierr = PetscOptionsHasName(NULL,NULL,"-rectA",&flg);CHKERRQ(ierr);
  if (flg) {n += 2; rect = 1;}
  ierr = PetscOptionsHasName(NULL,NULL,"-rectB",&flg);CHKERRQ(ierr);
  if (flg) {n -= 2; rect = 1;}

  /* ------- Assemble matrix --------- */
  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    for (j=0; j<n; j++) {
      v    = 10.0*i+j;
      ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* ----------------- Test MatNorm()  ----------------- */
  ierr = MatNorm(mat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(mat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"original A: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* --------------- Test MatTranspose()  -------------- */
  ierr = PetscOptionsHasName(NULL,NULL,"-in_place",&flg);CHKERRQ(ierr);
  if (!rect && flg) {
    ierr = MatTranspose(mat,MAT_REUSE_MATRIX,&mat);CHKERRQ(ierr);   /* in-place transpose */
    tmat = mat; mat = 0;
  } else {      /* out-of-place transpose */
    ierr = MatTranspose(mat,MAT_INITIAL_MATRIX,&tmat);CHKERRQ(ierr);
  }

  /* ----------------- Test MatNorm()  ----------------- */
  /* Print info about transpose matrix */
  ierr = MatNorm(tmat,NORM_FROBENIUS,&normf);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_1,&norm1);CHKERRQ(ierr);
  ierr = MatNorm(tmat,NORM_INFINITY,&normi);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"B = A^T: Frobenious norm = %g, one norm = %g, infinity norm = %g\n",(double)normf,(double)norm1,(double)normi);CHKERRQ(ierr);
  ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* ----------------- Test MatAXPY(), MatAYPX()  ----------------- */
  if (mat && !rect) {
    alpha = 1.0;
    ierr  = PetscOptionsGetScalar(NULL,NULL,"-alpha",&alpha,NULL);CHKERRQ(ierr);
    ierr  = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A\n");CHKERRQ(ierr);
    ierr  = MatAXPY(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr  = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAYPX:  B = alpha*B + A\n");CHKERRQ(ierr);
    ierr = MatAYPX(tmat,alpha,mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(tmat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  {
    Mat C;
    alpha = 1.0;
    ierr  = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  C = C + alpha * A, C=A, SAME_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr  = MatDuplicate(mat,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
    ierr  = MatAXPY(C,alpha,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr  = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr  = MatDestroy(&C);CHKERRQ(ierr);
    ierr  = TransposeAXPY(C,alpha,mat,MatCreateTranspose);CHKERRQ(ierr);
    ierr  = TransposeAXPY(C,alpha,mat,MatCreateHermitianTranspose);CHKERRQ(ierr);
  }

  {
    Mat matB;
    /* get matB that has nonzeros of mat in all even numbers of row and col */
    ierr = MatCreate(PETSC_COMM_WORLD,&matB);CHKERRQ(ierr);
    ierr = MatSetSizes(matB,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
    ierr = MatSetFromOptions(matB);CHKERRQ(ierr);
    ierr = MatSetUp(matB);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(matB,&rstart,&rend);CHKERRQ(ierr);
    if (rstart % 2 != 0) rstart++;
    for (i=rstart; i<rend; i += 2) {
      for (j=0; j<n; j += 2) {
        v    = 10.0*i+j;
        ierr = MatSetValues(matB,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," A: original matrix:\n");CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," B(a subset of A):\n");CHKERRQ(ierr);
    ierr = MatView(matB,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"MatAXPY:  B = B + alpha * A, SUBSET_NONZERO_PATTERN\n");CHKERRQ(ierr);
    ierr = MatAXPY(mat,alpha,matB,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&matB);CHKERRQ(ierr);
  }

  /* Test MatZeroRows */
  j = rstart - 1;
  if (j < 0) j = m-1;
  ierr = MatZeroRows(mat,1,&j,0.0,NULL,NULL);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Free data structures */
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     suffix: 3
     filter: grep -v type | grep -v "Mat Object"
     nsize: 2
     args: -rectA
     output_file: output/ex2_3.out
     test:
       args: -mat_type mpiaij
     test:
       suffix: aijcusparse
       requires: cuda
       args: -mat_type mpiaijcusparse
     test:
       suffix: aijhipsparse
       requires: hip
       args: -mat_type mpiaijhipsparse
     test:
       args: -mat_type mpidense
     test:
       requires: cuda
       suffix: densecuda
       args: -mat_type mpidensecuda
     test:
       requires: hip
       suffix: densehip
       args: -mat_type mpidensehip

   testset:
     suffix: 12_A
     args: -rectA
     filter: grep -v type | grep -v "Mat Object"
     output_file: output/ex2_12_A.out
     test:
       args: -mat_type seqaij
     test:
       args: -mat_type seqdense
     test:
       requires: cuda
       suffix: aijcusparse
       args: -mat_type seqaijcusparse
     test:
       requires: cuda
       suffix: densecuda
       args: -mat_type seqdensecuda
     test:
       requires: kokkos_kernels
       suffix: kokkos
       args: -mat_type seqaijkokkos
     test:
       requires: hip
       suffix: seqaijhip
       args: -mat_type seqaijhip
     test:
       requires: hip
       suffix: seqdensehip
       args: -mat_type seqdensehip

   testset:
     args: -rectB
     suffix: 12_B
     filter: grep -v type | grep -v "Mat Object"
     output_file: output/ex2_12_B.out
     test:
       args: -mat_type seqaij
     test:
       args: -mat_type seqdense
     test:
       requires: cuda
       suffix: aijcusparse
       args: -mat_type seqaijcusparse
     test:
       requires: cuda
       suffix: seqdensecuda
       args: -mat_type seqdensecuda
     test:
       requires: hip
       suffix: aijhipsparse
       args: -mat_type seqaijhipsparse
     test:
       requires: hip
       suffix: seqdensehip
       args: -mat_type seqdensehip
     test:
       requires: kokkos_kernels
       suffix: kokkos
       args: -mat_type seqaijkokkos

   testset:
     suffix: 21
     filter: grep -v type | grep -v "MPI processes"
     output_file: output/ex2_21.out
     test:
       args: -mat_type mpiaij
     test:
       args: -mat_type mpidense
     test:
       requires: cuda
       suffix: mpidensecuda
       args: -mat_type mpidensecuda
     test:
       requires: hip
       suffix: mpidensehip
       args: -mat_type mpidensehip
     test:
       requires: kokkos_kernels
       suffix: kokkos
       args: -mat_type mpiaijkokkos
     test:
       suffix: aijcusparse
       requires: cuda
       args: -mat_type mpiaijcusparse
     test:
       suffix: aijhipsparse
       requires: hip
       args: -mat_type mpiaijhipsparse
     test:
       suffix: aijkokkos
       requires: kokkos_kernels
       args: -mat_type mpiaijkokkos

   testset:
     suffix: 23
     nsize: 3
     filter: grep -v type | grep -v "Mat Object"
     output_file: output/ex2_23.out
     test:
       args: -mat_type mpiaij
     test:
       args: -mat_type mpidense
     test:
       requires: cuda
       suffix: mpidensecuda
       args: -mat_type mpidensecuda
     test:
       suffix: aijcusparse
       requires: cuda
       args: -mat_type mpiaijcusparse
     test:
       suffix: aijhipsparse
       requires: hip
       args: -mat_type mpiaijhipsparse
     test:
       suffix: aijkokkos
       requires: kokkos_kernels
       args: -mat_type mpiaijkokkos
TEST*/
