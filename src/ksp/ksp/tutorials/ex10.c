
static char help[] = "Solve a small system and a large system through preloading\n\
  Input arguments are:\n\
   -f0 <small_sys_binary> -f1 <large_sys_binary> \n\n";

/*T
   Concepts: KSP^basic parallel example
   Concepts: Mat^loading a binary matrix and vector;
   Concepts: PetscLog^preloading executable
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

typedef enum {
  RHS_FILE,
  RHS_ONE,
  RHS_RANDOM
} RHSType;
const char *const RHSTypes[] = {"FILE", "ONE", "RANDOM", "RHSType", "RHS_", NULL};

/* ATTENTION: this is the example used in the Profiling chaper of the PETSc manual,
   where we referenced its profiling stages, preloading and output etc.
   When you modify it, please make sure it is still consistent with the manual.
 */
int main(int argc,char **args)
{
  PetscErrorCode    ierr;
  Vec               x,b,b2;
  Mat               A;           /* linear system matrix */
  KSP               ksp;         /* Krylov subspace method context */
  PetscReal         norm;        /* norm of solution error */
  char              file[2][PETSC_MAX_PATH_LEN];
  PetscViewer       viewer;      /* viewer */
  PetscBool         flg,preload=PETSC_FALSE,same,trans=PETSC_FALSE;
  RHSType           rhstype = RHS_FILE;
  PetscInt          its,j,len,start,idx,n1,n2;
  const PetscScalar *val;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /*
     Determine files from which we read the two linear systems
     (matrix and right-hand-side vector).
  */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-trans",&trans,&flg));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file[0],sizeof(file[0]),&flg));
  if (flg) {
    CHKERRQ(PetscStrcpy(file[1],file[0]));
    preload = PETSC_FALSE;
  } else {
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f0",file[0],sizeof(file[0]),&flg));
    PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f0 or -f option");
    CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f1",file[1],sizeof(file[1]),&flg));
    if (!flg) preload = PETSC_FALSE;   /* don't bother with second system */
  }
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-rhs",RHSTypes,(PetscEnum*)&rhstype,NULL));

  /*
    To use preloading, one usually has code like the following:

    PetscPreLoadBegin(preload,"first stage);
      lines of code
    PetscPreLoadStage("second stage");
      lines of code
    PetscPreLoadEnd();

    The two macro PetscPreLoadBegin() and PetscPreLoadEnd() implicitly form a
    loop with maximal two iterations, depending whether preloading is turned on or
    not. If it is, either through the preload arg of PetscPreLoadBegin or through
    -preload command line, the trip count is 2, otherwise it is 1. One can use the
    predefined variable PetscPreLoadIt within the loop body to get the current
    iteration number, which is 0 or 1. If preload is turned on, the runtime doesn't
    do profiling for the first iteration, but it will do profiling for the second
    iteration instead.

    One can solve a small system in the first iteration and a large system in
    the second iteration. This process preloads the instructions with the small
    system so that more accurate performance monitoring (via -log_view) can be done
    with the large one (that actually is the system of interest).

    But in this example, we turned off preloading and duplicated the code for
    the large system. In general, it is a bad practice and one should not duplicate
    code. We do that because we want to show profiling stages for both the small
    system and the large system.
  */
  PetscPreLoadBegin(preload,"Load System 0");

  /*=========================
      solve a small system
    =========================*/

  /* open binary file. Note that we use FILE_MODE_READ to indicate reading from this file */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&viewer));

  /* load the matrix and vector; then destroy the viewer */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  switch (rhstype) {
  case RHS_FILE:
    /* Vectors in the file might a different size than the matrix so we need a
     * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
     * can create the correct size Vec. */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
    CHKERRQ(VecLoad(b,viewer));
    break;
  case RHS_ONE:
    CHKERRQ(MatCreateVecs(A,&b,NULL));
    CHKERRQ(VecSet(b,1.0));
    break;
  case RHS_RANDOM:
    CHKERRQ(MatCreateVecs(A,&b,NULL));
    CHKERRQ(VecSetRandom(b,NULL));
    break;
  }
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* if the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector
   */
  CHKERRQ(MatGetLocalSize(A,NULL,&n1));
  CHKERRQ(VecGetLocalSize(b,&n2));
  same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PETSC_COMM_WORLD));

  if (!same) { /* create a new vector b by padding the old one */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b2));
    CHKERRQ(VecSetSizes(b2,n1,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(b2));
    CHKERRQ(VecGetOwnershipRange(b,&start,NULL));
    CHKERRQ(VecGetLocalSize(b,&len));
    CHKERRQ(VecGetArrayRead(b,&val));
    for (j=0; j<len; j++) {
      idx = start+j;
      CHKERRQ(VecSetValues(b2,1,&idx,val+j,INSERT_VALUES));
    }
    CHKERRQ(VecRestoreArrayRead(b,&val));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecAssemblyBegin(b2));
    CHKERRQ(VecAssemblyEnd(b2));
    b    = b2;
  }
  CHKERRQ(VecDuplicate(b,&x));

  PetscPreLoadStage("KSPSetUp 0");
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));

  /*
    Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
    enable more precise profiling of setting up the preconditioner.
    These calls are optional, since both will be called within
    KSPSolve() if they haven't been called already.
  */
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSetUpOnBlocks(ksp));

  PetscPreLoadStage("KSPSolve 0");
  if (trans) CHKERRQ(KSPSolveTranspose(ksp,b,x));
  else       CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(KSPGetTotalIterations(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its));

  CHKERRQ(KSPGetResidualNorm(ksp,&norm));
  if (norm < 1.e-12) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-12\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %e\n",(double)norm));
  }

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));

  /*=========================
    solve a large system
    =========================*/
  /* the code is duplicated. Bad practice. See comments above */
  PetscPreLoadStage("Load System 1");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&viewer));

  /* load the matrix and vector; then destroy the viewer */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,viewer));
  switch (rhstype) {
  case RHS_FILE:
    /* Vectors in the file might a different size than the matrix so we need a
     * Vec whose size hasn't been set yet.  It'll get fixed below.  Otherwise we
     * can create the correct size Vec. */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
    CHKERRQ(VecLoad(b,viewer));
    break;
  case RHS_ONE:
    CHKERRQ(MatCreateVecs(A,&b,NULL));
    CHKERRQ(VecSet(b,1.0));
    break;
  case RHS_RANDOM:
    CHKERRQ(MatCreateVecs(A,&b,NULL));
    CHKERRQ(VecSetRandom(b,NULL));
    break;
  }
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatGetLocalSize(A,NULL,&n1));
  CHKERRQ(VecGetLocalSize(b,&n2));
  same = (n1 == n2)? PETSC_TRUE : PETSC_FALSE;
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,&same,1,MPIU_BOOL,MPI_LAND,PETSC_COMM_WORLD));

  if (!same) { /* create a new vector b by padding the old one */
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b2));
    CHKERRQ(VecSetSizes(b2,n1,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(b2));
    CHKERRQ(VecGetOwnershipRange(b,&start,NULL));
    CHKERRQ(VecGetLocalSize(b,&len));
    CHKERRQ(VecGetArrayRead(b,&val));
    for (j=0; j<len; j++) {
      idx = start+j;
      CHKERRQ(VecSetValues(b2,1,&idx,val+j,INSERT_VALUES));
    }
    CHKERRQ(VecRestoreArrayRead(b,&val));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecAssemblyBegin(b2));
    CHKERRQ(VecAssemblyEnd(b2));
    b    = b2;
  }
  CHKERRQ(VecDuplicate(b,&x));

  PetscPreLoadStage("KSPSetUp 1");
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  /*
    Here we explicitly call KSPSetUp() and KSPSetUpOnBlocks() to
    enable more precise profiling of setting up the preconditioner.
    These calls are optional, since both will be called within
    KSPSolve() if they haven't been called already.
  */
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSetUpOnBlocks(ksp));

  PetscPreLoadStage("KSPSolve 1");
  if (trans) CHKERRQ(KSPSolveTranspose(ksp,b,x));
  else       CHKERRQ(KSPSolve(ksp,b,x));

  CHKERRQ(KSPGetTotalIterations(ksp,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n",its));

  CHKERRQ(KSPGetResidualNorm(ksp,&norm));
  if (norm < 1.e-12) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-12\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm %e\n",(double)norm));
  }

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  PetscPreLoadEnd();
  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 1
      nsize: 4
      output_file: output/ex10_1.out
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/arco6 -ksp_gmres_classicalgramschmidt -mat_type baij -matload_block_size 3 -pc_type bjacobi

   test:
      TODO: Matrix row/column sizes are not compatible with block size
      suffix: 2
      nsize: 4
      output_file: output/ex10_2.out
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f0 ${DATAFILESPATH}/matrices/medium -f1 ${DATAFILESPATH}/matrices/arco6 -ksp_gmres_classicalgramschmidt -mat_type baij -matload_block_size 3 -pc_type bjacobi -trans

   test:
      suffix: 3
      requires: double complex !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/nh-complex-int32-float64 -ksp_type bicg

TEST*/
