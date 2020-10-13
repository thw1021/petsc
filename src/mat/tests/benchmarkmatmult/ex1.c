static char help[] = "Reads a matrix from the SuiteSparse Matrix Collection and vector from a file and tests the matrix-vector multiplication.\n\
Input arguments are:\n\
  -A <input_file> : file to load.  For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscmat.h>
#include <petscksp.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  PetscReal      norm,ratio,norm2;
  Vec            b,u,u2;
  Mat            A;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      flg,test_sell = PETSC_FALSE, verify_sell = PETSC_FALSE;
  PetscInt       maxslicewidth;
  PetscReal      avgslicewidth;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(NULL,NULL,"-A",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -A option");
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_sell",&test_sell,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-verify_sell",&verify_sell,NULL);CHKERRQ(ierr);
  if (verify_sell) test_sell = PETSC_TRUE; /* overwrite test_sell */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);

  /* Let the vec object trigger the first CUDA call, which takes a relatively long time to init CUDA */
  ierr = PetscOptionsGetString(NULL,NULL,"-b",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = VecLoad(b,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else {
    ierr = VecSetSizes(b,PETSC_DECIDE,m);CHKERRQ(ierr);
    ierr = VecSet(b,1.0);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);

  if (test_sell) {
    if (verify_sell) {
      Mat B;
      ierr = MatConvert(A,MATAIJCUSPARSE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      ierr = VecDuplicate(b,&u2);CHKERRQ(ierr);
      ierr = MatMult(B,b,u2);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
    }
    /* two-step convert is much faster than the basic convert */
    ierr = MatConvert(A,MATSELL,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatConvert(A,MATSELLCUDA,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatSeqSELLGetFillRatio(A,&ratio);CHKERRQ(ierr);
    ierr = MatSeqSELLGetMaxSliceWidth(A,&maxslicewidth);CHKERRQ(ierr);
    ierr = MatSeqSELLGetAvgSliceWidth(A,&avgslicewidth);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(A,MATAIJCUSPARSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  }
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  /* Timing MatMult */
  ierr = MatMult(A,b,u);CHKERRQ(ierr);

  /* Show result */
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  if (verify_sell) {
    ierr = VecAXPY(u2,-1,u);CHKERRQ(ierr);
    ierr = VecNorm(u2,NORM_2,&norm2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Relative error: %.4e\n", norm2/norm);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  if (!verify_sell) {
    PetscLogEvent      event;
    PetscEventPerfInfo eventInfo;
    PetscReal          gpuflopRate;

    if (test_sell) {
      ierr = PetscLogEventGetId("MatCUDACopyTo",&event);CHKERRQ(ierr);
    } else {
      ierr = PetscLogEventGetId("MatCUSPARSCopyTo",&event);CHKERRQ(ierr);
    }
    ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%.4e ", eventInfo.time);CHKERRQ(ierr);

    ierr = PetscLogEventGetId("MatMult",&event);CHKERRQ(ierr);
    ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &eventInfo);CHKERRQ(ierr);
    gpuflopRate = eventInfo.GpuFlops/eventInfo.GpuTime;
    if (test_sell) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%.2lf %.4e %.4e %.6lf %d %.2lf\n", gpuflopRate/1.e6,eventInfo.GpuTime,eventInfo.time,ratio,maxslicewidth,avgslicewidth);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%.2lf %.4e %.4e\n", gpuflopRate/1.e6,eventInfo.GpuTime,eventInfo.time);CHKERRQ(ierr);
    }
  }
  ierr = PetscFinalize();
  return ierr;
}
