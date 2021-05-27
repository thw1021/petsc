#include <petscmat.h>
#if defined(PETSC_HAVE_HDF5)
#include <petscviewerhdf5.h>
#endif

#define NNORMS 6

static PetscErrorCode MatLoadComputeNorms(Mat data_mat, PetscViewer inp_viewer, PetscReal norms[])
{
  Mat            corr_mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLoad(data_mat, inp_viewer);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(data_mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(data_mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatViewFromOptions(data_mat, NULL, "-view_mat");CHKERRQ(ierr);

  /* compute matrix norms */
  ierr = MatNorm(data_mat, NORM_1, &norms[0]);CHKERRQ(ierr);
  ierr = MatNorm(data_mat, NORM_INFINITY, &norms[1]);CHKERRQ(ierr);
  ierr = MatNorm(data_mat, NORM_FROBENIUS, &norms[2]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Data matrix norms: %g %g %g\n", (double)norms[0],(double)norms[1],(double)norms[2]);CHKERRQ(ierr);

  /* compute autocorrelation matrix */
  ierr = MatMatTransposeMult(data_mat, data_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &corr_mat);CHKERRQ(ierr);

  /* compute autocorrelation matrix norms */
  ierr = MatNorm(corr_mat, NORM_1, &norms[3]);CHKERRQ(ierr);
  ierr = MatNorm(corr_mat, NORM_INFINITY, &norms[4]);CHKERRQ(ierr);
  ierr = MatNorm(corr_mat, NORM_FROBENIUS, &norms[5]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Autocorrelation matrix norms: %g %g %g\n", (double)norms[3],(double)norms[4],(double)norms[5]);CHKERRQ(ierr);

  ierr = MatDestroy(&corr_mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      norms0[NNORMS], norms1[NNORMS];
  PetscViewer    inp_viewer;
  Mat            data_mat;
  char           file[PETSC_MAX_PATH_LEN]="", mat_name[PETSC_MAX_PATH_LEN]="dmatrix";
  PetscBool      flg;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscOptionsGetString(NULL,NULL,"-mat_name",mat_name,sizeof(mat_name),NULL);CHKERRQ(ierr);

  /* load matrix sequentially */
  ierr = MatCreate(PETSC_COMM_SELF, &data_mat);CHKERRQ(ierr);
  ierr = MatSetType(data_mat,MATDENSE);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)data_mat, mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, file, FILE_MODE_READ, &inp_viewer);CHKERRQ(ierr);
  ierr = MatLoadComputeNorms(data_mat, inp_viewer, norms0);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&inp_viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&data_mat);CHKERRQ(ierr);

  /* load matrix in parallel */
  ierr = MatCreate(PETSC_COMM_WORLD, &data_mat);CHKERRQ(ierr);
  ierr = MatSetType(data_mat,MATDENSE);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)data_mat, mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, file, FILE_MODE_READ, &inp_viewer);CHKERRQ(ierr);
  ierr = MatLoadComputeNorms(data_mat, inp_viewer, norms1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&inp_viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&data_mat);CHKERRQ(ierr);

  for (i=0; i<NNORMS; i++) {
    if (norms0[i] != norms1[i]) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "norm0[%D] = %g != %g = norms1[%D]", i, norms0[i], norms1[i], i);
  }

  ierr = PetscFinalize();
  return ierr;
}

#undef NNORMS

/*TEST

   testset:
      suffix: 1
      requires: datafilespath complex
      args:  -f ${DATAFILESPATH}/matrices/hdf5/sample_data.h5
      test:
        nsize: 1
      test:
        TODO: broken
        suffix: 1p
        nsize: {{2 4}}

TEST*/
