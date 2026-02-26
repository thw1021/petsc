#include <petscmat.h>

int main(int argc, char **args)
{
  PetscViewer viewer_read, viewer_write, viewer_read2;
  Mat         adj_mat, adj_mat2, adj_aij_mat;
  PetscBool   adj_mats_equal;
  char        adj_mat_file[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  PetscCall(PetscStrncpy(adj_mat_file, "${PETSC_DIR}/share/petsc/datafiles/matrices/adj_mat", sizeof(adj_mat_file)));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, adj_mat_file, FILE_MODE_READ, &viewer_read));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &adj_aij_mat));
  // Binary file contains an AIJ matrix
  PetscCall(MatLoad(adj_aij_mat, viewer_read));
  PetscCall(MatConvert(adj_aij_mat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj_mat));
  // Now write out again as AIJ
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "adj_mat2", FILE_MODE_WRITE, &viewer_write));
  PetscCall(MatView(adj_mat, viewer_write));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "adj_mat2", FILE_MODE_READ, &viewer_read2));
  PetscCall(MatLoad(adj_aij_mat, viewer_read2));
  PetscCall(MatConvert(adj_aij_mat, MATMPIADJ, MAT_INITIAL_MATRIX, &adj_mat2));
  PetscCall(MatEqual(adj_mat, adj_mat2, &adj_mats_equal));
  PetscCheck(adj_mats_equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Adjacency matrix not reproduced");
  PetscCall(MatDestroy(&adj_mat));
  PetscCall(MatDestroy(&adj_mat2));
  PetscCall(MatDestroy(&adj_aij_mat));
  PetscCall(PetscViewerDestroy(&viewer_read));
  PetscCall(PetscViewerDestroy(&viewer_read2));
  PetscCall(PetscViewerDestroy(&viewer_write));
}

/*TEST

   test:
      nsize: 1
      output_file: output/empty.out

   test:
      suffix: 2
      nsize: 2
      output_file: output/empty.out

TEST*/
