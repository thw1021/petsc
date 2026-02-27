#include <petscmat.h>

int main(int argc, char **args)
{
  PetscViewer     viewer_read, viewer_write, viewer_read2;
  Mat             adj_mat, adj_mat2, adj_aij_mat, transpose;
  MatPartitioning part;
  PetscBool       mats_equal;
  char            adj_mat_file[PETSC_MAX_PATH_LEN];
  PetscMPIInt     size;
  IS              is;

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
  PetscCall(MatEqual(adj_mat, adj_mat2, &mats_equal));
  PetscCheck(mats_equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Adjacency matrix not reproduced");
  PetscCall(MatConvert(adj_mat2, MATAIJ, MAT_REUSE_MATRIX, &adj_aij_mat));
  PetscCall(MatTranspose(adj_aij_mat, MAT_INITIAL_MATRIX, &transpose));
  PetscCall(MatEqual(adj_aij_mat, transpose, &mats_equal));
  PetscCheck(mats_equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Adjacency matrix not symmetric");

  // Now let's make sure we can construct a partition from this
  PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
  PetscCall(MatPartitioningSetUseEdgeWeights(part, PETSC_TRUE));
  PetscCall(MatPartitioningSetAdjacency(part, adj_mat2));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(MatPartitioningSetNParts(part, size));
  PetscCall(MatPartitioningSetFromOptions(part));
  PetscCall(MatPartitioningApply(part, &is));
  PetscCall(MatDestroy(&adj_mat));
  PetscCall(MatDestroy(&adj_mat2));
  PetscCall(MatDestroy(&adj_aij_mat));
  PetscCall(MatDestroy(&transpose));
  PetscCall(PetscViewerDestroy(&viewer_read));
  PetscCall(PetscViewerDestroy(&viewer_read2));
  PetscCall(PetscViewerDestroy(&viewer_write));
  PetscCall(ISDestroy(&is));
}

/*TEST

   test:
      suffix: view_adj_parmetis
      requires: defined(PETSC_USE_64BIT_INDICES) parmetis
      nsize: 2
      output_file: output/empty.out
      args: -mat_partitioning_type parmetis

   test:
      suffix: view_adj_ptscotch
      requires: defined(PETSC_USE_64BIT_INDICES) ptscotch
      nsize: 2
      output_file: output/empty.out
      args: -mat_partitioning_type ptscotch

   test:
      suffix: view_adj_chaco
      requires: defined(PETSC_USE_64BIT_INDICES) chaco
      nsize: 2
      output_file: output/empty.out
      args: -mat_partitioning_type chaco

   test:
      suffix: view_adj_party
      requires: defined(PETSC_USE_64BIT_INDICES) party
      nsize: 2
      output_file: output/empty.out
      args: -mat_partitioning_type party

   test:
      suffix: view_adj_hierarch
      requires: defined(PETSC_USE_64BIT_INDICES) hierarch
      nsize: 2
      output_file: output/empty.out
      args: -mat_partitioning_type hierarch

TEST*/
