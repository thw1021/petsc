static char help[] = "Tests MatPartitioning with PTScotch when a process owns no graph vertices.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat             adj;
  MatPartitioning part;
  IS              is;
  PetscInt        N = 2, start, end, n, nlocal, nz, v, i;
  PetscInt       *xadj, *adjncy, *vweights = NULL, *eweights = NULL;
  PetscMPIInt     rank, size;
  PetscBool       use_vertex_weights = PETSC_FALSE, use_edge_weights = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_vertex_weights", &use_vertex_weights, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_edge_weights", &use_edge_weights, NULL));

  /* Spread the vertices of a path graph over the processes. Two vertices on the three processes
     the tests below run on is the smallest case that reaches the bug: one process owns no vertices
     and no edges, so its weight arrays come back null from a zero-size allocation while the other
     processes supply real weights, which is the disagreement SCOTCH_dgraphBuild() rejects. Two
     processes must still own a vertex, since MatPartitioningApply_PTScotch_Private() partitions
     with the sequential SCOTCH_graphBuild() when every vertex lives on one process. */
  start = (N * rank) / size;
  end   = (N * (rank + 1)) / size;
  n     = end - start;

  PetscCall(PetscMalloc1(n + 1, &xadj));
  nz      = 0;
  xadj[0] = 0;
  for (i = 0; i < n; i++) {
    v = start + i;
    if (v > 0) nz++;
    if (v < N - 1) nz++;
    xadj[i + 1] = nz;
  }
  PetscCall(PetscMalloc1(nz, &adjncy));
  nz = 0;
  for (i = 0; i < n; i++) {
    v = start + i;
    if (v > 0) adjncy[nz++] = v - 1;
    if (v < N - 1) adjncy[nz++] = v + 1;
  }
  if (use_edge_weights && nz) {
    PetscCall(PetscMalloc1(nz, &eweights));
    for (i = 0; i < nz; i++) eweights[i] = 2;
  }
  if (use_vertex_weights && n) {
    PetscCall(PetscMalloc1(n, &vweights));
    for (i = 0; i < n; i++) vweights[i] = rank + 1;
  }

  /* MatCreateMPIAdj() takes ownership of xadj, adjncy and eweights, and
     MatPartitioningSetVertexWeights() takes ownership of vweights */
  PetscCall(MatCreateMPIAdj(PETSC_COMM_WORLD, n, N, xadj, adjncy, eweights, &adj));
  PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD, &part));
  if (use_edge_weights) PetscCall(MatPartitioningSetUseEdgeWeights(part, PETSC_TRUE));
  PetscCall(MatPartitioningSetAdjacency(part, adj));
  if (use_vertex_weights) PetscCall(MatPartitioningSetVertexWeights(part, vweights));
  PetscCall(MatPartitioningSetNParts(part, size));
  PetscCall(MatPartitioningSetFromOptions(part));
  PetscCall(MatPartitioningApply(part, &is));

  /* Which part each vertex lands in depends on the partitioner, but every process must be given
     the destination of each of its own vertices */
  PetscCall(ISGetLocalSize(is, &nlocal));
  PetscCheck(nlocal == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Partitioning has %" PetscInt_FMT " local indices, expected %" PetscInt_FMT, nlocal, n);

  PetscCall(ISDestroy(&is));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(MatDestroy(&adj));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 3
      requires: ptscotch
      args: -mat_partitioning_type ptscotch
      output_file: output/empty.out

      test:
         suffix: vertex_weights
         args: -use_vertex_weights 1

      test:
         suffix: edge_weights
         args: -use_edge_weights 1

      test:
         suffix: both_weights
         args: -use_vertex_weights 1 -use_edge_weights 1

TEST*/
