static char help[] = "Testing MatCreateSeqAIJKokkosWithKokkosViews, MatCreateSeqAIJKokkosWithKokkosCsrMatrix and MatSetMPIAIJWithSplitSeqAIJ().\n\n";

#include <petscvec_kokkos.hpp>
#include <petscdevice.h>
#include <petscmat.h>
#include <petscmat_kokkos.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

int main(int argc, char **argv)
{
  Mat             A, B;
  PetscInt        i, j, column;
  PetscInt       *di, *dj, *oi, *oj, nd;
  const PetscInt *garray;
  PetscInt       *garray_host;
  PetscScalar    *oa, *da;
  PetscScalar     value;
  PetscRandom     rctx;
  PetscBool       equal, done;
  Mat             AA, AB;
  PetscMPIInt     size, rank;
  MatType         mat_type;

  // ~~~~~~~~~~~~~~~~~~~~~
  // This test shows the routines needed to build a kokkos matrix without preallocation
  // on the host
  // ~~~~~~~~~~~~~~~~~~~~~

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size > 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 2 or more processes");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create a mpiaij matrix for checking */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 0, NULL, 0, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetUp(A));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));

  for (i = 5 * rank; i < 5 * rank + 5; i++) {
    for (j = 0; j < 5 * size; j++) {
      PetscCall(PetscRandomGetValue(rctx, &value));
      column = (PetscInt)(5 * size * PetscRealPart(value));
      PetscCall(PetscRandomGetValue(rctx, &value));
      PetscCall(MatSetValues(A, 1, &i, 1, &column, &value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatMPIAIJGetSeqAIJ(A, &AA, &AB, &garray));
  PetscCall(MatGetRowIJ(AA, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&di, (const PetscInt **)&dj, &done));
  PetscCall(MatSeqAIJGetArray(AA, &da));
  PetscCall(MatGetRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&oi, (const PetscInt **)&oj, &done));
  PetscCall(MatSeqAIJGetArray(AB, &oa));

  Mat output_mat_local, output_mat_nonlocal;
  // Be careful about scope given the kokkos memory reference counts
  {
    // Local
    PetscScalarKokkosView a_local_d;
    PetscIntKokkosView    i_local_d;
    PetscIntKokkosView    j_local_d;

    // Nonlocal
    PetscScalarKokkosView a_nonlocal_d;
    PetscIntKokkosView    i_nonlocal_d;
    PetscIntKokkosView    j_nonlocal_d;

    // Create device memory
    PetscCallCXX(a_local_d = PetscScalarKokkosView("a_local_d", di[5]));
    PetscCallCXX(i_local_d = PetscIntKokkosView("i_local_d", AA->rmap->n + 1));
    PetscCallCXX(j_local_d = PetscIntKokkosView("j_local_d", di[5]));

    // Create non-local device memory
    PetscCallCXX(a_nonlocal_d = PetscScalarKokkosView("a_nonlocal_d", oi[5]));
    PetscCallCXX(i_nonlocal_d = PetscIntKokkosView("i_nonlocal_d", AB->rmap->n + 1));
    PetscCallCXX(j_nonlocal_d = PetscIntKokkosView("j_nonlocal_d", oi[5]));

    // ~~~~~~~~~~~~~~~~~~~~~
    // Could fill the aij on the device - we're just going to test
    // by copying in the existing host values
    // ~~~~~~~~~~~~~~~~~~~~~
    PetscScalarKokkosViewHost a_local_h;
    PetscIntKokkosViewHost    i_local_h;
    PetscIntKokkosViewHost    j_local_h;
    PetscScalarKokkosViewHost a_nonlocal_h;
    PetscIntKokkosViewHost    i_nonlocal_h;
    PetscIntKokkosViewHost    j_nonlocal_h;

    PetscCallCXX(a_local_h = PetscScalarKokkosViewHost(da, di[5]));
    PetscCallCXX(i_local_h = PetscIntKokkosViewHost(di, AA->rmap->n + 1));
    PetscCallCXX(j_local_h = PetscIntKokkosViewHost(dj, di[5]));
    PetscCallCXX(a_nonlocal_h = PetscScalarKokkosViewHost(oa, oi[5]));
    PetscCallCXX(i_nonlocal_h = PetscIntKokkosViewHost(oi, AB->rmap->n + 1));
    PetscCallCXX(j_nonlocal_h = PetscIntKokkosViewHost(oj, oi[5]));

    // Haven't specified an exec space so these will all be synchronous
    // and finish without a need to call fence after
    PetscCallCXX(Kokkos::deep_copy(a_local_d, a_local_h));
    PetscCallCXX(Kokkos::deep_copy(i_local_d, i_local_h));
    PetscCallCXX(Kokkos::deep_copy(j_local_d, j_local_h));
    PetscCallCXX(Kokkos::deep_copy(a_nonlocal_d, a_nonlocal_h));
    PetscCallCXX(Kokkos::deep_copy(i_nonlocal_d, i_nonlocal_h));
    PetscCallCXX(Kokkos::deep_copy(j_nonlocal_d, j_nonlocal_h));

    // ~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~
    // Test MatCreateSeqAIJKokkosWithKokkosViews
    // ~~~~~~~~~~~~~~~~~

    // We can create our local diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, AA->rmap->n, AA->cmap->n, i_local_d, j_local_d, a_local_d, &output_mat_local));

    // We can create our nonlocal diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, AA->rmap->n, AB->cmap->n, i_nonlocal_d, j_nonlocal_d, a_nonlocal_d, &output_mat_nonlocal));

    // Build our mpi kokkos matrix by passing in the local and
    // nonlocal kokkos matrices and the colmap
    // MatSetMPIAIJWithSplitSeqAIJ allows us to pass in B using local indices
    // as long as garray has the global indices in it
    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetSizes(B, 5, 5, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatGetType(A, &mat_type));
    PetscCall(MatSetType(B, mat_type));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));

    // The garray passed in has to be on the host, but it can be created
    // on device and copied to the host
    // We're just going to copy the existing host values here
    PetscCall(PetscMalloc1(AB->cmap->n, &garray_host));
    for (int i = 0; i < AB->cmap->n; i++) { garray_host[i] = garray[i]; }

    // Skip the compactification - this means almost nothing happens on the host
    PetscCall(MatSetMPIAIJWithSplitSeqAIJ(B, output_mat_local, output_mat_nonlocal, garray_host));

    PetscCall(MatEqual(A, B, &equal));
    PetscCall(MatRestoreRowIJ(AA, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&di, (const PetscInt **)&dj, &done));
    PetscCall(MatSeqAIJRestoreArray(AA, &da));
    PetscCall(MatRestoreRowIJ(AB, 0, PETSC_FALSE, PETSC_FALSE, &nd, (const PetscInt **)&oi, (const PetscInt **)&oj, &done));
    PetscCall(MatSeqAIJRestoreArray(AB, &oa));

    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateSeqAIJKokkosWithKokkosViews()");
    PetscCall(MatDestroy(&B));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~
    // Test MatCreateSeqAIJKokkosWithKokkosViews but giving it host views as input
    // ~~~~~~~~~~~~~~~~~
    // We can create our local diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, AA->rmap->n, AA->cmap->n, i_local_h, j_local_h, a_local_h, &output_mat_local));

    // We can create our nonlocal diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, AA->rmap->n, AB->cmap->n, i_nonlocal_h, j_nonlocal_h, a_nonlocal_h, &output_mat_nonlocal));

    // Build our mpi kokkos matrix by passing in the local and
    // nonlocal kokkos matrices and the colmap
    // MatSetMPIAIJWithSplitSeqAIJ allows us to pass in B using local indices
    // as long as garray has the global indices in it
    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetSizes(B, 5, 5, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatGetType(A, &mat_type));
    PetscCall(MatSetType(B, mat_type));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));

    // The garray passed in has to be on the host, but it can be created
    // on device and copied to the host
    // We're just going to copy the existing host values here
    PetscCall(PetscMalloc1(AB->cmap->n, &garray_host));
    for (int i = 0; i < AB->cmap->n; i++) { garray_host[i] = garray[i]; }

    // Skip the compactification - this means almost nothing happens on the host
    PetscCall(MatSetMPIAIJWithSplitSeqAIJ(B, output_mat_local, output_mat_nonlocal, garray_host));

    PetscCall(MatEqual(A, B, &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateSeqAIJKokkosWithKokkosViews()");
    PetscCall(MatDestroy(&B));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /* Free spaces */
    PetscCall(PetscRandomDestroy(&rctx));
    PetscCall(MatDestroy(&A));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: kokkos_kernels

  test:
    nsize: 2
    args: -mat_type aijkokkos
    requires: kokkos_kernels
    output_file: output/empty.out

TEST*/
