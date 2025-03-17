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
    MatScalarKokkosDualView a_local_dual;
    MatRowMapKokkosDualView i_local_dual;
    MatColIdxKokkosDualView j_local_dual;
    MatScalarKokkosView     a_local_d;
    MatRowMapKokkosView     i_local_d;
    MatColIdxKokkosView     j_local_d;

    // Nonlocal
    MatScalarKokkosDualView a_nonlocal_dual;
    MatRowMapKokkosDualView i_nonlocal_dual;
    MatColIdxKokkosDualView j_nonlocal_dual;
    MatScalarKokkosView     a_nonlocal_d;
    MatRowMapKokkosView     i_nonlocal_d;
    MatColIdxKokkosView     j_nonlocal_d;

    // Create device & host memory
    PetscCallCXX(a_local_dual = MatScalarKokkosDualView("a_local_dual", di[5]));
    PetscCallCXX(i_local_dual = MatRowMapKokkosDualView("i_local_dual", AA->rmap->n + 1));
    PetscCallCXX(j_local_dual = MatColIdxKokkosDualView("j_local_dual", di[5]));

    // Get device views
    PetscCallCXX(a_local_d = a_local_dual.view_device());
    PetscCallCXX(i_local_d = i_local_dual.view_device());
    PetscCallCXX(j_local_d = j_local_dual.view_device());

    // Create non-local host and device memory
    PetscCallCXX(a_nonlocal_dual = MatScalarKokkosDualView("a_nonlocal_dual", oi[5]));
    PetscCallCXX(i_nonlocal_dual = MatRowMapKokkosDualView("i_nonlocal_dual", AB->rmap->n + 1));
    PetscCallCXX(j_nonlocal_dual = MatColIdxKokkosDualView("j_nonlocal_dual", oi[5]));

    // Get device views
    PetscCallCXX(a_nonlocal_d = a_nonlocal_dual.view_device());
    PetscCallCXX(i_nonlocal_d = i_nonlocal_dual.view_device());
    PetscCallCXX(j_nonlocal_d = j_nonlocal_dual.view_device());

    // ~~~~~~~~~~~~~~~~~~~~~
    // Could fill the aij on the device - we're just going to test
    // by copying in the existing host values
    // ~~~~~~~~~~~~~~~~~~~~~
    MatScalarKokkosViewHost a_local_h;
    MatRowMapKokkosViewHost i_local_h;
    MatColIdxKokkosViewHost j_local_h;
    MatScalarKokkosViewHost a_nonlocal_h;
    MatRowMapKokkosViewHost i_nonlocal_h;
    MatColIdxKokkosViewHost j_nonlocal_h;

    PetscCallCXX(a_local_h = PetscScalarKokkosViewHost(da, di[5]));
    PetscCallCXX(i_local_h = MatRowMapKokkosViewHost(di, AA->rmap->n + 1));
    PetscCallCXX(j_local_h = MatColIdxKokkosViewHost(dj, di[5]));
    PetscCallCXX(a_nonlocal_h = PetscScalarKokkosViewHost(oa, oi[5]));
    PetscCallCXX(i_nonlocal_h = MatRowMapKokkosViewHost(oi, AB->rmap->n + 1));
    PetscCallCXX(j_nonlocal_h = MatColIdxKokkosViewHost(oj, oi[5]));

    PetscCallCXX(Kokkos::deep_copy(a_local_d, a_local_h));
    PetscCallCXX(Kokkos::deep_copy(i_local_d, i_local_h));
    PetscCallCXX(Kokkos::deep_copy(j_local_d, j_local_h));
    PetscCallCXX(Kokkos::deep_copy(a_nonlocal_d, a_nonlocal_h));
    PetscCallCXX(Kokkos::deep_copy(i_nonlocal_d, i_nonlocal_h));
    PetscCallCXX(Kokkos::deep_copy(j_nonlocal_d, j_nonlocal_h));

    // ~~~~~~~~~~~~~~~~~~~~~

    // Have to specify that we've modified the device data
    PetscCallCXX(a_local_dual.modify_device());
    PetscCallCXX(i_local_dual.modify_device());
    PetscCallCXX(j_local_dual.modify_device());
    PetscCallCXX(a_nonlocal_dual.modify_device());
    PetscCallCXX(i_nonlocal_dual.modify_device());
    PetscCallCXX(j_nonlocal_dual.modify_device());

    // ~~~~~~~~~~~~~~~~~
    // Test MatCreateSeqAIJKokkosWithKokkosViews
    // ~~~~~~~~~~~~~~~~~

    // We can create our local diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, AA->rmap->n, AA->cmap->n, i_local_dual, j_local_dual, a_local_dual, &output_mat_local));

    // We can create our nonlocal diagonal block matrix directly on the device
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, AA->rmap->n, AB->cmap->n, i_nonlocal_dual, j_nonlocal_dual, a_nonlocal_dual, &output_mat_nonlocal));

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
    // Also test MatCreateSeqAIJKokkosWithKokkosCsrMatrix
    // ~~~~~~~~~~~~~~~~~
    // We can create our local diagonal block matrix directly on the device
    KokkosCsrMatrix csrmat_local, csrmat_nonlocal;
    PetscCallCXX(csrmat_local = KokkosCsrMatrix("csrmat_local", AA->rmap->n, AA->cmap->n, a_local_dual.extent(0), a_local_dual.view_device(), i_local_dual.view_device(), j_local_dual.view_device()));
    PetscCall(MatCreateSeqAIJKokkosWithKokkosCsrMatrix(PETSC_COMM_SELF, csrmat_local, &output_mat_local));

    // We can create our nonlocal diagonal block matrix directly on the device
    PetscCallCXX(csrmat_nonlocal = KokkosCsrMatrix("csrmat_nonlocal", AA->rmap->n, AB->cmap->n, a_nonlocal_dual.extent(0), a_nonlocal_dual.view_device(), i_nonlocal_dual.view_device(), j_nonlocal_dual.view_device()));
    PetscCall(MatCreateSeqAIJKokkosWithKokkosCsrMatrix(PETSC_COMM_SELF, csrmat_nonlocal, &output_mat_nonlocal));

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
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Likely a bug in MatCreateSeqAIJKokkosWithKokkosCsrMatrix()");
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
