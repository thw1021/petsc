#pragma once
#include <petsc_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc/private/kokkosimpl.hpp>
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscvec_kokkos.hpp>

/*
  MATSEQDENSEKOKKOS stores its values in a Kokkos::DualView<PetscScalar*> whose host view aliases the
  underlying Mat_SeqDense->v array (conventional column-major dense storage). The device side is a mirror
  in the Kokkos default memory space. The DualView's modify/sync flags are the single source of truth for
  which side holds the up-to-date data; A->offloadmask is set to PETSC_OFFLOAD_KOKKOS like the other
  Kokkos matrix types and does not track freshness.

  We require the leading dimension to equal the number of rows (lda == m), which is the default allocation
  for a dense matrix. This keeps the device mirror packed so that the 2D column-major views used for BLAS
  are plain LayoutLeft views. MatDenseSetLDA() with lda != m raises PETSC_ERR_SUP for this type, except
  when m == 0 (there is no data, and LAPACK's max(1,m) convention makes lda = 1 with m = 0 common).
*/

/* Column-major (LayoutLeft) 2D view built on demand over the DualView's raw pointers, for KokkosBlas calls */
using MatDenseKokkosView2D = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/* For mat->spptr of a (non-factored) MATSEQDENSEKOKKOS matrix */
struct Mat_SeqDenseKokkos {
  PetscScalarKokkosDualView a_dual; /* length m*n (packed, lda == m); host view aliases Mat_SeqDense->v */

  /* Wrap a host array (Mat_SeqDense->v, of length at least m*n) in a DualView, creating a device mirror */
  Mat_SeqDenseKokkos(PetscInt m, PetscInt n, PetscScalar *h_array)
  {
    PetscScalarKokkosViewHost a_h(h_array, (size_t)m * n);
    PetscScalarKokkosView     a_d = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, DefaultMemorySpace(), a_h);

    a_dual = PetscScalarKokkosDualView(a_d, a_h);
  }

  PetscScalar *DeviceData() { return a_dual.view_device().data(); }
  PetscScalar *HostData() { return a_dual.view_host().data(); }

  /* Column-major device view of the m x n matrix (packed, lda == m) */
  MatDenseKokkosView2D DeviceView2D(PetscInt m, PetscInt n) { return MatDenseKokkosView2D(DeviceData(), m, n); }
};

PETSC_INTERN PetscErrorCode MatSeqDenseKokkosSyncDevice(Mat);
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosModifyDevice(Mat);
