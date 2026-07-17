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
  which side holds the up-to-date data; A->offloadmask is only used as a cheap "is this a Kokkos matrix"
  type tag (set to PETSC_OFFLOAD_KOKKOS).

  We assume the leading dimension equals the number of rows (lda == m), which is the default allocation for
  a dense matrix. This keeps the device mirror packed so that the 2D column-major views used for BLAS are
  plain LayoutLeft views. Custom leading dimensions are handled on the host through the SeqDense fallback.
*/

/* Column-major (LayoutLeft) 2D views built on demand over the DualView's raw pointers, for KokkosBlas calls */
using MatDenseKokkosView2D          = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ConstMatDenseKokkosView2D     = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using MatDenseKokkosView2DHost      = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, HostMirrorMemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/* For mat->spptr of a (non-factored) MATSEQDENSEKOKKOS matrix */
struct Mat_SeqDenseKokkos {
  PetscScalarKokkosDualView a_dual; /* length lda*n; host view aliases Mat_SeqDense->v */

  /* Wrap a host array (Mat_SeqDense->v, of length lda*n) in a DualView, creating a device mirror.
     If d_array is given, it is used as the device side (assumed to hold the same values). */
  Mat_SeqDenseKokkos(PetscInt lda, PetscInt n, PetscScalar *h_array, PetscScalar *d_array = nullptr)
  {
    PetscScalarKokkosViewHost a_h(h_array, (size_t)lda * n);
    PetscScalarKokkosView     a_d;

    if (d_array) a_d = PetscScalarKokkosView(d_array, (size_t)lda * n);
    else a_d = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, DefaultMemorySpace(), a_h);
    a_dual = PetscScalarKokkosDualView(a_d, a_h);
  }

  PetscScalar *DeviceData() { return a_dual.view_device().data(); }
  PetscScalar *HostData() { return a_dual.view_host().data(); }

  /* Column-major device view of the m x n matrix (packed, lda == m) */
  MatDenseKokkosView2D DeviceView2D(PetscInt m, PetscInt n) { return MatDenseKokkosView2D(DeviceData(), m, n); }
};

PETSC_INTERN PetscErrorCode MatSeqDenseKokkosSyncDevice(Mat);
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosModifyDevice(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqDenseKokkos(Mat, MatType, MatReuse, Mat *);

/* Overloaded accessors must have C++ linkage (not PETSC_INTERN/extern "C"), like MatSeqAIJGetKokkosView() */
PetscErrorCode MatSeqDenseGetKokkosView(Mat, ConstMatDenseKokkosView2D *);
PetscErrorCode MatSeqDenseRestoreKokkosView(Mat, ConstMatDenseKokkosView2D *);
PetscErrorCode MatSeqDenseGetKokkosView(Mat, MatDenseKokkosView2D *);
PetscErrorCode MatSeqDenseRestoreKokkosView(Mat, MatDenseKokkosView2D *);
PetscErrorCode MatSeqDenseGetKokkosViewWrite(Mat, MatDenseKokkosView2D *);
PetscErrorCode MatSeqDenseRestoreKokkosViewWrite(Mat, MatDenseKokkosView2D *);
