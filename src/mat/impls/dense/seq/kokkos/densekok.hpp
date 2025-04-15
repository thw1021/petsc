#pragma once
#include <petsc_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc/private/kokkosimpl.hpp>
#include <../src/mat/impls/dense/seq/dense.h>
#include <string>

#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/seq/kokkos/veckokkosimpl.hpp> // for VecSeq_Kokkos

namespace
{
PETSC_NODISCARD inline decltype(auto) NoInit(std::string label)
{
  return Kokkos::view_alloc(Kokkos::WithoutInitializing, std::move(label));
}
} // namespace

using MatScalarType = PetscScalar;

template <class MemorySpace>
using KokkosDenseMatrixType = typename Kokkos::View<PetscScalar **, Kokkos::LayoutRight, MemorySpace>;

using KokkosDenseMatrix     = KokkosDenseMatrixType<DefaultMemorySpace>;
using KokkosDenseMatrixHost = KokkosDenseMatrixType<HostMirrorMemorySpace>;

using KokkosDenseMatrixView = KokkosDenseMatrix::non_const_type;

using KokkosDenseMatrixViewHost = KokkosDenseMatrixHost::non_const_type;

using ConstKokkosDenseMatrixView = KokkosDenseMatrix::const_type;

using ConstKokkosDenseMatrixViewHost = KokkosDenseMatrixHost::const_type;

using KokkosDenseMatrixDualView = Kokkos::DualView<MatScalarType *>;

//?
using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowMapType, MatColIdxType, MatScalarType, DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;

using KokkosTeamMemberType = Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type;

/* For mat->spptr of a regular matrix */
struct Mat_SeqDenseKokkos {
  KokkosDenseMatrixDualView m_dual;

  KokkosDenseMatrix densemat; /* The Dense matrix, used to call KK functions */

  /* TODO do I want create seqdensekokkos from seqdense? */
  Mat_SeqDenseKokkos(PetscInt nrows, PetscInt ncols)
  {
    densemat = KokkosDenseMatrix("densemat", nrows, ncols);
    Init();
  }

  //TODO can we even create View2D with dual view?
  Mat_SeqDenseKokkos(PetscInt nrows, PetscInt ncols, MatScalarKokkosDualView &m) : m_dual(m)
  {
    densemat = KokkosDenseMatrix("densemat", nrows, ncols, a.view_device(), i.view_device(), j.view_device());
    Init();
  }

  MatScalarType *a_host_data() { return a_dual.view_host().data(); }

  MatScalarType *a_device_data() { return a_dual.view_device().data(); }

  PetscInt nrows() { return densemat.extent(0); }
  PetscInt ncols() { return densemat.extent(1); }

  void SetDiagonal(const MatRowMapType *diag)
  {
    MatRowMapKokkosViewHost diag_h(const_cast<MatRowMapType *>(diag), nrows());
    auto                    diag_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), diag_h);
    diag_dual                      = MatRowMapKokkosDualView(diag_d, diag_h);
  }

  /* Shared init stuff */
  void Init(PetscObjectState nzstate = 0)
  {
    nonzerostate      = nzstate;
    transpose_updated = PETSC_FALSE;
    hermitian_updated = PETSC_FALSE;
  }
};

//tODO KokkosDenseMatrixView or MatScalarDenseKokkosView ?
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosSyncDevice(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqDenseKokkos(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosModifyDevice(Mat);

PETSC_INTERN PetscErrorCode MatSeqDenseGetKokkosView(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatSeqDenseRestoreKokkosView(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatSeqDenseGetKokkosViewWrite(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatSeqDenseRestoreKokkosViewWrite(Mat, MatScalarKokkosView *);
