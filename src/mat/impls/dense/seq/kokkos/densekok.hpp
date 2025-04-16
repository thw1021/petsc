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
using KokkosDenseMatrixType = typename Kokkos::View<MatScalarType**, Kokkos::LayoutRight, MemorySpace>;

using KokkosDenseMatrix     = KokkosDenseMatrixType<DefaultMemorySpace>;
using KokkosDenseMatrixHost = KokkosDenseMatrixType<HostMirrorMemorySpace>;

using MatScalarKokkosDenseView = KokkosDenseMatrix::non_const_type;

using MatScalarKokkosDenseViewHost = KokkosDenseMatrixHost::non_const_type;

using ConstMatScalarKokkosDenseView = KokkosDenseMatrix::const_type;

using ConstMatScalarKokkosDenseViewHost = KokkosDenseMatrixHost::const_type;

using MatScalarKokkosDenseDualView = Kokkos::DualView<MatScalarType **>;

//?
//using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowMapType, MatColIdxType, MatScalarType, DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;

using KokkosTeamMemberType = Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type;

/* For mat->spptr of a regular matrix */
struct Mat_SeqDenseKokkos {
  MatScalarKokkosDenseDualView m_dual;

  KokkosDenseMatrix densemat; /* The Dense matrix, used to call KK functions */

  /* TODO do I want create seqdensekokkos from seqdense? */
  Mat_SeqDenseKokkos(PetscInt nrows, PetscInt ncols)
  {
    densemat = KokkosDenseMatrix("densemat", nrows, ncols);
  }

  /* Construct a nrows by ncols matrix with given aseq on host. */
  Mat_SeqDenseKokkos(PetscInt nrows, PetscInt ncols, Mat_SeqDense *aseq, PetscBool copyValues = PETSC_TRUE)
  {
    auto exec = PetscGetKokkosExecutionSpace();

    MatScalarKokkosDenseViewHost a_h(aseq->v);

    auto a_d = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, exec, a_h);
    m_dual   = MatScalarKokkosDenseDualView(a_d, a_h);

    m_dual.modify_host();
    if (copyValues) m_dual.sync_device(exec);

   //TODO not copying values from aseq...
    densemat = KokkosDenseMatrix("densemat", nrows, ncols);
    //TODO something like this?
    //I dont understand memory pattern.. is the array on host or device?
    //Kokkos::parallel_for("FillMatrixFromFlat", rows * cols, KOKKOS_LAMBDA(int idx) {
    //int i = idx / cols;
    //int j = idx % cols;
    //matrix(i, j) = data[idx];
    // });
  }

  //TODO can we even create View2D with dual view?
  Mat_SeqDenseKokkos(PetscInt nrows, PetscInt ncols, MatScalarKokkosDenseDualView &m) : m_dual(m)
  {
    densemat = KokkosDenseMatrix("densemat", nrows, ncols);
    //TODO fill densemet with dualview?
  }

  MatScalarType *m_host_data() { return m_dual.view_host().data(); }

  MatScalarType *m_device_data() { return m_dual.view_device().data(); }

  PetscInt nrows() { return densemat.extent(0); }
  PetscInt ncols() { return densemat.extent(1); }

//  void SetDiagonal(const MatRowMapType *diag)
//  {
//    MatRowMapKokkosViewHost diag_h(const_cast<MatRowMapType *>(diag), nrows());
//    auto                    diag_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), diag_h);
//    diag_dual                      = MatRowMapKokkosDualView(diag_d, diag_h);
//  }

};

PETSC_INTERN PetscErrorCode MatSetSeqDenseKokkosWithDenseMatrix(Mat, Mat_SeqDenseKokkos *);

PETSC_INTERN PetscErrorCode MatSeqDenseKokkosSyncDevice(Mat);
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqDenseKokkos(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqDenseKokkosModifyDevice(Mat);

PETSC_INTERN PetscErrorCode MatSeqDenseGetKokkosView(Mat, MatScalarKokkosDenseView *);
PETSC_INTERN PetscErrorCode MatSeqDenseRestoreKokkosView(Mat, MatScalarKokkosDenseView *);
PETSC_INTERN PetscErrorCode MatSeqDenseGetKokkosViewWrite(Mat, MatScalarKokkosDenseView *);
PETSC_INTERN PetscErrorCode MatSeqDenseRestoreKokkosViewWrite(Mat, MatScalarKokkosDenseView *);
