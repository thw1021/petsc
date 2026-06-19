#pragma once
#include <petsc_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc/private/kokkosimpl.hpp>
#include <../src/mat/impls/baij/seq/baij.h>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosKernels_Handle.hpp>
#include <string>

using MatRowMapType = PetscInt;
using MatColIdxType = PetscInt;
using MatScalarType = PetscScalar;

template <class MemorySpace>
using KokkosCsrMatrixType = typename KokkosSparse::CrsMatrix<MatScalarType, MatColIdxType, MemorySpace, void /* MemoryTraits */, MatRowMapType>;
template <class MemorySpace>
using KokkosCsrGraphType = typename KokkosCsrMatrixType<MemorySpace>::staticcrsgraph_type;

using KokkosCsrGraph     = KokkosCsrGraphType<DefaultMemorySpace>;
using KokkosCsrGraphHost = KokkosCsrGraphType<HostMirrorMemorySpace>;

using KokkosCsrMatrix     = KokkosCsrMatrixType<DefaultMemorySpace>;
using KokkosCsrMatrixHost = KokkosCsrMatrixType<HostMirrorMemorySpace>;

using MatRowMapKokkosView = KokkosCsrGraph::row_map_type::non_const_type;
using MatColIdxKokkosView = KokkosCsrGraph::entries_type::non_const_type;
using MatScalarKokkosView = KokkosCsrMatrix::values_type::non_const_type;

using MatRowMapKokkosViewHost = KokkosCsrGraphHost::row_map_type::non_const_type;
using MatColIdxKokkosViewHost = KokkosCsrGraphHost::entries_type::non_const_type;
using MatScalarKokkosViewHost = KokkosCsrMatrixHost::values_type::non_const_type;

using ConstMatRowMapKokkosView = KokkosCsrGraph::row_map_type::const_type;
using ConstMatColIdxKokkosView = KokkosCsrGraph::entries_type::const_type;
using ConstMatScalarKokkosView = KokkosCsrMatrix::values_type::const_type;

using ConstMatRowMapKokkosViewHost = KokkosCsrGraphHost::row_map_type::const_type;
using ConstMatColIdxKokkosViewHost = KokkosCsrGraphHost::entries_type::const_type;
using ConstMatScalarKokkosViewHost = KokkosCsrMatrixHost::values_type::const_type;

using MatRowMapKokkosDualView = Kokkos::DualView<MatRowMapType *>;
using MatColIdxKokkosDualView = Kokkos::DualView<MatColIdxType *>;
using MatScalarKokkosDualView = Kokkos::DualView<MatScalarType *>;

using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowMapType, MatColIdxType, MatScalarType, DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;

using KokkosTeamMemberType = Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type;

/*
  Mat_SeqBAIJKokkos - rectangular-block generalized CSR matrix held in Kokkos Views.

  Storage layout:
  - row_bs: block row size (e.g., 3 for elasticity velocity)
  - col_bs: block column size (e.g., 6 for prolongator to coarse dofs)
  - Block-CSR graph: i, j arrays index BLOCKS (not scalar entries)
    - i[i]: row map for block-row i; length mbs+1, where mbs = (m / row_bs)
    - j[k]: column index (block-column) for block k in the block-CSR graph
    - nblk: number of blocks (length of j)
  - Block values: a array holds all block data in ROW-MAJOR order:
    - Total length: nblk * row_bs * col_bs
    - Block (block_i, block_j) at index idx occupies a[idx*row_bs*col_bs : (idx+1)*row_bs*col_bs]
    - Within each block, elements are stored row-major: a[idx*row_bs*col_bs + i*col_bs + j]
      is the (i, j) entry of the block (0 <= i < row_bs, 0 <= j < col_bs)

  DualView synchronization:
  - i_dual, j_dual: block-CSR graph on host and device
  - a_dual: block-value data on host and device; DualView modify/sync flags track which space is current

  csrmat_graph (KokkosSparse::CrsMatrix scalar block graph): holds the block-CSR structure as a
  scalar CSR graph (values are a length-nblk dummy); used to call KokkosSparse::spgemm_symbolic for
  the AB symbolic phase.
*/

/*
  BAIJKokkosTeamSizeDefault - default device team size for the block kernels (MatMult family, AB, fused
  PtAP): a warp cooperates on a block-row on a device backend, one thread per team on a host backend so
  the league parallelizes over block-rows. 16 was tuned on the A100 (F8f). NOT Kokkos::AUTO (it has
  chosen poorly in this tree). Overridable per matrix with -mat_baijkokkos_team_size.
*/
static inline PetscInt BAIJKokkosTeamSizeDefault()
{
  constexpr bool on_device = !Kokkos::SpaceAccessibility<DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
  return on_device ? 16 : 1;
}

struct Mat_SeqBAIJKokkos {
  PetscInt row_bs, col_bs; /* block row and column sizes (generalized; row_bs != col_bs allowed) */
  PetscInt mbs, nbs;       /* number of block-rows and block-columns (m/row_bs, n/col_bs) */

  /* Device-kernel tuning, per matrix (set by MatSetFromOptions_SeqBAIJKokkos / at preallocation). */
  PetscInt  team_size         = BAIJKokkosTeamSizeDefault(); /* TeamPolicy team size; -mat_baijkokkos_team_size */
  PetscBool use_generic       = PETSC_FALSE;                 /* force generic kernel; -mat_baijkokkos_generic_kernel */
  PetscBool use_noatomic_spmv = PETSC_FALSE;                 /* opt in to the atomic-free reduction MatMult (experimental, slower on the A100 than the default atomic kernel); -mat_baijkokkos_spmv_noatomic */

  MatRowMapKokkosDualView i_dual; /* block-row map (length mbs+1) */
  MatColIdxKokkosDualView j_dual; /* block-column indices (length nblk) */
  MatScalarKokkosDualView a_dual; /* block values (length nblk*row_bs*col_bs, row-major per block) */

  /* Host-side assembly bookkeeping (allocated at preallocation, used during MatSetValues[Blocked]) */
  PetscInt *imax = NULL; /* imax[i] = allocated block-slots per block-row i (length mbs) */
  PetscInt *ilen = NULL; /* ilen[i] = currently-used block-slots in block-row i (length mbs) */

  KokkosCsrMatrix csrmat_graph; /* Scalar block-graph CSR (i,j only) used to call KokkosSparse::spgemm_symbolic. numRows()=mbs, numCols()=nbs, nnz()=nblk. */

  /*
    Construct from device-allocated block-CSR Views. Host mirrors of i, j are created; values
    remain on device until explicitly synced. Builds the scalar block-graph CSR for symbolic ops.
  */
  Mat_SeqBAIJKokkos(PetscInt row_bs_in, PetscInt col_bs_in, PetscInt mbs_in, PetscInt nbs_in, PetscInt nblk_in, const MatRowMapKokkosView &i_d, const MatColIdxKokkosView &j_d, const MatScalarKokkosView &a_d) :
    row_bs(row_bs_in), col_bs(col_bs_in), mbs(mbs_in), nbs(nbs_in)
  {
    auto a_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, HostMirrorMemorySpace(), a_d);
    auto i_h = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), i_d);
    auto j_h = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), j_d);

    a_dual = MatScalarKokkosDualView(a_d, a_h);
    a_dual.modify_device(); /* Mark device has latest data since we did not copy a_d to a_h */
    i_dual = MatRowMapKokkosDualView(i_d, i_h);
    j_dual = MatColIdxKokkosDualView(j_d, j_h);

    /* Scalar block-graph CSR for symbolic operations. Only the (i,j) graph is used; its values array
       must have length nblk (one scalar per block), not nblk*row_bs*col_bs, so use a dummy view. */
    MatScalarKokkosView graph_vals("csrmat_graph_vals", nblk_in);
    csrmat_graph = KokkosCsrMatrix("csrmat_graph", nbs_in, graph_vals, KokkosCsrGraph(j_d, i_d));
  }

  MatScalarType *a_host_data() { return a_dual.view_host().data(); }
  MatRowMapType *i_host_data() { return i_dual.view_host().data(); }
  MatColIdxType *j_host_data() { return j_dual.view_host().data(); }

  MatRowMapType nblks() { return j_dual.view_device().extent(0); /* number of blocks */ }
};

/* PtAP numeric algorithm choice (set in symbolic from -mat_product_algorithm). */
typedef enum {
  MAT_BAIJKOK_PTAP_SPEED     = 0, /* two-product: materialize W=A*P, then Ac=P^T*W (work-optimal) */
  MAT_BAIJKOK_PTAP_MEMORY_M1 = 1  /* fused single-pass, no W; atomic accumulate into Ac */
} MatBAIJKokkosPtAPAlg;

struct MatProductCtx_SeqBAIJKokkos {
  KernelHandle         kh;
  PetscBool            reusesym;
  Mat                  At;                   /* cached transpose of product->A (AtB), or of P (PtAP, = R) */
  MatColIdxKokkosView  transpose_block_perm; /* At block-slot p -> source block-slot (length nblk); empty until built in symbolic */
  Mat                  W        = NULL;      /* cached intermediate W=A*P for the PtAP SPEED path (NULL otherwise) */
  MatBAIJKokkosPtAPAlg ptap_alg = MAT_BAIJKOK_PTAP_SPEED;
  MatProductCtx_SeqBAIJKokkos() : reusesym(PETSC_FALSE), At(NULL) { }
};

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJKokkos(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJKokkos_SeqAIJ(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqBAIJKokkosModifyDevice(Mat);
PETSC_INTERN PetscErrorCode MatSeqBAIJKokkosSyncDevice(Mat);

/* Native seq block-product helpers, reused by the parallel MPIBAIJKOKKOS products (F2.2 Option B). */
PETSC_INTERN PetscErrorCode MatProductSymbolicAB_SeqBAIJKokkos_Helper(Mat, Mat, Mat, MatProductCtx_SeqBAIJKokkos *);
PETSC_INTERN PetscErrorCode MatProductNumericAB_SeqBAIJKokkos_Helper(Mat, Mat, Mat);
PETSC_INTERN PetscErrorCode MatTransposeWithPerm_SeqBAIJKokkos_Private(Mat, Mat *, MatColIdxKokkosView *);
PETSC_INTERN PetscErrorCode MatRefreshTransposeValues_SeqBAIJKokkos(Mat, Mat, MatColIdxKokkosView);
