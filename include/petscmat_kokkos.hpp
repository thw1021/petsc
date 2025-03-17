#pragma once

#include <petscconf.h>

/* SUBMANSEC = Mat */

#if defined(PETSC_HAVE_KOKKOS)

  #include <petsc_kokkos.hpp>
  #include <petsc/private/kokkosimpl.hpp>
  #include <../src/mat/impls/aij/seq/aij.h>
  #include <KokkosSparse_CrsMatrix.hpp>
  #include <KokkosSparse_spiluk.hpp>
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

/*@C
   MatCreateSeqAIJKokkosWithKokkosViews - Creates a MATSEQAIJKOKKOS matrix with Kokkos views of the aij data

   Synopsis:
   #include <petscmat_kokkos.hpp>
   PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews  (MPI_Comm comm, PetscInt m, PetscInt n, MatRowMapKokkosDualView &i, MatColIdxKokkosDualView &j, MatScalarKokkosDualView a, Mat *A);

   Logically Collective, No Fortran Support

   Input Parameter:
+  comm  - the MPI communicator
-  m     - row size
-  n     - the column size
-  i     - the dual Kokkos view of row data
-  j     - the dual Kokkos view of the column data
-  a     - the dual Kokkos view of the values

   Output Parameter:
.  A  - the `MATSEQAIJKOKKOS` matrix

   Level: intermediate

   Notes:
   Creates a Mat given the csr data input as Kokkos dual vectors. This routine allows a Mat
   to be built without involving the host.

.seealso:`MatCreateSeqAIJKokkosWithKokkosCsrMatrix()`
@*/
PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews(MPI_Comm, PetscInt, PetscInt, MatRowMapKokkosDualView &, MatColIdxKokkosDualView &, MatScalarKokkosDualView, Mat *);

/*@C
   MatCreateSeqAIJKokkosWithKokkosCsrMatrix - Creates a MATSEQAIJKOKKOS matrix from a Kokkos CSR matrix

   Synopsis:
   #include <petscmat_kokkos.hpp>
   PetscErrorCode MatCreateSeqAIJKokkosWithKokkosCsrMatrix  (MPI_Comm comm, KokkosCsrMatrix A_csr, Mat *A);

   Logically Collective, No Fortran Support

   Input Parameter:
+  comm  - the MPI communicator
-  A_csr - the Kokkos CSR matrix

   Output Parameter:
.  A  - the `MATSEQAIJKOKKOS` matrix

   Level: intermediate

   Notes:
   Creates a Mat given an existing Kokkos CSR matrix. This routine allows a Mat
   to be built without involving the host.

.seealso: `MatCreateSeqAIJKokkosWithKokkosViews()`
@*/
PetscErrorCode MatCreateSeqAIJKokkosWithKokkosCsrMatrix(MPI_Comm, KokkosCsrMatrix, Mat *);

#endif
