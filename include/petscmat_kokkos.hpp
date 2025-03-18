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

/*@C
   MatCreateSeqAIJKokkosWithKokkosViews - Creates a MATSEQAIJKOKKOS matrix with Kokkos views of the aij data

   Synopsis:
   #include <petscmat_kokkos.hpp>
   PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews  (MPI_Comm comm, PetscInt m, PetscInt n, Kokkos::View<PetscInt *, MemorySpace>&, Kokkos::View<PetscInt *, MemorySpace>&, Kokkos::View<PetscScalar *, MemorySpace>&, Mat *A);

   Logically Collective, No Fortran Support

   Input Parameter:
+  comm  - the MPI communicator
-  m     - row size
-  n     - the column size
-  i     - the Kokkos view of row data (can be in either HostMirrorMemorySpace or Kokkos::DefaultExecutionSpace)
-  j     - the Kokkos view of the column data (can be in either HostMirrorMemorySpace or Kokkos::DefaultExecutionSpace)
-  a     - the Kokkos view of the values (can be in either HostMirrorMemorySpace or Kokkos::DefaultExecutionSpace)

   Output Parameter:
.  A  - the `MATSEQAIJKOKKOS` matrix

   Level: intermediate

   Notes:
   Creates a Mat given the csr data input as Kokkos views. This routine allows a Mat
   to be built without involving the host. Don't modify entries in the views after this routine.
   There should be no outstanding asynchronous operations on the views (ie this routine does not call fence()
   before using the views)

.seealso:
@*/
template <class MemorySpace>
PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews(MPI_Comm, PetscInt, PetscInt, Kokkos::View<PetscInt *, MemorySpace> &, Kokkos::View<PetscInt *, MemorySpace> &, Kokkos::View<PetscScalar *, MemorySpace> &, Mat *);

#endif
