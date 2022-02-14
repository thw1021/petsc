#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include "ex18.h"

using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

PetscErrorCode FillMatrixCOO_Kokkos(FEStruct *fe,Mat A)
{
  PetscErrorCode             ierr;
  Vec                        values;
  Kokkos::View<PetscScalar*,DefaultMemorySpace> v;

  PetscFunctionBeginUser;
  // Simulation of GPU based finite assembly process with COO
  // Uses a vector to provide the needed buffer space for convenience since it's memory be naturally in the correct location for Kokkos;
  // one could also allocate the memory explicitly
  ierr = VecCreateSeqKokkos(PETSC_COMM_SELF,3*3*fe->Ne,&values);CHKERRQ(ierr);
  ierr = VecGetKokkosViewWrite(values,&v);CHKERRQ(ierr);
  Kokkos::parallel_for("AssembleElementMatrices", fe->Ne, KOKKOS_LAMBDA (PetscInt i) {
      PetscScalar *s = &v(3*3*i);
      for (PetscInt vi=0; vi<3; vi++) {
        for (PetscInt vj=0; vj<3; vj++) {
          s[vi*3+vj] = vi+2*vj;
        }
      }
    });
  ierr = MatSetValuesCOO(A,v.data(),INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreKokkosViewWrite(values,&v);CHKERRQ(ierr);
  ierr = VecDestroy(&values);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
