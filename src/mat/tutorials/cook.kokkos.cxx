#include <Kokkos_Core.hpp>
#include <petscmat.h>
#include "Kokkos_Core_fwd.hpp"
#include "coo.h"
#include <petscdevicetypes.h>

PetscErrorCode FillMatrixKokkosCOO(FEStruct *fe,Mat A)
{
  PetscErrorCode ierr;
  PetscScalar    *v;
  Vec            values;
  PetscMemType   mem_type;

  PetscFunctionBeginUser;
  // Simulation of GPU based finite assembly process with COO
  // Uses a vector to provide the needed buffer space for convenience since it's memory be naturally in the correct location for Kokkos;
  // one could also allocate the memory explicitly
  ierr = VecCreateSeqKokkos(PETSC_COMM_SELF, 3*3*fe->Ne, &values);CHKERRQ(ierr);
  // Could use VecGetKokkosView() instead of VecGetArrayAndMemType()
  ierr = VecGetArrayAndMemType(values, &v, &mem_type);CHKERRQ(ierr);
  Kokkos::parallel_for("AssembleElementMatrices", fe->Ne, KOKKOS_LAMBDA (PetscInt i) {
      PetscScalar *s = &v[3*3*i];
      for (PetscInt vi=0; vi<3; vi++) {
        for (PetscInt vj=0; vj<3; vj++) {
          s[vi*3+vj] = vi+2*vj;
        }
      }
    });
  ierr = MatSetValuesCOO(A,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayAndMemType(values, &v);CHKERRQ(ierr);
  ierr = VecDestroy(&values);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
