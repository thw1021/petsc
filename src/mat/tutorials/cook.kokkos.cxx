#include <Kokkos_Core.hpp>
#include <petscmat.h>
#include "Kokkos_Core_fwd.hpp"
#include "coo.h"
#include "petscdevicetypes.h"
#include "petscerror.h"
#include "petscsys.h"

PETSC_EXTERN PetscErrorCode FillMatrixKokkosCOO(FEStruct *fe,Mat A)
{
  PetscErrorCode ierr;
  PetscScalar    *v;
  Vec            values;
  PetscMemType   mem_type;

  PetscFunctionBeginUser;
  /* simulation of GPU based finite assembly process with COO */
  ierr = VecCreateSeqKokkos(PETSC_COMM_SELF, 3*3*fe->Ne, &values);CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(values, &v, &mem_type);CHKERRQ(ierr);
  if (!PetscMemTypeDevice(mem_type)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Expected device pointer but got MemType %d",(int)mem_type);
  Kokkos::parallel_for("AssembleElementMatrices", fe->Ne, KOKKOS_LAMBDA (PetscInt i) {
      PetscScalar *s = &v[3*3*i];
      for (PetscInt vi=0; vi<3; vi++) {
        for (PetscInt vj=0; vj<3; vj++) {
          s[vi*3+vj] = vi+vj;
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
