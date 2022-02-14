#include "ex18.h"

PetscErrorCode FillMatrixCOO_CPU(FEStruct *fe,Mat A)
{
  PetscErrorCode             ierr;
  PetscScalar                *v;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(3*3*fe->Ne,&v);CHKERRQ(ierr);
  for (PetscInt i=0; i<fe->Ne; i++) {
    PetscScalar *s = &v[3*3*i];
    for (PetscInt vi=0; vi<3; vi++) {
      for (PetscInt vj=0; vj<3; vj++) {
        s[vi*3+vj] = vi+2*vj;
      }
    }
  }
  ierr = MatSetValuesCOO(A,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
