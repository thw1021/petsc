#if !defined(PETSCDSTYPES_H)
#define PETSCDSTYPES_H

#include <petscdmlabel.h>

/*S
  PetscDS - PETSc object that manages a discrete system, which is a set of discretizations + continuum equations from a PetscWeakForm

  Level: intermediate

.seealso: PetscDSCreate(), PetscDSSetType(), PetscDSType, PetscWeakForm, PetscFECreate(), PetscFVCreate()
S*/
typedef struct _p_PetscDS *PetscDS;

/*S
  PetscWeakForm - PETSc object that manages a sets of pointwise functions defining a system of equations

  Level: intermediate

.seealso: PetscWeakFormCreate(), PetscDS, PetscFECreate(), PetscFVCreate()
S*/
typedef struct _p_PetscWeakForm *PetscWeakForm;

typedef struct _PetscHashFormKey
{
  DMLabel  label;
  PetscInt value;
  PetscInt field;
} PetscHashFormKey;

/*E
  PetscWeakFormKind - The kind of weak form being specified.

  Supported kinds include:
$ OBJECTIVE                  - Objective form
$ F0, F1                     - Residual forms
$ G0, G1, G2, G3             - Jacobian forms
$ GP0, GP1, GP2, GP3         - Jacobian preconditioner matrix forms
$ GT0, GT1, GT2, GT3         - Dynamic Jacobian matrix forms
$ BDF0, BDF1                 - Boundary Residual forms
$ BDG0, BDG1, BDG2, BDG3     - Jacobian forms
$ BDGP0, BDGP1, BDGP2, BDGP3 - Jacobian preconditioner matrix forms
$ R                          - Riemann solver

  Level: beginner

.seealso: PetscWeakFormSetIndexResidual(), PetscWeakFormClearIndex()
E*/
typedef enum {PETSC_WF_OBJECTIVE, PETSC_WF_F0, PETSC_WF_F1, PETSC_WF_G0, PETSC_WF_G1, PETSC_WF_G2, PETSC_WF_G3, PETSC_WF_GP0, PETSC_WF_GP1, PETSC_WF_GP2, PETSC_WF_GP3, PETSC_WF_GT0, PETSC_WF_GT1, PETSC_WF_GT2, PETSC_WF_GT3, PETSC_WF_BDF0, PETSC_WF_BDF1, PETSC_WF_BDG0, PETSC_WF_BDG1, PETSC_WF_BDG2, PETSC_WF_BDG3, PETSC_WF_BDGP0, PETSC_WF_BDGP1, PETSC_WF_BDGP2, PETSC_WF_BDGP3, PETSC_WF_R} PetscWeakFormKind;

#endif
