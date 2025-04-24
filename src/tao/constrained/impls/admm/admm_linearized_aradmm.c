#include "admm.h"

PETSC_INTERN PetscErrorCode TaoADMMSetUp_Linearized_ARADMM(Tao tao)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_PLIB, "not implemented");
  PetscFunctionReturn(PETSC_SUCCESS);
}
