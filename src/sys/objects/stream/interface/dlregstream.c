#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/

static PetscBool PetscStreamPackageInitialized = PETSC_FALSE;
static PetscBool PetscEventPackageInitialized = PETSC_FALSE;
static PetscBool PetscStreamScalarPackageInitialized = PETSC_FALSE;
static PetscBool PetscStreamGraphPackageInitialized = PETSC_FALSE;

PetscErrorCode PetscStreamInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamPackageInitialized) PetscFunctionReturn(0);
  PetscStreamPackageInnitialized = PETSC_TRUE;
  ierr = PetscStreamRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscStreamFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscStreamPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscEventPackageInitialized) PetscFunctionReturn(0);
  PetscEventPackageInnitialized = PETSC_TRUE;
  ierr = PetscEventRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscEventFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscEventFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscEventPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamScalarPackageInitialized) PetscFunctionReturn(0);
  PetscStreamScalarPackageInnitialized = PETSC_TRUE;
  ierr = PetscStreamScalarRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscStreamScalarFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscStreamScalarPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscStreamGraphPackageInitialized) PetscFunctionReturn(0);
  PetscStreamGraphPackageInnitialized = PETSC_TRUE;
  ierr = PetscStreamGraphRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscStreamGraphFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamGraphFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscStreamGraphPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}
