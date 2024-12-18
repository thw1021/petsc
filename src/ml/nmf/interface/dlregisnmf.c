#include <petsc/private/nmfimpl.h>

static PetscBool PetscNMFPackageInitialized = PETSC_FALSE;

/*@C
  PetscNMFFinalizePackage - This function destroys everything in the PETSc/PetscNMF
  interface to the PetscNMF package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscNMFInitializePackage()`, `PetscFinalize()`, `PetscNMFRegister()`, `PetscNMFRegisterAll()`
@*/
PetscErrorCode PetscNMFFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscNMFList));
  PetscNMFPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscNMFInitializePackage - This function sets up PETSc to use the PetscNMF
  package.

  Level: developer

.seealso: `PetscNMFCreate()`, `PetscNMFFinalizePackage()`, `PetscNMFRegister()`, `PetscNMFRegisterAll()`
@*/
PetscErrorCode PetscNMFInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PetscNMFPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscNMFPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("PetscNMF", &PETSCNMF_CLASSID));
  /* Register Constructors */
  PetscCall(PetscNMFRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("PetscNMFFit", PETSCNMF_CLASSID, &PETSCNMF_Fit));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSCNMF_CLASSID;
    PetscCall(PetscInfoProcessClass("petscnmf", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("petscnmf", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSCNMF_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscNMFFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
