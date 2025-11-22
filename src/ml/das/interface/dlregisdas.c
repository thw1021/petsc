#include <petsc/private/dasimpl.h>

static PetscBool PetscDASPackageInitialized = PETSC_FALSE;

/*@C
  PetscDASInitializePackage - Initialize `PetscDAS` package

  Logically Collective

  Level: developer

.seealso: `PetscDASFinalizePackage()`
@*/
PetscErrorCode PetscDASInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscDASPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDASPackageInitialized = PETSC_TRUE;
  /* Register Class */
  PetscCall(PetscClassIdRegister("Data Assimilation System", &PETSCDAS_CLASSID));
  /* Register Constructors */
  PetscCall(PetscDASRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("DASSetUp", PETSCDAS_CLASSID, &PetscDAS_SetUp));
  PetscCall(PetscLogEventRegister("DASAssimilate", PETSCDAS_CLASSID, &PetscDAS_Assimilate));
  PetscCall(PetscLogEventRegister("DASForecast", PETSCDAS_CLASSID, &PetscDAS_Forecast));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSCDAS_CLASSID;
    PetscCall(PetscInfoProcessClass("das", 1, classids));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscDASFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDASFinalizePackage - Finalize `PetscDAS` package; it is called from `PetscFinalize()`

  Logically Collective

  Level: developer

.seealso: `PetscDASInitializePackage()`
@*/
PetscErrorCode PetscDASFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscDASList));
  PetscDASPackageInitialized = PETSC_FALSE;
  PetscDASRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}