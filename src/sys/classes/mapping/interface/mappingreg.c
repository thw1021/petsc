#include <petsc/private/mappingimpl.h>

PetscClassId      PETSC_MAPPING_CLASSID;
PetscFunctionList PetscMappingList = NULL;
static PetscBool  PetscMappingPackageInitialized = PETSC_FALSE;

PetscErrorCode PetscMappingFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscMappingPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMappingPackageInitialized) PetscFunctionReturn(0);
  PetscMappingPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("Mapping",&PETSC_MAPPING_CLASSID);CHKERRQ(ierr);
  {
    PetscClassId classids[1];

    classids[0] = PETSC_MAPPING_CLASSID;
    ierr = PetscInfoProcessClass("mapping", 1, classids);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    PetscBool pkg;

    ierr = PetscStrInList("mapping",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_MAPPING_CLASSID);CHKERRQ(ierr);}
  }
  ierr = PetscMappingRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscMappingFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMappingRegister(const char tname[], PetscErrorCode (*create)(PetscMapping))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMappingInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscMappingList, tname, create);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
