#include <petsc/private/imimpl.h>

PetscClassId      IM_CLASSID;
PetscFunctionList IMList = NULL;
static PetscBool  IMPackageInitialized = PETSC_FALSE;

PetscErrorCode IMFinalizePackage(void)
{
  PetscFunctionBegin;
  IMPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode IMInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (IMPackageInitialized) PetscFunctionReturn(0);
  IMPackageInitialized = PETSC_TRUE;
  ierr = PetscClassIdRegister("IM",&IM_CLASSID);CHKERRQ(ierr);
  {
    PetscClassId classids[1];

    classids[0] = IM_CLASSID;
    ierr = PetscInfoProcessClass("IM", 1, classids);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    PetscBool pkg;

    ierr = PetscStrInList("im",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(IM_CLASSID);CHKERRQ(ierr);}
  }
  ierr = IMRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(IMFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMRegister(const char tname[], PetscErrorCode (*create)(IM))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&IMList, tname, create);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
