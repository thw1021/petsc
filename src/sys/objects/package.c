
#include <petsc/private/petscimpl.h>        /*I    "petscsys.h"   I*/

const char *const PetscExternalPackages[] = {"hdf5","parmetis","PetscExternalPackage","PETSC_",NULL};

//TODO manpage
PetscErrorCode  PetscHaveExternalPackageName(const char pkg[], PetscBool *have)
{
  PetscExternalPackage  pkge=PETSC_EXT_PKG_PARMETIS;
  PetscBool             found;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscEListFind(PETSC_EXT_PKG_NUM, PetscExternalPackages, pkg, (PetscExternalPackage*) &pkge, &found);CHKERRQ(ierr);
  if (!found) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "PetscExternalPackage %s not defined", pkg);
  ierr = PetscHaveExternalPackage(pkge, have);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//TODO manpage
PetscErrorCode  PetscHaveExternalPackage(PetscExternalPackage pkg, PetscBool *have)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidBoolPointer(have,2);
  *have = PETSC_FALSE;
  switch (pkg) {
  case PETSC_EXT_PKG_HDF5:
#if defined(PETSC_HAVE_HDF5)
     *have = PETSC_TRUE;
#endif
     break;

  case PETSC_EXT_PKG_PARMETIS:
#if defined(PETSC_HAVE_PARMETIS)
     *have = PETSC_TRUE;
#endif
     break;

  default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "PetscExternalPackage with value %d not defined", pkg);
  }
  PetscFunctionReturn(0);
}

