#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

/* dummy routines if CGNS is not found, the true functions are in impls/cgns */
#if !defined(PETSC_HAVE_CGNS)

PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --with-cgns-dir");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCreateCGNS(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
  PetscFunctionBegin;
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --download-cgns");
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
