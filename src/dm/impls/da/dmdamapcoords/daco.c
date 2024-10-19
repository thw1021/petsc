#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

/* -----------------------------------------------------------------------------
   Checks for periodicity of a DM and Maps points outside of a domain back onto the domain
   using appropriate periodicity. At the moment assumes only a 2-D DMDA.
   ----------------------------------------------------------------------------------------*/
PetscErrorCode DMDAMapCoordsToPeriodicDomain(DM da, PetscScalar *x, PetscScalar *y)
{
  DMBoundaryType bx, by;
  PetscInt       dim, gx, gy;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, &dim, &gx, &gy, NULL, NULL, NULL, NULL, NULL, NULL, &bx, &by, NULL, NULL));

  if (bx == DM_BOUNDARY_PERIODIC) {
    while (*x >= (PetscScalar)gx) *x -= (PetscScalar)gx;
    while (*x < 0.0) *x += (PetscScalar)gx;
  }
  if (by == DM_BOUNDARY_PERIODIC) {
    while (*y >= (PetscScalar)gy) *y -= (PetscScalar)gy;
    while (*y < 0.0) *y += (PetscScalar)gy;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
