/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

PetscErrorCode DMDestroy_DA(DM da)
{
  DM_DA *dd = (DM_DA *)da->data;

  PetscFunctionBegin;
  /* destroy the external/common part */
  for (PetscInt i = 0; i < DMDA_MAX_WORK_ARRAYS; i++) {
    PetscCall(PetscFree(dd->startghostedout[i]));
    PetscCall(PetscFree(dd->startghostedin[i]));
    PetscCall(PetscFree(dd->startout[i]));
    PetscCall(PetscFree(dd->startin[i]));
  }

  PetscCall(VecScatterDestroy(&dd->gtol));
  PetscCall(VecScatterDestroy(&dd->ltol));
  PetscCall(VecDestroy(&dd->natural));
  PetscCall(VecScatterDestroy(&dd->gton));
  PetscCall(AODestroy(&dd->ao));
  PetscCall(PetscFree(dd->aotype));

  PetscCall(PetscFree(dd->lx));
  PetscCall(PetscFree(dd->ly));
  PetscCall(PetscFree(dd->lz));

  PetscCall(PetscFree(dd->refine_x_hier));
  PetscCall(PetscFree(dd->refine_y_hier));
  PetscCall(PetscFree(dd->refine_z_hier));

  if (dd->fieldname) {
    for (PetscInt i = 0; i < dd->w; i++) PetscCall(PetscFree(dd->fieldname[i]));
    PetscCall(PetscFree(dd->fieldname));
  }
  if (dd->coordinatename) {
    for (PetscInt i = 0; i < da->dim; i++) PetscCall(PetscFree(dd->coordinatename[i]));
    PetscCall(PetscFree(dd->coordinatename));
  }
  PetscCall(ISColoringDestroy(&dd->localcoloring));
  PetscCall(ISColoringDestroy(&dd->ghostedcoloring));

  PetscCall(PetscFree(dd->neighbors));
  PetscCall(PetscFree(dd->dfill));
  PetscCall(PetscFree(dd->ofill));
  PetscCall(PetscFree(dd->ofillcols));
  PetscCall(PetscFree(dd->e));
  PetscCall(ISDestroy(&dd->ecorners));

  PetscCall(PetscObjectComposeFunction((PetscObject)da, "DMSetUpGLVisViewer_C", NULL));

  if (dd->bc) {
    for (PetscInt b = 0; b < dd->bc->numBC; b++) {
      if (dd->bc->bcPoints[b]) PetscCall(ISDestroy(&dd->bc->bcPoints[b]));
      if (dd->bc->bcComps[b]) PetscCall(ISDestroy(&dd->bc->bcComps[b]));
    }
    PetscCall(PetscFree(dd->bc->bcPoints));
    PetscCall(PetscFree(dd->bc->bcComps));
    PetscCall(PetscFree(dd->bc));
  }

  PetscCall(PetscFree(dd));
  PetscFunctionReturn(PETSC_SUCCESS);
}
