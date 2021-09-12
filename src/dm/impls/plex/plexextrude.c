#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscdmplextransform.h>

PetscErrorCode DMExtrude_Plex(DM dm, PetscInt layers, DM *edm)
{
  DMPlexTransform tr;
  DM              cdm, ecdm;
  const char     *prefix;
  PetscOptions    options;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetDM(tr, dm);CHKERRQ(ierr);
  ierr = DMPlexTransformSetType(tr, DMPLEXEXTRUDE);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) dm, &prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) tr,  prefix);CHKERRQ(ierr);
  ierr = PetscObjectGetOptions((PetscObject) dm, &options);CHKERRQ(ierr);
  ierr = PetscObjectSetOptions((PetscObject) tr, options);CHKERRQ(ierr);
  ierr = DMPlexTransformExtrudeSetLayers(tr, layers);CHKERRQ(ierr);
  ierr = DMPlexTransformSetFromOptions(tr);CHKERRQ(ierr);
  ierr = PetscObjectSetOptions((PetscObject) tr, NULL);CHKERRQ(ierr);
  ierr = DMPlexTransformSetUp(tr);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) tr, NULL, "-dm_plex_transform_view");CHKERRQ(ierr);
  ierr = DMPlexTransformApply(tr, dm, edm);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm, *edm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(*edm, &ecdm);CHKERRQ(ierr);
  ierr = DMCopyDisc(cdm, ecdm);CHKERRQ(ierr);
  ierr = DMPlexTransformCreateDiscLabels(tr, *edm);CHKERRQ(ierr);
  ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
  if (*edm) {
    ((DM_Plex *) (*edm)->data)->printFEM = ((DM_Plex *) dm->data)->printFEM;
    ((DM_Plex *) (*edm)->data)->printL2  = ((DM_Plex *) dm->data)->printL2;
  }
  PetscFunctionReturn(0);
}
