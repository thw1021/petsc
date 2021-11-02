#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*@
  DMPlexSetMigrationSF - Sets the SF for migrating from a parent DM into this DM

  Input Parameters:
+ dm        - The DM
- naturalSF - The PetscSF

  Note: It is necessary to call this in order to have DMCreateSubDM() or DMCreateSuperDM() build the Global-To-Natural map

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateMigrationSF(), DMPlexGetMigrationSF()
@*/
PetscErrorCode DMPlexSetMigrationSF(DM dm, PetscSF migrationSF)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  dm->sfMigration = migrationSF;
  ierr = PetscObjectReference((PetscObject) migrationSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetMigrationSF - Gets the SF for migrating from a parent DM into this DM

  Input Parameter:
. dm          - The DM

  Output Parameter:
. migrationSF - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateMigrationSF(), DMPlexSetMigrationSF
@*/
PetscErrorCode DMPlexGetMigrationSF(DM dm, PetscSF *migrationSF)
{
  PetscFunctionBegin;
  *migrationSF = dm->sfMigration;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetGlobalToNaturalSF - Sets the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The DM
- naturalSF   - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateGlobalToNaturalSF(), DMPlexGetGlobaltoNaturalSF()
@*/
PetscErrorCode DMPlexSetGlobalToNaturalSF(DM dm, PetscSF naturalSF)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  dm->sfNatural = naturalSF;
  ierr = PetscObjectReference((PetscObject) naturalSF);CHKERRQ(ierr);
  dm->useNatural = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetGlobalToNaturalSF - Gets the SF for mapping Global Vec to the Natural Vec

  Input Parameter:
. dm          - The DM

  Output Parameter:
. naturalSF   - The PetscSF

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexCreateGlobalToNaturalSF(), DMPlexSetGlobaltoNaturalSF
@*/
PetscErrorCode DMPlexGetGlobalToNaturalSF(DM dm, PetscSF *naturalSF)
{
  PetscFunctionBegin;
  *naturalSF = dm->sfNatural;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateGlobalToNaturalSF - Creates the SF for mapping Global Vec to the Natural Vec

  Input Parameters:
+ dm          - The redistributed DM
. section     - The local PetscSection describing the Vec before the mesh was distributed, or NULL if not available
- sfMigration - The PetscSF used to distribute the mesh, or NULL if it cannot be computed

  Output Parameter:
. sfNatural   - PetscSF for mapping the Vec in PETSc ordering to the canonical ordering

  Note: This is not typically called by the user.

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField()
 @*/
PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM dm, PetscSection section, PetscSF sfMigration, PetscSF *sfNatural)
{
  MPI_Comm       comm;
  Vec            tmpVec;
  PetscSF        sf, sfEmbed, sfField;
  PetscSection   gSection, sectionDist, gLocSection;
  PetscInt      *spoints, *remoteOffsets;
  PetscInt       ssize, pStart, pEnd, p, globalSize;
  PetscBool      destroy = PETSC_FALSE, debug = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (!sfMigration) {
    /* If sfMigration is missing, sfNatural cannot be computed and is set to NULL */
    *sfNatural = NULL;
    PetscFunctionReturn(0);
  } else if (!section) {
    /* If the sequential section is not provided (NULL), it is reconstructed from the parallel section */
    PetscSF sfMigrationInv;
    PetscSection localSection;

    ierr = DMGetLocalSection(dm, &localSection);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(sfMigration, &sfMigrationInv);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMigrationInv, localSection, NULL, section);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfMigrationInv);CHKERRQ(ierr);
    destroy = PETSC_TRUE;
  }
  if (debug) {ierr = PetscSFView(sfMigration, NULL);CHKERRQ(ierr);}
  /* Create a new section from distributing the original section */
  ierr = PetscSectionCreate(comm, &sectionDist);CHKERRQ(ierr);
  ierr = PetscSFDistributeSection(sfMigration, section, &remoteOffsets, sectionDist);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) sectionDist, "Migrated Section");CHKERRQ(ierr);
  if (debug) {ierr = PetscSectionView(sectionDist, NULL);CHKERRQ(ierr);}
  ierr = DMSetLocalSection(dm, sectionDist);CHKERRQ(ierr);
  /* If a sequential section is provided but no dof is affected, sfNatural cannot be computed and is set to NULL */
  ierr = DMCreateGlobalVector(dm, &tmpVec);CHKERRQ(ierr);
  ierr = VecGetSize(tmpVec, &globalSize);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &tmpVec);CHKERRQ(ierr);
  if (globalSize) {
    const PetscInt *leaves;
    PetscInt       *sortleaves, *indices;
    PetscInt        Nl;

    /* Get a pruned version of migration SF */
    ierr = DMGetGlobalSection(dm, &gSection);CHKERRQ(ierr);
    if (debug) {ierr = PetscSectionView(gSection, NULL);CHKERRQ(ierr);}
    ierr = PetscSFGetGraph(sfMigration, NULL, &Nl, &leaves, NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(gSection, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off;

      ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(gSection, p, &off);CHKERRQ(ierr);
      if ((dof > 0) && (off >= 0)) ++ssize;
    }
    ierr = PetscMalloc3(ssize, &spoints, Nl, &sortleaves, Nl, &indices);CHKERRQ(ierr);
    for (p = 0; p < Nl; ++p) {sortleaves[p] = leaves ? leaves[p] : p; indices[p] = p;}
    ierr = PetscSortIntWithArray(Nl, sortleaves, indices);CHKERRQ(ierr);
    for (p = pStart, ssize = 0; p < pEnd; ++p) {
      PetscInt dof, off, loc;

      ierr = PetscSectionGetDof(gSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(gSection, p, &off);CHKERRQ(ierr);
      if ((dof > 0) && (off >= 0)) {
        ierr = PetscFindInt(p, Nl, sortleaves, &loc);CHKERRQ(ierr);
        PetscCheck(loc >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %D with nonzero dof is not a leaf of the migration SF", p);
        spoints[ssize++] = indices[loc];
      }
    }
    ierr = PetscSFCreateEmbeddedLeafSF(sfMigration, ssize, spoints, &sfEmbed);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) sfEmbed, "Embedded SF");CHKERRQ(ierr);
    ierr = PetscFree3(spoints, sortleaves, indices);CHKERRQ(ierr);
    if (debug) {ierr = PetscSFView(sfEmbed, NULL);CHKERRQ(ierr);}
    /* Create the SF associated with this section
         Roots are natural dofs, leaves are global dofs */
    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(sectionDist, sf, PETSC_FALSE, PETSC_TRUE, &gLocSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfEmbed, section, remoteOffsets, gLocSection, &sfField);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfEmbed);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&gLocSection);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) sfField, "Natural-to-Global SF");CHKERRQ(ierr);
    if (debug) {ierr = PetscSFView(sfField, NULL);CHKERRQ(ierr);}
    /* Invert the field SF
         Roots are global dofs, leaves are natural dofs */
    ierr = PetscSFCreateInverseSF(sfField, sfNatural);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *sfNatural, "Global-to-Natural SF");CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject) *sfNatural, NULL, "-globaltonatural_sf_view");CHKERRQ(ierr);
    /* Clean up */
    ierr = PetscSFDestroy(&sfField);CHKERRQ(ierr);
  } else {
    *sfNatural = NULL;
  }
  ierr = PetscSectionDestroy(&sectionDist);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  if (destroy) {ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalBegin - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameters:
. nv - Vec in the canonical ordering distributed over all processors associated with gv

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexGlobalToNaturalBegin(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_GlobalToNaturalBegin,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArray(nv, &outarray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray,MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(nv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
    ierr = VecCopy(gv, nv);CHKERRQ(ierr);
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  ierr = PetscLogEventEnd(DMPLEX_GlobalToNaturalBegin,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexGlobalToNaturalEnd - Rearranges a global Vector in the natural order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- gv - The global Vec

  Output Parameter:
. nv - The natural Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

 .seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexGlobalToNaturalEnd(DM dm, Vec gv, Vec nv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_GlobalToNaturalEnd,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecGetArray(nv, &outarray);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray,MPI_REPLACE);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(gv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(nv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  ierr = PetscLogEventEnd(DMPLEX_GlobalToNaturalEnd,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalBegin - Rearranges a Vector in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- nv - The natural Vec

  Output Parameters:
. gv - The global Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(),DMPlexGlobalToNaturalEnd()
@*/
PetscErrorCode DMPlexNaturalToGlobalBegin(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_NaturalToGlobalBegin,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    /* We only have access to the SF that goes from Global to Natural.
       Instead of inverting dm->sfNatural, we can call PetscSFReduceBegin/End with MPI_Op MPI_SUM.
       Here the SUM really does nothing since sfNatural is one to one, as long as gV is set to zero first. */
    ierr = VecZeroEntries(gv);CHKERRQ(ierr);
    ierr = VecGetArray(gv, &outarray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
    ierr = VecCopy(nv, gv);CHKERRQ(ierr);
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  ierr = PetscLogEventEnd(DMPLEX_NaturalToGlobalBegin,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexNaturalToGlobalEnd - Rearranges a Vector in the natural order to the Global order.

  Collective on dm

  Input Parameters:
+ dm - The distributed DMPlex
- nv - The natural Vec

  Output Parameters:
. gv - The global Vec

  Note: The user must call DMSetUseNatural(dm, PETSC_TRUE) before DMPlexDistribute().

  Level: intermediate

.seealso: DMPlexDistribute(), DMPlexDistributeField(), DMPlexNaturalToGlobalBegin(), DMPlexGlobalToNaturalBegin()
 @*/
PetscErrorCode DMPlexNaturalToGlobalEnd(DM dm, Vec nv, Vec gv)
{
  const PetscScalar *inarray;
  PetscScalar       *outarray;
  PetscMPIInt        size;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_NaturalToGlobalEnd,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRMPI(ierr);
  if (dm->sfNatural) {
    ierr = VecGetArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecGetArray(gv, &outarray);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(dm->sfNatural, MPIU_SCALAR, (PetscScalar *) inarray, outarray, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(nv, &inarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gv, &outarray);CHKERRQ(ierr);
  } else if (size == 1) {
  } else PetscCheckFalse(dm->useNatural,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "DM global to natural SF not present.\nIf DMPlexDistribute() was called and a section was defined, report to petsc-maint@mcs.anl.gov.");
  else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONGSTATE, "DM global to natural SF was not created.\nYou must call DMSetUseNatural() before DMPlexDistribute().");
  ierr = PetscLogEventEnd(DMPLEX_NaturalToGlobalEnd,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
