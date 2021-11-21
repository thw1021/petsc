#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

/*

   Reorders marks every DM point (vertex, edge, ...)

    core   : owned and on an element all of whose points are private to this rank, definitely not a root in the PETSc SF
    owned  : owned but on an element that shares some points with other ranks, this does not mean the point is necessarily shared so it may
              or may not be a root in the PETSc SF
    ghost  : not owned, a leaf in the PetscSF.

   by inspecting the `pointSF` graph.

   Could use PetscSFGetLeafRanks() to determine the points that are roots in the PETSc SF

   The code was translated from FireDrake Cython code
*/
PetscErrorCode DMPlexLabelPointOwnershipType(DM dm)
{
  PetscErrorCode    ierr;
  PetscInt          pStart, pEnd, cStart, cEnd;
  PetscInt          c, ci, p;
  PetscInt          nleaves;
  PetscInt          *closure = NULL;
  PetscInt          nclosure;
  const PetscInt    *ilocal = NULL;
  PetscSF           point_sf = NULL;
  PetscBool         is_ghost, is_owned;
  DMLabel           lbl_core, lbl_owned, lbl_ghost;
  PetscMPIInt       size;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"pt_core");CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"pt_owned");CHKERRQ(ierr);
  ierr = DMCreateLabel(dm,"pt_ghost");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "pt_core", &lbl_core);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "pt_owned", &lbl_owned);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "pt_ghost", &lbl_ghost);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    /* Mark ghosts from point overlap SF */
    ierr = DMGetPointSF(dm,&point_sf);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(point_sf, NULL, &nleaves, &ilocal, NULL);CHKERRQ(ierr);
    for (p=0; p<nleaves; p++) {
      ierr = DMLabelSetValue(lbl_ghost, ilocal[p], 1);CHKERRQ(ierr);
    }
  } else {
    /* If sequential mark all points as core */
    for (p=pStart; p<pEnd; p++) {
      ierr = DMLabelSetValue(lbl_core, p, 1);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMLabelCreateIndex(lbl_ghost, pStart, pEnd);CHKERRQ(ierr);
  /*
     If any entity in closure(cell) is in the halo, then all those
     entities in closure(cell) that are not in the halo are owned,
     but not core.
  */
  for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &nclosure, &closure);CHKERRQ(ierr);
    is_owned = PETSC_FALSE;
    for (ci=0; ci<nclosure; ci++){
       p = closure[2*ci];
       ierr = DMLabelHasPoint(lbl_ghost, p, &is_ghost);CHKERRQ(ierr);
       if (is_ghost) {
         is_owned = PETSC_TRUE;
         break;
       }
    }
    if (is_owned) {
      for (ci=0; ci<nclosure; ci++) {
        p = closure[2*ci];
        ierr = DMLabelHasPoint(lbl_ghost, p, &is_ghost);CHKERRQ(ierr);
        if (!is_ghost) {
          ierr = DMLabelSetValue(lbl_owned, p, 1);CHKERRQ(ierr);
        }
      }
    }
    if (closure) {
      ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &nclosure, &closure);CHKERRQ(ierr);
    }
  }
  /* Mark all remaining points as core */
  ierr = DMLabelCreateIndex(lbl_owned, pStart, pEnd);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++){
    ierr = DMLabelHasPoint(lbl_owned, p, &is_owned);CHKERRQ(ierr);
    ierr = DMLabelHasPoint(lbl_ghost, p, &is_ghost);CHKERRQ(ierr);
    if ((!is_ghost) && (!is_owned)) {
      ierr = DMLabelSetValue(lbl_core, p, 1);CHKERRQ(ierr);
    }
  }
  ierr = DMLabelDestroyIndex(lbl_owned);CHKERRQ(ierr);
  ierr = DMLabelDestroyIndex(lbl_ghost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Builds PyOP2 entity class offsets for all entity levels.

    :arg dm: The DM object encapsulating the mesh topology
*/
PetscErrorCode DMPlexGetPointOwnershipType(DM dm,PetscInt *depth,PetscInt **entity_class_sizes)
{
  PetscErrorCode ierr;
  PetscInt       *eStart, *eEnd;
  PetscInt       d, i, ci, class_size;
  const PetscInt *indices = NULL;
  IS             class_is;
  char     const *op2class[3] = {"pt_core","pt_owned","pt_ghost"};

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,depth);CHKERRQ(ierr);
  (*depth)++;

  ierr = PetscCalloc1(3*(*depth),entity_class_sizes);CHKERRQ(ierr);
  ierr = PetscMalloc2(*depth,&eStart,*depth,&eEnd);CHKERRQ(ierr);
  for (d=0; d<*depth; d++){
    ierr = DMPlexGetDepthStratum(dm, d, &eStart[d], &eEnd[d]);CHKERRQ(ierr);
  }

  for (i=0; i<3; i++) {
    ierr = DMGetStratumIS(dm,op2class[i], 1,&class_is);CHKERRQ(ierr);
    ierr = DMGetStratumSize(dm,op2class[i], 1,&class_size);CHKERRQ(ierr);
    if (class_size > 0) {
      ierr = ISGetIndices(class_is, &indices);CHKERRQ(ierr);
      for (ci=0; ci<class_size; ci++) {
        for (d=0; d<*depth; d++) {
          if ((eStart[d] <= indices[ci]) && (indices[ci] < eEnd[d])) {
            (*entity_class_sizes)[d+i*(*depth)]++;
            break;
          }
        }
      }
      ierr = ISRestoreIndices(class_is, &indices);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&class_is);CHKERRQ(ierr);
  }
  ierr = PetscFree2(eStart,eEnd);CHKERRQ(ierr);

  /* PyOP2 entity class indices are additive */
  for (d=0; d<*depth; d++) {
    for (i=1; i<3; i++){
      (*entity_class_sizes)[d+i*(*depth)] += (*entity_class_sizes)[d+(i-1)*(*depth)];
    }
  }
  PetscFunctionReturn(0);
}

/*@
    DMPlexSetUseVecGhostPermutation - Reorders a DMPLEX so that DMPlexCreateVecGhost() may be used

    Input Parameter:
.    DM - the DMPLEX object

    Options Database:
.    -dm_plex_use_vec_ghost_permutation

    Level: basic

    Notes:
      Must be called before DMCreateGlobalVector() or DMCreateLocalVector() and DMGetGlobalSection() or DMGetLocalSection()

      Using this in conjuction with DMGlobalGetLocal*()/DMGlobalRestoreLocal() will eliminate copying of local data when moving between
      the local and the global vector representations.

.seealso: DMPlexCreateGhostVector(), DMPlexSetUpVecGhostPermutation()
@*/
PetscErrorCode DMPlexSetUseVecGhostPermutation(DM dm)
{
  DM_Plex        *plex = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  plex->useghostperm = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
    DMPlexSetUpVecGhostPermutation - Reorders a DMPLEX so that DMPlexCreateVecGhost() may be used

    Input Parameter:
.    DM - the DMPLEX object

    Level: basic

    Notes:
      Must be called before DMCreateGlobalVector() or DMCreateLocalVector() and DMGetGlobalSection() or DMGetLocalSection()

    Developer Notes:
      Builds a global point renumbering as a permutation of Plex points.
      The node permutation is derived from a depth-first traversal of
      the Plex graph over each entity class in turn. The returned IS
      is the Plex -> PyOP2 permutation.

    The code was translated from FireDrake Cython code

.seealso: DMPlexCreateGhostVector(), DMPlexSetUseVecGhostPermutation()
@*/
PetscErrorCode DMPlexSetUpVecGhostPermutation(DM dm)
{
  PetscErrorCode ierr;
  PetscInt       dim, cStart, cEnd, nclosure, ci, l, p, i,d;
  PetscInt       pStart, pEnd, c, depth, *entity_class_sizes;
  PetscInt       lidx[3] = {0,0,0};
  PetscInt       *closure = NULL;
  PetscInt       *perm = NULL;
  PetscBT        seen = NULL;
  PetscBool      has_point;
  DMLabel        labels[3];
  DM_Plex        *plex = (DM_Plex*)dm->data;

  PetscFunctionBegin;
  if (plex->vecghostperm) PetscFunctionReturn(0);

  ierr = DMPlexLabelPointOwnershipType(dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd - pStart, &perm);CHKERRQ(ierr);
  ierr = PetscBTCreate(pEnd - pStart, &seen);CHKERRQ(ierr); /* this will be incorrect if pStart is not 0 */

  /* Get label pointers and label-specific array indices */
  ierr = DMGetLabel(dm, "pt_core", &labels[0]);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "pt_owned", &labels[1]);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "pt_ghost", &labels[2]);CHKERRQ(ierr);
  for (l=0;l<3;l++) {
    ierr = DMLabelCreateIndex(labels[l], pStart, pEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexGetPointOwnershipType(dm,&depth,&entity_class_sizes);CHKERRQ(ierr);
  for (i=1; i<3; i++) {
    for (d=0; d<depth; d++) {
      lidx[i] += entity_class_sizes[d + (i-1)*depth];
    }
  }
  ierr = PetscFree(entity_class_sizes);CHKERRQ(ierr);

  for (c=pStart; c<pEnd; c++) {
    /*
       We always re-order cell-wise so that we inherit any cache
       coherency from the reordering provided by the Plex
    */
    if ((cStart <= c) && (c < cEnd)) {
      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &nclosure, &closure);CHKERRQ(ierr);
      for (ci=0; ci<nclosure; ci++) {
        p = closure[2*ci];
        if (!PetscBTLookup(seen, p)) {
          for (l=0; l<3; l++){
            ierr = DMLabelHasPoint(labels[l], p, &has_point);CHKERRQ(ierr);
            if (has_point) {
              PetscBTSet(seen, p);
              perm[lidx[l]++] = p;
              if (lidx[l] > pEnd) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"lidx[%D] %D overflow",l,lidx[l]);
              break;
            }
          }
        }
      }
    }
    if (closure) {
      ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &nclosure, &closure);CHKERRQ(ierr);
    }
  }
  for (l=0; l<3; l++){
    ierr = DMLabelDestroyIndex(labels[l]);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&seen);CHKERRQ(ierr);
  ierr = ISCreate(PetscObjectComm((PetscObject)dm),&plex->vecghostperm);CHKERRQ(ierr);
  ierr = ISSetType(plex->vecghostperm,ISGENERAL);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(plex->vecghostperm, pEnd - pStart,perm,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISSetPermutation(plex->vecghostperm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    DMPlexCreateGhostVector - Creates a ghost vector for a DMPLEX

    Input Parameter:
.    DM - the DMPLEX object

    Output Parameter:
.    v - the ghosted vector

    Level: basic

    Notes:
      Must be called after DMCreateDS()

.seealso: DMPlexSetUseVecGhostPermutation(), VecGhostGetLocalForm(), VecGhostRestoreLocalForm(), VecGhostUpdateBegin(),
          VecCreateGhostWithArray(), VecGhostUpdateEnd()
@*/
PetscErrorCode DMPlexCreateGhostVector(DM dm,Vec *v)
{
  PetscErrorCode ierr;
  PetscInt       nghosts = 0,*ghosts,nextra = 0;
  MPI_Comm       comm;
  PetscSection   section,localsection;
  PetscInt       localSize, globalSize, pStart, pEnd, p, dof, idx, i;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);

  /*
       ghosts are the global numbers of the roots on the other processes we receive from
  */
  ierr = DMGetGlobalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &globalSize);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section,&pStart,&pEnd);CHKERRQ(ierr);
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
    if (dof < 0) {
      nghosts += -(dof+1);
    }
  }
  ierr   = PetscMalloc1(nghosts,&ghosts);CHKERRQ(ierr);
  nghosts = 0;
  for (p=pStart; p<pEnd; p++) {
    ierr = PetscSectionGetDof(section,p,&dof);CHKERRQ(ierr);
    if (dof < 0) {
      ierr = PetscSectionGetOffset(section,p,&idx);CHKERRQ(ierr);
      dof = -(dof+1);
      for (i=0; i<dof; i++) {
        ghosts[nghosts++] = -(idx+1) + i;
      }
    }
  }

  ierr = DMGetLocalSection(dm, &localsection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(localsection, &localSize);CHKERRQ(ierr);
  nextra = localSize - nghosts - globalSize;
  ierr = VecCreateGhost(comm,globalSize,PETSC_DETERMINE,nghosts,ghosts,nextra,v);CHKERRQ(ierr);
  ierr = PetscFree(ghosts);CHKERRQ(ierr);
  ierr = VecSetDM(*v, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
