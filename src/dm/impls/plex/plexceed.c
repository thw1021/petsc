#include <petsc/private/dmpleximpl.h>           /*I      "petscdmplex.h"          I*/

/*@C
  DMPlexGetLocalOffsets - Allocate and populate array of local offsets.

  Allocate and populate array of shape [num_elem, elem_size] defining offsets for each value (elem, node) for local vector of dm field. All offsets are in the range [0, l_size - 1]. Caller is responsible for freeing the offsets array.

  Input Parameters:
  dm - The DMPlex object
  domain_label - label for DMPlex domain
  label_value - Stratum value
  height - Height of target cells in DMPlex topology
  dm_field - Index of DMPlex field

  Output Parameters:
  num_elem - Number of local elements
  elem_size - Number of dofs per local element
  num_comp - Number of components per dof
  l_size - Size of local vector
  offsets - Allocated offsets array for elements

  Level: developer

@*/
PetscErrorCode DMPlexGetLocalOffsets(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field, PetscInt *num_elem, PetscInt *elem_size, PetscInt *num_comp, PetscInt *l_size, PetscInt **offsets)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscDS      ds = NULL;
  PetscFE      fe;
  PetscSection section;
  PetscInt     *restr_indices;
  const PetscInt *iter_indices;
  IS           iter_is;

  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  if (domain_label) {ierr = DMGetFirstLabelEntry_Internal(dm, dm, domain_label, 1, &label_value, height, NULL, &ds);CHKERRQ(ierr);}

  // Translate dm_field to ds_field
  PetscInt ds_field = -1;
  for (PetscInt i=0; i<dm->Nds; i++) {
    if (!domain_label && !dm->probs[i].label) {
      ds = dm->probs[i].ds;
    }
    if (ds == dm->probs[i].ds) {
      const PetscInt *arr;
      PetscInt nf;
      IS is = dm->probs[i].fields;
      ierr = ISGetIndices(is, &arr);CHKERRQ(ierr);
      ierr = ISGetSize(is, &nf);CHKERRQ(ierr);
      for (PetscInt j=0; j<nf; j++) {
        if (dm_field == arr[j]) {
          ds_field = j;
          break;
        }
      }
      ierr = ISRestoreIndices(is, &arr);CHKERRQ(ierr);
    }
  }
  if (ds_field == -1) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Could not find dm_field %D in DS", dm_field);
  
  {
    PetscInt depth;
    DMLabel depth_label;
    IS depth_is;
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMPlexGetDepthLabel(dm, &depth_label);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(depth_label, depth - height, &depth_is);CHKERRQ(ierr);
    if (domain_label) {
      IS domain_is;
      ierr = DMLabelGetStratumIS(domain_label, label_value, &domain_is);CHKERRQ(ierr);
      if (domain_is) { // domainIS is non-empty
        ierr = ISIntersect(depth_is, domain_is, &iter_is);CHKERRQ(ierr);
        ierr = ISDestroy(&domain_is);CHKERRQ(ierr);
      } else { // domainIS is NULL (empty)
        iter_is = NULL;
      }
      ierr = ISDestroy(&depth_is);CHKERRQ(ierr);
    } else {
      iter_is = depth_is;
    }
    if (iter_is) {
      ierr = ISGetLocalSize(iter_is, num_elem);CHKERRQ(ierr);
      ierr = ISGetIndices(iter_is, &iter_indices);CHKERRQ(ierr);
    } else {
      *num_elem = 0;
      iter_indices = NULL;
    }
  }

  {
    PetscDualSpace dual_space;
    PetscInt num_dual_basis_vectors;
    ierr = PetscDSGetDiscretization(ds, ds_field, (PetscObject*)&fe);CHKERRQ(ierr);
    ierr = PetscFEGetDualSpace(fe, &dual_space);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(dual_space, &num_dual_basis_vectors);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetNumComponents(dual_space, num_comp);CHKERRQ(ierr);
    if (num_dual_basis_vectors % *num_comp != 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for number of dual basis vectors %D not divisible by %D components", num_dual_basis_vectors, *num_comp);
    *elem_size = num_dual_basis_vectors / *num_comp;
  }
  PetscInt restr_size = (*num_elem)*(*elem_size);
  ierr = PetscMalloc1(restr_size, &restr_indices);CHKERRQ(ierr);
  PetscInt elem_offset = 0;
  for (PetscInt p = 0; p < *num_elem; p++) {
    PetscInt c = iter_indices[p];
    PetscInt num_indices, *indices;
    PetscInt field_offsets[17]; // max number of fields plus 1
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL);CHKERRQ(ierr);

    for (PetscInt i = 0; i < *elem_size; i++) {
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = indices[field_offsets[dm_field] + i*(*num_comp)];
      restr_indices[elem_offset++] = loc >= 0 ? loc : -(loc +1);
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &num_indices, &indices, field_offsets, NULL);CHKERRQ(ierr);
  }
  if (elem_offset != restr_size) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_SUP, "Shape mismatch, offsets array of shape (%D, %D) initialized for %D nodes", *num_elem, (*elem_size), elem_offset);
  if (iter_is) { ierr = ISRestoreIndices(iter_is, &iter_indices);CHKERRQ(ierr); }
  ierr = ISDestroy(&iter_is); CHKERRQ(ierr);

  *offsets = restr_indices;
  ierr = PetscSectionGetStorageSize(section, l_size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_LIBCEED)
#include <petscdmplexceed.h>

/* Define the map from the local vector (Lvector) to the cells (Evector) */
PetscErrorCode DMPlexGetCeedRestriction(DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field, CeedElemRestriction *ERestrict)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(ERestrict, 2);
  if (!dm->ceedERestrict) {
    PetscInt     num_elem, elem_size, num_comp, lvec_size, *restr_indices;
    CeedElemRestriction elem_restr;
    Ceed         ceed;

    ierr = DMPlexGetLocalOffsets(dm, domain_label, label_value, height, dm_field, &num_elem, &elem_size, &num_comp, &lvec_size, &restr_indices); CHKERRQ(ierr);

    ierr = DMGetCeed(dm, &ceed);CHKERRQ(ierr);
    ierr = CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, lvec_size, CEED_MEM_HOST, CEED_COPY_VALUES, restr_indices, &elem_restr);CHKERRQ_CEED(ierr);
    ierr = PetscFree(restr_indices);CHKERRQ(ierr);
  }
  *ERestrict = dm->ceedERestrict;
  PetscFunctionReturn(0);
}

#endif
