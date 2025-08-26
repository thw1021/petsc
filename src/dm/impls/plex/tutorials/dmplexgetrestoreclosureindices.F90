!
!   Split a tetrahedron using transform type "refine_tobox" to obtain 4 hexahedra,
!   all of which share a face (4 nodes) with each other.
!   Use DMPlexGetClosureIndices to get DOF of all cells, and check that there are
!   exactly 12 indices shared (1 face = 4 nodes x 3 DOF)
!
program main
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petsc.h>

  use PETScDM
  use PETScDMplex

  implicit none

  DM :: dm, trdm, cdm
  PetscInt :: c, cStart, cEnd
  PetscInt :: cdim, nIdx, idx, cnt, Nf
  PetscInt, parameter :: sharedNodes = 4, zero = 0
  PetscSection :: gS
  PetscErrorCode :: ierr
  DMPlexTransform :: tr

  PetscInt, allocatable :: idxMatrix(:,:), offsets(:)
  PetscInt, pointer, dimension(:) :: indices

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(DMSetType(dm, DMPLEX, ierr))
  PetscCallA(DMSetFromOptions(dm, ierr))

  PetscCallA(DMPlexTransformCreate(PETSC_COMM_WORLD, tr, ierr))
  PetscCallA(DMPlexTransformSetFromOptions(tr, ierr))
  PetscCallA(DMPlexTransformSetDM(tr, dm, ierr))
  PetscCallA(DMPlexTransformSetUp(tr, ierr))
  PetscCallA(DMPlexTransformApply(tr, dm, trdm, ierr))

  PetscCallA(DMGetCoordinateDM(trdm, cdm, ierr))
  PetscCallA(DMGetCoordinateDim(cdm, cdim, ierr))
  PetscCallA(DMGetGlobalSection(cdm, gS, ierr))

  PetscCallA(DMPlexGetHeightStratum(trdm, zero, cStart, cEnd, ierr))
  PetscCallA(PetscSectionGetNumFields(gS, Nf, ierr))
  allocate(offsets(Nf))

  ! Indices per cell
  PetscCallA(DMPlexGetClosureIndices(cdm, gS, gS, cStart, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
  allocate(idxMatrix(nIdx, cEnd - cStart))
  idxMatrix(1:nIdx, cStart + 1) = indices
  PetscCallA(DMPlexRestoreClosureIndices(cdm, gS, gS, cStart, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
  do c = cStart + 1, cEnd - 1
    PetscCallA(DMPlexGetClosureIndices(cdm, gS, gS, c, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
    idxMatrix(1:nIdx, c + 1) = indices
    ! Check size and content of output field offsets array
    PetscCheck(size(offsets) == 1 .and. offsets(1) == zero, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wrong field offsets")
    PetscCallA(DMPlexRestoreClosureIndices(cdm, gS, gS, c, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
  end do

  ! Check number of shared indices between cell 0 and all others
  do c = cStart + 1, cEnd - 1
    cnt = 0
    do idx = 1, nIdx
      cnt = cnt + count(idxMatrix(idx, 1) == idxMatrix(1:nIdx, c + 1))
    end do
    PetscCheck(cnt == sharedNodes * cdim, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wrong DOF indices")
  end do

  PetscCallA(DMPlexTransformDestroy(tr, ierr))
  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))

end program main

! /*TEST
!
! test:
!   args : -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/gmsh-tet.msh  -dm_plex_transform_type refine_tobox
!   output_file: output/empty.out
!
! TEST*/
