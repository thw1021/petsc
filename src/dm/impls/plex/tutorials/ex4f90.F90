! setting up DMPlex for finite elements
! Contributed by Pratheek Shanthraj <p.shanthraj@mpie.de>
      program main
#include <petsc/finclude/petsc.h>
      use petsc
      implicit none
      DM :: dm
      PetscDS :: prob
      PetscInt :: dim = 3
      PetscBool :: simplex = PETSC_TRUE
      PetscBool :: interpolate = PETSC_TRUE
      PetscReal :: refinementLimit = 0.0
      PetscErrorCode :: ierr
      PetscTabulation, pointer :: tab(:)
      PetscFE fe

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
      PetscCallA(DMPlexCreateDoublet(PETSC_COMM_WORLD, dim, simplex,interpolate, refinementLimit, dm, ierr))
      PetscCallA(PetscFECreateDefault(PETSC_COMM_WORLD, dim, 1, simplex, 'name', -1, fe, ierr));
      PetscCallA(PetscObjectSetName(fe, 'name', ierr));
      PetscCallA(DMSetField(dm, 0, PETSC_NULL_DMLABEL, PetscObjectCast(fe), ierr));
      PetscCallA(DMSetField(dm, 1, PETSC_NULL_DMLABEL, PetscObjectCast(fe), ierr));

      PetscCallA(DMSetUp(dm,ierr))
      PetscCallA(DMCreateDS(dm,ierr))
      PetscCallA(DMGetDS(dm,prob,ierr))
      PetscCallA(PetscDSGetTabulation(prob,tab,ierr))
      print*,'Tab values 1 function',tab(1)%ptr%T(1)%ptr
      print*,'Tab values 1 derivative',tab(1)%ptr%T(2)%ptr
      print*,'Tab values 2 function',tab(2)%ptr%T(1)%ptr
      print*,'Tab values 2 derivative',tab(2)%ptr%T(2)%ptr
      PetscCallA(PetscDSRestoreTabulation(prob,tab,ierr))

      PetscCallA(PetscFEDestroy(fe, ierr));
      PetscCallA(DMDestroy(dm, ierr))
      PetscCallA(PetscFinalize(ierr))
      end program main
!/*TEST
!
!  test:
!    nsize: 1
!
!TEST*/
