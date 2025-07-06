!     Contributed by Noem T
program test
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdm.h>
use PETScDM
use PETScDMplex
implicit none
DM       :: dm
PetscInt :: numCells
PetscInt :: cStart
PetscInt :: numVertices, numCorners
PetscErrorCode :: ierr
PetscInt, parameter :: numDim = 2
PetscReal,    parameter :: eps = 5.0 * epsilon(1.0)
PetscCallA(PetscInitialize(ierr))
triangle: block

!
! Single triangle element
! Corner coordinates (1, 1), (5, 5), (3, 6)
! Using a 3-point quadrature rule
!
PetscInt :: i
PetscInt :: zero = 0

! Number of quadrature points
PetscInt, parameter :: tria_qpts = 3
! Quadrature order
PetscInt, parameter :: tria_qorder = 2
! Mapped quadrature points
PetscReal, parameter :: tria_v(tria_qpts*numDim)    = [2.0, 2.5, 3.0, 5.0, 4.0, 4.5]
! Jacobian (constant, repeated tria_qpts times)
PetscReal, parameter :: tria_J(tria_qpts*numDim**2) = [([2.0, 1.0, 2.0, 2.5], i = 1, tria_qpts)]

PetscQuadrature :: q
PetscReal :: J(tria_qpts*numDim**2), invJ(tria_qpts*numDim**2), v(tria_qpts*numDim), detJ(tria_qpts)
PetscReal :: vertexCoords(6)
PetscInt  :: cells(3)

numCells = 1
numCorners = 3
numVertices = 3
cells = [0, 1, 2]
vertexCoords = 1.0 * [1, 1, 5, 5, 3, 6]

PetscCallA(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD,numDim,numCells,numVertices,numCorners,PETSC_FALSE,cells,numDim,vertexCoords,dm,ierr))
PetscCallA(PetscDTSimplexQuadrature(numDim, tria_qorder, PETSCDTSIMPLEXQUAD_DEFAULT, q, ierr))
PetscCallA(DMPlexGetHeightStratum(dm, zero, cStart, PETSC_NULL_INTEGER, ierr))
PetscCallA(DMPlexComputeCellGeometryFEM(dm, cStart, q, v, J, invJ, detJ, ierr))
PetscCheck(all(abs(v - tria_v) < eps), PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Wrong mapped quadrature points (triangle)')
PetscCheck(all(abs(J - tria_J) < eps), PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Wrong jacobian (triangle)')
PetscCallA(PetscQuadratureDestroy(q, ierr))
PetscCallA(DMDestroy(dm,ierr))
end block triangle

quadrilateral: block

!
! Single quadrilateral element
! Corner coordinates (-3, -2), (3, -1), (2, 4), (-2, 3)
! Using a 4-point (2x2) Gauss quadrature rule
!

! Number of quadrature points
PetscInt, parameter :: quad_qpts = 4

PetscQuadrature :: q
PetscReal :: vertexCoords(8)
PetscReal :: J(quad_qpts*numDim**2), invJ(quad_qpts*numDim**2), v(quad_qpts*numDim), detJ(quad_qpts)
PetscReal :: a, b
PetscInt  :: cells(4)
PetscInt  :: nc, npoints
PetscInt  :: k
PetscInt :: zero = 0

numCells = 1
numCorners = 4
numVertices = 4
cells = [0, 1, 2, 3]
vertexCoords = 1.0 * [-3, -2, 3, -1, 2, 4, -2, 3]

PetscCallA(DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD,numDim,numCells,numVertices, numCorners,PETSC_FALSE,cells,numDim,vertexCoords,dm,ierr))
nc = 1
npoints = 2
a = -1.0
b = 1.0
PetscCallA(PetscDTGaussTensorQuadrature(numDim, nc, npoints, a, b, q, ierr))
PetscCallA(DMPlexGetHeightStratum(dm, zero, cStart, PETSC_NULL_INTEGER, ierr))
PetscCallA(DMPlexComputeCellGeometryFEM(dm, cStart, q, v, J, invJ, detJ, ierr))
do k = 1, quad_qpts
  PetscCheck(all(abs(v((k-1)*numDim+1:k*numDim) - quad_v(Gauss_rs(k))) < eps), PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Wrong mapped quadrature points (quadrilateral)')
  PetscCheck(all(abs(J((k-1)*numDim**2+1:k*numDim**2) - quad_J(Gauss_rs(k))) < eps), PETSC_COMM_WORLD, PETSC_ERR_PLIB,  'Wrong jacobian (quadrilateral)')
end do
PetscCallA(PetscQuadratureDestroy(q, ierr))
PetscCallA(DMDestroy(dm,ierr))
end block quadrilateral
PetscCallA(PetscFinalize(ierr))

contains
! Quadrature points in a quadrilateral in [-1,+1]
function Gauss_rs(n)
PetscInt, intent(in) :: n
PetscReal :: Gauss_rs(2)

  PetscReal, parameter :: p = 1.0/sqrt(3.0)

  select case(n)
  case(1)
    Gauss_rs = [-p, -p]
  case(2)
    Gauss_rs = [-p, +p]
  case(3)
    Gauss_rs = [+p, -p]
  case(4)
    Gauss_rs = [+p, +p]
end select
end function Gauss_rs
! Mapped quadrature points
function quad_v(rs)
PetscReal, intent(in) :: rs(2)
PetscReal :: quad_v(2)

  PetscReal :: r, s

  r = rs(1)
  s = rs(2)
  quad_v(1) = -0.5*r*(s-5)       ! sum N_i * x_i
  quad_v(2) = 0.5*(r+5*s+2)      ! sum N_i * y_i

end function quad_v
! Jacobian
function quad_J(rs)
PetscReal, intent(in) :: rs(2)
PetscReal :: quad_J(4)

  PetscReal :: r, s
  PetscReal :: pfive = .5, twopfive = 2.5

  r = rs(1)
  s = rs(2)
  quad_J = [-0.5*(s-5), -0.5*r, pfive, twopfive]
end function quad_J
end program test

! /*TEST
!
! test:
!
! TEST*/
