!
!  Part of the base include file for Fortran use of PETSc Vec
!  Note: This file should contain only define statements

! No spaces for #defines as some compilers (PGI) also adds
! those additional spaces during preprocessing - bad for fixed format
!
#if !defined (PETSCVECBASEDEF_H)
#define PETSCVECBASEDEF_H

#define VecScatter PetscSF
#define VecScatterType PetscSFType
#define tVecScatter tPetscSF

#endif
