!
!  Include file for Fortran use of the DMDA (distributed array) package in PETSc
!
#if !defined (PETSCDMDADEF_H)
#define PETSCDMDADEF_H

#include "petsc/finclude/petscis.h"
#include "petsc/finclude/petscvec.h"
#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscdm.h"

#define DMDAStencilType PetscEnum

#define DMDALocalInfo type(tDMDALocalInfo)

#define DMDAInterpolationType PetscEnum
#define DMDAElementType PetscEnum

#define XG_RANGE in%GXS + 1:in%GXS + in%GXM
#define YG_RANGE in%GYS + 1:in%GYS + in%GYM
#define ZG_RANGE in%GZS + 1:in%GZS + in%GZM
#define X_RANGE  in%XS  + 1:in%XS  + in%XM
#define Y_RANGE  in%YS  + 1:in%YS  + in%YM
#define Z_RANGE  in%ZS  + 1:in%ZS  + in%ZM

!   depecated API

#define DMDAVecGetArrayF90         DMDAVecGetArray
#define DMDAVecRestoreArrayF90     DMDAVecRestoreArray
#define DMDAVecGetArrayReadF90     DMDAVecGetArrayRead
#define DMDAVecRestoreArrayReadF90 DMDAVecRestoreArrayRead

#endif
