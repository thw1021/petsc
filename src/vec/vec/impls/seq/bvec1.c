
/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
#include <petscblaslapack.h>

PetscErrorCode VecDot_Seq(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx)
{
  const PetscScalar *ya,*xa;
  PetscScalar       *zptr;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  PetscCall(VecGetArrayRead(xin,&xa));
  PetscCall(VecGetArrayRead(yin,&ya));
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc the second */
  PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,&zptr,NULL));
  PetscStackCallBLAS("BLASdot",*zptr = BLASdot_(&bn,ya,&one,xa,&one));
  PetscCall(VecRestoreArrayRead(xin,&xa));
  PetscCall(VecRestoreArrayRead(yin,&ya));
  if (xin->map->n > 0) PetscCall(PetscLogFlops(2.0*xin->map->n-1));
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_Seq(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx)
{
  const PetscScalar *ya,*xa;
  PetscScalar       *zptr;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  PetscCall(VecGetArrayRead(xin,&xa));
  PetscCall(VecGetArrayRead(yin,&ya));
  PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,&zptr,NULL));
  PetscStackCallBLAS("BLASdot",*zptr = BLASdotu_(&bn,xa,&one,ya,&one));
  PetscCall(VecRestoreArrayRead(xin,&xa));
  PetscCall(VecRestoreArrayRead(yin,&ya));
  if (xin->map->n > 0) PetscCall(PetscLogFlops(2.0*xin->map->n-1));
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_Seq(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx)
{
  PetscErrorCode ierr;
  PetscBLASInt   bn;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  if (PetscManagedScalarEq(alpha,0.0)) {
    PetscCall(VecSet_Seq(xin,alpha,dctx));
  } else if (!PetscManagedScalarEq(alpha,1.0)) {
    PetscScalar  *xarray;
    PetscBLASInt one = 1;

    PetscCall(VecGetArray(xin,&xarray));
    PetscStackCallBLAS("BLASscal",BLASscal_(&bn,alpha.ptr,xarray,&one));
    PetscCall(VecRestoreArray(xin,&xarray));
  }
  PetscCall(PetscLogFlops(bn));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_Seq(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext PETSC_UNUSED dctx)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(yin->map->n,&bn));
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (!PetscManagedScalarEq(alpha,0.0)) {
    PetscCall(VecGetArrayRead(xin,&xarray));
    PetscCall(VecGetArray(yin,&yarray));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bn,alpha.ptr,xarray,&one,yarray,&one));
    PetscCall(VecRestoreArrayRead(xin,&xarray));
    PetscCall(VecRestoreArray(yin,&yarray));
    PetscCall(PetscLogFlops(2.0*bn));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_Seq(Vec yin, PetscManagedScalar a, PetscManagedScalar b, Vec xin, PetscDeviceContext dctx)
{
  PetscErrorCode    ierr;
  const PetscInt    n = yin->map->n;
  const PetscScalar *xx;
  PetscScalar       *yy;

  PetscFunctionBegin;
  if (PetscManagedScalarEq(a,0.0)) {
    PetscCall(VecScale_Seq(yin,b,dctx));
  } else if (PetscManagedScalarEq(b,1.0)) {
    PetscCall(VecAXPY_Seq(yin,a,xin,dctx));
  } else if (PetscManagedScalarEq(a,1.0)) {
    PetscCall(VecAYPX_Seq(yin,b,xin,dctx));
  } else if (PetscManagedScalarEq(b,0.0)) {
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArray(yin,&yy));
    for (PetscInt i = 0; i < n; ++i) yy[i] = (*a.ptr)*xx[i];
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArray(yin,&yy));
    PetscCall(PetscLogFlops(n));
  } else {
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArray(yin,&yy));
    for (PetscInt i = 0; i < n; ++i) yy[i] = (*a.ptr)*xx[i] + (*b.ptr)*yy[i];
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArray(yin,&yy));
    PetscCall(PetscLogFlops(3.0*n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec xin, Vec yin, PetscDeviceContext dctx)
{
  PetscInt          n = zin->map->n,i;
  const PetscScalar *yy,*xx;
  PetscScalar       *zz;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  PetscCall(VecGetArrayRead(yin,&yy));
  PetscCall(VecGetArray(zin,&zz));
  if (PetscManagedScalarEq(alpha,1.0)) {
    for (i=0; i<n; i++) zz[i] = xx[i] + (*beta.ptr)*yy[i] + (*gamma.ptr)*zz[i];
    PetscCall(PetscLogFlops(4.0*n));
  } else if (PetscManagedScalarEq(gamma,1.0)) {
    for (i=0; i<n; i++) zz[i] = (*alpha.ptr)*xx[i] + (*beta.ptr)*yy[i] + zz[i];
    PetscCall(PetscLogFlops(4.0*n));
  } else if (PetscManagedScalarEq(gamma,0.0)) {
    for (i=0; i<n; i++) zz[i] = (*alpha.ptr)*xx[i] + (*beta.ptr)*yy[i];
    PetscCall(PetscLogFlops(3.0*n));
  } else {
    for (i=0; i<n; i++) zz[i] = (*alpha.ptr)*xx[i] + (*beta.ptr)*yy[i] + (*gamma.ptr)*zz[i];
    PetscCall(PetscLogFlops(5.0*n));
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscCall(VecRestoreArrayRead(yin,&yy));
  PetscCall(VecRestoreArray(zin,&zz));
  PetscFunctionReturn(0);
}
