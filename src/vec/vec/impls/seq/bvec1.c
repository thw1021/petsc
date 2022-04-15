
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
  PetscBLASInt       one = 1,bn = 0;

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
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&aptr,NULL));
  if (*aptr == (PetscScalar)0.0) {
    PetscCall(VecSet_Seq(xin,alpha,dctx));
  } else if (*aptr != (PetscScalar)1.0) {
    PetscScalar  *xarray;
    PetscBLASInt  one = 1;
    PetscBLASInt  bn;

    PetscCall(PetscBLASIntCast(xin->map->n,&bn));
    PetscCall(VecGetArray(xin,&xarray));
    PetscStackCallBLAS("BLASscal",BLASscal_(&bn,aptr,xarray,&one));
    PetscCall(VecRestoreArray(xin,&xarray));
    PetscCall(PetscLogFlops(bn));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_Seq(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx)
{
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&aptr,NULL));
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (*aptr != (PetscScalar)0.0) {
    const PetscScalar *xarray;
    PetscScalar       *yarray;
    PetscBLASInt       one = 1,bn;

    PetscCall(PetscBLASIntCast(yin->map->n,&bn));
    PetscCall(VecGetArrayRead(xin,&xarray));
    PetscCall(VecGetArray(yin,&yarray));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bn,aptr,xarray,&one,yarray,&one));
    PetscCall(VecRestoreArrayRead(xin,&xarray));
    PetscCall(VecRestoreArray(yin,&yarray));
    PetscCall(PetscLogFlops(2.0*bn));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_Seq(Vec yin, PetscManagedScalar a, PetscManagedScalar b, Vec xin, PetscDeviceContext dctx)
{
  PetscScalar *aptr,*bptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,a,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&aptr,NULL));
  PetscCall(PetscManagedScalarGetValues(dctx,b,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&bptr,NULL));
  {
    const PetscScalar aval = *aptr, bval = *bptr;

    if (aval == (PetscScalar)0.0) {
      PetscCall(VecScale_Seq(yin,b,dctx));
    } else if (bval == (PetscScalar)1.0) {
      PetscCall(VecAXPY_Seq(yin,a,xin,dctx));
    } else if (aval == (PetscScalar)1.0) {
      PetscCall(VecAYPX_Seq(yin,b,xin,dctx));
    } else {
      const PetscInt     n = yin->map->n;
      const PetscScalar *xx;
      PetscInt           flops;
      PetscScalar       *yy;

      PetscCall(VecGetArrayRead(xin,&xx));
      PetscCall(VecGetArray(yin,&yy));
      if (bval == (PetscScalar)0.0) {
        flops = n;
        for (PetscInt i = 0; i < n; ++i) yy[i] = aval*xx[i];
      } else {
        flops = 3*n;
        for (PetscInt i = 0; i < n; ++i) yy[i] = aval*xx[i] + bval*yy[i];
      }
      PetscCall(VecRestoreArrayRead(xin,&xx));
      PetscCall(VecRestoreArray(yin,&yy));
      PetscCall(PetscLogFlops(flops));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec xin, Vec yin, PetscDeviceContext dctx)
{

  PetscScalar *aptr,*bptr,*gptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&aptr,NULL));
  PetscCall(PetscManagedScalarGetValues(dctx,beta,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&bptr,NULL));
  PetscCall(PetscManagedScalarGetValues(dctx,gamma,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,&gptr,NULL));
  {
    const PetscScalar  aval  = *aptr,bval = *bptr,gval = *gptr;
    const PetscInt     n     = zin->map->n;
    const PetscScalar *yy,*xx;
    PetscInt           flops = 4*n; // common case
    PetscScalar       *zz;

    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArrayRead(yin,&yy));
    PetscCall(VecGetArray(zin,&zz));
    if (aval == (PetscScalar)1.0) {
      for (PetscInt i = 0; i < n; ++i) zz[i] = xx[i] + bval*yy[i] + gval*zz[i];
    } else if (gval == (PetscScalar)1.0) {
      for (PetscInt i = 0; i < n; ++i) zz[i] = aval*xx[i] + bval*yy[i] + zz[i];
    } else if (gval == (PetscScalar)0.0) {
      for (PetscInt i = 0; i < n; ++i) zz[i] = aval*xx[i] + bval*yy[i];
      flops -= n;
    } else {
      for (PetscInt i = 0; i < n; ++i) zz[i] = aval*xx[i] + bval*yy[i] + gval*zz[i];
      flops += n;
    }
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArrayRead(yin,&yy));
    PetscCall(VecRestoreArray(zin,&zz));
    PetscCall(PetscLogFlops(flops));
  }
  PetscFunctionReturn(0);
}
