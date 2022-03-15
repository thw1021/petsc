
/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
#include <petscblaslapack.h>

PetscErrorCode VecDot_Seq(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext PETSC_UNUSED dctx)
{
  const PetscScalar *ya,*xa;
  PetscBLASInt      one = 1,bn = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc the second */
  PetscStackCallBLAS("BLASdot",*z.ptr = BLASdot_(&bn,ya,&one,xa,&one));
  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  if (xin->map->n > 0) {ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_Seq(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx)
{
  const PetscScalar *ya,*xa;
  PetscBLASInt      one = 1,bn = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&ya);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASdot",*z.ptr = BLASdotu_(&bn,xa,&one,ya,&one));
  ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&ya);CHKERRQ(ierr);
  if (xin->map->n > 0) {ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_Seq(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx)
{
  PetscErrorCode ierr;
  PetscBLASInt   bn;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (PetscManagedScalarEq(alpha,0.0)) {
    ierr = VecSet_Seq(xin,alpha,dctx);CHKERRQ(ierr);
  } else if (!PetscManagedScalarEq(alpha,1.0)) {
    PetscScalar  *xarray;
    PetscBLASInt one = 1;

    ierr = VecGetArray(xin,&xarray);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASscal",BLASscal_(&bn,alpha.ptr,xarray,&one));
    ierr = VecRestoreArray(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(bn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_Seq(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext PETSC_UNUSED dctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (!PetscManagedScalarEq(alpha,0.0)) {
    ierr = VecGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yarray);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bn,alpha.ptr,xarray,&one,yarray,&one));
    ierr = VecRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*bn);CHKERRQ(ierr);
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
    ierr = VecScale_Seq(yin,b,dctx);CHKERRQ(ierr);
  } else if (PetscManagedScalarEq(b,1.0)) {
    ierr = VecAXPY_Seq(yin,a,xin,dctx);CHKERRQ(ierr);
  } else if (PetscManagedScalarEq(a,1.0)) {
    ierr = VecAYPX_Seq(yin,b,xin,dctx);CHKERRQ(ierr);
  } else if (PetscManagedScalarEq(b,0.0)) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
    for (PetscInt i = 0; i < n; ++i) yy[i] = (*a.ptr)*xx[i];
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
    for (PetscInt i = 0; i < n; ++i) yy[i] = (*a.ptr)*xx[i] + (*b.ptr)*yy[i];
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec xin, Vec yin, PetscDeviceContext dctx)
{
  PetscErrorCode    ierr;
  PetscInt          n = zin->map->n,i;
  const PetscScalar *yy,*xx;
  PetscScalar       *zz;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(zin,&zz);CHKERRQ(ierr);
  if (PetscManagedScalarEq(alpha,1.0)) {
    for (i=0; i<n; i++) zz[i] = xx[i] + (*beta.ptr)*yy[i] + (*gamma.ptr)*zz[i];
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (PetscManagedScalarEq(gamma,1.0)) {
    for (i=0; i<n; i++) zz[i] = (*alpha.ptr)*xx[i] + (*beta.ptr)*yy[i] + zz[i];
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (PetscManagedScalarEq(gamma,0.0)) {
    for (i=0; i<n; i++) zz[i] = (*alpha.ptr)*xx[i] + (*beta.ptr)*yy[i];
    ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
  } else {
    for (i=0; i<n; i++) zz[i] = (*alpha.ptr)*xx[i] + (*beta.ptr)*yy[i] + (*gamma.ptr)*zz[i];
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(zin,&zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
