
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>

static PetscErrorCode VecMXDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx, PetscErrorCode (*VecMXDot_SeqFn)(Vec,PetscInt,const Vec[],PetscManagedScalar,PetscDeviceContext))
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);}
  ierr = VecMXDot_SeqFn(xin,nv,y,PetscManagedScalarCreate(work),dctx);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z.ptr,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {ierr = PetscFree(work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecMXDot_MPI(xin,nv,y,z,dctx,VecMDot_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecMXDot_MPI(xin,nv,y,z,dctx,VecMTDot_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscReal         sum,work = 0.0;
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n   = xin->map->n;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    work = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z.ptr = PetscSqrtReal(sum);
    ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_Seq(xin,NORM_1,PetscManagedRealCreate(&work),dctx);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_Seq(xin,NORM_INFINITY,PetscManagedRealCreate(&work),dctx);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_Seq(xin,NORM_1,PetscManagedRealCreate(temp),dctx);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,PetscManagedRealCreate(temp+1),dctx);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z.ptr,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    z.ptr[1] = PetscSqrtReal(z.ptr[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  ierr = VecMax_Seq(xin,idx,PetscManagedRealCreate(&work),dctx);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIUNI)
  *z.ptr = work;
#else
  /* Find the global max */
  if (!idx.ptr) {
    ierr = MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx.ptr + xin->map->rstart;
    ierr  = MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z.ptr   = out.v;
    *idx.ptr = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_Seq(xin,idx,PetscManagedRealCreate(&work),dctx);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIUNI)
  *z.ptr = work;
#else
  /* Find the global Min */
  if (!idx.ptr) {
    ierr = MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx.ptr + xin->map->rstart;
    ierr  = MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z.ptr   = out.v;
    *idx.ptr = out.i;
  }
#endif
  PetscFunctionReturn(0);
}
