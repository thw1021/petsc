
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>

static PetscErrorCode VecMXDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx, PetscErrorCode (*VecMXDot_SeqFn)(Vec,PetscInt,const Vec[],PetscManagedScalar,PetscDeviceContext))
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) PetscCall(PetscMalloc1(nv,&work));
  PetscCall(VecMXDot_SeqFn(xin,nv,y,PetscManagedScalarCreate(work),dctx));
  PetscCall(MPIU_Allreduce(work,z.ptr,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI(xin,nv,y,z,dctx,VecMDot_Seq));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI(xin,nv,y,z,dctx,VecMTDot_Seq));
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscReal         sum,work = 0.0;
  const PetscScalar *xx;
  PetscInt          n   = xin->map->n;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&bn));
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCall(VecGetArrayRead(xin,&xx));
    work = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    *z.ptr = PetscSqrtReal(sum);
    PetscCall(PetscLogFlops(2.0*xin->map->n));
  } else if (type == NORM_1) {
    /* Find the local part */
    PetscCall(VecNorm_Seq(xin,NORM_1,PetscManagedRealCreate(&work),dctx));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    PetscCall(VecNorm_Seq(xin,NORM_INFINITY,PetscManagedRealCreate(&work),dctx));
    /* Find the global max */
    PetscCall(MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    PetscCall(VecNorm_Seq(xin,NORM_1,PetscManagedRealCreate(temp),dctx));
    PetscCall(VecNorm_Seq(xin,NORM_2,PetscManagedRealCreate(temp+1),dctx));
    temp[1] = temp[1]*temp[1];
    PetscCall(MPIU_Allreduce(temp,z.ptr,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    z.ptr[1] = PetscSqrtReal(z.ptr[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  PetscCall(VecMax_Seq(xin,idx,PetscManagedRealCreate(&work),dctx));
#if defined(PETSC_HAVE_MPIUNI)
  *z.ptr = work;
#else
  /* Find the global max */
  if (!idx.ptr) {
    PetscCall(MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx.ptr + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin)));
    *z.ptr   = out.v;
    *idx.ptr = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  PetscCall(VecMin_Seq(xin,idx,PetscManagedRealCreate(&work),dctx));
#if defined(PETSC_HAVE_MPIUNI)
  *z.ptr = work;
#else
  /* Find the global Min */
  if (!idx.ptr) {
    PetscCall(MPIU_Allreduce(&work,z.ptr,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx.ptr + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin)));
    *z.ptr   = out.v;
    *idx.ptr = out.i;
  }
#endif
  PetscFunctionReturn(0);
}
