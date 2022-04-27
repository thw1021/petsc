
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>
#include <petsc/private/deviceimpl.h>

static PetscErrorCode VecMXDot_MPI(Vec xin, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx, PetscErrorCode (*const VecMXDot_SeqFn)(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext))
{
  PetscInt *nvptr;

  PetscFunctionBegin;
  PetscCall(VecMXDot_SeqFn(xin,nv,y,z,dctx));
  PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx,z,nvptr,MPIU_SUM,(PetscObject)xin));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPI(Vec xin, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI(xin,nv,y,z,dctx,VecMDot_Seq));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPI(Vec xin, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI(xin,nv,y,z,dctx,VecMTDot_Seq));
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscReal *zptr;
  PetscInt   zn = 1;
  MPI_Op     op = MPIU_SUM;

  PetscFunctionBegin;
  switch (type) {
  case NORM_2:
  case NORM_FROBENIUS: {
    const PetscInt      n   = xin->map->n;
    const PetscBLASInt  one = 1;
    const PetscScalar  *xx;
    PetscReal           ztmp;
    PetscBLASInt        bn;

    PetscCall(PetscBLASIntCast(n,&bn));
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscStackCallBLAS("BLASDot",ztmp = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one)));
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(PetscLogFlops(2*n));
    PetscCall(PetscManagedRealSetValues(dctx,z,PETSC_MEMTYPE_HOST,&ztmp,1));
  } break;
  case NORM_1_AND_2:
  case NORM_1:
  case NORM_INFINITY:
    PetscCall(VecNorm_Seq(xin,type,z,dctx));
    if (type == NORM_INFINITY) {
      op = MPIU_MAX;
    } else if (type == NORM_1_AND_2) {
      PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&zptr));
      zptr[1] *= zptr[1];
      zn       = 2;
    }
    break;
  }
  PetscCall(PetscDeviceContextAllReduceManagedReal_Internal(dctx,z,&zn,op,(PetscObject)xin));
  if (type == NORM_2 || type == NORM_FROBENIUS || type == NORM_1_AND_2) {
    PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&zptr));
    zptr[type == NORM_1_AND_2] = PetscSqrtReal(zptr[type == NORM_1_AND_2]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMinMax_MPI_Private(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx, MPI_Op ops[2], PetscErrorCode(*const VecMinMax_Seq)(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext))
{
  PetscFunctionBegin;
  /* Find the local min/max */
  PetscCall(VecMinMax_Seq(xin,idx,z,dctx));
  if (PetscDefined(HAVE_MPIUNI)) PetscFunctionReturn(0);
  /* Find the global min/max */
  if (idx) {
    PetscReal *zptr;
    PetscInt  *idxptr;
    struct { PetscReal v; PetscInt i; } in,out;

    PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&zptr));
    PetscCall(PetscManagedIntGetValues(dctx,idx,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&idxptr));
    in.v = zptr[0];
    in.i = idxptr[0] + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,ops[0],PetscObjectComm((PetscObject)xin)));
    *zptr   = out.v;
    *idxptr = out.i;
  } else {
    const PetscInt one = 1;
    PetscCall(PetscDeviceContextAllReduceManagedReal_Internal(dctx,z,&one,ops[1],(PetscObject)xin));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  MPI_Op ops[] = {MPIU_MAXLOC,MPIU_MAX};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Private(xin,idx,z,dctx,ops,VecMax_Seq));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPI(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  MPI_Op ops[] = {MPIU_MINLOC,MPIU_MIN};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Private(xin,idx,z,dctx,ops,VecMin_Seq));
  PetscFunctionReturn(0);
}
