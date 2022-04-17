
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>
#include <petsc/private/deviceimpl.h>

static PetscErrorCode VecMXDot_MPI(Vec xin, PetscInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx, PetscErrorCode (*const VecMXDot_SeqFn)(Vec,PetscInt,const Vec[],PetscManagedScalar,PetscDeviceContext))
{
  PetscFunctionBegin;
  PetscCall(VecMXDot_SeqFn(xin,nv,y,z,dctx));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx,z,(PetscObject)xin,nv,MPIU_SUM));
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
  PetscReal *zptr;
  PetscInt   zn;
  MPI_Op     op = MPIU_SUM;

  PetscFunctionBegin;
  PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,&zptr,&zn));
  PetscAssert(zn >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"%s() needs managed type of size >= 1, have %" PetscInt_FMT,PETSC_FUNCTION_NAME,zn);
  switch (type) {
  case NORM_2:
  case NORM_FROBENIUS: {
    const PetscInt     n   = xin->map->n;
    const PetscScalar *xx;
    PetscBLASInt       one = 1,bn;

    PetscCall(PetscBLASIntCast(n,&bn));
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscStackCallBLAS("BLASDot",*zptr = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one)));
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(PetscLogFlops(2*n));
  } break;
  case NORM_1_AND_2:
    PetscAssert(zn >= 2,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"NORM_1_AND_2 needs managed type of size >= 2, have %" PetscInt_FMT,zn);
  case NORM_1:
  case NORM_INFINITY:
    PetscCall(VecNorm_Seq(xin,type,z,dctx));
    if (type == NORM_INFINITY) {
      op = MPIU_MAX;
    } else if (type == NORM_1_AND_2) {
      zptr[1] *= zptr[1];
      zn       = 2;
    }
    break;
  }
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,zptr,(PetscMPIInt)zn,MPIU_REAL,op,PetscObjectComm((PetscObject)xin)));
  if (type == NORM_1 || type == NORM_FROBENIUS) {
    zptr[0] = PetscSqrtReal(zptr[0]);
  } else if (type == NORM_1_AND_2) {
    zptr[1] = PetscSqrtReal(zptr[1]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMinMax_MPI_Private(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx, MPI_Op ops[2], PetscErrorCode(*const SeqFn)(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext))
{
  PetscReal *zptr;
  PetscInt  *idxptr;

  PetscFunctionBegin;
  /* Find the local min/max */
  PetscCall(SeqFn(xin,idx,z,dctx));
  if (PetscDefined(HAVE_MPIUNI)) PetscFunctionReturn(0);
  PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,&zptr,NULL));
  PetscCall(PetscManagedIntGetValues(dctx,idx,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,&idxptr,NULL));
  /* Find the global min/max */
  if (idx) {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = *zptr;
    in.i  = *idxptr + xin->map->rstart;
    PetscCall(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,ops[0],PetscObjectComm((PetscObject)xin)));
    *zptr   = out.v;
    *idxptr = out.i;
  } else {
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE,zptr,1,MPIU_REAL,ops[1],PetscObjectComm((PetscObject)xin)));
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
