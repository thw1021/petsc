
/*
   This file contains routines for Parallel vector operations.
 */
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <petsc/private/cudavecimpl.h>

/*MC
   VECCUDA - VECCUDA = "cuda" - A VECSEQCUDA on a single-process communicator, and VECMPICUDA otherwise.

   Options Database Keys:
. -vec_type cuda - sets the vector type to VECCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQCUDA, VECMPICUDA, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecDestroy_MPICUDA(Vec v)
{
  Vec_MPI        *vecmpi = (Vec_MPI*)v->data;
  Vec_CUDA       *veccuda;
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  if (v->spptr) {
    veccuda = (Vec_CUDA*)v->spptr;
    if (veccuda->GPUarray_allocated) {
      err = cudaFree(((Vec_CUDA*)v->spptr)->GPUarray_allocated);CHKERRCUDA(err);
      veccuda->GPUarray_allocated = NULL;
    }
    if (veccuda->stream) {
      err = cudaStreamDestroy(((Vec_CUDA*)v->spptr)->stream);CHKERRCUDA(err);
    }
    if (v->pinned_memory) {
      ierr = PetscMallocSetCUDAHost();CHKERRQ(ierr);
      ierr = PetscFree(vecmpi->array_allocated);CHKERRQ(ierr);
      ierr = PetscMallocResetCUDAHost();CHKERRQ(ierr);
      v->pinned_memory = PETSC_FALSE;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  if (v->workscalars_d) {
    ierr = v->ops->freeworkscalars(v);CHKERRQ(ierr);
    v->workscalars_d=NULL;
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPICUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqCUDA(xin,NORM_2,&work);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqCUDA(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqCUDA(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqCUDA(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqCUDA(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqCUDA(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPICUDA(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqCUDA(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   VECMPICUDA - VECMPICUDA = "mpicuda" - The basic parallel vector, modified to use CUDA

   Options Database Keys:
. -vec_type mpicuda - sets the vector type to VECMPICUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/


PetscErrorCode VecDuplicate_MPICUDA(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPICUDA_Private(*v,PETSC_TRUE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI*)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep);CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  /* change type_name appropriately */
  ierr = VecCUDAAllocateCheck(*v);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)(*v),VECMPICUDA);CHKERRQ(ierr);

  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPICUDA(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqCUDA(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRQ(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

/* ======================================================
                  Async versions
  =======================================================
*/
__global__ static void PetscCudaSqr (PetscReal *r,const PetscReal *a) {r[0] = a[0]*a[0];}
__global__ static void PetscCudaSqrt(PetscReal *r,const PetscReal *a) {r[0] = sqrt(a[0]);}

PetscErrorCode VecDotAsync_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDotAsync_SeqCUDA(xin,yin,z);CHKERRQ(ierr);
  cerr = cudaStreamSynchronize(PetscDefaultCudaStream);CHKERRCUDA(cerr); /* Make sure z is ready for MPI */
  ierr = MPIU_Allreduce(MPI_IN_PLACE,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDotAsync_MPICUDA(Vec xin,Vec yin,PetscScalar *z)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDotAsync_SeqCUDA(xin,yin,z);CHKERRQ(ierr);
  cerr = cudaStreamSynchronize(PetscDefaultCudaStream);CHKERRCUDA(cerr); /* Make sure z is ready for MPI */
  ierr = MPIU_Allreduce(MPI_IN_PLACE,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNormAsync_MPICUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  /* Find the local part */
  ierr = VecNormAsync_SeqCUDA(xin,type,z);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCudaSqr<<<1,1,0,PetscDefaultCudaStream>>>(z,z);
    cerr = cudaStreamSynchronize(PetscDefaultCudaStream);CHKERRCUDA(cerr); /* Make sure z is ready for MPI */
    ierr = MPIU_Allreduce(MPI_IN_PLACE,z,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
    PetscCudaSqrt<<<1,1,0,PetscDefaultCudaStream>>>(z,z); /* MPI should make sure z is ready for any stream */
  } else if (type == NORM_1) {
    /* Find the global sum */
    cerr = cudaStreamSynchronize(PetscDefaultCudaStream);CHKERRCUDA(cerr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,z,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the global max */
    cerr = cudaStreamSynchronize(PetscDefaultCudaStream);CHKERRCUDA(cerr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,z,1,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscCudaSqr<<<1,1,0,PetscDefaultCudaStream>>>(z+1,z+1);
    cerr = cudaStreamSynchronize(PetscDefaultCudaStream);CHKERRCUDA(cerr);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,z,2,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
    PetscCudaSqrt<<<1,1,0,PetscDefaultCudaStream>>>(z+1,z+1);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_NVSHMEM)
PetscErrorCode VecAllocateWorkScalars_NVSHMEM(Vec x)
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  ierr = PetscNvshmemInitializeCheck();CHKERRQ(ierr);
  ierr = PetscNvshmemMalloc(VEC_MAX_WORK_SCALARS*sizeof(PetscScalar),(void**)&x->workscalars_d);CHKERRQ(ierr);
  for (i=0; i<VEC_MAX_WORK_SCALARS; i++) {
    if (x->workscalars_inuse[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Work scalars are not restored on host before getting new ones on device");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecFreeWorkScalars_NVSHMEM(Vec x)
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;
  for (i=0; i<VEC_MAX_WORK_SCALARS; i++) {
    if (x->workscalars_inuse[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Some work scalars are still in use before free");
  }
  ierr = PetscNvshmemFree(x->workscalars_d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Check if the scalar was allocated by VecGetWorkScalar/Real/Norm() on this vector */
PETSC_STATIC_INLINE PetscErrorCode VecAllocatedScalar_NVSHMEM(Vec xin,const PetscScalar *z,PetscBool *allocated)
{
  PetscInt64 offset;

  PetscFunctionBegin;
  offset     = (PetscInt64)(z - xin->workscalars_d);
  *allocated = (0<= offset && offset < VEC_MAX_WORK_SCALARS)? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotAsync_NVSHMEM(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode ierr;
  PetscScalar    *beta = z;
  PetscBool      allocated = PETSC_TRUE;

  PetscFunctionBegin;
  //ierr = VecAllocatedScalar_NVSHMEM(xin,z,&allocated);CHKERRQ(ierr);
  if (!allocated) {ierr = VecGetWorkScalar(xin,&beta);CHKERRQ(ierr);} /* So that beta is in nvshmem */
  ierr = VecDotAsync_SeqCUDA(xin,yin,beta);CHKERRQ(ierr);
  ierr = PetscNvshmemSum(1,beta,beta);CHKERRQ(ierr);
  if (!allocated) {
    ierr = VecAssignWorkScalar(xin,z,beta);CHKERRQ(ierr);
    ierr = VecRestoreWorkScalar(xin,&beta);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDotAsync_NVSHMEM(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode ierr;
  PetscScalar    *beta = z;
  PetscBool      allocated = PETSC_TRUE;

  PetscFunctionBegin;
  //ierr = VecAllocatedScalar_NVSHMEM(xin,z,&allocated);CHKERRQ(ierr);
  if (!allocated) {ierr = VecGetWorkScalar(xin,&beta);CHKERRQ(ierr);} /* So that beta is in nvshmem */
  ierr = VecTDotAsync_SeqCUDA(xin,yin,beta);CHKERRQ(ierr);
  ierr = PetscNvshmemSum(1,beta,beta);CHKERRQ(ierr);
  if (!allocated) {
    ierr = VecAssignWorkScalar(xin,z,beta);CHKERRQ(ierr);
    ierr = VecRestoreWorkScalar(xin,&beta);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecNormAsync_NVSHMEM(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      *beta = z;
  PetscBool      allocated = PETSC_TRUE;

  PetscFunctionBegin;
  /* Find the local part */
  //ierr = VecAllocatedScalar_NVSHMEM(xin,(PetscScalar*)z,&allocated);CHKERRQ(ierr);
  if (!allocated) {ierr = VecGetWorkNorm(xin,type,&beta);CHKERRQ(ierr);}
  ierr = VecNormAsync_SeqCUDA(xin,type,beta);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCudaSqr<<<1,1,0,PetscDefaultCudaStream>>>(beta,beta);
    ierr = PetscNvshmemSum(1,beta,beta);CHKERRQ(ierr); /* Compute sum on PETSC_COMM_WORLD in PetscDefaultCudaStream */
    PetscCudaSqrt<<<1,1,0,PetscDefaultCudaStream>>>(beta,beta);
  } else if (type == NORM_1) {
    /* Find the global sum */
    ierr = PetscNvshmemSum(1,beta,beta);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the global max */
    ierr = PetscNvshmemMax(1,beta,beta);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscCudaSqr<<<1,1,0,PetscDefaultCudaStream>>>(beta+1,beta+1);
    ierr = PetscNvshmemSum(2,beta,beta);CHKERRQ(ierr);
    PetscCudaSqrt<<<1,1,0,PetscDefaultCudaStream>>>(beta+1,beta+1);
  }
  if (!allocated) {
    ierr = VecAssignWorkReal(xin,z,beta);CHKERRQ(ierr);
    ierr = VecRestoreWorkNorm(xin,type,&beta);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode VecCreate_MPICUDA(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(vv->map);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(vv);CHKERRQ(ierr);
  ierr = VecCreate_MPICUDA_Private(vv,PETSC_FALSE,0,((Vec_CUDA*)vv->spptr)->GPUarray_allocated);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheckHost(vv);CHKERRQ(ierr);
  ierr = VecSet(vv,0.0);CHKERRQ(ierr);
  ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
  vv->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_CUDA(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQCUDA);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPICUDA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPICUDAWithArray - Creates a parallel, array-style vector,
   where the user provides the GPU array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  array - the user provided GPU array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecCUDAPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/
PetscErrorCode  VecCreateMPICUDAWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == PETSC_DECIDE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must set local size of vector");
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = VecCreate(comm,vv);CHKERRQ(ierr);
  ierr = VecSetSizes(*vv,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vv,bs);CHKERRQ(ierr);
  ierr = VecCreate_MPICUDA_Private(*vv,PETSC_FALSE,0,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateMPICUDAWithArrays - Creates a parallel, array-style vector,
   where the user provides the GPU array space to store the vector values.

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - block size, same meaning as VecSetBlockSize()
.  n     - local vector length, cannot be PETSC_DECIDE
.  N     - global vector length (or PETSC_DECIDE to have calculated)
-  cpuarray - the user provided CPU array to store the vector values
-  gpuarray - the user provided GPU array to store the vector values

   Output Parameter:
.  vv - the vector

   Notes:
   If both cpuarray and gpuarray are provided, the caller must ensure that
   the provided arrays have identical values.

   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   PETSc does NOT free the provided arrays when the vector is destroyed via
   VecDestroy(). The user should not free the array until the vector is
   destroyed.

   Level: intermediate

.seealso: VecCreateSeqCUDAWithArrays(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecCUDAPlaceArray(), VecPlaceArray(),
          VecCUDAAllocateCheckHost()
@*/
PetscErrorCode  VecCreateMPICUDAWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,PetscInt N,const PetscScalar cpuarray[],const PetscScalar gpuarray[],Vec *vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateMPICUDAWithArray(comm,bs,n,N,gpuarray,vv);CHKERRQ(ierr);

  if (cpuarray && gpuarray) {
    Vec_MPI *s         = (Vec_MPI*)((*vv)->data);
    s->array           = (PetscScalar*)cpuarray;
    (*vv)->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (cpuarray) {
    Vec_MPI *s         = (Vec_MPI*)((*vv)->data);
    s->array           = (PetscScalar*)cpuarray;
    (*vv)->offloadmask =  PETSC_OFFLOAD_CPU;
  } else if (gpuarray) {
    (*vv)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*vv)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode VecBindToCPU_MPICUDA(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = pin;
  if (pin) {
    ierr = VecCUDACopyFromGPU(V);CHKERRQ(ierr);
    V->offloadmask = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dotnorm2               = NULL;
    V->ops->waxpy                  = VecWAXPY_Seq;
    V->ops->dot                    = VecDot_MPI;
    V->ops->mdot                   = VecMDot_MPI;
    V->ops->tdot                   = VecTDot_MPI;
    V->ops->norm                   = VecNorm_MPI;
    V->ops->scale                  = VecScale_Seq;
    V->ops->copy                   = VecCopy_Seq;
    V->ops->set                    = VecSet_Seq;
    V->ops->swap                   = VecSwap_Seq;
    V->ops->axpy                   = VecAXPY_Seq;
    V->ops->axpby                  = VecAXPBY_Seq;
    V->ops->maxpy                  = VecMAXPY_Seq;
    V->ops->aypx                   = VecAYPX_Seq;
    V->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult          = VecPointwiseMult_Seq;
    V->ops->setrandom              = VecSetRandom_Seq;
    V->ops->placearray             = VecPlaceArray_Seq;
    V->ops->replacearray           = VecReplaceArray_SeqCUDA;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->dot_local              = VecDot_Seq;
    V->ops->tdot_local             = VecTDot_Seq;
    V->ops->norm_local             = VecNorm_Seq;
    V->ops->mdot_local             = VecMDot_Seq;
    V->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;

    V->ops->allocateworkscalars    = NULL;
    V->ops->freeworkscalars        = NULL;

    V->ops->getworkscalar          = NULL;
    V->ops->restoreworkscalar      = NULL;
    V->ops->getworknorm            = NULL;
    V->ops->restoreworknorm        = NULL;

    V->ops->assignworkscalar       = NULL;
    V->ops->assignworkreal         = NULL;
    V->ops->setworkscalar          = NULL;
    V->ops->setworkreal            = NULL;
    V->ops->copyworkscalartohost   = NULL;
    V->ops->copyworkrealtohost     = NULL;
    V->ops->addworkscalar          = NULL;
    V->ops->subworkscalar          = NULL;
    V->ops->multworkscalar         = NULL;
    V->ops->divideworkscalar       = NULL;
    V->ops->sqrworkreal            = NULL;
    V->ops->sqrtworkreal           = NULL;

    V->ops->dot_async              = NULL;
    V->ops->tdot_async             = NULL;
    V->ops->norm_async             = NULL;
    V->ops->axpy_async             = NULL;
    V->ops->aypx_async             = NULL;
  } else {
    V->ops->dotnorm2               = VecDotNorm2_MPICUDA;
    V->ops->waxpy                  = VecWAXPY_SeqCUDA;
    V->ops->duplicate              = VecDuplicate_MPICUDA;
    V->ops->dot                    = VecDot_MPICUDA;
    V->ops->mdot                   = VecMDot_MPICUDA;
    V->ops->tdot                   = VecTDot_MPICUDA;
    V->ops->norm                   = VecNorm_MPICUDA;
    V->ops->scale                  = VecScale_SeqCUDA;
    V->ops->copy                   = VecCopy_SeqCUDA;
    V->ops->set                    = VecSet_SeqCUDA;
    V->ops->swap                   = VecSwap_SeqCUDA;
    V->ops->axpy                   = VecAXPY_SeqCUDA;
    V->ops->axpby                  = VecAXPBY_SeqCUDA;
    V->ops->maxpy                  = VecMAXPY_SeqCUDA;
    V->ops->aypx                   = VecAYPX_SeqCUDA;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqCUDA;
    V->ops->pointwisemult          = VecPointwiseMult_SeqCUDA;
    V->ops->setrandom              = VecSetRandom_SeqCUDA;
    V->ops->placearray             = VecPlaceArray_SeqCUDA;
    V->ops->replacearray           = VecReplaceArray_SeqCUDA;
    V->ops->resetarray             = VecResetArray_SeqCUDA;
    V->ops->dot_local              = VecDot_SeqCUDA;
    V->ops->tdot_local             = VecTDot_SeqCUDA;
    V->ops->norm_local             = VecNorm_SeqCUDA;
    V->ops->mdot_local             = VecMDot_SeqCUDA;
    V->ops->destroy                = VecDestroy_MPICUDA;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqCUDA;
    V->ops->getlocalvector         = VecGetLocalVector_SeqCUDA;
    V->ops->restorelocalvector     = VecRestoreLocalVector_SeqCUDA;
    V->ops->getlocalvectorread     = VecGetLocalVector_SeqCUDA;
    V->ops->restorelocalvectorread = VecRestoreLocalVector_SeqCUDA;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqCUDA;
    V->ops->getarray               = VecGetArray_SeqCUDA;
    V->ops->restorearray           = VecRestoreArray_SeqCUDA;
    V->ops->getarrayandmemtype        = VecGetArrayAndMemType_SeqCUDA;
    V->ops->restorearrayandmemtype    = VecRestoreArrayAndMemType_SeqCUDA;
   #if defined(PETSC_HAVE_NVSHMEM)
    Vec_MPI *vecmpi = (Vec_MPI*)V->data;
    if (vecmpi->use_nvshmem) {
      V->ops->allocateworkscalars  = VecAllocateWorkScalars_NVSHMEM;
      V->ops->freeworkscalars      = VecFreeWorkScalars_NVSHMEM;
      V->ops->dot_async            = VecDotAsync_NVSHMEM; /* Involves communication */
      V->ops->tdot_async           = VecTDotAsync_NVSHMEM;
      V->ops->norm_async           = VecNormAsync_NVSHMEM;
    } else
   #endif
    {
      V->ops->allocateworkscalars  = VecAllocateWorkScalars_SeqCUDA;
      V->ops->freeworkscalars      = VecFreeWorkScalars_SeqCUDA;
      V->ops->dot_async            = VecDotAsync_MPICUDA;
      V->ops->tdot_async           = VecTDotAsync_MPICUDA;
      V->ops->norm_async           = VecNormAsync_MPICUDA;
    }

    V->ops->getworknorm            = VecGetWorkNorm_SeqCUDA;
    V->ops->restoreworknorm        = VecRestoreWorkNorm_SeqCUDA;
    V->ops->getworkscalar          = VecGetWorkScalar_SeqCUDA;
    V->ops->restoreworkscalar      = VecRestoreWorkScalar_SeqCUDA;

    V->ops->getworkscalar          = VecGetWorkScalar_SeqCUDA;
    V->ops->restoreworkscalar      = VecRestoreWorkScalar_SeqCUDA;
    V->ops->getworknorm            = VecGetWorkNorm_SeqCUDA;
    V->ops->restoreworknorm        = VecRestoreWorkNorm_SeqCUDA;

    V->ops->assignworkscalar       = VecAssignWorkScalar_SeqCUDA;
    V->ops->assignworkreal         = VecAssignWorkReal_SeqCUDA;
    V->ops->setworkscalar          = VecSetWorkScalar_SeqCUDA;
    V->ops->setworkreal            = VecSetWorkReal_SeqCUDA;
    V->ops->copyworkscalartohost   = VecCopyWorkScalarToHost_SeqCUDA;
    V->ops->copyworkrealtohost     = VecCopyWorkRealToHost_SeqCUDA;
    V->ops->addworkscalar          = VecAddWorkScalar_SeqCUDA;
    V->ops->subworkscalar          = VecSubWorkScalar_SeqCUDA;
    V->ops->multworkscalar         = VecMultWorkScalar_SeqCUDA;
    V->ops->divideworkscalar       = VecDivideWorkScalar_SeqCUDA;
    V->ops->sqrworkreal            = VecSqrWorkReal_SeqCUDA;
    V->ops->sqrtworkreal           = VecSqrtWorkReal_SeqCUDA;

    V->ops->axpy_async             = VecAXPYAsync_SeqCUDA;
    V->ops->aypx_async             = VecAYPXAsync_SeqCUDA;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_MPICUDA_Private(Vec vv,PetscBool alloc,PetscInt nghost,const PetscScalar array[])
{
  PetscErrorCode ierr;
  Vec_CUDA       *veccuda;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPICUDA);CHKERRQ(ierr);

#if defined(PETSC_HAVE_NVSHMEM)
  PetscMPIInt    result;
  Vec_MPI        *vecmpi = (Vec_MPI*)vv->data;

  vecmpi->use_nvshmem = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_nvshmem",&vecmpi->use_nvshmem,NULL);CHKERRQ(ierr);
  if (vecmpi->use_nvshmem) {
    ierr = MPI_Comm_compare(PETSC_COMM_WORLD,PetscObjectComm((PetscObject)vv),&result);CHKERRMPI(ierr);
    if (result != MPI_IDENT && result != MPI_CONGRUENT) vecmpi->use_nvshmem = PETSC_FALSE;
  }
 #endif

  ierr = VecBindToCPU_MPICUDA(vv,PETSC_FALSE);CHKERRQ(ierr);
  vv->ops->bindtocpu = VecBindToCPU_MPICUDA;

  /* Later, functions check for the Vec_CUDA structure existence, so do not create it without array */
  if (alloc && !array) {
    ierr = VecCUDAAllocateCheck(vv);CHKERRQ(ierr);
    ierr = VecCUDAAllocateCheckHost(vv);CHKERRQ(ierr);
    ierr = VecSet(vv,0.0);CHKERRQ(ierr);
    ierr = VecSet_Seq(vv,0.0);CHKERRQ(ierr);
    vv->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  if (array) {
    if (!vv->spptr) {
      PetscReal pinned_memory_min;
      PetscBool flag;
      ierr = PetscCalloc(sizeof(Vec_CUDA),&vv->spptr);CHKERRQ(ierr);
      veccuda = (Vec_CUDA*)vv->spptr;
      vv->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
      vv->minimum_bytes_pinned_memory = 0;

      /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
         Note: This same code duplicated in VecCreate_SeqCUDA_Private() and VecCUDAAllocateCheck(). Is there a good way to avoid this? */
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)vv),((PetscObject)vv)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
      pinned_memory_min = vv->minimum_bytes_pinned_memory;
      ierr = PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&flag);CHKERRQ(ierr);
      if (flag) vv->minimum_bytes_pinned_memory = pinned_memory_min;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    veccuda = (Vec_CUDA*)vv->spptr;
    veccuda->GPUarray = (PetscScalar*)array;
  }
  PetscFunctionReturn(0);
}
