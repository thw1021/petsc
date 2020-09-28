/*
 Implementation of the sequential hip vectors.

 This file contains the code that can be compiled with a C
 compiler.  The companion file vechip2.hip.cpp contains the code that
 must be compiled with hipcc compiler.
 */

#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/hipvecimpl.h>

/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscHIPFlag for the vector
    Does NOT zero the HIP array
 */
extern "C"
PetscErrorCode VecHIPAllocateCheckHost(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec_Seq        *s = (Vec_Seq*)v->data;
  PetscInt       n = v->map->n;

  PetscFunctionBegin;
  if (!s) {
    ierr = PetscNewLog((PetscObject)v,&s);CHKERRQ(ierr);
    v->data = s;
  }
  if (!s->array) {
    ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      v->offloadmask = PETSC_OFFLOAD_CPU;
    }
  }
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecCopy_SeqHIP_Private(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecHIPAllocateCheckHost(xin);CHKERRQ(ierr);
  ierr = VecHIPAllocateCheckHost(yin);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscArraycpy(ya,xa,xin->map->n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecSetRandom_SeqHIP_Private(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) { ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr); }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecDestroy_SeqHIP_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  if (vs) {
    if (vs->array_allocated) { ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr); }
    ierr = PetscFree(vs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecResetArray_SeqHIP_Private(Vec vin)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecHIPAllocateCheck_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPAllocateCheck(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecHIPCopyToGPU_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPCopyToGPU(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    VecHIPCopyToGPUSome_Public - Copies certain entries down to the GPU from the CPU of a vector

   Input Parameters:
 +  v    - the vector
 .  ci   - the requested indices, this should be created with HIPIndicesCreate()
 -  mode - vec scatter mode used in VecScatterBegin/End
*/
extern "C"
PetscErrorCode VecHIPCopyToGPUSome_Public(Vec v,PetscHIPIndices ci,ScatterMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPCopyToGPUSome(v,ci,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  VecHIPCopyFromGPUSome_Public - Copies certain entries up to the CPU from the GPU of a vector

  Input Parameters:
 +  v    - the vector
 .  ci   - the requested indices, this should be created with HIPIndicesCreate()
 -  mode - vec scatter mode used in VecScatterBegin/End
*/
extern "C"
PetscErrorCode VecHIPCopyFromGPUSome_Public(Vec v,PetscHIPIndices ci,ScatterMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPCopyFromGPUSome(v,ci,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecSetRandom_SeqHIP(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetRandom_SeqHIP_Private(xin,r);CHKERRQ(ierr);
  xin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecResetArray_SeqHIP(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecResetArray_SeqHIP_Private(vin);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecPlaceArray_SeqHIP(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecReplaceArray_SeqHIP(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

/*@
 VecCreateSeqHIP - Creates a standard, sequential array-style vector.

 Collective

 Input Parameter:
 +  comm - the communicator, should be PETSC_COMM_SELF
 -  n - the vector length

 Output Parameter:
 .  v - the vector

 Notes:
 Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
 same type as an existing vector.

 Level: intermediate

 .seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
 @*/
extern "C"
PetscErrorCode VecCreateSeqHIP(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQHIP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecDuplicate_SeqHIP(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqHIP(PetscObjectComm((PetscObject)win),win->map->n,V);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecCreate_SeqHIP(Vec V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(V->map);CHKERRQ(ierr);
  ierr = VecHIPAllocateCheck(V);CHKERRQ(ierr);
  ierr = VecCreate_SeqHIP_Private(V,((Vec_HIP*)V->spptr)->GPUarray_allocated);CHKERRQ(ierr);
  ierr = VecHIPAllocateCheckHost(V);CHKERRQ(ierr);
  ierr = VecSet(V,0.0);CHKERRQ(ierr);
  ierr = VecSet_Seq(V,0.0);CHKERRQ(ierr);
  V->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqHIPWithArray - Creates a HIP sequential array-style vector,
   where the user provides the array space to store the vector values. The array
   provided must be a GPU array.

   Collective

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - GPU memory where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecHIPPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateMPIHIPWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
          VecCreateGhost(), VecCreateSeq(), VecHIPPlaceArray(), VecCreateSeqWithArray(),
          VecCreateMPIWithArray()
@*/
extern "C"
PetscErrorCode  VecCreateSeqHIPWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar array[],Vec *V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  ierr = VecCreate_SeqHIP_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecGetArrayWrite_SeqHIP(Vec v,PetscScalar **vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHIPAllocateCheckHost(v);CHKERRQ(ierr);
  v->offloadmask = PETSC_OFFLOAD_CPU;
  *vv = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecBindToCPU_SeqHIP(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = pin;
  if (pin) {
    ierr = VecHIPCopyFromGPU(V);CHKERRQ(ierr);
    V->offloadmask                 = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dot                    = VecDot_Seq;
    V->ops->norm                   = VecNorm_Seq;
    V->ops->tdot                   = VecTDot_Seq;
    V->ops->scale                  = VecScale_Seq;
    V->ops->copy                   = VecCopy_Seq;
    V->ops->set                    = VecSet_Seq;
    V->ops->swap                   = VecSwap_Seq;
    V->ops->axpy                   = VecAXPY_Seq;
    V->ops->axpby                  = VecAXPBY_Seq;
    V->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult          = VecPointwiseMult_Seq;
    V->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    V->ops->setrandom              = VecSetRandom_Seq;
    V->ops->dot_local              = VecDot_Seq;
    V->ops->tdot_local             = VecTDot_Seq;
    V->ops->norm_local             = VecNorm_Seq;
    V->ops->mdot_local             = VecMDot_Seq;
    V->ops->mtdot_local            = VecMTDot_Seq;
    V->ops->maxpy                  = VecMAXPY_Seq;
    V->ops->mdot                   = VecMDot_Seq;
    V->ops->mtdot                  = VecMTDot_Seq;
    V->ops->aypx                   = VecAYPX_Seq;
    V->ops->waxpy                  = VecWAXPY_Seq;
    V->ops->dotnorm2               = NULL;
    V->ops->placearray             = VecPlaceArray_Seq;
    V->ops->replacearray           = VecReplaceArray_Seq;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->duplicate              = VecDuplicate_Seq;
    V->ops->conjugate              = VecConjugate_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
  } else {
    V->ops->dot                    = VecDot_SeqHIP;
    V->ops->norm                   = VecNorm_SeqHIP;
    V->ops->tdot                   = VecTDot_SeqHIP;
    V->ops->scale                  = VecScale_SeqHIP;
    V->ops->copy                   = VecCopy_SeqHIP;
    V->ops->set                    = VecSet_SeqHIP;
    V->ops->swap                   = VecSwap_SeqHIP;
    V->ops->axpy                   = VecAXPY_SeqHIP;
    V->ops->axpby                  = VecAXPBY_SeqHIP;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqHIP;
    V->ops->pointwisemult          = VecPointwiseMult_SeqHIP;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqHIP;
    V->ops->setrandom              = VecSetRandom_SeqHIP;
    V->ops->dot_local              = VecDot_SeqHIP;
    V->ops->tdot_local             = VecTDot_SeqHIP;
    V->ops->norm_local             = VecNorm_SeqHIP;
    V->ops->mdot_local             = VecMDot_SeqHIP;
    V->ops->maxpy                  = VecMAXPY_SeqHIP;
    V->ops->mdot                   = VecMDot_SeqHIP;
    V->ops->aypx                   = VecAYPX_SeqHIP;
    V->ops->waxpy                  = VecWAXPY_SeqHIP;
    V->ops->dotnorm2               = VecDotNorm2_SeqHIP;
    V->ops->placearray             = VecPlaceArray_SeqHIP;
    V->ops->replacearray           = VecReplaceArray_SeqHIP;
    V->ops->resetarray             = VecResetArray_SeqHIP;
    V->ops->destroy                = VecDestroy_SeqHIP;
    V->ops->duplicate              = VecDuplicate_SeqHIP;
    V->ops->conjugate              = VecConjugate_SeqHIP;
    V->ops->getlocalvector         = VecGetLocalVector_SeqHIP;
    V->ops->restorelocalvector     = VecRestoreLocalVector_SeqHIP;
    V->ops->getlocalvectorread     = VecGetLocalVector_SeqHIP;
    V->ops->restorelocalvectorread = VecRestoreLocalVector_SeqHIP;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqHIP;
  }
  PetscFunctionReturn(0);
}

extern "C"
PetscErrorCode VecCreate_SeqHIP_Private(Vec V,const PetscScalar *array)
{
  PetscErrorCode ierr;
  Vec_HIP       *vechip;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQHIP on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQHIP);CHKERRQ(ierr);
  ierr = VecBindToCPU_SeqHIP(V,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->bindtocpu = VecBindToCPU_SeqHIP;

  /* Later, functions check for the Vec_HIP structure existence, so do not create it without array */
  if (array) {
    if (!V->spptr) {
      ierr = PetscMalloc(sizeof(Vec_HIP),&V->spptr);CHKERRQ(ierr);
      vechip = (Vec_HIP*)V->spptr;
      vechip->stream = 0; /* using default stream */
      vechip->GPUarray_allocated = 0;
      V->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
    }
    vechip = (Vec_HIP*)V->spptr;
    vechip->GPUarray = (PetscScalar*)array;
  }

  PetscFunctionReturn(0);
}
