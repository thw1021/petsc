#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcupm/vecseqcupm.hpp>

/*MC
   VECCUDA - VECCUDA = "cuda" - A VECSEQCUDA on a single-process communicator, and VECMPICUDA otherwise.

   Options Database Keys:
. -vec_type cuda - sets the vector type to VECCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECSEQCUDA, VECMPICUDA, VECSTANDARD, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

namespace Petsc
{

namespace Vector
{

namespace CUPM
{

namespace Impl
{

template <Device::CUPM::DeviceType T>
struct VecMPI_CUPM : Vec_CUPMBase<T,VecMPI_CUPM<T>>
{
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type,T,VecMPI_CUPM<T>);

protected:
  PETSC_CXX_COMPAT_DECL(constexpr Vec_MPI* VecIMPLCast_(Vec v))
  {
    return static_cast<Vec_MPI*>(v->data);
  }

  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 VecType VECTYPE_()) { return VECMPICUPM(); }

private:
  template <typename SeqFunction>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(Vec,PetscInt*,PetscReal*,SeqFunction,MPI_Op,MPI_Op));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupm_async_(Vec,PetscBool/*allocate_missing*/=PETSC_TRUE,PetscInt/*nghost*/=0,PetscScalar*/*host_array*/=nullptr,PetscScalar*/*device_array*/=nullptr));

public:
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupm_async(MPI_Comm,PetscInt,PetscInt,PetscInt,Vec*,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupmwitharrays_async(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec,NormType,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec,PetscInt,const Vec[],PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec,Vec,PetscScalar*,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscInt*,PetscReal*));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*));
};

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async_(Vec v, PetscBool allocate_missing, PetscInt nghost, PetscScalar *host_array, PetscScalar *device_array))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(v,PETSC_FALSE,nghost,host_array);CHKERRQ(ierr);
  ierr = Initialize_CUPMBase_(v,allocate_missing,host_array,device_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                             constructors/destructors                               //

// v->ops->create
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::create_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = creatempicupm_async_(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 VecCreateMPICUDA - Creates a standard, parallel array-style vector for CUDA devices.

 Collective

 Input Parameters:
 +  comm - the MPI communicator to use
 .  n - local vector length (or PETSC_DECIDE to have calculated if N is given)
 -  N - global vector length (or PETSC_DETERMINE to have calculated if n is given)

    Output Parameter:
 .  v - the vector

    Notes:
    Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
    same type as an existing vector.

    Level: intermediate

 .seealso: VecCreateMPICUDAWithArray(), VecCreateMPICUDAWithArrays(), VecCreateSeqCUDA(), VecCreateSeq(),
           VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
           VecCreateMPIWithArray(), VecCreateGhostWithArray(), VecMPISetGhost()

 @*/

// VecCreateMPICUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, Vec *v, PetscBool call_set_type))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Create_CUPMBase_(comm,bs,n,N,v,call_set_type);CHKERRQ(ierr);
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

.seealso: VecCreateMPICUDA(), VecCreateSeqCUDAWithArray(), VecCreateMPIWithArray(), VecCreateSeqWithArray(),
          VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost(),
          VecCreateMPI(), VecCreateGhostWithArray(), VecPlaceArray()

@*/

// VecCreateMPICUPMWithArray[s]()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupmwitharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // don't call VecSetType()
  ierr = creatempicupm_async(comm,bs,n,N,v,PETSC_FALSE);CHKERRQ(ierr);
  ierr = creatempicupm_async_(*v,PETSC_FALSE,0,PetscRemoveConstCast(host_array),PetscRemoveConstCast(device_array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::duplicate_async(Vec v, Vec *y))
{
  const auto     vimpl = VecIMPLCast(v);
  const auto     nghost = vimpl->nghost;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // don't call VecSetType(), we don't want to set up the data structures yet
  ierr = Duplicate_CUPMBase_(v,y,PETSC_FALSE);CHKERRQ(ierr);
  // now we do
  ierr = creatempicupm_async_(*y,PETSC_FALSE,nghost);CHKERRQ(ierr);
  // in case the user has done some VecSetOps() tomfoolery
  ierr = PetscMemcpy((*y)->ops,v->ops,sizeof(*v->ops));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (const auto locrep = vimpl->localrep) {
    const auto  ops   = locrep->ops;
    const auto  yimpl = VecIMPLCast(*y);
    PetscScalar *array;

    ierr = VecGetArray(*y,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,v->map->n+nghost,array,&yimpl->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(yimpl->localrep->ops,ops,sizeof(*ops));CHKERRQ(ierr);
    ierr = VecRestoreArray(*y,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(PetscObjectCast(*y),PetscObjectCast(yimpl->localrep));CHKERRQ(ierr);
    yimpl->localupdate = vimpl->localupdate;
    if (yimpl->localupdate) {
      ierr = PetscObjectReference(PetscObjectCast(yimpl->localupdate));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::destroy_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Destroy_CUPMBase_(v);CHKERRQ(ierr);
  {
    const auto useit = UseCUPMHostAlloc(v->pinned_memory);
    ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

// v->ops->norm
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::norm_async(Vec v, NormType type, PetscReal *z))
{
  PetscReal      work[2]  = {0};
  const auto     bothNorm = type == NORM_1_AND_2;
  const auto     norm2    = bothNorm || type == NORM_2 || type == NORM_FROBENIUS;
  const auto     count    = bothNorm ? 2 : 1;
  const auto     op       = type == NORM_INFINITY ? MPIU_MAX : MPIU_SUM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_CUPM<T>::norm_async(v,type,work);CHKERRQ(ierr);
  if (norm2) work[bothNorm] *= work[bothNorm];
  ierr = MPIU_Allreduce(work,z,count,MPIU_REAL,op,PetscObjectComm(PetscObjectCast(v)));CHKERRMPI(ierr);
  if (norm2) z[bothNorm] = PetscSqrtReal(z[bothNorm]);
  PetscFunctionReturn(0);
}

// v->ops->dot
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dot_async(Vec x, Vec y, PetscScalar *z))
{
  PetscScalar    work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_CUPM<T>::dot_async(x,y,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

// v->ops->tdot
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::tdot_async(Vec x, Vec y, PetscScalar *z))
{
  PetscScalar    work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_CUPM<T>::tdot_async(x,y,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::mdot_async(Vec x, PetscInt nv, const Vec y[], PetscScalar *z))
{
  auto           stackwork = std::array<PetscScalar,128>{};
  const auto     allocate  = nv > stackwork.size();
  auto           *work     = stackwork.data();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (allocate) {ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);}
  ierr = VecSeq_CUPM<T>::mdot_async(x,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  if (allocate) {ierr = PetscFree(work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dotnorm2_async(Vec x, Vec y, PetscScalar *dp, PetscScalar *nm))
{
  PetscScalar    work[2],sum[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_CUPM<T>::dotnorm2_async(x,y,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename SeqFunction>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::minmax_async_(Vec x, PetscInt *idx, PetscReal *z, SeqFunction seqfn, MPI_Op idxOp, MPI_Op noIdxOp))
{
  PetscReal      work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = seqfn(x,idx,&work);CHKERRQ(ierr);
  if (PetscDefined(HAVE_MPIUNI)) {
    *z = work;
  } else {
    const auto comm = PetscObjectComm(PetscObjectCast(x));

    if (idx) {
      struct { PetscReal v; PetscInt i; } in,out;

      in.v  = work;
      in.i  = *idx + x->map->rstart;
      ierr  = MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,idxOp,comm);CHKERRMPI(ierr);
      *z    = out.v;
      *idx  = out.i;
    } else {ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,noIdxOp,comm);CHKERRMPI(ierr);}
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::max_async(Vec x, PetscInt *idx, PetscReal *z))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(x,idx,z,VecSeq_CUPM<T>::max_async,MPIU_MAXLOC,MPIU_MAX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::min_async(Vec x, PetscInt *idx, PetscReal *z))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(x,idx,z,VecSeq_CUPM<T>::min_async,MPIU_MINLOC,MPIU_MIN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace CUPM

} // namespace Vector

} // namespace Petsc

/*MC
   VECMPICUDA - VECMPICUDA = "mpicuda" - The basic parallel vector, modified to use CUDA

   Options Database Keys:
. -vec_type mpicuda - sets the vector type to VECMPICUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecCreate_CUDA(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm(Petsc::PetscObjectCast(v)),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQCUDA);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPICUDA);CHKERRQ(ierr);
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
    V->ops->max                    = VecMax_MPI;
    V->ops->min                    = VecMin_MPI;
    V->ops->reciprocal             = VecReciprocal_Default;
    V->ops->sum                    = NULL;
    V->ops->shift                  = NULL;
    /* default random number generator */
    ierr = PetscFree(V->defaultrandtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCRANDER48,&V->defaultrandtype);CHKERRQ(ierr);
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
    V->ops->getlocalvectorread     = VecGetLocalVectorRead_SeqCUDA;
    V->ops->restorelocalvectorread = VecRestoreLocalVectorRead_SeqCUDA;
    V->ops->getarraywrite          = VecGetArrayWrite_SeqCUDA;
    V->ops->getarray               = VecGetArray_SeqCUDA;
    V->ops->restorearray           = VecRestoreArray_SeqCUDA;
    V->ops->getarrayandmemtype     = VecGetArrayAndMemType_SeqCUDA;
    V->ops->restorearrayandmemtype = VecRestoreArrayAndMemType_SeqCUDA;
    V->ops->max                    = VecMax_MPICUDA;
    V->ops->min                    = VecMin_MPICUDA;
    V->ops->reciprocal             = VecReciprocal_SeqCUDA;
    V->ops->sum                    = VecSum_SeqCUDA;
    V->ops->shift                  = VecShift_SeqCUDA;
    /* default random number generator */
    ierr = PetscFree(V->defaultrandtype);CHKERRQ(ierr);
    ierr = PetscStrallocpy(PETSCCURAND,&V->defaultrandtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
