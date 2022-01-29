#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/veccupmbase.hpp>   /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq
#include <petsc/private/randomimpl.h>      // for _p_PetscRandom

#if 0
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#endif

// TODO
// - refactor the AXPY's for code reuse
// - figure out how to template which thrust namespace to use so we can do
//   thrust::<backend>::par.on(stream)
// - maybe reintroduce PetscDeviceMalloc()?
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync
// - touch up the docs for both implementations
// - pick one of the VecGetArray<modifier>() to explain data movement semantics in the docs and
//   have everyone else refer to it
// - remove the cuda and hip separate versions
// - remove bindtocpu?
// - do rocblas instead of hipblas
// - remove this define and use the right header (i.e. clean up the headers first)
#define PetscNvshmemFree(...) 0

namespace Petsc
{

namespace Vector
{

namespace CUPM
{

namespace Impl
{

namespace
{

template <bool> struct UseComplexTag { };

} // anonymous namespace

template <Device::CUPM::DeviceType T>
struct VecSeq_CUPM : Vec_CUPMBase<T,VecSeq_CUPM<T>>
{
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type,T,VecSeq_CUPM<T>);

protected:
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast_(Vec v)) PETSC_DECLTYPE_RETURNS(static_cast<Vec_Seq*>(v->data))
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 auto VECTYPE_()) PETSC_DECLTYPE_RETURNS(VECSEQCUPM())

private:
  // common core for min and max
  template <typename TupleFuncT, typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(TupleFuncT&&,UnaryFuncT&&,PetscReal,Vec,PetscInt*,PetscReal*));
  // common core for pointwise binary and pointwise unary thrust functions
  template <typename BinaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisebinary_async_(BinaryFuncT&&,Vec,Vec,Vec));
  template <typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwiseunary_async_(UnaryFuncT&&,Vec,Vec/*out*/=nullptr));
  // mdot dispatchers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<true>,Vec,PetscInt,const Vec[],PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<false>,Vec,PetscInt,const Vec[],PetscScalar*));
  // dispatcher for the actual kernels for mdot when NOT configured for complex, called by
  // mdot_async_(use_complex_tag<false>,...)
  template <int>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext,cupmStream_t,const PetscScalar*,const Vec[],PetscInt,PetscScalar*,PetscInt*));
  // common core for the various create routines
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async_(Vec,PetscScalar*/*host_ptr*/=nullptr,PetscScalar*/*device_ptr*/=nullptr));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm,PetscInt,PetscInt,Vec*,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupmwithbotharrays_async(MPI_Comm,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Vec*));

  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode replacearray_async(Vec,const PetscScalar*));

  // callable indirectly via function pointers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode aypx_async(Vec,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpy_async(Vec,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisedivide_async(Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisemult_async(Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reciprocal_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waxpy_async(Vec,PetscScalar,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_async(Vec,PetscInt,const PetscScalar*,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec,PetscInt,const Vec[],PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode scale_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode swap_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpby_async(Vec,PetscScalar,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpbypcz_async(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec,NormType,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec,Vec,PetscScalar*,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode conjugate_async(Vec));
  template <bool>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec,Vec));
  template <bool>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode sum_async(Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode shift_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setrandom_async(Vec,PetscRandom));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec,PetscBool));
};

// ================================================================================== //
//                                                                                    //
//                                  utility methods                                   //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                                  array accessors                                   //

#define CHKERRTHRUST(...)  do {                                                 \
    try {                                                                       \
      __VA_ARGS__;                                                              \
    } catch (const thrust::system_error& ex) {                                  \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());     \
    }                                                                           \
  } while (0)

template <Device::CUPM::DeviceType T>
template <typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisebinary_async_(BinaryFuncT&& unary, Vec win, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  CHKERRTHRUST(
    auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,xin).ptr);
    auto yptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,yin).ptr);
    auto wptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,win).ptr);

    thrust::transform(xptr,xptr+n,yptr,wptr,std::forward<BinaryFuncT>(unary));
  );
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwiseunary_async_(UnaryFuncT&& unary, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  CHKERRTHRUST(
    if (xin == yin) { // in-place
      auto xptr = thrust::device_pointer_cast(DeviceArrayReadWrite(dctx,xin).ptr);

      thrust::transform(xptr,xptr+n,xptr,std::forward<UnaryFuncT>(unary));
    } else {
      auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,xin).ptr);
      auto yptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,yin).ptr);

      thrust::transform(xptr,xptr+n,yptr,std::forward<UnaryFuncT>(unary));
    }
  );
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async_(Vec v, PetscScalar *host_array, PetscScalar *device_array))
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)),&size);CHKERRMPI(ierr);
  if (PetscUnlikely(size > 1)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must create VecSeq on communicator of size 1, have size %d",size);
  // REVIEW ME: remove me
  if (PetscUnlikely(VecIMPLCast(v))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Creating VecSeq for the second time!");
  ierr = VecCreate_Seq_Private(v,host_array);CHKERRQ(ierr);
  ierr = Initialize_CUPMBase_(v,PETSC_FALSE,host_array,device_array);CHKERRQ(ierr);
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = createseqcupm_async_(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, Vec *v, PetscBool call_set_type))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Create_CUPMBase_(comm,bs,n,n,v,call_set_type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPMWithArrays()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupmwithbotharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // createseqcupm_async_() is called!
  ierr = createseqcupm_async(comm,bs,n,v,PETSC_FALSE);CHKERRQ(ierr);
  ierr = createseqcupm_async_(*v,PetscRemoveConstCast(host_array),PetscRemoveConstCast(device_array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::duplicate_async(Vec v, Vec *y))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Duplicate_CUPMBase_(v,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Destroy_CUPMBase_(v);CHKERRQ(ierr);
  {
    const auto useit = UseCUPMHostAlloc(v->pinned_memory);
    ierr = VecDestroy_Seq(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define VecSetOp_CUPM(op_name,op_host,...)  v->ops->op_name = usehost ? op_host : __VA_ARGS__

// VecCUPMBindToCPU()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->boundtocpu == usehost) PetscFunctionReturn(0);
  ierr = BindToCPU_CUPMBase_(v,usehost);CHKERRQ(ierr);

  // REVIEW ME: this absolutely should be some sort of bulk mempcy rather than this mess
  v->ops->dot   = VecSetOp_CUPM(dot_local,VecDot_Seq,dot_async);
  v->ops->norm  = VecSetOp_CUPM(norm_local,VecNorm_Seq,norm_async);
  v->ops->tdot  = VecSetOp_CUPM(tdot_local,VecTDot_Seq,tdot_async);
  v->ops->mdot  = VecSetOp_CUPM(mdot_local,VecMDot_Seq,mdot_async);
  v->ops->mtdot = VecSetOp_CUPM(mtdot_local,VecMTDot_Seq,nullptr);
  VecSetOp_CUPM(scale,VecScale_Seq,scale_async);
  VecSetOp_CUPM(copy,VecCopy_Seq,copy_async);
  VecSetOp_CUPM(set,VecSet_Seq,set_async);
  VecSetOp_CUPM(swap,VecSwap_Seq,swap_async);
  VecSetOp_CUPM(axpy,VecAXPY_Seq,axpy_async);
  VecSetOp_CUPM(axpby,VecAXPBY_Seq,axpby_async);
  VecSetOp_CUPM(axpbypcz,VecAXPBYPCZ_Seq,axpbypcz_async);
  VecSetOp_CUPM(pointwisemult,VecPointwiseMult_Seq,pointwisemult_async);
  VecSetOp_CUPM(pointwisedivide,VecPointwiseDivide_Seq,pointwisedivide_async);
  VecSetOp_CUPM(setrandom,VecSetRandom_Seq,setrandom_async);
  VecSetOp_CUPM(maxpy,VecMAXPY_Seq,maxpy_async);
  VecSetOp_CUPM(aypx,VecAYPX_Seq,aypx_async);
  VecSetOp_CUPM(waxpy,VecWAXPY_Seq,waxpy_async);
  VecSetOp_CUPM(dotnorm2,nullptr,dotnorm2_async);
  VecSetOp_CUPM(conjugate,VecConjugate_Seq,conjugate_async);
  VecSetOp_CUPM(max,VecMax_Seq,max_async);
  VecSetOp_CUPM(min,VecMin_Seq,min_async);
  VecSetOp_CUPM(reciprocal,VecReciprocal_Default,reciprocal_async);
  VecSetOp_CUPM(sum,nullptr,sum_async);
  VecSetOp_CUPM(shift,nullptr,shift_async);
  VecSetOp_CUPM(placearray,VecPlaceArray_Seq,placearray_async<PETSC_MEMTYPE_HOST>);
  v->ops->replacearray = replacearray_async<PETSC_MEMTYPE_HOST>;
  VecSetOp_CUPM(resetarray,VecResetArray_Seq,resetarray_async<PETSC_MEMTYPE_HOST>);

  VecSetOp_CUPM(duplicate,VecDuplicate_Seq,duplicate_async);
  VecSetOp_CUPM(getlocalvector,nullptr,&getlocalvector_async</*read = */false>);
  VecSetOp_CUPM(getlocalvectorread,nullptr,&getlocalvector_async</*read = */true>);
  VecSetOp_CUPM(restorelocalvector,nullptr,&restorelocalvector_async</*read = */false>);
  VecSetOp_CUPM(restorelocalvectorread,nullptr,&restorelocalvector_async</*read = */true>);
  PetscFunctionReturn(0);
}

#undef VecSetOp_CUPM

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->resetarray or VecCUPMResetArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
// yes (probably Jed :)), ideal world these should be arguments not template parameters. But I
// need to assign this function to a C compatible function pointer, so something like default
// arguments don't work no? Stubs seem like overkill too...
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  // REVIEW ME:
  // this is wildly inefficient but must be done if we assume that the placed array must have
  // correct values
  if (PetscMemTypeHost(mtype)) {
    ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    ierr = VecResetArray_Seq(v);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    const auto vseq = VecIMPLCast(v);

    ierr = CopyToDevice_(dctx,v);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    VecCUPMCast(v)->device_array = vseq->unplacedarray;
    vseq->unplacedarray          = nullptr;
    v->offloadmask               = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if (PetscMemTypeHost(mtype)) {
    ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    ierr = VecPlaceArray_Seq(v,a);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    const auto vseq = VecIMPLCast(v);

    if (PetscUnlikely(vseq->unplacedarray)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
    ierr = getarray_async<PETSC_MEMTYPE_DEVICE,MemoryAccess::READ_WRITE>(v,&vseq->unplacedarray);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    VecCUPMCast(v)->device_array = const_cast<PetscScalar*>(a);
    // offload mask set by getarray
  }
  PetscFunctionReturn(0);
}

// v->ops->replacearray or VecCUPMReplaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::replacearray_async(Vec v, const PetscScalar *a))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED(mtype);
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (PetscMemTypeHost(mtype)) {
    const auto vseq = VecIMPLCast(v);

    if (vseq->array != vseq->array_allocated) {
      PetscDeviceContext dctx;
      // make sure the users array has the latest values.
      // REVIEW ME: why? we're about to free it
      ierr = GetHandles_(&dctx);CHKERRQ(ierr);
      ierr = CopyToHost_(dctx,v);CHKERRQ(ierr);
    }
    if (vseq->array_allocated) {
      const auto useit = UseCUPMHostAlloc(v->pinned_memory);
      ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
    }
    vseq->array_allocated = vseq->array = PetscRemoveConstCast(a);
    v->pinned_memory      = PETSC_FALSE; // REVIEW ME: we can determine this
    v->offloadmask        = PETSC_OFFLOAD_CPU;
  } else {
    const auto vcu = VecCUPMCast(v);

    switch (vcu->ptr_ownership) {
    case PETSC_COPY_VALUES:
    case PETSC_OWN_POINTER:
      if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
        ierr = PetscNvshmemFree(vcu->device_array);CHKERRQ(ierr);
      } else {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = GetHandles_(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
    case PETSC_USE_POINTER:
      vcu->device_array = PetscRemoveConstCast(a);
      break;
    }
    ierr = PetscObjectStateIncrease(PetscObjectCast(v));CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscFunctionReturn(0);
}

#undef STATIC_ASSERT_THAT_ONLY_PETSC_MEMTYPE_HOST_OR_DEVICE_IS_USED

// v->ops->getlocalvector or v->ops->getlocalvectorread
template <Device::CUPM::DeviceType T>
template <bool read>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w))
{
  PetscErrorCode ierr;
  PetscBool      wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm);CHKERRQ(ierr);
  if (wisseqcupm) {
    if (const auto vseq = VecIMPLCast(w)) {
      if (vseq->array_allocated) {
        const auto useit = UseCUPMHostAlloc(w->pinned_memory);

        ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
        if (useit.value()) w->pinned_memory = PETSC_FALSE;
      }
      vseq->array         = nullptr;
      vseq->unplacedarray = nullptr;
    }
    if (const auto vcu = VecCUPMCast(w)) {
      if (vcu->device_array) {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = GetHandles_(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
      ierr = PetscFree(w->spptr /* vcu */);CHKERRQ(ierr);
    }
  }
  if (v->petscnative && wisseqcupm) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data          = v->data;
    w->offloadmask   = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr         = v->spptr;
    ierr = PetscObjectStateIncrease(PetscObjectCast(w));CHKERRQ(ierr);
  } else {
    const auto arrayptr = &VecIMPLCast(w)->array;
    if (read) {
      ierr = VecGetArrayRead(v,const_cast<const PetscScalar**>(arrayptr));CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(v,arrayptr);CHKERRQ(ierr);
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcupm) {
      PetscDeviceContext dctx;

      ierr = GetHandles_(&dctx);CHKERRQ(ierr);
      ierr = DeviceAllocateCheck_(dctx,w);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->restorelocalvector or v->ops->restorelocalvectorread
template <Device::CUPM::DeviceType T>
template <bool read>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorelocalvector_async(Vec v, Vec w))
{
  PetscErrorCode ierr;
  PetscBool      wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm);CHKERRQ(ierr);
  if (v->petscnative && wisseqcupm) {
    v->data          = w->data;
    v->offloadmask   = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr         = w->spptr;
    w->data          = nullptr;
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
    w->spptr         = nullptr;
  } else {
    auto array = &VecIMPLCast(w)->array;
    if (read) {
      ierr = VecRestoreArrayRead(v,const_cast<const PetscScalar**>(array));CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(v,array);CHKERRQ(ierr);
    }
    if (w->spptr && wisseqcupm) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = GetHandles_(&stream);CHKERRQ(ierr);
      cerr = cupmFreeAsync(VecCUPMCast(w)->device_array,stream);CHKERRCUPM(cerr);
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin))
{
  const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    const auto   nbytes = n*sizeof(typename DeviceArrayRead::value_type);
    cupmError_t  cerr;
    cupmStream_t stream;

    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).ptr,DeviceArrayRead(dctx,xin).ptr,nbytes,cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  } else {
    const auto       alphaIsOne = alpha == PetscScalar(1.0);
    cupmBlasHandle_t cupmBlasHandle;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      cupmBlasError_t cberr;
      auto            yarray = DeviceArrayWrite(dctx,yin);
      auto            xarray = DeviceArrayRead(dctx,xin);

      if (alphaIsOne) {
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        constexpr auto sone = PetscScalar(1.0);

        cberr = cupmBlasXscal(cupmBlasHandle,n,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&sone,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((alphaIsOne ? 1 : 2)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin))
{
  PetscErrorCode ierr;
  PetscBool      xiscupm;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompareAny(PetscObjectCast(xin),&xiscupm,VECSEQCUPM(),VECMPICUPM(),"");CHKERRQ(ierr);
  if (xiscupm) {
    const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,DeviceArrayRead(dctx,xin),1,DeviceArrayWrite(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    ierr = pointwisebinary_async_(thrust::divides<PetscScalar>(),win,xin,yin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseMult_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    ierr = pointwisebinary_async_(thrust::multiplies<PetscScalar>(),win,xin,yin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

namespace detail
{

struct reciprocal
{
  PETSC_HOSTDEVICE_DECL constexpr
  auto operator()(PetscScalar s) const PETSC_DECLTYPE_RETURNS(s ? PetscScalar(1.0)/s : s);
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pointwiseunary_async_(detail::reciprocal(),xin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = copy_async(yin,win);CHKERRQ(ierr);
  } else {
    const auto         n = win->map->n;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmStream_t       stream;

    ierr = GetHandles_(&dctx,&cupmBlasHandle,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto warray = DeviceArrayWrite(dctx,win);
      auto cerr   = cupmMemcpyAsync(warray.ptr,DeviceArrayRead(dctx,yin).ptr,n*sizeof(typename decltype(warray)::value_type),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
      auto cberr  = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,DeviceArrayRead(dctx,xin),1,warray,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscInt nv, const PetscScalar *alpha, Vec *y))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto xarray = DeviceArrayWrite(dctx,xin);

    for (decltype(nv) j = 0; j < nv; ++j) {
      auto cberr = cupmBlasXaxpy(cupmBlasHandle,n,alpha+j,DeviceArrayRead(dctx,y[j]),1,xarray,1);CHKERRCUPMBLAS(cberr);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(nv*2*n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(nv*sizeof(*alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
  // second
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdot(cupmBlasHandle,n,DeviceArrayRead(dctx,yin),1,DeviceArrayRead(dctx,xin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*(n-1),0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define MDOT_WORKGROUP_NUM  128
#define MDOT_WORKGROUP_SIZE MDOT_WORKGROUP_NUM

namespace kernels
{

PETSC_HOSTDEVICE_DECL static PetscInt EntriesPerGroup(PetscInt size)
{
  const auto group_entries = (size-1)/(MDOT_WORKGROUP_SIZE+1);
  // for very small vectors, a group should still do some work
  return group_entries ? group_entries : 1;
}

template <int N>
PETSC_KERNEL_DECL static void mdot_kernel(const PetscScalar *PETSC_RESTRICT x, const PetscScalar *PETSC_RESTRICT y[PETSC_RESTRICT N], PetscInt size, PetscScalar *PETSC_RESTRICT results)
{
  static_assert(N > 0,"");
  PETSC_SHAREDMEM_DECL PetscScalar shmem[N*MDOT_WORKGROUP_SIZE];
  const auto tx       = threadIdx.x,bx = blockIdx.x;
  const auto bdx      = blockDim.x,gdx = gridDim.x;
  const auto worksize = EntriesPerGroup(size);
  const auto begin    = tx+bx*worksize;
  const auto end      = min((bx+1)*worksize,size);
  PetscScalar sumlocal[N];
  PetscScalar *ylocal[N];

#pragma unroll
  for (auto i = 0; i < N; ++i) {
    sumlocal[i] = 0;
    ylocal[i]   = y[i]; // load pointer once
  }

#pragma unroll
  for (auto i = begin; i < end; i += bdx) {
    const auto xi = x[i]; // load only once from global memory!

#pragma unroll
    for (auto j = 0; j < N; ++j) sumlocal[j] += ylocal[j][i]*xi;
  }

#pragma unroll
  for (auto i = 0; i < N; ++i) shmem[tx+i*MDOT_WORKGROUP_SIZE] = sumlocal[i];

  // parallel reduction
#pragma unroll
  for (auto stride = bdx/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
#pragma unroll
      for (auto i = tx; i < N; i += MDOT_WORKGROUP_SIZE) shmem[i] += shmem[i+stride];
    }
  }
  // bottom N threads per block write to global memory
  // REVIEW ME: I am ~pretty~ sure we don't need another __syncthreads() here since each thread
  // writes to the same sections in the above loop that it is about to read from below
  if (tx < N) results[bx+tx*gdx] = shmem[tx*MDOT_WORKGROUP_SIZE];
  return;
}

} // namespace kernels

template <Device::CUPM::DeviceType T>
template <int N>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar *results, PetscInt *yidx))
{
  static_assert(N > 0,"");
  const auto   yidxt = *yidx;
  const auto   yint  = yin+yidxt;
  PetscScalar *device_y[N];
  cupmError_t  cerr;

  PetscFunctionBegin;
  for (auto i = 0; i < N; ++i) device_y[i] = DeviceArrayRead(dctx,yint[i]);
  cerr = cupmLaunchKernel(kernels::mdot_kernel<N>,dim3(MDOT_WORKGROUP_NUM),dim3(MDOT_WORKGROUP_SIZE),0,stream,xarr,device_y,size,results+(yidxt*MDOT_WORKGROUP_NUM));CHKERRCUPM(cerr);
  *yidx += N;
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<false>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto          n      = xin->map->n;
  const auto          nv1    = ((nv % 4) == 1) ? nv-1 : nv;
  const auto          nbytes = nv1*MDOT_WORKGROUP_NUM*sizeof(*VecIMPLCast(xin)->array);
  PetscScalar         *d_results;
  PetscDeviceContext  dctx;
  cupmStream_t        stream;
  cupmError_t         cerr;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  // allocate scratchpad memory for the results of individual work groups
  cerr = cupmMallocAsync(reinterpret_cast<void**>(&d_results),nbytes,stream);CHKERRCUPM(cerr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto yidx = 0;
    auto xptr = DeviceArrayRead(dctx,xin);

    // REVIEW ME: Can fork-join here
    while (yidx < nv)
    {
      switch (nv-yidx) {
      case 7:
      case 6:
      case 5:
      case 4:
        ierr = mdot_kernel_dispatch_<4>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 3:
        ierr = mdot_kernel_dispatch_<3>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 2:
        ierr = mdot_kernel_dispatch_<2>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 1:
        ierr = mdot_kernel_dispatch_<1>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
      case 0:
        break;
      default: // 8 or more
        ierr = mdot_kernel_dispatch_<8>(dctx,stream,xptr,yin,n,d_results,&yidx);CHKERRQ(ierr);
        break;
      }
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // copy results to CPU
  {
    std::array<PetscScalar,PETSC_MAX_PATH_LEN> stackarray;
    const auto allocate   = nv1*MDOT_WORKGROUP_NUM > stackarray.size();
    auto       h_results  = stackarray.data();

    if (allocate) {ierr = PetscMalloc1(nv1*MDOT_WORKGROUP_NUM,&h_results);CHKERRQ(ierr);}
    cerr = cupmMemcpyAsync(h_results,d_results,nbytes,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
    // REVIEW ME: double count of flops??
    // do these now while memcpy is in flight
    ierr = PetscLogFlops(nv1*MDOT_WORKGROUP_NUM);CHKERRQ(ierr);
    ierr = PetscLogGpuToCpuScalar(nbytes);CHKERRQ(ierr);
    // for systems without async free this will synchronize implicitly
    cerr = cupmFreeAsync(d_results,stream);CHKERRCUPM(cerr);
    // REVIEW ME: need to hard sync here...
    ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
    // REVIEW ME: it is likely faster to do this in a micro kernel rather than do it on the
    // host which that requires synchronization
    // sum group results into z
    for (auto j = decltype(nv1)(0); j < nv1; ++j) {
      for (auto i = j*MDOT_WORKGROUP_NUM; i < (j+1)*MDOT_WORKGROUP_NUM; ++i) z[j] += h_results[i];
    }
    if (allocate) {ierr = PetscFree(h_results);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_NUM
#undef MDOT_WORKGROUP_SIZE

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<true>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto xptr = DeviceArrayRead(dctx,xin);

    // can fork-join here
    for (decltype(nv) i = 0; i < nv; ++i) {
      auto cberr = cupmBlasXdot(cupmBlasHandle,n,DeviceArrayRead(dctx,yin+i),1,xptr,1,z+i);CHKERRCUPMBLAS(cberr);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: flops?????
  ierr = PetscLogGpuToCpuScalar(nv*sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  using complex_tag = UseComplexTag<PetscDefined(USE_COMPLEX)>;
  const auto     n = xin->map->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscUnlikely(nv <= 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to %s %" PetscInt_FMT " not positive",PETSC_FUNCTION_NAME,nv);
  else if (PetscUnlikely(nv == 1)) {
    ierr = dot_async(xin,PetscRemoveConstCast(yin[0]),z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  // z will always need to be zeroed first, either for a quick return or for summing later on
  ierr = PetscArrayzero(z,nv);CHKERRQ(ierr);
  // nothing to do if x has no entries
  if (!n) PetscFunctionReturn(0);
  ierr = mdot_async_(complex_tag(),xin,nv,yin,z);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha))
{
  const auto         n = xin->map->n;
  PetscErrorCode     ierr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0)) {
    const auto   nbytes = n*sizeof(typename DeviceArrayWrite::value_type);
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemsetAsync(DeviceArrayWrite(dctx,xin).ptr,0,nbytes,stream);CHKERRCUPM(cerr);
  } else {
    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,xin).ptr);

      thrust::fill(xptr,xptr+n,alpha);
    );
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscScalar alpha))
{
  PetscFunctionBegin;
  if (alpha == PetscScalar(1.0)) PetscFunctionReturn(0);
  else if (alpha == PetscScalar(0.0)) {
    auto ierr = set_async(xin,alpha);CHKERRQ(ierr);
  } else {
    const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscErrorCode     ierr;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXscal(cupmBlasHandle,n,&alpha,DeviceArrayWrite(dctx,xin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscScalar *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdotu(cupmBlasHandle,n,DeviceArrayRead(dctx,xin),1,DeviceArrayRead(dctx,yin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*n-1,0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin))
{y
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    const auto         nbytes = n*sizeof(*VecIMPLCast(xin)->array);
    auto               yiscupm = PETSC_TRUE,xondevice = PETSC_TRUE; // assume we start on device
    cupmMemcpyKind_t   mode;
    PetscDeviceContext dctx;
    cupmStream_t       stream;
    cupmError_t        cerr;
    PetscErrorCode     ierr;

    switch (xin->offloadmask) {
    case PETSC_OFFLOAD_KOKKOS:      // technically an error
    case PETSC_OFFLOAD_UNALLOCATED: // technically an error
    case PETSC_OFFLOAD_CPU:
      xondevice = PETSC_FALSE; // we assumed partially wrong
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      break;
      // no default case so warnings are thrown for new offloadmasks
    }

    switch (yin->offloadmask) {
    case PETSC_OFFLOAD_KOKKOS:
    case PETSC_OFFLOAD_UNALLOCATED:
    case PETSC_OFFLOAD_CPU:
      ierr = PetscObjectTypeCompareAny(PetscObjectCast(yin),&yiscupm,VECSEQCUPM(),VECMPICUPM(),"");CHKERRQ(ierr);
    case PETSC_OFFLOAD_GPU:
    case PETSC_OFFLOAD_BOTH:
      if (yiscupm) { // PETSC_TRUE by default (unless on the host)
        // even though y may be on the host, its a cupm vector, so it ought to be on the device
        mode = xondevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;
      } else {
        // we assumed really wrong
        mode = xondevice ? cupmMemcpyDeviceToHost : cupmMemcpyHostToHost;
      }
      break;
    }

    ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
    switch (mode) {
    case cupmMemcpyDeviceToDevice:
      // the best case
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).ptr,DeviceArrayRead(dctx,xin).ptr,nbytes,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyHostToDevice:
      // not terrible
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).ptr,HostArrayRead(dctx,xin).ptr,nbytes,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,DeviceArrayRead(dctx,xin).ptr,nbytes,mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    case cupmMemcpyHostToHost:   {
      // the worst case
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscArraycpy(yarray,HostArrayRead(dctx,xin),n);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
    }
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscErrorCode     ierr;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXswap(cupmBlasHandle,n,DeviceArrayWrite(dctx,xin),1,DeviceArrayWrite(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscScalar alpha, PetscScalar beta, Vec xin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = scale_async(yin,beta);CHKERRQ(ierr);
  } else if (beta == PetscScalar(1.0)) {
    ierr = axpy_async(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == PetscScalar(1.0)) {
    ierr = aypx_async(yin,beta,xin);CHKERRQ(ierr);
  } else {
    const auto         betaIsZero = beta == PetscScalar(0.0);
    const auto         n = static_cast<cupmBlasInt_t>(yin->map->n);
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto yarray = DeviceArrayWrite(dctx,yin);

      if (betaIsZero) {
        const auto   nbytes = n*sizeof(typename decltype(yarray)::value_type);
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
        cerr = cupmMemcpyAsync(yarray,DeviceArrayRead(dctx,xin).ptr,nbytes,cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
        cberr = cupmBlasXscal(cupmBlasHandle,n,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        cberr = cupmBlasXscal(cupmBlasHandle,n,&beta,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&alpha,DeviceArrayRead(dctx,xin),1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((betaIsZero ? 1 : 3)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar((betaIsZero ? 1 : 2)*sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (gamma != PetscScalar(1.0)) {
    // z <- a*x + b*y + c*z
    ierr = scale_async(zin,gamma);CHKERRQ(ierr);
  }
  ierr = axpy_async(zin,alpha,xin);CHKERRQ(ierr);
  ierr = axpy_async(zin,beta,yin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscReal *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscInt           flopCount = 0;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!n) {
    z[0] = 0.0;
    // yes this technically sets z[0] = 0 again half the time
    z[type == NORM_1_AND_2] = 0.0;
    PetscFunctionReturn(0);
  }
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    cupmBlasError_t cberr;
    auto            xarray = DeviceArrayRead(dctx,xin);

    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      cberr = cupmBlasXasum(cupmBlasHandle,n,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount = PetscMax(n-1,0);
      if (type == NORM_1) break;
      ++z; // fall-through
    case NORM_2:
    case NORM_FROBENIUS:
      cberr = cupmBlasXnrm2(cupmBlasHandle,n,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount += PetscMax(2*n-1,0); // +=  in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmError_t  cerr;
      cupmStream_t stream;
      PetscScalar  zs;
      int          i;

      // REVIEW ME: this needs to be redone by hand
      cberr = cupmBlasXamax(cupmBlasHandle,n,xarray,1,&i);CHKERRCUPMBLAS(cberr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(&zs,xarray.ptr+i-1,sizeof(zs),cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      *z   = PetscAbsScalar(zs);
      // REVIEW ME: flopCount = ???
    } break;
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flopCount);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = dot_async(s,t,dp);CHKERRQ(ierr);
  ierr = dot_async(t,t,nm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace detail
{

struct conjugate
{
  PETSC_HOSTDEVICE_DECL
  constexpr PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec xin))
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    auto ierr = pointwiseunary_async_(detail::conjugate(),xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

namespace detail
{

struct real_part
{
  PETSC_HOSTDEVICE_DECL
  thrust::tuple<PetscReal,PetscInt> operator()(const thrust::tuple<PetscScalar,PetscInt>& x) const
  {
    return {PetscRealPart(x.get<0>()),x.get<1>()};
  }

  PETSC_HOSTDEVICE_DECL
  constexpr PetscReal operator()(PetscScalar x) const { return PetscRealPart(x); }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
template <typename TupleFuncT, typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::minmax_async_(TupleFuncT&& tuple_ftr, UnaryFuncT&& unary_ftr, PetscReal initval, Vec v, PetscInt *p, PetscReal *m))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (!n) {
    *m = initval;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  // REVIEW ME: why not cupmBlasIXamin()/cupmBlasIXamax()?
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  CHKERRTHRUST(
    auto vptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).ptr);

    if (p) {
      const auto init = thrust::make_tuple(initval,PetscInt(-1));
      auto       zip  = thrust::make_zip_iterator(
        thrust::make_tuple(vptr,thrust::make_counting_iterator(PetscInt(0)))
      );

      if (PetscDefined(USE_COMPLEX)) {
        thrust::tie(*m,*p) = thrust::transform_reduce(
          zip,zip+n,detail::real_part(),init,std::forward<TupleFuncT>(tuple_ftr)
        );
      } else {
        thrust::tie(*m,*p) = thrust::reduce(zip,zip+n,init,std::forward<TupleFuncT>(tuple_ftr));
      }
    } else {
      if (PetscDefined(USE_COMPLEX)) {
        *m = thrust::transform_reduce(
          vptr,vptr+n,detail::real_part(),initval,std::forward<UnaryFuncT>(unary_ftr)
        );
      } else {
        *m = thrust::reduce(vptr,vptr+n,initval,std::forward<UnaryFuncT>(unary_ftr));
      }
    }
  );
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

namespace detail
{

struct max_tuple
{
  using tuple_type = thrust::tuple<PetscReal,PetscInt>;

  PETSC_HOSTDEVICE_DECL tuple_type operator()(const tuple_type& x, const tuple_type& y) const
  {
    if ((x.get<0>() > y.get<0>()) || (x.get<1>() <  y.get<1>())) {
      return thrust::make_tuple(x.get<0>(),x.get<1>());
    } else {
      return thrust::make_tuple(y.get<0>(),y.get<1>());
    }
  }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscInt *p, PetscReal *m))
{
  using tuple_ftr = detail::max_tuple;
  using unary_ftr = thrust::maximum<util::remove_pointer_t<decltype(m)>>;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(tuple_ftr(),unary_ftr(),PETSC_MIN_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace detail
{

struct min_tuple
{
  using tuple_type = thrust::tuple<PetscReal,PetscInt>;

  PETSC_HOSTDEVICE_DECL
  constexpr tuple_type operator()(const tuple_type& x, const tuple_type& y) const
  {
    return ((x.get<0>() < y.get<0>()) || (x.get<1>() < y.get<1>())) ? x : y;
    // if ((x.get<0>() < y.get<0>()) || (x.get<1>() < y.get<1>())) {
    //   return thrust::make_tuple(x.get<0>(),x.get<1>());
    // } else {
    //   return thrust::make_tuple(y.get<0>(),y.get<1>());
    // }
  }
};

} // namespace detail
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscInt *p, PetscReal *m))
{
  using tuple_functor = detail::min_tuple;
  using unary_functor = thrust::minimum<util::remove_pointer_t<decltype(m)>>;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(tuple_functor(),unary_functor(),PETSC_MAX_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscScalar *sum))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  // REVIEW ME: why not cupmBlasXasum()?
  CHKERRTHRUST(
    auto dptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).ptr);

    *sum = thrust::reduce(dptr,dptr+n,PetscScalar(0.0));
  );
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: must be at least n additions
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace detail
{

struct shifter
{
  const PetscScalar s;

  PETSC_HOSTDEVICE_DECL constexpr PetscScalar operator()(PetscScalar x) const { return x+s; }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscScalar shift))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pointwiseunary_async_(detail::shifter{shift},v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  PetscBool          iscurand;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare(PetscObjectCast(rand),PETSCCURAND,&iscurand);CHKERRQ(ierr);
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  if (iscurand) {
    ierr = PetscRandomGetValues(rand,n,DeviceArrayWrite(dctx,v));CHKERRQ(ierr);
  } else {
    ierr = PetscRandomGetValues(rand,n,HostArrayWrite(dctx,v));CHKERRQ(ierr);
  }
  // REVIEW ME: flops????
  // REVIEW ME: Timing???
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace CUPM

} // namespace Vec

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
