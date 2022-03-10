#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/veccupmimpl.h>
#include <petsc/private/randomimpl.h> // for _p_PetscRandom
#include "../src/sys/objects/device/impls/cupm/cupmthrustutility.hpp"

#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// TODO
// - refactor the AXPY's for code reuse
// - maybe reintroduce PetscDeviceMalloc()?
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync
// - touch up the docs for both implementations
// - pick one of the VecGetArray<modifier>() to explain data movement semantics in the docs and
//   have everyone else refer to it
// - remove the cuda and hip separate versions
// - do rocblas instead of hipblas

namespace Petsc
{

// REVIEW ME: using "Vec" as the namespace causes ambiguity errors when referring to "Vec" the
// type later on
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

private:
  PETSC_CXX_COMPAT_DECL(auto VecIMPLCast_(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_Seq*>(v->data));
  PETSC_CXX_COMPAT_DECL(constexpr auto VECTYPE_()) PETSC_DECLTYPE_AUTO_RETURNS(VECSEQCUPM());

  // common core for min and max
  template <typename TupleFuncT,typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(TupleFuncT&&,UnaryFuncT&&,PetscReal,Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext));
  // common core for pointwise binary and pointwise unary thrust functions
  template <typename BinaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisebinary_async_(BinaryFuncT&&,PetscDeviceContext,Vec,Vec,Vec));
  template <typename UnaryFuncT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwiseunary_async_(UnaryFuncT&&,PetscDeviceContext,Vec,Vec/*out*/=nullptr));
  // mdot dispatchers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<true>,Vec,PetscInt,const Vec[],PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async_(UseComplexTag<false>,Vec,PetscInt,const Vec[],PetscManagedScalar,PetscDeviceContext));
  // dispatcher for the actual kernels for mdot when NOT configured for complex, called by
  // mdot_async_(use_complex_tag<false>,...)
  template <int>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext,cupmStream_t,const PetscScalar*,const Vec[],PetscInt,PetscScalar**,PetscScalar*,PetscInt&));
  // common core for the various create routines
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async_(Vec,PetscDeviceContext,PetscScalar*/*host_ptr*/=nullptr,PetscScalar*/*device_ptr*/=nullptr));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm,PetscInt,PetscInt,PetscDeviceContext,Vec*,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupmwithbotharrays_async(MPI_Comm,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],PetscDeviceContext,Vec*));

  // callable indirectly via function pointers
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode aypx_async(Vec,PetscManagedScalar,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpy_async(Vec,PetscManagedScalar,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisedivide_async(Vec,Vec,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisemult_async(Vec,Vec,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reciprocal_async(Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waxpy_async(Vec,PetscManagedScalar,Vec,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_async(Vec,PetscManagedInt,PetscManagedScalar,Vec*,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode set_async(Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode scale_async(Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec,Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode copy_async(Vec,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode swap_async(Vec,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpby_async(Vec,PetscManagedScalar,PetscManagedScalar,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpbypcz_async(Vec,PetscManagedScalar,PetscManagedScalar,PetscManagedScalar,Vec,Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec,NormType,PetscManagedReal,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec,Vec,PetscManagedScalar,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode conjugate_async(Vec,PetscDeviceContext));
  template <PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec,Vec,PetscDeviceContext));
  template <PetscMemoryAccessMode>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec,Vec,PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec,PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode sum_async(Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode shift_async(Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setrandom_async(Vec,PetscRandom,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec,PetscBool,PetscDeviceContext));
};

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async_(Vec v, PetscDeviceContext dctx, PetscScalar *host_array, PetscScalar *device_array))
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)),&size));
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must create VecSeq on communicator of size 1, have size %d",size);
  // REVIEW ME: remove me
  PetscCheck(!VecIMPLCast(v),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Creating VecSeq for the second time!");
  PetscCall(VecCreate_Seq_Private(v,host_array,dctx));
  PetscCall(Initialize_CUPMBase(v,PETSC_FALSE,host_array,device_array,dctx));
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async(Vec v, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(createseqcupm_async_(v,dctx));
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscDeviceContext dctx, Vec *v, PetscBool call_set_type))
{
  PetscFunctionBegin;
  PetscCall(Create_CUPMBase(comm,bs,n,n,dctx,v,call_set_type));
  PetscFunctionReturn(0);
}

// VecCreateSeqCUPMWithArrays()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupmwithbotharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar host_array[], const PetscScalar device_array[], PetscDeviceContext dctx, Vec *v))
{
  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // createseqcupm_async_() is called!
  PetscCall(createseqcupm_async(comm,bs,n,dctx,v,PETSC_FALSE));
  PetscCall(createseqcupm_async_(*v,dctx,PetscRemoveConstCast(host_array),PetscRemoveConstCast(device_array)));
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::duplicate_async(Vec v, Vec *y, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(Duplicate_CUPMBase(v,y,dctx));
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(Destroy_CUPMBase(v,dctx,VecDestroy_Seq));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  utility methods                                   //
//                                                                                    //
// ================================================================================== //

// v->ops->bindtocpu
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(BindToCPU_CUPMBase(v,usehost,dctx));

  // REVIEW ME: this absolutely should be some sort of bulk mempcy rather than this mess
  VecSetOp_CUPM(dot,VecDot_Seq,dot_async);
  VecSetOp_CUPM(norm,VecNorm_Seq,norm_async);
  VecSetOp_CUPM(tdot,VecTDot_Seq,tdot_async);
  VecSetOp_CUPM(mdot,VecMDot_Seq,mdot_async);
  VecSetOp_CUPM(resetarray,VecResetArray_Seq,resetarray_async<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(placearray,VecPlaceArray_Seq,placearray_async<PETSC_MEMTYPE_HOST>);
  v->ops->mtdot = v->ops->mtdot_local = VecMTDot_Seq;
  VecSetOp_CUPM(conjugate,VecConjugate_Seq,conjugate_async);
  VecSetOp_CUPM(max,VecMax_Seq,max_async);
  VecSetOp_CUPM(min,VecMin_Seq,min_async);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisebinary_async_(BinaryFuncT&& binary, PetscDeviceContext dctx, Vec xin, Vec yin, Vec zout))
{
  PetscFunctionBegin;
  PetscCall(Device::CUPM::Impl::ThrustApplyPointwise<T>(dctx,std::forward<BinaryFuncT>(binary),zout->map->n,DeviceArrayRead(dctx,xin).data(),DeviceArrayRead(dctx,yin).data(),DeviceArrayWrite(dctx,zout).data()));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwiseunary_async_(UnaryFuncT&& unary, PetscDeviceContext dctx, Vec xinout, Vec yin))
{
  PetscFunctionBegin;
  if (!yin || (xinout == yin)) {
    // in-place
    PetscCall(Device::CUPM::Impl::ThrustApplyPointwise<T>(dctx,std::forward<UnaryFuncT>(unary),xinout->map->n,DeviceArrayReadWrite(dctx,xinout).data()));
  } else {
    PetscCall(Device::CUPM::Impl::ThrustApplyPointwise<T>(dctx,std::forward<UnaryFuncT>(unary),xinout->map->n,DeviceArrayRead(dctx,xinout).data(),DeviceArrayWrite(dctx,yin).data()));
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->resetarray or VecCUPMResetArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(base_type::template ResetArray_CUPMBase<mtype>(v,VecResetArray_Seq,dctx));
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(base_type::template PlaceArray_CUPMBase<mtype>(v,a,VecPlaceArray_Seq,dctx));
  PetscFunctionReturn(0);
}

// v->ops->getlocalvector or v->ops->getlocalvectorread
template <Device::CUPM::DeviceType T>
template <PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w, PetscDeviceContext dctx))
{
  PetscBool wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm));
  if (wisseqcupm) {
    if (const auto wseq = VecIMPLCast(w)) {
      if (auto& alloced = wseq->array_allocated) {
        const auto useit = UseCUPMHostAlloc(w);

        PetscCall(PetscFree(alloced));
        w->pinned_memory = PETSC_FALSE;
      }
      wseq->array = wseq->unplacedarray = nullptr;
    }
    if (const auto wcu = VecCUPMCast(w)) {
      if (auto& device_array = wcu->device_array) {
        cupmStream_t stream;

        PetscCall(GetHandles_(dctx,&stream));
        PetscCallCUPM(cupmFreeAsync(device_array,stream));
        device_array = nullptr;
      }
      PetscCall(PetscFree(w->spptr /* wcu */));
    }
  }
  if (v->petscnative && wisseqcupm) {
    PetscCall(PetscFree(w->data));
    w->data          = v->data;
    w->offloadmask   = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr         = v->spptr;
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(w)));
  } else {
    const auto arrayptr = &VecIMPLCast(w)->array;
    if (access == PETSC_MEMORY_ACCESS_READ) {
      PetscCall(VecGetArrayReadAsync(v,const_cast<const PetscScalar**>(arrayptr),dctx));
    } else {
      PetscCall(VecGetArrayAsync(v,arrayptr,dctx));
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcupm) PetscCall(DeviceAllocateCheck_(dctx,w));
  }
  PetscFunctionReturn(0);
}

// v->ops->restorelocalvector or v->ops->restorelocalvectorread
template <Device::CUPM::DeviceType T>
template <PetscMemoryAccessMode access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorelocalvector_async(Vec v, Vec w, PetscDeviceContext dctx))
{
  PetscBool wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm));
  if (v->petscnative && wisseqcupm) {
    v->data          = w->data;
    v->offloadmask   = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr         = w->spptr;
    w->data          = nullptr; // these assignments are __critical__, as w may persist
    w->spptr         = nullptr; // after this call returns and shouldn't share data with v!
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  } else {
    auto array = &VecIMPLCast(w)->array;
    if (access == PETSC_MEMORY_ACCESS_READ) {
      PetscCall(VecRestoreArrayReadAsync(v,const_cast<const PetscScalar**>(array),dctx));
    } else {
      PetscCall(VecRestoreArrayAsync(v,array,dctx));
    }
    if (w->spptr && wisseqcupm) {
      cupmStream_t stream;

      PetscCall(GetHandles_(dctx,&stream));
      PetscCallCUPM(cupmFreeAsync(VecCUPMCast(w)->device_array,stream));
      PetscCall(PetscFree(w->spptr));
    }
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

namespace detail
{

// thrust::complex constructors are not constexpr
// (https://github.com/NVIDIA/thrust/issues/1677), and so you cannot constant-initialize them
// otherwise nvcc throws a
// error: dynamic initialization is not supported for a __constant__ variable
// so make a dummy "complex" type with a constexpr constructor
struct complex_one
{
  const PetscReal r = 1.0;
  const PetscReal i = 0.0;
};
// REVIEW ME:
// 1. find a way to not have this be a global variable. problem is that you cant attach device
//    memory spaces to class member variables (static or otherwise) or inside host functions...
// 2. allocating and de-allocating this one demand seems dumb too...
static const PETSC_CONSTMEM_DECL PETSC_DEVICE_DECL complex_one GlobalDeviceOne = {};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx))
{
  const auto n = static_cast<cupmBlasInt_t>(yin->map->n);
  PetscBool  known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarEqual(alpha,0.0,&known,&equal));
  if (known && equal) {
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx,&stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPM(cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).data(),DeviceArrayRead(dctx,xin).data(),n*sizeof(*DeviceArrayRead(dctx,xin).data()),cupmMemcpyDeviceToDevice,stream));
    PetscCall(PetscLogGpuTimeEnd());
  } else {
    PetscBool         alphaIsOne;
    cupmBlasHandle_t  cupmBlasHandle;
    PetscScalar      *aptr;

    PetscCall(PetscManagedScalarEqual(alpha,1.0,&known,&equal));
    alphaIsOne = static_cast<decltype(alphaIsOne)>(known && equal);
    PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&aptr));
    PetscCall(GetHandles_(dctx,&cupmBlasHandle));
    {
      const auto cmode  = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};
      const auto calpha = cupmScalarCast(aptr);
      const auto yptr   = DeviceArrayReadWrite(dctx,yin);
      const auto xptr   = DeviceArrayRead(dctx,xin);

      PetscCall(PetscLogGpuTimeBegin());
      if (alphaIsOne) {
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle,n,calpha,xptr.cupmdata(),1,yptr.cupmdata(),1));
      } else {
        cupmScalar_t *d_one;

        PetscCallCUPM(cupmGetSymbolAddress(reinterpret_cast<void**>(&d_one),detail::GlobalDeviceOne));
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle,n,calpha,yptr.cupmdata(),1));
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle,n,d_one,xptr.cupmdata(),1,yptr.cupmdata(),1));
      }
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops((alphaIsOne ? 1 : 2)*n));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx))
{
  PetscBool xiscupm,known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarEqual(alpha,0.0,&known,&equal));
  if (PetscUnlikely(known && equal)) PetscFunctionReturn(0);
  PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(xin),&xiscupm,VECSEQCUPM(),VECMPICUPM(),""));
  if (xiscupm) {
    const auto       n = static_cast<cupmBlasInt_t>(yin->map->n);
    PetscScalar     *aptr;
    cupmBlasHandle_t cupmBlasHandle;

    PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&aptr));
    PetscCall(GetHandles_(dctx,&cupmBlasHandle));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle,n,cupmScalarCast(aptr),DeviceArrayRead(dctx,xin),1,DeviceArrayReadWrite(dctx,yin),1));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops(2*n));
  } else {
    PetscCall(VecAXPY_Seq(yin,alpha,xin,dctx));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseDivide_Seq(win,xin,yin,dctx));
  } else {
    // note order of arguments! xin and yin are read, win is written!
    PetscCall(pointwisebinary_async_(thrust::divides<PetscScalar>{},dctx,xin,yin,win));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseMult_Seq(win,xin,yin,dctx));
  } else {
    // note order of arguments! xin and yin are read, win is written!
    PetscCall(pointwisebinary_async_(thrust::multiplies<PetscScalar>{},dctx,xin,yin,win));
  }
  PetscFunctionReturn(0);
}

namespace detail
{

struct reciprocal
{
  PETSC_HOSTDEVICE_INLINE_DECL PetscScalar operator()(PetscScalar s) const
  {
    // yes all of this verbosity is needed because sometimes PetscScalar is a thrust::complex
    // and then it matter whether we do s ? true : false vs s == 0, as well as whether we wrap
    // everything in PetscScalar...
    return s == PetscScalar{0.0} ? s : PetscScalar{1.0}/s;
  }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(pointwiseunary_async_(detail::reciprocal{},dctx,xin));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscManagedScalar alpha, Vec xin, Vec yin, PetscDeviceContext dctx))
{
  PetscBool known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarEqual(alpha,0.0,&known,&equal));
  if (known && equal /* alpha = 0 */) {
    PetscCall(copy_async(yin,win,dctx));
  } else {
    const auto        n    = win->map->n;
    const auto        wptr = DeviceArrayWrite(dctx,win);
    cupmBlasHandle_t  cupmBlasHandle;
    cupmStream_t      stream;
    PetscScalar      *aptr;

    PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&aptr));
    PetscCall(GetHandles_(dctx,&cupmBlasHandle,&stream));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(wptr.data(),DeviceArrayRead(dctx,yin).data(),n*sizeof(*wptr.data()),cupmMemcpyDeviceToDevice,stream));
      PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle,n,cupmScalarCast(aptr),DeviceArrayRead(dctx,xin),1,wptr.cupmdata(),1));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops(2*n));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscManagedInt nv, PetscManagedScalar alpha, Vec *y, PetscDeviceContext dctx))
{
  const auto        n    = xin->map->n;
  const auto        xptr = DeviceArrayReadWrite(dctx,xin);
  cupmBlasHandle_t  cupmBlasHandle;
  PetscInt         *nvptr;
  PetscScalar      *aptr;

  PetscFunctionBegin;
  // implicit sync
  PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&aptr));
  PetscCall(GetHandles_(dctx,&cupmBlasHandle));
  {
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

    PetscCall(PetscLogGpuTimeBegin());
    for (PetscInt j = 0, nvval = *nvptr; j < nvval; ++j) {
      PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle,n,cupmScalarCast(aptr++),DeviceArrayRead(dctx,y[j]),1,xptr.cupmdata(),1));
    }
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops((*nvptr)*2*n));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx))
{
  const auto        n = xin->map->n;
  PetscScalar      *zptr;
  cupmBlasHandle_t  cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx,&cupmBlasHandle));
  PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&zptr));
  {
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

    PetscCall(PetscLogGpuTimeBegin());
    // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
    // second
    PetscCallCUPMBLAS(cupmBlasXdot(cupmBlasHandle,n,DeviceArrayRead(dctx,yin),1,DeviceArrayRead(dctx,xin),1,cupmScalarCast(zptr)));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(PetscMax(2*n-1,0)));
  PetscFunctionReturn(0);
}

#define MDOT_WORKGROUP_NUM  128
#define MDOT_WORKGROUP_SIZE MDOT_WORKGROUP_NUM

namespace kernels
{

PETSC_HOSTDEVICE_INLINE_DECL static PetscInt EntriesPerGroup(PetscInt size)
{
  const auto group_entries = (size-1)/gridDim.x+1;
  // for very small vectors, a group should still do some work
  return group_entries ? group_entries : 1;
}

template <int N>
PETSC_KERNEL_DECL static void mdot_kernel(const PetscScalar *PETSC_RESTRICT x, const PetscScalar *PETSC_RESTRICT y[N], PetscInt size, PetscScalar *PETSC_RESTRICT results)
{
  static_assert(N > 0,"");
  PETSC_SHAREDMEM_DECL PetscScalar shmem[N*MDOT_WORKGROUP_SIZE];
  // HIP -- for whatever reason -- has threadIdx, blockIdx, blockDim, and gridDim as separate
  // types, so each of these go on separate lines...
  const auto         tx       = threadIdx.x;
  const auto         bx       = blockIdx.x;
  const auto         bdx      = blockDim.x;
  const auto         gdx      = gridDim.x;
  const auto         worksize = EntriesPerGroup(size);
  const auto         begin    = tx+bx*worksize;
  const auto         end      = min((bx+1)*worksize,size);
  const PetscScalar *ylocal[N];
  PetscScalar        sumlocal[N];

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
      //for (auto i = tx; i < N; i += MDOT_WORKGROUP_SIZE) shmem[i] += shmem[i+stride];
      for (auto i = 0; i < N; ++i) shmem[tx+i*MDOT_WORKGROUP_SIZE] += shmem[tx+stride+i*MDOT_WORKGROUP_SIZE];
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar **device_y, PetscScalar *results, PetscInt &yidx))
{
  PetscScalar *host_y[N];

  PetscFunctionBegin;
  for (auto i = 0; i < N; ++i) host_y[i] = DeviceArrayRead(dctx,yin[i+yidx]);
  PetscCallCUPM(cupmMemcpyAsync(device_y,host_y,N*sizeof(*device_y),cupmMemcpyDefault,stream));
  PetscCallCUPM(cupmLaunchKernel(kernels::mdot_kernel<N>,MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE,0,stream,xarr,device_y,size,results+yidx*MDOT_WORKGROUP_NUM));
  yidx += N;
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<false>, Vec xin, PetscInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx))
{
  constexpr auto   batchsize = 8;
  const auto       n         = xin->map->n;
  // number of vectors that we handle via the batches. note any singletons are handled by
  // cublas, hence the nv-1.
  const auto       nvbatch   = ((nv % batchsize) == 1) ? nv-1 : nv;
  const auto       nwork     = nvbatch*MDOT_WORKGROUP_NUM;
  PetscScalar     *d_results;
  const auto       nbytes    = nwork*sizeof(*d_results);
  PetscScalar    **d_y;
  cupmStream_t     stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx,&stream));
  // will hold all the device y pointers
  PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void**>(&d_y),batchsize*sizeof(*d_y),stream));
  PetscAssert(nv > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nv %" PetscInt_FMT " < 0",nv);
  // allocate scratchpad memory for the results of individual work groups
  PetscCallCUPM(cupmMallocAsync(reinterpret_cast<void**>(&d_results),nbytes,stream));
  {
    const auto xptr = DeviceArrayRead(dctx,xin);
    auto       yidx = PetscInt{0};

    // REVIEW ME: Can fork-join here, but should probably only have a single-sized kernel then
    // REVIEW ME: Should probably try and load-balance these. Consider the case where nv = 9;
    // it is very likely better to do 4+5 rather and 8+1
    PetscCall(PetscLogGpuTimeBegin());
    do {
      switch (nv-yidx) {
      case 7:
        PetscCall(mdot_kernel_dispatch_<7>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      case 6:
        PetscCall(mdot_kernel_dispatch_<6>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      case 5:
        PetscCall(mdot_kernel_dispatch_<5>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      case 4:
        PetscCall(mdot_kernel_dispatch_<4>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      case 3:
        PetscCall(mdot_kernel_dispatch_<3>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      case 2:
        PetscCall(mdot_kernel_dispatch_<2>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      case 1: {
        cupmBlasHandle_t  cupmBlasHandle;
        PetscScalar      *zptr;

        // REVIEW ME: this is done with device memory since cublas can be run asynchronously in
        // this case, but we copy everything back up to the cpu below...
        PetscCall(GetHandles_(dctx,&cupmBlasHandle));
        PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&zptr));
        const auto with = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};
        PetscCallCUPMBLAS(cupmBlasXdot(cupmBlasHandle,static_cast<cupmBlasInt_t>(n),DeviceArrayRead(dctx,yin[yidx]),1,xptr.cupmdata(),1,cupmScalarCast(zptr+yidx)));
        // the reason this ensure offload exists is because:
        // 1. We do getvalues(HOST) below with PETSC_MEMORY_ACCESS_WRITE
        // 2. This just passes the host pointer directly which is populated (up until the last
        //    value) on the host.
        // 3. This sets offload to PETSC_OFFLOAD_CPU
        // 4. On next read op on device the host array is copied to the device overwriting
        //    whatever cublas put there with whatever junk was in the last array index
        //
        // so we need to copy the _whole_ array up just to make sure the host gets it. note
        // this would not be necessary if we had a device-only implementation
        PetscCall(PetscManagedScalarEnsureOffload(dctx,z,PETSC_OFFLOAD_BOTH));
        ++yidx;
      } break;
      default: // 8 or more
        PetscCall(mdot_kernel_dispatch_<8>(dctx,stream,xptr.data(),yin,n,d_y,d_results,yidx));
        break;
      }
    } while (yidx < nv);
    PetscCall(PetscLogGpuTimeEnd());
  }
  // copy results to CPU
  // REVIEW ME: it is likely better to do this in a micro kernel rather than do it on the
  // host which requires synchronization and possibly an additional allocation
  {
    auto         stackarray = std::array<PetscScalar,PETSC_MAX_PATH_LEN>{};
    const auto   allocate   = static_cast<decltype(stackarray.size())>(nwork) > stackarray.size();
    auto         h_results  = stackarray.data();
    PetscScalar *zptr;

    if (allocate) PetscCallCUPM(cupmMallocHost(reinterpret_cast<void**>(&h_results),nbytes));
    PetscCallCUPM(cupmMemcpyAsync(h_results,d_results,nbytes,cupmMemcpyDeviceToHost,stream));
    // do these now while memcpy is in flight
    PetscCall(PetscLogFlops(nwork));
    PetscCall(PetscLogGpuToCpuScalar(nbytes));
    // implicit sync
    PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,PETSC_TRUE,&zptr));
    // sum group results into z
    for (PetscInt j = 0; j < nvbatch; ++j) {
      zptr[j] = 0;
      for (auto i = j*MDOT_WORKGROUP_NUM; i < (j+1)*MDOT_WORKGROUP_NUM; ++i) zptr[j] += h_results[i];
    }
    if (allocate) PetscCallCUPM(cupmFreeHost(h_results));
  }
  // free these here, we will already have synchronized
  PetscCallCUPM(cupmFreeAsync(d_results,stream));
  PetscCallCUPM(cupmFreeAsync(d_y,stream));
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_NUM
#undef MDOT_WORKGROUP_SIZE

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<true>, Vec xin, PetscInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx))
{
  // probably not worth it to run more than 8 of these at a time?
  const auto          n_sub = PetscMin(nv,8);
  const auto          n     = static_cast<cupmBlasInt_t>(xin->map->n);
  const auto          xptr  = DeviceArrayRead(dctx,xin);
  PetscScalar        *zptr;
  PetscDeviceContext *subctx;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&zptr));
  PetscCall(PetscDeviceContextFork(dctx,n_sub,&subctx));
  PetscCall(PetscLogGpuTimeBegin());
  for (decltype(nv) i = 0; i < nv; ++i) {
    cupmBlasHandle_t handle;

    PetscCall(GetHandles_(subctx[i%n_sub],&handle));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{handle,CUPMBLAS_POINTER_MODE_DEVICE};
      PetscCallCUPMBLAS(cupmBlasXdot(handle,n,DeviceArrayRead(dctx,yin[i]),1,xptr.cupmdata(),1,cupmScalarCast(zptr+i)));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscDeviceContextJoin(dctx,n_sub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subctx));
  // REVIEW ME: flops?????
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async(Vec xin, PetscManagedInt nv, const Vec yin[], PetscManagedScalar z, PetscDeviceContext dctx))
{
  PetscBool known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntEqual(nv,1,&known,&equal));
  if (PetscUnlikely(known && equal) /* nv = 1 */) {
    PetscCall(dot_async(xin,PetscRemoveConstCast(yin[0]),z,dctx));
  } else if (const auto n = xin->map->n) {
    PetscInt *nvptr;

    // implicity sync
    PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
    PetscCheck(*nvptr > 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to %s %" PetscInt_FMT " not positive",PETSC_FUNCTION_NAME,*nvptr);

    PetscCall(mdot_async_(UseComplexTag<PetscDefined(USE_COMPLEX)>{},xin,*nvptr,yin,z,dctx));
    // REVIEW ME: double count of flops??
    PetscCall(PetscLogGpuFlops(PetscMax((*nvptr)*(2*n-1),0)));
  } else {
    const auto zero = PetscScalar{0};

    PetscCall(PetscManagedScalarApplyOperator(dctx,z,PETSC_OPERATOR_EQUAL,PETSC_MEMTYPE_HOST,&zero,nullptr));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx))
{
  const auto   n    = xin->map->n;
  const auto   xptr = DeviceArrayWrite(dctx,xin);
  PetscBool    known,equal;
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx,&stream));
  PetscCall(PetscManagedScalarEqual(alpha,0.0,&known,&equal));
  if (known && equal /* alpha = 0 */) {
    PetscCallCUPM(cupmMemsetAsync(xptr.data(),0,n*sizeof(*xptr.data()),stream));
  } else {
    PetscScalar *ptr;

    // implicit sync
    PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&ptr));
    PetscCall(Device::CUPM::Impl::ThrustSet<T>(stream,n,xptr.data(),ptr));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx))
{
  PetscBool known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarEqual(alpha,1.0,&known,&equal));
  if (known && equal) PetscFunctionReturn(0);
  PetscCall(PetscManagedScalarEqual(alpha,0.0,&known,&equal));
  if (known && equal) {
    PetscCall(set_async(xin,alpha,dctx));
  } else {
    const auto        n = static_cast<cupmBlasInt_t>(xin->map->n);
    PetscScalar      *aptr;
    cupmBlasHandle_t  cupmBlasHandle;

    PetscCall(GetHandles_(dctx,&cupmBlasHandle));
    PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&aptr));
    {
      const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle,n,cupmScalarCast(aptr),DeviceArrayReadWrite(dctx,xin),1));
      PetscCall(PetscLogGpuTimeEnd());
    }
    PetscCall(PetscLogGpuFlops(n));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx))
{
  const auto        n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscScalar      *zptr;
  cupmBlasHandle_t  cupmBlasHandle;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx,&cupmBlasHandle));
  PetscCall(PetscManagedScalarGetValues(dctx,z,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&zptr));
  {
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXdotu(cupmBlasHandle,n,DeviceArrayRead(dctx,xin),1,DeviceArrayRead(dctx,yin),1,cupmScalarCast(zptr)));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(PetscMax(2*n-1,0)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto   n       = xin->map->n;
    const auto   nbytes  = n*sizeof(*VecIMPLCast(xin)->array);
    auto         yiscupm = PETSC_TRUE,xondevice = PETSC_TRUE; // assume we start on device
    // silence buggy gcc warning: ‘mode’ may be used uninitialized in this function
    auto         mode    = cupmMemcpyDeviceToDevice;
    cupmStream_t stream;

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
      PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(yin),&yiscupm,VECSEQCUPM(),VECMPICUPM(),""));
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

    PetscCall(GetHandles_(dctx,&stream));
    switch (mode) {
    case cupmMemcpyDeviceToDevice:
      // the best case
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).data(),DeviceArrayRead(dctx,xin).data(),nbytes,mode,stream));
      PetscCall(PetscLogGpuTimeEnd());
      break;
    case cupmMemcpyHostToDevice:
      // not terrible
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(DeviceArrayWrite(dctx,yin).data(),HostArrayRead(dctx,xin).data(),nbytes,mode,stream));
      PetscCall(PetscLogGpuTimeEnd());
      break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yptr;

      PetscCall(VecGetArrayWriteAsync(yin,&yptr,dctx));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPM(cupmMemcpyAsync(yptr,DeviceArrayRead(dctx,xin).data(),nbytes,mode,stream));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecRestoreArrayWriteAsync(yin,&yptr,dctx));
    } break;
    case cupmMemcpyHostToHost:   {
      // the worst case
      PetscScalar *yptr;

      PetscCall(VecGetArrayWrite(yin,&yptr));
      PetscCall(PetscArraycpy(yptr,HostArrayRead(dctx,xin).data(),n));
      PetscCall(VecRestoreArrayWrite(yin,&yptr));
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
    }
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  if (xin != yin) {
    cupmBlasHandle_t cupmBlasHandle;

    PetscCall(GetHandles_(dctx,&cupmBlasHandle));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXswap(cupmBlasHandle,static_cast<cupmBlasInt_t>(xin->map->n),DeviceArrayReadWrite(dctx,xin),1,DeviceArrayReadWrite(dctx,yin),1));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpby_async(Vec yin, PetscManagedScalar alpha, PetscManagedScalar beta, Vec xin, PetscDeviceContext dctx))
{
  PetscBool known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarEqual(alpha,0.0,&known,&equal));
  if (known && equal /* alpha = 0 */) {
    PetscCall(scale_async(yin,beta,dctx));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscManagedScalarEqual(beta,1.0,&known,&equal));
  if (known && equal /* beta = 1 */) {
    PetscCall(axpy_async(yin,alpha,xin,dctx));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscManagedScalarEqual(alpha,1.0,&known,&equal));
  if (known && equal /* alpha = 1 */) {
    PetscCall(aypx_async(yin,beta,xin,dctx));
  } else {
    const auto        n    = static_cast<cupmBlasInt_t>(yin->map->n);
    const auto        xptr = DeviceArrayRead(dctx,xin);
    cupmBlasHandle_t  cupmBlasHandle;
    PetscScalar      *aptr;

    PetscCall(PetscManagedScalarEqual(beta,0.0,&known,&equal));
    PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&aptr));
    PetscCall(GetHandles_(dctx,&cupmBlasHandle));
    {
      const auto bzero  = known && equal;
      const auto calpha = cupmScalarCast(aptr);
      const auto cmode  = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};

      if (bzero /* beta = 0 */) {
        // here we can get away with purely write-only as we memcpy into it first
        const auto   yptr = DeviceArrayWrite(dctx,yin);
        cupmStream_t stream;

        PetscCall(GetHandles_(dctx,&stream));
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUPM(cupmMemcpyAsync(yptr.data(),xptr.data(),n*sizeof(*yptr.data()),cupmMemcpyDeviceToDevice,stream));
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle,n,calpha,yptr.cupmdata(),1));
      } else {
        const auto   yptr = DeviceArrayReadWrite(dctx,yin);
        PetscScalar *bptr;

        PetscCall(PetscManagedScalarGetValues(dctx,beta,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&bptr));
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUPMBLAS(cupmBlasXscal(cupmBlasHandle,n,cupmScalarCast(bptr),yptr.cupmdata(),1));
        PetscCallCUPMBLAS(cupmBlasXaxpy(cupmBlasHandle,n,calpha,xptr.cupmdata(),1,yptr.cupmdata(),1));
      }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(PetscLogGpuFlops((bzero ? 1 : 3)*n));
    }
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpbypcz_async(Vec zin, PetscManagedScalar alpha, PetscManagedScalar beta, PetscManagedScalar gamma, Vec xin, Vec yin, PetscDeviceContext dctx))
{
  PetscBool known,equal;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarEqual(gamma,1.0,&known,&equal));
  // note this fires even if gamma may secretely be equal to 1 since known may return false
  if (!(known && equal) /* gamma != 1.0 */) PetscCall(scale_async(zin,gamma,dctx));
  PetscCall(axpy_async(zin,alpha,xin,dctx));
  PetscCall(axpy_async(zin,beta,yin,dctx));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  if (const auto n = static_cast<cupmBlasInt_t>(xin->map->n)) {
    const auto        xptr      = DeviceArrayRead(dctx,xin);
    PetscInt          flopCount = 0;
    cupmBlasHandle_t  cupmBlasHandle;
    PetscReal        *zptr;

    PetscCall(GetHandles_(dctx,&cupmBlasHandle));
    // in case NORM_INFINITY we may technically undo this, but cleaner this way
    const auto cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_DEVICE};
    PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_WRITE,PETSC_FALSE,&zptr));
    PetscCall(PetscLogGpuTimeBegin());
    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      PetscCallCUPMBLAS(cupmBlasXasum(cupmBlasHandle,n,xptr.cupmdata(),1,cupmRealCast(zptr)));
      flopCount = PetscMax(n-1,0);
      if (type == NORM_1) break;
      ++zptr; // fall-through
    case NORM_2:
    case NORM_FROBENIUS:
      PetscCallCUPMBLAS(cupmBlasXnrm2(cupmBlasHandle,n,xptr.cupmdata(),1,cupmRealCast(zptr)));
      flopCount += PetscMax(2*n-1,0); // += in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      const auto    cmode = WithCUPMBlasPointerMode<T>{cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST};
      cupmStream_t  stream;
      PetscScalar   zs;
      cupmBlasInt_t i;

      // REVIEW ME: this needs to be redone by hand
      PetscCall(GetHandles_(dctx,&stream));
      PetscCallCUPMBLAS(cupmBlasXamax(cupmBlasHandle,n,xptr.cupmdata(),1,&i));
      PetscCallCUPM(cupmMemcpyAsync(&zs,xptr.data()+i-1,sizeof(zs),cupmMemcpyDeviceToHost,stream));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      const auto zr = PetscAbsScalar(zs);
      PetscCall(PetscManagedRealSetValues(dctx,z,PETSC_MEMTYPE_HOST,&zr,1));
      // REVIEW ME: flopCount = ???
    } break;
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(flopCount));
  } else {
    constexpr PetscReal zero[] = {0.0,0.0};

    PetscCall(PetscManagedRealSetValues(dctx,z,PETSC_MEMTYPE_HOST,zero,1+(type == NORM_1_AND_2)));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscManagedScalar dp, PetscManagedScalar nm, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(dot_async(s,t,dp,dctx));
  PetscCall(dot_async(t,t,nm,dctx));
  PetscFunctionReturn(0);
}

namespace detail
{

struct conjugate
{
  PETSC_HOSTDEVICE_INLINE_DECL PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec xin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) PetscCall(pointwiseunary_async_(detail::conjugate{},dctx,xin));
  PetscFunctionReturn(0);
}

namespace detail
{

struct real_part
{
  PETSC_HOSTDEVICE_INLINE_DECL
  thrust::tuple<PetscReal,PetscInt> operator()(const thrust::tuple<PetscScalar,PetscInt>& x) const
  {
    return thrust::make_tuple(PetscRealPart(x.get<0>()),x.get<1>());
  }

  PETSC_HOSTDEVICE_INLINE_DECL PetscReal operator()(PetscScalar x) const { return PetscRealPart(x); }
};

template <typename Operator>
struct tuple_compare
{
  using tuple_type = thrust::tuple<PetscReal,PetscInt>;

  PETSC_HOSTDEVICE_INLINE_DECL tuple_type operator()(const tuple_type& x, const tuple_type& y) const
  {
    if (Operator{}(y.get<0>(),x.get<0>())) {
      // if y is strictly greater/less than x, return y
      return y;
    } else if (y.get<0>() == x.get<0>()) {
      // if equal, prefer lower index
      return y.get<1>() < x.get<1>() ? y : x;
    } else {
      // otherwise return x
      return x;
    }
  }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
template <typename TupleFuncT, typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::minmax_async_(TupleFuncT&& tuple_ftr, UnaryFuncT&& unary_ftr, PetscReal mval, Vec v, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (const auto n = v->map->n) {
    const auto   vptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).data());
    cupmStream_t stream;

    PetscCall(GetHandles_(dctx,&stream));
    // needed to:
    // 1. switch between transform_reduce and reduce
    // 2. strip the real_part functor from the arguments
#if PetscDefined(USE_COMPLEX)
#  define THRUST_MINMAX_REDUCE(...) THRUST_CALL(thrust::transform_reduce,__VA_ARGS__)
#else
#  define THRUST_MINMAX_REDUCE(s,b,e,real_part__,...) THRUST_CALL(thrust::reduce,s,b,e,__VA_ARGS__)
#endif
    if (p) {
      auto       pval = PetscInt{-1};
      const auto zip  = thrust::make_zip_iterator(
        thrust::make_tuple(std::move(vptr),thrust::make_counting_iterator(PetscInt{0}))
      );
      // need to use preprocessor conditionals since otherwise thrust complains about not being
      // able to convert a thrust::device_reference<PetscScalar> to a PetscReal on complex
      // builds...
      PetscCallThrust(
        thrust::tie(mval,pval) = THRUST_MINMAX_REDUCE(
          stream,zip,zip+n,detail::real_part{},
          thrust::make_tuple(mval,pval),std::forward<TupleFuncT>(tuple_ftr)
        );
      );
      PetscCall(PetscManagedIntSetValues(dctx,p,PETSC_MEMTYPE_HOST,&pval,1));
    } else {
      PetscCallThrust(
        mval = THRUST_MINMAX_REDUCE(
          stream,vptr,vptr+n,detail::real_part{},mval,std::forward<UnaryFuncT>(unary_ftr)
        );
      );
    }
#undef THRUST_MINMAX_REDUCE
  }
  PetscCall(PetscManagedRealSetValues(dctx,m,PETSC_MEMTYPE_HOST,&mval,1));
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx))
{
  using value_type    = util::remove_pointer_t<decltype(m->device)>;
  using tuple_functor = detail::tuple_compare<thrust::greater<value_type>>;
  using unary_functor = thrust::maximum<value_type>;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  PetscCall(minmax_async_(tuple_functor{},unary_functor{},PETSC_MIN_REAL,v,p,m,dctx));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscManagedInt p, PetscManagedReal m, PetscDeviceContext dctx))
{
  using value_type    = util::remove_pointer_t<decltype(m->device)>;
  using tuple_functor = detail::tuple_compare<thrust::less<value_type>>;
  using unary_functor = thrust::minimum<value_type>;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  PetscCall(minmax_async_(tuple_functor{},unary_functor{},PETSC_MAX_REAL,v,p,m,dctx));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscManagedScalar sum, PetscDeviceContext dctx))
{
  const auto   n    = v->map->n;
  const auto   dptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).data());
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(dctx,&stream));
  // REVIEW ME: why not cupmBlasXasum()?
  PetscCallThrust(
    const auto ret = THRUST_CALL(thrust::reduce,stream,dptr,dptr+n,PetscScalar{0.0});
    // need this inside since we have to capture the output
    PetscCall(PetscManagedScalarSetValues(dctx,sum,PETSC_MEMTYPE_HOST,&ret,1));
  );
  // REVIEW ME: must be at least n additions
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscManagedScalar shift, PetscDeviceContext dctx))
{
  PetscScalar *ptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,shift,PETSC_MEMTYPE_DEVICE,PETSC_MEMORY_ACCESS_READ,PETSC_FALSE,&ptr));
  PetscCall(pointwiseunary_async_(Device::CUPM::Impl::make_shift_operator(ptr,thrust::plus<PetscScalar>{}),dctx,v));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand, PetscDeviceContext dctx))
{
  const auto n = v->map->n;
  PetscBool  iscurand;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(rand),PETSCCURAND,&iscurand));
  if (iscurand) PetscCall(PetscRandomGetValues(rand,n,DeviceArrayWrite(dctx,v)));
  else PetscCall(PetscRandomGetValues(rand,n,HostArrayWrite(dctx,v)));
  // REVIEW ME: flops????
  // REVIEW ME: Timing???
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                 implementations                                    //

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCreateSeqCUPMAsync(T&& VecSeq_CUPM_Impls, MPI_Comm comm, PetscInt n, PetscDeviceContext dctx, Vec *v))
{
  PetscFunctionBegin;
  PetscValidPointer(v,4);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.createseqcupm_async(comm,0,n,dctx,v,PETSC_TRUE));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCreateSeqCUPMWithArraysAsync(T&& VecSeq_CUPM_Impls, MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar cpuarray[], const PetscScalar gpuarray[], PetscDeviceContext dctx, Vec *v))
{
  PetscFunctionBegin;
  if (n && cpuarray) PetscValidScalarPointer(cpuarray,5);
  PetscValidPointer(v,7);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.createseqcupmwithbotharrays_async(comm,bs,n,cpuarray,gpuarray,dctx,v));
  PetscFunctionReturn(0);
}

template <PetscMemoryAccessMode mode, typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayAsync_Private(T&& VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidPointer(a,3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template getarray_async<PETSC_MEMTYPE_DEVICE,mode>(v,a,dctx));
  PetscFunctionReturn(0);
}

template <PetscMemoryAccessMode mode, typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayAsync_Private(T&& VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidPointer(a,3);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template restorearray_async<PETSC_MEMTYPE_DEVICE,mode>(v,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayAsync(T&& VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_READ_WRITE>(std::forward<T>(VecSeq_CUPM_Impls),v,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayAsync(T&& VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_READ_WRITE>(std::forward<T>(VecSeq_CUPM_Impls),v,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayReadAsync(T&& VecSeq_CUPM_Impls, Vec v, const PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_READ>(std::forward<T>(VecSeq_CUPM_Impls),v,const_cast<PetscScalar**>(a),dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayReadAsync(T&& VecSeq_CUPM_Impls, Vec v, const PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_READ>(std::forward<T>(VecSeq_CUPM_Impls),v,const_cast<PetscScalar**>(a),dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMGetArrayWriteAsync(T&& VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(VecCUPMGetArrayAsync_Private<PETSC_MEMORY_ACCESS_WRITE>(std::forward<T>(VecSeq_CUPM_Impls),v,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMRestoreArrayWriteAsync(T&& VecSeq_CUPM_Impls, Vec v, PetscScalar **a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(VecCUPMRestoreArrayAsync_Private<PETSC_MEMORY_ACCESS_WRITE>(std::forward<T>(VecSeq_CUPM_Impls),v,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMPlaceArrayAsync(T&& VecSeq_CUPM_Impls, Vec vin, const PetscScalar a[], PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template placearray_async<PETSC_MEMTYPE_DEVICE>(vin,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMReplaceArrayAsync(T&& VecSeq_CUPM_Impls, Vec vin, const PetscScalar a[], PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template replacearray_async<PETSC_MEMTYPE_DEVICE>(vin,a,dctx));
  PetscFunctionReturn(0);
}

template <typename T>
PETSC_CXX_COMPAT_DECL(PetscErrorCode VecCUPMResetArrayAsync(T&& VecSeq_CUPM_Impls, Vec vin, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(VecSeq_CUPM_Impls.template resetarray_async<PETSC_MEMTYPE_DEVICE>(vin,dctx));
  PetscFunctionReturn(0);
}

// REVIEW ME: TODO figure out a way around gcc 9.3.1 linker bug to get these to
// work. Everything compiles but get linker error in optimized build for mdot_async_(). I think
// this is because it is overloaded function and since we __always__ only call one overload,
// compiler doesn't generate the object code for the other overload...
// /usr/bin/ld: arch-linux-c-opt/lib/libpetsc.so: undefined reference to
// `Petsc::Vector::CUPM::Impl::VecSeq_CUPM<(Petsc::Device::CUPM::DeviceType)0>::mdot_async_(Petsc::Vector::CUPM::Impl::(anonymousnamespace)::UseComplexTag<false>,
// _p_Vec*, int, _p_Vec* const*, double*)'

// declare the extern templates, each is explicitly instantiated in the respective
// implementation directories
#if PetscDefined(HAVE_CUDA)
//extern template struct VecSeq_CUPM<Device::CUPM::DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
//extern template struct VecSeq_CUPM<Device::CUPM::DeviceType::HIP>;
#endif

} // namespace Impl

} // namespace CUPM

} // namespace Vector

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
