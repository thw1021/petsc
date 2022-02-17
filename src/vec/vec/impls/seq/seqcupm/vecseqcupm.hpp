#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/veccupmimpl.h>   /*I <petscvec.h> I*/
#include <petsc/private/randomimpl.h>      // for _p_PetscRandom

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

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

private:
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast_(Vec v)) PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_Seq*>(v->data));
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 auto VECTYPE_()) PETSC_DECLTYPE_AUTO_RETURNS(VECSEQCUPM());

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
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_kernel_dispatch_(PetscDeviceContext,cupmStream_t,const PetscScalar*,const Vec[],PetscInt,PetscScalar**,PetscScalar*,PetscInt*));
  // common core for the various create routines
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async_(Vec,PetscScalar*/*host_ptr*/=nullptr,PetscScalar*/*device_ptr*/=nullptr));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm,PetscInt,PetscInt,Vec*,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupmwithbotharrays_async(MPI_Comm,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Vec*));

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
  template <MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec,Vec));
  template <MemoryAccess>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec,Vec));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*));
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
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());      \
    }                                                                           \
  } while (0)

// rocThrust has no par_nosync
#if defined(__NVCC__) && !PetscDefined(USE_DEBUG) && (THRUST_VERSION >= 101600)
#  define THRUST_ASYNC_T par_nosync
#else
#  define THRUST_ASYNC_T par
#endif

namespace
{

struct PetscLogGpuTime_
{
  /* lets hope barry doesn't notice these */
  PetscLogGpuTime_()  noexcept { CHKERRABORT(PETSC_COMM_SELF,PetscLogGpuTimeBegin()); }
  ~PetscLogGpuTime_() noexcept { CHKERRABORT(PETSC_COMM_SELF,PetscLogGpuTimeEnd());   }
};

} // anonymous namespace

#define THRUST_CALL(thrust_func,stream,...) [&]{                                   \
    const auto timer = PetscLogGpuTime_{};                                         \
    return thrust_func(thrust::cuda::THRUST_ASYNC_T.on(stream),__VA_ARGS__);       \
  }()

template <Device::CUPM::DeviceType T>
template <typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisebinary_async_(BinaryFuncT&& binary, Vec win, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  CHKERRTHRUST(
    auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,xin).ptr);
    auto yptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,yin).ptr);
    auto wptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,win).ptr);

    THRUST_CALL(thrust::transform,stream,xptr,xptr+n,yptr,wptr,std::forward<BinaryFuncT>(binary));
  );
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwiseunary_async_(UnaryFuncT&& unary, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  if (xin == yin || !yin) { // in-place
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(DeviceArrayReadWrite(dctx,xin).ptr);

      THRUST_CALL(thrust::transform,stream,xptr,xptr+n,xptr,std::forward<UnaryFuncT>(unary));
    );
  } else {
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,xin).ptr);
      auto yptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,yin).ptr);

      THRUST_CALL(thrust::transform,stream,xptr,xptr+n,yptr,std::forward<UnaryFuncT>(unary));
    );
  }
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
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must create VecSeq on communicator of size 1, have size %d",size);
  // REVIEW ME: remove me
  PetscCheck(!VecIMPLCast(v),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Creating VecSeq for the second time!");
  ierr = VecCreate_Seq_Private(v,host_array);CHKERRQ(ierr);
  ierr = Initialize_CUPMBase(v,PETSC_FALSE,host_array,device_array);CHKERRQ(ierr);
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
  ierr = Create_CUPMBase(comm,bs,n,n,v,call_set_type);CHKERRQ(ierr);
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
  ierr = Duplicate_CUPMBase(v,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Destroy_CUPMBase(v);CHKERRQ(ierr);
  ierr = VecDestroy_Seq(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// v->ops->bindtocpu
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BindToCPU_CUPMBase(v,usehost);CHKERRQ(ierr);

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

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->getlocalvector or v->ops->getlocalvectorread
template <Device::CUPM::DeviceType T>
template <MemoryAccess access>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w))
{
  PetscErrorCode ierr;
  PetscBool      wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm);CHKERRQ(ierr);
  if (wisseqcupm) {
    if (const auto wseq = VecIMPLCast(w)) {
      if (wseq->array_allocated) {
        const auto useit = UseCUPMHostAlloc(w);

        ierr = PetscFree(wseq->array_allocated);CHKERRQ(ierr);
        w->pinned_memory = PETSC_FALSE;
      }
      wseq->array = wseq->unplacedarray = nullptr;
    }
    if (const auto wcu = VecCUPMCast(w)) {
      if (auto device_array = wcu->device_array) {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = GetHandles_(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(device_array,stream);CHKERRCUPM(cerr);
      }
      ierr = PetscFree(w->spptr /* wcu */);CHKERRQ(ierr);
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
    if (access == MemoryAccess::READ) {
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
template <MemoryAccess access>
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
    w->spptr         = nullptr;
    w->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  } else {
    auto array = &VecIMPLCast(w)->array;
    if (access == MemoryAccess::READ) {
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

// v->ops->resetarray or VecCUPMResetArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = base_type::template ResetArray_CUPMBase<mtype>(v,VecResetArray_Seq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = base_type::template PlaceArray_CUPMBase<mtype>(v,a,VecPlaceArray_Seq);CHKERRQ(ierr);
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
    const auto   nbytes = n*sizeof(PetscScalar);
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
    {
      cupmBlasError_t cberr;
      const auto      calpha = makeCupmScalar(alpha);
      const auto      yarray = DeviceArrayReadWrite(dctx,yin);
      const auto      xarray = DeviceArrayRead(dctx,xin);

      cberr = cupmBlasSetPointerMode(cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      if (alphaIsOne) {
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&calpha,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        const auto sone = makeCupmScalar(1.0);

        cberr = cupmBlasXscal(cupmBlasHandle,n,&calpha,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&sone,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    }
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
    const auto         calpha = makeCupmScalar(alpha);
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    cberr = cupmBlasSetPointerMode(cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXaxpy(cupmBlasHandle,n,&calpha,DeviceArrayRead(dctx,xin),1,DeviceArrayReadWrite(dctx,yin),1);CHKERRCUPMBLAS(cberr);
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
  PETSC_HOSTDEVICE_DECL PetscScalar operator()(PetscScalar s) const
  {
    // yes all of this verbosity is needed because sometimes PetscScalar is a thrust::complex
    // and then it matter whether we do s ? true : false vs s == 0, as well as whether we wrap
    // everything in PetscScalar...
    return s == PetscScalar{0.0} ? s : PetscScalar{1.0}/s;
  }
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
    const auto         calpha = makeCupmScalar(alpha);
    PetscDeviceContext dctx;
    cupmBlasError_t    cberr;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmStream_t       stream;

    ierr = GetHandles_(&dctx,&cupmBlasHandle,&stream);CHKERRQ(ierr);
    cberr = cupmBlasSetPointerMode(cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      const auto warray = DeviceArrayWrite(dctx,win);
      const auto cerr   = cupmMemcpyAsync(warray.ptr,DeviceArrayRead(dctx,yin).ptr,n*sizeof(typename decltype(warray)::value_type),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
      cberr  = cupmBlasXaxpy(cupmBlasHandle,n,&calpha,DeviceArrayRead(dctx,xin),1,warray,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(calpha));CHKERRQ(ierr);
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
  {
    const auto xarray = DeviceArrayReadWrite(dctx,xin);

    ierr = cupmBlasSetPointerModeFromPointer(cupmBlasHandle,alpha);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    for (decltype(nv) j = 0; j < nv; ++j) {
      auto cberr = cupmBlasXaxpy(cupmBlasHandle,n,cupmScalarCast(alpha+j),DeviceArrayRead(dctx,y[j]),1,xarray,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
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
  ierr = cupmBlasSetPointerModeFromPointer(cupmBlasHandle,z);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdot(cupmBlasHandle,n,DeviceArrayRead(dctx,yin),1,DeviceArrayRead(dctx,xin),1,cupmScalarCast(z));CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*n-1,0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define MDOT_WORKGROUP_NUM  128
#define MDOT_WORKGROUP_SIZE MDOT_WORKGROUP_NUM

namespace kernels
{

PETSC_HOSTDEVICE_DECL static PetscInt EntriesPerGroup(PetscInt size)
{
  const auto group_entries = (size-1)/gridDim.x+1;
  // for very small vectors, a group should still do some work
  return group_entries ? group_entries : 1;
}

template <int N>
PETSC_KERNEL_DECL static void mdot_kernel(const PetscScalar *PETSC_RESTRICT x, const PetscScalar *PETSC_RESTRICT y[N], PetscInt size, PetscScalar *PETSC_RESTRICT results)
{
  static_assert(N > 1,"");
  PETSC_SHAREDMEM_DECL PetscScalar shmem[N*MDOT_WORKGROUP_SIZE];
  const auto tx       = threadIdx.x,bx = blockIdx.x;
  const auto bdx      = blockDim.x,gdx = gridDim.x;
  const auto worksize = EntriesPerGroup(size);
  const auto begin    = tx+bx*worksize;
  const auto end      = min((bx+1)*worksize,size);
  const PetscScalar *ylocal[N];
  PetscScalar       sumlocal[N];

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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_kernel_dispatch_(PetscDeviceContext dctx, cupmStream_t stream, const PetscScalar *xarr, const Vec yin[], PetscInt size, PetscScalar **device_y, PetscScalar *results, PetscInt *yidx))
{
  const auto   yidxt = *yidx;
  const auto   yint  = yin+yidxt;
  PetscScalar *host_y[N];
  cupmError_t  cerr;

  PetscFunctionBegin;
  for (auto i = 0; i < N; ++i) host_y[i] = DeviceArrayRead(dctx,yint[i]);
  cerr = cupmMemcpyAsync(device_y,host_y,N*sizeof(*device_y),cupmMemcpyHostToDevice,stream);CHKERRCUPM(cerr);
  cerr = cupmLaunchKernel(kernels::mdot_kernel<N>,dim3(MDOT_WORKGROUP_NUM),dim3(MDOT_WORKGROUP_SIZE),0,stream,xarr,device_y,size,results+(yidxt*MDOT_WORKGROUP_NUM));CHKERRCUPM(cerr);
  *yidx += N;
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<false>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto          n      = xin->map->n;
  const auto          nv1    = ((nv % 4) == 1) ? nv-1 : nv;
  const auto          nwork  = nv1*MDOT_WORKGROUP_NUM;
  const auto          nbytes = nwork*sizeof(PetscScalar);
  PetscBool           device_mem;
  PetscScalar         **d_y;
  PetscScalar         *d_results;
  PetscDeviceContext  dctx;
  cupmStream_t        stream;
  cupmError_t         cerr;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  // will hold all the device y pointers
  cerr = cupmMallocAsync(reinterpret_cast<void**>(&d_y),8*sizeof(*d_y),stream);CHKERRCUPM(cerr);
  // allocate scratchpad memory for the results of individual work groups
  cerr = cupmMallocAsync(reinterpret_cast<void**>(&d_results),nbytes,stream);CHKERRCUPM(cerr);
  {
    auto yidx = PetscInt{0};
    auto xptr = DeviceArrayRead(dctx,xin);

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    // REVIEW ME: Can fork-join here, but should probably only have a single-sized kernel then
    while (yidx < nv)
    {
      switch (nv-yidx) {
      case 7:
      case 6:
      case 5:
      case 4:
        ierr = mdot_kernel_dispatch_<4>(dctx,stream,xptr,yin,n,d_y,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 3:
        ierr = mdot_kernel_dispatch_<3>(dctx,stream,xptr,yin,n,d_y,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 2:
        ierr = mdot_kernel_dispatch_<2>(dctx,stream,xptr,yin,n,d_y,d_results,&yidx);CHKERRQ(ierr);
        break;
      case 1: {
        cupmBlasHandle_t handle;
        cupmBlasError_t  cberr;

        ierr  = PetscDeviceContextGetBLASHandle_Internal(dctx,&handle);CHKERRQ(ierr);
        ierr  = cupmBlasSetPointerModeFromPointer(handle,z);CHKERRQ(ierr);
        cberr = cupmBlasXdot(handle,n,DeviceArrayRead(dctx,yin[yidx]),1,xptr,1,cupmScalarCast(z+yidx));CHKERRCUPMBLAS(cberr);
        ++yidx;
      }
      case 0:
        break;
      default: // 8 or more
        ierr = mdot_kernel_dispatch_<8>(dctx,stream,xptr,yin,n,d_y,d_results,&yidx);CHKERRQ(ierr);
        break;
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  ierr = cupmIsDeviceMemory(z,&device_mem);CHKERRQ(ierr);
  // copy results to CPU
  if (device_mem) {
    // REVIEW ME: TODO
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device memory MDOT implementation pending");
  } else {
    auto       stackarray = std::array<PetscScalar,PETSC_MAX_PATH_LEN>{};
    const auto allocate   = static_cast<decltype(stackarray.size())>(nwork) > stackarray.size();
    auto       h_results  = stackarray.data();

    if (allocate) {ierr = PetscMalloc1(nwork,&h_results);CHKERRQ(ierr);}
    cerr = cupmMemcpyAsync(h_results,d_results,nbytes,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
    // do these now while memcpy is in flight
    ierr = PetscLogFlops(nwork);CHKERRQ(ierr);
    ierr = PetscLogGpuToCpuScalar(nbytes);CHKERRQ(ierr);
    // REVIEW ME: need to hard sync here...
    ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
    // REVIEW ME: it is likely faster to do this in a micro kernel rather than do it on the
    // host which requires synchronization and possibly an additional allocation
    // sum group results into z
    for (auto j = PetscInt{0}; j < nv1; ++j) {
      for (auto i = j*MDOT_WORKGROUP_NUM; i < (j+1)*MDOT_WORKGROUP_NUM; ++i) z[j] += h_results[i];
    }
    if (allocate) {ierr = PetscFree(h_results);CHKERRQ(ierr);}
  }
  // free these here, we will already have synchronized
  cerr = cupmFreeAsync(d_results,stream);CHKERRCUPM(cerr);
  cerr = cupmFreeAsync(d_y,stream);CHKERRCUPM(cerr);
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_NUM
#undef MDOT_WORKGROUP_SIZE

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async_(UseComplexTag<true>, Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto         n = static_cast<cupmBlasInt_t>(xin->map->n);
  PetscBool          device_mem;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = cupmIsDeviceMemory(z,&device_mem);CHKERRQ(ierr);
  {
    const auto         xptr = DeviceArrayRead(dctx,xin);
    const auto         mode = device_mem ? CUPMBLAS_POINTER_MODE_DEVICE : CUPMBLAS_POINTER_MODE_HOST;
    // probably not worth it to run more than 8 of these at a time?
    const auto         n_sub = PetscMin(nv,8);
    PetscDeviceContext *subctx;

    ierr = PetscDeviceContextFork(dctx,n_sub,&subctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    for (auto i = PetscInt{0}; i < nv; ++i) {
      cupmBlasHandle_t handle;

      ierr = PetscDeviceContextGetBLASHandle_Internal(subctx[i%n_sub],&handle);CHKERRQ(ierr);
      ierr = cupmBlasSetPointerMode(handle,mode);CHKERRQ(ierr);
      auto cberr = cupmBlasXdot(handle,n,DeviceArrayRead(dctx,yin[i]),1,xptr,1,cupmScalarCast(z+i));CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscDeviceContextJoin(dctx,n_sub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subctx);CHKERRQ(ierr);
  }
  // REVIEW ME: flops?????
  ierr = PetscLogGpuToCpuScalar(nv*sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::mdot_async(Vec xin, PetscInt nv, const Vec yin[], PetscScalar *z))
{
  const auto     n = xin->map->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheck(nv > 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to %s %" PetscInt_FMT " not positive",PETSC_FUNCTION_NAME,nv);
  else if (PetscUnlikely(nv == 1)) {
    ierr = dot_async(xin,PetscRemoveConstCast(yin[0]),z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  // z will always need to be zeroed first, either for a quick return or for summing later on
  ierr = PetscArrayzero(z,nv);CHKERRQ(ierr);
  // nothing to do if x has no entries
  if (n) {
    ierr = mdot_async_(UseComplexTag<PetscDefined(USE_COMPLEX)>{},xin,nv,yin,z);CHKERRQ(ierr);
    // REVIEW ME: double count of flops??
    ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha))
{
  const auto         n = xin->map->n;
  PetscErrorCode     ierr;
  cupmStream_t       stream;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  if (alpha == PetscScalar(0)) {
    auto cerr = cupmMemsetAsync(DeviceArrayWrite(dctx,xin).ptr,0,n*sizeof(PetscScalar),stream);CHKERRCUPM(cerr);
  } else {
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(DeviceArrayWrite(dctx,xin).ptr);

      THRUST_CALL(thrust::fill,stream,xptr,xptr+n,alpha);
    );
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
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
    const auto         calpha = makeCupmScalar(alpha);
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscErrorCode     ierr;

    ierr  = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    cberr = cupmBlasSetPointerMode(cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
    ierr  = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXscal(cupmBlasHandle,n,&calpha,DeviceArrayReadWrite(dctx,xin),1);CHKERRCUPMBLAS(cberr);
    ierr  = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr  = PetscLogCpuToGpuScalar(sizeof(calpha));CHKERRQ(ierr);
    ierr  = PetscLogGpuFlops(n);CHKERRQ(ierr);
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
  ierr = cupmBlasSetPointerModeFromPointer(cupmBlasHandle,z);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdotu(cupmBlasHandle,n,DeviceArrayRead(dctx,xin),1,DeviceArrayRead(dctx,yin),1,cupmScalarCast(z));CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*n-1,0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    const auto         nbytes = n*sizeof(*VecIMPLCast(xin)->array);
    auto               yiscupm = PETSC_TRUE,xondevice = PETSC_TRUE; // assume we start on device
    // silence buggy gcc warning: ‘mode’ may be used uninitialized in this function
    cupmMemcpyKind_t   mode = cupmMemcpyDeviceToDevice;
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
      ierr = PetscArraycpy(yarray,HostArrayRead(dctx,xin).ptr,n);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
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
    cberr = cupmBlasXswap(cupmBlasHandle,n,DeviceArrayReadWrite(dctx,xin),1,DeviceArrayReadWrite(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
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
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    {
      cupmBlasError_t cberr;
      const auto      calpha = makeCupmScalar(alpha);
      const auto      xarray = DeviceArrayRead(dctx,xin);

      cberr = cupmBlasSetPointerMode(cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
      ierr  = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      if (betaIsZero) {
        // here we can get away with purely write-only as we memcpy into it first
        const auto   yarray = DeviceArrayWrite(dctx,yin);
        const auto   nbytes = n*sizeof(PetscScalar);
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
        cerr = cupmMemcpyAsync(yarray.ptr,xarray.ptr,nbytes,cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
        cberr = cupmBlasXscal(cupmBlasHandle,n,&calpha,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        const auto cbeta  = makeCupmScalar(beta);
        const auto yarray = DeviceArrayReadWrite(dctx,yin);

        cberr = cupmBlasXscal(cupmBlasHandle,n,&cbeta,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,n,&calpha,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    }
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
  PetscBool          device_mem;
  PetscInt           flopCount = 0;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = cupmIsDeviceMemory(z,&device_mem);CHKERRQ(ierr);
  if (!n) {
    const auto norm1and2 = type == NORM_1_AND_2;

    if (device_mem) {
      cupmStream_t stream;

      ierr = GetHandles_(&stream);CHKERRQ(ierr);
      auto cerr = cupmMemsetAsync(z,0,(1+norm1and2)*sizeof(*z),stream);CHKERRCUPM(cerr);
    } else {
      z[0] = 0.0;
      // yes this technically sets z[0] = 0 again half the time
      z[norm1and2] = 0.0;
    }
    PetscFunctionReturn(0);
  }
  ierr = GetHandles_(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  {
    cupmBlasError_t cberr;
    auto            xarray = DeviceArrayRead(dctx,xin);

    // in case NORM_INFINITY we may technically undo this, but cleaner this way
    cberr = cupmBlasSetPointerMode(cupmBlasHandle,device_mem ? CUPMBLAS_POINTER_MODE_DEVICE : CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      cberr = cupmBlasXasum(cupmBlasHandle,n,xarray,1,cupmRealCast(z));CHKERRCUPMBLAS(cberr);
      flopCount = PetscMax(n-1,0);
      if (type == NORM_1) break;
      ++z; // fall-through
    case NORM_2:
    case NORM_FROBENIUS:
      cberr = cupmBlasXnrm2(cupmBlasHandle,n,xarray,1,cupmRealCast(z));CHKERRCUPMBLAS(cberr);
      flopCount += PetscMax(2*n-1,0); // += in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmError_t   cerr;
      cupmStream_t  stream;
      PetscScalar   zs;
      cupmBlasInt_t i;

      PetscCheck(!device_mem,PETSC_COMM_SELF,PETSC_ERR_SUP,"Device memory norm pending implementation");
      // REVIEW ME: this needs to be redone by hand
      cberr = cupmBlasSetPointerMode(cupmBlasHandle,CUPMBLAS_POINTER_MODE_HOST);CHKERRCUPMBLAS(cberr);
      cberr = cupmBlasXamax(cupmBlasHandle,n,xarray,1,&i);CHKERRCUPMBLAS(cberr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(&zs,xarray.ptr+i-1,sizeof(zs),cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
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
  PETSC_HOSTDEVICE_DECL PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
};

} // namespace detail

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec xin))
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    auto ierr = pointwiseunary_async_(detail::conjugate{},xin);CHKERRQ(ierr);
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

  PETSC_HOSTDEVICE_DECL PetscReal operator()(PetscScalar x) const { return PetscRealPart(x); }
};

template <typename Operator>
struct tuple_compare
{
  using tuple_type = thrust::tuple<PetscReal,PetscInt>;

  PETSC_HOSTDEVICE_DECL tuple_type operator()(const tuple_type& x, const tuple_type& y) const
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::minmax_async_(TupleFuncT&& tuple_ftr, UnaryFuncT&& unary_ftr, PetscReal init_val, Vec v, PetscInt *p, PetscReal *m))
{
  constexpr auto     init_ptr = PetscInt{-1};
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (!n) {
    *m = init_val;
    if (p) *p = init_ptr;
    PetscFunctionReturn(0);
  }
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  // REVIEW ME: why not cupmBlasIXamin()/cupmBlasIXamax()?
  CHKERRTHRUST(
    auto vptr = thrust::device_pointer_cast<PetscScalar>(DeviceArrayRead(dctx,v));

    if (p) {
      auto tup = thrust::make_tuple(init_val,init_ptr);
      auto zip = thrust::make_zip_iterator(
        thrust::make_tuple(vptr,thrust::make_counting_iterator(PetscInt{0}))
      );

      // need to use preprocessor conditionals since otherwise thrust complains about not being
      // able to convert a thrust::device_reference<PetscScalar> to a PetscReal on complex
      // builds...
#if PetscDefined(USE_COMPLEX)
      thrust::tie(*m,*p) = THRUST_CALL(thrust::transform_reduce,stream,zip,zip+n,detail::real_part(),tup,std::forward<TupleFuncT>(tuple_ftr));
#else
      thrust::tie(*m,*p) = THRUST_CALL(thrust::reduce,stream,zip,zip+n,tup,std::forward<TupleFuncT>(tuple_ftr));
#endif
    } else {
#if PetscDefined(USE_COMPLEX)
      *m = THRUST_CALL(thrust::transform_reduce,stream,vptr,vptr+n,detail::real_part(),init_val,std::forward<UnaryFuncT>(unary_ftr));
#else
      *m = THRUST_CALL(thrust::reduce,stream,vptr,vptr+n,init_val,std::forward<UnaryFuncT>(unary_ftr));
#endif
    }
  );
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscInt *p, PetscReal *m))
{
  using value_type = util::remove_pointer_t<decltype(m)>;
  using tuple_ftr  = detail::tuple_compare<thrust::greater<value_type>>;
  using unary_ftr  = thrust::maximum<value_type>;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  ierr = minmax_async_(tuple_ftr{},unary_ftr{},PETSC_MIN_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscInt *p, PetscReal *m))
{
  using value_type    = util::remove_pointer_t<decltype(m)>;
  using tuple_functor = detail::tuple_compare<thrust::less<value_type>>;
  using unary_functor = thrust::minimum<value_type>;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // use {} constructor syntax otherwise most vexing parse
  ierr = minmax_async_(tuple_functor{},unary_functor{},PETSC_MAX_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscScalar *sum))
{
  const auto         n = v->map->n;
  PetscDeviceContext dctx;
  cupmStream_t       stream;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = GetHandles_(&dctx,&stream);CHKERRQ(ierr);
  // REVIEW ME: why not cupmBlasXasum()?
  CHKERRTHRUST(
    auto dptr = thrust::device_pointer_cast(DeviceArrayRead(dctx,v).ptr);

    *sum = THRUST_CALL(thrust::reduce,stream,dptr,dptr+n,PetscScalar{0.0});
  );
  // REVIEW ME: must be at least n additions
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

namespace detail
{

struct shifter
{
  const PetscScalar s;

  PETSC_HOSTDEVICE_DECL PetscScalar operator()(PetscScalar x) const { return x+s; }
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
