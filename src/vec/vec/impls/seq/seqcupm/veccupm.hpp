#ifndef PETSCVECSEQCUPM_HPP
#define PETSCVECSEQCUPM_HPP

#define PETSC_SKIP_SPINLOCK // REVIEW ME: why

#include <petsc/private/vecimpl.h>         /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq
#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupmblasinterface.hpp>

#if !defined(__cplusplus) || !PetscDefined(HAVE_CXX_DIALECT_CXX11)
#  error "VecSeq_CUPM requires C++11"
#endif

#if PetscDefined(HAVE_CUDA) || PetscDefined(HAV_HIP)
#  define PETSC_HAVE_CUPM 1
#endif

#if PetscDefined(HAVE_CUPM)
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#endif

// TODO
// - refactor the AXPY's for code reuse
// - remove designated initializers
// - figure out how to template which thrust namespace to use so we can do
//   thrust::<backend>::par.on(stream)
// - get rid of these undefs, they are for error checking purposes only
#undef CHKERRCUDA
#undef CHKERRCUBLAS
// - fix cublasXdot alias wrapper since sometimes it wraps to cublasXdotc
// - maybe reintroduce PetscDeviceMalloc()?
// - There is also an overloaded version of cudaMallocAsync that takes the same arguments as
//   cudaMallocFromPoolAsync
// - Make createseqcupm_async callable from C
// - port bindtocpu

namespace Petsc
{

namespace Impl
{

#define CHKERRCXXCTOR(expr) CHKERRABORT(PETSC_COMM_SELF,expr)

#if PetscDefined(HAVE_CUPM)
#  define PETSC_HOST_DECL       __host__
#  define PETSC_DEVICE_DECL     __device__ __forceinline__
#else
#  define PETSC_HOST_DECL
#  define PETSC_DEVICE_DECL
#endif

#define PETSC_HOSTDEVICE_DECL PETSC_HOST_DECL PETSC_DEVICE_DECL

// VEC<PAR_TYPE>CUPM() gives the full expansion of the macro for each variant, (for cuda this
// would be VECSEQCUDA, i.e. "seqcuda") while VEC<PAR_TYPE>CUPMMACRO() gives the
// stringification of the macro name, i.e. for cuda "VECSEQCUDA"
#define PETSC_VECCUPM_DEFINE_UTILITY(PAR_TYPE,CUPM_TYPE)                \
  PETSC_CXX_COMPAT_DECL(constexpr const char* VEC ## PAR_TYPE ## CUPM()) \
  {                                                                     \
    return PetscStringize(VEC ## PAR_TYPE ## CUPM_TYPE);                \
  };                                                                    \
  PETSC_CXX_COMPAT_DECL(constexpr const char* VEC ## PAR_TYPE ## CUPMMACRO()) \
  {                                                                     \
    return PetscStringize_(VEC ## PAR_TYPE ## CUPM_TYPE);               \
  };

#define PETSC_VECCUPM_DEFINE_BASE_CLASS(TYPE)                           \
  template <>                                                           \
  struct VecSeq_CUPMBase<CUPMDeviceType::TYPE>                          \
  {                                                                     \
    static constexpr auto& VEC_CUPMCopyToGPU   = VEC_ ## TYPE ## CopyToGPU; \
    static constexpr auto& VEC_CUPMCopyFromGPU = VEC_ ## TYPE ## CopyFromGPU; \
                                                                        \
    PETSC_VECCUPM_DEFINE_UTILITY(SEQ,TYPE)                              \
    PETSC_VECCUPM_DEFINE_UTILITY(MPI,TYPE)                              \
  }

template <CUPMDeviceType T> struct VecSeq_CUPMBase;

PETSC_VECCUPM_DEFINE_BASE_CLASS(CUDA);
PETSC_VECCUPM_DEFINE_BASE_CLASS(HIP);

template <CUPMDeviceType T>
struct VecSeq_CUPM : VecSeq_CUPMBase<T>,CUPMBlasInterface<T>
{
public:
  PETSC_CUPMBLAS_INHERIT_INTERFACE_TYPEDEFS_USING(cupmBlasInterface_t,T);

  using base_type = VecSeq_CUPMBase<T>;
  using base_type::VEC_CUPMCopyToGPU;
  using base_type::VEC_CUPMCopyFromGPU;
  using base_type::VECSEQCUPM;
  using base_type::VECSEQCUPMMACRO;
  using base_type::VECMPICUPM;
  using base_type::VECMPICUPMMACRO;

  struct Vec_CUPM
  {
    PetscScalar      *device_array;     // gpu data
    PetscCopyMode     ptr_ownership;    // does PETSc own the array ptr?
    PetscBool         nvshmem;          // is array allocated in nvshmem? It is used to
                                        // allocate Mvctx->lvec in nvshmem
  };

private:
  // casting
  PETSC_CXX_COMPAT_DECL(constexpr Vec_Seq* __vec_impls_cast(Vec v))
  {
    return static_cast<Vec_Seq*>(v->data);
  }

  PETSC_CXX_COMPAT_DECL(constexpr Vec_CUPM* __cupm_impls_cast(Vec v))
  {
    return static_cast<Vec_CUPM*>(v->spptr);
  }

  // retrieving the various handles
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __get_handle_dispatch(PetscDeviceContext*,cupmBlasHandle_t*,cupmStream_t*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __get_handles(PetscDeviceContext*,cupmBlasHandle_t* = nullptr,cupmStream_t* = nullptr));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __get_handles(cupmStream_t*));

  // the dispatcher for min and max
  template <typename Tt, typename Tu>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __minmax_async(Tt&&,Tu&&,PetscReal,Vec,PetscInt*,PetscReal*));
  template <typename UnaryT>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __pointwiseunary_async(UnaryT&&,Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __create_async(Vec,PetscScalar* = nullptr));

  // data movement
  PETSC_CXX_COMPAT_DECL(PetscErrorCode __host_allocate_check(PetscDeviceContext PETSC_UNUSED dctx, Vec v))
  {
    PetscErrorCode ierr;
    auto           vseq = __vec_impls_cast(v);

    PetscFunctionBegin;
    if (!vseq) {
      ierr = PetscNewLog(PetscObjectCast(v),&vseq);CHKERRQ(ierr);
      v->data = vseq;
    }
    if (!vseq->array) {
      const auto n      = v->map->n;
      const auto nbytes = n*sizeof(*vseq->array_allocated);
      const auto useit  = use_cupm_host_alloc(nbytes > v->minimum_bytes_pinned_memory);

      if (useit.value()) v->pinned_memory = PETSC_TRUE;
      ierr = PetscMalloc1(n,&vseq->array_allocated);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory(PetscObjectCast(v),nbytes);CHKERRQ(ierr);
      vseq->array = vseq->array_allocated;
      if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) v->offloadmask = PETSC_OFFLOAD_CPU;
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode __device_allocate_check(PetscDeviceContext dctx, Vec v, PetscScalar *device_array = nullptr))
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (v->spptr) PetscFunctionReturn(0);
    ierr = PetscCalloc(sizeof(*__cupm_impls_cast(v)),&v->spptr);CHKERRQ(ierr);
    {
      PetscBool flg;
      auto      mem = static_cast<PetscInt>(v->minimum_bytes_pinned_memory);

      // Need to parse command line for minimum size to use for pinned memory allocations on
      // host here
      ierr = PetscObjectOptionsBegin(PetscObjectCast(v));CHKERRQ(ierr);
      ierr = PetscOptionsRangeInt("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",mem,&mem,&flg,0,std::numeric_limits<decltype(mem)>::max());CHKERRQ(ierr);
      if (flg) v->minimum_bytes_pinned_memory = mem;
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    if (device_array) {
      auto vcu = __cupm_impls_cast(v);

      vcu->device_array  = device_array;
      vcu->ptr_ownership = PETSC_USE_POINTER;
      v->offloadmask     = PETSC_OFFLOAD_GPU;
    } else {
      auto         vcu = __cupm_impls_cast(v);
      const auto   n   = v->map->n*sizeof(*vcu->device_array);
      PetscBLASInt bn;
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMallocAsync(static_cast<void**>(&vcu->device_array),bn,stream);CHKERRCUPM(cerr);
      vcu->ptr_ownership = PETSC_OWN_POINTER;
      if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
        if (v->data && __vec_impls_cast(v)->array) {
          v->offloadmask = PETSC_OFFLOAD_CPU;
        } else {
          v->offloadmask = PETSC_OFFLOAD_GPU;
        }
      }
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode __copy_to_device(PetscDeviceContext dctx, Vec v))
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __device_allocate_check(dctx,v);CHKERRQ(ierr);
    if (v->offloadmask == PETSC_OFFLOAD_CPU) {
      const auto   xfersize = v->map->n*sizeof(*__vec_impls_cast(v)->array);
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(VEC_CUPMCopyToGPU,v,0,0,0);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(__cupm_impls_cast(v)->device_array,__vec_impls_cast(v)->array,xfersize,cupmMemcpyHostToDevice,stream);CHKERRCUPM(cerr);
      ierr = PetscLogEventEnd(VEC_CUPMCopyToGPU,v,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogCpuToGpu(xfersize);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  PETSC_CXX_COMPAT_DECL(PetscErrorCode __copy_to_host(PetscDeviceContext dctx, Vec v))
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = __host_allocate_check(dctx,v);CHKERRQ(ierr);
    if (v->offloadmask == PETSC_OFFLOAD_GPU) {
      const auto   xfersize = v->map->n*sizeof(*__cupm_impls_cast(v)->device_array);
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(VEC_CUPMCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(__vec_impls_cast(v)->array,__cupm_impls_cast(v)->device_array,xfersize,cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      ierr = PetscLogEventEnd(VEC_CUPMCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogGpuToCpu(xfersize);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
    PetscFunctionReturn(0);
  }

  struct use_cupm_host_alloc
  {
    // in order to use 'auto' the member needs to be static; in order to be static, it must
    // also be constexpr, which in turn requires an initializer (also implpicitly required by
    // auto). But constexpr obviously needs a constant expression initializer, so we can't
    // initialize it with global (mutable) variables...
#define DECLTYPE_AUTO(left,right) decltype(right) left = right
    const DECLTYPE_AUTO(oldmalloc,PetscTrMalloc);
    const DECLTYPE_AUTO(oldfree,PetscTrFree);
    const DECLTYPE_AUTO(oldrealloc,PetscTrRealloc);
#undef DECLTYPE_AUTO
    const bool v;

    constexpr use_cupm_host_alloc(bool useit) noexcept : v(useit)
    {
      if (useit) {
        PetscTrMalloc  = [](size_t sz,PetscBool,int,const char*,const char*,void **ptr)
        {
          PetscFunctionBegin;
          {auto cerr = cupmMallocHost(ptr,sz);CHKERRCUPM(cerr);}
          PetscFunctionReturn(0);
        };
        PetscTrFree    = [](void *ptr,int,const char*,const char*)
        {
          PetscFunctionBegin;
          {auto cerr = cupmFreeHost(ptr);CHKERRCUPM(cerr);}
          PetscFunctionReturn(0);
        };
        PetscTrRealloc = [](size_t,int,const char*,const char*,void**)
        {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"CUDA has no realloc()");
        };
      }
    }

    constexpr const bool& value() const noexcept { return this->v; }

    ~use_cupm_host_alloc() noexcept
    {
      if (this->v) {
        PetscTrMalloc  = this->oldmalloc;
        PetscTrFree    = this->oldfree;
        PetscTrRealloc = this->oldrealloc;
      }
    }
  };

  template <bool read>
  struct vector_array
  {
    using pointer_type = util::conditional_t<read,const PetscScalar*,PetscScalar*>;

    const pointer_type ptr;

    operator pointer_type() const noexcept { return this->ptr; }

  protected:
    constexpr vector_array(pointer_type pointer) noexcept : ptr(pointer) { }
  };

  // RAII VecCUPMGetArrayRead()
  struct device_array_read  : vector_array<true>
  {
    using base_type = vector_array<true>;

    constexpr device_array_read(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__initialize_device_array(dctx,vector))
    { }

  private:
    PETSC_CXX_COMPAT_DECL(PetscScalar* __initialize_device_array(PetscDeviceContext dctx, Vec vector))
    {
      CHKERRCXXCTOR(__copy_to_device(dctx,vector));
      return __cupm_impls_cast(vector)->device_array;
    }
  };

  // RAII VecCUPMGetArray()
  struct device_array_write : vector_array<false>
  {
    using base_type = vector_array<false>;

    constexpr device_array_write(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__initialize_device_array(dctx,vector)), _v(vector)
    { }

    ~device_array_write() noexcept
    {
      // 1. can't actually do anything about the error since we may already be seterrq-ing out
      // 2. no clue what happens if two different error codes are in flight simultaneously
      auto PETSC_UNUSED ierr = PetscObjectStateIncrease(PetscObjectCast(_v));
      _v->offloadmask = PETSC_OFFLOAD_GPU;
    }

  private:
    const Vec _v;

    PETSC_CXX_COMPAT_DECL(PetscScalar* __initialize_device_array(PetscDeviceContext dctx, Vec vector))
    {
      CHKERRCXXCTOR(__device_allocate_check(dctx,vector));
      return __cupm_impls_cast(vector)->device_array;
    }
  };

  struct host_array_read    : vector_array<true>
  {
    using base_type = vector_array<true>;

    constexpr host_array_read(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__initialize_host_array(dctx,vector))
    { }

  private:
    PETSC_CXX_COMPAT_DECL(PetscScalar* __initialize_host_array(PetscDeviceContext dctx, Vec vector))
    {
      CHKERRCXXCTOR(__copy_to_host(dctx,vector));
      return __vec_impls_cast(vector)->array;
    }
  };

  struct host_array_write   : vector_array<false>
  {
    using base_type = vector_array<false>;

    constexpr host_array_write(PetscDeviceContext dctx, Vec vector) noexcept
      : base_type(__initialize_host_array(dctx,vector)), _v(vector)
    { }

    ~host_array_write() noexcept
    {
      // 1. can't actually do anything about the error since we may already be seterrq-ing out
      // 2. no clue what happens if two different error codes are in flight simultaneously
      auto PETSC_UNUSED ierr = PetscObjectStateIncrease(PetscObjectCast(_v));
      _v->offloadmask = PETSC_OFFLOAD_CPU;
    }

  private:
    const Vec _v;

    PETSC_CXX_COMPAT_DECL(PetscScalar* __initialize_host_array(PetscDeviceContext dctx, Vec vector))
    {
      CHKERRCXXCTOR(__host_allocate_check(dctx,vector));
      return __vec_impls_cast(vector)->array;
    }
  };

public:
  const struct _VecOps ops = {
    .create                 = create_async,
    .createwitharray        = createwitharray_async,
    .createwitharrays       = createwithbotharrays_async,
    .duplicate              = duplicate_async,
    .resetarray             = resetarray_async,
    .placearray             = placearray_async,
    .replacearray           = replacearray_async,
    .getarray               = getarray_async,
    .restorearray           = restorearray_async,
    .getarraywrite          = getarraywrite_async,
    .getarrayandmemtype     = getarrayandmemtype_async,
    .restorearrayandmemtype = restorearrayandmemtype_async,
    .axpy                   = axpy_async,
    .aypx                   = aypx_async,
    .pointwisedivide        = pointwisedivide_async,
    .pointwisemult          = pointwisemult_async,
    .reciprocal             = reciprocal_async,
    .waxpy                  = waxpy_async,
    .maxpy                  = maxpy_async,
    .dot                    = dot_async,
    .set                    = set_async,
    .scale                  = scale_async,
    .tdot                   = tdot_async,
    .copy                   = copy_async,
    .swap                   = swap_async,
    .axpby                  = axpby_async,
    .axpbypcz               = axpbypcz_async,
    .norm                   = norm_async,
    .dotnorm2               = dotnorm2_async,
    .destroy                = destroy_async,
    .conjugate              = conjugate_async,
    .getlocalvector         = getlocalvector_async<false>,
    .getlocalvectorread     = getlocalvector_async<true>,
    .restorelocalvector     = restorelocalvector_async<false>,
    .restorelocalvectorread = restorelocalvector_async<true>,
    .max                    = max_async,
    .min                    = min_async,
    .sum                    = sum_async,
    .shift                  = shift_async,
    .setrandom              = setrandom_async
  };

  // needs C binding
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createseqcupm_async(MPI_Comm,PetscInt,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createwitharray_async(MPI_Comm,PetscInt,PetscInt,const PetscScalar*,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode createwithbotharrays_async(MPI_Comm,PetscInt,PetscInt,const PetscScalar*,const PetscScalar*,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode replacearray_async(Vec,const PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarray_async(Vec,PetscScalar**));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearray_async(Vec,PetscScalar**));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarraywrite_async(Vec,PetscScalar**));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getarrayandmemtype_async(Vec,PetscScalar**,PetscMemType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorearrayandmemtype_async(Vec,PetscScalar**,PetscMemType*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode aypx_async(Vec,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode axpy_async(Vec,PetscScalar,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisedivide_async(Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode pointwisemult_async(Vec,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode reciprocal_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode waxpy_async(Vec,PetscScalar,Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode maxpy_async(Vec,PetscInt,const PetscScalar*,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscScalar*));
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
  template <bool read>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode getlocalvector_async(Vec,Vec));
  template <bool read>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode restorelocalvector_async(Vec,Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode sum_async(Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode shift_async(Vec,PetscScalar));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setrandom_async(Vec,PetscRandom));
};

// ================================================================================== //
//                                                                                    //
//                                  private methods                                   //
//                                                                                    //
// ================================================================================== //

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::__get_handle_dispatch(PetscDeviceContext *ctx, cupmBlasHandle_t *handle, cupmStream_t *stream))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextGetCurrentContextAssertType_Internal(&dctx,cupmDeviceTypeToPetscDeviceType());CHKERRQ(ierr);
  if (handle) {ierr = PetscDeviceContextGetBLASHandle_Internal(dctx,handle);CHKERRQ(ierr);}
  if (stream) {ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,stream);CHKERRQ(ierr);}
  if (ctx) *ctx = dctx;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::__get_handles(PetscDeviceContext *dctx, cupmBlasHandle_t *handle, cupmStream_t *stream))
{
  return __get_handle_dispatch(dctx,handle,stream);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::__get_handles(cupmStream_t *handle))
{
  return __get_handles(nullptr,nullptr,handle); // other overload
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::__create_async(Vec v, PetscScalar *device_array))
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)),&size);CHKERRMPI(ierr);
  if (PetscUnlikely(size > 1)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create %s on more than one process",VECSEQCUPMMACRO());
  ierr = PetscDeviceInitialize(cupmDeviceTypeToPetscDeviceType());CHKERRQ(ierr);
  ierr = VecCreate_Seq_Private(v,nullptr);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName(PetscObjectCast(v),VECSEQCUPM());CHKERRQ(ierr);
  ierr = VecBindToCPU_SeqCUDA(v,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->bindtocpu = VecBindToCPU_SeqCUDA;

  // Later, functions check for the Vec_CUPM structure existence, so do not create it without an
  // array attached
  if (device_array) {
    PetscDeviceContext dctx;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = __device_allocate_check(dctx,v,device_array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename Tt, typename Tu>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::__minmax_async(Tt&& tuple_functor, Tu&& unary_functor, PetscReal initialValue, Vec v, PetscInt *p, PetscReal *m))
{
  PetscErrorCode     ierr;
  const auto         n = v->map->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  if (!n) {
    *m = initialValue;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto vptr = thrust::device_pointer_cast(device_array_read(dctx,v).ptr);

    if (p) {
      const auto make_init = [=](){ return thrust::make_tuple(initialValue,PetscInt(-1)); };
      auto       zibit     = thrust::make_zip_iterator(
        thrust::make_tuple(vptr,thrust::make_counting_iterator(PetscInt(0)))
      );

      if (PetscDefined(USE_COMPLEX)) {
        struct real_part
        {
          PETSC_HOSTDEVICE_DECL
          thrust::tuple<PetscReal,PetscInt> operator()(const thrust::tuple<PetscScalar,PetscInt>& x) const
          {
            return thrust::make_tuple(PetscRealPart(x.get<0>()),x.get<1>());
          }
        };

        thrust::tie(*m,*p) = thrust::transform_reduce(
          zibit,zibit+n,real_part(),make_init(),std::forward<Tt>(tuple_functor)
        );
      } else {
        thrust::tie(*m,*p) = thrust::reduce(
          zibit,zibit+n,make_init(),std::forward<Tt>(tuple_functor)
        );
      }
    } else {
      if (PetscDefined(USE_COMPLEX)) {
        struct real_part
        {
          PETSC_HOSTDEVICE_DECL
          constexpr PetscReal operator()(const PetscScalar& x) const { return PetscRealPart(x); }
        };

        *m = thrust::transform_reduce(
          vptr,vptr+n,real_part(),initialValue,std::forward<Tu>(unary_functor)
        );
      } else {
        *m = thrust::reduce(vptr,vptr+n,initialValue,std::forward<Tu>(unary_functor));
      }
    }
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: flops?
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <typename Tu>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::__pointwiseunary_async(Tu&& unary, Vec win, Vec xin, Vec yin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto xptr = thrust::device_pointer_cast(device_array_read(dctx,xin).ptr);
    auto yptr = thrust::device_pointer_cast(device_array_read(dctx,yin).ptr);
    auto wptr = thrust::device_pointer_cast(device_array_write(dctx,win).ptr);

    thrust::transform(xptr,xptr+n,yptr,wptr,std::forward<Tu>(unary));
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                                   constructors                                     //

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createseqcupm_async(MPI_Comm comm, PetscInt n, Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQCUPM());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::create_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(v->map);CHKERRQ(ierr);
  ierr = __create_async(v);CHKERRQ(ierr);
  ierr = set_async(v,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createwitharray_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar *device_array, Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v,5);
  ierr = createwithbotharrays_async(comm,bs,n,nullptr,device_array,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::createwithbotharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, const PetscScalar *host_array, const PetscScalar *device_array, Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n) PetscValidScalarPointer(host_array,4);
  PetscValidPointer(v,6);
  // set V's gpuarray to be gpuarray, do not allocate memory on host yet.
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,/* REVIEW ME: was previously n*/PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*v,bs);CHKERRQ(ierr);
  // REVIEW ME: why no PetscLayoutSetUp()????
  ierr = __create_async(*v,PetscRemoveConstCast(device_array));CHKERRQ(ierr);
  if (host_array) {
    __vec_impls_cast(*v)->array = PetscRemoveConstCast(host_array);
    (*v)->offloadmask = device_array ? PETSC_OFFLOAD_BOTH : PETSC_OFFLOAD_CPU;
  } else if (device_array) {
    (*v)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*v)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::duplicate_async(Vec v, Vec *y))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = createseqcupm_async(PetscObjectComm(PetscObjectCast(v)),v->map->n,y);CHKERRQ(ierr);
  ierr = PetscLayoutReference(v->map,&(*y)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(PetscObjectCast(v)->olist,&(PetscObjectCast(*y)->olist));CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(PetscObjectCast(v)->qlist,&(PetscObjectCast(*y)->qlist));CHKERRQ(ierr);
  (*y)->stash.ignorenegidx = v->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                    mutatators                                      //

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::resetarray_async(Vec v))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  // REVIEW ME:
  // this is wildly inefficient but must be done if we assume that the placed array must have
  // correct values
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = __copy_to_host(dctx,v);CHKERRQ(ierr);
  ierr = VecResetArray_Seq(v);CHKERRQ(ierr);
  v->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::placearray_async(Vec v, const PetscScalar *a))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = __copy_to_host(dctx,v);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(v,a);CHKERRQ(ierr);
  v->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::replacearray_async(Vec v, const PetscScalar *a))
{
  auto           vseq = __vec_impls_cast(v);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vseq->array != vseq->array_allocated) {
    PetscDeviceContext dctx;
    // make sure the users array has the latest values.
    // REVIEW ME: why? we're about to free it
    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = __copy_to_host(dctx,v);CHKERRQ(ierr);
  }
  if (vseq->array_allocated) {
    const auto x = use_cupm_host_alloc(v->pinned_memory);
    ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
  }
  vseq->array_allocated = vseq->array = PetscRemoveConstCast(a);
  v->pinned_memory = PETSC_FALSE; // REVIEW ME: we can determine this
  v->offloadmask   = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
template <bool read>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getlocalvector_async(Vec v, Vec w))
{
  PetscErrorCode ierr;
  PetscBool      wisseqcupm;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUPM(),VECMPICUPM());
  ierr = PetscObjectTypeCompare(PetscObjectCast(w),VECSEQCUPM(),&wisseqcupm);CHKERRQ(ierr);
  if (wisseqcupm) {
    if (auto vseq = __vec_impls_cast(w)) {
      if (vseq->array_allocated) {
        const auto useit = use_cupm_host_alloc(w->pinned_memory);

        ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
        if (useit.value()) w->pinned_memory = PETSC_FALSE;
      }
      vseq->array         = nullptr;
      vseq->unplacedarray = nullptr;
    }
    if (auto vcu = __cupm_impls_cast(w)) {
      if (vcu->device_array) {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = __get_handles(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
        vcu->device_array = nullptr;
      }
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
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
    if (read) {
      ierr = VecGetArrayRead(v,&__vec_impls_cast(w)->array);CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(v,&__vec_impls_cast(w)->array);CHKERRQ(ierr);
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcupm) {
      PetscDeviceContext dctx;

      ierr = __get_handles(&dctx);CHKERRQ(ierr);
      ierr = __device_allocate_check(dctx,w);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
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
    ierr = (read ? VecRestoreArrayRead : VecRestoreArray)(v,&__vec_impls_cast(w)->array);CHKERRQ(ierr);
    if (w->spptr && wisseqcupm) {
      cupmStream_t stream;
      cupmError_t  cerr;

      ierr = __get_handles(&stream);CHKERRQ(ierr);
      cerr = cupmFreeAsync(__cupm_impls_cast(w)->device_array,stream);CHKERRCUPM(cerr);
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getarray_async(Vec v, PetscScalar **a))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = __copy_to_host(dctx,v);CHKERRQ(ierr);
  *a   = *static_cast<decltype(a)>(v->data);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorearray_async(Vec v, PetscScalar**))
{
  PetscFunctionBegin;
  v->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getarraywrite_async(Vec v, PetscScalar **a))
{
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = __host_allocate_check(dctx,v);CHKERRQ(ierr);
  *a   = *static_cast<decltype(a)>(v->data);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::getarrayandmemtype_async(Vec v, PetscScalar **a, PetscMemType *mtype))
{
  PetscFunctionBegin;
  if (v->offloadmask & PETSC_OFFLOAD_GPU) {
    auto vcu = __cupm_impls_cast(v);
    // return device pointer when device has up-to-date data, such as when offloadmask is
    // PETSC_OFFLOAD_BOTH
    *a = vcu->device_array;
    // change the mask once GPU gets write access, don't wait until restore array
    v->offloadmask = PETSC_OFFLOAD_GPU;
    if (mtype) {
      // I could just as easily have done
      //
      // if PETSC_CONSTEXPR_17 (T == CUPMDeviceType::HIP) *mtype = PETSC_MEMTYPE_HIP;
      // else *mtype = __cupm_impls_cast(v)->nvshmem ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUDA;
      //
      // but that would be very brittle to additions to CUPMDeviceType, as it would still
      // "work" silently
      if (vcu->nvshmem) *mtype = PETSC_MEMTYPE_NVSHMEM;
      else *mtype = cupmDeviceTypeToPetscMemType();
    }
  } else {
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = __host_allocate_check(dctx,v);CHKERRQ(ierr);
    *a   = *static_cast<decltype(a)>(v->data);
    if (mtype) *mtype = PETSC_MEMTYPE_HOST;
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::restorearrayandmemtype_async(Vec v, PetscScalar PETSC_UNUSED **a, PetscMemType PETSC_UNUSED *mtype))
{
  PetscFunctionBegin;
  if (v->offloadmask & PETSC_OFFLOAD_GPU) {
    v->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    v->offloadmask = PETSC_OFFLOAD_CPU;
  }
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::aypx_async(Vec yin, PetscScalar alpha, Vec xin))
{
  const auto         n = yin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    cupmError_t  cerr;
    cupmStream_t stream;

    ierr = __get_handles(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n*sizeof(PetscScalar),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  } else {
    const auto       alphaIsOne = alpha == PetscScalar(1.0);
    cupmBlasError_t  cberr;
    cupmBlasHandle_t cupmBlasHandle;
    PetscBLASInt     bn;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto yarray = device_array_write(dctx,yin);
      auto xarray = device_array_read(dctx,xin);

      if (alphaIsOne) {
        cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {
        constexpr PetscScalar sone = 1.0;

        cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&sone,xarray,1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((alphaIsOne ? 1 : 2)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::axpy_async(Vec yin, PetscScalar alpha, Vec xin))
{
  PetscErrorCode ierr;
  PetscBool      xiscupm;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompareAny(PetscObjectCast(xin),&xiscupm,VECSEQCUPM(),VECMPICUPM(),"");CHKERRQ(ierr);
  if (xiscupm) {
    PetscBLASInt       bn;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,device_array_write(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*bn);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  } else {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisedivide_async(Vec win, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseDivide_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    ierr = __pointwiseunary_async(thrust::divides<PetscScalar>(),win,xin,yin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::pointwisemult_async(Vec win, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    ierr = VecPointwiseMult_Seq(win,xin,yin);CHKERRQ(ierr);
  } else {
    ierr = __pointwiseunary_async(thrust::multiplies<PetscScalar>(),win,xin,yin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::reciprocal_async(Vec xin))
{
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    struct reciprocal
    {
      PETSC_HOSTDEVICE_DECL constexpr PetscScalar operator()(const PetscScalar& s) const
      {
        return s ? PetscScalar(1.0)/s : 0;
      }
    };

    auto xptr = thrust::device_pointer_cast(device_array_write(dctx,xin).ptr);

    thrust::transform(xptr,xptr+n,xptr,reciprocal());
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::waxpy_async(Vec win, PetscScalar alpha, Vec xin, Vec yin))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = copy_async(yin,win);CHKERRQ(ierr);
  } else {
    PetscBLASInt       bn;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmStream_t       stream;

    ierr = PetscBLASIntCast(win->map->n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      cupmBlasError_t cberr;
      cupmError_t     cerr;
      auto            warray = device_array_write(dctx,win);

      cerr = cupmMemcpyAsync(warray.ptr,device_array_read(dctx,yin).ptr,bn*sizeof(*warray.ptr),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
      cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,warray,1);CHKERRCUPMBLAS(cberr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*bn);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::maxpy_async(Vec xin, PetscInt nv, const PetscScalar *alpha, Vec *y))
{
  PetscBLASInt       bn;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    auto xarray = device_array_write(dctx,xin);

    for (decltype(nv) j = 0; j < nv; ++j) {
      auto cberr = cupmBlasXaxpy(cupmBlasHandle,bn,alpha+j,device_array_read(dctx,y[j]),1,xarray,1);CHKERRCUPMBLAS(cberr);
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(nv*2*bn);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(nv*sizeof(*alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dot_async(Vec xin, Vec yin, PetscScalar *z))
{
  PetscBLASInt       bn;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  // arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the
  // second
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdot(cupmBlasHandle,bn,device_array_read(dctx,yin),1,device_array_read(dctx,xin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*(bn-1),0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::set_async(Vec xin, PetscScalar alpha))
{
  const auto         n = xin->map->n;
  PetscErrorCode     ierr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0)) {
    cupmStream_t stream;
    cupmError_t  cerr;

    ierr = __get_handles(&dctx,&stream);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cerr = cupmMemsetAsync(device_array_write(dctx,xin).ptr,0,n*sizeof(PetscScalar),stream);CHKERRCUPM(cerr);
  } else {
    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      auto xptr = thrust::device_pointer_cast(device_array_write(dctx,xin).ptr);

      thrust::fill(xptr,xptr+n,alpha);
    } catch (const std::exception& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::scale_async(Vec xin, PetscScalar alpha))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == PetscScalar(0.0)) {
    ierr = set_async(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != PetscScalar(1.0)) {
    PetscBLASInt       bn;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;

    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,device_array_write(dctx,xin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar(sizeof(alpha));CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(bn);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::tdot_async(Vec xin, Vec yin, PetscScalar *z))
{
  PetscBLASInt       bn;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   cupmBlasHandle;
  cupmBlasError_t    cberr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cberr = cupmBlasXdotu(cupmBlasHandle,bn,device_array_read(dctx,xin),1,device_array_read(dctx,yin),1,z);CHKERRCUPMBLAS(cberr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(PetscMax(2*bn-1,0));CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::copy_async(Vec xin, Vec yin))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    const auto         nbytes = n*sizeof(xin->array)
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

    ierr = __get_handles(&dctx,&stream);CHKERRQ(ierr);
    switch (mode) {
    case cupmMemcpyDeviceToDevice:
      // the best case
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,device_array_read(dctx,xin).ptr,n*sizeof(PetscScalar),mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyHostToDevice:
      // not terrible
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(device_array_write(dctx,yin).ptr,host_array_read(dctx,xin).ptr,n*sizeof(PetscScalar),mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      break;
    case cupmMemcpyDeviceToHost: {
      // not great
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(yarray,device_array_read(dctx,xin).ptr,n*sizeof(*yarray),mode,stream);CHKERRCUPM(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    case cupmMemcpyHostToHost:   {
      // the worst case
      PetscScalar *yarray;

      ierr = VecGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      ierr = PetscArraycpy(ya,host_array_read(dctx,xin),n);CHKERRQ(ierr);
      ierr = VecRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
    } break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"Unknown cupmMemcpyKind %d",mode);
    }
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::swap_async(Vec xin, Vec yin))
{
  PetscFunctionBegin;
  if (xin != yin) {
    const auto         n = xin->map->n;
    PetscDeviceContext dctx;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscBLASInt       bn;
    PetscErrorCode     ierr;

    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    cberr = cupmBlasXswap(cupmBlasHandle,bn,device_array_write(dctx,xin),1,device_array_write(dctx,yin),1);CHKERRCUPMBLAS(cberr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
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
    const auto         n          = yin->map->n;
    const auto         betaIsZero = beta == PetscScalar(0.0);
    PetscBLASInt       bn;
    cupmBlasHandle_t   cupmBlasHandle;
    cupmBlasError_t    cberr;
    PetscDeviceContext dctx;

    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    {
      auto yarray = device_array_write(dctx,yin);

      if (betaIsZero) {
        cupmStream_t stream;

        ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
        cerr = cupmMemcpyAsync(yarray,device_array_read(dctx,xin).ptr,n*sizeof(*yarray.ptr),cupmMemcpyDeviceToDevice,stream);CHKERRCUPM(cerr);
        cberr = cupmBlasXscal(cupmBlasHandle,bn,&alpha,yarray,1);CHKERRCUPMBLAS(cberr);
      } else {

        cberr = cupmBlasXscal(cupmBlasHandle,bn,&beta,yarray,1);CHKERRCUPMBLAS(cberr);
        cberr = cupmBlasXaxpy(cupmBlasHandle,bn,&alpha,device_array_read(dctx,xin),1,yarray,1);CHKERRCUPMBLAS(cberr);
      }
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops((betaIsZero ? 1 : 3)*n);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpuScalar((betaIsZero ? 1 : 2)*sizeof(alpha));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
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

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::norm_async(Vec xin, NormType type, PetscReal *z))
{
  const auto         n         = xin->map->n;
  PetscInt           flopCount = 0;
  PetscBLASInt       bn;
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
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  ierr = __get_handles(&dctx,&cupmBlasHandle);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    cupmBlasError_t cberr;
    auto            xarray = device_array_read(dctx,xin);

    switch (type) {
    case NORM_1_AND_2:
    case NORM_1:
      cberr = cupmBlasXasum(cupmBlasHandle,bn,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount = PetscMax(n-1,0);
      if (type == NORM_1_AND_2) ++z;
      else break;
    case NORM_2:
    case NORM_FROBENIUS:
      cberr = cupmBlasXnrm2(cupmBlasHandle,bn,xarray,1,z);CHKERRCUPMBLAS(cberr);
      flopCount += PetscMax(2*n-1,0); // +=  in case we've fallen through from NORM_1_AND_2
      break;
    case NORM_INFINITY: {
      cupmError_t  cerr;
      cupmStream_t stream;
      PetscScalar  zs;
      int          i;

      // REVIEW ME: this needs to be redone by hand
      cberr = cupmBlasIXamax(cupmBlasHandle,bn,xarray,1,&i);CHKERRCUPMBLAS(cberr);
      ierr = PetscDeviceContextGetStreamHandle_Internal(dctx,&stream);CHKERRQ(ierr);
      cerr = cupmMemcpyAsync(&zs,xarray.ptr+i-1,sizeof(zs),cupmMemcpyDeviceToHost,stream);CHKERRCUPM(cerr);
      *z   = PetscAbsScalar(zs);
      // flopCount = ???
    } break;
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flopCount);CHKERRQ(ierr);
  ierr = PetscLogGpuToCpuScalar(sizeof(*z));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::dotnorm2_async(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = dot_async(s,t,dp);CHKERRQ(ierr);
  ierr = dot_async(t,t,nm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::destroy_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (auto vcu  = __cupm_impls_cast(v)) {
    switch (vcu->ptr_ownership) {
    case PETSC_COPY_VALUES:
    case PETSC_OWN_POINTER:
      if (PetscDefined(HAVE_NVSHMEM) && vcu->nvshmem) {
        ierr = PetscNvshmemFree(vcu->device_array);CHKERRQ(ierr);
      } else {
        cupmStream_t stream;
        cupmError_t  cerr;

        ierr = __get_handles(&stream);CHKERRQ(ierr);
        cerr = cupmFreeAsync(vcu->device_array,stream);CHKERRCUPM(cerr);
      }
    case PETSC_USE_POINTER:
      break;
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogObjectState((PetscObject)v,"Length=%" PetscInt_FMT,v->map->n);CHKERRQ(ierr);
#endif
  if (auto vseq = __vec_impls_cast(v)) {
    if (vseq->array_allocated) {
      const auto useit = use_cupm_host_alloc(v->pinned_memory);

      ierr = PetscFree(vseq->array_allocated);CHKERRQ(ierr);
      if (useit.value()) v->pinned_memory = PETSC_FALSE;
    }
    ierr = PetscFree(v->data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::conjugate_async(Vec v))
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    const auto         n = v->map->n;
    PetscDeviceContext dctx;
    PetscErrorCode     ierr;

    ierr = __get_handles(&dctx);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      struct conjugate
      {
        PETSC_HOSTDEVICE_DECL
        constexpr PetscScalar operator()(PetscScalar x) const { return PetscConj(x); }
      };
      auto xptr = thrust::device_pointer_cast(device_array_write(dctx,v).ptr);

      thrust::transform(xptr,xptr+n,xptr,conjugate());
    } catch (const std::exception& ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    // REVIEW ME: also at least n?
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::max_async(Vec v, PetscInt *p, PetscReal *m))
{
  PetscErrorCode ierr;
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

  PetscFunctionBegin;
  ierr = __minmax_async(max_tuple(),thrust::maximum<decltype(*m)>(),PETSC_MIN_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::min_async(Vec v, PetscInt *p, PetscReal *m))
{
  PetscErrorCode ierr;
  struct min_tuple
  {
    using tuple_type = thrust::tuple<PetscReal,PetscInt>;

    PETSC_HOSTDEVICE_DECL tuple_type operator()(const tuple_type& x, const tuple_type& y) const
    {
      if ((x.get<0>() < y.get<0>()) || (x.get<1>() < y.get<1>())) {
        return thrust::make_tuple(x.get<0>(),x.get<1>());
      } else {
        return thrust::make_tuple(y.get<0>(),y.get<1>());
      }
    }
  };

  PetscFunctionBegin;
  ierr = __minmax_async(min_tuple(),thrust::minimum<decltype(*m)>(),PETSC_MAX_REAL,v,p,m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::sum_async(Vec v, PetscScalar *sum))
{
  PetscErrorCode     ierr;
  const auto         n = v->map->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    auto dptr = thrust::device_pointer_cast(device_array_read(dctx,v).ptr);

    *sum = thrust::reduce(dptr,dptr+n,PetscScalar(0.0));
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  // REVIEW ME: must be at least n additions
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::shift_async(Vec v, PetscScalar shift))
{
  PetscErrorCode     ierr;
  const auto         n = v->map->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    struct shifter
    {
      using value_type = decltype(shift);
      const value_type _s;

      constexpr shifter(value_type sft) noexcept : _s(sft) { }

      PETSC_HOSTDEVICE_DECL constexpr value_type operator()(value_type x) const { return x+_s; }
    };
    auto dptr = thrust::device_pointer_cast(device_array_write(dctx,v).ptr);

    thrust::transform(dptr,dptr+n,dptr,shifter(shift)); /* in-place transform */
  } catch (const std::exception& ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <CUPMDeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecSeq_CUPM<T>::setrandom_async(Vec v, PetscRandom rand))
{
  PetscErrorCode     ierr;
  const auto         n = xin->map->n;
  PetscDeviceContext dctx;
  PetscBool          iscurand;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare(PetscObjectCast(r),PETSCCURAND,&iscurand);CHKERRQ(ierr);
  ierr = __get_handles(&dctx);CHKERRQ(ierr);
  if (iscurand) {
    ierr = PetscRandomGetValues(r,n,device_array_write(dctx,v));CHKERRQ(ierr);
  } else {
    ierr = PetscRandomGetValues(r,n,host_array_write(dctx,v));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Petsc

#endif // PETSCVECSEQCUPM_HPP
