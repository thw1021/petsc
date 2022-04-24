#ifndef PETSC_CUPM_THRUST_UTILITY_HPP
#define PETSC_CUPM_THRUST_UTILITY_HPP

#if defined(__cplusplus)

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

namespace
{

#if PetscDefined(USING_NVCC)
#  if !defined(THRUST_VERSION)
#    error "THRUST_VERSION not defined!"
#  endif
#  if !PetscDefined(USE_DEBUG) && (THRUST_VERSION >= 101600)
#    define thrust_call_par_on(func,s,...) func(thrust::cuda::par_nosync.on(s),__VA_ARGS__)
#  else
#    define thrust_call_par_on(func,s,...) func(thrust::cuda::par.on(s),__VA_ARGS__)
#  endif
#elif PetscDefined(USING_HCC) // rocThrust has no par_nosync
#  define thrust_call_par_on(func,s,...)   func(thrust::hip::par.on(s),__VA_ARGS__)
#else
#  define thrust_call_par_on(func,s,...)   func(__VA_ARGS__)
#endif

struct PetscLogGpuTime_
{
  PetscLogGpuTime_()  noexcept { PetscCallAbort(PETSC_COMM_SELF,PetscLogGpuTimeBegin()); }
  ~PetscLogGpuTime_() noexcept { PetscCallAbort(PETSC_COMM_SELF,PetscLogGpuTimeEnd());   }
};

#define THRUST_CALL(...) [&]{                   \
    const auto timer = PetscLogGpuTime_{};      \
    return thrust_call_par_on(__VA_ARGS__);     \
  }()


#define CHKERRTHRUST(...)  do {                                                 \
    try {                                                                       \
      __VA_ARGS__;                                                              \
    } catch (const thrust::system_error& ex) {                                  \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());      \
    }                                                                           \
  } while (0)

template <typename T, typename UnaryOperator>
struct shift_operator
{
  const T             *s;
  const UnaryOperator  op;

  PETSC_HOSTDEVICE_DECL
  auto operator()(T&& x) const PETSC_DECLTYPE_AUTO_RETURNS(op(std::forward<T>(x),*s));
};

template <typename T, typename BinaryOperator>
static inline auto make_shift_operator(T&& s, BinaryOperator&& op)
PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(shift_operator<T,BinaryOperator>{std::forward<T>(s),std::forward<BinaryOperator>(op)});

template <DeviceType DT, typename T, typename UnaryFuncT, typename StreamT = typename Impl::Interface<DT>::cupmStream_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwiseUnary(StreamT stream, UnaryFuncT&& unary, PetscInt n, T *xin, T *yin = nullptr))
{
  PetscFunctionBegin;
  if (xin == yin || !yin) { // in-place
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(xin);

      THRUST_CALL(thrust::transform,stream,xptr,xptr+n,xptr,std::forward<UnaryFuncT>(unary));
    );
  } else {
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(xin);
      auto yptr = thrust::device_pointer_cast(yin);

      THRUST_CALL(thrust::transform,stream,xptr,xptr+n,yptr,std::forward<UnaryFuncT>(unary));
    );
  }
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

template <DeviceType DT, typename T, typename BinaryFuncT, typename StreamT = typename Impl::Interface<DT>::cupmStream_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwiseBinary(StreamT stream, BinaryFuncT&& binary, PetscInt n, T *xin, T *yin, T *zin))
{
  PetscFunctionBegin;
  CHKERRTHRUST(
    auto xptr = thrust::device_pointer_cast(xin);
    auto yptr = thrust::device_pointer_cast(yin);
    auto zptr = thrust::device_pointer_cast(zin);

    THRUST_CALL(thrust::transform,stream,xptr,xptr+n,yptr,zptr,std::forward<BinaryFuncT>(binary));
  );
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

template <DeviceType DT, typename ...Args, typename FT = PetscErrorCode(*)(Args...)>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(PetscDeviceContext dctx, FT&& ThrustApplyFunction, Args&&... rest))
{
  typename Impl::Interface<DT>::cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&stream));
  PetscCall(ThrustApplyFunction(stream,std::forward<Args>(rest)...));
  PetscFunctionReturn(0);
}

template <DeviceType DT, typename T, typename UnaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwiseUnary(PetscDeviceContext dctx, UnaryFuncT&& unary, PetscInt n, T *xin, T *yin = nullptr))
{
  PetscFunctionBegin;
  PetscCall(ThrustApplyPointwise<DT>(dctx,ThrustApplyPointwiseUnary,std::forward<UnaryFuncT>(unary),xin,yin));
  PetscFunctionReturn(0);
}

template <DeviceType DT, typename T, typename BinaryFuncT>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwiseBinary(PetscDeviceContext dctx, BinaryFuncT&& binary, PetscInt n, T *xin, T *yin, T *zin))
{
  PetscFunctionBegin;
  PetscCall(ThrustApplyPointwise<DT>(dctx,ThrustApplyPointwiseBinary,std::forward<BinaryFuncT>(binary),xin,yin,zin));
  PetscFunctionReturn(0);
}

#define PetscCallCUPM_(...) do {                                                               \
    using      interface          = Impl::Interface<DT>;                                       \
    using      cupmError_t        = typename interface::cupmError_t;                           \
    const auto cupmName           = [](             ){ return interface::cupmName          ( ); }; \
    const auto cupmGetErrorName   = [](cupmError_t e){ return interface::cupmGetErrorName  (e); }; \
    const auto cupmGetErrorString = [](cupmError_t e){ return interface::cupmGetErrorString(e); }; \
    const auto cupmSuccess = interface::cupmSuccess;                                           \
    PetscCallCUPM(__VA_ARGS__);                                                                \
  } while (0)

template <DeviceType DT, typename T, typename StreamT = typename Impl::Interface<DT>::cupmStream_t>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustSet(StreamT stream, PetscInt n, T *ptr, const T *val))
{
  const auto size = n*sizeof(T);

  PetscFunctionBegin;
  if (*val == T{0}) {
    PetscCallCUPM_(Impl::Interface<DT>::cupmMemsetAsync(ptr,0,size,stream));
  } else {
    CHKERRTHRUST(
      auto xptr = thrust::device_pointer_cast(ptr);

      THRUST_CALL(thrust::fill,stream,xptr,xptr+n,*val);
    );
    PetscCall(PetscLogCpuToGpu(size));
  }
  PetscFunctionReturn(0);
}

#undef PetscCallCUPM_

template <DeviceType DT, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustSet(PetscDeviceContext dctx, PetscInt n, T *ptr, const T *val))
{
  typename Impl::Interface<DT>::cupmStream_t stream;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&stream));
  PetscCall(ThrustSet(stream,n,ptr,val));
  PetscFunctionReturn(0);
}

} // anonymous namespace

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CUPM_THRUST_UTILITY_HPP
