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

namespace detail
{

struct PetscLogGpuTimer
{
  PetscLogGpuTimer()  noexcept { PetscCallAbort(PETSC_COMM_SELF,PetscLogGpuTimeBegin()); }
  ~PetscLogGpuTimer() noexcept { PetscCallAbort(PETSC_COMM_SELF,PetscLogGpuTimeEnd());   }
};

} // namespace detail

#define THRUST_CALL(...) [&]{                                                   \
    const auto timer = ::Petsc::Device::CUPM::detail::PetscLogGpuTimer{};       \
    return thrust_call_par_on(__VA_ARGS__);                                     \
  }()

#define CHKERRTHRUST(...)  do {                                                 \
    try {                                                                       \
      __VA_ARGS__;                                                              \
    } catch (const thrust::system_error& ex) {                                  \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s",ex.what());      \
    }                                                                           \
  } while (0)

template <typename T, typename BinaryOperator>
struct shift_operator
{
  const T              *s;
  const BinaryOperator  op;

  PETSC_HOSTDEVICE_DECL
  auto operator()(T&& x) const PETSC_DECLTYPE_AUTO_RETURNS(op(std::forward<T>(x),*s));
};

template <typename T, typename BinaryOperator>
static inline auto make_shift_operator(T&& s, BinaryOperator&& op)
PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  shift_operator<T,BinaryOperator>{std::forward<T>(s),std::forward<BinaryOperator>(op)}
);

// actual implementation that calls thrust
template <
  DeviceType DT,
  typename T,
  typename FunctorType,
  typename StreamType = typename Impl::Interface<DT>::cupmStream_t
  >
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(StreamType stream, FunctorType&& functor, PetscInt n, T *xin, T *yin = nullptr, T *zin = nullptr))
{
  auto xptr = thrust::device_pointer_cast(xin);

  PetscFunctionBegin;
  if (yin && (yin != xin)) {
    auto yptr = thrust::device_pointer_cast(yin);

    if (zin) {
      PetscAssert((xin != zin) && (zin != yin),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Must have disjoint pointers when passing all three!");
      CHKERRTHRUST(
        THRUST_CALL(
          thrust::transform,
          stream,xptr,xptr+n,yptr,thrust::device_pointer_cast(zin),std::forward<FunctorType>(functor)
        )
      );
    } else {
      CHKERRTHRUST(THRUST_CALL(thrust::transform,stream,xptr,xptr+n,yptr,std::forward<FunctorType>(functor)));
    }
  } else {
    PetscAssert(!zin,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot pass zin and not yin");
    CHKERRTHRUST(THRUST_CALL(thrust::transform,stream,xptr,xptr+n,xptr,std::forward<FunctorType>(functor)));
  }
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

// serves as setup to the real implementation above
template <
  DeviceType DT,
  typename ...Args,
  typename StreamType   = typename Impl::Interface<DT>::cupmStream_t,
  typename FunctionType = PetscErrorCode(*)(StreamType,Args...)
  >
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(PetscDeviceContext dctx, FunctionType&& ThrustApplyFunction, Args&&... rest))
{
  StreamType stream;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&stream));
  PetscCall(ThrustApplyFunction(stream,std::forward<Args>(rest)...));
  PetscFunctionReturn(0);
}

template <DeviceType DT, typename T, typename UnaryFunctionType = T(T)>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwiseUnary(PetscDeviceContext dctx, UnaryFunctionType&& unary, PetscInt n, T *xin, T *yin = nullptr))
{
  PetscFunctionBegin;
  PetscCall(ThrustApplyPointwise<DT>(dctx,ThrustApplyPointwiseUnary,std::forward<UnaryFunctionType>(unary),xin,yin));
  PetscFunctionReturn(0);
}

template <DeviceType DT, typename T, typename BinaryFunctionType = T(T,T)>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwiseBinary(PetscDeviceContext dctx, BinaryFunctionType&& binary, PetscInt n, T *xin, T *yin, T *zin))
{
  PetscFunctionBegin;
  PetscCall(ThrustApplyPointwise<DT>(dctx,ThrustApplyPointwiseBinary,std::forward<BinaryFunctionType>(binary),xin,yin,zin));
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
    auto xptr = thrust::device_pointer_cast(ptr);

    CHKERRTHRUST(THRUST_CALL(thrust::fill,stream,xptr,xptr+n,*val));
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
