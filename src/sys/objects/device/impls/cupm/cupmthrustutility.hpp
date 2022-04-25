#ifndef PETSC_CUPM_THRUST_UTILITY_HPP
#define PETSC_CUPM_THRUST_UTILITY_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cupminterface.hpp>

#if defined(__cplusplus)
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace Petsc
{

namespace Device
{

namespace CUPM
{

namespace Impl
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
    const auto timer = ::Petsc::Device::CUPM::Impl::detail::PetscLogGpuTimer{}; \
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
  const T *const       s;
  const BinaryOperator op;

  PETSC_HOSTDEVICE_DECL auto operator()(T x) const PETSC_DECLTYPE_AUTO_RETURNS(op(std::move(x),*s));
};

template <typename T, typename BinaryOperator>
static inline auto make_shift_operator(T *s, BinaryOperator&& op)
PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  shift_operator<T,BinaryOperator>{s,std::forward<BinaryOperator>(op)}
);

// actual implementation that calls thrust, 2 argument version
template <DeviceType DT, typename FunctorType, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(typename Interface<DT>::cupmStream_t stream, FunctorType&& functor, PetscInt n, T *xinout, T *yin = nullptr))
{
  const auto xptr = thrust::device_pointer_cast(xinout);
  const auto retptr = (yin && (yin != xinout)) ? thrust::device_pointer_cast(yin) : xptr;

  PetscFunctionBegin;
  CHKERRTHRUST(THRUST_CALL(thrust::transform,stream,xptr,xptr+n,retptr,std::forward<FunctorType>(functor)));
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

// actual implementation that calls thrust, 3 argument version
template <DeviceType DT, typename FunctorType, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(typename Interface<DT>::cupmStream_t stream, FunctorType&& functor, PetscInt n, const T *xin, const T *yin, T *zin))
{
  const auto xptr = thrust::device_pointer_cast(xin);

  PetscFunctionBegin;
  PetscAssert((xin != yin) && (xin != zin) && (zin != yin),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Must have disjoint pointers when passing all three!");
  CHKERRTHRUST(
    THRUST_CALL(
      thrust::transform,stream,
      xptr,xptr+n,
      thrust::device_pointer_cast(yin),
      thrust::device_pointer_cast(zin),
      std::forward<FunctorType>(functor)
    )
  );
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

// serves as setup to the real implementation above
template <DeviceType T, typename... Args>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustApplyPointwise(PetscDeviceContext dctx, Args&&... rest))
{
  typename Interface<T>::cupmStream_t stream;

  PetscFunctionBegin;
  static_assert(sizeof...(Args) <= 5,"");
  PetscValidDeviceContext(dctx,1);
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&stream));
  PetscCall(ThrustApplyPointwise<T>(stream,std::forward<Args>(rest)...));
  PetscFunctionReturn(0);
}

#define PetscCallCUPM_(...) do {                                                               \
    using      interface          = Interface<DT>;                                             \
    using      cupmError_t        = typename interface::cupmError_t;                           \
    const auto cupmName           = [](             ){ return interface::cupmName          ( ); }; \
    const auto cupmGetErrorName   = [](cupmError_t e){ return interface::cupmGetErrorName  (e); }; \
    const auto cupmGetErrorString = [](cupmError_t e){ return interface::cupmGetErrorString(e); }; \
    const auto cupmSuccess = interface::cupmSuccess;                                           \
    PetscCallCUPM(__VA_ARGS__);                                                                \
  } while (0)

template <DeviceType DT, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustSet(typename Interface<DT>::cupmStream_t stream, PetscInt n, T *ptr, const T *val))
{
  const auto size = n*sizeof(T);

  PetscFunctionBegin;
  PetscValidPointer(val,4);
  if (*val == T{0}) {
    PetscCallCUPM_(Interface<DT>::cupmMemsetAsync(ptr,0,size,stream));
  } else {
    auto xptr = thrust::device_pointer_cast(ptr);

    CHKERRTHRUST(THRUST_CALL(thrust::fill,stream,xptr,xptr+n,*val));
    if (std::is_same<util::remove_cv_t<T>,PetscScalar>::value) {
      PetscCall(PetscLogCpuToGpuScalar(size));
    } else {
      PetscCall(PetscLogCpuToGpu(size));
    }
  }
  PetscFunctionReturn(0);
}

#undef PetscCallCUPM_

template <DeviceType DT, typename T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode ThrustSet(PetscDeviceContext dctx, PetscInt n, T *ptr, const T *val))
{
  typename Interface<DT>::cupmStream_t stream;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscCall(PetscDeviceContextGetStreamHandle_Internal(dctx,&stream));
  PetscCall(ThrustSet(stream,n,ptr,val));
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace CUPM

} // namespace Device

} // namespace Petsc

#endif // __cplusplus

#endif // PETSC_CUPM_THRUST_UTILITY_HPP
