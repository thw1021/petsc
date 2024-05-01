#pragma once

#include <petsclog.h>         // PetscLogGpuTimeBegin()/End()
#include <petscerror.h>       // SETERRQ()
#include <petscdevice_cupm.h> // PETSC_USING_NVCC

#include <thrust/version.h>          // THRUST_VERSION
#include <thrust/system_error.h>     // thrust::system_error
#include <thrust/execution_policy.h> // thrust::cuda/hip::par

namespace Petsc
{

namespace device
{

namespace cupm
{
#if PetscDefined(USING_NVCC)
  #if !defined(THRUST_VERSION)
    #error "THRUST_VERSION not defined!"
  #endif
  #if THRUST_VERSION >= 101600
    #define PETSC_THRUST_HAS_ASYNC                 1
    #define THRUST_PAR_ON(S) thrust::cuda::par_nosync.on(S)
  #else
    #define THRUST_PAR_ON(S) thrust::cuda::par.on(S)
  #endif
#elif PetscDefined(USING_HCC) // rocThrust has no par_nosync
  #define THRUST_PAR_ON(S) thrust::hip::par.on(S)
#else
  #define THRUST_PAR_ON(S)
#endif

#ifndef PETSC_THRUST_HAS_ASYNC
  #define PETSC_THRUST_HAS_ASYNC 0
#endif

namespace detail
{

struct PetscLogGpuTimer {
  PetscLogGpuTimer() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeBegin());
    PetscFunctionReturnVoid();
  }

  ~PetscLogGpuTimer() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, PetscLogGpuTimeEnd());
    PetscFunctionReturnVoid();
  }
};

} // namespace detail

#define THRUST_CALL(F, S, ...) \
  [&] { \
    const auto timer = ::Petsc::device::cupm::detail::PetscLogGpuTimer{}; \
    return F(THRUST_PAR_ON(S), __VA_ARGS__); \
  }()

#define PetscCallThrust(...) \
  do { \
    try { \
      { \
        __VA_ARGS__; \
      } \
    } catch (const thrust::system_error &ex) { \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Thrust error: %s", ex.what()); \
    } \
  } while (0)

} // namespace cupm

} // namespace device

} // namespace Petsc
