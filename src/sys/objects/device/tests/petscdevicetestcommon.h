#ifndef PETSCDEVICETESTCOMMON_H
#define PETSCDEVICETESTCOMMON_H

/* this file needs to be the one to include petsc/private/deviceimpl.h since it needs to define
 * a special macro to ensure that the error checking macros stay defined even in optimized
 * builds
 */
#if defined(PETSCDEVICEIMPL_H)
#  error "must #include this file before petsc/private/deviceimpl.h"
#endif

#if !defined(PETSC_DEVICE_KEEP_ERROR_CHECKING_MACROS)
#  define PETSC_DEVICE_KEEP_ERROR_CHECKING_MACROS 1
#endif
#include <petsc/private/deviceimpl.h>

static inline PetscErrorCode AssertDeviceExists(PetscDevice device)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertDeviceDoesNotExist(PetscDevice device)
{
  PetscFunctionBegin;
  PetscCheck(!device,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDevice was not destroyed for type %s",PetscDeviceTypes[device->type]);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertDeviceContextExists(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertDeviceContextDoesNotExist(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCheck(!dctx,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext was not destroyed");
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscStreamTypesValidAndEqual(PetscStreamType left, PetscStreamType right, const char *errStr)
{
  PetscFunctionBegin;
  PetscValidStreamType(left,1);
  PetscValidStreamType(right,2);
  PetscCheck(left == right,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,errStr,PetscStreamTypes[left],PetscStreamTypes[right]);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscDevicesValidAndEqual(PetscDevice left, PetscDevice right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDevices(left,1,right,2);
  PetscCheck(left == right,PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"%s",errStr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode AssertPetscDeviceContextsValidAndEqual(PetscDeviceContext left, PetscDeviceContext right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDeviceContexts(left,1,right,2);
  PetscCheck(left == right,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"%s",errStr);
  PetscFunctionReturn(0);
}

#if defined(__cplusplus)
#include <petsc/private/cpputil.hpp>
#include <array>
#include <memory>
#include <functional>

template <
  typename PetscType_,
  typename PetscManagedType_,
  typename CreateT,
  typename CreateDefaultT,
  typename DestroyT,
  typename GetValuesT,
  typename GetSizeT
  >
struct ManagedTypeInterface
{
  using PetscType        = PetscType_;
  using PetscManagedType = PetscManagedType_;

  const CreateT        PetscManagedTypeCreate;
  const CreateDefaultT PetscManagedTypeCreateDefault;
  const DestroyT       PetscManagedTypeDestroy;
  const GetValuesT     PetscManagedTypeGetValues;
  const GetSizeT       PetscManagedTypeGetSize;

  constexpr ManagedTypeInterface(CreateT&& c, CreateDefaultT&& cd, DestroyT&& d, GetValuesT&& gv, GetSizeT&& gs)
    noexcept :
    PetscManagedTypeCreate(std::forward<CreateT>(c)),
    PetscManagedTypeCreateDefault(std::forward<CreateDefaultT>(cd)),
    PetscManagedTypeDestroy(std::forward<DestroyT>(d)),
    PetscManagedTypeGetValues(std::forward<GetValuesT>(gv)),
    PetscManagedTypeGetSize(std::forward<GetSizeT>(gs))
  { }

  template <typename T>
  PetscErrorCode TestCreateAndOpSingleton(PetscDeviceContext dctx, T&& TestOp, PetscInt nmax = 20) const noexcept
  {
    constexpr auto    masks      = std::array<PetscOffloadMask,2>{PETSC_OFFLOAD_CPU,PETSC_OFFLOAD_GPU};
    auto              host_value = PetscType{15};
    PetscType        *host_ptr   = nullptr,*device_ptr = nullptr;
    PetscManagedType  scal;
    auto              host_ptrs  = std::array<std::pair<PetscType*,PetscCopyMode>,3>{
      std::make_pair(&host_value,PETSC_COPY_VALUES),
      std::make_pair(host_ptr,   PETSC_OWN_POINTER),
      std::make_pair(&host_value,PETSC_USE_POINTER)
    };

    PetscFunctionBegin;
    // single size
    for (auto& host : host_ptrs) {
      const auto alloc = host.second == PETSC_OWN_POINTER;

      for (const auto mask : masks) {
        for (PetscInt k = 0; k < nmax; ++k) {
          constexpr auto size = 1;

          // need to keep reallocating the host pointer since we will pass over ownership
          if (alloc) PetscCall(PetscMalloc1(size,&host.first));
          PetscCall(PetscManagedTypeCreate(dctx,host.first,device_ptr,size,host.second,PETSC_OWN_POINTER,mask,&scal));
          if (k) PetscCall(TestOp(dctx,scal));
          PetscCall(PetscManagedTypeDestroy(dctx,&scal));
        }
      }
    }
    PetscFunctionReturn(0);
  }

  template <PetscInt n_scal = 20, typename T>
  PetscErrorCode TestCreateAndOpGroup(PetscDeviceContext dctx, T&& TestOp) const noexcept
  {
    static_assert(n_scal > 0,"");
    PetscManagedType scal_arr[n_scal];

    PetscFunctionBegin;
    static_assert(n_scal % 2 == 0,"");
    // destroy in original order
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,1,scal_arr+i));
      if (i) PetscCall(TestOp(dctx,scal_arr[i]));
    }
    for (PetscInt i = 0; i < n_scal; ++i) PetscCall(PetscManagedTypeDestroy(dctx,scal_arr+i));

    // destroy in reverse order
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,100*i,scal_arr+i));
      if (i) PetscCall(TestOp(dctx,scal_arr[i]));
    }
    for (PetscInt i = n_scal-1; i >= 0; --i) PetscCall(PetscManagedTypeDestroy(dctx,scal_arr+i));

    // destroy as we create
    for (PetscInt i = 0, j = 0; i < n_scal+(n_scal/2); ++i) {
      if (i < n_scal) {
        PetscCall(PetscManagedTypeCreateDefault(dctx,i,scal_arr+i));
        if (i) PetscCall(TestOp(dctx,scal_arr[i]));
      }
      if (i >= n_scal/2) {
        PetscCall(PetscManagedTypeDestroy(dctx,scal_arr+j));
        ++j;
      }
    }
    PetscFunctionReturn(0);
  }

  template <typename T>
  PetscErrorCode TestGetValuesAndOp(PetscDeviceContext dctx, PetscManagedType scal, T&& TestOp) const noexcept
  {
    const auto syncs  = std::array<PetscBool,2>{PETSC_FALSE,PETSC_TRUE};
    const auto mtypes = std::array<PetscMemType,2>{
      PETSC_MEMTYPE_HOST,
      PETSC_MEMTYPE_DEVICE
    };
    const auto modes  = std::array<PetscMemoryAccessMode,3>{
      PETSC_MEMORY_ACCESS_WRITE,
      PETSC_MEMORY_ACCESS_READ_WRITE,
      PETSC_MEMORY_ACCESS_READ
    };
    PetscDevice     dev;
    PetscDeviceType dtype;

    PetscFunctionBegin;
    PetscCall(PetscDeviceContextGetDevice(dctx,&dev));
    PetscCall(PetscDeviceGetType(dev,&dtype));
    for (auto sync : syncs) {
      for (auto mtype : mtypes) {
        // host device cannot handle device memory
        if (mtype == PETSC_MEMTYPE_DEVICE && dtype == PETSC_DEVICE_HOST) continue;
        for (auto mode : modes) {
          PetscType *ptr;

          PetscCall(PetscManagedTypeGetValues(dctx,scal,mtype,mode,sync,&ptr));
          PetscCall(TestOp(dctx,scal,mtype,mode,sync,ptr,dtype));
        }
      }
    }
    PetscFunctionReturn(0);
  }

  virtual ~ManagedTypeInterface()                      noexcept = default;
  virtual PetscErrorCode run(PetscDeviceContext) const noexcept = 0;
};

#define PETSC_MANAGED_TYPE_INTERFACE_HEADER(...)                               \
  using BaseType = ManagedTypeInterface<__VA_ARGS__>;                          \
  using typename BaseType::PetscManagedType;                                   \
  using typename BaseType::PetscType;                                          \
  using BaseType::BaseType;                                                    \
  using BaseType::PetscManagedTypeCreate;                                      \
  using BaseType::PetscManagedTypeCreateDefault;                               \
  using BaseType::PetscManagedTypeDestroy;                                     \
  using BaseType::PetscManagedTypeGetValues;                                   \
  using BaseType::PetscManagedTypeGetSize

template <typename PetscType, typename PetscManagedType, typename ...FunctionTypes>
static auto make_managed_interface(FunctionTypes&&... fns) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  ManagedTypeInterface<PetscType,PetscManagedType,FunctionTypes...>{std::forward<FunctionTypes>(fns)...}
);

template <template <typename...> typename T, typename PT, typename PMT, typename... Args>
static auto make_managed_test(Args&&... functions) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  std::unique_ptr<T<PT,PMT,Args...>>{new T<PT,PMT,Args...>{std::forward<Args>(functions)...}}
);

template <template <typename...> typename T>
static auto make_managed_scalar_test() PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  make_managed_test<T,PetscScalar,PetscManagedScalar>(
    PetscManagedScalarCreate,
    PetscManagedScalarCreateDefault,
    PetscManagedScalarDestroy,
    PetscManagedScalarGetValues,
    PetscManagedScalarGetSize
  )
);

template <template <typename...> typename T>
static auto make_managed_real_test() PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  make_managed_test<T,PetscReal,PetscManagedReal>(
    PetscManagedRealCreate,
    PetscManagedRealCreateDefault,
    PetscManagedRealDestroy,
    PetscManagedRealGetValues,
    PetscManagedRealGetSize
  )
);

template <template <typename...> typename T>
static auto make_managed_int_test() PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  make_managed_test<T,PetscInt,PetscManagedInt>(
    PetscManagedIntCreate,
    PetscManagedIntCreateDefault,
    PetscManagedIntDestroy,
    PetscManagedIntGetValues,
    PetscManagedIntGetSize
  )
);
#endif /* __cplusplus */

#endif /* PETSCDEVICETESTCOMMON_H */
