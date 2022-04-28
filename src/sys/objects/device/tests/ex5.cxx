static const char help[] = "Tests creation and destruction of PetscManagedScalar/Real/Int.\n\n";

#include "petscdevicetestcommon.h"
#include <petsc/private/cpputil.hpp>
#include <array>

template <
  typename PetscType,
  typename PetscManagedType,
  typename CreateT,
  typename CreateDefaultT,
  typename DestroyT,
  typename GetValuesT
  >
struct ManagedTypeInterface
{
  const CreateT        PetscManagedTypeCreate;
  const CreateDefaultT PetscManagedTypeCreateDefault;
  const DestroyT       PetscManagedTypeDestroy;
  const GetValuesT     PetscManagedTypeGetValues;

  PetscErrorCode TestGetValues(PetscDeviceContext dctx, PetscManagedType scal) const noexcept
  {
    const auto syncs  = std::array<PetscBool,2>{PETSC_TRUE,PETSC_FALSE};
    const auto mtypes = std::array<PetscMemType,2>{
      PETSC_MEMTYPE_HOST,
      PETSC_MEMTYPE_DEVICE
    };
    const auto modes  = std::array<PetscMemoryAccessMode,3>{
      PETSC_MEMORY_ACCESS_READ,
      PETSC_MEMORY_ACCESS_READ_WRITE,
      PETSC_MEMORY_ACCESS_WRITE
    };

    PetscFunctionBegin;
    for (auto sync : syncs) {
      for (auto mtype : mtypes) {
        if (mtype == PETSC_MEMTYPE_DEVICE) continue;
        for (auto mode : modes) {
          PetscType *ptr;

          PetscCall(PetscManagedTypeGetValues(dctx,scal,mtype,mode,sync,&ptr));
        }
      }
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingletonDefault(PetscDeviceContext dctx, PetscInt nmax = 20) const noexcept
  {
    PetscManagedType  scal;

    PetscFunctionBegin;
    // single size
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,1,&scal));
      PetscCall(TestGetValues(dctx,scal));
      PetscCall(PetscManagedTypeDestroy(dctx,&scal));
    }

    // single large size
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,10,&scal));
      PetscCall(TestGetValues(dctx,scal));
      PetscCall(PetscManagedTypeDestroy(dctx,&scal));
    }

    // different sizes
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,i,&scal));
      if (i) PetscCall(TestGetValues(dctx,scal));
      PetscCall(PetscManagedTypeDestroy(dctx,&scal));
    }

    // different large sizes
    for (PetscInt i = 0; i < nmax; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,100*i,&scal));
      if (i) PetscCall(TestGetValues(dctx,scal));
      PetscCall(PetscManagedTypeDestroy(dctx,&scal));
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode TestSingleton(PetscDeviceContext dctx, PetscInt nmax = 20) const noexcept
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
          if (k) PetscCall(TestGetValues(dctx,scal));
          PetscCall(PetscManagedTypeDestroy(dctx,&scal));
        }
      }
    }
    PetscFunctionReturn(0);
  }

  template <PetscInt n_scal = 20>
  PetscErrorCode TestGroupDefault(PetscDeviceContext dctx) const noexcept
  {
    PetscManagedType scal_arr[n_scal];

    PetscFunctionBegin;
    static_assert(n_scal % 2 == 0,"");
    // destroy in original order
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,1,scal_arr+i));
      if (i) PetscCall(TestGetValues(dctx,scal_arr[i]));
    }
    for (PetscInt i = 0; i < n_scal; ++i) PetscCall(PetscManagedTypeDestroy(dctx,scal_arr+i));

    // destroy in reverse order
    for (PetscInt i = 0; i < n_scal; ++i) {
      PetscCall(PetscManagedTypeCreateDefault(dctx,100*i,scal_arr+i));
      if (i) PetscCall(TestGetValues(dctx,scal_arr[i]));
    }
    for (PetscInt i = n_scal-1; i >= 0; --i) PetscCall(PetscManagedTypeDestroy(dctx,scal_arr+i));

    // destroy as we create
    for (PetscInt i = 0, j = 0; i < n_scal+(n_scal/2); ++i) {
      if (i < n_scal) {
        PetscCall(PetscManagedTypeCreateDefault(dctx,i,scal_arr+i));
        if (i) PetscCall(TestGetValues(dctx,scal_arr[i]));
      }
      if (i >= n_scal/2) {
        PetscCall(PetscManagedTypeDestroy(dctx,scal_arr+j));
        ++j;
      }
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode run(PetscDeviceContext dctx) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(TestSingletonDefault(dctx));
    PetscCall(TestGroupDefault(dctx));
    PetscCall(TestSingleton(dctx));
    PetscFunctionReturn(0);
  }
};

template <typename PetscType, typename PetscManagedType, typename ...FunctionTypes>
static auto make_managed_interface(FunctionTypes&&... fns) PETSC_DECLTYPE_NOEXCEPT_AUTO_RETURNS(
  ManagedTypeInterface<PetscType,PetscManagedType,FunctionTypes...>{std::forward<FunctionTypes>(fns)...}
);

static PetscErrorCode TestPetscManagedScalar(PetscDeviceContext dctx)
{
  const auto interface = make_managed_interface<PetscScalar,PetscManagedScalar>(
    PetscManagedScalarCreate,
    PetscManagedScalarCreateDefault,
    PetscManagedScalarDestroy,
    PetscManagedScalarGetValues
  );

  PetscFunctionBegin;
  PetscCall(interface.run(dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestPetscManagedReal(PetscDeviceContext dctx)
{
  const auto interface = make_managed_interface<PetscReal,PetscManagedReal>(
    PetscManagedRealCreate,
    PetscManagedRealCreateDefault,
    PetscManagedRealDestroy,
    PetscManagedRealGetValues
  );

  PetscFunctionBegin;
  PetscCall(interface.run(dctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestPetscManagedInt(PetscDeviceContext dctx)
{
  const auto interface = make_managed_interface<PetscInt,PetscManagedInt>(
    PetscManagedIntCreate,
    PetscManagedIntCreateDefault,
    PetscManagedIntDestroy,
    PetscManagedIntGetValues
  );

  PetscFunctionBegin;
  PetscCall(interface.run(dctx));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  // test on current context (whatever that may be)
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(TestPetscManagedScalar(dctx));
  PetscCall(TestPetscManagedReal(dctx));
  PetscCall(TestPetscManagedInt(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 2 5}}
   args: -device_enable {{lazy eager}}
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -root_device_context_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -root_device_context_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -root_device_context_device_type hip
     suffix: hip
   test:
     requires: sycl
     args: -root_device_context_device_type sycl
     suffix: sycl

TEST*/
