#ifndef IMPLDEVICEBASE_HPP
#define IMPLDEVICEBASE_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpputil.hpp>
#include <petsc/private/viewerimpl.h>
#include <array>
#include <cstring> // for std::strlen

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogInitialize(void);
#else
#define PetscLogInitialize() 0
#endif

namespace Petsc
{

namespace Device
{

namespace Impl
{

template <typename Derived> // CRTP
struct DeviceBase
{
  using createContextFunction_t = PetscErrorCode (*)(PetscDeviceContext);

  // default constructor
  constexpr DeviceBase(createContextFunction_t f) noexcept : create_(f) { }

  template <typename D = Derived>
  PETSC_CXX_COMPAT_DECL(constexpr PetscDeviceType GetPetscDeviceType())
  { return D::GetPetscDeviceType_(); }

protected:
  // function to create a PetscDeviceContext (the (*create) function pointer usually set
  // via XXXSetType() for other PETSc objects)
  const createContextFunction_t create_;

  // if you want the base class to handle the entire options query, has the same arguments as
  // PetscOptionDeviceBasic
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceAll(MPI_Comm,std::pair<PetscDeviceInitType,PetscBool>&,std::pair<PetscInt,PetscBool>&,std::pair<PetscBool,PetscBool>&));

  // if you want to start and end the options query yourself, but still want all the default
  // options
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceBasic(PetscOptionItems*,std::pair<PetscDeviceInitType,PetscBool>&,std::pair<PetscInt,PetscBool>&,std::pair<PetscBool,PetscBool>&));


  // option templates to follow, each one has two forms:
  // - A simple form returning only the value and flag. This gives no control over the message,
  //   arguments to the options query or otherwise
  // - A complex form, which allows you to pass most of the options query arguments *EXCEPT*
  //   - The options query function called
  //   - The option string

  // option template for initializing the device
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceInitialize(PetscOptionItems*,PetscDeviceInitType*,PetscBool*));
  template <typename...T, util::enable_if_t<sizeof...(T)>=3,int> = 0>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceInitialize(PetscOptionItems*,T&&...));
  // option template for selecting the default device
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceSelect(PetscOptionItems*,PetscInt*,PetscBool*));
  template <typename...T, util::enable_if_t<sizeof...(T)>=3,int> = 0>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceSelect(PetscOptionItems*,T&&...));
  // option templates for viewing a device
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceView(PetscOptionItems*,PetscBool*,PetscBool*));
  template <typename...T, util::enable_if_t<sizeof...(T)>=3,int> = 0>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDeviceView(PetscOptionItems*,T&&...));


private:
  // base function for all options templates above, they basically just reformat the arguments,
  // create the option string and pass it off to this function
  template <typename...T, typename F = PetscErrorCode(*)(PetscOptionItems*,const char*,T&&...)>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode PetscOptionDevice(F&&,PetscOptionItems*,const char[],T&&...));
};

template <typename D>
template <typename...T, typename F>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDevice(F&& OptionsFunction, PetscOptionItems *PetscOptionsObject, const char optstub[], T&&... args))
{
  constexpr auto dtype    = GetPetscDeviceType();
  const auto     implname = PetscDeviceTypes[dtype];
  auto           buf      = std::array<char,128>{};
  constexpr auto buflen   = buf.size()-1;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    const auto len = std::strlen(optstub)+std::strlen(implname);

    PetscCheck(len < buflen,PetscOptionsObject->comm,PETSC_ERR_PLIB,"char buffer is not large enough to hold '%s%s'; have %zu need %zu",optstub,implname,buflen,len);
  }
  PetscCall(PetscSNPrintf(buf.data(),buflen,"%s%s",optstub,implname));
  PetscCall(OptionsFunction(PetscOptionsObject,buf.data(),std::forward<T>(args)...));
  PetscFunctionReturn(0);
}

template <typename D>
template <typename...T, util::enable_if_t<sizeof...(T)>=3,int>>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceInitialize(PetscOptionItems *PetscOptionsObject, T&&... args))
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDevice(PetscOptionsEList_Private,PetscOptionsObject,"-device_enable_",std::forward<T>(args)...));
  PetscFunctionReturn(0);
}

template <typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceInitialize(PetscOptionItems *PetscOptionsObject, PetscDeviceInitType *inittype, PetscBool *flag))
{
  auto type = static_cast<PetscInt>(util::integral_value(*inittype));

  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceInitialize(PetscOptionsObject,"How (or whether) to initialize a device","PetscDeviceInitialize()",PetscDeviceInitTypes,3,PetscDeviceInitTypes[type],&type,flag));
  *inittype = static_cast<PetscDeviceInitType>(type);
  PetscFunctionReturn(0);
}

template <typename D>
template <typename...T, util::enable_if_t<sizeof...(T)>=3,int>>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceSelect(PetscOptionItems *PetscOptionsObject, T&&... args))
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDevice(PetscOptionsInt_Private,PetscOptionsObject,"-device_select_",std::forward<T>(args)...));
  PetscFunctionReturn(0);
}

template <typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceSelect(PetscOptionItems *PetscOptionsObject, PetscInt *id, PetscBool *flag))
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceSelect(PetscOptionsObject,"Which device to use. Pass " PetscStringize(PETSC_DECIDE) " to have PETSc decide or (given they exist) [0-" PetscStringize(PETSC_DEVICE_MAX_DEVICES) ") for a specific device","PetscDeviceCreate()",*id,id,flag,PETSC_DECIDE,PETSC_DEVICE_MAX_DEVICES));
  PetscFunctionReturn(0);
}

template <typename D>
template <typename...T, util::enable_if_t<sizeof...(T)>=3,int>>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceView(PetscOptionItems *PetscOptionsObject, T&&... args))
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDevice(PetscOptionsBool_Private,PetscOptionsObject,"-device_view_",std::forward<T>(args)...));
  PetscFunctionReturn(0);
}

template <typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceView(PetscOptionItems *PetscOptionsObject, PetscBool *view, PetscBool *flag))
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceView(PetscOptionsObject,"Display device information and assignments (forces eager initialization)","PetscDeviceView()",*view,view,flag));
  PetscFunctionReturn(0);
}

template <typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceBasic(PetscOptionItems *PetscOptionsObject, std::pair<PetscDeviceInitType,PetscBool> &initType, std::pair<PetscInt,PetscBool> &initId, std::pair<PetscBool,PetscBool> &initView))
{
  PetscFunctionBegin;
  PetscCall(PetscOptionDeviceInitialize(PetscOptionsObject,&initType.first,&initType.second));
  PetscCall(PetscOptionDeviceSelect(PetscOptionsObject,&initId.first,&initId.second));
  PetscCall(PetscOptionDeviceView(PetscOptionsObject,&initView.first,&initView.second));
  PetscFunctionReturn(0);
}

template <typename D>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode DeviceBase<D>::PetscOptionDeviceAll(MPI_Comm comm, std::pair<PetscDeviceInitType,PetscBool> &initType, std::pair<PetscInt,PetscBool> &initId, std::pair<PetscBool,PetscBool> &initView))
{
  constexpr char optname[] = "PetscDevice %s Options";
  constexpr auto dtype     = GetPetscDeviceType();
  const auto     implname  = PetscDeviceTypes[dtype];
  auto           buf       = std::array<char,128>{};
  constexpr auto buflen    = buf.size()-1; // -1 to leave room for null

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    // -3 since '%s' is replaced and dont count null char for optname
    const auto len = std::strlen(implname)+sizeof(optname)-3;

    PetscCheck(len < buflen,comm,PETSC_ERR_PLIB,"char buffer is not large enough to hold 'PetscDevice %s Options'; have %zu need %zu",implname,buflen,len);
  }
  PetscCall(PetscSNPrintf(buf.data(),buflen,optname,implname));
  PetscOptionsBegin(comm,nullptr,buf.data(),"Sys");
  PetscCall(PetscOptionDeviceBasic(PetscOptionsObject,initType,initId,initView));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Device

} // namespace Petsc

#define PETSC_DEVICE_IMPL_BASE_CLASS_HEADER(base_name,T)        \
  using base_name = Petsc::Device::Impl::DeviceBase<T>;         \
  friend base_name;                                             \
  using base_name::base_name

#endif // IMPLDEVICEBASE_HPP
