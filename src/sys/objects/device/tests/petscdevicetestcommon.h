#ifndef PETSCDEVICE_H
#error "included this file before petscdevice.h, this file must be included last to ensure that public petsc headers are well formed"
#endif
#ifndef PETSCDEVICETESTCOMMON_H
#define PETSCDEVICETESTCOMMON_H
/*  This header file should NEVER #include another file and should be the last thing included
 *  in the test file. This is to guard against ill-formed PetscDevice header files!
 */
PETSC_STATIC_INLINE PetscErrorCode AssertDeviceExists(PetscDevice device)
{
  PetscFunctionBegin;
  PetscValidDevice(device,1);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertDeviceDoesNotExist(PetscDevice device)
{
  PetscFunctionBegin;
  if (device) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDevice was not destroyed for kind %s",device->kind);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertDeviceContextExists(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertDeviceContextDoesNotExist(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  if (dctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContext was not destroyed");
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertPetscStreamTypesValidAndEqual(PetscStreamType left, PetscStreamType right, const char *errStr)
{
  PetscFunctionBegin;
  PetscValidStreamType(left,1);
  PetscValidStreamType(right,2);
  if (left != right) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,errStr,PetscStreamTypes[left],PetscStreamTypes[right]);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertPetscDevicesValidAndEqual(PetscDevice left, PetscDevice right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDevices(left,1,right,2);
  if (left != right) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,errStr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode AssertPetscDeviceContextsValidAndEqual(PetscDeviceContext left, PetscDeviceContext right, const char *errStr)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDeviceContexts(left,1,right,2);
  if (left != right) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,errStr);
  PetscFunctionReturn(0);
}
#endif /* PETSCDEVICETESTCOMMON_H */
