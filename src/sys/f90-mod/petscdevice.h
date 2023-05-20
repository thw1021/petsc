!
! Used by petscsysmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdevice.h"

!
! PetscDevice
!
      type tPetscDevice
        PetscFortranAddr:: d PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscDevice

!
! PetscDeviceTypes
!
      PetscDeviceType, parameter :: PETSC_DEVICE_HOST = 0
      PetscDeviceType, parameter :: PETSC_DEVICE_CUDA = 1
      PetscDeviceType, parameter :: PETSC_DEVICE_HIP  = 2
      PetscDeviceType, parameter :: PETSC_DEVICE_SYCL = 3
      PetscDeviceType, parameter :: PETSC_DEVICE_MAX  = 4

!
!     PetscDeviceInitTypes
!
      PetscDeviceInitType, parameter :: PETSC_DEVICE_INIT_NONE = 0
      PetscDeviceInitType, parameter :: PETSC_DEVICE_LAZY      = 1
      PetscDeviceInitType, parameter :: PETSC_DEVICE_EAGER     = 2

!
! PetscDeviceContext
!
      type tPetscDeviceContext
        PetscFortranAddr:: d PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscDeviceContext

!
! PetscStreamTypes
!
      PetscStreamType, parameter :: PETSC_STREAM_GLOBAL_BLOCKING    = 0
      PetscStreamType, parameter :: PETSC_STREAM_DEFAULT_BLOCKING   = 1
      PetscStreamType, parameter :: PETSC_STREAM_GLOBAL_NONBLOCKING = 2
      PetscStreamType, parameter :: PETSC_STREAM_MAX                = 3

!
! PetscDeviceContextJoinModes
!
      PetscDeviceContextJoinMode, parameter :: &
      PETSC_DEVICE_CONTEXT_JOIN_DESTROY = 0
      PetscDeviceContextJoinMode, parameter :: &
      PETSC_DEVICE_CONTEXT_JOIN_SYNC    = 1
      PetscDeviceContextJoinMode, parameter :: &
      PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC = 2

!
! PetscDeviceCopyMode
!
      PetscDeviceCopyMode, parameter :: PETSC_DEVICE_COPY_HTOH = 0
      PetscDeviceCopyMode, parameter :: PETSC_DEVICE_COPY_DTOH = 1
      PetscDeviceCopyMode, parameter :: PETSC_DEVICE_COPY_HTOD = 2
      PetscDeviceCopyMode, parameter :: PETSC_DEVICE_COPY_DTOD = 3
      PetscDeviceCopyMode, parameter :: PETSC_DEVICE_COPY_AUTO = 4

!
! PetscMemoryAccessMode
!
      PetscMemoryAccessMode, parameter :: &
      PETSC_MEMORY_ACCESS_READ       = 1
      PetscMemoryAccessMode, parameter :: &
      PETSC_MEMORY_ACCESS_WRITE      = 2
      PetscMemoryAccessMode, parameter :: &
      PETSC_MEMORY_ACCESS_READ_WRITE = 3
