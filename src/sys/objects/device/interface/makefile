#requiresdefine 'PETSC_HAVE_CXX'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCECXX = dcontext.cxx device.cxx devicereg.cxx managedtype.cxx
SOURCEF	  =
SOURCEH   = objpool.hpp cupmdevice.hpp sycldevice.hpp hostdevice.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/device/interface/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

