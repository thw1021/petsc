#requiresdefine 'PETSC_HAVE_CXX'
#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCECXX = contextcupm.cxx
SOURCECU  =
SOURCEF	  =
SOURCEH   = contextcupm.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/device/impls/cupm/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

