#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCECU  = contextcuda.cu
SOURCEF	  =
SOURCEHXX = contextcuda.hpp
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/device/impls/cuda/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

