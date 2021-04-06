#requirespackage 'PETSC_HAVE_HIP'
ALL: lib

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
SOURCEC	  =
SOURCEHIP = streamhip.hip.cpp eventhip.hip.cpp scalhip.hip.cpp
SOURCEF	  =
SOURCEH	  = streamhip.h
MANSEC	  = Sys
LIBBASE	  = libpetscsys
DIRS	  =
LOCDIR	  = src/sys/objects/stream/impls/hip/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

