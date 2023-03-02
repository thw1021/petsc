-include ../../../../../petscdir.mk
#requirespackage  'PETSC_HAVE_AMGX'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECXX = amgx.cxx
SOURCEH   =
LIBBASE   = libpetscksp
MANSEC    = KSP
SUBMANSEC = PC

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
