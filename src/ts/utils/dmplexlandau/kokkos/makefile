#requirespackage 'PETSC_HAVE_KOKKOS'
#requiresdefine 'PETSC_NOT_USE_CTABLE'
-include ../../../../../petscdir.mk
ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEKOKKOS = landau.kokkos.cxx
SOURCECU =
SOURCEF  =
SOURCEH  =
DIRS     =
LIBBASE  = libpetscts
MANSEC   = TS
SUBMANSEC= LANDAU
LOCDIR   = src/ts/utils/dmplexlandau/kokkos

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

