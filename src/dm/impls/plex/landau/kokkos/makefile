#requiresdefine 'PETSC_HAVE_KOKKOS'
-include ../../../../../../petscdir.mk
ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCECXX= landau_kokkos.cxx
SOURCECU =
SOURCEF  =
SOURCEH  =
DIRS     = tutorials
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMPLEX
LOCDIR   = src/dm/impls/plex/landau/kokkos

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
2
