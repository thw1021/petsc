-include ../../../../../petscdir.mk
ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = plexland.c
SOURCECU =
SOURCEF  =
SOURCEH  = land_kernel.h
DIRS     = tutorials cuda kokkos
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMPLEX
LOCDIR   = src/dm/impls/plex/landau

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
