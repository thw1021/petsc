#requirespackage 'PETSC_HAVE_CUDA'
-include ../../../../../petscdir.mk
ALL: lib

CPPFLAGS =
CFLAGS   =
FFLAGS   =
SOURCEC  = plexland.c
SOURCECU = landau.cu
SOURCEF  =
SOURCEH  = land_kernel.h
DIRS     = tutorials
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMPLEX
LOCDIR   = src/dm/impls/plex/landau

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
