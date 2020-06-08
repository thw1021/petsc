#requirespackage 'PETSC_HAVE_CUDA'
-include ../../../../../petscdir.mk
ALL: lib

CPPFLAGS = ${NETCFD_INCLUDE} ${EXODUSII_INCLUDE}
CFLAGS   =
FFLAGS   =
SOURCEC  = 
SOURCECU = landau.cu
SOURCEF  =
SOURCEH  =
DIRS     = generators tests tutorials
LIBBASE  = libpetscdm
MANSEC   = DM
SUBMANSEC= DMPLEX
LOCDIR   = src/dm/impls/plex/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
