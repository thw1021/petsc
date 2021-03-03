-include ../../../../../../petscdir.mk
#requirespackage 'PETSC_HAVE_HIP'
ALL: lib

CFLAGS    =
FFLAGS    =
SOURCEHIP  = aijhipsparse.hip.cpp
SOURCEF   =
SOURCEH   = hipsparsematimpl.h
LIBBASE   = libpetscmat
DIRS      =
MANSEC    = Mat
LOCDIR    = src/mat/impls/aij/seq/seqhipsparse/

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
