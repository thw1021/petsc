#requirespackage 'PETSC_HAVE_CUDA'
ALL: lib

CFLAGS   =
FFLAGS   =
SOURCECU = veccupm.cu
SOURCEF  =
LIBBASE  = libpetscvec
MANSEC   = Vec
LOCDIR   = src/vec/vec/impls/seq/seqcupm/cuda/
DIRS     =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

