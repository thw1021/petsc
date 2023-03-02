-include ../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_FFTW'

CFLAGS   = ${FFTW_INCLUDE}
SOURCEC  = fftw.c
SOURCEH  =
LIBBASE  = libpetscmat
MANSEC   = Mat

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
