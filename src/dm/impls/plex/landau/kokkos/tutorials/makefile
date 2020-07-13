#requiresdefine 'PETSC_HAVE_KOKKOS'
#requiresscalar !complex
-include ../../../../../../../petscdir.mk

CFLAGS	        =
FFLAGS	        =
CPPFLAGS        = -I${KOKKOS_DIR}/include -Wl,-rpath,${KOKKOS_DIR}/lib -L${KOKKOS_DIR}/lib -lkokkoscontainers.3.1.1 -lkokkoscore.3.1.1 -D FIX_KOKKOS
FPPFLAGS        =
LOCDIR          = src/dm/impls/plex/landau/kokkos/tutorials/
EXAMPLESCXX     = ex1.cxx
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = DM

#ex1: ex1.o
#	-${CXXLINKER} -o ex1 ex1.o -L${KOKKOS_DIR}/lib -lkokkoscore.3.1.1 -lkokkoscontainers.3.1.1 -Wl,-undefined,dynamic_lookup ${PETSC_SNES_LIB}
#	${RM} ex1.o


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
