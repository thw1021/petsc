-include ../../../../../petscdir.mk
#requirespackage   'PETSC_HAVE_SUNDIALS'

MANSEC   = TS

PRELIBS = ${SUNDIALS_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

V ?= 0
QUIET = $(shell [ "$(V)" = "0" ] && echo @)

runcv: cv_petsc_ex7.PETSc
	-${QUIET}${MPIEXEC} -n 1 ${MPIEXEC_TAIL} cv_petsc_ex7 > ex19.tmp 2>&1;
	-${QUIET}${DIFF} output/cv_petsc_ex7.out ex19.tmp > ex19.dtmp;
	-@if [ ! -s ex19.dtmp ];  then \
		echo "C/C++ example src/ts/impls/sundials/tests run successfully with SUNDIALS"; \
	else  \
		printf "${PWD}\nPossible problem with cv_petsc_ex7 running with SUNDIALS, diffs above\n=========================================\n";\
		touch ../../../../../check_error;\
	fi; \
	${RM} -f ex19.tmp ex19.dtmp;
