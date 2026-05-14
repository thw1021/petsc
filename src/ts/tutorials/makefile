-include ../../../petscdir.mk

CLEANFILES      = TS-data-* ex40_*_dir

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

V ?= 0
QUIET = $(shell [ "$(V)" = "0" ] && echo @)

runex7: ex7.PETSc
	-${QUIET}${MPIEXEC} -n 1 ${MPIEXEC_TAIL} ./ex7 -da_refine 4 -ts_max_time .05 -snes_mf -pc_type none -ts_exact_final_time matchstep -ts_type cvodebdf > ex7.tmp 2>&1;
	-${QUIET}${PETSC_DIR}/lib/petsc/bin/petscdiff output/ex7_cvodebdf.out ex7.tmp > ex7.dtmp;
	-@if [ ! -s ex7.dtmp ]; then \
		echo "C/C++ example src/ts/tutorials/ex7 run successfully with SUNDIALS CVode"; \
	else \
                cat ex7.tmp ;\
		printf "${PWD}\nPossible problem with src/ts/tutorials/ex7 with SUNDIALS CVode, diffs above\n=========================================\n";\
		touch ../../../../check_error;\
	fi; \
	${RM} -f ex7.tmp ex7.dtmp
