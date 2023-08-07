-include ../../../../petscdir.mk

DIRS = benchmarkscatters raja

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bench_cu.o: bench.cu bench_impl.hpp
	${PETSC_CUCOMPILE_SINGLE} -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include `pwd`/$<

rm_bench_cu:
	${RM} bench_cu bench_cu.o

bench_cu_debug.o: bench_cu.o

bench_cu_debug: rm_bench_cu
	${MAKE} bench_cu && ${MV} -f bench_cu bench_cu_debug && ${RM} bench_cu.o

bench_hip.o: bench_hip.hip.cpp bench_impl.hpp
	${PETSC_HIPCOMPILE_SINGLE} -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include `pwd`/$<

rm_bench_hip:
	${RM} bench_hip bench_hip.o

bench_hip_debug.o: bench_hip.o

bench_hip_debug: rm_bench_hip
	${MAKE} bench_hip && ${MV} -f bench_hip bench_hip_debug && ${RM} bench_hip.o
