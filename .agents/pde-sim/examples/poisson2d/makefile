# Build the 2-D Poisson MMS verification program.
# Requires PETSC_DIR (and usually PETSC_ARCH) to be set in the environment.
#
#   make poisson
#   mpiexec -n 4 ./poisson -mms_base 17 -mms_levels 4 -mms_csv convergence.csv
#
# The default solver is CG; add -pc_type gamg (or -pc_type mg) on the command
# line for mesh-independent iteration counts.

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson: poisson.o
	${CLINKER} -o poisson poisson.o ${PETSC_LIB}
	${RM} poisson.o

# Convenience: run the full convergence study on 4 ranks.
run: poisson
	mpiexec -n 4 ./poisson -mms_base 17 -mms_levels 4 -pc_type gamg -mms_csv convergence.csv

# Convenience: single-resolution solution-output solve (129x129) on 4 ranks.
# Writes poisson_solution.xdmf + *.bin (ParaView-openable) and poisson_solution.vts.
# NOTE: this PETSc build has no HDF5, so XDMF wraps raw binary field dumps.
run-output: poisson
	mpiexec -n 4 ./poisson -out_grid 129 -mms_levels 0 -ksp_type cg -pc_type gamg -out_prefix poisson_solution

clean::
	${RM} poisson poisson.o convergence.csv poisson_solution.xdmf poisson_solution.vts poisson_solution_*.bin
