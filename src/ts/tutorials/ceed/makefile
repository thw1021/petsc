# REQUIRES: libceed
ex11: ex11.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

run_sw:
	./ex11 -physics sw -ufv_vtk_interval 0 -dm_plex_adj_cone -dm_plex_adj_closure 0 \
		      -sw_riemann rusanov_ceed -bc_wall 100,101 -ufv_cfl 5 -petsclimiter_type sin \
					-dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -dm_plex_use_ceed \
					-ts_max_time 1 -ts_ssp_type rks2 -ts_ssp_nstages 10 \
					-monitor height,energy ${EXTRA_OPTIONS}

run_euler:
	./ex11 -physics euler -eu_type linear_wave -eu_gamma 1.4 -dm_plex_adj_cone -dm_plex_adj_closure 0 \
         -ufv_vtk_interval 0 -ufv_vtk_basename ${PETSC_DIR}/ex11 -monitor density,energy \
         -eu_riemann godunov_ceed -bc_wall 100,101 -ufv_cfl 5 -petsclimiter_type sin \
         -dm_plex_filename ${PETSC_DIR}/share/petsc/datafiles/meshes/annulus-20.exo -dm_plex_use_ceed \
         -ts_max_time 1 -ts_ssp_type rks2 -ts_ssp_nstages 10 ${EXTRA_OPTIONS}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
