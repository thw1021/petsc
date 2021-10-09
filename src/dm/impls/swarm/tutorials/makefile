-include ../petscdir.mk
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/swarm/tutorials/
MANSEC          = DM

run:
	-${MPIEXEC} -n 1 ./ex1 -dm_plex_simplex 0 -np 20 -dm_plex_box_faces 2,3 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0  -petscspace_degree 2 -dm_plex_hash_location -ftop_ksp_type lsqr -ftop_pc_type none -dm_view ${EXTRA} 

runpc:
	-${MPIEXEC} -n 1 ./ex1 -dm_plex_simplex 0 -np 20 -dm_plex_box_faces 2,3 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -dm_plex_hash_location -ftop_ksp_type lsqr -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero -dm_view ${EXTRA}

lldb:
	-@lldb ./ex1 -- -dm_plex_simplex 0 -np 20 -dm_plex_box_faces 2,3 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -petscspace_degree 2 -dm_plex_hash_location -ftop_ksp_type lsqr -ftop_ksp_rtol 1e-15 -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero -dm_view ${EXTRA}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test
