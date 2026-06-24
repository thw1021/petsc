export PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-perlmutter-opt-gcc-kokkos-cuda
cd $PETSC_DIR/src/ksp/ksp/tutorials
make ex56 2>&1 | tail -2
ARGS="-ne 7 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -ksp_converged_reason -use_mat_nearnullspace -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_threshold 0.001 -pc_gamg_coarse_eq_limit 100 -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_rtol 1e-8 -ksp_norm_type unpreconditioned -vec_type kokkos"
for MT in aijkokkos mpibaijkokkos; do
  echo "=== mat_type=$MT (n=8) ==="
  srun -n 8 --gpus-per-node 4 --gpu-bind=none ./ex56 $ARGS -mat_type $MT 2>&1 | grep -i "converged\|Error Message\|wrong state\|incompatible\|not set\|exceeds\|nnz" | head -6
done
