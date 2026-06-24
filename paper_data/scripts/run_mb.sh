export PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-perlmutter-opt-gcc-kokkos-cuda
cd $PETSC_DIR/src/ksp/ksp/tutorials
timeout 150 srun -n 8 --gpus-per-node 4 --gpu-bind=none ./ex56 -ne 7 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -ksp_converged_reason -use_mat_nearnullspace -ksp_rtol 1e-8 -ksp_norm_type unpreconditioned -mat_type mpibaijkokkos > $HOME/mpibaij.out 2>&1
echo "EXIT=$?"
echo "---HEAD---"; head -40 $HOME/mpibaij.out
echo "---TAIL---"; tail -12 $HOME/mpibaij.out
