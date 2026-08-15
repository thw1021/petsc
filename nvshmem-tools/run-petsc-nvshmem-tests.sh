#!/bin/bash
# PETSc + NVSHMEM suite. Each case runs twice -- -use_nvshmem 0 then 1 -- and requires
# (a) identical output and (b) positive proof NVSHMEM engaged. NVSHMEM_VERSION=1 makes
# NVSHMEM print a banner on init, and PETSc only initializes it once an SF passes
# PetscSFLinkNvshmemCheck(); no banner == silent fallback to the MPI path.
set -u
source /home/hsuh/petsc/.claude/worktrees/nvshmem-v1/janus-env-nvshmem.sh >/dev/null 2>&1
OUT=$(dirname "${BASH_SOURCE[0]}")/petsc-nvshmem-results
mkdir -p $OUT
pass=0; fail=0; incon=0

run_case() {
  local label=$1 np=$2 exe=$3; shift 3
  local off=$OUT/$label.np$np.off.txt on=$OUT/$label.np$np.on.txt
  printf '%-30s np=%s  ' "$label" "$np"
  timeout 600 mpirun -n $np "$exe" "$@" -use_nvshmem 0 > $off 2>&1; local r0=$?
  NVSHMEM_VERSION=1 timeout 600 mpirun -n $np "$exe" "$@" -use_nvshmem 1 > $on 2>&1; local r1=$?
  if [ $r0 -ne 0 ]; then echo "BASELINE FAILED rc=$r0"; fail=$((fail+1)); return; fi
  if [ $r1 -ne 0 ]; then echo "NVSHMEM FAILED rc=$r1"; fail=$((fail+1)); return; fi
  local eng=no; grep -qiE "NVSHMEM v?[0-9]+\.[0-9]+" $on && eng=yes
  local f='/NVSHMEM/d;/nvshmem/d;/ibrc/d;/peermem/d;/^[[:space:]]*$/d'
  sed "$f" $off > $off.clean; sed "$f" $on > $on.clean
  if diff -q $off.clean $on.clean >/dev/null; then
    if [ $eng = yes ]; then echo "PASS (identical, NVSHMEM engaged)"; pass=$((pass+1))
    else echo "INCONCLUSIVE (identical but NVSHMEM never engaged)"; incon=$((incon+1)); fi
  else echo "MISMATCH (engaged=$eng)"; fail=$((fail+1)); fi
}

echo "================ PETSc + NVSHMEM suite, 1 PE per GPU ================"
echo "node   : $(hostname -s)   job ${PBS_JOBID%%.*}"
echo "GPUs   : $($(dirname "${BASH_SOURCE[0]}")/devorder | head -1)"
echo "UCX_TLS=$UCX_TLS  NVSHMEM_SYMMETRIC_SIZE=$NVSHMEM_SYMMETRIC_SIZE"
echo "====================================================================="
SF=$PETSC_DIR/src/vec/is/sf/tests
SN=$PETSC_DIR/src/snes/tutorials
for np in 2 4; do
  run_case "sf-ex22-fetchandop-put" $np $SF/ex22 -vec_type cuda
  run_case "sf-ex22-fetchandop-get" $np $SF/ex22 -vec_type cuda -use_nvshmem_get 1
done
for np in 2 4; do
  run_case "snes-ex19-lidcavity" $np $SN/ex19 -da_refine 4 -dm_vec_type cuda \
     -dm_mat_type aijcusparse -pc_type jacobi -lidvelocity 10 -grashof 100 \
     -snes_monitor_short -snes_converged_reason
done
echo "====================================================================="
echo "PASS=$pass  FAIL=$fail  INCONCLUSIVE=$incon"
[ $fail -eq 0 ] && [ $incon -eq 0 ]
