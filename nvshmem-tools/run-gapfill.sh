#!/bin/bash
# Matrix gap fill: heavy-compute cells for graph/device arms at larger sizes, plus
# 10-rep error-bar samples of the four headline 64 KB numbers. 2026-08-15 second job.
cd "$(dirname "$0")"
source ../janus-env-nvshmem.sh >/dev/null 2>&1
export UCX_TLS=sm,self,cuda_copy,cuda_ipc NVSHMEM_BOOTSTRAP=MPI

LOG=results-20260815-gapfill.txt
: > $LOG
run() { timeout 300 mpirun -np 4 --bind-to none -x UCX_TLS -x LD_LIBRARY_PATH -x NVSHMEM_BOOTSTRAP -x NVSHMEM_SYMMETRIC_SIZE "$@" 2>/dev/null | grep -E "per_iter|FAIL" >> $LOG; }

# heavy-compute (G=32 P=16) at the missing sizes: graph and device arms
for d in 65536 131072 524288; do
  for rep in 1 2 3; do run ./acgnbench-nccl-graph -d $d -gaxpy 32 -paxpy 16 -graph_iters 10 -iters 300; done
  for rep in 1 2 3; do run ./acgnbench-nvdev -d $d -gaxpy 32 -paxpy 16 -persistent -blocks 64 -iters 300; done
done

# error-bar samples: 10 reps of the headline 64 KB arms
for rep in $(seq 10); do run ./acgnbench -d 8192 -iters 300; done                                   # branch-MPI
for rep in $(seq 10); do run ./acgnbench-nccl -d 8192 -iters 300; done                             # eager NCCL
for rep in $(seq 10); do run ./acgnbench-nccl-graph -d 8192 -graph_iters 10 -iters 300; done       # graph NCCL
for rep in $(seq 10); do run ./acgnbench-nvdev -d 8192 -persistent -iters 300; done                # device NVSHMEM
echo "GAPFILL-DONE"
