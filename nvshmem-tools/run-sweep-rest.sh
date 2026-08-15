#!/bin/bash
# Remainder of the 2026-08-15 sweep: heavy-compute single-node block + 2-node NCCL suite.
cd "$(dirname "$0")"
source ../janus-env-nvshmem.sh >/dev/null 2>&1

LOG1=results-20260815-nccl-1node.txt
LOG2=results-20260815-nccl-2node.txt

run1() { # args: extra mpirun args ... -- bench args ...
  timeout 300 mpirun -np 4 --bind-to none -x UCX_TLS -x LD_LIBRARY_PATH "$@" 2>/dev/null
}

# ---- finish heavy block, single node ----
export UCX_TLS=sm,self,cuda_copy,cuda_ipc
G=32; P=16
for rep in 1 2 3; do run1 ./acgnbench -d 8192 -monolithic -gaxpy $G -paxpy $P -iters 300 | grep per_iter >> $LOG1; done
for rep in 1 2 3; do run1 ./acgnbench -d 8192 -skip_comm  -gaxpy $G -paxpy $P -iters 300 | grep per_iter >> $LOG1; done
for d in 65536 131072 524288; do
  for mode in "" "-allreduce"; do
    for rep in 1 2 3; do run1 ./acgnbench-nccl -d $d $mode -gaxpy $G -paxpy $P -iters 300 | grep -E "per_iter|FAIL" >> $LOG1; done
  done
  for rep in 1 2 3; do run1 ./acgnbench -d $d -monolithic -gaxpy $G -paxpy $P -iters 300 | grep per_iter >> $LOG1; done
  for rep in 1 2 3; do run1 ./acgnbench -d $d -skip_comm  -gaxpy $G -paxpy $P -iters 300 | grep per_iter >> $LOG1; done
done
echo "HEAVY-BLOCK-DONE"

# ---- 2-node NCCL suite (2+2), light and heavy ----
export UCX_TLS=sm,self,cuda_copy,cuda_ipc,rc
export NCCL_DEBUG=WARN
: > $LOG2
for gp in "8 4" "32 16"; do set -- $gp; G=$1; P=$2
  for d in 8192 65536 131072 524288; do
    for mode in "" "-allreduce"; do
      for rep in 1 2 3; do
        timeout 300 mpirun -np 4 --hostfile $PBS_NODEFILE --map-by ppr:2:node --bind-to none \
          -x UCX_TLS -x LD_LIBRARY_PATH -x NCCL_DEBUG \
          ./acgnbench-nccl -d $d $mode -gaxpy $G -paxpy $P -iters 300 2>/dev/null | grep -E "per_iter|FAIL" >> $LOG2
      done
    done
  done
done
echo "2NODE-BLOCK-DONE"
