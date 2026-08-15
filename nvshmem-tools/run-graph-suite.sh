#!/bin/bash
# CUDA-graph replay suite (route 2a), single node + a 2-node probe. 2026-08-15.
cd "$(dirname "$0")"
source ../janus-env-nvshmem.sh >/dev/null 2>&1

LOG=results-20260815-nccl-graph.txt
: > $LOG

# ---- single node ----
export UCX_TLS=sm,self,cuda_copy,cuda_ipc
for mode in "" "-allreduce"; do
  for d in 8192 65536 131072 524288; do
    for k in 1 10; do
      for rep in 1 2; do
        timeout 300 mpirun -np 4 --bind-to none -x UCX_TLS -x LD_LIBRARY_PATH \
          ./acgnbench-nccl-graph -d $d $mode -graph_iters $k -iters 300 2>/dev/null | grep -E "per_iter|FAIL" >> $LOG
      done
    done
  done
done
# one heavy-compute point: does the graph win persist when compute dominates?
for rep in 1 2; do
  timeout 300 mpirun -np 4 --bind-to none -x UCX_TLS -x LD_LIBRARY_PATH \
    ./acgnbench-nccl-graph -d 8192 -gaxpy 32 -paxpy 16 -graph_iters 10 -iters 300 2>/dev/null | grep -E "per_iter|FAIL" >> $LOG
done
echo "GRAPH-1NODE-DONE"

# ---- 2-node probe: NCCL graph replay over the network transport ----
# NCCL_IB_HCA=mlx5_bond_0 is REQUIRED: letting NCCL use the raw bond-slave devices
# (mlx5_0/mlx5_1) gives ~78 ms/iter of retransmit timeouts instead of ~300 us.
export UCX_TLS=sm,self,cuda_copy,cuda_ipc,rc
export NCCL_IB_HCA=mlx5_bond_0
for mode in "" "-allreduce"; do
  for rep in 1 2; do
    timeout 300 mpirun -np 4 --hostfile $PBS_NODEFILE --map-by ppr:2:node --bind-to none \
      -x UCX_TLS -x LD_LIBRARY_PATH -x NCCL_IB_HCA \
      ./acgnbench-nccl-graph -d 8192 $mode -graph_iters 10 -iters 300 2>/dev/null | grep -E "per_iter|FAIL" >> $LOG
  done
done
echo "GRAPH-2NODE-DONE"

# ---- redo the 2-node eager-NCCL block with the HCA fix ----
LOG2=results-20260815-nccl-2node.txt
: > $LOG2
for gp in "8 4" "32 16"; do set -- $gp; G=$1; P=$2
  for d in 8192 65536 131072 524288; do
    for mode in "" "-allreduce"; do
      for rep in 1 2 3; do
        timeout 300 mpirun -np 4 --hostfile $PBS_NODEFILE --map-by ppr:2:node --bind-to none \
          -x UCX_TLS -x LD_LIBRARY_PATH -x NCCL_IB_HCA \
          ./acgnbench-nccl -d $d $mode -gaxpy $G -paxpy $P -iters 300 2>/dev/null | grep -E "per_iter|FAIL" >> $LOG2
      done
    done
  done
done
echo "2NODE-REDO-DONE"
