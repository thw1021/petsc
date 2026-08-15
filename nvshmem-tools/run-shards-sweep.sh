#!/bin/bash
# Shards (m>>1) sweep v2: sender-side pipelining x transport x put-protocol. 2026-08-15.
cd "$(dirname "$0")"
source ../janus-env-nvshmem.sh >/dev/null 2>&1
export UCX_TLS=sm,self,cuda_copy,cuda_ipc

LOG=results-20260815-shards.txt
: > $LOG

for d in 8192 131072 524288; do
  for G in 8 32; do
    P=$((G / 2))
    for m in 1 2 4 8; do
      for arm in "0 0" "1 0" "1 1"; do
        set -- $arm; nv=$1; ps=$2
        for rep in 1 2; do
          timeout 300 mpirun -np 4 --bind-to none -x UCX_TLS -x LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE \
            ./acgnbench -d $d -gaxpy $G -paxpy $P -shards $m -use_nvshmem $nv -use_nvshmem_putsig $ps -iters 300 2>/dev/null | grep per_iter | sed "s/$/ ps=$ps/" >> $LOG
        done
      done
    done
  done
done
echo "SHARDS-SWEEP-DONE"
