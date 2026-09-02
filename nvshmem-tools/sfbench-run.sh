#!/bin/bash
# Build and run sfbench.c, sweeping message size for the GPU-aware MPI path and both
# NVSHMEM protocols, and print a comparison table.
#
#   ./sfbench-run.sh [nranks] [reps]
#
# --bind-to none is NOT optional: NVSHMEM's proxy thread spins, and under Open MPI's
# default one-core-per-rank binding it starves the rank, making NVSHMEM look ~10x slower
# than it is. See section 10.1 of NVSHMEM-BUILD-NOTES.md.
set -eu
NP=${1:-4}
REPS=${2:-3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Sweep bounds are overridable so the script can be smoke-tested quickly.
NMIN=${NMIN:-8}
NMAX=${NMAX:-4194304}
if [ ! -r "$HERE/../janus-env-nvshmem.sh" ]; then
  echo "error: $HERE/../janus-env-nvshmem.sh not found -- run this script from its own directory" >&2
  exit 1
fi
source "$HERE/../janus-env-nvshmem.sh" >/dev/null 2>&1
# The binary must be visible from every node, so build it beside the script (home is
# shared), not under the node-local /tmp.
OUT=$(mktemp -d "$HERE/.sfbench-run.XXXXXX")
trap 'rm -rf "$OUT"' EXIT
# Multi-node: MPIRUN_EXTRA="--hostfile $PBS_NODEFILE --map-by ppr:4:node" ./sfbench-run.sh 8
MPIRUN_EXTRA=${MPIRUN_EXTRA:-}
MPIX="-x UCX_TLS -x UCX_NET_DEVICES -x LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE -x NVSHMEM_HCA_LIST ${NVSHMEM_IB_ENABLE_IBGDA:+-x NVSHMEM_IB_ENABLE_IBGDA}"

mpicc -O3 -o "$OUT/sfbench" "$HERE/sfbench.c" \
  -I"$PETSC_DIR/include" -I"$PETSC_DIR/$PETSC_ARCH/include" -I"$CUDA_HOME/include" \
  -Wl,-rpath,"$PETSC_DIR/$PETSC_ARCH/lib" -L"$PETSC_DIR/$PETSC_ARCH/lib" -lpetsc \
  -L"$CUDA_HOME/lib64" -lcudart

echo "node $(hostname -s), np=$NP, best of $REPS reps, UCX_TLS=$UCX_TLS, mpirun extra: ${MPIRUN_EXTRA:-(single node)}"
for rep in $(seq 1 "$REPS"); do
  for mode in mpi nvput nvget; do
    case $mode in
      mpi)   a="-use_nvshmem 0" ;;
      nvput) a="-use_nvshmem 1" ;;
      nvget) a="-use_nvshmem 1 -use_nvshmem_get 1" ;;
    esac
    mpirun -n "$NP" $MPIRUN_EXTRA --bind-to none $MPIX "$OUT/sfbench" -nmin "$NMIN" -nmax "$NMAX" $a > "$OUT/$mode.$rep" 2>&1
  done
done

python3 - "$OUT" "$REPS" <<'PY'
import sys
out, reps = sys.argv[1], int(sys.argv[2])
def load(f):
    d = {}
    for ln in open(f):
        p = ln.split()
        if len(p) == 4 and p[0].isdigit(): d[int(p[0])] = float(p[2])
    return d
def best(mode):
    rs = [load(f"{out}/{mode}.{r}") for r in range(1, reps + 1)]
    ns = set(rs[0])
    for r in rs: ns &= set(r)
    return {n: min(r[n] for r in rs) for n in ns}
m, p, g = best('mpi'), best('nvput'), best('nvget')
print(f"{'bytes':>10} | {'MPI us':>8} {'NVput us':>9} {'NVget us':>9} | {'put/MPI':>8} {'get/MPI':>8}")
print("-" * 64)
for n in sorted(m):
    print(f"{n*8:>10} | {m[n]:>8.2f} {p[n]:>9.2f} {g[n]:>9.2f} | {p[n]/m[n]:>8.2f} {g[n]/m[n]:>8.2f}")
PY
