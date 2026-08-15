# janus-env-nvshmem.sh — toolchain for the hsuh/nvshmem-pg1 worktree on ALCF Janus.
#
# Derived from ../../../janus-env.sh (main worktree) with three deltas:
#   1. PETSC_DIR points at THIS worktree, PETSC_ARCH is arch-janus-nvshmem.
#   2. NVSHMEM is given as the 3.0+ split pair libnvshmem_host.so + libnvshmem_device.a,
#      which is what this branch's config/BuildSystem/config/packages/NVSHMEM.py liblist
#      and the gmakefile device-link rule (-lnvshmem_device) expect. The old monolithic
#      libnvshmem.a is deliberately NOT used.
#   3. Kokkos is dropped — the PetscSF NVSHMEM path needs only VECCUDA, and Kokkos-Kernels
#      dominates build time. Add it back once NVSHMEM itself is verified.
#
# MUST be sourced from bash (PETSC_CONFIGURE_OPTS is an array), and NOT through a pipe
# (a pipe puts `source` in a subshell and silently discards every export).

NVHPC_ROOT=/soft/compilers/nvhpc/Linux_x86_64/24.11

# HPC-X CUDA-aware Open MPI 4.1.7a1. Chosen over /soft/compilers/openmpi-gnu (5.1) because
# NVSHMEM's prebuilt nvshmem_bootstrap_mpi.so links libmpi.so.40 = the Open MPI 4.x ABI.
source "$NVHPC_ROOT/comm_libs/12.6/hpcx/hpcx-2.20/hpcx-init-ompi.sh"
hpcx_load

# CUDA 12.9.1 matches the toolkit NVSHMEM 3.4.5's device objects were built with; nvlink
# rejects device code from a newer toolkit than nvcc.
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.9.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran

export NVSHMEM_DIR=/soft/libraries/nvshmem/libnvshmem-linux-x86_64-3.4.5_cuda12-archive
export LD_LIBRARY_PATH=$NVSHMEM_DIR/lib:$LD_LIBRARY_PATH
export PATH=$NVSHMEM_DIR/bin:$PATH

export AOCL_DIR=/soft/libraries/math_libs/aocl-4.2/4.2.0/gcc/lib_LP64
export LD_LIBRARY_PATH=$AOCL_DIR:$LD_LIBRARY_PATH

# --- RUNTIME settings (discovered 2026-08-14 on x2000c0s1b0n0) ----------------
# UCX_TLS: the default UCX config on Janus compute nodes tries the InfiniBand UD
# path even for intra-node traffic and times out
# ("ibv_create_ah(...) failed: Connection timed out" -> MPI_Init aborts). This is
# reproducible with a bare 4-rank MPI_Init+Allreduce and has nothing to do with
# NVSHMEM. Restricting UCX to shared-memory + CUDA transports fixes it and keeps
# CUDA-awareness. Needed at >=4 ranks; harmless at 2.
export UCX_TLS=${UCX_TLS:-sm,self,cuda_copy,cuda_ipc}

# NVSHMEM's symmetric heap defaults to 1 GiB per PE; trim it so several PEs fit.
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-256M}

# NVSHMEM notes for this site:
#   - The prebuilt nvshmem_bootstrap_mpi.so needs libmpi.so.40 (Open MPI 4.x ABI),
#     which HPC-X 4.1.7a1 provides. Do NOT switch to /soft/compilers/openmpi-gnu
#     (5.x, libmpi.so.80) or the MPI bootstrap will not load.
#   - The IBRC transport is unavailable ("neither nv_peer_mem, or nvidia_peermem
#     detected"), so NVSHMEM here is single-node only (P2P/IPC over NVLink).
#   - PETSc calls cudaSetDevice() before nvshmemx_init_attr(), so it picks GPUs
#     correctly on its own. NVIDIA's bundled perftest binaries do it in the wrong
#     order and land every PE on one GPU; do not use them as the reference.

export PETSC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PETSC_ARCH=arch-janus-nvshmem

export PETSC_CONFIGURE_OPTS=(
  --with-cc=mpicc --with-cxx=mpicxx --with-fc=0
  --with-cuda=1
  --with-cuda-dir="$CUDA_HOME"
  --with-cuda-arch=90
  --with-cudac="$CUDA_HOME/bin/nvcc"
  --CUDAFLAGS="-ccbin g++"
  --with-nvshmem=1
  --with-nvshmem-include="$NVSHMEM_DIR/include"
  --with-nvshmem-lib="$NVSHMEM_DIR/lib/libnvshmem_host.so $NVSHMEM_DIR/lib/libnvshmem_device.a $CUDA_HOME/lib64/stubs/libcuda.so"
  --with-blaslapack-lib="$AOCL_DIR/libflame.so $AOCL_DIR/libblis.so $AOCL_DIR/libaoclutils.so"
  --with-debugging=0
  COPTFLAGS=-O3 CXXOPTFLAGS=-O3
)
