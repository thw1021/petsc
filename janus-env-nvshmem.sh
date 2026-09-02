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

# --- MULTI-NODE (2026-09-01, first job after ALCF installed DOCA-OFED 26.04 + a real
# nvidia_peermem) -----------------------------------------------------------------
# Fabric facts, measured -- they CORRECT the 2026-08 notes, which misread the devices:
#   mlx5_0      = ens2f0np0, 400 Gb/s RoCEv2 (BlueField-3 / ConnectX-7), NUMA 0, PCIe
#                 switch shared with GPU1. THE data port. Up on every node seen so far.
#   mlx5_1      = ens1f0np0, the second 400 Gb/s port; link DOWN on some nodes
#                 (x2000c0s5b0n0), so never rely on it without checking.
#   mlx5_bond_0 = bond0 = mgmt0+mgmt1, a 25 GbE ConnectX-5 MANAGEMENT bond (NUMA 1).
#                 It is NOT a bond of mlx5_0/mlx5_1. Pinning NCCL to it (the 2026-08
#                 "fix") put NCCL on the management network: 2.7 GB/s = 25 Gb/s line rate.
# Every multi-node stack must therefore be pinned to mlx5_0 explicitly; default
# round-robin over the three devices lands PEs on a down port or the 25G bond.
export NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST:-mlx5_0:1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1}
# NCCL on mlx5_0 stalls at ~78 ms per exchange with its RoCE v2 default (GID 3) no matter
# the TC/QP/GDR knobs (sweep 2026-09-01); RoCE v1 on GID 2 runs at full speed. This is the
# combination that turned the 2026-08 "78 ms bond-slave stall" into a working 400G path.
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0}
export NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-2}
# Multi-node UCX_TLS: append ,rc (see the 2026-08 notes: ud is broken; rc slows small
# intra-node messages, so keep the sm-only set for single-node measurements).
# Verified 2026-09-01 (nvshmem_smoke, 8 PEs over 2 nodes): NVSHMEM IBRC transport
# initializes and inter-node put+signal validates. peermem is detected through
# /sys/module/nvidia_peermem/version (NVSHMEM 3.4.5, NCCL 2.28, UCX >= 1.19); the
# HPC-X UCX 1.17 only probes /sys/kernel/mm/memory_peers/nv_mem/version, which this
# DOCA-OFED does not create, so UCX 1.17 stays host-staged even now.
# Put SOURCES for IB transports must be on the symmetric heap (or registered);
# an unregistered cudaMalloc source fails with IBV_WC_LOC_PROT_ERR in the proxy.
#
# IBGDA (GPU-initiated RDMA): NVSHMEM_IB_ENABLE_IBGDA=1 initializes ("It will be used for
# device-side APIs over IB") but the GPU cannot map the NIC doorbell page --
# "cudaHostRegister with IoMemory failed with error=1" -- because the nvidia module lacks
# NVreg_RegistryDwords="PeerMappingOverride=1;" (and EnableStreamMemOPs=0). NVSHMEM then
# runs its hybrid "NIC handler will be CPU with host memory backend" mode (GPU writes the
# WQEs into GPU memory, a CPU thread rings doorbells). Opt in per run; measured in
# nvshmem-tools/results-20260901-2node-ladder4-ibgda.txt.
#   export NVSHMEM_IB_ENABLE_IBGDA=1
# NVSHMEM_IBGDA_NIC_HANDLER=cpu is rejected without GDRCopy (not installed: no /dev/gdrdrv).
#
# GPUDirect MPI (UCX 1.19 from the DOCA repo, extracted to shared home) -- opt in with
# JANUS_UCX119=1 before sourcing. LD_PRELOAD is required: LD_LIBRARY_PATH is not enough
# because HPC-X's Open MPI resolves libucp through RUNPATH-relative paths ("ucx/mt/lib")
# that win over it. Every rank must see the same path (shared home, not /tmp).
if [ "${JANUS_UCX119:-0}" = 1 ]; then
  UCX119=/home/hsuh/opt/ucx-1.19-doca
  export LD_PRELOAD=$UCX119/lib64/libucs.so.0:$UCX119/lib64/libucm.so.0:$UCX119/lib64/libuct.so.0:$UCX119/lib64/libucp.so.0
  export UCX_MODULE_DIR=$UCX119/lib64/ucx
  export UCX_IB_GPU_DIRECT_RDMA=y
  # UCX 1.19's default rendezvous threshold leaves a 35 us hole at 32 KB (10 us at 4 KB and
  # 15 us at 256 KB); 16k closes it with no penalty elsewhere (PN 28.12, 2026-09-02)
  export UCX_RNDV_THRESH=${UCX_RNDV_THRESH:-16k}
  # forward with: mpirun -x LD_PRELOAD -x UCX_MODULE_DIR -x UCX_IB_GPU_DIRECT_RDMA -x UCX_RNDV_THRESH
fi

# NVSHMEM notes for this site:
#   - The prebuilt nvshmem_bootstrap_mpi.so needs libmpi.so.40 (Open MPI 4.x ABI),
#     which HPC-X 4.1.7a1 provides. Do NOT switch to /soft/compilers/openmpi-gnu
#     (5.x, libmpi.so.80) or the MPI bootstrap will not load.
#   - (Obsolete since 2026-09-01) The IBRC transport used to be unavailable ("neither
#     nv_peer_mem, or nvidia_peermem detected"); NVSHMEM is now multi-node capable.
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
