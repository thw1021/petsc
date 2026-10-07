#!/usr/bin/env bash
# PETSc environment — PETSc 3.26.0 (release branch, commit a878298e75e).
#
# Generated alongside install_petsc.sh; mirrors the layout of libROM's
# librom_env.sh. Sourcing this file is sufficient on its own: it exports the
# same PETSC_DIR/PETSC_ARCH as ~/.bashrc and additionally exposes the
# installed prefix and ready-to-use compile/link flags.
#
# Two identical library trees are available (build tree and installed prefix):
#   1. Build tree    $PETSC_ARCH_DIR   (also set in ~/.bashrc)
#   2. Installed     $PETSC_PREFIX     (/home/tang/packages/petsc/install)

export PETSC_DIR="/home/tang/packages/petsc"
export PETSC_ARCH="arch-linux-c-opt"
export PETSC_VERSION="3.26.0"
export PETSC_ARCH_DIR="${PETSC_DIR}/${PETSC_ARCH}"   # build tree
export PETSC_PREFIX="${PETSC_DIR}/install"           # installed prefix

# Dependencies as configured (MPICH 4.3.0, CUDA 12.2)
export PETSC_MPI_DIR="/home/tang/packages/mpichInstall"
export PETSC_CUDA_DIR="/usr/local/cuda-12.2"

# The system mpicc on PATH is OpenMPI; PETSc was built with MPICH and its
# headers hard-error on an MPI-implementation mismatch, so the MPICH wrappers
# must come first.
export PATH="${PETSC_MPI_DIR}/bin:${PATH}"

# pkg-config metadata (compile/link flags for user programs)
export PETSC_PC="${PETSC_ARCH_DIR}/lib/pkgconfig/PETSc.pc"

# Header search paths (build tree first, then installed prefix)
export CPATH="${PETSC_ARCH_DIR}/include:${PETSC_PREFIX}/include:${CPATH}"
export C_INCLUDE_PATH="${PETSC_ARCH_DIR}/include:${PETSC_PREFIX}/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="${PETSC_ARCH_DIR}/include:${PETSC_PREFIX}/include:${CPLUS_INCLUDE_PATH}"

# Library search / runtime paths
export LIBRARY_PATH="${PETSC_ARCH_DIR}/lib:${PETSC_PREFIX}/lib:${PETSC_MPI_DIR}/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${PETSC_ARCH_DIR}/lib:${PETSC_PREFIX}/lib:${PETSC_MPI_DIR}/lib:${LD_LIBRARY_PATH}"

# Convenience compile/link flags. The rpath entries are required: PETSc.pc
# carries no rpath, so without them binaries only run when LD_LIBRARY_PATH
# is set (e.g. by this file or ~/.bashrc).
export PETSC_CFLAGS="-I${PETSC_DIR}/include -I${PETSC_ARCH_DIR}/include"
export PETSC_LDFLAGS="-Wl,-rpath,${PETSC_ARCH_DIR}/lib -Wl,-rpath,${PETSC_PREFIX}/lib -L${PETSC_ARCH_DIR}/lib -lpetsc"

# Multi-rank GPU runs require a GPU-aware MPI; this MPICH build is not.
# Uncomment to let PETSc stage device data through host memory instead of
# aborting (single-rank GPU runs are unaffected):
# export PETSC_OPTIONS="-use_gpu_aware_mpi 0"

echo "PETSc environment variables set (PETSc ${PETSC_VERSION}, ${PETSC_ARCH}):"
echo "  PETSC_DIR=$PETSC_DIR"
echo "  PETSC_ARCH=$PETSC_ARCH"
echo "  PETSC_ARCH_DIR=$PETSC_ARCH_DIR  (build tree)"
echo "  PETSC_PREFIX=$PETSC_PREFIX  (installed)"
echo "  PETSC_PC=$PETSC_PC"
echo "  PETSC_MPI_DIR=$PETSC_MPI_DIR  (mpicc/mpicxx/mpif90/mpiexec on PATH)"
echo "  PETSC_CFLAGS=$PETSC_CFLAGS"
echo "  PETSC_LDFLAGS=$PETSC_LDFLAGS"
