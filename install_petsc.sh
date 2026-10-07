#!/usr/bin/env bash
#
# PETSc build & installation script.
#
# Rebuilds PETSc from this repository (updated to the upstream 'release'
# branch, PETSc 3.26.x) with the same feature set as the previous install
# (/home/tang/packages/petsc-bak, PETSc 3.24.3):
#   1. Dependency check (MPICH 4.3, CUDA 12.2, BLAS/LAPACK, compilers, Python)
#   2. Configure    -> PETSC_ARCH=arch-linux-c-opt (same arch name as before)
#   3. make all     -> shared libraries in ${PETSC_DIR}/${PETSC_ARCH}
#   4. Install      -> /home/tang/packages/petsc/install (config/install.py)
#   5. Smoke test   -> 2-rank KSP solve (CPU path + CUDA support report)
#   6. ~/.bashrc environment check
#
# Reference configuration (from the petsc-bak reconfigure script):
#   --with-mpi-dir=/home/tang/packages/mpichInstall   (MPICH 4.3.0, gcc 11.4)
#   --with-cuda=1  --with-cuda-dir=/usr/local/cuda-12.2
#   --with-debugging=0, -O3 for C/C++/Fortran
#   system netlib BLAS/LAPACK, Fortran bindings enabled
#   real scalars, double precision, 32-bit indices, shared libraries
#
# Every step is skipped when its artifact already exists, so the script can
# be re-run to only rebuild what changed. Delete ${PETSC_DIR}/${PETSC_ARCH}
# to force a full reconfigure + rebuild.

set -e

print_info()    { echo -e "\033[1;34m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
print_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }

PETSC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PETSC_ARCH=arch-linux-c-opt
ARCH_DIR="${PETSC_DIR}/${PETSC_ARCH}"
PREFIX="${PETSC_DIR}/install"
MPI_DIR="/home/tang/packages/mpichInstall"
CUDA_DIR="/usr/local/cuda-12.2"
NPROC=$(nproc)
BASHRC="${HOME}/.bashrc"

cd "${PETSC_DIR}"

print_info "=========================================="
print_info "        PETSc Installation Script"
print_info "=========================================="
print_info "Source:           ${PETSC_DIR}"
print_info "Reference build:  /home/tang/packages/petsc-bak (PETSc 3.24.3)"
print_info "PETSC_ARCH:       ${PETSC_ARCH}"
print_info "Install prefix:   ${PREFIX}"
print_info "Build jobs:       ${NPROC}"
echo ""

# ============================================================================
# Step 1: Check required dependencies
# ============================================================================
print_info "Step 1: Checking required dependencies..."

MISSING_DEPS=()

check_file() {
    local path=$1
    local name=$2
    if [ -e "${path}" ]; then
        print_success "$name found: ${path}"
    else
        print_warning "$name not found: ${path}"
        MISSING_DEPS+=("$name")
    fi
}

check_file "${MPI_DIR}/bin/mpicc"    "MPICH C compiler (mpicc)"
check_file "${MPI_DIR}/bin/mpicxx"   "MPICH C++ compiler (mpicxx)"
check_file "${MPI_DIR}/bin/mpif90"   "MPICH Fortran compiler (mpif90)"
check_file "${MPI_DIR}/bin/mpiexec"  "MPICH launcher (mpiexec)"
check_file "${CUDA_DIR}/bin/nvcc"    "CUDA compiler (nvcc 12.2)"

for cmd in python3 make gfortran; do
    if command -v "$cmd" &>/dev/null; then
        print_success "$cmd found: $(command -v "$cmd")"
    else
        print_warning "$cmd not found"
        MISSING_DEPS+=("$cmd")
    fi
done

if command -v pkg-config &>/dev/null; then
    print_success "pkg-config found: $(command -v pkg-config)"
else
    print_warning "pkg-config not found (smoke test will use fallback link flags)"
fi

print_info "Checking BLAS/LAPACK..."
for lib in liblapack libblas; do
    if ldconfig -p | grep -q "$lib"; then
        print_success "$lib found"
    else
        print_warning "$lib not found"
        MISSING_DEPS+=("$lib (liblapack-dev libblas-dev)")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_error "The following required dependencies are missing:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  - $dep"
    done
    echo ""
    print_error "Installation aborted."
    exit 1
fi

print_success "All required dependencies are installed."
echo ""

# ============================================================================
# Step 2: Configure (same options as the reference petsc-bak install)
# ============================================================================
print_info "Step 2: Configuring PETSc (arch-linux-c-opt)..."
print_info "Features: MPI (MPICH 4.3), CUDA 12.2 device support, Fortran"
print_info "bindings, system BLAS/LAPACK, optimized build (-O3, no debugging),"
print_info "real scalars / double precision / 32-bit indices, shared libraries."
echo ""

CONFIGURE_LOG="${PETSC_DIR}/configure-run.log"
if [ -f "${ARCH_DIR}/lib/petsc/conf/petscvariables" ]; then
    print_success "PETSc already configured: ${ARCH_DIR}"
else
    print_info "Running configure (log: ${CONFIGURE_LOG})..."
    if ! ./configure \
        PETSC_ARCH=${PETSC_ARCH} \
        --prefix="${PREFIX}" \
        --with-mpi-dir="${MPI_DIR}" \
        --with-cuda=1 \
        --with-cuda-dir="${CUDA_DIR}" \
        --with-debugging=0 \
        --COPTFLAGS=-O3 \
        --CXXOPTFLAGS=-O3 \
        --FOPTFLAGS=-O3 \
        --with-make-np=${NPROC} \
        > "${CONFIGURE_LOG}" 2>&1; then
        print_error "Configure failed. Last 40 lines of ${CONFIGURE_LOG}:"
        tail -40 "${CONFIGURE_LOG}"
        exit 1
    fi
    print_success "Configure completed"
fi

# ============================================================================
# Step 3: Build all libraries (shared, includes Fortran bindings)
# ============================================================================
print_info "Step 3: Building PETSc libraries..."

BUILD_LOG="${PETSC_DIR}/make-all.log"
if [ -f "${ARCH_DIR}/lib/libpetsc.so" ]; then
    print_success "PETSc libraries already built: ${ARCH_DIR}/lib"
else
    print_info "Compiling PETSc with ${NPROC} jobs (this takes a while)..."
    if ! make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}" all \
        > "${BUILD_LOG}" 2>&1; then
        print_error "Build failed. Last 40 lines of ${BUILD_LOG}:"
        tail -40 "${BUILD_LOG}"
        exit 1
    fi
    [ -f "${ARCH_DIR}/lib/libpetsc.so" ] || {
        print_error "libpetsc.so not found after build; see ${BUILD_LOG}";
        exit 1;
    }
    print_success "PETSc libraries built: ${ARCH_DIR}/lib/libpetsc.so"
fi

# ============================================================================
# Step 4: Install to ${PREFIX}
# ============================================================================
print_info "Step 4: Installing PETSc to ${PREFIX}..."

# A prefix directory named "install" (created by configure) shadows the
# non-.PHONY install target in the root makefile, which would make
# "make install" report "'install' is up to date" and do nothing. Invoke
# the installer that the target itself runs instead.
INSTALL_LOG="${PETSC_DIR}/make-install.log"
if [ -f "${PREFIX}/lib/libpetsc.so" ]; then
    print_success "PETSc already installed: ${PREFIX}"
else
    if ! PETSC_DIR="${PETSC_DIR}" PETSC_ARCH="${PETSC_ARCH}" \
            python3 ./config/install.py > "${INSTALL_LOG}" 2>&1; then
        print_error "Install failed. Last 40 lines of ${INSTALL_LOG}:"
        tail -40 "${INSTALL_LOG}"
        exit 1
    fi
    [ -f "${PREFIX}/lib/libpetsc.so" ] || {
        print_error "libpetsc.so not found in ${PREFIX}/lib after install; see ${INSTALL_LOG}";
        exit 1;
    }
    print_success "PETSc installed: ${PREFIX}"
fi

# ============================================================================
# Step 5: Smoke test (2-rank KSP solve, CPU path)
# ============================================================================
print_info "Step 5: Smoke testing (2-rank KSP solve)..."

TEST_DIR=$(mktemp -d /tmp/test_petsc.XXXXXX)
cat > "${TEST_DIR}/test_petsc.c" << 'EOF'
#include <petsc.h>

int main(int argc, char **argv)
{
    Mat       A;
    Vec       x, b, e;
    KSP       ksp;
    PetscInt  rstart, rend, r;
    PetscReal err;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    /* Solve A x = b with A = diag(2, 2), b = (2, 2); exact x = (1, 1).
       Each rank only touches its own rows, so the test is rank-safe. */
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
    for (r = rstart; r < rend; r++) PetscCall(MatSetValue(A, r, r, 2.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecGetOwnershipRange(b, &rstart, &rend));
    for (r = rstart; r < rend; r++) PetscCall(VecSetValue(b, r, 2.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(VecDuplicate(x, &e));
    PetscCall(VecCopy(x, e));
    PetscCall(VecShift(e, -1.0));
    PetscCall(VecNorm(e, NORM_2, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "solution error norm: %.3e\n", (double)err));
#if defined(PETSC_HAVE_CUDA)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PETSc CUDA support: enabled\n"));
#else
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PETSc CUDA support: disabled\n"));
#endif
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&e));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return err < 1.0e-10 ? 0 : 1;
}
EOF

if command -v pkg-config >/dev/null 2>&1 && [ -f "${ARCH_DIR}/lib/pkgconfig/PETSc.pc" ]; then
    PETSC_FLAGS=$(pkg-config --cflags --libs "${ARCH_DIR}/lib/pkgconfig/PETSc.pc")
else
    PETSC_FLAGS="-I${PETSC_DIR}/include -I${ARCH_DIR}/include -L${ARCH_DIR}/lib -lpetsc"
fi
# PETSc.pc does not embed rpath; add it so the test binary runs without
# relying on LD_LIBRARY_PATH exported by ~/.bashrc.
PETSC_FLAGS="${PETSC_FLAGS} -Wl,-rpath,${ARCH_DIR}/lib -Wl,-rpath,${PREFIX}/lib"

if "${MPI_DIR}/bin/mpicc" "${TEST_DIR}/test_petsc.c" ${PETSC_FLAGS} \
        -o "${TEST_DIR}/test_petsc" 2>&1 && \
        (cd "${TEST_DIR}" && "${MPI_DIR}/bin/mpiexec" -n 2 ./test_petsc); then
    print_success "Smoke test passed (2-rank KSP solve)"
else
    print_error "Smoke test failed"
    rm -rf "${TEST_DIR}"
    exit 1
fi
rm -rf "${TEST_DIR}"

# ============================================================================
# Step 6: Check ~/.bashrc environment settings
# ============================================================================
print_info "Step 6: Checking ${BASHRC} environment settings..."

if grep -q "export PETSC_DIR=${PETSC_DIR}" "${BASHRC}" && \
       grep -q "export PETSC_ARCH=${PETSC_ARCH}" "${BASHRC}"; then
    print_success "${BASHRC} already sets PETSC_DIR and PETSC_ARCH correctly (no change needed)"
else
    print_warning "${BASHRC} is missing PETSc settings; add the following lines:"
    echo ""
    echo "  export PETSC_DIR=${PETSC_DIR}"
    echo "  export PETSC_ARCH=${PETSC_ARCH}"
    echo "  export PATH=\$PETSC_DIR/\$PETSC_ARCH/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"
    echo ""
fi

echo ""
print_info "=========================================="
print_success "      PETSc Installation Complete!"
print_info "=========================================="
echo ""
print_info "Installed features (${ARCH_DIR}):"
echo "  - PETSc $(grep -oP 'define PETSC_VERSION_MAJOR\s+\K[0-9]+' include/petscversion.h).$(grep -oP 'define PETSC_VERSION_MINOR\s+\K[0-9]+' include/petscversion.h).$(grep -oP 'define PETSC_VERSION_SUBMINOR\s+\K[0-9]+' include/petscversion.h) (release branch)"
echo "  - Core packages: Vec, Mat, KSP/PC, SNES, TS, DM, Tao, Sys"
echo "  - MPI: MPICH 4.3 (${MPI_DIR})"
echo "  - CUDA device support: 12.2 (${CUDA_DIR})"
echo "  - Fortran bindings: enabled (mpif90)"
echo "  - BLAS/LAPACK: system netlib"
echo "  - Scalar type: real, double precision; 32-bit indices"
echo "  - Shared libraries: yes; optimized build (-O3, debugging off)"
echo ""
print_info "Install prefix (make install): ${PREFIX}"
echo ""
print_info "To use PETSc (already set in ~/.bashrc):"
echo ""
echo "  export PETSC_DIR=${PETSC_DIR}"
echo "  export PETSC_ARCH=${PETSC_ARCH}"
echo ""
print_info "Compile a program against the build tree:"
echo ""
echo "  mpicc solver.c -I\${PETSC_DIR}/include -I\${PETSC_DIR}/\${PETSC_ARCH}/include \\"
echo "      -L\${PETSC_DIR}/\${PETSC_ARCH}/lib -lpetsc -o solver.out"
echo ""
print_info "or with pkg-config:"
echo ""
echo "  mpicc solver.c \$(pkg-config --cflags --libs \\"
echo "      \${PETSC_DIR}/\${PETSC_ARCH}/lib/pkgconfig/PETSc.pc) -o solver.out"
echo ""
print_info "Run GPU vectors/matrices at runtime, e.g.:"
echo ""
echo "  mpiexec -n 2 ./solver.out -vec_type cuda -mat_type aijcusparse"
echo ""
print_info "Key headers:"
echo "  #include <petscvec.h>   // 向量"
echo "  #include <petscmat.h>   // 矩阵"
echo "  #include <petscksp.h>   // 线性求解器 KSP/PC"
echo "  #include <petscsnes.h>  // 非线性求解器 SNES"
echo "  #include <petscts.h>    // 时间积分 TS"
echo "  #include <petscdm.h>    // 网格/拓扑管理 DM"
echo "  #include <petsctao.h>   // 优化 Tao"
