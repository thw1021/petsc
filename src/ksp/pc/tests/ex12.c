static const char help[] = "Tests user-supplied diagonal scaling for weighted PCASM.\n\n";

#include <petscksp.h>

static inline PetscScalar TestScalar(PetscReal real, PetscReal imag)
{
#if PetscDefined(USE_COMPLEX)
  return PetscCMPLX(real, imag);
#else
  (void)imag;
  return real;
#endif
}

static PetscErrorCode CreateOperator(Mat *A)
{
  PetscMPIInt size;
  PetscInt    start, end, N;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  N = 4 * size;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 4, N, N, 3, NULL, 2, NULL, A));
  PetscCall(MatGetOwnershipRange(*A, &start, &end));
  for (PetscInt row = start; row < end; ++row) {
    PetscCall(MatSetValue(*A, row, row, TestScalar(4.0 + 0.1 * row, 0.2), INSERT_VALUES));
    PetscCall(MatSetValue(*A, row, (row + 1) % N, TestScalar(-0.7, 0.15), INSERT_VALUES));
    PetscCall(MatSetValue(*A, row, (row + N - 1) % N, TestScalar(-0.2, -0.1), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSubdomains(PetscInt nlocal, IS is[], IS inner[])
{
  PetscMPIInt rank, size;
  PetscInt    idx[5], N;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  N = 4 * size;
  for (PetscInt block = 0; block < nlocal; ++block) {
    PetscInt m = nlocal == 2 ? 3 : PetscMin(5, N);

    for (PetscInt j = 0; j < m; ++j) idx[j] = (4 * rank + 2 * block + j) % N;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, idx, PETSC_COPY_VALUES, &is[block]));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 4 / nlocal, 4 * rank + 2 * block, 1, &inner[block]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureSolvers(PC pc)
{
  KSP     *subksp;
  PetscInt nlocal;

  PetscFunctionBeginUser;
  PetscCall(PCASMGetSubKSP(pc, &nlocal, NULL, &subksp));
  for (PetscInt i = 0; i < nlocal; ++i) {
    PC subpc;

    PetscCall(KSPSetType(subksp[i], KSPPREONLY));
    PetscCall(KSPGetPC(subksp[i], &subpc));
    PetscCall(PCSetType(subpc, PCLU));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateScaling(PC pc, PetscScalar first, Vec scaling[])
{
  IS         *is;
  PetscMPIInt rank, size;
  PetscInt    nlocal;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PCASMGetLocalSubdomains(pc, &nlocal, &is, NULL));
  for (PetscInt i = 0; i < nlocal; ++i) {
    const PetscInt *idx;
    PetscScalar    *values;
    PetscInt        m, start = 4 * rank + 2 * i;

    PetscCall(ISGetLocalSize(is[i], &m));
    PetscCall(ISGetIndices(is[i], &idx));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &scaling[i]));
    PetscCall(VecGetArray(scaling[i], &values));
    for (PetscInt j = 0; j < m; ++j) {
      values[j] = 1.0;
      if (nlocal == 2 || size > 1) {
        if (idx[j] == start) values[j] = first;
        else if (idx[j] == (start + m - 1) % (4 * size)) values[j] = 1.0 - first;
      }
    }
    PetscCall(VecRestoreArray(scaling[i], &values));
    PetscCall(ISRestoreIndices(is[i], &idx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PCASMGetLocalScaling() must hand back exactly what was installed, without a copy. */
static PetscErrorCode CheckStoredScaling(PC pc, PetscInt nlocal, Vec scaling[])
{
  Vec      *stored;
  PetscInt  n;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(PCASMGetLocalScaling(pc, &n, &stored));
  PetscCheck(n == nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCASMGetLocalScaling() returned %" PetscInt_FMT " subdomains, expected %" PetscInt_FMT, n, nlocal);
  PetscCheck(stored, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCASMGetLocalScaling() returned no weights after PCASMSetLocalScaling()");
  for (PetscInt i = 0; i < nlocal; ++i) {
    PetscCall(VecEqual(stored[i], scaling[i], &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stored scaling vector %" PetscInt_FMT " differs from the supplied one", i);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InstallScaling(PC pc)
{
  Vec      scaling[2];
  PetscInt nlocal;

  PetscFunctionBeginUser;
  PetscCall(PCASMGetLocalSubdomains(pc, &nlocal, NULL, NULL));
  PetscCall(CreateScaling(pc, 0.25, scaling));
  PetscCall(PCASMSetLocalScaling(pc, nlocal, scaling));
  PetscCall(CheckStoredScaling(pc, nlocal, scaling));
  for (PetscInt i = 0; i < nlocal; ++i) PetscCall(VecDestroy(&scaling[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Assemble sum_i R_i^T D_i A_i^{-1} R_i independently of the ASM scatter and apply code. */
static PetscErrorCode CreateReference(Mat A, PC pc, Vec scaling[], Mat *B)
{
  IS      *is;
  Mat     *sub;
  PetscInt nlocal, m, N;

  PetscFunctionBeginUser;
  PetscCall(PCASMGetLocalSubdomains(pc, &nlocal, &is, NULL));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatGetSize(A, &N, NULL));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, m, m, N, N, m, NULL, N - m, NULL, B));
  PetscCall(MatCreateSubMatrices(A, nlocal, is, is, MAT_INITIAL_MATRIX, &sub));
  for (PetscInt i = 0; i < nlocal; ++i) {
    KSP             ksp;
    PC              subpc;
    Vec             x, y;
    const PetscInt *idx;
    PetscInt        ns;

    PetscCall(ISGetLocalSize(is[i], &ns));
    PetscCall(ISGetIndices(is[i], &idx));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPSetOperators(ksp, sub[i], sub[i]));
    PetscCall(KSPGetPC(ksp, &subpc));
    PetscCall(PCSetType(subpc, PCLU));
    PetscCall(MatCreateVecs(sub[i], &x, &y));
    for (PetscInt j = 0; j < ns; ++j) {
      const PetscScalar *values;

      PetscCall(VecSet(x, 0.0));
      PetscCall(VecSetValue(x, j, 1.0, INSERT_VALUES));
      PetscCall(VecAssemblyBegin(x));
      PetscCall(VecAssemblyEnd(x));
      PetscCall(KSPSolve(ksp, x, y));
      PetscCall(VecPointwiseMult(y, scaling[i], y));
      PetscCall(VecGetArrayRead(y, &values));
      PetscCall(MatSetValues(*B, ns, idx, 1, &idx[j], values, ADD_VALUES));
      PetscCall(VecRestoreArrayRead(y, &values));
    }
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(ISRestoreIndices(is[i], &idx));
  }
  PetscCall(MatDestroySubMatrices(nlocal, &sub));
  PetscCall(MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckVector(Vec actual, Vec expected)
{
  PetscReal error, norm;

  PetscFunctionBeginUser;
  PetscCall(VecNorm(expected, NORM_INFINITY, &norm));
  PetscCall(VecAXPY(expected, -1.0, actual));
  PetscCall(VecNorm(expected, NORM_INFINITY, &error));
  PetscCheck(error <= 100.0 * PETSC_MACHINE_EPSILON * PetscMax(1.0, norm), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Weighted ASM error %g exceeds tolerance", (double)error);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckApplications(PC pc, Mat B, PetscInt nlocal)
{
  Vec      x, y, expected;
  PetscInt N, m, start, end;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(B, &N, NULL));
  PetscCall(MatGetLocalSize(B, &m, NULL));
  PetscCall(MatCreateVecs(B, &x, &y));
  PetscCall(VecDuplicate(y, &expected));
  PetscCall(VecGetOwnershipRange(x, &start, &end));
  for (PetscInt j = 0; j < N; ++j) {
    PetscCall(VecSet(x, 0.0));
    if (start <= j && j < end) PetscCall(VecSetValue(x, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    PetscCall(PCApply(pc, x, y));
    PetscCall(MatMult(B, x, expected));
    PetscCall(CheckVector(y, expected));
    PetscCall(PCApplyTranspose(pc, x, y));
    PetscCall(MatMultTranspose(B, x, expected));
    PetscCall(CheckVector(y, expected));
  }
  if (nlocal == 1) {
    Mat X, Y;

    PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, N, 3, NULL, &X));
    PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
    for (PetscInt j = 0; j < 3; ++j) {
      Vec          column;
      PetscScalar *values;

      PetscCall(MatDenseGetColumnVecWrite(X, j, &column));
      PetscCall(VecGetArray(column, &values));
      for (PetscInt i = 0; i < m; ++i) values[i] = TestScalar(0.1 * (start + i + 1) * (j + 1), 0.2 * (j - start - i));
      PetscCall(VecRestoreArray(column, &values));
      PetscCall(MatDenseRestoreColumnVecWrite(X, j, &column));
    }
    for (PetscInt transpose = 0; transpose < 2; ++transpose) {
      if (transpose) PetscCall(PCMatApplyTranspose(pc, X, Y));
      else PetscCall(PCMatApply(pc, X, Y));
      for (PetscInt j = 0; j < 3; ++j) {
        Vec xc, yc;

        PetscCall(MatDenseGetColumnVecRead(X, j, &xc));
        PetscCall(MatDenseGetColumnVecRead(Y, j, &yc));
        if (transpose) PetscCall(MatMultTranspose(B, xc, expected));
        else PetscCall(MatMult(B, xc, expected));
        PetscCall(CheckVector(yc, expected));
        PetscCall(MatDenseRestoreColumnVecRead(Y, j, &yc));
        PetscCall(MatDenseRestoreColumnVecRead(X, j, &xc));
      }
    }
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&Y));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckInvalidScaling(PC pc, PetscInt nlocal, Vec scaling[])
{
  Vec            invalid, saved, extra[3];
  PetscMPIInt    size;
  PetscInt       m;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(VecGetLocalSize(scaling[0], &m));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCASMSetLocalType(pc, PC_COMPOSITE_MULTIPLICATIVE);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Multiplicative composition was accepted for weighted ASM");
  for (PetscInt i = 0; i <= nlocal; ++i) extra[i] = scaling[0];
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCASMSetLocalScaling(pc, nlocal + 1, extra);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wrong scaling block count was accepted");
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m + 1, &invalid));
  saved      = scaling[0];
  scaling[0] = invalid;
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCASMSetLocalScaling(pc, nlocal, scaling);
  PetscCall(PetscPopErrorHandler());
  scaling[0] = saved;
  PetscCall(VecDestroy(&invalid));
  PetscCheck(ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wrong scaling vector length was accepted");
  PetscCall(VecCreateMPI(PETSC_COMM_SELF, m, m, &invalid));
  scaling[0] = invalid;
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCASMSetLocalScaling(pc, nlocal, scaling);
  PetscCall(PetscPopErrorHandler());
  scaling[0] = saved;
  PetscCall(VecDestroy(&invalid));
  PetscCheck(ierr == PETSC_ERR_ARG_INCOMP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Incompatible scaling vector type was not rejected");
  if (size > 1) {
    PetscCall(VecCreateMPI(PETSC_COMM_WORLD, m, PETSC_DECIDE, &invalid));
    scaling[0] = invalid;
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = PCASMSetLocalScaling(pc, nlocal, scaling);
    PetscCall(PetscPopErrorHandler());
    scaling[0] = saved;
    PetscCall(VecDestroy(&invalid));
    PetscCheck(ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Distributed scaling vector was accepted");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PC             pc;
  Mat            A, B;
  IS             is[2], inner[2];
  Vec            scaling[2], early;
  Vec           *stored;
  PetscInt       nlocal = 2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-local_blocks", &nlocal, NULL));
  PetscCheck(nlocal == 1 || nlocal == 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Use one or two local blocks");
  PetscCall(CreateOperator(&A));
  PetscCall(CreateSubdomains(nlocal, is, inner));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCASM));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCASMSetType(pc, PC_ASM_WEIGHTED));
  PetscCall(PCASMSetLocalSubdomains(pc, nlocal, is, inner));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &early));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = PCASMSetLocalScaling(pc, 1, &early);
  PetscCall(PetscPopErrorHandler());
  PetscCall(VecDestroy(&early));
  PetscCheck(ierr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Scaling was accepted before setup");
  PetscCall(PCASMGetLocalScaling(pc, NULL, &stored));
  PetscCheck(!stored, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "PCASMGetLocalScaling() reported weights before any were supplied");
  PetscCall(PCSetPostSetUp(pc, InstallScaling));
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetPostSetUp(pc, NULL));
  PetscCall(ConfigureSolvers(pc));
  for (PetscInt pass = 0; pass < 3; ++pass) {
    if (pass == 2) {
      PetscCall(PCReset(pc));
      PetscCall(PCSetOperators(pc, A, A));
      PetscCall(PCASMSetType(pc, PC_ASM_RESTRICT));
      PetscCall(PCASMSetLocalSubdomains(pc, nlocal, is, inner));
      PetscCall(PCSetUp(pc));
      PetscCall(ConfigureSolvers(pc));
      PetscCall(PCASMSetType(pc, PC_ASM_WEIGHTED));
    }
    PetscCall(CreateScaling(pc, 0.25, scaling));
    if (pass) PetscCall(PCASMSetLocalScaling(pc, nlocal, scaling));
    if (pass == 1) {
      Vec updated[2];

      PetscCall(CreateScaling(pc, TestScalar(0.6, 0.2), updated));
      for (PetscInt i = 0; i < nlocal; ++i) {
        PetscCall(VecCopy(updated[i], scaling[i]));
        PetscCall(VecDestroy(&updated[i]));
      }
    }
    PetscCall(CheckInvalidScaling(pc, nlocal, scaling));
    PetscCall(CreateReference(A, pc, scaling, &B));
    for (PetscInt i = 0; i < nlocal; ++i) PetscCall(VecDestroy(&scaling[i]));
    PetscCall(CheckApplications(pc, B, nlocal));
    if (pass == 2) {
      Vec ones[2];
      Mat basic;

      PetscCall(CreateScaling(pc, 0.25, ones));
      for (PetscInt i = 0; i < nlocal; ++i) PetscCall(VecSet(ones[i], 1.0));
      PetscCall(CreateReference(A, pc, ones, &basic));
      for (PetscInt i = 0; i < nlocal; ++i) PetscCall(VecDestroy(&ones[i]));
      PetscCall(PCASMSetType(pc, PC_ASM_BASIC));
      PetscCall(CheckApplications(pc, basic, nlocal));
      PetscCall(PCASMSetType(pc, PC_ASM_WEIGHTED));
      PetscCall(CheckApplications(pc, B, nlocal));
      PetscCall(MatDestroy(&basic));
    }
    PetscCall(MatDestroy(&B));
  }
  for (PetscInt i = 0; i < nlocal; ++i) {
    PetscCall(ISDestroy(&is[i]));
    PetscCall(ISDestroy(&inner[i]));
  }
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: weighted
    nsize: {{1 2 3}}
    args: -pc_asm_type weighted -local_blocks {{1 2}}
    output_file: output/empty.out

TEST*/
