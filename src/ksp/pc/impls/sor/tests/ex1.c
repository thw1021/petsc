static char help[] = "Tests that PCApplyTranspose_SOR()'s flag swaps (apply upper/lower, forward/backward,\n\
and local forward/backward sweeps) are genuine transposes of each other for a symmetric matrix, both\n\
directly at the MatSOR() level and through PCSOR's PCApply()/PCApplyTranspose().\n\n";

#include <petscksp.h>

typedef PetscErrorCode (*ApplyFn)(void *, Vec, Vec);

typedef struct {
  Mat        A;
  MatSORType flag;
} SORCtx;

typedef struct {
  PC        pc;
  PetscBool transpose;
} PCCtx;

static PetscErrorCode CreateSymmetricAIJ(MPI_Comm comm, PetscInt n, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(*A, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*A, 5, NULL));
  for (PetscInt i = 0; i < n; i++) {
    PetscScalar diag = 4.0 + i;

    PetscCall(MatSetValue(*A, i, i, diag, INSERT_VALUES));
    if (i > 0) {
      PetscScalar off = -1.0 - 0.3 * i;

      PetscCall(MatSetValue(*A, i, i - 1, off, INSERT_VALUES));
      PetscCall(MatSetValue(*A, i - 1, i, off, INSERT_VALUES));
    }
    if (i > 1) {
      PetscCall(MatSetValue(*A, i, i - 2, 0.2, INSERT_VALUES));
      PetscCall(MatSetValue(*A, i - 2, i, 0.2, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SOR_APPLY_UPPER/SOR_APPLY_LOWER are pure multiplies, unaffected by y's leftover contents, but the
   sweep flags solve using y as an initial guess unless told otherwise: force a zero initial guess so
   that repeated calls (one per column, reusing y) each produce the single-sweep linear operator. */
static PetscErrorCode ApplyMatSOR(void *actx, Vec e, Vec y)
{
  SORCtx    *ctx  = (SORCtx *)actx;
  MatSORType flag = ctx->flag;

  PetscFunctionBegin;
  if (!(flag & (SOR_APPLY_UPPER | SOR_APPLY_LOWER))) flag = (MatSORType)(flag | SOR_ZERO_INITIAL_GUESS);
  PetscCall(MatSOR(ctx->A, e, 1.0, flag, 0.0, 1, 1, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplyPC(void *actx, Vec e, Vec y)
{
  PCCtx *ctx = (PCCtx *)actx;

  PetscFunctionBegin;
  if (ctx->transpose) PetscCall(PCApplyTranspose(ctx->pc, e, y));
  else PetscCall(PCApply(ctx->pc, e, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Explicitly forms the dense operator y = apply(ctx, x) column by column. */
static PetscErrorCode ComputeDenseOperator(MPI_Comm comm, Mat A, ApplyFn apply, void *ctx, Mat *Op)
{
  Vec      e, y;
  PetscInt n;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatCreateSeqDense(comm, n, n, NULL, Op));
  PetscCall(MatCreateVecs(A, &e, &y));
  for (PetscInt j = 0; j < n; j++) {
    const PetscScalar *yarr;
    PetscScalar       *col;

    PetscCall(VecZeroEntries(e));
    PetscCall(VecSetValue(e, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(e));
    PetscCall(VecAssemblyEnd(e));
    PetscCall(apply(ctx, e, y));
    PetscCall(MatDenseGetColumn(*Op, j, &col));
    PetscCall(VecGetArrayRead(y, &yarr));
    PetscCall(PetscArraycpy(col, yarr, n));
    PetscCall(VecRestoreArrayRead(y, &yarr));
    PetscCall(MatDenseRestoreColumn(*Op, &col));
  }
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMatEqual(Mat X, Mat Y, const char msg[])
{
  Mat       diff;
  PetscReal err, scale;
  PetscInt  m, n;

  PetscFunctionBegin;
  PetscCall(MatGetSize(X, &m, &n));
  PetscCall(MatDuplicate(X, MAT_COPY_VALUES, &diff));
  PetscCall(MatAXPY(diff, -1.0, Y, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(diff, NORM_FROBENIUS, &err));
  PetscCall(MatNorm(X, NORM_FROBENIUS, &scale));
  PetscCheck(err < PETSC_SMALL * m * n * PetscMax(scale, 1.0), PetscObjectComm((PetscObject)X), PETSC_ERR_PLIB, "%s: mismatch of norm %g (scale %g)", msg, (double)err, (double)scale);
  PetscCall(MatDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatSOR(sym) and MatSOR(tsym) are, for a symmetric A, exact transposes of each other -- the same
   pairing PCApplyTranspose_SOR() swaps between (apply upper/lower, forward/backward, local
   forward/backward). This identity is specific to symmetric A; PCApplyTranspose_SOR() does not check
   symmetry but relies on it, just as the pre-existing forward/backward swap already did. */
static PetscErrorCode TestMatSOR(MPI_Comm comm, Mat A, MatSORType sym, MatSORType tsym, const char symname[])
{
  SORCtx sctx = {A, sym};
  SORCtx tctx = {A, tsym};
  Mat    S, T, Ttranspose;

  PetscFunctionBegin;
  PetscCall(ComputeDenseOperator(comm, A, ApplyMatSOR, &sctx, &S));
  PetscCall(ComputeDenseOperator(comm, A, ApplyMatSOR, &tctx, &T));
  PetscCall(MatTranspose(T, MAT_INITIAL_MATRIX, &Ttranspose));
  PetscCall(CheckMatEqual(S, Ttranspose, "MatSOR(sym) is not the transpose of MatSOR(tsym)"));
  PetscCall(PetscPrintf(comm, "MatSOR sym=%-16s transpose matches MatSOR(swapped flag)  : PASS\n", symname));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&T));
  PetscCall(MatDestroy(&Ttranspose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* For PCSOR configured with sym, checks that PCApply() matches the direct MatSOR() of the same flag,
   and that PCApplyTranspose() matches the direct MatSOR() of tsym -- the flag PCApplyTranspose_SOR()
   is expected to swap to. */
static PetscErrorCode TestPCSOR(MPI_Comm comm, Mat A, MatSORType sym, MatSORType tsym, const char symname[])
{
  PC     pc;
  PCCtx  actx, tctx;
  SORCtx dctx, tdctx;
  Mat    direct, transposeDirect, applied, appliedTranspose;

  PetscFunctionBegin;
  PetscCall(PCCreate(comm, &pc));
  PetscCall(PCSetType(pc, PCSOR));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSORSetSymmetric(pc, sym));
  PetscCall(PCSetUp(pc));

  actx.pc        = pc;
  actx.transpose = PETSC_FALSE;
  tctx.pc        = pc;
  tctx.transpose = PETSC_TRUE;
  dctx.A         = A;
  dctx.flag      = sym;
  tdctx.A        = A;
  tdctx.flag     = tsym;

  PetscCall(ComputeDenseOperator(comm, A, ApplyPC, &actx, &applied));
  PetscCall(ComputeDenseOperator(comm, A, ApplyPC, &tctx, &appliedTranspose));
  PetscCall(ComputeDenseOperator(comm, A, ApplyMatSOR, &dctx, &direct));
  PetscCall(ComputeDenseOperator(comm, A, ApplyMatSOR, &tdctx, &transposeDirect));

  PetscCall(CheckMatEqual(applied, direct, "PCApply() does not match direct MatSOR() of the same flag"));
  PetscCall(CheckMatEqual(appliedTranspose, transposeDirect, "PCApplyTranspose() does not match direct MatSOR() of the swapped flag"));

  PetscCall(PetscPrintf(comm, "PCSOR sym=%-16s PCApply/PCApplyTranspose match MatSOR: PASS\n", symname));

  PetscCall(MatDestroy(&applied));
  PetscCall(MatDestroy(&appliedTranspose));
  PetscCall(MatDestroy(&direct));
  PetscCall(MatDestroy(&transposeDirect));
  PetscCall(PCDestroy(&pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;
  Mat      A;
  PetscInt n = 7;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_SELF;

  PetscCall(CreateSymmetricAIJ(comm, n, &A));

  PetscCall(TestMatSOR(comm, A, SOR_APPLY_UPPER, SOR_APPLY_LOWER, "apply_upper"));
  PetscCall(TestMatSOR(comm, A, SOR_FORWARD_SWEEP, SOR_BACKWARD_SWEEP, "forward"));
  PetscCall(TestMatSOR(comm, A, SOR_LOCAL_FORWARD_SWEEP, SOR_LOCAL_BACKWARD_SWEEP, "local_forward"));

  PetscCall(TestPCSOR(comm, A, SOR_APPLY_UPPER, SOR_APPLY_LOWER, "apply_upper"));
  PetscCall(TestPCSOR(comm, A, SOR_APPLY_LOWER, SOR_APPLY_UPPER, "apply_lower"));
  PetscCall(TestPCSOR(comm, A, SOR_FORWARD_SWEEP, SOR_BACKWARD_SWEEP, "forward"));
  PetscCall(TestPCSOR(comm, A, SOR_BACKWARD_SWEEP, SOR_FORWARD_SWEEP, "backward"));
  PetscCall(TestPCSOR(comm, A, SOR_LOCAL_FORWARD_SWEEP, SOR_LOCAL_BACKWARD_SWEEP, "local_forward"));
  PetscCall(TestPCSOR(comm, A, SOR_LOCAL_BACKWARD_SWEEP, SOR_LOCAL_FORWARD_SWEEP, "local_backward"));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 0
     requires: !single

TEST*/
