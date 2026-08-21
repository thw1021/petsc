static char help[] = "Regression test for the MatInvertBlockDiagonal() cached-inverse invalidation.\n\n\
   MatInvertBlockDiagonal() caches the inverted diagonal blocks inside the matrix.  If that\n\
   cache is not invalidated when the matrix is re-assembled with new values but an unchanged\n\
   nonzero structure, the routine hands back the previous values' inverse.\n\
   PCSetUp_PBJacobi_Host() obtains its blocks from that call, so PCPBJACOBI then preconditions\n\
   with stale blocks -- and with -ksp_type preonly on a block-diagonal operator, where the\n\
   preconditioner is the exact solve, KSPSolve() returns a silently wrong answer.\n\
   MATAIJ (with a block size set) and MATBAIJ hold separate caches, so both are checked, both\n\
   directly and through KSP.  The matrix is nb blocks of d*I, whose exact block inverse is I/d;\n\
   pass 0 uses d = 1 and pass 1 re-assembles the same sparsity with d = 2.\n\n";

#include <petscksp.h>

static const PetscInt bs = 2, nb = 4;

static PetscErrorCode MatCreateBlockDiagonal(MatType type, Mat *A)
{
  PetscInt  j;
  PetscInt *dnnz;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, bs * nb, bs * nb, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*A, type));
  PetscCall(MatSetBlockSize(*A, bs));
  PetscCall(PetscMalloc1(nb, &dnnz));
  for (j = 0; j < nb; j++) dnnz[j] = 1;
  PetscCall(MatXAIJSetPreallocation(*A, bs, dnnz, NULL, NULL, NULL));
  PetscCall(PetscFree(dnnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Fill every diagonal block with d*I and assemble. Called twice, so the second call is the
   "new values, unchanged nonzero structure" re-assembly that must invalidate the cache. */
static PetscErrorCode MatAssembleBlockDiagonal(Mat A, PetscScalar d)
{
  PetscInt    b, rstart, rend;
  PetscScalar blk[4] = {0.0, 0.0, 0.0, 0.0};

  PetscFunctionBeginUser;
  blk[0] = d;
  blk[3] = d;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (b = rstart / bs; b < rend / bs; b++) PetscCall(MatSetValuesBlocked(A, 1, &b, 1, &b, blk, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckInverse(MatType type, PetscBool *ok)
{
  Mat                A;
  const PetscScalar *inv;
  PetscInt           b, i, j, pass;
  PetscScalar        d;
  PetscReal          err, want;

  PetscFunctionBeginUser;
  PetscCall(MatCreateBlockDiagonal(type, &A));
  for (pass = 0; pass < 2; pass++) {
    d    = (pass == 0) ? 1.0 : 2.0;
    want = 1.0 / PetscRealPart(d);
    PetscCall(MatAssembleBlockDiagonal(A, d));
    PetscCall(MatInvertBlockDiagonal(A, &inv));
    /* every local block must be the exact inverse of d*I, that is I/d */
    err = 0.0;
    for (b = 0; b < nb; b++) {
      for (i = 0; i < bs; i++) {
        for (j = 0; j < bs; j++) err = PetscMax(err, PetscAbsScalar(inv[b * bs * bs + i * bs + j] - (i == j ? want : 0.0)));
      }
    }
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &err, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
    if (err > PETSC_SMALL) *ok = PETSC_FALSE;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %-8s pass %" PetscInt_FMT ": MatInvertBlockDiagonal()[0] = %-6g (exact %g) %s\n", type, pass, (double)PetscRealPart(inv[0]), (double)want, err > PETSC_SMALL ? "STALE" : "ok"));
  }
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The user-visible consequence: KSP(preonly) + PCPBJACOBI on a block-diagonal matrix is an
   exact solve, so x must equal inv(A)*b. Re-assemble A with new values, solve again, and the
   answer is silently wrong if the preconditioner reused the previous blocks. */
static PetscErrorCode CheckSolve(MatType type, PetscBool *ok)
{
  Mat         A;
  Vec         b, x;
  KSP         ksp;
  PC          pc;
  PetscInt    pass;
  PetscScalar d;
  PetscReal   err, got, want;

  PetscFunctionBeginUser;
  PetscCall(MatCreateBlockDiagonal(type, &A));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPBJACOBI));
  for (pass = 0; pass < 2; pass++) {
    d    = (pass == 0) ? 1.0 : 2.0;
    want = 1.0 / PetscRealPart(d);
    PetscCall(MatAssembleBlockDiagonal(A, d));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    /* b is all ones and A is d*I, so every entry of the exact solution is 1/d */
    PetscCall(VecMax(x, NULL, &got));
    PetscCall(VecShift(x, -want));
    PetscCall(VecNorm(x, NORM_INFINITY, &err));
    if (err > PETSC_SMALL) *ok = PETSC_FALSE;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %-8s pass %" PetscInt_FMT ": KSPSolve() x[0] = %-6g (exact %g) %s\n", type, pass, (double)got, (double)want, err > PETSC_SMALL ? "WRONG ANSWER" : "ok"));
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscBool ok = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatInvertBlockDiagonal() after re-assembly with unchanged sparsity:\n"));
  PetscCall(CheckInverse(MATAIJ, &ok));
  PetscCall(CheckInverse(MATBAIJ, &ok));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "KSP(preonly) + PCPBJACOBI on the same block-diagonal matrix:\n"));
  PetscCall(CheckSolve(MATAIJ, &ok));
  PetscCall(CheckSolve(MATBAIJ, &ok));
  PetscCheck(ok, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatInvertBlockDiagonal() returned a stale cached inverse after the matrix was re-assembled with new values and an unchanged nonzero structure");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     nsize: 1

   test:
     suffix: 2
     nsize: 2
     output_file: output/ex311_1.out

TEST*/
