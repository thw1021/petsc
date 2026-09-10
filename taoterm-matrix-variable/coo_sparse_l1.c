/*
  min_x  0.5 || X(x) v - b ||_2^2 + lambda ||x||_1

  X(x): M x N sparse matrix with a FIXED nonzero pattern (coo_i, coo_j) and values x.

  The unknown is the Vec x of nonzero values. Its local length is ncoo, the number of COO
  entries this rank supplies, in the order they were given to MatSetPreallocationCOO().
  X is refilled from x with one MatSetValuesCOO() call per evaluation: no reassembly, and
  any rank may supply any entry because PETSc routes remote entries to their owner.

  Gradient: X(x) v is linear in x, with d f / d x_k = r[i_k] * v[j_k] where r = X v - b.
  v[j_k] is gathered once (v is data); r[i_k] is gathered on every call. If each rank
  supplied only entries in rows it owns, the r-gather would be a local permutation.
*/

#include <petsctao.h>

static char help[] = "Least squares in the nonzero values of a fixed-pattern sparse matrix, with L1 regularization.\n";

typedef struct {
  Mat        X;
  Vec        v, b, r;   /* data and residual */
  Vec        ri, vj;    /* ri[k] = r[coo_i[k]], vj[k] = v[coo_j[k]]; sequential, length ncoo */
  VecScatter gather_r, gather_v;
  PetscInt   ncoo;
} AppCtx;

static PetscErrorCode ObjGrad(TaoTerm term, Vec x, Vec params, PetscReal *f, Vec g)
{
  AppCtx            *user;
  const PetscScalar *xa, *ria, *vja;
  PetscScalar       *ga;
  PetscReal          nrm;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(VecGetArrayRead(x, &xa));
  PetscCall(MatSetValuesCOO(user->X, xa, INSERT_VALUES)); /* X <- x; pattern fixed, assembly handled inside */
  PetscCall(VecRestoreArrayRead(x, &xa));
  PetscCall(MatMult(user->X, user->v, user->r)); /* r = X v - b */
  PetscCall(VecAXPY(user->r, -1.0, user->b));
  PetscCall(VecNorm(user->r, NORM_2, &nrm));
  *f = 0.5 * nrm * nrm;
  PetscCall(VecScatterBegin(user->gather_r, user->r, user->ri, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(user->gather_r, user->r, user->ri, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(user->ri, &ria));
  PetscCall(VecGetArrayRead(user->vj, &vja));
  PetscCall(VecGetArrayWrite(g, &ga));
  for (PetscInt k = 0; k < user->ncoo; k++) ga[k] = ria[k] * vja[k];
  PetscCall(VecRestoreArrayWrite(g, &ga));
  PetscCall(VecRestoreArrayRead(user->vj, &vja));
  PetscCall(VecRestoreArrayRead(user->ri, &ria));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Central-difference check of the shell term's gradient along a random direction; run with -check_gradient */
static PetscErrorCode CheckGradient(TaoTerm term, Vec x)
{
  Vec         x0, d, g;
  PetscRandom rctx;
  PetscScalar gd;
  PetscReal   f0, fp, fm, fd, h = 1.e-5;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(x, &x0));
  PetscCall(VecDuplicate(x, &d));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)x), &rctx));
  PetscCall(PetscRandomSetInterval(rctx, -1.0, 1.0));
  PetscCall(VecSetRandom(x0, rctx));
  PetscCall(VecSetRandom(d, rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x0, NULL, &f0, g));
  PetscCall(VecDot(g, d, &gd));
  PetscCall(VecAXPY(x0, h, d));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x0, NULL, &fp, g));
  PetscCall(VecAXPY(x0, -2.0 * h, d));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x0, NULL, &fm, g));
  fd = (fp - fm) / (2.0 * h);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)x), "gradient check: g.d = %g, central difference = %g, relative difference = %g\n", (double)PetscRealPart(gd), (double)fd, (double)(PetscAbsReal(PetscRealPart(gd) - fd) / PetscMax(PetscAbsReal(fd), 1.e-12))));
  PetscCall(VecDestroy(&x0));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx             user;
  Tao                tao;
  TaoTerm            fterm, gterm;
  Vec                x, xtrue;
  IS                 is_i, is_j;
  PetscInt          *coo_i, *coo_j;
  PetscScalar       *va, *xt;
  PetscReal          lambda = 1.e-3, f, lip, lip_local, rnorm;
  PetscInt           M = 20, N = 12, rlo, rhi, its, vlo, vhi;
  PetscMPIInt        rank, size;
  TaoConvergedReason reason;
  PetscBool          check = PETSC_FALSE, no_reg = PETSC_FALSE;
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &lambda, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_gradient", &check, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_reg", &no_reg, NULL)); /* smooth term only, for use with classical Tao solvers */
  PetscCheck(N >= 3, comm, PETSC_ERR_USER, "Need N >= 3");

  /* This rank supplies three entries for each row in [rlo, rhi). The split is deliberately
     the naive one, which differs from PETSc's row ownership when M is not divisible by size,
     so some entries are remote and get routed. */
  rlo       = (M * rank) / size;
  rhi       = (M * (rank + 1)) / size;
  user.ncoo = 3 * (rhi - rlo);
  PetscCall(PetscMalloc2(user.ncoo, &coo_i, user.ncoo, &coo_j));
  for (PetscInt i = rlo, k = 0; i < rhi; i++) {
    for (PetscInt c = 0; c < 3; c++, k++) {
      coo_i[k] = i;
      coo_j[k] = (i + c) % N;
    }
  }
  /* copies first: MatSetPreallocationCOO() may scramble its index arrays */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, user.ncoo, coo_i, PETSC_COPY_VALUES, &is_i));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, user.ncoo, coo_j, PETSC_COPY_VALUES, &is_j));

  PetscCall(MatCreate(comm, &user.X));
  PetscCall(MatSetSizes(user.X, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetFromOptions(user.X));
  PetscCall(MatSetPreallocationCOO(user.X, user.ncoo, coo_i, coo_j)); /* pattern fixed here, once */
  PetscCall(PetscFree2(coo_i, coo_j));
  PetscCall(MatCreateVecs(user.X, &user.v, &user.r));
  PetscCall(VecDuplicate(user.r, &user.b));

  /* v: fixed data */
  PetscCall(VecGetOwnershipRange(user.v, &vlo, &vhi));
  PetscCall(VecGetArrayWrite(user.v, &va));
  for (PetscInt j = vlo; j < vhi; j++) va[j - vlo] = 0.5 * (PetscScalar)((j % 5) - 2) + 0.25;
  PetscCall(VecRestoreArrayWrite(user.v, &va));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ncoo, &user.ri));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user.ncoo, &user.vj));
  PetscCall(VecScatterCreate(user.r, is_i, user.ri, NULL, &user.gather_r));
  PetscCall(VecScatterCreate(user.v, is_j, user.vj, NULL, &user.gather_v));
  PetscCall(VecScatterBegin(user.gather_v, user.v, user.vj, INSERT_VALUES, SCATTER_FORWARD)); /* v is data: once */
  PetscCall(VecScatterEnd(user.gather_v, user.v, user.vj, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(ISDestroy(&is_i));
  PetscCall(ISDestroy(&is_j));

  /* the unknown as Tao sees it: local entries are this rank's COO values, in COO order */
  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, user.ncoo, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(x));

  /* x_true and b = X(x_true) v */
  PetscCall(VecDuplicate(x, &xtrue));
  PetscCall(VecGetArrayWrite(xtrue, &xt));
  for (PetscInt i = rlo, k = 0; i < rhi; i++) {
    for (PetscInt c = 0; c < 3; c++, k++) xt[k] = (PetscScalar)((i % 4) + 1) * (c == 0 ? 1.0 : (c == 1 ? -0.5 : 0.25));
  }
  PetscCall(VecRestoreArrayWrite(xtrue, &xt));
  PetscCall(VecGetArrayRead(xtrue, (const PetscScalar **)&xt));
  PetscCall(MatSetValuesCOO(user.X, xt, INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(xtrue, (const PetscScalar **)&xt));
  PetscCall(MatMult(user.X, user.v, user.b));
  PetscCall(VecSet(x, 0.0));

  /* Lipschitz bound: X(x) v = K x with K[i_k, k] = v[j_k], so ||K||_2^2 <= ||K||_F^2 = sum_k v[j_k]^2 */
  PetscCall(VecDot(user.vj, user.vj, &lip_local));
  PetscCallMPI(MPIU_Allreduce(&lip_local, &lip, 1, MPIU_REAL, MPIU_SUM, comm));

  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetType(tao, TAOFB));

  PetscCall(TaoTermCreate(comm, &fterm));
  PetscCall(TaoTermSetType(fterm, TAOTERMSHELL));
  PetscCall(TaoTermSetParametersMode(fterm, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetContext(fterm, &user));
  PetscCall(TaoTermShellSetObjectiveAndGradient(fterm, ObjGrad));
  PetscCall(TaoTermSetLipschitz(fterm, lip));
  PetscCall(TaoTermSetSolutionTemplate(fterm, x));

  PetscCall(TaoTermCreate(comm, &gterm));
  PetscCall(TaoTermSetType(gterm, TAOTERML1));
  PetscCall(TaoTermSetSolutionTemplate(gterm, x));

  PetscCall(TaoAddTerm(tao, "f_", 1.0, fterm, NULL, NULL));
  if (!no_reg) PetscCall(TaoAddTerm(tao, "g_", lambda, gterm, NULL, NULL));
  PetscCall(TaoSetFromOptions(tao));
  if (check) {
    PetscCall(TaoSetUp(tao));
    PetscCall(CheckGradient(fterm, x));
  }
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolutionStatus(tao, &its, &f, NULL, NULL, NULL, &reason));

  /* residual at the solution, through the same refill path */
  PetscCall(VecGetArrayRead(x, (const PetscScalar **)&xt));
  PetscCall(MatSetValuesCOO(user.X, xt, INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(x, (const PetscScalar **)&xt));
  PetscCall(MatMult(user.X, user.v, user.r));
  PetscCall(VecAXPY(user.r, -1.0, user.b));
  PetscCall(VecNorm(user.r, NORM_2, &rnorm));
  PetscCall(PetscPrintf(comm, "iterations %" PetscInt_FMT ", reason %s, objective %g, ||X(x) v - b||_2 = %g\n", its, TaoConvergedReasons[reason], (double)f, (double)rnorm));

  PetscCall(TaoTermDestroy(&fterm));
  PetscCall(TaoTermDestroy(&gterm));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xtrue));
  PetscCall(VecScatterDestroy(&user.gather_r));
  PetscCall(VecScatterDestroy(&user.gather_v));
  PetscCall(VecDestroy(&user.ri));
  PetscCall(VecDestroy(&user.vj));
  PetscCall(VecDestroy(&user.v));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.r));
  PetscCall(MatDestroy(&user.X));
  PetscCall(PetscFinalize());
  return 0;
}
