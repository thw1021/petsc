// Common data structures and rosenbrock1.c and rosenbrock1_taoterm.c

#include <petsctao.h>
/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/
typedef struct {
  MPI_Comm  comm;
  PetscInt  n;     /* dimension */
  PetscReal alpha; /* condition parameter */
  PetscBool chained;
} AppCtx;

static PetscErrorCode AppCtxCreateHessianMatrices(AppCtx *usr, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF, 2, usr->n, usr->n, 1, NULL, H));
  if (Hpre) *Hpre = *H;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  AppCtxFormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ user - AppCtx
- X    - input vector

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient
*/
static PetscErrorCode AppCtxFormFunctionGradient(AppCtx *user, Vec X, PetscReal *f, Vec G)
{
  PetscInt           nn = user->n / 2;
  PetscReal          ff = 0, t1, t2, alpha = user->alpha;
  PetscScalar       *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(G, &g));

  /* Compute G(X) */
  if (user->chained) {
    g[0] = 0;
    for (PetscInt i = 0; i < user->n - 1; i++) {
      t1 = x[i + 1] - x[i] * x[i];
      ff += PetscSqr(1 - x[i]) + alpha * t1 * t1;
      g[i] += -2 * (1 - x[i]) + 2 * alpha * t1 * (-2 * x[i]);
      g[i + 1] = 2 * alpha * t1;
    }
  } else {
    for (PetscInt i = 0; i < nn; i++) {
      t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      t2 = 1 - x[2 * i];
      ff += alpha * t1 * t1 + t2 * t2;
      g[2 * i]     = -4 * alpha * t1 * x[2 * i] - 2.0 * t2;
      g[2 * i + 1] = 2 * alpha * t1;
    }
  }

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(G, &g));
  *f = ff;

  PetscCall(PetscLogFlops(15.0 * nn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ tao   - the Tao context
- x     - input vector

  Output Parameters:
+ H     - Hessian matrix
- Hpre  - Hessian preconditioner
*/
PetscErrorCode AppCtxFormHessian(AppCtx *user, Vec X, Mat H, Mat Hpre)
{
  PetscInt           ind[2];
  PetscReal          alpha = user->alpha;
  PetscReal          v[2][2];
  const PetscScalar *x;
  PetscBool          assembled;

  PetscFunctionBeginUser;
  /* Zero existing matrix entries */
  PetscCall(MatAssembled(H, &assembled));
  if (assembled) PetscCall(MatZeroEntries(H));

  /* Get a pointer to vector data */
  PetscCall(VecGetArrayRead(X, &x));

  /* Compute H(X) entries */
  if (user->chained) {
    PetscCall(MatZeroEntries(H));
    for (PetscInt i = 0; i < user->n - 1; i++) {
      PetscScalar t1 = x[i + 1] - x[i] * x[i];
      v[0][0]        = 2 + 2 * alpha * (t1 * (-2) - 2 * x[i]);
      v[0][1]        = 2 * alpha * (-2 * x[i]);
      v[1][0]        = 2 * alpha * (-2 * x[i]);
      v[1][1]        = 2 * alpha * t1;
      ind[0]         = i;
      ind[1]         = i + 1;
      PetscCall(MatSetValues(H, 2, ind, 2, ind, v[0], ADD_VALUES));
    }
  } else {
    for (PetscInt i = 0; i < user->n / 2; i++) {
      v[1][1] = 2 * alpha;
      v[0][0] = -4 * alpha * (x[2 * i + 1] - 3 * x[2 * i] * x[2 * i]) + 2;
      v[1][0] = v[0][1] = -4.0 * alpha * x[2 * i];
      ind[0]            = 2 * i;
      ind[1]            = 2 * i + 1;
      PetscCall(MatSetValues(H, 2, ind, 2, ind, v[0], INSERT_VALUES));
    }
  }
  PetscCall(VecRestoreArrayRead(X, &x));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogFlops(9.0 * user->n / 2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
