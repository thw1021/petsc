/* TAOFB example. Solves 0.5 |Ax-b|_2^2 + \lambda |x|_1,
 * with A is Gaussian random with size M*N, and b is a measurement vector of size M. */

#include <petsctao.h>
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates TaoFB to solve proximal algorithm. \n";

//ADAPGM example. Nesterov lasso construction

typedef enum {
  PROB_LASSO,
  PROB_LOG_REG
} ProbType;

typedef struct {
  ProbType  probType;
  PetscInt  m, n, k; //A : m x n, k : signal sparsity
  Mat       A;
  Vec       x0, x, workvec, workvec2, workvec3, b, xsub, gsub, workvecM, workvecM2, workvecM3;
  IS        is_set;
  PetscReal scale, optimum, lip;
} AppCtx;

PetscErrorCode Log_UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx        *user  = (AppCtx *)ptr;
  const PetscInt ix[1] = {user->n};
  PetscReal      xlast, gradmean;
  PetscMPIInt    size, rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* workvecM = A @ x[0:end-1] + x[end] */
  PetscCall(VecGetSubVector(X, user->is_set, &user->xsub));
  PetscCall(MatMult(user->A, user->xsub, user->workvecM));
  PetscCall(VecRestoreSubVector(X, user->is_set, &user->xsub));

  if (rank == (size - 1)) PetscCall(VecGetValues(X, 1, ix, &xlast));
  PetscCallMPI(MPI_Bcast(&xlast, 1, MPIU_REAL, size - 1, comm));
  PetscCall(VecShift(user->workvecM, xlast));

  /* workvecM2 = 1 + exp(-workvecM) */
  PetscCall(VecCopy(user->workvecM, user->workvecM2));
  PetscCall(VecScale(user->workvecM2, -1));
  PetscCall(VecExp(user->workvecM2));
  PetscCall(VecShift(user->workvecM2, 1));

  /* f = -avg((b - 1) * workvecM - log(workvecM2)) */
  PetscCall(VecCopy(user->b, user->workvecM3));
  PetscCall(VecShift(user->workvecM3, -1));
  /* workvecM3 = (b-1)*workvecM */
  PetscCall(VecPointwiseMult(user->workvecM3, user->workvecM3, user->workvecM));
  /* overwriting workvecM wih workvecM2, as it is no longer needed */
  PetscCall(VecCopy(user->workvecM2, user->workvecM));
  PetscCall(VecLog(user->workvecM));
  PetscCall(VecAXPY(user->workvecM3, -1., user->workvecM));
  PetscCall(VecMean(user->workvecM3, f));
  *f *= -1;

  /* grad[0:end-1] = A.T @ (1/workvecM2 - y) / user->m,
   * grad[end ]    = avg(1/workvecM2 - y) */
  PetscCall(VecCopy(user->workvecM2, user->workvecM));
  PetscCall(VecReciprocal(user->workvecM));
  PetscCall(VecAXPY(user->workvecM, -1., user->b));
  PetscCall(VecMean(user->workvecM, &gradmean));

  /* grad[0:end-1] */
  PetscCall(VecGetSubVector(G, user->is_set, &user->gsub));
  PetscCall(MatMultTranspose(user->A, user->workvecM, user->gsub));
  PetscCall(VecScale(user->gsub, 1. / ((double)user->m)));
  PetscCall(VecRestoreSubVector(G, user->is_set, &user->gsub));
  /* grad[end] = mean(1/workvcM2 - y) */
  PetscCall(VecSetValue(G, user->n, gradmean, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(G));
  PetscCall(VecAssemblyEnd(G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Log_UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx        *user  = (AppCtx *)ptr;
  const PetscInt ix[1] = {user->n};
  PetscReal      xlast, gradmean;
  PetscMPIInt    size, rank;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* workvecM = A @ x[0:end-1] + x[end] */
  PetscCall(VecGetSubVector(X, user->is_set, &user->xsub));
  PetscCall(MatMult(user->A, user->xsub, user->workvecM));
  PetscCall(VecRestoreSubVector(X, user->is_set, &user->xsub));

  if (rank == (size - 1)) PetscCall(VecGetValues(X, 1, ix, &xlast));
  PetscCallMPI(MPI_Bcast(&xlast, 1, MPIU_REAL, size - 1, comm));
  PetscCall(VecShift(user->workvecM, xlast));

  /* workvecM2 = 1 + exp(-workvecM) */
  PetscCall(VecCopy(user->workvecM, user->workvecM2));
  PetscCall(VecScale(user->workvecM2, -1));
  PetscCall(VecExp(user->workvecM2));
  PetscCall(VecShift(user->workvecM2, 1));

  /* f = -avg((b - 1) * workvecM - log(workvecM2)) */
  PetscCall(VecCopy(user->b, user->workvecM3));
  PetscCall(VecShift(user->workvecM3, -1));
  /* workvecM3 = (b-1)*workvecM */
  PetscCall(VecPointwiseMult(user->workvecM3, user->workvecM3, user->workvecM));
  /* overwriting workvecM wih workvecM2, as it is no longer needed */
  PetscCall(VecCopy(user->workvecM2, user->workvecM));
  PetscCall(VecLog(user->workvecM));
  PetscCall(VecAXPY(user->workvecM3, -1., user->workvecM));
  PetscCall(VecMean(user->workvecM3, f));
  *f *= -1;

  /* grad[0:end-1] = A.T @ (1/workvecM2 - y) / user->m,
   * grad[end ]    = avg(1/workvecM2 - y) */
  PetscCall(VecCopy(user->workvecM2, user->workvecM));
  PetscCall(VecReciprocal(user->workvecM));
  PetscCall(VecAXPY(user->workvecM, -1., user->b));
  PetscCall(VecMean(user->workvecM, &gradmean));

  /* grad[0:end-1] */
  PetscCall(VecGetSubVector(G, user->is_set, &user->gsub));
  PetscCall(MatMultTranspose(user->A, user->workvecM, user->gsub));
  PetscCall(VecScale(user->gsub, 1. / ((double)user->m)));
  PetscCall(VecRestoreSubVector(G, user->is_set, &user->gsub));
  /* grad[end] = mean(1/workvcM2 - y) */
  PetscCall(VecSetValue(G, user->n, gradmean, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(G));
  PetscCall(VecAssemblyEnd(G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 |Ax-b|_2^2
 * grad f = A^T (A x - b)               */
PetscErrorCode UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecAXPY(user->workvec, -1., user->b));
  PetscCall(MatMultTranspose(user->A, user->workvec, G));
  PetscCall(VecTDot(user->workvec, user->workvec, f));
  *f *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 |Ax-b|_2^2
 * grad f = A^T (A x - b)               */
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecAXPY(user->workvec, -1., user->b));
  PetscCall(MatMultTranspose(user->A, user->workvec, G));
  PetscCall(VecTDot(user->workvec, user->workvec, f));
  *f *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscRandom rctx;
  PetscReal   norm, *array, *array2, *array3, temp2, randreal;
  PetscInt    i, *indices, p, temp;
  MPI_Comm    comm;
  PetscMPIInt size, rank;

  PetscFunctionBegin;
  switch (user->probType) {
  case PROB_LASSO:
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Lasso problem only supports size 1 MPI");
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));

    p = PetscCeilInt(user->n, user->k);

    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n, &user->x));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n, &user->x0));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->n, &user->workvec2));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->workvec3));
    PetscCall(VecCreateSeq(PETSC_COMM_WORLD, user->m, &user->b));
    PetscCall(VecSet(user->x0, 0.));

    //workvec: y_star
    PetscCall(VecSetRandom(user->workvec, rctx));
    PetscCall(VecNorm(user->workvec, NORM_2, &norm));
    PetscCall(VecScale(user->workvec, 1 / norm));
    PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD, user->m, user->n, NULL, &user->A));
    PetscCall(MatDenseGetArray(user->A, &array));
    /* Gaussian Matrix, MATLAB equivalent of A = randn(m,n) */
    for (i = 0; i < user->m * user->n; i++) {
      PetscReal a[1] = {0.};

      PetscCall(PetscRandomGetValueReal(rctx, &a[0]));
      PetscCall(PetscPDFSampleGaussian1D(a, NULL, &array[i]));
    }
    PetscCall(MatDenseRestoreArray(user->A, &array));
    PetscCall(MatScale(user->A, 2.));
    PetscCall(MatShift(user->A, -1.));
    PetscCall(MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY));

    PetscCall(MatMultTranspose(user->A, user->workvec, user->workvec2));
    PetscCall(VecAbs(user->workvec2));
    PetscCall(VecCopy(user->workvec2, user->x)); // using x vec as workvec for sortperm
    PetscCall(PetscMalloc1(user->n, &indices));
    for (i = 0; i < user->n; i++) indices[i] = i;

    /* indices: perm, workvec2: CTy, x0: alpha */
    PetscCall(VecGetArray(user->x, &array));
    PetscCall(VecGetArray(user->workvec2, &array2));
    PetscCall(VecGetArray(user->x0, &array3));
    PetscCall(PetscSortRealWithArrayInt(user->n, array, indices)); //in increasing order

    for (i = user->n - 1; i >= 0; i--) {
      temp = indices[i];
      if (i >= user->n - p) {
        array3[temp] = user->scale / array2[temp];
      } else {
        temp2 = array2[temp];
        if (temp2 < 0.1 * user->scale) {
          array3[temp] = user->scale;
        } else {
          PetscCall(PetscRandomGetValueReal(rctx, &randreal));
          array3[temp] = user->scale * randreal / temp2;
        }
      }
    }
    PetscCall(VecRestoreArray(user->x, &array));
    PetscCall(VecRestoreArray(user->workvec2, &array2));
    PetscCall(VecRestoreArray(user->x0, &array3));

    PetscCall(MatDiagonalScale(user->A, NULL, user->x0));

    // generate the primal solution
    // x0 is x_star
    PetscCall(VecSet(user->x0, 0.));
    PetscCall(VecGetArray(user->x0, &array));

    for (i = user->n - 1; i >= 0; i--) {
      if (i >= user->n - p) {
        temp = indices[i];
        PetscCall(PetscRandomGetValueReal(rctx, &randreal));
        PetscCall(MatGetColumnVector(user->A, user->workvec3, temp));
        PetscCall(VecDot(user->workvec3, user->workvec, &norm));
        array[temp] = randreal * PetscSqrtReal((int)p) * PetscSign(norm);
      }
    }
    PetscCall(VecRestoreArray(user->x0, &array));
    PetscCall(MatMultAdd(user->A, user->x0, user->workvec, user->b));
    PetscCall(VecNorm(user->workvec, NORM_2, &norm));
    user->optimum = norm / 2.;

    PetscCall(VecNorm(user->x0, NORM_1, &norm));
    user->optimum += user->scale * norm;

    PetscCall(PetscFree(indices));
    PetscCall(PetscRandomDestroy(&rctx));

    /* Note: technically, one can use (|A|_op)^2 as Lipschitz constant, but we avoid it here */
    user->lip = 0.;
    break;
  case PROB_LOG_REG: {
    PetscViewer viewer;
    PetscInt    low, high;
    char        mat_data[] = "matrix-heart-scale.dat";
    char        vec_data[] = "vector-heart-scale_1_0.dat";

    user->scale = 0.01;

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_data, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &user->A));
    PetscCall(MatSetType(user->A, MATMPIAIJ));
    PetscCall(MatLoad(user->A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(MatGetSize(user->A, &user->m, &user->n));

    PetscCall(MatCreateVecs(user->A, NULL, &user->workvecM));
    PetscCall(MatCreateVecs(user->A, NULL, &user->workvecM2));
    PetscCall(MatCreateVecs(user->A, NULL, &user->workvecM3));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
    PetscCall(VecSetSizes(user->x, PETSC_DECIDE, user->n + 1));
    PetscCall(VecSetFromOptions(user->x));
    PetscCall(VecDuplicate(user->x, &user->workvec));
    PetscCall(VecDuplicate(user->x, &user->workvec2));
    PetscCall(VecSet(user->x, 0));

    /* Lip is computed via
       * A1 = [A; VecOnes(m)];
       * Lip = Norm(A1 @ A1') / 4 / m */
    user->lip = 1.052285051684358;

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, vec_data, FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->b));
    PetscCall(VecLoad(user->b, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscObjectGetComm((PetscObject)user->x, &comm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    PetscCall(MatGetOwnershipRangeColumn(user->A, &low, &high));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, high - low, low, 1, &user->is_set));
  } break;
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->b));
  PetscCall(VecDestroy(&user->workvec));
  PetscCall(VecDestroy(&user->workvec2));
  PetscCall(MatDestroy(&user->A));
  switch (user->probType) {
  case PROB_LASSO:
    PetscCall(VecDestroy(&user->x0));
    PetscCall(VecDestroy(&user->workvec3));
    break;
  case PROB_LOG_REG:
    PetscCall(VecDestroy(&user->workvecM));
    PetscCall(VecDestroy(&user->workvecM2));
    PetscCall(VecDestroy(&user->workvecM3));
    PetscCall(ISDestroy(&user->is_set));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid problem formulation type.");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *probTypes[2] = {"prob_lasso", "prob_log_reg"};

  PetscInt probtype;

  PetscFunctionBegin;
  user->k        = 5;
  user->n        = 20;
  user->m        = 10;
  user->scale    = 1.;
  user->probType = PROB_LASSO;
  probtype       = user->probType;

  PetscOptionsBegin(comm, "", "Forward-backward example", "TAO");
  PetscCall(PetscOptionsEList("-problem", "Decide which problem to solve.", "fb_example.c", probTypes, 2, probTypes[user->probType], &probtype, NULL));

  user->probType = (ProbType)probtype;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-k", &user->k, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &user->m, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-scale", &user->scale, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        fdm, gdm;
  Tao       tao;
  AppCtx    user;
  PetscReal v1, v2;
  PetscInt  dm_idx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCheck(user.n > 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "Problem size needs to be greater than 2.");
  PetscCall(DataCreate(&user));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, user.x));
  PetscCall(TaoSetType(tao, TAOFB));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &fdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &gdm));
  PetscCall(DMTaoSetType(gdm, DMTAOL1));
  PetscCall(DMTaoL1SetContext(gdm, user.scale));

  switch (user.probType) {
  case PROB_LASSO: {
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, UserObjGrad_DM, (void *)&user));
    PetscCall(DMTaoSetLipschitz(fdm, user.lip));
    PetscCall(TaoAddDM(tao, fdm, 1.));
    PetscCall(TaoPSSetSmoothTerm(tao, dm_idx));
    dm_idx++;
  } break;
  case PROB_LOG_REG: {
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, Log_UserObjGrad_DM, (void *)&user));
    PetscCall(DMTaoSetLipschitz(fdm, user.lip));
    PetscCall(TaoAddDM(tao, fdm, 1.));
    PetscCall(TaoPSSetSmoothTerm(tao, dm_idx));
    dm_idx++;
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  PetscCall(TaoAddDM(tao, gdm, 1.));
  PetscCall(TaoPSSetNonSmoothTerm(tao, dm_idx));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* compute the error */
  switch (user.probType) {
  case PROB_LASSO:
    PetscCall(VecNorm(user.x0, NORM_2, &v2));
    PetscCall(VecAXPY(user.x0, -1, user.x));
    PetscCall(VecNorm(user.x0, NORM_2, &v1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1 / v2)));
    break;
  case PROB_LOG_REG:
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem formulation type.");
  }

  PetscCall(DataDestroy(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(DMDestroy(&fdm));
  PetscCall(DMDestroy(&gdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: lasso_no_ls
      args: -problem prob_lasso -tao_fb_accel 0 -tao_fb_adaptive 0 -scale 10 -tao_ls_max_funcs 0 -tao_max_it 1000
      output_file: output/fb_example_lasso_no_ls.out
      requires: !single

   test:
      suffix: lasso_ls
      args: -problem prob_lasso -tao_fb_accel 0 -tao_fb_adaptive 0 -scale 10 -tao_ls_max_funcs 30 -tao_fb_ls_scale 1.05 -tao_max_it 1000
      output_file: output/fb_example_lasso_ls.out
      requires: !single

   test:
      suffix: lasso_non_mon_ls
      args: -problem prob_lasso -tao_fb_accel 0 -tao_fb_adaptive 0 -scale 10 -tao_ls_max_funcs 30 -tao_ls_PS_memory_size 5 -tao_fb_ls_scale 1.05 -tao_max_it 1000
      output_file: output/fb_example_lasso_non_mon_ls.out
      requires: !single

   test:
      suffix: lasso_fista
      args: -problem prob_lasso -tao_fb_accel 1 -tao_fb_adaptive 0 -scale 10 -tao_max_it 1000
      output_file: output/fb_example_lasso_fista.out
      requires: !single

   test:
      suffix: lasso_ada
      args: -problem prob_lasso -tao_fb_accel 0 -tao_fb_adaptive 1 -scale 10 -tao_max_it 2000
      output_file: output/fb_example_lasso_ada.out
      requires: !single

   test:
      suffix: logreg_fista
      nsize: {{1 2}}
      localrunfiles: matrix-heart-scale.dat vector-heart-scale_1_0.dat
      args: -problem prob_log_reg -scale 0.01 -tao_fb_accel 1 -tao_fb_adaptive 0 -tao_converged_reason -tao_max_it 2000
      output_file: output/fb_example_logreg_fista.out
      requires: !single

   test:
      suffix: logreg_ada
      nsize: {{1 2 4}}
      localrunfiles: matrix-heart-scale.dat vector-heart-scale_1_0.dat
      args: -problem prob_log_reg -scale 0.01 -tao_fb_accel 0 -tao_fb_adaptive 1 -tao_max_it 1000 -tao_converged_reason -tao_monitor
      output_file: output/fb_example_logreg_ada.out
      requires: !single

TEST*/
