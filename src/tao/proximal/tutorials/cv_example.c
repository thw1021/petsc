/* TAOCV example. Solves dual SVM problem, or least absolute deviation (LAD),
 *
 * dual svm:
 * minimize_{a_1,...a_N} 0.5*|\sum a_i g_i d_i |_2^2 - \sum a_i
 * s.t. 0 \leq a_i \leq C,  \sum a_i g_i = 0
 *
 * -> f(x) + g(x) + h(Ax),
 *  where f(x) = 0.5 x^T Q x + x^T q
 *        g(x) : box constraint indicator function
 *        h(x) : zero cone indicator function
 *
 * least absolute deviation:
 *
 * min_x |Ax - b|_1 + scale*|x|_1
 *
 * -> f(x) + g(x) + h(Ax),
 *  where f(x) = Zero(),
 *        g(x) = |x|_1
 *        h(x) = |\cdot - b|_1        */

#include <petsctao.h>
#include <petscdm.h>
#include <petscksp.h>
#include <petscmat.h>

static char help[] = "This example demonstrates TaoCV to solve proximal primal-dual algorithm. \n";

typedef enum {
  DUAL_SVM,
  LAD, //Least absolute deviation
} ProbType;

typedef struct {
  ProbType  probType;
  PetscInt  m, n;
  Mat       Q, A;
  Vec       x0, x, workvec, workvec2, workvec3, q, y_translation;
  PetscReal C, lip, matnorm, g_scale;
  PetscBool set_norm;
} AppCtx;

PetscErrorCode LAD_UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscFunctionBegin;
  *f = 0;
  PetscCall(VecSet(G, 0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Least absolute deviation
 * f(x) = zero()              */
PetscErrorCode LAD_UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  PetscFunctionBegin;
  *f = 0;
  PetscCall(VecSet(G, 0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x^T Q x + x^T b
 * grad f = Qx + b                */
PetscErrorCode SVM_UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal temp1, temp2;

  PetscFunctionBegin;
  PetscCall(MatMult(user->Q, X, G));
  PetscCall(VecTDot(G, X, &temp1));
  PetscCall(VecTDot(X, user->q, &temp2));
  PetscCall(VecAXPY(G, +1., user->q));
  *f = 0.5 * temp1 + temp2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x^T Q x + x^T b
 * grad f = Qx + b                */
PetscErrorCode SVM_UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx   *user = (AppCtx *)ptr;
  PetscReal temp1, temp2;

  PetscFunctionBegin;
  PetscCall(MatMult(user->Q, X, G));
  PetscCall(VecTDot(G, X, &temp1));
  PetscCall(VecTDot(X, user->q, &temp2));
  PetscCall(VecAXPY(G, +1., user->q));
  *f = 0.5 * temp1 + temp2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CompareSolution(AppCtx *user)
{
  Vec       sol;
  PetscReal v1, v2;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(user->x, &sol));
  switch (user->probType) {
  case DUAL_SVM:
    break;
  case LAD:
    PetscCall(VecSetValue(sol, 0, -4.3283317776698340, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 1, 0.1883780045924437, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 2, 0.0, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 3, 0.5622123045061072, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 4, -1.6734195047896858, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 5, 12.790103687505056, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 6, -0.7053336912159722, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 7, -3.0384547478121746, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 8, 0.0, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 9, -0.42888657254277773, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 10, -3.196986857899165, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 11, 2.2476106964899394, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 12, -6.2628282167848885, INSERT_VALUES));
    PetscCall(VecSetValue(sol, 13, 12.085848625113325, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(sol));
    PetscCall(VecAssemblyEnd(sol));
    PetscCall(VecNorm(sol, NORM_2, &v2));
    PetscCall(VecAXPY(sol, -1., user->x));
    PetscCall(VecNorm(sol, NORM_2, &v1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1 / v2)));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid problem type.");
  }
  PetscCall(VecDestroy(&sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscViewer viewer;
  Mat         X;
  Vec         y;
  PetscReal  *matarr, *vecarr;
  PetscInt    i;
  VecScatter  vscat;
  Vec         yseq;
  PetscInt    seq_local_n;

  char mat_data1[] = "matrix-heart-scale.dat";
  char vec_data1[] = "vector-heart-scale.dat";
  char mat_data2[] = "matrix-housing-scale.dat";
  char vec_data2[] = "vector-housing-scale.dat";

  PetscFunctionBegin;
  user->Q = NULL;
  user->A = NULL;
  switch (user->probType) {
  case DUAL_SVM:
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, vec_data1, FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
    PetscCall(VecLoad(y, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_data1, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
    PetscCall(MatSetType(X, MATMPIAIJ));
    PetscCall(MatLoad(X, viewer));
    PetscCall(MatGetSize(X, &user->m, &user->n));
    PetscCall(MatMatTransposeMult(X, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user->Q));
    PetscCall(MatDiagonalScale(user->Q, y, y));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &user->A));
    PetscCall(MatSetType(user->A, MATMPIDENSE));
    PetscCall(MatSetSizes(user->A, PETSC_DECIDE, PETSC_DECIDE, 1, user->m));
    PetscCall(MatSetUp(user->A));

    PetscCall(VecScatterCreateToZero(y, &vscat, &yseq));
    PetscCall(VecScatterBegin(vscat, y, yseq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(vscat, y, yseq, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetLocalSize(yseq, &seq_local_n));

    /* Since A mat is 1-by-m, only first rank gets full row
     * and rest procs get zero row. So its probably better to
     * do it this way */
    if (seq_local_n > 0) {
      PetscCall(MatDenseGetArray(user->A, &matarr));
      PetscCall(VecGetArray(yseq, &vecarr));
      for (i = 0; i < user->m; i++) matarr[i] = vecarr[i];
      PetscCall(VecRestoreArray(yseq, &vecarr));
      PetscCall(MatDenseRestoreArray(user->A, &matarr));
    }
    PetscCall(MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY));

    /* A = Matrix(y'), with size = (1, m) */
    PetscCall(MatNorm(user->A, NORM_FROBENIUS, &user->matnorm));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
    PetscCall(VecSetSizes(user->x, PETSC_DECIDE, user->m));
    PetscCall(VecSetFromOptions(user->x));
    PetscCall(VecDuplicate(user->x, &user->x0));
    PetscCall(VecDuplicate(user->x, &user->workvec));
    PetscCall(VecDuplicate(user->x, &user->workvec2));
    PetscCall(VecDuplicate(user->x, &user->workvec3));
    PetscCall(VecDuplicate(user->x, &user->y_translation));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&yseq));
    PetscCall(VecScatterDestroy(&vscat));
    break;
  case LAD: {
    Vec x_col, a_col;

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, vec_data2, FILE_MODE_READ, &viewer));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->y_translation));
    PetscCall(VecLoad(user->y_translation, viewer));
    PetscCall(VecScale(user->y_translation, -1.));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, mat_data2, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
    PetscCall(MatSetType(X, MATMPIAIJ));
    PetscCall(MatLoad(X, viewer));
    PetscCall(MatGetSize(X, &user->m, &user->n));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &user->A));
    PetscCall(MatSetType(user->A, MATMPIDENSE));
    PetscCall(MatSetSizes(user->A, PETSC_DECIDE, PETSC_DECIDE, user->m, user->n + 1));
    PetscCall(MatSetFromOptions(user->A));
    PetscCall(MatSetUp(user->A));
    PetscCall(MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY));

    PetscCall(MatCreateVecs(X, NULL, &x_col));
    PetscCall(VecSetFromOptions(x_col));

    PetscInt m_col, n_col;

    PetscCall(MatGetOwnershipRangeColumn(user->A, &m_col, &n_col));

    user->matnorm = 62.68935267753693;
    for (i = 0; i < user->n; i++) {
      PetscCall(MatDenseGetColumnVecWrite(user->A, i, &a_col));
      PetscCall(MatGetColumnVector(X, x_col, i));
      PetscCall(VecCopy(x_col, a_col));
      PetscCall(MatDenseRestoreColumnVecWrite(user->A, i, &a_col));
    }
    PetscCall(MatDenseGetColumnVecWrite(user->A, user->n, &a_col));
    PetscCall(VecSet(a_col, 1.));
    PetscCall(MatDenseRestoreColumnVecWrite(user->A, user->n, &a_col));

    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
    PetscCall(VecSetSizes(user->x, PETSC_DECIDE, user->n + 1));
    PetscCall(VecSetFromOptions(user->x));
    PetscCall(VecDuplicate(user->x, &user->x0));
    PetscCall(VecDuplicate(user->x, &user->workvec));
    PetscCall(VecDuplicate(user->x, &user->workvec2));
    PetscCall(VecDuplicate(user->x, &user->workvec3));
    PetscCall(VecDestroy(&a_col));
    PetscCall(VecDestroy(&x_col));
  } break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid problem type.");
  }

  PetscCall(VecSet(user->x0, 0.));
  PetscCall(VecSet(user->x, 0.));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->q));
  PetscCall(VecSetSizes(user->q, PETSC_DECIDE, user->m));
  PetscCall(VecSetUp(user->q));
  PetscCall(VecSet(user->q, -1));

  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x0));
  PetscCall(VecDestroy(&user->q));
  PetscCall(VecDestroy(&user->workvec));
  PetscCall(VecDestroy(&user->workvec2));
  PetscCall(VecDestroy(&user->workvec3));
  if (user->Q) PetscCall(MatDestroy(&user->Q));
  PetscCall(MatDestroy(&user->A));
  PetscCall(VecDestroy(&user->y_translation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *probTypes[2] = {"dual_svm", "lad"};

  PetscInt probtype;

  PetscFunctionBegin;
  user->C        = 0.1;
  user->g_scale  = 10.;
  user->probType = DUAL_SVM;
  user->set_norm = PETSC_FALSE;
  probtype       = user->probType;

  PetscOptionsBegin(comm, "", "Forward-backward example", "TAO");
  PetscCall(PetscOptionsEList("-problem", "Decide which problem to solve - dual svm or least absolute deviation.", "cv_example.c", probTypes, 2, probTypes[user->probType], &probtype, NULL));

  user->probType = (ProbType)probtype;

  /* Trigger Linesearch by setting norm to zero */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-set_norm", &user->set_norm, NULL));
  /* Box constraint for SVM */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-C", &user->C, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-g_scale", &user->g_scale, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       fdm, gdm, hdm;
  Tao      tao;
  AppCtx   user;
  PetscInt dm_idx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetSolution(tao, user.x));
  PetscCall(TaoSetType(tao, TAOCV));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &fdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &gdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &hdm));

  switch (user.probType) {
  case DUAL_SVM: {
    PetscCall(DMTaoSetType(gdm, DMTAOBOX));
    PetscCall(DMTaoSetType(hdm, DMTAOZERO));
    PetscCall(DMTaoBoxSetContext(gdm, 0, user.C, NULL, NULL));
  } break;
  case LAD:
    PetscCall(DMTaoSetType(gdm, DMTAOL1));
    PetscCall(DMTaoSetType(hdm, DMTAOL1));
    PetscCall(DMTaoSetTranslationVector(hdm, user.y_translation));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem type.");
  }

  switch (user.probType) {
  case DUAL_SVM: {
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, SVM_UserObjGrad_DM, (void *)&user));
    PetscCall(TaoAddDM(tao, fdm, 1.));
    PetscCall(TaoPSSetSmoothTerm(tao, dm_idx));
    dm_idx++;
  } break;
  case LAD: {
    // LAD's f(x) = Zero()
    PetscCall(DMTaoSetObjectiveAndGradient(fdm, LAD_UserObjGrad_DM, (void *)&user));
    PetscCall(TaoAddDM(tao, fdm, 0.));
    PetscCall(TaoPSSetSmoothTerm(tao, dm_idx));
    dm_idx++;
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid problem type.");
  }

  PetscCall(TaoAddDM(tao, gdm, user.g_scale));
  PetscCall(TaoPSSetNonSmoothTerm(tao, dm_idx));
  dm_idx++;

  PetscCall(TaoAddDM(tao, hdm, 1.));
  if (!user.set_norm) PetscCall(TaoPSSetNonSmoothTermWithLinearMap(tao, dm_idx, user.A, 0.));
  else PetscCall(TaoPSSetNonSmoothTermWithLinearMap(tao, dm_idx, user.A, user.matnorm));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  PetscCall(CompareSolution(&user));
  PetscCall(DataDestroy(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(DMDestroy(&fdm));
  PetscCall(DMDestroy(&gdm));
  PetscCall(DMDestroy(&hdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: svm_norm
      nsize: {{1 2 4}}
      localrunfiles: matrix-heart-scale.dat vector-heart-scale.dat
      args: -problem dual_svm -g_scale 1 -tao_converged_reason -tao_max_it 1000 -tao_cv_primal_dual_ratio 1 -C 0.1 -tao_gttol 1.e-5 -tao_ls_max_funcs 0 -set_norm 1
      output_file: output/cv_example_svm_norm.out
      requires: !single

   test:
      suffix: svm_norm_ls
      nsize: {{1 2 4}}
      localrunfiles: matrix-heart-scale.dat vector-heart-scale.dat
      args: -problem dual_svm -g_scale 1 -tao_converged_reason -tao_max_it 2000 -tao_cv_primal_dual_ratio 1 -C 0.1 -tao_gttol 1.e-5 -tao_ls_max_funcs 30 -set_norm 1
      output_file: output/cv_example_svm_norm_ls.out
      requires: !single

   test:
      suffix: svm_ls
      nsize: {{1 2 4}}
      localrunfiles: matrix-heart-scale.dat vector-heart-scale.dat
      args: -problem dual_svm -g_scale 1 -tao_max_it 20 -tao_cv_primal_dual_ratio 1 -C 0.1 -tao_gttol 1.e-5 -set_norm 0 -tao_ls_max_funcs 30 -tao_monitor
      output_file: output/cv_example_svm_ls.out
      requires: !single

   test:
      suffix: lad_norm
      nsize: {{1 2 4}}
      localrunfiles: matrix-housing-scale.dat vector-housing-scale.dat
      args: -problem lad -g_scale 10 -tao_max_it 1000 -tao_cv_primal_dual_ratio 1 -C 0.1 -tao_gttol 1.e-5 -tao_ls_max_funcs 0 -set_norm 1
      output_file: output/cv_example_lad_norm.out
      requires: !single

   test:
      suffix: lad_norm_ls
      nsize: {{1 2 4}}
      localrunfiles: matrix-housing-scale.dat vector-housing-scale.dat
      args: -problem lad -g_scale 10 -tao_max_it 1000 -tao_cv_primal_dual_ratio 1 -C 0.1 -tao_gttol 1.e-5 -set_norm 1 -tao_ls_max_funcs 30
      output_file: output/cv_example_lad_norm_ls.out
      requires: !single

   test:
      suffix: lad_ls
      nsize: {{1 2 4}}
      localrunfiles: matrix-housing-scale.dat vector-housing-scale.dat
      args: -problem lad -g_scale 10 -tao_max_it 1000 -tao_cv_primal_dual_ratio 1 -C 0.1 -tao_gttol 1.e-5 -set_norm 0 -tao_ls_max_funcs 30
      output_file: output/cv_example_lad_ls.out
      requires: !single

TEST*/
