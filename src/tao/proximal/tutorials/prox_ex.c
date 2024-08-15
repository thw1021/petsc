/* DMApplyProximalMap example */

#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/taoimpl.h>
#include <petscbag.h>

static char help[] = "This example demonstrates various ways to use DM to solve proximal algorithms.\n";

typedef enum {
  PROBLEM_L1,
  PROBLEM_SIMPLEX,
  PROBLEM_BOX,
  PROBLEM_ZERO,
} ProblemType;

typedef struct {
  ProblemType problem;
  PetscScalar lb, ub;
  PetscInt    n;                   /* dimension */
  PetscReal   stepsize, lam, simp; /* simp: size of simplex */
  PetscReal   tol;
  PetscBool   l2_null;
  PetscBool   compare; /* compare: compare against known implementation's output, for a given fixed input */
  PetscBool   conj, trans;
  PetscBool   lb_use_vec, ub_use_vec;
  Vec         x, x_test, y, translation;
  Vec         lb_vec, ub_vec;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *probTypes[4] = {"l1", "simplex", "box", "zero"};

  PetscInt probtype;

  PetscFunctionBeginUser;
  user->n           = 10;
  user->lam         = 1;
  user->lb          = -1;
  user->ub          = 1;
  user->simp        = 1.;
  user->tol         = 1.e-12;
  user->stepsize    = 1.;
  user->problem     = PROBLEM_L1;
  user->l2_null     = PETSC_FALSE;
  user->lb_use_vec  = PETSC_FALSE;
  user->ub_use_vec  = PETSC_FALSE;
  user->lb_vec      = NULL;
  user->ub_vec      = NULL;
  user->translation = NULL;
  user->compare     = PETSC_FALSE;
  user->conj        = PETSC_FALSE;
  user->trans       = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "DMTaoApplyProximalMap example", "DMTAO");
  probtype = user->problem;

  PetscCall(PetscOptionsEList("-problem", "Decide which problem to solve.", "prox_ex.c", probTypes, 4, probTypes[user->problem], &probtype, NULL));

  user->problem = (ProblemType)probtype;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user->stepsize, NULL));
  /* stepsize of L1 case */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-l1_scale", &user->lam, NULL));
  /* Four cases for Box:
   * Case 1: lb_real, ub_real
   * Case 2: lb_real, ub_vec
   * Case 3: lb_vec,  ub_real
   * Case 4: lb_vec,  ub_vec  */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lb", &user->lb, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ub", &user->ub, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-lb_use_vec", &user->lb_use_vec, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ub_use_vec", &user->ub_use_vec, NULL));
  /* size of simplex */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-simplex", &user->simp, NULL));
  /* whether to use NULL for regularizer */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-l2_null", &user->l2_null, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-compare", &user->compare, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-conjugate", &user->conj, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-trans", &user->trans, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolveSmallerProblem(AppCtx *user, DM dm0, DM dm1)
{
  PetscInt i;
  Vec      y_small, x_small, trans_small;
  Vec      lb_vec_small = NULL;
  Vec      ub_vec_small = NULL;

  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y_small));
  PetscCall(VecSetSizes(y_small, PETSC_DECIDE, 2));
  PetscCall(VecSetUp(y_small));
  PetscCall(VecDuplicate(y_small, &x_small));
  PetscCall(VecDuplicate(y_small, &trans_small));
  if (user->lb_use_vec) {
    PetscCall(VecDuplicate(y_small, &lb_vec_small));
    PetscCall(VecSet(lb_vec_small, user->lb));
  }
  if (user->ub_use_vec) {
    PetscCall(VecDuplicate(y_small, &ub_vec_small));
    PetscCall(VecSet(ub_vec_small, user->ub));
  }
  PetscCall(VecSetRandom(y_small, NULL));
  PetscCall(VecSetRandom(trans_small, NULL));
  /* Using different sized input to test */
  if (user->trans) PetscCall(DMTaoSetTranslationVector(dm0, trans_small));
  if (user->problem == PROBLEM_BOX) PetscCall(DMTaoBoxSetContext(dm0, user->lb, user->ub, lb_vec_small, ub_vec_small));
  for (i = 0; i < 5; i++) {
    if (user->l2_null) PetscCall(DMTaoApplyProximalMap(dm0, NULL, user->stepsize, y_small, x_small, user->conj));
    else PetscCall(DMTaoApplyProximalMap(dm0, dm1, user->stepsize, y_small, x_small, user->conj));
  }
  PetscCall(VecDestroy(&y_small));
  PetscCall(VecDestroy(&x_small));
  PetscCall(VecDestroy(&trans_small));
  PetscCall(VecDestroy(&ub_vec_small));
  PetscCall(VecDestroy(&lb_vec_small));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscRandom rctx;

  PetscFunctionBeginUser;
  if (user->compare) user->n = 10;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
  PetscCall(VecSetSizes(user->x, PETSC_DECIDE, user->n));
  PetscCall(VecSetFromOptions(user->x));
  PetscCall(VecDuplicate(user->x, &user->y));
  PetscCall(VecDuplicate(user->x, &user->x_test));

  if (user->compare) {
    /* For compare, we only consider n=10 case */
    PetscViewer fd;
    char        inputFile[] = "prox_ex_compare_input_x_y";

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, inputFile, FILE_MODE_READ, &fd));
    PetscCall(VecLoad(user->y, fd));
    PetscCall(VecDuplicate(user->x, &user->translation));
    PetscCall(VecLoad(user->translation, fd));
    PetscCall(PetscViewerDestroy(&fd));

    switch (user->problem) {
    case PROBLEM_L1:
      user->lam = 0.1;
      break;
    case PROBLEM_BOX:
      user->lb = -0.2;
      user->ub = 0.3;
      break;
    case PROBLEM_SIMPLEX:
      user->simp     = 1.1;
      user->stepsize = 1.; //Currently not allowing stepsize for simplex
      break;
    case PROBLEM_ZERO:
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }
  } else {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(PetscRandomSetInterval(rctx, -10, 10));
    /* x : Random vec, from -10 to 10 */
    PetscCall(PetscRandomSetSeed(rctx, 1234));
    PetscCall(VecSetRandom(user->x, rctx));
    PetscCall(VecCopy(user->x, user->x_test));
    /* y : Random vec, from -10 to 10 */
    PetscCall(PetscRandomSetSeed(rctx, 5678));
    PetscCall(VecSetRandom(user->y, rctx));

    if (user->lb_use_vec) {
      PetscCall(VecDuplicate(user->x, &user->lb_vec));
      PetscCall(VecSet(user->lb_vec, user->lb));
    }
    if (user->ub_use_vec) {
      PetscCall(VecDuplicate(user->x, &user->ub_vec));
      PetscCall(VecSet(user->ub_vec, user->ub));
    }
    PetscCall(PetscRandomDestroy(&rctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CheckSolution(AppCtx *user)
{
  PetscReal   vec_dist, vec_sum;
  PetscViewer fd;

  PetscFunctionBeginUser;
  if (!user->x_test) PetscCall(VecDuplicate(user->x, &user->x_test));

  if (user->compare) {
    Vec sol, sol_trans, sol_conj, sol_conj_trans;

    PetscCall(VecDuplicate(user->x, &sol));
    PetscCall(VecDuplicate(user->x, &sol_trans));
    PetscCall(VecDuplicate(user->x, &sol_conj));
    PetscCall(VecDuplicate(user->x, &sol_conj_trans));

    switch (user->problem) {
    case PROBLEM_L1: {
      // SoftTresh with scale 0.1
      char filename[] = "prox_ex_compare_l1_0_dot_1_sol";
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &fd));
      PetscCall(VecLoad(sol, fd));
      PetscCall(VecLoad(sol_trans, fd));
      PetscCall(VecLoad(sol_conj, fd));
      PetscCall(VecLoad(sol_conj_trans, fd));
      PetscCall(PetscViewerDestroy(&fd));
    } break;
    case PROBLEM_SIMPLEX: {
      char filename[] = "prox_ex_compare_simplex_1_dot_1_sol";
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &fd));
      PetscCall(VecLoad(sol, fd));
      PetscCall(VecLoad(sol_trans, fd));
      PetscCall(VecLoad(sol_conj, fd));
      PetscCall(VecLoad(sol_conj_trans, fd));
      PetscCall(PetscViewerDestroy(&fd));
    } break;
    case PROBLEM_BOX: {
      // Box with [-0.2, 0.3]
      char filename[] = "prox_ex_compare_box_neg_0_dot_2_pos_0_dot_3_sol";
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &fd));
      PetscCall(VecLoad(sol, fd));
      PetscCall(VecLoad(sol_trans, fd));
      PetscCall(VecLoad(sol_conj, fd));
      PetscCall(VecLoad(sol_conj_trans, fd));
      PetscCall(PetscViewerDestroy(&fd));
    } break;
    case PROBLEM_ZERO:
      PetscCall(VecSet(sol, 0.));
      PetscCall(VecCopy(user->y, sol_conj));
      PetscCall(VecCopy(user->translation, sol_trans));
      PetscCall(VecScale(sol_trans, -1.));
      PetscCall(VecWAXPY(sol_conj_trans, 1., user->y, user->translation));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }

    if (user->conj && user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, sol_conj_trans));
    else if (user->conj && !user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, sol_conj));
    else if (!user->conj && user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, sol_trans));
    else if (!user->conj && !user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, sol));
    else PetscUnreachable();

    PetscCall(VecAbs(user->x_test));
    PetscCall(VecSum(user->x_test, &vec_sum));
    if (vec_sum < 1.e-12) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between ground truth and solution: < 1.e-12\n"));
    else if (vec_sum < 1.e-6) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between ground truth and solution: < 1.e-6\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance ground truth and solution: %e\n", (double)vec_sum));

    PetscCall(VecDestroy(&sol));
    PetscCall(VecDestroy(&sol_trans));
    PetscCall(VecDestroy(&sol_conj));
    PetscCall(VecDestroy(&sol_conj_trans));
  } else {
    switch (user->problem) {
    case PROBLEM_L1:
      /* Testing Regularizer version vs Full version */
      PetscCall(VecAXPY(user->x, -1., user->x_test));
      PetscCall(VecNorm(user->x, NORM_2, &vec_dist));
      if (vec_dist < 1.e-11) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between DMTaoApplyProximalMap and SoftThreshold: < 1.e-11\n"));
      } else if (vec_dist < 1.e-6) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between DMTaoApplyProximalMap and SoftThreshold: < 1.e-6\n"));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between DMTaoApplyProximalMap and SoftThreshold: %e\n", (double)vec_dist));
      }
      break;
    case PROBLEM_SIMPLEX: {
      PetscReal sum, min, max;
      PetscCall(VecSum(user->x, &sum));
      PetscCall(VecMin(user->x, NULL, &min));
      PetscCall(VecMax(user->x, NULL, &max));
      vec_dist = PetscAbsReal(sum - user->simp);
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Smallest element of solution: %e\n", (double)min));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Largest element of solution: %e\n", (double)max));
      if (vec_dist < 1.e-11) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: < 1.e-11\n"));
      } else if (vec_dist < 1.e-6) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: < 1.e-6\n"));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between VecSum and Simplex Size: %e\n", (double)vec_dist));
      }
    } break;
    case PROBLEM_BOX:
    case PROBLEM_ZERO: {
      PetscReal min, max;

      PetscCall(VecMin(user->x, NULL, &min));
      PetscCall(VecMax(user->x, NULL, &max));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Smallest element of solution: %e\n", (double)min));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Largest element of solution: %e\n", (double)max));
    } break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&user->x));
  PetscCall(VecDestroy(&user->x_test));
  PetscCall(VecDestroy(&user->y));
  if (user->lb_vec) PetscCall(VecDestroy(&user->lb_vec));
  if (user->ub_vec) PetscCall(VecDestroy(&user->ub_vec));
  if (user->translation) PetscCall(VecDestroy(&user->translation));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       dm0, dm1;
  AppCtx   user;
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm0));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm1));

  switch (user.problem) {
  case PROBLEM_L1:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOL1 Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOL1));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoL1SetContext(dm0, user.lam));
    /* Try built-in Soft-Threshold */
    PetscCall(TaoSoftThreshold(user.y, -user.lam * user.stepsize, user.lam * user.stepsize, user.x_test));
    break;
  case PROBLEM_SIMPLEX:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOSIMPLEX Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOSIMPLEX));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoSimplexSetContext(dm0, user.simp, user.tol));
    break;
  case PROBLEM_BOX:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOBOX Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOBOX));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    PetscCall(DMTaoBoxSetContext(dm0, user.lb, user.ub, user.lb_vec, user.ub_vec));
    break;
  case PROBLEM_ZERO:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMTAOZERO Case\n"));
    PetscCall(DMTaoSetType(dm0, DMTAOZERO));
    PetscCall(DMTaoSetType(dm1, DMTAOL2));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
  }

  if (user.trans) PetscCall(DMTaoSetTranslationVector(dm0, user.translation));

  /* Solving same problem few times to simulate iteration */
  for (i = 0; i < 5; i++) {
    if (user.l2_null) {
      PetscCall(DMTaoApplyProximalMap(dm0, NULL, user.stepsize, user.y, user.x, user.conj));
    } else {
      PetscCall(DMTaoApplyProximalMap(dm0, dm1, user.stepsize, user.y, user.x, user.conj));
    }
  }

  PetscCall(CheckSolution(&user));
  /* Change input size to see if this works */
  PetscCall(SolveSmallerProblem(&user, dm0, dm1));

  PetscCall(DMDestroy(&dm0));
  PetscCall(DMDestroy(&dm1));
  PetscCall(VecViewFromOptions(user.x, NULL, "-solution_vec_view"));
  PetscCall(DataDestroy(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      nsize: {{1 2 4}}
      suffix: soft
      args: -problem l1 -l2_null {{0 1}}
      output_file: output/prox_ex_soft.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: soft_compare
      localrunfiles: prox_ex_compare_input_x_y prox_ex_compare_l1_0_dot_1_sol
      args: -problem l1 -l2_null {{0 1}} -compare 1 -conjugate {{1 0}} -trans{{0 1}}
      output_file: output/prox_ex_soft_compare.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: simplex
      args: -problem simplex -l2_null {{0 1}}
      output_file: output/prox_ex_simplex.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: simplex_compare
      localrunfiles: prox_ex_compare_input_x_y prox_ex_compare_simplex_1_dot_1_sol
      args: -problem simplex -l2_null {{0 1}} -compare 1 -conjugate {{0 1}} -trans {{0 1}}
      output_file: output/prox_ex_simplex_compare.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: box
      args: -problem box -l2_null {{0 1}} -lb_use_vec {{0 1}} -ub_use_vec {{0 1}}
      output_file: output/prox_ex_box.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: box_compare
      localrunfiles: prox_ex_compare_input_x_y prox_ex_compare_box_neg_0_dot_2_pos_0_dot_3_sol
      args: -problem box -l2_null {{0 1}} -compare 1 -conjugate {{0 1}} -trans {{0 1}}
      output_file: output/prox_ex_box_compare.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: zero
      args: -problem zero -l2_null {{0 1}}
      output_file: output/prox_ex_zero.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: zero_compare
      localrunfiles: prox_ex_compare_input_x_y
      args: -problem zero -l2_null {{0 1}} -compare 1 -conjugate {{0 1}} -trans {{0 1}}
      output_file: output/prox_ex_zero_compare.out
      requires: !single

TEST*/
