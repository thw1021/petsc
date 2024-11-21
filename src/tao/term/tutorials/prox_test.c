/* TaoTermProximalMap example */

#include <petsctao.h>
#include <petsc/private/taoimpl.h>
#include <petscbag.h>

static char help[] = "This example demonstrates various ways to use TaoTerm to solve proximal algorithms.\n";

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
  PetscBool   conj, trans, view, test_small;
  PetscBool   lb_use_vec, ub_use_vec;
  Vec         x, x_test, y, translation, trans_p;
  Vec         lb_vec, ub_vec;
  Vec         sol, sol_trans, sol_conj, sol_conj_trans;
  char        file[PETSC_MAX_PATH_LEN];
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
  user->view        = PETSC_FALSE;
  user->test_small  = PETSC_TRUE;
  user->lb_vec      = NULL;
  user->ub_vec      = NULL;
  user->translation = NULL;
  user->trans_p     = NULL;
  user->compare     = PETSC_FALSE;
  user->conj        = PETSC_FALSE;
  user->trans       = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "TaoTermProximalMap example", "TAOTERM");
  probtype = user->problem;

  PetscCall(PetscOptionsEList("-problem", "Decide which problem to solve.", "prox_ex.c", probTypes, 4, probTypes[user->problem], &probtype, NULL));

  user->problem = (ProblemType)probtype;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user->stepsize, NULL));
  /* stepsize of L1 case */
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lambda", &user->lam, NULL));
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
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view", &user->view, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_small", &user->test_small, NULL));
  /* file name */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", user->file, sizeof(user->file), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Change problem size and solve again
PetscErrorCode SolveSmallerProblem(AppCtx *user, TaoTerm term0, TaoTerm term1)
{
  PetscInt  i;
  PetscBool is_cj;
  Vec       y_small, x_small, trans_small, trans_vec;
  Vec       lb_vec_small = NULL;
  Vec       ub_vec_small = NULL;

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

  trans_vec = (user->trans) ? trans_small : NULL;

  /* Special inner treatment for Conjugate case */
  PetscCall(TaoTermSetSolutionTemplate(term0, x_small));
  PetscCall(PetscObjectTypeCompare((PetscObject)term0, TAOTERMCONJUGATE, &is_cj));
  if (is_cj) {
    TaoTerm subterm;

    PetscCall(TaoTermConjugateGetOriginalTerm(term0, &subterm));
    PetscCall(TaoTermSetSolutionTemplate(subterm, x_small));
    if (user->problem == PROBLEM_BOX) PetscCall(TaoTermBoxSetContext(subterm, user->lb, user->ub, lb_vec_small, ub_vec_small));
  } else if (user->problem ==PROBLEM_BOX) PetscCall(TaoTermBoxSetContext(term0, user->lb, user->ub, lb_vec_small, ub_vec_small));

  /* Using different sized input to test */
  for (i = 0; i < 5; i++) {
    if (user->l2_null) PetscCall(TaoTermProximalMap(term0, trans_vec, 1., term1, y_small, user->stepsize, x_small));
    else PetscCall(TaoTermProximalMap(term0, trans_vec, 1., NULL, y_small, user->stepsize, x_small));
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
    PetscViewer viewer;
    int         fd;
    off_t       offset, off = 0;

    PetscCall(VecDuplicate(user->x, &user->translation));
    PetscCall(VecDuplicate(user->x, &user->sol));
    PetscCall(VecDuplicate(user->x, &user->sol_trans));
    PetscCall(VecDuplicate(user->x, &user->sol_conj));
    PetscCall(VecDuplicate(user->x, &user->sol_conj_trans));

    /* load viewer */
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, user->file, FILE_MODE_READ, &viewer));
    /* load x, y */
    PetscCall(VecLoad(user->y, viewer));
    PetscCall(VecLoad(user->translation, viewer));

    switch (user->problem) {
    case PROBLEM_BOX:
      user->lb = -0.2;
      user->ub = 0.3;
      off      = 0;
      break;
    case PROBLEM_L1:
      user->lam = 0.1;
      /* Skip two empty int per Vec */
      off = PETSC_BINARY_INT_SIZE * 8 + PETSC_BINARY_DOUBLE_SIZE * 40;
      // PetscBinarySeek skip four 10 sized vectors
      break;
    case PROBLEM_SIMPLEX:
      user->simp     = 1.1;
      user->stepsize = 1.; //Currently not allowing stepsize for simplex
      off            = PETSC_BINARY_INT_SIZE * 16 + PETSC_BINARY_DOUBLE_SIZE * 80;
      break;
    case PROBLEM_ZERO:
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }
    // Order of vectors in binary: Box, L1, Simplex
    PetscCall(PetscViewerBinaryGetDescriptor(viewer, &fd));
    PetscCall(PetscBinarySeek(fd, off, PETSC_BINARY_SEEK_CUR, &offset));
    switch (user->problem) {
    case PROBLEM_BOX:
    case PROBLEM_L1:
    case PROBLEM_SIMPLEX:
      PetscCall(VecLoad(user->sol, viewer));
      PetscCall(VecLoad(user->sol_trans, viewer));
      PetscCall(VecLoad(user->sol_conj, viewer));
      PetscCall(VecLoad(user->sol_conj_trans, viewer));
      break;
    case PROBLEM_ZERO:
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }
    PetscCall(PetscViewerDestroy(&viewer));
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
  PetscReal vec_dist, vec_sum;

  PetscFunctionBeginUser;
  if (!user->x_test) PetscCall(VecDuplicate(user->x, &user->x_test));

  if (user->compare) {
    switch (user->problem) {
    case PROBLEM_L1:
    case PROBLEM_SIMPLEX:
    case PROBLEM_BOX:
      break;
    case PROBLEM_ZERO:
      PetscCall(VecSet(user->sol, 0.));
      PetscCall(VecCopy(user->y, user->sol_conj));
      PetscCall(VecCopy(user->translation, user->sol_trans));
      PetscCall(VecScale(user->sol_trans, -1.));
      PetscCall(VecWAXPY(user->sol_conj_trans, 1., user->y, user->translation));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }

    if (user->conj && user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, user->sol_conj_trans));
    else if (user->conj && !user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, user->sol_conj));
    else if (!user->conj && user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, user->sol_trans));
    else if (!user->conj && !user->trans) PetscCall(VecWAXPY(user->x_test, -1., user->x, user->sol));
    else PetscUnreachable();

    PetscCall(VecAbs(user->x_test));
    PetscCall(VecSum(user->x_test, &vec_sum));
    if (vec_sum < 1.e-12) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between ground truth and solution: < 1.e-12\n"));
    else if (vec_sum < 1.e-6) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance between ground truth and solution: < 1.e-6\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "distance ground truth and solution: %e\n", (double)vec_sum));

  } else {
    switch (user->problem) {
    case PROBLEM_L1:
      /* Testing Regularizer version vs Full version */
      PetscCall(VecAXPY(user->x, -1., user->x_test));
      PetscCall(VecNorm(user->x, NORM_2, &vec_dist));
      if (vec_dist < 1.e-11) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between TaoTermProximalMap and SoftThreshold: < 1.e-11\n"));
      } else if (vec_dist < 1.e-6) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between TaoTermProximalMap and SoftThreshold: < 1.e-6\n"));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "error between TaoTermProximalMap and SoftThreshold: %e\n", (double)vec_dist));
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
  if (user->compare) {
    PetscCall(VecDestroy(&user->sol));
    PetscCall(VecDestroy(&user->sol_trans));
    PetscCall(VecDestroy(&user->sol_conj));
    PetscCall(VecDestroy(&user->sol_conj_trans));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TaoTerm  term0, term1, term0_conj;
  AppCtx   user;
  PetscInt i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &term0));
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &term1));

  switch (user.problem) {
  case PROBLEM_L1:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "TAOTERML1 Case\n"));
    PetscCall(TaoTermSetType(term0, TAOTERML1));
    PetscCall(TaoTermSetType(term1, TAOTERMHALFL2SQUARED));
    /* Try built-in Soft-Threshold */
    PetscCall(TaoSoftThreshold(user.y, -user.lam * user.stepsize, user.lam * user.stepsize, user.x_test));
    break;
  case PROBLEM_SIMPLEX:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "TAOTERMSIMPLEX Case\n"));
    PetscCall(TaoTermSetType(term0, TAOTERMSIMPLEX));
    PetscCall(TaoTermSetType(term1, TAOTERMHALFL2SQUARED));
    PetscCall(TaoTermSimplexSetSize(term0, user.simp));
    PetscCall(TaoTermSimplexSetTolerance(term0, user.tol));
    break;
  case PROBLEM_BOX:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "TAOTERMBOX Case\n"));
    PetscCall(TaoTermSetType(term0, TAOTERMBOX));
    PetscCall(TaoTermSetType(term1, TAOTERMHALFL2SQUARED));
    PetscCall(TaoTermBoxSetContext(term0, user.lb, user.ub, user.lb_vec, user.ub_vec));
    break;
  case PROBLEM_ZERO:
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "TAOTERMZERO Case\n"));
    PetscCall(TaoTermSetType(term0, TAOTERMZERO));
    PetscCall(TaoTermSetType(term1, TAOTERMHALFL2SQUARED));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
  }

  user.trans_p = (user.trans) ? user.translation : NULL;

  /* Conjugate case */
  if (user.conj) { PetscCall(TaoTermCreateConjugate(term0, &term0_conj)); }

  if (user.conj) {
    PetscCall(TaoTermSetSolutionTemplate(term0_conj, user.x));
    PetscCall(TaoTermSetSolutionTemplate(term0, user.x));
    PetscCall(TaoTermSetFromOptions(term0_conj));
    PetscCall(TaoTermSetUp(term0_conj));
  } else {
    PetscCall(TaoTermSetSolutionTemplate(term0, user.x));
    PetscCall(TaoTermSetFromOptions(term0));
    PetscCall(TaoTermSetUp(term0));
  }

  /* Solving same problem few times to simulate iteration */
  for (i = 0; i < 5; i++) {
    if (user.conj) {
      if (user.l2_null) {
        PetscCall(TaoTermProximalMap(term0_conj, user.trans_p, user.lam, term1, user.y, user.stepsize, user.x));
      } else {
        PetscCall(TaoTermProximalMap(term0_conj, user.trans_p, user.lam, NULL, user.y, user.stepsize, user.x));
      }
    } else {
      if (user.l2_null) {
        PetscCall(TaoTermProximalMap(term0, user.trans_p, user.lam, term1, user.y, user.stepsize, user.x));
      } else {
        PetscCall(TaoTermProximalMap(term0, user.trans_p, user.lam, NULL, user.y, user.stepsize, user.x));
      }
    }
  }

  if (!user.view) PetscCall(CheckSolution(&user));
  /* Change input size to see if this works */
  if (user.test_small) {
    if (user.conj) {
      if (user.view) PetscCall(TaoTermView(term0_conj, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(SolveSmallerProblem(&user, term0_conj, term1));
    } else {
      if (user.view) PetscCall(TaoTermView(term0, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(SolveSmallerProblem(&user, term0, term1));
    }
  }

  if (user.view && user.conj) {
    TaoTermType orig_type;
    PetscBool   flg;

    PetscCall(TaoTermView(term0_conj, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(TaoTermConjugateGetOriginalType(term0_conj, &orig_type));
    switch (user.problem) {
    case PROBLEM_L1:
      PetscCall(PetscStrcmp(orig_type, "l1", &flg));
      break;
    case PROBLEM_BOX:
      PetscCall(PetscStrcmp(orig_type, "box", &flg));
      break;
    case PROBLEM_ZERO:
      PetscCall(PetscStrcmp(orig_type, "zero", &flg));
      break;
    case PROBLEM_SIMPLEX:
      PetscCall(PetscStrcmp(orig_type, "simplex", &flg));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported problem type!");
    }
    PetscAssert(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_NOTSAMETYPE, "Conjugate term's original type not same");
  } else if (user.view) PetscCall(TaoTermView(term0, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(TaoTermDestroy(&term0));
  PetscCall(TaoTermDestroy(&term1));
  if (user.conj) PetscCall(TaoTermDestroy(&term0_conj));
  PetscCall(VecViewFromOptions(user.x, NULL, "-solution_vec_view"));
  PetscCall(DataDestroy(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES) datafilespath

   test:
      nsize: {{1 2 4}}
      suffix: soft
      args: -problem l1 -l2_null {{0 1}}
      output_file: output/prox_ex_soft.out
      requires: !single

   test:
      nsize: {{1 2 4}}
      suffix: soft_compare
      args: -problem l1 -l2_null {{0 1}} -compare 1 -conjugate {{1 0}} -trans {{0 1}} -f ${DATAFILESPATH}/tao/prox_ex_compare.dat
      output_file: output/prox_ex_soft_compare.out
      requires: !single

   test:
      suffix: soft_view
      args: -problem l1 -l2_null 1 -view 1
      output_file: output/prox_ex_soft_view.out
      requires: !single

   test:
      suffix: soft_view_conjugate
      args: -problem l1 -l2_null 1 -view 1 -conjugate 1
      output_file: output/prox_ex_soft_view_cj.out
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
      args: -problem simplex -l2_null {{0 1}} -compare 1 -conjugate {{0 1}} -trans {{0 1}} -f ${DATAFILESPATH}/tao/prox_ex_compare.dat
      output_file: output/prox_ex_simplex_compare.out
      requires: !single

   test:
      suffix: simplex_view
      args: -problem simplex -l2_null 1 -view 1 -taoterm_simplex_tol 1.e-12 -taoterm_simplex_size 1.2
      output_file: output/prox_ex_simplex_view.out
      requires: !single

   test:
      suffix: simplex_view_conjugate
      args: -problem simplex -l2_null 1 -view 1 -conjugate 1 -taoterm_simplex_tol 1.e-12 -taoterm_simplex_size 1.2
      output_file: output/prox_ex_simplex_view_cj.out
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
      args: -problem box -l2_null {{0 1}} -compare 1 -conjugate {{0 1}} -trans {{0 1}} -f ${DATAFILESPATH}/tao/prox_ex_compare.dat
      output_file: output/prox_ex_box_compare.out
      requires: !single

   test:
      suffix: box_view_1
      args: -problem box -l2_null 1 -view 1 -taoterm_box_lb_real -1.2 -lb 1.2 -taoterm_box_ub_real 3.3 -ub 3.3
      output_file: output/prox_ex_box_view_1.out
      requires: !single

   test:
      suffix: box_view_2
      args: -problem box -l2_null 1 -view 1 -lb_use_vec 1 -taoterm_box_ub_real 3.3 -ub 3.3
      output_file: output/prox_ex_box_view_2.out
      requires: !single

   test:
      suffix: box_view_3
      args: -problem box -l2_null 1 -view 1 -ub_use_vec 1 -taoterm_box_lb_real -1.2 -lb -1.2
      output_file: output/prox_ex_box_view_3.out
      requires: !single

   test:
      suffix: box_view_4
      args: -problem box -l2_null 1 -view 1 -lb_use_vec 1 -ub_use_vec 1
      output_file: output/prox_ex_box_view_4.out
      requires: !single

   test:
      suffix: box_view_1_cj
      args: -problem box -l2_null 1 -view 1 -conjugate 1
      output_file: output/prox_ex_box_view_1_cj.out
      requires: !single

   test:
      suffix: box_view_2_cj
      args: -problem box -l2_null 1 -view 1 -lb_use_vec 1 -conjugate 1
      output_file: output/prox_ex_box_view_2_cj.out
      requires: !single

   test:
      suffix: box_view_3_cj
      args: -problem box -l2_null 1 -view 1 -ub_use_vec 1 -conjugate 1
      output_file: output/prox_ex_box_view_3_cj.out
      requires: !single

   test:
      suffix: box_view_4_cj
      args: -problem box -l2_null 1 -view 1 -lb_use_vec 1 -ub_use_vec 1 -conjugate 1
      output_file: output/prox_ex_box_view_4_cj.out
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
      args: -problem zero -l2_null {{0 1}} -compare 1 -conjugate {{0 1}} -trans {{0 1}} -f ${DATAFILESPATH}/tao/prox_ex_compare.dat
      output_file: output/prox_ex_zero_compare.out
      requires: !single

   test:
      suffix: zero_view
      args: -problem zero -l2_null 1 -view 1
      output_file: output/prox_ex_zero_view.out
      requires: !single

   test:
      suffix: zero_view_conjugate
      args: -problem zero -l2_null 1 -view 1 -conjugate 1
      output_file: output/prox_ex_zero_view_cj.out
      requires: !single

TEST*/
