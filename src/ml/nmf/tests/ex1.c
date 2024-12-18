static char help[] = "Trying to figure out NMF interface, so I can have rough idea about internals.\n\n";

#include <petscnmf.h>

typedef struct {
  Mat         X, X_new, W, W_get, W_new, H, H_get;
  TaoTerm     loss, wl1, wl2, hl1, hl2;
  PetscRandom rctx;
  PetscBool   reg_cli;
  PetscInt    rank, m, n;
  PetscReal   loss_scale, l1_ratio, alpha_w, alpha_h;
  char        file[PETSC_MAX_PATH_LEN];
} AppCtx;

/* Options explained:
 *
 * Fianl objective:
 *
 *        loss_           w_l1_         w_l2_         h_l1_       h_l2_
 * \alpha*f(x;wh) + \beta*g(w) + \gamma*h(w) + \delta*j(h) + \eps*k(h)
 *
 *
 * Main objective:
 *
 *        loss_, halfl2squared
 * \alpha*0.5*||X - WH||_F^2 +
 *
 * L1 W Regularizer
 *
 *        w_l1_, l1
 * \beta*||Vec(W)||_1 +
 *
 * L2 W Regularizer
 *
 *        w_l2_, halfl2squared
 * \gamma*0.5*||Vec(W)||_2^2 +
 *
 * L1 H Regularizer
 *
 *        h_l1_, l1
 * \delta*||Vec(H)||_1 +
 *
 * L2 H Regularizer
 *
 *      h_l2_, halfl2squared
 * \eps*0.5*||Vec(H)||_2^2 +
 *
 *                                */

char cli_options[] = "-nmf_add_objective_terms loss_\
                      -loss_taoterm_type halfl2squared\
                      -nmf_taoterm_sum_loss_scale 1.234\
                      -nmf_taoterm_sum_loss_x_parameter_type x\
                      -nmf_taoterm_sum_loss_p_parameter_type wh\
                      -nmf_add_objective_terms w_l1_\
                      -w_l1_taoterm_type l1\
                      -nmf_taoterm_sum_w_l1_scale 1.234\
                      -nmf_taoterm_sum_w_l1_x_parameter_type w\
                      -nmf_taoterm_sum_w_l1_p_parameter_type none\
                      -nmf_add_objective_terms w_l2_\
                      -w_l2_taoterm_type halfl2squared\
                      -nmf_taoterm_sum_w_l2_scale 1.234\
                      -nmf_taoterm_sum_w_l2_x_parameter_type w\
                      -nmf_taoterm_sum_w_l2_p_parameter_type none\
                      -nmf_add_objective_terms h_l1_\
                      -h_l1_taoterm_type l1\
                      -nmf_taoterm_sum_h_l1_scale 1.234\
                      -nmf_taoterm_sum_h_l1_x_parameter_type h\
                      -nmf_taoterm_sum_h_l1_p_parameter_type none\
                      -nmf_add_objective_terms h_l2_\
                      -h_l2_taoterm_type halfl2squared\
                      -nmf_taoterm_sum_h_l2_scale 1.234\
                      -nmf_taoterm_sum_h_l2_x_parameter_type h\
                      -nmf_taoterm_sum_h_l2_p_parameter_type none";

PetscErrorCode DataCreate(AppCtx *user)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  // X: m by n, n_samples by n_features
  // Load Swimmer.dat
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, user->file, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user->X));
  PetscCall(MatLoad(user->X, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatGetSize(user->X, &user->m, &user->n));
  //TODO is there good Xnew data for swimmers? Maybe I need to partitian original into two for backtesting?
  PetscCall(MatDuplicate(user->X, MAT_DO_NOT_COPY_VALUES, &user->X_new));
  //TODO how to duplicate-create W,H, but with different sizes?
  // W: m by k, n_samples by rank
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user->W));
  // H: k by n, rank by n_features
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user->H));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &user->rctx));
  // 0.5*||X - WH||_F^2. L2 on Vec -> Fro for Mat
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &user->loss));
  // ||Vec(W)||_1
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &user->wl1));
  // 0.5*||Vec(W)||_2^2
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &user->wl2));
  // ||Vec(H)||_1
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &user->hl1));
  // 0.5*||Vec(H)||_2^2
  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &user->hl2));
  PetscCall(TaoTermSetType(user->loss, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetType(user->wl1, TAOTERML1));
  PetscCall(TaoTermSetType(user->wl2, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetType(user->hl1, TAOTERML1));
  PetscCall(TaoTermSetType(user->hl2, TAOTERMHALFL2SQUARED));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DataDestroy(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&user->X));
  PetscCall(MatDestroy(&user->X_new));
  PetscCall(MatDestroy(&user->W));
  PetscCall(MatDestroy(&user->H));
  PetscCall(TaoTermDestroy(&user->loss));
  PetscCall(TaoTermDestroy(&user->wl1));
  PetscCall(TaoTermDestroy(&user->wl2));
  PetscCall(TaoTermDestroy(&user->hl1));
  PetscCall(TaoTermDestroy(&user->hl2));
  PetscCall(PetscRandomDestroy(&user->rctx));
  PetscCall(PetscFree(user));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  PetscFunctionBegin;
  PetscOptionsBegin(comm, "", "NMF example using Swimmer", "NMF");
  user->reg_cli    = PETSC_FALSE;
  user->loss_scale = 1.;
  user->l1_ratio   = 1.;
  user->alpha_w    = 1.;
  user->alpha_h    = 1.;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-rank", &user->rank, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-loss_scale", &user->loss_scale, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-l1_ratio", &user->l1_ratio, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha_w", &user->alpha_w, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha_h", &user->alpha_h, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reg_cli", &user->reg_cli, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", user->file, sizeof(user->file), NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx   user;
  PetscNMF nmf;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(DataCreate(&user));

  if (user.reg_cli) PetscCall(PetscOptionsInsertString(NULL, cli_options));

  PetscCall(PetscNMFCreate(PETSC_COMM_WORLD, &nmf));
  PetscCall(PetscNMFSetType(nmf, PETSCNMFAOADMM));
  //TODO Data? Mat?
  PetscCall(PetscNMFSetTrainingMat(nmf, user.X));
  //TODO Q: Do I need this? or should be done internally?
  //Maybe either or?
  PetscCall(PetscNMFSetTransformMat(nmf, user.W));
  PetscCall(PetscNMFSetComponentsMat(nmf, user.H));
  //TODO different init parrent. lets ignore svd
  PetscCall(PetscNMFSetInitializationType(nmf, PETSCNMF_INIT_RANDOM));
  PetscCall(PetscNMFSetRandomContext(nmf, user.rctx));

  // Main loss term ||X - WH||_F^2
  // TODO need to figure out what to do for WH ...
  // For Fro its fine. But for something else like KL, need to be clever
  PetscCall(PetscNMFAddObjectiveTerm(nmf, "loss_", user.loss_scale, user.loss, PETSCNMF_PARAM_X, PETSCNMF_PARAM_WH, NULL));

  if (user.reg_cli) {
    PetscCall(PetscOptionsInsertString(NULL, cli_options));
  } else {
    //Regularizers
    // Q: Is there optimization benefit from having this in internals?
    // alpha_W * l1_ratio * n_features * ||vec(W)||_1
    PetscCall(PetscNMFAddObjectiveTerm(nmf, "w_l1_", user.alpha_w * user.l1_ratio * user.n, user.wl1, PETSCNMF_PARAM_W, PETSCNMF_PARAM_NONE, NULL));
    // alpha_H * l1_ratio * n_samples * ||vec(H)||_1
    PetscCall(PetscNMFAddObjectiveTerm(nmf, "h_l1_", user.alpha_h * user.l1_ratio * user.m, user.hl1, PETSCNMF_PARAM_H, PETSCNMF_PARAM_NONE, NULL));
    // 0.5 * alpha_W * (1 - l1_ratio) * n_features * ||W||_F^2
    PetscCall(PetscNMFAddObjectiveTerm(nmf, "w_l1_", user.alpha_w * (1 - user.l1_ratio) * user.n, user.wl2, PETSCNMF_PARAM_W, PETSCNMF_PARAM_NONE, NULL));
    // 0.5 * alpha_H * (1 - l1_ratio) * n_samples  * ||H||_F^2
    PetscCall(PetscNMFAddObjectiveTerm(nmf, "h_l1_", user.alpha_h * (1 - user.l1_ratio) * user.m, user.hl2, PETSCNMF_PARAM_H, PETSCNMF_PARAM_NONE, NULL));
  }
  // if reg_cli: TODO taoterm_sum inside NMF object
  // TODO -objective_taoterm_sum_{...}_x_parameter_type is probably wrong, but gets the idea across for now i think
  // -nmf_taoterm_sum_loss_x_parameter_type x -nmf_taoterm_sum_loss_p_parameter_type wh
  //
  // -nmf_add_objective_terms w_l1_ -w_l1_taoterm_type l1            -objective_taoterm_sum_w_l1_scale 1.234
  // -objective_taoterm_sum_w_l1_x_parameter_type w -objective_taoterm_sum_w_l1_p_parameter_type none
  //
  // -nmf_add_objective_terms w_l2_ -w_l2_taoterm_type halfl2squared -objective_taoterm_sum_w_l2_scale 1.234
  // -objective_taoterm_sum_w_l2_x_parameter_type w -objective_taoterm_sum_w_l2_p_parameter_type none
  //
  // -nmf_add_objective_terms h_l1_ -h_l1_taoterm_type l1            -objective_taoterm_sum_h_l1_scale 1.234
  // -objective_taoterm_sum_h_l1_x_parameter_type h -objective_taoterm_sum_h_l1_p_parameter_type none
  //
  // -nmf_add_objective_terms h_l2_ -h_l2_taoterm_type halfl2squared -objective_taoterm_sum_h_l2_scale 1.234
  // -objective_taoterm_sum_h_l2_x_parameter_type h -objective_taoterm_sum_h_l2_p_parameter_type none

  //Q1: Is "SetRank" good nomenclature?
  //Q2: In scikit-learn, None=all features,
  //                     auto= automatic (?)
  PetscCall(PetscNMFSetRank(nmf, user.rank));
  PetscCall(PetscNMFSetUp(nmf));
  PetscCall(PetscNMFSetFromOptions(nmf));
  // Main part
  PetscCall(PetscNMFFit(nmf));

  // In scikit-learn, W = model.transform(X)
  // Note: no W_get neded.
  PetscCall(PetscNMFGetTransformMat(nmf, &user.W));
  // Do some sanity check here
  // In scikit-learn, H = mode.components_
  PetscCall(PetscNMFGetComponentsMat(nmf, &user.H));

  //Below is not normal use case... TODO maybe skip this for now.. not that important
  //Decompose Xnew based on H
  //Trying to emulate W_new = model.transform(X_new)
  PetscCall(PetscNMFSetTrainingMat(nmf, user.X_new));
  PetscCall(PetscNMFTransform(nmf));
  PetscCall(PetscNMFGetTransformMat(nmf, &user.W_new));

  PetscCall(DataDestroy(&user));
  PetscCall(PetscNMFDestroy(&nmf));
  PetscCall(PetscFinalize());
}

/*TEST

   build:
     requires: !complex

   test:
     suffix: add_terms_cli
     requires: !single
     args: -f ${DATAFILESPATH}/ml/nmf/Swimmer.dat -nmf_add_objective_terms w_l1_ -w_l1_taoterm_type l1 -objective_taoterm_sum_w_l1_scale 1.234

TEST*/
