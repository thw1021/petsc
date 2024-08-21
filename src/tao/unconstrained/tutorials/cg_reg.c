#include <petsctao.h>
#include <petscdm.h>
#include <petsc/private/taoimpl.h>

static char help[] = "This example demonstrates various ways to use DMTao. \n\
                      It solves f(x) + scale*g(x,y), where f(x) = 0.5 x.T A x - b.T x, and \n\
                      g(x,y) = stepsize* ||x - y||_2^2. \n\
                      f(x) can be with TaoSetObjectiveAndGradient, TaoSetObjective and TaoSetGradient, \n\
                      or TaoAddDM(Tao, DM). Here, DM would carry all the necesary information about the problem description. \n\
                      Similarly, g(x,y) can be formed via DMTaoUseTaoRoutines with appropriate sub Tao, Built-in DMTao type,\n\
                      or with DMTaoSet... routines. \n\
                      Composite formulation can be formed in three ways: \n\
                      First, set f(x) with TaoSet..., and set g(x,y) via TaoSetRegularizer. \n\
                      Second, set f(x) with TaoSet..., and set g(x,y) via TaoAddDM. \n\
                      Third, set f(x) with TaoAddDM, and set g(x,y) with TaoAddDM. \n";

/* How to add f(x) */
typedef enum {
  COMPOSE_MAIN_VIA_DM,        /* Problem will be set via TaoAddDM(tao, fdm), with DMTaoSetObj,Grad,ObjGrad */
  COMPOSE_MAIN_VIA_DM_SUBTAO, /* Set Main via TaoAddDM(tao, gdm), with DMTaoUseTaoRoutine(fdm, subtao)     */
  COMPOSE_MAIN_VIA_TAO        /* Problem will be set via TaoSetObj,Grad,or ObjGrad                         */
} MainComposeType;

/* How to setup f(x) */
typedef enum {
  MAIN_FORM_OBJGRAD,      /* f: ObjGrad         */
  MAIN_FORM_OBJ_AND_GRAD, /* f: Obj And Grad    */
} MainFormType;

/* How to add g(x,y) */
typedef enum {
  COMPOSE_REG_VIA_REG, /* Use TaoSetRegularizer to add regularizer DM */
  COMPOSE_REG_VIA_ADD  /* Use TaoAddDM to add regularizer DM          */
} RegComposeType;

/* How to setup g(x,y) */
typedef enum {
  REG_FORM_OBJGRAD,             /* g: ObjGrad                 */
  REG_FORM_OBJ_AND_GRAD,        /* g: Obj and Grad            */
  REG_FORM_BUILT_IN,            /* g: Built-in type DMTAOL2   */
  REG_FORM_SUBTAO_OBJGRAD,      /* g: sub Tao w/ objgrad      */
  REG_FORM_SUBTAO_OBJ_AND_GRAD, /* g: sub Tao w/ obj and grad */
} RegFormType;

typedef struct {
  MainComposeType mainComposeType;
  MainFormType    mainFormType;
  RegComposeType  regComposeType;
  RegFormType     regFormType;
  PetscBool       yvec;
  PetscInt        n; /* dimension */
  PetscReal       stepsize;
  PetscReal       mu1; /* Parameter for soft-threshold */
  Mat             A;
  Vec             b, workvec, y;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  const char *mainComposeTypes[3] = {"use_dm", "use_dm_subtao", "use_tao"};
  const char *mainFormTypes[2]    = {"objgrad", "objandgrad"};
  const char *regComposeTypes[2]  = {"set_reg", "add_dm"};
  const char *regFormTypes[5]     = {"objgrad", "objandgrad", "builtin", "subtao_objgrad", "subtao_obj_and_grad"};

  PetscInt regcomp, maincomp, mainform, regform;

  PetscFunctionBegin;
  user->mainComposeType = COMPOSE_MAIN_VIA_DM;
  user->mainFormType    = MAIN_FORM_OBJGRAD;
  user->regComposeType  = COMPOSE_REG_VIA_REG;
  user->regFormType     = REG_FORM_OBJGRAD;
  user->yvec            = PETSC_TRUE;
  user->n               = 10;
  user->stepsize        = 0.5;
  user->mu1             = 1;
  PetscOptionsBegin(comm, "", "Regularizer CG via DMTao example", "DMTAO");

  maincomp = user->mainComposeType;
  mainform = user->mainFormType;
  regcomp  = user->regComposeType;
  regform  = user->regFormType;

  PetscCall(PetscOptionsEList("-main_compose", "Main problem (f) compose type", "cg_reg.c", mainComposeTypes, 3, mainComposeTypes[user->mainComposeType], &maincomp, NULL));
  PetscCall(PetscOptionsEList("-main_form", "Main problem formation type", "cg_reg.c", mainFormTypes, 2, mainFormTypes[user->mainFormType], &mainform, NULL));
  PetscCall(PetscOptionsEList("-reg_compose", "Regularizer (g) compose type", "cg_reg.c", regComposeTypes, 2, regComposeTypes[user->regComposeType], &regcomp, NULL));
  PetscCall(PetscOptionsEList("-reg_form", "Regularizer formation type", "cg_reg.c", regFormTypes, 5, regFormTypes[user->regFormType], &regform, NULL));
  PetscCall(PetscOptionsBool("-yvec", "The y-vec option", "cg_reg.c", user->yvec, &user->yvec, NULL));

  user->mainComposeType = (MainComposeType)maincomp;
  user->mainFormType    = (MainFormType)mainform;
  user->regComposeType  = (RegComposeType)regcomp;
  user->regFormType     = (RegFormType)regform;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &user->n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-stepsize", &user->stepsize, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &user->mu1, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective
 *
 * f(x) = 0.5 x.T A x - b.T x */
PetscErrorCode UserObj(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= 0.5;
  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient: grad f = Ax - b */
PetscErrorCode UserGrad(Tao tao, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecAXPY(G, -1., user->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x.T A x - b.T x
 * grad f = A x - b                         */
PetscErrorCode UserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecWAXPY(G, -1., user->b, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= 0.5;

  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective
 *
 * f(x) = 0.5 x.T A x - b.T x */
PetscErrorCode UserObj_DM(DM dm, Vec X, PetscReal *f, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= 0.5;
  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Gradient: grad f = Ax - b */
PetscErrorCode UserGrad_DM(DM dm, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecAXPY(G, -1., user->b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Objective and Gradient
 *
 * f(x) = 0.5 x.T A x - b.T x
 * grad f = A x - b                         */
PetscErrorCode UserObjGrad_DM(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp;

  PetscFunctionBegin;
  PetscCall(MatMult(user->A, X, user->workvec));
  PetscCall(VecWAXPY(G, -1., user->b, user->workvec));
  PetscCall(VecTDot(user->workvec, X, f));
  *f *= 0.5;

  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Full Obj and Grad for comparison
 *
 * f(x)   = 0.5 x.T A x - b.T x + step * \|x-y\|_2^2
 * grad f = A x - b + x - y                           */
PetscErrorCode FullUserObjGrad(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscScalar temp, reg_val, stepsize;

  PetscFunctionBegin;
  stepsize = user->stepsize;
  /* workvec :  x-y */
  if (user->yvec) {
    PetscCall(VecWAXPY(user->workvec, -1, user->y, X));
  } else {
    PetscCall(VecCopy(X, user->workvec));
  }
  PetscCall(VecTDot(user->workvec, user->workvec, &reg_val));
  reg_val *= stepsize;

  /* f = 0.5 x^T A x */
  PetscCall(MatMult(user->A, X, G));
  PetscCall(VecTDot(G, X, f));
  *f *= 0.5;

  PetscCall(VecAXPY(G, -1., user->b));
  PetscCall(VecAXPY(G, 1., user->workvec));
  /* f -= b^T x */
  PetscCall(VecTDot(user->b, X, &temp));
  *f -= temp;
  /* Add reg term */
  *f += reg_val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. \|X-Y\|_2^2
 * This is to set Tao routines for DMTao */
PetscErrorCode L2_ObjGrad_Tao(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(VecCopy(X, G));
  /* Note: Scale part will be done internally */
  if (user->yvec) {
    Vec y;
    DM  dm;

    PetscCall(TaoGetParentDM(tao, &dm));
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecTDot(G, G, f));
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Obj_Tao(Tao tao, Vec X, PetscReal *f, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  if (user->yvec) {
    Vec y;
    DM  dm;

    PetscCall(TaoGetParentDM(tao, &dm));
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecAXPBYPCZ(user->workvec, 1, -1, 0, X, y));
    PetscCall(VecTDot(user->workvec, user->workvec, f));
  } else {
    PetscCall(VecTDot(X, X, f));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Grad_Tao(Tao tao, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(VecCopy(X, G));
  if (user->yvec) {
    Vec y;
    DM  dm;

    PetscCall(TaoGetParentDM(tao, &dm));
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L2 Metric. \|X-Y\|_2^2 */
PetscErrorCode L2_ObjGrad(DM dm, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(VecCopy(X, G));
  if (user->yvec) {
    Vec y;
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecAXPY(G, -1., y));
  }
  PetscCall(VecTDot(G, G, f));
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Obj(DM dm, Vec X, PetscReal *f, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  if (user->yvec) {
    Vec y;
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecWAXPY(user->workvec, -1., y, X));
    PetscCall(VecTDot(user->workvec, user->workvec, f));
  } else {
    PetscCall(VecTDot(X, X, f));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode L2_Grad(DM dm, Vec X, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  if (user->yvec) {
    Vec y;
    PetscCall(DMTaoGetCentralVector(dm, &y));
    PetscCall(VecWAXPY(G, -1., y, X));
  } else {
    PetscCall(VecCopy(X, G));
  }
  PetscCall(VecScale(G, 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Tao         tao, tao_full, fdm_subtao, gdm_subtao;
  DM          fdm, gdm;
  Vec         x, x_full;
  Mat         temp_mat;
  AppCtx      user;
  PetscRandom rctx;
  PetscReal   vec_dist;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DETERMINE, user.n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecZeroEntries(x));
  PetscCall(VecDuplicate(x, &x_full));
  PetscCall(VecDuplicate(x, &user.y));
  PetscCall(VecDuplicate(x, &user.workvec));
  PetscCall(VecDuplicate(x, &user.b));

  /* A,b data */
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DETERMINE, PETSC_DETERMINE, user.n, user.n, NULL, &temp_mat));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetSeed(rctx, 1234));
  PetscCall(MatSetRandom(temp_mat, rctx));
  PetscCall(MatAssemblyBegin(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(temp_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatTransposeMatMult(temp_mat, temp_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &user.A));
  PetscCall(PetscRandomSetSeed(rctx, 5678));
  PetscCall(VecSetRandom(user.b, rctx));
  PetscCall(MatDestroy(&temp_mat));
  /* y: random vec */
  PetscCall(PetscRandomSetSeed(rctx, 9012));
  PetscCall(VecSetRandom(user.y, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  /* tao      = 0.5 x^T A x - b^T x
   * tao_full = 0.5 x^T A x - b^T x + 0.5 \|x-y\|_2^2 */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao_full));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &fdm_subtao));
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &gdm_subtao));

  PetscCall(TaoSetType(tao, TAOCG));
  PetscCall(TaoSetType(tao_full, TAOCG));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetSolution(tao_full, x_full));
  PetscCall(TaoSetOptionsPrefix(tao, "added_"));
  PetscCall(TaoSetOptionsPrefix(tao_full, "normal_"));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSetFromOptions(tao_full));
  PetscCall(TaoSetObjectiveAndGradient(tao_full, NULL, FullUserObjGrad, (void *)&user));

  /* Sketch: try to set DM for main objective */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &fdm));
  PetscCall(DMTaoSetType(fdm, DMTAOSHELL));
  PetscCall(DMTaoSetFromOptions(fdm));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &gdm));

  /* f(x) form */
  switch (user.mainFormType) {
  case MAIN_FORM_OBJGRAD: {
    switch (user.mainComposeType) {
    case COMPOSE_MAIN_VIA_TAO:
      PetscCall(TaoSetObjectiveAndGradient(tao, NULL, UserObjGrad, (void *)&user));
      break;
    case COMPOSE_MAIN_VIA_DM:
      PetscCall(DMTaoSetObjectiveAndGradient(fdm, UserObjGrad_DM, (void *)&user));
      PetscCall(TaoAddDM(tao, fdm, 1.));
      break;
    case COMPOSE_MAIN_VIA_DM_SUBTAO:
      PetscCall(TaoSetObjectiveAndGradient(fdm_subtao, NULL, UserObjGrad, (void *)&user));
      PetscCall(DMTaoUseTaoRoutines(fdm, fdm_subtao));
      PetscCall(TaoAddDM(tao, fdm, 1.));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid main problem formulation type.");
    }
  } break;
  case MAIN_FORM_OBJ_AND_GRAD: {
    switch (user.mainComposeType) {
    case COMPOSE_MAIN_VIA_TAO:
      PetscCall(TaoSetObjective(tao, UserObj, (void *)&user));
      PetscCall(TaoSetGradient(tao, NULL, UserGrad, (void *)&user));
      break;
    case COMPOSE_MAIN_VIA_DM:
      PetscCall(DMTaoSetObjective(fdm, UserObj_DM, (void *)&user));
      PetscCall(DMTaoSetGradient(fdm, UserGrad_DM, (void *)&user));
      PetscCall(TaoAddDM(tao, fdm, 1.));
      break;
    case COMPOSE_MAIN_VIA_DM_SUBTAO:
      PetscCall(TaoSetObjective(fdm_subtao, UserObj, (void *)&user));
      PetscCall(TaoSetGradient(fdm_subtao, NULL, UserGrad, (void *)&user));
      PetscCall(DMTaoUseTaoRoutines(fdm, fdm_subtao));
      PetscCall(TaoAddDM(tao, fdm, 1.));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid main problem formulation type.");
    }
  } break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid main problem formulation type.");
  }

  /* Set DMTao */
  switch (user.regFormType) {
  case REG_FORM_OBJGRAD:
    PetscCall(DMTaoSetType(gdm, DMTAOSHELL));
    PetscCall(DMTaoSetObjectiveAndGradient(gdm, L2_ObjGrad, (void *)&user));
    break;
  case REG_FORM_OBJ_AND_GRAD:
    PetscCall(DMTaoSetType(gdm, DMTAOSHELL));
    PetscCall(DMTaoSetObjective(gdm, L2_Obj, (void *)&user));
    PetscCall(DMTaoSetGradient(gdm, L2_Grad, (void *)&user));
    break;
  case REG_FORM_BUILT_IN:
    PetscCall(DMTaoSetType(gdm, DMTAOL2));
    break;
  case REG_FORM_SUBTAO_OBJGRAD:
    PetscCall(DMTaoSetType(gdm, DMTAOSHELL));
    PetscCall(TaoSetObjectiveAndGradient(gdm_subtao, NULL, L2_ObjGrad_Tao, (void *)&user));
    PetscCall(DMTaoUseTaoRoutines(gdm, gdm_subtao));
    break;
  case REG_FORM_SUBTAO_OBJ_AND_GRAD:
    PetscCall(DMTaoSetType(gdm, DMTAOSHELL));
    PetscCall(TaoSetObjective(gdm_subtao, L2_Obj_Tao, (void *)&user));
    PetscCall(TaoSetGradient(gdm_subtao, NULL, L2_Grad_Tao, (void *)&user));
    PetscCall(DMTaoUseTaoRoutines(gdm, gdm_subtao));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Regularizer formulation type.");
  }

  PetscCall(DMTaoSetFromOptions(gdm));
  if (user.yvec) PetscCall(DMTaoSetCentralVector(gdm, user.y));

  switch (user.regComposeType) {
  case COMPOSE_REG_VIA_REG:
    PetscCall(TaoSetRegularizer(tao, gdm, user.stepsize));
    break;
  case COMPOSE_REG_VIA_ADD:
    PetscCall(TaoAddDM(tao, gdm, user.stepsize));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Regularizer composition type.");
  }

  /* Solve full version */
  /* Solve Regularizer version */
  PetscCall(TaoSolve(tao));
  PetscCall(TaoSolve(tao_full));

  /* Testing Regularizer version vs Full version */
  PetscCall(VecAXPY(x, -1., x_full));
  PetscCall(VecNorm(x, NORM_2, &vec_dist));
  if (vec_dist < 1.e-12) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Full TaoSolve: < 1.e-12\n"));
  } else {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "error between TaoSolve with Regularizer and Full TaoSolve: %e\n", (double)vec_dist));
  }

  PetscCall(DMDestroy(&fdm));
  PetscCall(DMDestroy(&gdm));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoDestroy(&tao_full));
  PetscCall(TaoDestroy(&fdm_subtao));
  PetscCall(TaoDestroy(&gdm_subtao));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x_full));
  PetscCall(VecDestroy(&user.y));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.workvec));
  PetscCall(MatDestroy(&user.A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: {{1 2 4}}
      args: -main_compose {{use_dm use_dm_subtao use_tao}} -main_form {{objgrad objandgrad}} -reg_compose {{set_reg add_dm}} -reg_form {{objgrad objandgrad builtin subtao_objgrad subtao_obj_and_grad}} -yvec {{0 1}}
      output_file: output/cg_reg_000.out
      requires: !single

TEST*/
