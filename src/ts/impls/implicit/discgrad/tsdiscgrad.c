/*
  Code for timestepping with discrete gradient integrators
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdm.h>

PetscBool  DGCite       = PETSC_FALSE;
const char DGCitation[] = "@article{Gonzalez1996,\n"
                          "  title   = {Time integration and discrete Hamiltonian systems},\n"
                          "  author  = {Oscar Gonzalez},\n"
                          "  journal = {Journal of Nonlinear Science},\n"
                          "  volume  = {6},\n"
                          "  pages   = {449--467},\n"
                          "  doi     = {10.1007/978-1-4612-1246-1_10},\n"
                          "  year    = {1996}\n}\n";

typedef struct {
  PetscReal stage_time;
  Vec       X0, X, Xdot;
  void     *funcCtx;
  PetscBool gonzalez;
  PetscBool runningJacobian;
  PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *);
  PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *);
  PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *);
  PetscErrorCode (*userSMat)(TS, PetscInt, PetscInt, Mat*);
} TS_DiscGrad;

static PetscErrorCode TSDiscGradGetX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSDiscGrad_X0", X0));
    else *X0 = ts->vec_sol;
  }
  if (Xdot) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSDiscGrad_Xdot", Xdot));
    else *Xdot = dg->Xdot;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGradRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSDiscGrad_X0", X0));
  }
  if (Xdot) {
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSDiscGrad_Xdot", Xdot));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCoarsenHook_TSDiscGrad(DM fine, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMRestrictHook_TSDiscGrad(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec X0, Xdot, X0_c, Xdot_c;

  PetscFunctionBegin;
  PetscCall(TSDiscGradGetX0AndXdot(ts, fine, &X0, &Xdot));
  PetscCall(TSDiscGradGetX0AndXdot(ts, coarse, &X0_c, &Xdot_c));
  PetscCall(MatRestrict(restrct, X0, X0_c));
  PetscCall(MatRestrict(restrct, Xdot, Xdot_c));
  PetscCall(VecPointwiseMult(X0_c, rscale, X0_c));
  PetscCall(VecPointwiseMult(Xdot_c, rscale, Xdot_c));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, fine, &X0, &Xdot));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, coarse, &X0_c, &Xdot_c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainHook_TSDiscGrad(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainRestrictHook_TSDiscGrad(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec X0, Xdot, X0_sub, Xdot_sub;

  PetscFunctionBegin;
  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(TSDiscGradGetX0AndXdot(ts, subdm, &X0_sub, &Xdot_sub));

  PetscCall(VecScatterBegin(gscat, X0, X0_sub, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat, X0, X0_sub, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecScatterBegin(gscat, Xdot, Xdot_sub, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat, Xdot, Xdot_sub, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, subdm, &X0_sub, &Xdot_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_DiscGrad(TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  DM           dm;

  PetscFunctionBegin;
  if (!dg->X) PetscCall(VecDuplicate(ts->vec_sol, &dg->X));
  if (!dg->X0) PetscCall(VecDuplicate(ts->vec_sol, &dg->X0));
  if (!dg->Xdot) PetscCall(VecDuplicate(ts->vec_sol, &dg->Xdot));

  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMCoarsenHookAdd(dm, DMCoarsenHook_TSDiscGrad, DMRestrictHook_TSDiscGrad, ts));
  PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_TSDiscGrad, DMSubDomainRestrictHook_TSDiscGrad, ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_DiscGrad(TS ts, PetscOptionItems *PetscOptionsObject)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Discrete Gradients ODE solver options");
  {
    PetscCall(PetscOptionsBool("-ts_discgrad_gonzalez", "Use Gonzalez term in discrete gradients formulation", "TSDiscGradUseGonzalez", dg->gonzalez, &dg->gonzalez, NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_DiscGrad(TS ts, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  Discrete Gradients\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGradIsGonzalez_DiscGrad(TS ts, PetscBool *gonzalez)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  *gonzalez = dg->gonzalez;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGradUseGonzalez_DiscGrad(TS ts, PetscBool flg)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  dg->gonzalez = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_DiscGrad(TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&dg->X));
  PetscCall(VecDestroy(&dg->X0));
  PetscCall(VecDestroy(&dg->Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_DiscGrad(TS ts)
{
  DM dm;

  PetscFunctionBegin;
  PetscCall(TSReset_DiscGrad(ts));
  PetscCall(TSGetDM(ts, &dm));
  if (dm) {
    PetscCall(DMCoarsenHookRemove(dm, DMCoarsenHook_TSDiscGrad, DMRestrictHook_TSDiscGrad, ts));
    PetscCall(DMSubDomainHookRemove(dm, DMSubDomainHook_TSDiscGrad, DMSubDomainRestrictHook_TSDiscGrad, ts));
  }
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradGetFormulation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradSetFormulation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradIsGonzalez_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradUseGonzalez_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_DiscGrad(TS ts, PetscReal t, Vec X)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  PetscReal    dt = t - ts->ptime;

  PetscFunctionBegin;
  PetscCall(VecCopy(ts->vec_sol, dg->X));
  PetscCall(VecWAXPY(X, dt, dg->Xdot, dg->X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGrad_SNESSolve(TS ts, Vec b, Vec x)
{
  SNES     snes;
  PetscInt nits, lits;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESSolve(snes, b, x));
  PetscCall(SNESGetIterationNumber(snes, &nits));
  PetscCall(SNESGetLinearSolveIterations(snes, &lits));
  ts->snes_its += nits;
  ts->ksp_its += lits;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_DiscGrad(TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  TSAdapt      adapt;
  TSStepStatus status     = TS_STEP_INCOMPLETE;
  PetscInt     rejections = 0;
  PetscBool    stageok, accept = PETSC_TRUE;
  PetscReal    next_time_step = ts->time_step;

  PetscFunctionBegin;
  PetscCall(TSGetAdapt(ts, &adapt));
  if (!ts->steprollback) PetscCall(VecCopy(ts->vec_sol, dg->X0));

  while (!ts->reason && status != TS_STEP_COMPLETE) {
    PetscReal shift = 1 / (0.5 * ts->time_step);

    dg->stage_time = ts->ptime + 0.5 * ts->time_step;

    PetscCall(VecCopy(dg->X0, dg->X));
    PetscCall(TSPreStage(ts, dg->stage_time));
    PetscCall(TSDiscGrad_SNESSolve(ts, NULL, dg->X));
    PetscCall(TSPostStage(ts, dg->stage_time, 0, &dg->X));
    PetscCall(TSAdaptCheckStage(adapt, ts, dg->stage_time, dg->X, &stageok));
    if (!stageok) goto reject_step;

    status = TS_STEP_PENDING;
    PetscCall(VecAXPBYPCZ(dg->Xdot, -shift, shift, 0, dg->X0, dg->X));
    PetscCall(VecAXPY(ts->vec_sol, ts->time_step, dg->Xdot));
    PetscCall(TSAdaptChoose(adapt, ts, ts->time_step, NULL, &next_time_step, &accept));
    status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      PetscCall(VecCopy(dg->X0, ts->vec_sol));
      ts->time_step = next_time_step;
      goto reject_step;
    }
    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++;
    accept = PETSC_FALSE;
    if (!ts->reason && ts->max_reject >= 0 && ++rejections > ts->max_reject) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, rejections));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSGetStages_DiscGrad(TS ts, PetscInt *ns, Vec **Y)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  if (ns) *ns = 1;
  if (Y) *Y = &dg->X;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGradFormSMat_Default(PetscInt n, PetscInt m, Mat *S)
{

  PetscFunctionBegin;
  PetscCall(MatCreate(PETSC_COMM_WORLD, S));
  PetscCall(MatSetSizes(*S, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(*S));
  PetscCall(MatSetUp(*S));
  PetscCall(MatAssemblyBegin(*S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDiscGradSetSMat(TS ts, PetscErrorCode (*userSMat)(TS, PetscInt, PetscInt, Mat*)){
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  PetscFunctionBegin;
  dg->userSMat = userSMat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
    G(U) = F[t0 + 0.5*dt, U, (U-U0)/dt] = 0
*/

/* x = (x+x')/2 */
/* NEED TO CALCULATE x_{n+1} from x and x_{n}*/
static PetscErrorCode SNESTSFormFunction_DiscGrad(SNES snes, Vec x, Vec y, TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  PetscReal    norm, shift = 1 / (0.5 * ts->time_step);
  PetscInt     n, dbg = 0, taylor = 0;
  Vec          X0, Xdot, Xp, Xdiff, unit;
  Mat          S;
  PetscBool    runningJacobian;
  PetscScalar  F = 0, F0 = 0, Gp;
  Vec          G, SgF;
  DM           dm, dmsave;
  PetscFunctionBegin;

  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(VecDuplicate(y, &Xp));
  PetscCall(VecDuplicate(y, &Xdiff));
  PetscCall(VecDuplicate(y, &SgF));
  PetscCall(VecDuplicate(y, &G));

  PetscCall(VecDuplicate(y, &unit));
  PetscCall(VecZeroEntries(unit));
  PetscCall(VecShift(unit, 1.));
  //PetscPrintf(PETSC_COMM_WORLD, "shift %.16g\n", shift);
  PetscCall(VecGetLocalSize(y, &n));
  if (!dg->userSMat) PetscCall(TSDiscGradFormSMat_Default(n, n, &S));
  else PetscCall((*dg->userSMat)(ts, n, n, &S));
  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(VecAXPBYPCZ(Xdot, -shift, shift, 0, X0, x)); /* Xdot = shift (x - X0) */

  PetscCall(VecAXPBYPCZ(Xp, -1, 2, 0, X0, x));     /* Xp = 2*x - X0 + (0)*Xp */
  PetscCall(VecAXPBYPCZ(Xdiff, -1, 1, 0, X0, Xp)); /* Xdiff = xp - X0 + (0)*Xdiff */
  // hack
  //PetscCall(VecAXPBYPCZ(Xdiff, -1, 1, 0, X0, x)); /* Xdiff = xp - X0 + (0)*Xdiff */
  VecViewFromOptions(Xdiff, NULL, "-xdiff_view");
  if (dg->gonzalez & taylor){
    Vec taylorX, taylorXp, taylorX0, taylorG;
    Vec         du, uhat, r, rhat, df;
    PetscReal   h;
    PetscReal   taylorF0, taylorF;
    PetscReal  *es, *hs, *errors;
    PetscReal   hMax = 1.0, hMin = 1e-6, hMult = 0.1;
    PetscInt    Nv, v;
    
    PetscCall(VecDuplicate(G, &taylorG));
    PetscCall(VecDuplicate(x, &taylorX));
    PetscCall(VecDuplicate(Xp, &taylorXp));
    PetscCall(VecDuplicate(X0, &taylorX0));

    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv)
    ;
    PetscCall(PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors));
    /*
      F(x+\epsilon h) \approx F(x) - \epsilon \nabla F(x+h) \\
      \frac{ F(x+\epsilon h - F(x) - \epsilon \nabla F(x+h))}{\epsilon} = O(\epsilon^2)

      so compute F shifted by perturbation, F at x, and \epsilon nabla F(x+h), divide epsilon
      and plot the error term vs epsilon
    */
    Vec perturbation, taylorXhat;
    PetscRandom rnd;
    PetscReal   val;

    
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
    PetscCall(VecDuplicate(taylorX, &perturbation));
    PetscCall(VecDuplicate(taylorX, &taylorXhat));
    PetscCall(PetscRandomGetValueReal(rnd, &val));
    PetscCall(VecZeroEntries(perturbation));
    PetscCall(VecShift(perturbation, val));

    PetscCall(PetscRandomDestroy(&rnd));
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      
      PetscCall(VecWAXPY(taylorXhat, h, perturbation, taylorX));
      /* F(\hat u) \approx F(u) + J(u) (uhat - u) = F(u) + h * J(u) du */
      /* F(x+\epsilon h) \approx F(x) + \epsilon \nabla  F\cdot h*/
      // evaluate F(x+ \ epsilon h)
      PetscCall((*dg->Ffunc)(ts, dg->stage_time, taylorXhat, &taylorF, dg->funcCtx));
      // evaluate F(x)
      PetscCall((*dg->Ffunc)(ts, dg->stage_time, taylorX, &taylorF0, dg->funcCtx));
      // evaluate \nabla F(x + \epsilon h)
      PetscCall((*dg->Gfunc)(ts, dg->stage_time, taylorXhat, taylorG, dg->funcCtx));
      PetscReal taylorNablaF;
      // evaluate \nabla F(x + \epsilon h) \cdot (x_n+1 - x_n)
      PetscCall(VecDot(Xdiff, taylorG, &taylorNablaF));
      errors[Nv] = (taylorF - taylorF0 - taylorNablaF)/h;
      es[Nv] = PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    PetscReal tol = -1;
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL) break;
    }
    PetscBool isLin;
    if (v == Nv) isLin = PETSC_TRUE;
    PetscReal slope, intercept;
    PetscCall(PetscLinearRegression(Nv, hs, es, &slope, &intercept));
    PetscCall(PetscFree3(es, hs, errors));
    /* Slope should be about 2 */
    if (tol >= 0) {
      PetscCheck(isLin || PetscAbsReal(2 - slope) <= tol, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double)slope);
    } else {
      if (!isLin) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Taylor approximation converging at order %3.2f\n", (double)slope));
      else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Function appears to be linear\n"));
    }
  }
  if (dg->gonzalez) {
    PetscCall((*dg->Sfunc)(ts, dg->stage_time, x, S, dg->funcCtx));
    PetscCall((*dg->Ffunc)(ts, dg->stage_time, Xp, &F, dg->funcCtx));
    PetscCall((*dg->Ffunc)(ts, dg->stage_time, X0, &F0, dg->funcCtx));
    PetscCall((*dg->Gfunc)(ts, dg->stage_time, x, G, dg->funcCtx));
    /* Adding Extra Gonzalez Term */
    PetscCall(VecDot(Xdiff, G, &Gp));
    PetscCall(VecNorm(Xdiff, NORM_2, &norm));
    //PetscPrintf(PETSC_COMM_WORLD, "norm: %g\n", norm);
    if (norm < PETSC_SQRT_MACHINE_EPSILON) { //
      Gp = 0;
      if (dbg) PetscPrintf(PETSC_COMM_WORLD, "Gp IS ZERO\n");
    } else {
      if (dbg) PetscPrintf(PETSC_COMM_WORLD, "Gp/norm: %g\n", Gp/norm);

      /* Gp = (1/|xn+1 - xn|^2) * (F(xn+1) - F(xn) - Gp) */
      #if 0
      PetscPrintf(PETSC_COMM_WORLD, "F: %g\n", F);
      PetscPrintf(PETSC_COMM_WORLD, "F0: %g\n", F0);
      PetscPrintf(PETSC_COMM_WORLD, "Gp_pre: %g\n", Gp);
      PetscReal diffF = F-F0;
      PetscPrintf(PETSC_COMM_WORLD, "diffF: %g\n", diffF);
      #endif 
      Gp = (F - F0 - Gp) / PetscSqr(norm);// *1.e4
      if (dbg) {
        PetscReal deltaNorm;
        PetscReal diffFF0dn2;
        PetscReal GdDelta;
        diffFF0dn2 = (F-F0)/norm;
        PetscCall(VecDot(G, Xdiff, &GdDelta));
        PetscReal diff = diffFF0dn2 - GdDelta;

        PetscReal alphaddnorm;

        alphaddnorm = (F-F0-GdDelta)/(norm);

        PetscPrintf(PETSC_COMM_WORLD, "FF0/norm - grad F dot delta: %.16g\n", diff);
        PetscPrintf(PETSC_COMM_WORLD, "F-F0 - grad F dot delta/norm: %.16g\n", alphaddnorm);
        PetscPrintf(PETSC_COMM_WORLD, "Gp nonzero: %g, F: %g, F0, %g\n", Gp, F, F0);
        PetscPrintf(PETSC_COMM_WORLD, "norm: %g\n", norm);
      }
    }
    Vec test;
    PetscReal unitVecNorm, gradVecNorm;
    
    PetscCall(VecDuplicate(G, &test));
    PetscCall(VecZeroEntries(test));
    PetscCall(VecShift(test, 1.0));
    PetscCall(VecScale(test, Gp));
    PetscCall(VecNorm(test, NORM_2, &unitVecNorm));
    PetscCall(VecNorm(G, NORM_2, &gradVecNorm));
    //PetscPrintf(PETSC_COMM_WORLD, "||grad F(midpoint)||/||Gon()I||: %.16g        ||grad F(midpoint)||: %.16g         ||Gon()I||: %.16g\n", gradVecNorm/unitVecNorm, gradVecNorm, unitVecNorm);
    PetscCall(VecAXPY(G, Gp, Xdiff));
    PetscCall(MatMult(S, G, SgF)); /* S*gradF */
    VecDestroy(&test);
  } else {
    PetscCall((*dg->Sfunc)(ts, dg->stage_time, x, S, dg->funcCtx));
    PetscCall((*dg->Gfunc)(ts, dg->stage_time, x, G, dg->funcCtx));
    PetscCall(MatMult(S, G, SgF)); /* Xdot = S*gradF */
  }
  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(VecAXPBYPCZ(y, 1, -1, 0, Xdot, SgF));
  ts->dm = dmsave;
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(VecDestroy(&Xp));
  PetscCall(VecDestroy(&Xdiff));
  PetscCall(VecDestroy(&SgF));
  PetscCall(VecDestroy(&unit));
  PetscCall(VecDestroy(&G));
  PetscCall(MatDestroy(&S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormJacobian_DiscGrad(SNES snes, Vec x, Mat A, Mat B, TS ts)
{
  TS_DiscGrad *dg    = (TS_DiscGrad *)ts->data;
  PetscReal    shift = 1 / (0.5 * ts->time_step);
  Vec          Xdot;
  DM           dm, dmsave;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  /* Xdot has already been computed in SNESTSFormFunction_DiscGrad (SNES guarantees this) */
  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, NULL, &Xdot));

  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIJacobian(ts, dg->stage_time, x, Xdot, shift, A, B, PETSC_FALSE));
  ts->dm = dmsave;
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, NULL, &Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGradGetFormulation_DiscGrad(TS ts, PetscErrorCode (**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  *Sfunc = dg->Sfunc;
  *Ffunc = dg->Ffunc;
  *Gfunc = dg->Gfunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDiscGradSetFormulation_DiscGrad(TS ts, PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  dg->Sfunc   = Sfunc;
  dg->Ffunc   = Ffunc;
  dg->Gfunc   = Gfunc;
  dg->funcCtx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSDISCGRAD - ODE solver using the discrete gradients version of the implicit midpoint method

  Level: intermediate

  Notes:
  This is the implicit midpoint rule, with an optional term that guarantees the discrete
  gradient property. This timestepper applies to systems of the form $u_t = S(u) \nabla F(u)$
  where $S(u)$ is a linear operator, and $F$ is a functional of $u$.

.seealso: [](ch_ts), `TSCreate()`, `TSSetType()`, `TS`, `TSDISCGRAD`, `TSDiscGradSetFormulation()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_DiscGrad(TS ts)
{
  TS_DiscGrad *th;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(DGCitation, &DGCite));
  ts->ops->reset          = TSReset_DiscGrad;
  ts->ops->destroy        = TSDestroy_DiscGrad;
  ts->ops->view           = TSView_DiscGrad;
  ts->ops->setfromoptions = TSSetFromOptions_DiscGrad;
  ts->ops->setup          = TSSetUp_DiscGrad;
  ts->ops->step           = TSStep_DiscGrad;
  ts->ops->interpolate    = TSInterpolate_DiscGrad;
  ts->ops->getstages      = TSGetStages_DiscGrad;
  ts->ops->snesfunction   = SNESTSFormFunction_DiscGrad;
  ts->ops->snesjacobian   = SNESTSFormJacobian_DiscGrad;
  ts->default_adapt_type  = TSADAPTNONE;

  ts->usessnes = PETSC_TRUE;

  PetscCall(PetscNew(&th));
  ts->data = (void *)th;

  th->gonzalez = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradGetFormulation_C", TSDiscGradGetFormulation_DiscGrad));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradSetFormulation_C", TSDiscGradSetFormulation_DiscGrad));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradIsGonzalez_C", TSDiscGradIsGonzalez_DiscGrad));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradUseGonzalez_C", TSDiscGradUseGonzalez_DiscGrad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSDiscGradGetFormulation - Get the construction method for S, F, and grad F from the
  formulation $u_t = S \nabla F$ for `TSDISCGRAD`

  Not Collective

  Input Parameter:
. ts - timestepping context

  Output Parameters:
+ Sfunc - constructor for the S matrix from the formulation
. Ffunc - functional F from the formulation
. Gfunc - constructor for the gradient of F from the formulation
- ctx   - the user context

  Calling sequence of `Sfunc`:
+ ts   - the integrator
. time - the current time
. u    - the solution
. S    - the S-matrix from the formulation
- ctx  - the user context

  Calling sequence of `Ffunc`:
+ ts   - the integrator
. time - the current time
. u    - the solution
. F    - the computed function from the formulation
- ctx  - the user context

  Calling sequence of `Gfunc`:
+ ts   - the integrator
. time - the current time
. u    - the solution
. G    - the gradient of the computed function from the formulation
- ctx  - the user context

  Level: intermediate

.seealso: [](ch_ts), `TS`, `TSDISCGRAD`, `TSDiscGradSetFormulation()`
@*/
PetscErrorCode TSDiscGradGetFormulation(TS ts, PetscErrorCode (**Sfunc)(TS ts, PetscReal time, Vec u, Mat S, void *ctx), PetscErrorCode (**Ffunc)(TS ts, PetscReal time, Vec u, PetscScalar *F, void *ctx), PetscErrorCode (**Gfunc)(TS ts, PetscReal time, Vec u, Vec G, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(Sfunc, 2);
  PetscAssertPointer(Ffunc, 3);
  PetscAssertPointer(Gfunc, 4);
  PetscUseMethod(ts, "TSDiscGradGetFormulation_C", (TS, PetscErrorCode(**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode(**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode(**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *), (ts, Sfunc, Ffunc, Gfunc, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TSDiscGradSetFormulation - Set the construction method for S, F, and grad F from the
  formulation $u_t = S(u) \nabla F(u)$ for `TSDISCGRAD`

  Not Collective

  Input Parameters:
+ ts    - timestepping context
. Sfunc - constructor for the S matrix from the formulation
. Ffunc - functional F from the formulation
. Gfunc - constructor for the gradient of F from the formulation
- ctx   - optional context for the functions

  Calling sequence of `Sfunc`:
+ ts   - the integrator
. time - the current time
. u    - the solution
. S    - the S-matrix from the formulation
- ctx  - the user context

  Calling sequence of `Ffunc`:
+ ts   - the integrator
. time - the current time
. u    - the solution
. F    - the computed function from the formulation
- ctx  - the user context

  Calling sequence of `Gfunc`:
+ ts   - the integrator
. time - the current time
. u    - the solution
. G    - the gradient of the computed function from the formulation
- ctx  - the user context

  Level: intermediate

.seealso: [](ch_ts), `TSDISCGRAD`, `TSDiscGradGetFormulation()`
@*/
PetscErrorCode TSDiscGradSetFormulation(TS ts, PetscErrorCode (*Sfunc)(TS ts, PetscReal time, Vec u, Mat S, void *ctx), PetscErrorCode (*Ffunc)(TS ts, PetscReal time, Vec u, PetscScalar *F, void *ctx), PetscErrorCode (*Gfunc)(TS ts, PetscReal time, Vec u, Vec G, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidFunction(Sfunc, 2);
  PetscValidFunction(Ffunc, 3);
  PetscValidFunction(Gfunc, 4);
  PetscTryMethod(ts, "TSDiscGradSetFormulation_C", (TS, PetscErrorCode(*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode(*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode(*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *), (ts, Sfunc, Ffunc, Gfunc, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSDiscGradIsGonzalez - Checks flag for whether to use additional conservative terms in
  discrete gradient formulation for `TSDISCGRAD`

  Not Collective

  Input Parameter:
. ts - timestepping context

  Output Parameter:
. gonzalez - `PETSC_TRUE` when using the Gonzalez term

  Level: advanced

.seealso: [](ch_ts), `TSDISCGRAD`, `TSDiscGradUseGonzalez()`
@*/
PetscErrorCode TSDiscGradIsGonzalez(TS ts, PetscBool *gonzalez)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscAssertPointer(gonzalez, 2);
  PetscUseMethod(ts, "TSDiscGradIsGonzalez_C", (TS, PetscBool *), (ts, gonzalez));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSDiscGradUseGonzalez - Sets discrete gradient formulation with or without additional
  conservative terms.

  Not Collective

  Input Parameters:
+ ts  - timestepping context
- flg - `PETSC_TRUE` to use the Gonzalez term

  Options Database Key:
. -ts_discgrad_gonzalez <flg> - use the Gonzalez term for the discrete gradient formulation

  Level: intermediate

  Notes:
  Without `flg`, the discrete gradients timestepper is just backwards Euler.

.seealso: [](ch_ts), `TSDISCGRAD`
@*/
PetscErrorCode TSDiscGradUseGonzalez(TS ts, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSDiscGradUseGonzalez_C", (TS, PetscBool), (ts, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}
