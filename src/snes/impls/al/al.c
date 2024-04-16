#include <../src/snes/impls/al/alimpl.h>

/*
     This file implements a truncated Newton method with arc length continuation,
     for solving a system of nonlinear equations, using the KSP, Vec,
     and Mat interfaces for linear solvers, vectors, and matrices,
     respectively.
*/
const char *const SNESNewtonALCorrectionTypes[] = {"EXACT", "NORMAL", "SNESNewtonALCorrectionType", "SNES_NEWTONAL_CORRECTION_", NULL};

static PetscErrorCode SNESNewtonALCheckArcLength(SNES snes, Vec XStep, PetscReal lambdaStep, PetscReal stepSize)
{
  PetscReal      arcLength, arcLengthError;
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  PetscCall(VecDot(XStep, XStep, &arcLength));
  arcLength += al->psisq * lambdaStep * lambdaStep;
  arcLength      = PetscSqrtReal(arcLength);
  arcLengthError = PetscAbsReal(arcLength - stepSize);

  if (arcLengthError > 1.0e-6) PetscCall(PetscInfo(snes, "Arc length differs from specified step size: computed=%18.16e, expected=%18.16e, error=%18.16e \n", (double)arcLength, (double)stepSize, (double)arcLengthError));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  SNESNewtonALSetCorrectionType - Set the type of correction to use in the arc-length continuation method.

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
- ctype - the type of correction to use

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESNewtonALCorrectionType`
@*/
PetscErrorCode SNESNewtonALSetCorrectionType(SNES snes, SNESNewtonALCorrectionType ctype)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  al->correction_type = ctype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESSetNewtonAL - Sets a user function that is called at each function evaluation to
  compute the tangent load vector for the arc-length continuation method. The tangent load vector is the
  partial derivative of external load with respect to the load parameter. In the case of proportional 
  loading, the tangent load vector is the full external load vector at the end of the load step.

  The signature of the user provided method is
.vb
   PetscErrorCode your_tangent_load_method(SNES snes, Vec U, Vec Q, void* ctx);
.ve
  where `snes` is the nonlinear solver object, `U` is the current solution vector, `Q` is the output tangent 
  load vector, and `ctx` is the user-defined context passed to this function. If the current value of the
  load parameter is needed, it can be obtained with `SNESNewtonALGetLoadParameter()`.

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
. func - [optional] tangent load function evaluation routine
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESNEWTONALGetTangentLoadFunction()`, `SNESNewtonALGetLoadParameter()`
@*/
PetscErrorCode SNESSetNewtonAL(SNES snes, SNESFunctionFn *func, void *ctx)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMSNESSetNewtonAL(dm, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESGetNewtonAL - Get the user function and context set with `SNESSetNewtonAL`

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
. func - [optional] tangent load function evaluation routine, see `SNESSetNewtonAL()` for the signature
- ctx  - [optional] user-defined context for private data for the function evaluation routine (may be `NULL`)

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESSetNewtonAL()`
@*/
PetscErrorCode SNESGetNewtonAL(SNES snes, SNESFunctionFn **func, void **ctx)
{
  DM dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMSNESGetNewtonAL(dm, func, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESNewtonALGetLoadParameter - Get the value of the load parameter `lambda` for the arc-length continuation method.
  This function should be used in the functions provided to `SNESSetFunction()` and `SNESSetNewtonAL()`
  to compute the residual and tangent load vectors for a given value of `lambda` (0 <= lambda <= 1). 
  Usually, `lambda` is used to scale the external force vector in the residual function, i.e. proportional loading,
  in which case the tangent load vector is the full external force vector.

  Logically Collective

  Input Parameters:
+ snes - the nonlinear solver object
- lambda - the arc-length parameter

  Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONAL`, `SNESSetNewtonAL()`
@*/
PetscErrorCode SNESNewtonALGetLoadParameter(SNES snes, PetscReal *lambda)
{
  SNES_NEWTONAL *al;

  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  al = (SNES_NEWTONAL *)snes->data;
  PetscAssertPointer(lambda, 2);
  *lambda = al->lambda;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESNewtonALScaleRHS(SNES snes)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;
  PetscReal      lambda;

  PetscFunctionBegin;
  if (!snes->vec_rhs) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(SNESNewtonALGetLoadParameter(snes, &lambda));
  if (!al->vec_rhs_orig) {
    PetscCall(VecDuplicate(snes->vec_rhs, &al->vec_rhs_orig));
    PetscCall(VecCopy(snes->vec_rhs, al->vec_rhs_orig));
    PetscCall(VecScale(snes->vec_rhs, lambda));
  } else {
    PetscCall(VecAXPBY(snes->vec_rhs, lambda, 0.0, al->vec_rhs_orig));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESComputeNewtonAL - Calls the function that has been set with `SNESSetNewtonAL()`.

  Collective

  Input Parameters:
+ snes - the `SNES` context
- X    - input vector

  Output Parameter:
. Q - tangent load vector, as set by `SNESSetNewtonAL()`

  Level: developer

  Notes:
  `SNESComputeNewtonAL()` is typically used within nonlinear solvers
  implementations, so users would not generally call this routine themselves.

.seealso: [](ch_snes), `SNES`, `SNESSetNewtonAL()`, `SNESGetNewtonAL()`
@*/
PetscErrorCode SNESComputeNewtonAL(SNES snes, Vec X, Vec Q)
{
  DM              dm;
  DMSNES          sdm;
  void           *ctx;
  SNESFunctionFn *computealfunction;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Q, VEC_CLASSID, 3);
  PetscCheckSameComm(snes, 1, X, 2);
  PetscCheckSameComm(snes, 1, Q, 3);
  PetscCall(VecValidValues_Internal(X, 2, PETSC_TRUE));

  PetscCall(PetscLogEventBegin(SNES_NewtonALEval, snes, X, Q, 0));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetDMSNES(dm, &sdm));
  PetscCall(DMSNESGetFunction(dm, &computealfunction, &ctx));
  PetscCheck(computealfunction || snes->vec_rhs, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "No tangent load function or rhs vector has been set");
  if (computealfunction) {
    PetscCall(VecLockReadPush(X));
    PetscCallBack("SNES callback NewtonAL tangent load function", (*computealfunction)(snes, X, Q, ctx));
    PetscCall(VecLockReadPop(X));
  }
  if (snes->vec_rhs) {
    SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

    PetscCheck(al->vec_rhs_orig, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "No original rhs vector has been set");
    PetscCall(VecAXPBY(Q, 1, 0, al->vec_rhs_orig));
  }
  PetscCall(PetscLogEventEnd(SNES_NewtonALEval, snes, X, Q, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  SNESSolve_NEWTONAL - Solves a nonlinear system with an arc-length continuation method.

  Input Parameter:
. snes - the SNES context
@*/
static PetscErrorCode SNESSolve_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *data = (SNES_NEWTONAL *)snes->data;
  PetscInt       maxits, maxincs, lits;
  PetscReal      fnorm, xnorm, ynorm, stepSize;
  Vec            oldDeltaX, DeltaX, deltaX, X, R, Q, deltaX_Q, deltaX_R, W;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits    = snes->max_its;        /* maximum number of iterations */
  maxincs   = data->max_steps;      /* maximum number of increments */
  X         = snes->vec_sol;        /* solution vector */
  R         = snes->vec_func;       /* residual vector */
  Q         = snes->work[0];        /* tangent load vector */
  deltaX_Q  = snes->work[1];        /* variation of X with respect to lambda */
  deltaX_R  = snes->work[2];        /* linearized error correction */
  W         = snes->work[3];        /* work vector */
  DeltaX    = snes->work[4];        /* step from equilibrium */
  oldDeltaX = snes->work[5];        /* old step from equilibrium */
  deltaX    = snes->vec_sol_update; /* full newton step */
  stepSize  = data->step_size;      /* initial step size */

  PetscCall(VecZeroEntries(DeltaX));

  /* main incremental-iterative loop */
  for (PetscInt i = 0; i < maxincs; i++) {
    PetscReal deltaLambda, oldLambdaUpdate = data->lambda_update;
    if (i > 0) PetscCall(VecCopy(DeltaX, oldDeltaX));

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = 0;
    snes->norm = 0.0;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESNewtonALScaleRHS(snes));
    PetscCall(SNESComputeFunction(snes, X, R));
    PetscCall(SNESComputeNewtonAL(snes, X, Q));
    PetscCall(VecWAXPY(W, 1, R, Q)); /* W <- R + Q */

    PetscCall(VecNorm(W, NORM_2, &fnorm)); /* fnorm <- ||F||  */
    SNESCheckFunctionNorm(snes, fnorm);

    /* Monitor convergence */
    PetscCall(SNESConverged(snes, 0, 0.0, 0.0, fnorm));
    PetscCall(SNESMonitor(snes, 0, fnorm));
    if (i == 0 && snes->reason) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(VecZeroEntries(deltaX_Q));
    PetscCall(VecZeroEntries(deltaX_R));
    PetscCall(VecZeroEntries(deltaX));

    for (PetscInt j = 0; j < maxits; j++) {
      PetscReal normsqX_Q;

      /* Call general purpose update function */
      PetscTryTypeMethod(snes, update, snes->iter);

      PetscCall(SNESComputeJacobian(snes, X, snes->jacobian, snes->jacobian_pre));
      SNESCheckJacobianDomainerror(snes);
      PetscCall(KSPSetOperators(snes->ksp, snes->jacobian, snes->jacobian_pre));
      /* Solve J deltaX_Q = Q, where J is Jacobian matrix */
      PetscCall(KSPSolve(snes->ksp, Q, deltaX_Q));
      SNESCheckKSPSolve(snes);
      PetscCall(KSPGetIterationNumber(snes->ksp, &lits));
      PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", tangent load linear solve iterations=%" PetscInt_FMT "\n", snes->iter, lits));
      /* Compute load parameter variation */
      PetscCall(VecDot(deltaX_Q, deltaX_Q, &normsqX_Q));
      /* On first iter, use predictor */
      if (j == 0) {
        PetscReal sign = 1.0;
        if (i > 0) {
          PetscCall(VecDot(DeltaX, deltaX_Q, &sign));
          sign += data->psisq * data->lambda_update;
          sign = sign >= 0 ? 1.0 : -1.0;
        }
        data->lambda_update = 0.0;
        PetscCall(VecZeroEntries(DeltaX));
        deltaLambda = sign * stepSize / PetscSqrtReal(normsqX_Q + data->psisq);
        PetscCall(VecNorm(Q, NORM_2, &fnorm));
      } else {
        /* Solve J deltaX_R = -R */
        PetscCall(KSPSolve(snes->ksp, R, deltaX_R));
        SNESCheckKSPSolve(snes);
        PetscCall(KSPGetIterationNumber(snes->ksp, &lits));
        PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", residual linear solve iterations=%" PetscInt_FMT "\n", snes->iter, lits));
        PetscCall(VecScale(deltaX_R, -1));
        PetscCall(VecNorm(R, NORM_2, &fnorm));

        if (data->correction_type == SNES_NEWTONAL_CORRECTION_NORMAL) {
          /* Solve a*deltaLambda^2 + b*deltaLambda + c = 0 */
          PetscReal dot1, dot2;

          PetscCall(VecDot(DeltaX, deltaX_R, &dot1));
          PetscCall(VecDot(DeltaX, deltaX_Q, &dot2));
          deltaLambda = -dot1 * (1 / (dot2 + data->psisq * data->lambda_update));
        } else {
          /* Solve a*deltaLambda^2 + b*deltaLambda + c = 0 */
          PetscReal a, b, c, psisqLambdaUpdate, discriminant;

          psisqLambdaUpdate = data->psisq * data->lambda_update;
          a                 = normsqX_Q + data->psisq;
          PetscCall(VecWAXPY(W, data->delta_s, deltaX_R, DeltaX));
          PetscCall(VecDot(deltaX_Q, W, &b));
          b = 2.0 * (b + psisqLambdaUpdate);
          PetscCall(VecDot(W, W, &c));
          c = c + psisqLambdaUpdate * data->lambda_update - stepSize * stepSize;

          discriminant = b * b - 4.0 * a * c;
          PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", a=%18.16e, b=%18.16e, c=%18.16e, discriminant=%18.16e\n", snes->iter, (double)a, (double)b, (double)c, (double)discriminant));
          if (discriminant < 0) {
            /* If the discriminant is negative, we have only complex roots. Shrink step size and retry step. */
            stepSize *= 0.5;
            PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", discriminant=%18.16e < 0, shrinking step size to %18.16e\n", snes->iter, (double)discriminant, (double)stepSize));
            snes->reason = SNES_DIVERGED_AL_STEP_SIZE;
            break;
          } else {
            PetscReal dlambda1, dlambda2, pmpart, t;

            pmpart   = PetscSqrtReal(discriminant);
            dlambda1 = (-b - pmpart) / (2.0 * a);
            dlambda2 = (-b + pmpart) / (2.0 * a);
            PetscCall(VecDot(DeltaX, deltaX_Q, &t));
            t           = t + psisqLambdaUpdate;
            deltaLambda = t * dlambda1 > t * dlambda2 ? dlambda1 : dlambda2;
          }
        }
      }
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
      data->lambda_update = data->lambda_update + deltaLambda;
      data->lambda        = data->lambda + deltaLambda;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
      PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", lambda=%18.16e, lambda_update=%18.16e\n", snes->iter, (double)data->lambda, (double)data->lambda_update));
      PetscCall(VecAXPBYPCZ(deltaX, data->delta_s, deltaLambda, 0, deltaX_R, deltaX_Q));
      PetscCall(VecAXPY(DeltaX, 1, deltaX));
      PetscCall(VecAXPY(X, 1, deltaX));
      PetscCall(SNESNewtonALScaleRHS(snes));
      PetscCall(SNESComputeFunction(snes, X, R));
      PetscCall(SNESComputeNewtonAL(snes, X, Q));
      PetscCall(VecNorm(R, NORM_2, &fnorm));
      PetscCall(VecNorm(X, NORM_2, &xnorm));
      PetscCall(VecNorm(deltaX, NORM_2, &ynorm));

      if (PetscLogPrintInfo) PetscCall(SNESNewtonALCheckArcLength(snes, DeltaX, data->lambda_update, stepSize));

      /* Monitor convergence */
      PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
      snes->iter  = j + 1;
      snes->norm  = fnorm;
      snes->ynorm = ynorm;
      snes->xnorm = xnorm;
      PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
      PetscCall(SNESLogConvergenceHistory(snes, snes->norm, lits));
      /* Test for convergence */
      PetscCall(SNESConverged(snes, snes->iter, xnorm, ynorm, fnorm));
      PetscCall(SNESMonitor(snes, snes->iter, snes->norm));
      if (snes->reason) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)snes), "SNES solve converged reason %s\n", SNESConvergedReasons[snes->reason]));
        break;
      }
    }
    if (snes->reason == SNES_DIVERGED_AL_STEP_SIZE) {
      /* restarting step */
      i = i - 1;
      data->lambda -= data->lambda_update;
      data->lambda_update = 0;
      PetscCall(VecAXPY(X, -1, DeltaX));
      if (i > 0) PetscCall(VecCopy(oldDeltaX, DeltaX));
      else PetscCall(VecZeroEntries(DeltaX));
      if (stepSize < data->min_step_size) break;
      snes->reason = SNES_CONVERGED_ITERATING;
      continue;
    } else if (snes->reason < 0) {
      PetscCall(PetscInfo(snes, "SNES solve failed reason %s\n", SNESConvergedReasons[snes->reason]));
      break;
    }
    if (data->lambda >= 1.0) {
      snes->iter = i;
      break;
    } else if (i == maxincs - 1) {
      snes->reason = SNES_DIVERGED_MAX_IT;
      break;
    } else {
      snes->reason = SNES_CONVERGED_ITERATING;
      snes->iter   = 0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESSetUp_NEWTONAL - Sets up the internal data structures for the later use
   of the SNESNEWTONAL nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

 */
static PetscErrorCode SNESSetUp_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *data = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESSetWorkVecs(snes, 6));
  data->lambda_update = 0.0;
  data->lambda        = 0.0;
  data->delta_s       = 1.0;
  PetscCall(SNESSetUpMatrices(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESSetFromOptions_NEWTONAL - Sets various parameters for the SNESNEWTONAL method.

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_NEWTONAL(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  SNES_NEWTONAL *data = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SNES Newton Arc Length options");
  data->step_size = 1.0;
  PetscCall(PetscOptionsReal("-snes_newtonal_step_size", "Initial arc length increment step size", "SNESNewtonAL", data->step_size, &data->step_size, NULL));
  data->max_steps = 100;
  PetscCall(PetscOptionsInt("-snes_newtonal_max_steps", "Maximum number of increment steps", "SNESNewtonAL", data->max_steps, &data->max_steps, NULL));
  data->min_step_size = 1.0e-6;
  PetscCall(PetscOptionsReal("-snes_newtonal_min_step_size", "Minimum arc length increment step size", "SNESNewtonAL", data->min_step_size, &data->min_step_size, NULL));
  data->psisq = 1.0;
  PetscCall(PetscOptionsReal("-snes_newtonal_psisq", "Regularization parameter for arc length continuation, 0 for cylindrical", "SNESNewtonAL", data->psisq, &data->psisq, NULL));
  data->correction_type = SNES_NEWTONAL_CORRECTION_EXACT;
  PetscCall(PetscOptionsEnum("-snes_newtonal_correction_type", "Type of correction to use in the arc-length continuation method", "SNESNewtonALCorrectionType", SNESNewtonALCorrectionTypes, (PetscEnum)data->correction_type, (PetscEnum *)&data->correction_type, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESReset_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *al = (SNES_NEWTONAL *)snes->data;

  PetscFunctionBegin;
  al->lambda_update = 0.0;
  al->lambda        = 0.0;
  PetscCall(VecDestroy(&al->vec_rhs_orig));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESDestroy_NEWTONAL - Destroys the private SNES_NEWTONAL context that was created
   with SNESCreate_NEWTONAL().

   Input Parameter:
.  snes - the SNES context

   Application Interface Routine: SNESDestroy()
 */
static PetscErrorCode SNESDestroy_NEWTONAL(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   SNESView_NEWTONAL - Prints info from the SNESNEWTONAL data structure.

   Input Parameters:
.  SNES - the SNES context
.  viewer - visualization context

   Application Interface Routine: SNESView()
*/
static PetscErrorCode SNESView_NEWTONAL(SNES snes, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) { }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   SNESNewtonAL - Newton based nonlinear solver that uses a arc-length continuation method to solve the nonlinear system.

   Options Database Keys:
+   -snes_newtonal_step_size <1.0> - Initial arc length increment step size
.   -snes_newtonal_min_step_size <1.0e-6> - Minimum arc length increment step size
.   -snes_newtonal_max_steps <100> - Maximum number of increment steps
.   -snes_newtonal_psisq <1.0> - Regularization parameter for arc length continuation, 0 for cylindrical. Larger values generally lead to more steps.
.   -snes_linesearch_type <basic> - basic.  Select line search type
-   -snes_linesearch_damping - damping factor used for basic line search

   Level: intermediate

.seealso: [](ch_snes), `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESNEWTONAL`, `SNESSetNewtonAL()`, `SNESGetNewtonAL()`, `SNESNewtonALGetLoadParameter()`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONAL(SNES snes)
{
  SNES_NEWTONAL *arclengthParameters;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONAL;
  snes->ops->solve          = SNESSolve_NEWTONAL;
  snes->ops->destroy        = SNESDestroy_NEWTONAL;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONAL;
  snes->ops->view           = SNESView_NEWTONAL;
  snes->ops->reset          = SNESReset_NEWTONAL;

  snes->usesksp = PETSC_TRUE;
  snes->usesnpc = PETSC_FALSE;

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC));

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNew(&arclengthParameters));
  snes->data = (void *)arclengthParameters;
  PetscFunctionReturn(PETSC_SUCCESS);
}
