#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/psarmijo/psarmijo.h>
#include <../src/tao/proximal/impls/cv/cv.h>

static PetscErrorCode TaoLineSearchDestroy_PSArmijo(TaoLineSearch ls)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(armP->memory));
  if (armP->x) PetscCall(PetscObjectDereference((PetscObject)armP->x));
  PetscCall(VecDestroy(&armP->work));
  PetscCall(VecDestroy(&armP->work2));
  PetscCall(PetscFree(ls->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetFromOptions_PSArmijo(TaoLineSearch ls, PetscOptionItems *PetscOptionsObject)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PSArmijo linesearch options");
  PetscCall(PetscOptionsReal("-tao_ls_PSArmijo_eta", "decrease constant", "", armP->eta, &armP->eta, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_PSArmijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchView_PSArmijo(TaoLineSearch ls, PetscViewer pv)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;
  PetscBool               isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  PSArmijo linesearch"));
    PetscCall(PetscViewerASCIIPrintf(pv, "eta=%g ", (double)armP->eta));
    PetscCall(PetscViewerASCIIPrintf(pv, "memsize=%" PetscInt_FMT "\n", armP->memorySize));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @ TaoApply_PSArmijo - This routine performs a linesearch. It
   backtracks until the (nonmonotone) PSArmijo conditions are satisfied.

   Input Parameters:
+  ls   - TaoLineSearch context
.  xold - z_k
.  f    - f(z_k)
.  g    - grad_f(z_k). same as tao->gradient
.  xnew - x_k
-  step - initial estimate of step length (Set via TaoLSSetInitialStep)

   Output parameters:
+  f    - f(x_{k+1})
.  xnew - x_{k+1} that satisfies the condition
-  step - final step length
@ */
static PetscErrorCode TaoLineSearchApply_PSArmijo(TaoLineSearch ls, Vec xold, PetscReal *f, Vec g, Vec xnew)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;
  PetscInt                i, its = 0;
  PetscReal               ref, cert; /* cert = R + <gradf(x), xnew - x> + 1/2step * |xnew - x|_2^2 */
  PetscReal               diffnorm, inprod;
  MPI_Comm                comm;
  TaoType                 type;
  PetscBool               isfb, iscv;
  PetscReal               xi, grad_x_dot, xdiffnorm, graddiffnorm, L, C, D, min1, min2, min3;
  PetscReal               temp, temp2, temp3, step_new, rho, norm1, norm2;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ls, &comm));
  ls->nfeval = 0;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    armP->x = xold;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  } else if (xold != armP->x) {
    PetscCall(VecDestroy(&armP->work));
    PetscCall(VecDestroy(&armP->work2));
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    PetscCall(PetscObjectDereference((PetscObject)armP->x));
    armP->x = xold;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  }

  PetscCall(TaoLineSearchMonitor(ls, 0, *f, 0.0));

  /* Check linesearch parameters */
  if (armP->eta > 1) {
    PetscCall(PetscInfo(ls, "PSArmijo line search error: eta (%g) > 1\n", (double)armP->eta));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (armP->memorySize < 1) {
    PetscCall(PetscInfo(ls, "PSArmijo line search error: memory_size (%" PetscInt_FMT ") < 1\n", armP->memorySize));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "PSArmijo line search error: initial function inf or nan\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }

  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(TaoGetType(ls->tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)ls->tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)ls->tao, TAOCV, &iscv));

  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function values. */
  if (armP->memorySize > 1) {
    if (!armP->memory) PetscCall(PetscMalloc1(armP->memorySize, &armP->memory));

    if (!armP->memorySetup) {
      for (i = 0; i < armP->memorySize; i++) armP->memory[i] = 0.;
      armP->current               = 0;
      armP->memorySetup           = PETSC_TRUE;
      armP->memory[armP->current] = *f;
    }

    /* Calculate reference value (MAX) */
    ref = armP->memory[0];
    for (i = 1; i < armP->memorySize; i++) {
      if (armP->memory[i] > ref) { ref = armP->memory[i]; }
    }
  } else ref = *f;

  ls->step = ls->initstep;

  if (isfb) {
    // Input is prox_g(x- step * gradf(x))
    /* Calculate function at new iterate */
    PetscCall(TaoLineSearchComputeObjective(ls, xnew, &temp));
    /* Check criteria */
    PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
    PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
    PetscCall(VecTDot(armP->work2, g, &inprod));
    cert = temp - (inprod + (1 / (2 * ls->step)) * diffnorm + ref);

    /* accept xnew */
    if (cert < ls->rtol) {
      PetscCall(TaoLineSearchComputeObjective(ls, xnew, f));
      ls->reason = TAOLINESEARCH_SUCCESS;
    }
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)ls->tao->data;
    xi         = cv->pd_ratio * ls->tao->step * cv->eta * (1 + ls->tao->gatol);
    xi *= xi;

    PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, ls->tao->gradient));
    PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, ls->tao->solution));
    PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
    PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
    PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

    L = (xdiffnorm == 0) ? 0 : grad_x_dot / (xdiffnorm * xdiffnorm);
    C = (grad_x_dot == 0) ? 0 : (graddiffnorm * graddiffnorm) / grad_x_dot;
    D = ls->tao->step * L * (ls->tao->step * C - 1);

    cv->eta *= cv->R;
    cert = PETSC_INFINITY; // For TAOCV, linesearch needs to go at least once
  } else SETERRQ(PetscObjectComm((PetscObject)ls->tao), PETSC_ERR_USER, "Invalid Tao type.");

  while (cert >= ls->ftol && ls->nproxeval < ls->max_funcs) {
    /* Calculate iterate */
    ++its;

    if (isfb) {
      ls->step = ls->step * armP->eta;

      /* FB: input to prox: x_k - step*gradf(x_k)
       * Adaptive DY: input to prox: z - step*u - step*gradf(z) -> input g for DY needs to be u-gradf(z), not just gradf(z) TODO*/
      PetscCall(VecWAXPY(armP->work, -ls->step, g, xold));
      PetscCall(DMTaoApplyProximalMap(ls->prox, ls->prox_reg, ls->step * ls->prox_scale, armP->work, xnew, PETSC_FALSE));
      ls->nproxeval++;
      /* Calculate function at new iterate */
      PetscCall(TaoLineSearchComputeObjective(ls, xnew, f));
      PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
      PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
      PetscCall(VecTDot(armP->work2, g, &inprod));

      cert = *f - (inprod + (1 / (2 * ls->step)) * diffnorm + ref);

      PetscCall(TaoLineSearchMonitor(ls, its, *f, ls->step));
    } else if (iscv) {
      /* Note: LineSearch for CV does not follow regular Armijo condition.
       * Rather, this looks for
       *
       * eta \geq |A^T (y_new - y) | / |y_new - y|,
       * where y_new = prox_{sigma, h^*} (y + sigma((1+stepnew/stepold) Ax - stepnew/step Ax_old)) */
      TAO_CV *cv = (TAO_CV *)ls->tao->data;

      min1      = ls->tao->step * PetscSqrtReal(1 + ls->tao->step / cv->step_old);
      min2      = 1 / (2 * cv->nu * cv->pd_ratio * cv->eta);
      temp      = 1 - 4 * xi;
      temp2     = cv->pd_ratio * cv->eta * ls->tao->step; //Unlike no linesearch, this "xi" uses updated norm estimate
      temp3     = PetscSqrtReal(D * D + temp * temp2 * temp2);
      min3      = ls->tao->step * PetscSqrtReal(temp / (2 * (1 + ls->tao->gatol) * (temp3 + D)));
      step_new  = PetscMin(min1, PetscMin(min2, min3));
      rho       = step_new / ls->tao->step;
      cv->sigma = cv->pd_ratio * cv->pd_ratio * step_new;

      /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
      PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, ls->tao->dualvec));
      PetscCall(VecAXPY(cv->dualvec_work, cv->sigma * (1 + rho), cv->Ax));
      /* dualvec: y = prox_h*(w, sigma) */
      PetscCall(DMTaoApplyProximalMap(cv->h_prox, cv->reg, cv->sigma * cv->h_scale, cv->dualvec_work, cv->dualvec_test, PETSC_TRUE));
      /* workvec : A^T * y_test */
      PetscCall(MatMultTranspose(cv->h_lmap, cv->dualvec_test, cv->workvec));
      /* norm1 = norm(ATy_test - ATy) */
      PetscCall(VecWAXPY(cv->workvec2, -1., cv->ATy, cv->workvec));
      PetscCall(VecNorm(cv->workvec2, NORM_2, &norm1));
      /* norm2 = norm(y_test - y) */
      PetscCall(VecWAXPY(cv->dualvec_work2, -1., cv->dualvec_test, ls->tao->dualvec));
      PetscCall(VecNorm(cv->dualvec_work2, NORM_2, &norm2));
      cert = -cv->eta + norm1 / norm2;
      if (cert <= ls->ftol) {
        cv->step_old  = ls->tao->step;
        ls->tao->step = step_new;
        PetscCall(VecCopy(cv->dualvec_test, ls->tao->dualvec));
        PetscCall(VecCopy(cv->workvec, cv->ATy));
        break;
      }
      PetscCall(TaoLineSearchMonitor(ls, its, cert, ls->step)); //Technically not logging function value...
      cv->h_lmap_norm *= cv->r;
    } else SETERRQ(PetscObjectComm((PetscObject)ls->tao), PETSC_ERR_USER, "Invalid Tao type.");
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (ls->nproxeval >= ls->max_funcs) {
    PetscCall(PetscInfo(ls, "Number of line search prox evals (%" PetscInt_FMT ") > maximum allowed (%" PetscInt_FMT ")\n", ls->nproxeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  } else if (ls->step < ls->stepmin) {
    PetscCall(PetscInfo(ls, "Step length is below tolernace.\n"));
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  }

  if (isfb) {
    if (armP->memorySize > 1) {
      armP->current++;
      if (armP->current >= armP->memorySize) armP->current = 0;
      armP->memory[armP->current] = *f;
    }
  }
  if (ls->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Successful termination, update memory. Only FIFO for PSARMIJO */
  ls->reason = TAOLINESEARCH_SUCCESS;
  PetscCall(PetscInfo(ls, "%" PetscInt_FMT " prox evals in line search, step = %10.4f\n", ls->nproxeval, (double)ls->step));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   TAOLINESEARCHPSARMIJO - Special line-search type for proximal splittign algorithms.
   Should not be used with any other algorithm.

   Level: developer

seealso: `TaoLineSearch`, `TAOFB`, `Tao`
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_PSArmijo(TaoLineSearch ls)
{
  TaoLineSearch_PSARMIJO *armP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscCall(PetscNew(&armP));

  armP->memory            = NULL;
  armP->eta               = 0.5;
  armP->memorySize        = 1;
  ls->data                = (void *)armP;
  ls->initstep            = 0;
  ls->ops->monitor        = NULL;
  ls->ops->setup          = NULL;
  ls->ops->reset          = NULL;
  ls->ops->apply          = TaoLineSearchApply_PSArmijo;
  ls->ops->view           = TaoLineSearchView_PSArmijo;
  ls->ops->destroy        = TaoLineSearchDestroy_PSArmijo;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_PSArmijo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
