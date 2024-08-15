#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h>            /*I "petscdm.h" I*/
#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/

/*@
  TaoPSSetLipschitz - Sets Lipschitz constant of smooth term
  for `TAOFB` or `TAOCV`.

  Logically Collective

  Input Parameters:
+ tao - the `Tao` context
- lip - the Lipschitz constant

  Level: intermediate

  Note:
  This method is intended for a usecase where smooth term
  is set via `TaoSetObjective` like routines.

.seealso: [](ch_tao), `Tao`, `TAOFB`, `TAOCV`
@*/
PetscErrorCode TaoPSSetLipschitz(Tao tao, PetscReal lip)
{
  TaoType   type;
  PetscBool isfb, iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, lip, 2);
  /* Note: zero lip means to approximate it in the beginning */
  PetscCheck(lip >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Lipschitz value has to be nonnegative.");
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));

  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->lip     = lip;
    fb->lip_set = PETSC_TRUE;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->lip     = lip;
    cv->lip_set = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetNonSmoothTerm - Set non-smooth objective term for
  proximal splitting problem - g(x).

  Input Parameters:
+ tao   - the `Tao` context for the `TAOFB` and `TAOCV` solver
. dm    - the `DM` context containing non-smooth objective
- scale - scale of the nonsmooth term

  Level: advanced

.seealso: `TAOFB`, `TAOCV`, `Tao`, `TaoPSSetNonSmoothTermWithLinearMap()`
@*/
PetscErrorCode TaoPSSetNonSmoothTerm(Tao tao, DM dm, PetscReal scale)
{
  TaoType   type;
  PetscBool isfb, iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  PetscCheck(scale >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Scale cannot be negative");

  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->proxterm   = dm;
    fb->prox_scale = scale;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->g_prox  = dm;
    cv->g_scale = scale;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetNonSmoothTermWithLinearMap - Set non-smooth objective term
  with linear mapping for proximal splitting problem - h(Ax).

  Input Parameters:
+ tao   - the `Tao` context for the `TAOCV` solver
. dm    - the `DM` context containing non-smooth objective
. mat   - the linear mapping matrix
. norm  - norm of the linear mapping matrix, if avaliable. Set as zero if unknown
- scale - scale of the function

  Level: advanced

.seealso: `TAOCV`, `Tao`, `TaoPSSetNonSmoothTerm()`
@*/
PetscErrorCode TaoPSSetNonSmoothTermWithLinearMap(Tao tao, DM dm, Mat mat, PetscReal norm, PetscReal scale)
{
  TaoType   type;
  PetscBool iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  PetscCheck(scale >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Scale cannot be negative");
  /* Currently this function is only used by TAOCV, but
   * TODO TAODY would use this too */
  if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    if (mat) {
      PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
      PetscCall(PetscObjectReference((PetscObject)mat));
    }
    PetscValidLogicalCollectiveReal(tao, norm, 4);
    cv->h_prox  = dm;
    cv->h_lmap  = mat;
    cv->h_scale = scale;
    if (norm) {
      cv->h_lmap_norm   = norm;
      cv->lmap_norm_set = PETSC_TRUE;
    } else {
      cv->h_lmap_norm   = 0.;
      cv->lmap_norm_set = PETSC_FALSE;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetSmoothTerm - Set smooth objective term for proximal splitting problem.
  This smooth objective - f(x) -  must have gradient term available.

  Input Parameters:
+ tao   - the `Tao` context for the `TAOCV`, `TAOFB` solver
. dm    - the `DM` context containing smooth objective
- scale - the scale of smooth term

  Level: advanced

.seealso: `TAOFB`, `TAOCV`, `Tao`, `TaoPSSetNonSmoothTerm()`
@*/
PetscErrorCode TaoPSSetSmoothTerm(Tao tao, DM dm, PetscReal scale)
{
  TaoType   type;
  PetscBool isfb, iscv;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCheck(tdm->ops->computeobjectiveandgradient || tdm->ops->computegradient, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "DMTaoSetGradient() has not been called");
  PetscCheck(scale >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Scale cannot be negative");
  PetscCall(PetscObjectReference((PetscObject)dm));

  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->smoothterm = dm;
    fb->f_scale    = scale;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->smoothterm = dm;
    cv->f_scale    = scale;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSUseAcceleration - Determine whether to Nesterov-type
  acceleration strategy for `TAOCV, and `TAOFB`.

  Input Parameters:
+ tao  - the `Tao` context for the `TAOCV` or `TAOFB` solver
- flag - Bool to denote whether to acceleration

  Level: advanced

.seealso: `Tao`, `TAOFB`, `TAOCV`
@*/
PetscErrorCode TaoPSUseAcceleration(Tao tao, PetscBool flag)
{
  TaoType   type;
  PetscBool isfb, iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->use_accel = PETSC_TRUE;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->use_accel = PETSC_TRUE;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSUseAdaptiveStep - Determine whether to use adaptive stepsize
  for `TAOCV, and `TAOFB`.

  Input Parameters:
+ tao  - the `Tao` context for the `TAOCV` or `TAOFB` solver
- flag - Bool to denote whether to use adaptive stepsize

  Level: advanced

.seealso: `Tao`, `TAOFB`, `TAOCV`
@*/
PetscErrorCode TaoPSUseAdaptiveStep(Tao tao, PetscBool flag)
{
  TaoType   type;
  PetscBool isfb, iscv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOCV, &iscv));
  if (isfb) {
    TAO_FB *fb = (TAO_FB *)tao->data;

    fb->use_adapt = PETSC_TRUE;
  } else if (iscv) {
    TAO_CV *cv = (TAO_CV *)tao->data;

    cv->use_adapt = PETSC_TRUE;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid Tao type.");
  PetscFunctionReturn(PETSC_SUCCESS);
}
