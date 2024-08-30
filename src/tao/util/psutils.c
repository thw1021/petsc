#include <petsc/private/taoimpl.h>
#include <petsc/private/dmimpl.h>            /*I "petscdm.h" I*/
#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/

/*@
  TaoPSSetNonSmoothTerm - Set non-smooth objective term for
  proximal splitting problem - g(x).

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` and `TAOCV` solver
- idx - the index of `DM` context containing non-smooth objective

  Level: advanced

.seealso: `TAOFB`, `TAOCV`, `Tao`, `TaoPSSetNonSmoothTermWithLinearMap()`
@*/
PetscErrorCode TaoPSSetNonSmoothTerm(Tao tao, PetscInt idx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao, "TaoPSSetNonSmoothTerm_C", (Tao, PetscInt), (tao, idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetNonSmoothTermWithLinearMap - Set non-smooth objective term
  with linear mapping for proximal splitting problem - h(Ax).

  Input Parameters:
+ tao  - the `Tao` context for the `TAOCV` solver
. idx  - the index of `DM` context containing non-smooth objective
. mat  - the linear mapping matrix
- norm - norm of the linear mapping matrix, if avaliable. Set as zero if unknown

  Level: advanced

.seealso: `TAOCV`, `Tao`, `TaoPSSetNonSmoothTerm()`
@*/
PetscErrorCode TaoPSSetNonSmoothTermWithLinearMap(Tao tao, PetscInt idx, Mat mat, PetscReal norm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCheck(norm >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Matrix norm cannot be negative");
  PetscTryMethod(tao, "TaoPSSetNonSmoothTermWithLinearMap_C", (Tao, PetscInt, Mat, PetscReal), (tao, idx, mat, norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSSetSmoothTerm - Set smooth objective term for proximal splitting problem.
  This smooth objective - f(x) -  must have gradient term available.

  Input Parameters:
+ tao - the `Tao` context for the `TAOCV`, `TAOFB` solver
- idx - the index of `DM` context containing smooth objective

  Level: advanced

.seealso: `TAOFB`, `TAOCV`, `Tao`, `TaoPSSetNonSmoothTerm()`
@*/
PetscErrorCode TaoPSSetSmoothTerm(Tao tao, PetscInt idx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao, "TaoPSSetSmoothTerm_C", (Tao, PetscInt), (tao, idx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSUseAcceleration - Determine whether to Nesterov-type
  acceleration strategy for `TAOFB`.

  Input Parameters:
+ tao  - the `Tao` context for the `TAOFB` solver
- flag - Bool to denote whether to acceleration

  Level: advanced

.seealso: `Tao`, `TAOFB`
@*/
PetscErrorCode TaoPSUseAcceleration(Tao tao, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  /* Note: Acceleration flag for TAOCV is planned for the future */
  PetscTryMethod(tao, "TaoPSUseAcceleration_C", (Tao, PetscBool), (tao, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPSUseAdaptiveStep - Determine whether to use adaptive stepsize
  for `TAOFB`.

  Input Parameters:
+ tao  - the `Tao` context for the `TAOFB` solver
- flag - Bool to denote whether to use adaptive stepsize

  Level: advanced

.seealso: `Tao`, `TAOFB`
@*/
PetscErrorCode TaoPSUseAdaptiveStep(Tao tao, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  /* Note: Adaptive flag for TAOCV is planned for the future */
  PetscTryMethod(tao, "TaoPSUseAdaptiveStep_C", (Tao, PetscBool), (tao, flag));
  PetscFunctionReturn(PETSC_SUCCESS);
}
