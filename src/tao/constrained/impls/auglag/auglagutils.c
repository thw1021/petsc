#include <../src/tao/constrained/impls/auglag/auglag.h> /*I "petsctao.h" I*/ /*I "petscvec.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>

/* -------------------------------------------------------- */

/*@
   TaoAugLagGetType - Retreive the augmented Lagrangian formulation type for the subproblem.

   Input Parameters:
.  tao - the Tao context for the TAOAUGLAG solver

   Output Parameters:
.  type - augmented Lagragrangian type

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagSetType(), TaoAugLagType
@*/
PetscErrorCode TaoAugLagGetType(Tao tao, TaoAugLagType *type)
{
  TAO_AugLag      *auglag = (TAO_AugLag*)tao->data;
  PetscBool       same = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOAUGLAG, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "TAO solver must be TAOAUGLAG to retrieve the subsolver");
  *type = auglag->type;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagSetType - Determine the augmented Lagrangian formulation type for the subproblem.

   Input Parameters:
+  tao - the Tao context for the TAOAUGLAG solver
-  type - augmented Lagragrangian type

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagGetType(), TaoAugLagType
@*/
PetscErrorCode TaoAugLagSetType(Tao tao, TaoAugLagType type)
{
  TAO_AugLag      *auglag = (TAO_AugLag*)tao->data;
  PetscBool       same = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOAUGLAG, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "TAO solver must be TAOAUGLAG to set the subsolver");
  if (tao->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoAugLagSetType() must be called before TaoSetUp()");
  auglag->type = type;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagGetSubsolver - Retrieve a pointer to the TAOAUGLAG.

   Input Parameters:
.  tao - the Tao context for the TAOAUGLAG solver

   Output Parameter:
.  subsolver - the Tao context for the subsolver

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagSetSubsolver()
@*/
PetscErrorCode TaoAugLagGetSubsolver(Tao tao, Tao *subsolver)
{
  TAO_AugLag      *auglag = (TAO_AugLag*)tao->data;
  PetscBool       same = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOAUGLAG, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "TAO solver must be TAOAUGLAG to retrieve the subsolver");
  *subsolver = auglag->subsolver;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagSetSubsolver - Changes the subsolver inside TAOAUGLAG with the user provided one.

   Input Parameters:
+  tao - the Tao context for the TAOAUGLAG solver
-  subsolver - the Tao context for the subsolver

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagGetSubsolver()
@*/
PetscErrorCode TaoAugLagSetSubsolver(Tao tao, Tao subsolver)
{
  TAO_AugLag      *auglag = (TAO_AugLag*)tao->data;
  PetscBool       same = PETSC_FALSE, compatible = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(subsolver, TAO_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOAUGLAG, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Parent solver must be TAOAUGLAG");
  if (tao->bounded) {
    ierr = PetscObjectTypeCompareAny((PetscObject)subsolver, &compatible, TAOSHELL, TAOBNCG, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, "");CHKERRQ(ierr);
    if (!compatible) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Subsolver must be a bound-constrained first-order method");
  } else {
    ierr = PetscObjectTypeCompareAny((PetscObject)subsolver, &compatible, TAOSHELL, TAOCG, TAOLMVM, TAOBNCG, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, "");CHKERRQ(ierr);
    if (!compatible) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Subsolver must be a first-order method");
  }
  if (!compatible) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Subsolver must be a first-order method");
  ierr = TaoDestroy(&auglag->subsolver);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)subsolver);
  auglag->subsolver = subsolver;
  if (tao->setupcalled) {
    ierr = TaoSetInitialVector(auglag->subsolver, auglag->P);CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine(auglag->subsolver, TaoAugLagSubsolverObjectiveAndGradient_Private, (void*)auglag);CHKERRQ(ierr);
    ierr = TaoSetVariableBounds(auglag->subsolver, auglag->PL, auglag->PU);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagGetMultipliers - Retreive a pointer to the Lagrange multipliers.

                             For problems with both equality and inequality constraints,
                             the multipliers are combined together as Y = (Ye, Yi). Users
                             can recover copies of the subcomponents using TaoAugLagSplitDual().
                             Alternatively, users can also recover corresponding index sets with
                             TaoAugLagGetDualIS() and use VecGetSubVector() to access the
                             desired component in-place without making copies or triggering
                             parallel communication.

   Input Parameters:
.  tao - the Tao context for the TAOAUGLAG solver

   Output Parameters:
.  Y - vector of Lagrange multipliers

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagSetMultipliers(), TaoAugLagSplitDual(), TaoAugLagGetDualIS()
@*/
PetscErrorCode TaoAugLagGetMultipliers(Tao tao, Vec *Y)
{
  TAO_AugLag      *auglag = (TAO_AugLag*)tao->data;
  PetscBool       same = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOAUGLAG, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Tao solver must be TAOAUGLAG");
  if (!tao->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called first for scatters to be constructed");
  *Y = auglag->Y;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagSetMultipliers - Set user-defined Lagrange multipliers. The vector type and
                             parallel structure of the given vectormust match equality and
                             inequality constraints. The vector must have a local size equal
                             to the sum of the local sizes for the constraint vectors, and a
                             global size equal to the sum of the global sizes of the constraint
                             vectors.

                             This routine is only useful if the user wants to change the
                             parallel distribution of the combined dual vector in problems that
                             feature both equality and inequality constraints. For other tasks,
                             it is strongly recommended that the user retreive the dual vector
                             created by the solver using TaoAugLagGetMultipliers().

   Input Parameters:
+  tao - the Tao context for the TAOAUGLAG solver
-  Y - vector of Lagrange multipliers

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagGetMultipliers()
@*/
PetscErrorCode TaoAugLagSetMultipliers(Tao tao, Vec Y)
{
  TAO_AugLag     *auglag = (TAO_AugLag*)tao->data;
  VecType        Ytype;
  PetscInt       Nuser, Neq, Nineq, N;
  PetscBool      same = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOAUGLAG, &same);CHKERRQ(ierr);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Tao solver must be TAOAUGLAG");
  /* no-op if user provides vector from TaoAugLagGetMultipliers() */
  if (Y == auglag->Y) PetscFunctionReturn(0);
  /* make sure vector type is same as equality and inequality constraints */
  if (tao->eq_constrained) {
    ierr = VecGetType(tao->constraints_equality, &Ytype);
  } else {
    ierr = VecGetType(tao->constraints_inequality, &Ytype);
  }
  ierr = PetscObjectTypeCompare((PetscObject)Y, Ytype, &same);
  if (!same) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector for multipliers is not the same type as constraint vectors");
  /* make sure global size matches sum of equality and inequality */
  if (tao->eq_constrained) {
    ierr = VecGetSize(tao->constraints_equality, &Neq);
  } else {
    Neq = 0;
  }
  if (tao->ineq_constrained) {
    ierr = VecGetSize(tao->constraints_inequality, &Nineq);
  } else {
    Nineq = 0;
  }
  N = Neq + Nineq;
  ierr = VecGetSize(Y, &Nuser);CHKERRQ(ierr);
  if (Nuser != N) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector has wrong global size");
  /* if there is only one type of constraint, then we need the local size to match too */
  if (Neq == 0) {
    ierr = VecGetLocalSize(tao->constraints_inequality, &Nineq);CHKERRQ(ierr);
    ierr = VecGetLocalSize(Y, &Nuser);CHKERRQ(ierr);
    if (Nuser != Nineq) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector has wrong local size");
  }
  if (Nineq == 0) {
    ierr = VecGetLocalSize(tao->constraints_equality, &Neq);CHKERRQ(ierr);
    ierr = VecGetLocalSize(Y, &Nuser);CHKERRQ(ierr);
    if (Nuser != Neq) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given vector has wrong local size");
  }
  /* if we got here, the given vector is compatible so we can replace the current one */
  ierr = PetscObjectReference((PetscObject)Y);CHKERRQ(ierr);
  ierr = VecDestroy(&auglag->Y);CHKERRQ(ierr);
  auglag->Y = Y;
  /* if there are both types of constraints and the solver has already been set up,
     then we need to regenerate VecScatter objects for the new combined dual vector */
  if (tao->setupcalled && tao->eq_constrained && tao->ineq_constrained) {
    ierr = VecDestroy(&auglag->C);CHKERRQ(ierr);
    ierr = VecDuplicate(auglag->Y, &auglag->C);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&auglag->Yscatter[0]);CHKERRQ(ierr);
    ierr = VecScatterCreate(auglag->Y, auglag->Yis[0], auglag->Ye, NULL, &auglag->Yscatter[0]);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&auglag->Yscatter[1]);CHKERRQ(ierr);
    ierr = VecScatterCreate(auglag->Y, auglag->Yis[1], auglag->Yi, NULL, &auglag->Yscatter[1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagGetPrimalIS - Retreive a pointer to the index set that identifies optimization
                          and slack variable components of the primal vector returned by
                          TaoAugLagGetPrimalVector().

                          Not valid for problems with only equality constraints.

   Input Parameters:
.  tao - the Tao context for the TAOAUGLAG solver

   Output Parameters:
+  opt_is - index set associated with the optimization variables (NULL if not needed)
-  slack_is - index set associated with the slack variables (NULL if not needed)

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagGetPrimalVector()
@*/
PetscErrorCode TaoAugLagGetPrimalIS(Tao tao, IS *opt_is, IS *slack_is)
{
  TAO_AugLag *auglag = (TAO_AugLag*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (!tao->ineq_constrained) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Primal space has index sets only for inequality constrained problems");
  if (!tao->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called first for index sets to be constructed");
  if (!opt_is && !slack_is) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_NULL, "Both index set pointers cannot be NULL");
  if (opt_is) {
     *opt_is = auglag->Pis[0];
  }
  if (slack_is) {
     *slack_is = auglag->Pis[1];
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*@
   TaoAugLagGetDualIS - Retreive a pointer to the index set that identifies equality
                        and inequality constraint components of the dual vector returned
                        by TaoAugLagGetMultipliers().

                        Not valid for problems with only one type of constraint.

   Input Parameters:
.  tao - the Tao context for the TAOAUGLAG solver

   Output Parameters:
+  eq_is - index set associated with the equality constraints (NULL if not needed)
-  ineq_is - index set associated with the inequality constraints (NULL if not needed)

   Level: advanced

.seealso: TAOAUGLAG, TaoAugLagGetMultipliers()
@*/

PetscErrorCode TaoAugLagGetDualIS(Tao tao, IS *eq_is, IS *ineq_is)
{
  TAO_AugLag *auglag = (TAO_AugLag*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (!tao->ineq_constrained || !tao->ineq_constrained) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Dual space has index sets only when problem has both equality and inequality constraints");
  if (!tao->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoSetUp() must be called first for index sets to be constructed");
  if (!eq_is && !ineq_is) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_NULL, "Both index set pointers cannot be NULL");
  if (eq_is) {
     *eq_is = auglag->Yis[0];
  }
  if (ineq_is) {
     *ineq_is = auglag->Yis[1];
  }
  PetscFunctionReturn(0);
}