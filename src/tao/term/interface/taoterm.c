#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h> /*I "petsctao.h" I*/

PetscClassId TAOTERM_CLASSID;

PetscLogEvent TAOTERM_ObjectiveEval;
PetscLogEvent TAOTERM_GradientEval;
PetscLogEvent TAOTERM_ObjGradEval;
PetscLogEvent TAOTERM_HessianEval;

/*@
  TaoTermDestroy - Destroy a `TaoTerm`.

  Collective

  Input Parameters:
. term - a `TaoTerm`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreate()`, `TaoTermSetUp()`, `TaoTermSetFromOptions()`, `TaoTermView()`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermDestroy(TaoTerm *term)
{
  PetscFunctionBegin;
  if (!*term) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*term, TAOTERM_CLASSID, 1);
  if (--((PetscObject)*term)->refct > 0) {
    *term = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod(*term, destroy);
  PetscCall(MatDestroy(&(*term)->solution_factory));
  PetscCall(MatDestroy(&(*term)->parameters_factory));
  PetscCall(PetscHeaderDestroy(term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermView - View a description of a `TaoTerm`.

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
- viewer - a `PetscViewer`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreate()`, `TaoTermDestroy()`, `TaoTermSetUp()`, `TaoTermSetFromOptions()`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermView(TaoTerm term, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)term), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(term, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)term, viewer));
    if (term->ops->view) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscUseTypeMethod(term, view, viewer);
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetUp - Set up a `TaoTerm`.

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreate()`, `TaoTermDestroy()`, `TaoTermView()`, `TaoTermSetFromOptions()`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermSetUp(TaoTerm term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (term->setup_called) PetscFunctionReturn(PETSC_SUCCESS);
  term->setup_called = PETSC_TRUE;
  PetscTryTypeMethod(term, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetFromOptions - Configure a `TaoTerm` from options

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Options Database Keys:
. -taoterm_type <type> - tao, shell, dm, separable, l1, linf, l2squared, quadratic, kl, `TaoTermType` for complete list

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreate()`, `TaoTermDestroy()`, `TaoTermView()`, `TaoTermSetUp()`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermSetFromOptions(TaoTerm term)
{
  const char *deft = TAOTERMSHELL;
  char        type[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (((PetscObject)term)->type_name) deft = ((PetscObject)term)->type_name;
  PetscObjectOptionsBegin((PetscObject)term);
  PetscCall(PetscOptionsFList("-taoterm_type", "TaoTerm type", "TaoTermType", TaoTermList, deft, type, 256, &flg));
  if (flg) {
    PetscCall(TaoTermSetType(term, type));
  } else {
    PetscCall(TaoTermSetType(term, deft));
  }
  PetscTryTypeMethod(term, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetType - Set the type of a `TaoTerm` from options

  Collective

  Input Parameter:
+ term - a `TaoTerm`
- type - a `TaoTermType`

  Options Database Key:
. -taoterm_type <type> - Sets the method; use -help for a list
   of available methods (for instance, newtonls or newtontr)

  Level: beginner

  Note:
  New types of `TaoTerm` can be created with `TaoTermRegister()`

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermType`, `TaoTermCreate()`, `TaoTermDestroy()`, `TaoTermView()`, `TaoTermSetUp()`, `TaoTermGetType()`
@*/
PetscErrorCode TaoTermSetType(TaoTerm term, TaoTermType type)
{
  PetscErrorCode (*create)(TaoTerm);
  PetscBool issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);

  PetscCall(PetscObjectTypeCompare((PetscObject)term, type, &issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoTermList, type, (void (**)(void)) & create));
  PetscCheck(create, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested TaoTerm type %s", type);

  /* Destroy the existing term information */
  PetscTryTypeMethod(term, destroy);
  term->setup_called = PETSC_FALSE;

  PetscCall((*create)(term));
  PetscCall(PetscObjectChangeTypeName((PetscObject)term, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreate - Create a TaoTerm to use in `Tao` objective functions

  Collective

  Input Parameter:
. comm - communicator for MPI processes that compute the term

  Output Parameter:
. term - a new TaoTerm

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermDestroy()`, `TaoTermSetUp()`, `TaoTermSetFromOptions()`, `TaoTermView()`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermCreate(MPI_Comm comm, TaoTerm *term)
{
  TaoTerm _term;

  PetscFunctionBegin;
  PetscAssertPointer(term, 2);
  PetscCall(TaoInitializePackage());
  PetscCall(PetscHeaderCreate(_term, TAOTERM_CLASSID, "TaoTerm", "Objective function term", "Tao", comm, TaoTermDestroy, TaoTermView));
  *term = _term;
  PetscCall(MatCreate(comm, &(*term)->solution_factory));
  PetscCall(MatSetType((*term)->solution_factory, MATDUMMY));
  PetscCall(MatCreate(comm, &(*term)->parameters_factory));
  PetscCall(MatSetType((*term)->parameters_factory, MATDUMMY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermObjective - Evaluate a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameter:
. value - the value of $f(x; p)$

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`,
@*/
PetscErrorCode TaoTermObjective(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscAssertPointer(value, 4);
  if (term->ops->objective) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
  } else if (term->ops->objectiveandgradient) {
    Vec temp;

    PetscCall(PetscInfo(term, "Duplicating solution vector in order to call objective/gradient routine\n"));
    PetscCall(VecDuplicate(x, &temp));
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, temp);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscCall(VecDestroy(&temp));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have an objective function.  You should have called TaoSetObjective() or TaoTermShellSetObjective()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "TaoTerm value: %20.19e\n", (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGradient - Evaluate the gradient of a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameter:
. g - the value of $\nabla_x f(x; p)$

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`
@*/
PetscErrorCode TaoTermGradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscCheckSameComm(term, 1, g, 4);
  VecCheckSameSize(x, 2, g, 4);
  if (term->ops->gradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
  } else if (term->ops->objectiveandgradient) {
    PetscReal value;

    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, &value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have a gradient function.  You should have called TaoSetGradient() or TaoTermShellSetGradient()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermObjectiveAndGradient - Evaluate both the value and gradient of a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameters:
+ value - the value of $f(x; p)$
- g     - the value of $\nabla_x f(x; p)$

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermHessian()`
@*/
PetscErrorCode TaoTermObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscAssertPointer(value, 4);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 5);
  PetscCheckSameComm(term, 1, g, 5);
  VecCheckSameSize(x, 2, g, 5);
  if (term->ops->objectiveandgradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
  } else if (term->ops->objective && term->ops->gradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
  } else
    SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE,
            "TaoTerm does not have objective and gradient function.  "
            "You should have called some of the following functions: TaoSetObjective(), TaoSetGradient(), TaoSetObjectiveAndGradient(), TaoTermShellSetObjective(), TaoTermShellSetGradient(), TaoTermShellSetObjectiveAndGradient()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "TaoTerm value: %20.19e\n", (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermHessian - Evaluate the Hessian of a `TaoTerm` (with respect to the solution variables) for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameters:
+ H    - Hessian matrix $\nabla_x^2 f(x;p)$
- Hpre - precondiitoning matrix

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`
@*/
PetscErrorCode TaoTermHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  if (H) {
    PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
    PetscCheckSameComm(term, 1, H, 4);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre, MAT_CLASSID, 5);
    PetscCheckSameComm(term, 1, Hpre, 5);
  }
  PetscCall(PetscLogEventBegin(TAOTERM_HessianEval, term, NULL, NULL, NULL));
  PetscUseTypeMethod(term, hessian, x, params, H, Hpre);
  PetscCall(PetscLogEventEnd(TAOTERM_HessianEval, term, NULL, NULL, NULL));
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsObjectiveDefined - Whether this term can call `TaoTermObjective()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the objective is defined

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermObjective()`, `TaoTermShellSetObjective()`, `TaoTermIsGradientDefined()`, `TaoTermIsObjectiveAndGradientDefined()`, `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isobjectivedefined) PetscUseTypeMethod(term, isobjectivedefined, is_defined);
  else *is_defined = (term->ops->objective != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsGradientDefined - Whether this term can call `TaoTermGradient()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the gradient is defined

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermGradient()`, `TaoTermShellSetGradient()`, `TaoTermIsObjectiveDefined()`, `TaoTermIsObjectiveAndGradientDefined()`, `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsGradientDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isgradientdefined) PetscUseTypeMethod(term, isgradientdefined, is_defined);
  else *is_defined = (term->ops->gradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsObjectiveAndGradientDefined - Whether this term can call `TaoTermObjectiveAndGradient()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the objective/gradient is defined

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermObjectiveAndGradient()`, `TaoTermShellSetObjectiveAndGradient()`, `TaoTermIsObjectiveDefined()`, `TaoTermIsGradientDefined()`, `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isobjectiveandgradientdefined) PetscUseTypeMethod(term, isobjectiveandgradientdefined, is_defined);
  else *is_defined = (term->ops->objectiveandgradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsHessianDefined - Whether this term can call `TaoTermHessian()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the hessian is defined

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermHessian()`, `TaoTermShellSetHessian()`, `TaoTermIsObjectiveDefined()`, `TaoTermIsGradientDefined()`, `TaoTermIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoTermIsHessianDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->ishessiandefined) PetscUseTypeMethod(term, ishessiandefined, is_defined);
  else *is_defined = (term->ops->hessian != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsCreateHessianMatricesDefined - Whether this term can call `TaoTermCreateHessianMatrices()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the term can create new Hessian matrices

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermHessian()`, `TaoTermShellSetHessian()`, `TaoTermIsObjectiveDefined()`, `TaoTermIsGradientDefined()`, `TaoTermIsObjectiveAndGradientDefined()`, `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->iscreatehessianmatricesdefined) PetscUseTypeMethod(term, iscreatehessianmatricesdefined, is_defined);
  else *is_defined = (term->ops->createhessianmatrices != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetLayouts - Get the layouts describing the solution vectors and parameter vectors of a `TaoTerm`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ solution_layout   - (optional) the `PetscLayout` for the solution space
- parameters_layout - (optional) the `PetscLayout` for the parameter space

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
  `TaoTermGetVecTypes()`,
  `TaoGermSetSolutionTemplate()`,
  `TaoGermSetParametersTemplate()`,
  `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermGetLayouts(TaoTerm term, PetscLayout *solution_layout, PetscLayout *parameters_layout)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (solution_layout) PetscCall(MatGetLayouts(term->solution_factory, solution_layout, NULL));
  if (parameters_layout) PetscCall(MatGetLayouts(term->parameters_factory, parameters_layout, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetVecTypes - Get the vector types of the solution and parameter vectors of a `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ solution_type   - (optional) the `VecType` for the solution space
- parameters_type - (optional) the `VecType` for the parameter space

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
  `TaoTermGetLayouts()`,
  `TaoGermSetSolutionTemplate()`,
  `TaoGermSetParametersTemplate()`,
  `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermGetVecTypes(TaoTerm term, VecType *solution_type, VecType *parameters_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (solution_type) PetscCall(MatGetVecType(term->solution_factory, solution_type));
  if (parameters_type) PetscCall(MatGetVecType(term->parameters_factory, parameters_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateVecs - Create a solution and/or parameter vector for a `TaoTerm`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ solution   - (optional) a compatible solution vector for `term`
- parameters - (optional) a compatible parameter vector for `term`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
  `TaoTermSetSolutionTemplate()`,
  `TaoTermGetParametersTemplate()`,
  `TaoGermGetLayouts()`,
  `TaoTermGetVecTypes()`,
@*/
PetscErrorCode TaoTermCreateVecs(TaoTerm term, Vec *solution, Vec *parameters)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (term->ops->createvecs) {
    PetscUseTypeMethod(term, createvecs, solution, parameters);
  } else {
    PetscCall(MatCreateVecs(term->solution_factory, NULL, solution));
    PetscCall(MatCreateVecs(term->parameters_factory, NULL, parameters));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMatrices - Create the matrices that can be inputs to `TaoTermHessian()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ H    - (optional) a matrix that can store the Hessian computed in `TaoTermHessian()`
- Hpre - (optional) a preconditioner matrix that can be computed in `TaoTermHessian()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermHessian()`
@*/
PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (H) PetscAssertPointer(H, 2);
  if (Hpre) PetscAssertPointer(Hpre, 3);
  PetscUseTypeMethod(term, createhessianmatrices, H, Hpre);
  PetscFunctionReturn(PETSC_SUCCESS);
}
