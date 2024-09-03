#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h> /*I "petsctao.h" I*/

PetscClassId TAOTERM_CLASSID;

PetscLogEvent TAOTERM_ObjectiveEval;
PetscLogEvent TAOTERM_GradientEval;
PetscLogEvent TAOTERM_ObjGradEval;
PetscLogEvent TAOTERM_HessianEval;

/*@
  TaoTermDestroy - Destroy a description of a `TaoTerm`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
- viewer - a `PetscViewer`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermCreate()`, `TaoTermSetUp()`, `TaoTermSetFromOptions()`, `TaoTermView()`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermDestroy(TaoTerm *taoterm)
{
  PetscFunctionBegin;
  if (!*taoterm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*taoterm, TAOTERM_CLASSID, 1);
  if (--((PetscObject)*taoterm)->refct > 0) {
    *taoterm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod(*taoterm, destroy);
  PetscCall(VecDestroy(&(*taoterm)->solution_template));
  PetscCall(VecDestroy(&(*taoterm)->parameter_template));

  PetscCall(PetscHeaderDestroy(taoterm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermView - View a description of a `TaoTerm`

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
- viewer - a `PetscViewer`

  Level: intermediate

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
  TaoTermSetUp - Set up a `TaoTerm`

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
. -tao_term_type <type> - tao, shell, dm, separable, l1, linf, l2squared, quadratic, kl, `TaoTermType` for complete list

  Level: intermediate

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
  PetscCall(PetscOptionsFList("-tao_term_type", "TaoTerm type", "TaoTermType", TaoTermList, deft, type, 256, &flg));
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
. -tao_term_type <type> - Sets the method; use -help for a list
   of available methods (for instance, newtonls or newtontr)

  Level: intermediate

  Note: new types of `TaoTerm` can be created with `TaoTermRegister()`

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TaoTermType`, `TaoTermCreate()`, `TaoTermDestroy()`, `TaoTermView()`, `TaoTermSetUp()`, `TaoTermSetType()`
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
  PetscCheck(create, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested Tao type %s", type);

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

  Intput Parameter:
. comm - communicator for MPI processes that compute the term

  Output Parameter:
. term - a new TaoTerm

  Level: intermediate

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
. value  - the value of $f(x; p)$

  Level: intermediate

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
    term->nobj++;
  } else if (term->ops->objectiveandgradient) {
    Vec temp;

    PetscCall(PetscInfo(term, "Duplicating solution vector in order to call objective/gradient routine\n"));
    PetscCall(VecDuplicate(x, &temp));
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, temp);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscCall(VecDestroy(&temp));
    term->nobjgrad++;
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
. g      -  the value of $\nabla_x f(x; p)$

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`
@*/
PetscErrorCode TaoTermGradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBegin;
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
    term->ngrad++;
  } else if (term->ops->objectiveandgradient) {
    PetscReal value;

    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, &value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    term->nobjgrad++;
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
+ value  -  the value of $f(x; p)$
- g      -  the value of $\nabla_x f(x; p)$

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermHessian()`
@*/
PetscErrorCode TaoTermObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscFunctionBegin;
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
  PetscAssertPointer(value, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(term, 1, g, 5);
  VecCheckSameSize(x, 2, g, 5);
  if (term->ops->objectiveandgradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    term->nobjgrad++;
  } else if (term->ops->objective && term->ops->gradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    term->nobj++;
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    term->ngrad++;
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have objective and gradient function.  "
      "You should have called some of the following functions: TaoSetObjective(), TaoSetGradient(), TaoSetObjectiveAndGradient(), TaoTermShellSetObjective(), TaoTermShellSetGradient(), TaoTermShellSetObjectiveAndGradient()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "TaoTerm value: %20.19e\n", (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
