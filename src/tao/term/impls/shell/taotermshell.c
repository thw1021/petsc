#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Shell TaoTerm_Shell;

struct _n_TaoTerm_Shell {
  PetscContainer ctxcontainer;
};

/*@C
  TaoTermShellSetContextDestroy - Set a method to destroy user context resources when a `TAOTERMSHELL` is destroyed

  Logically collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSHELL`
- destroy - the context destroy function

  Calling sequence of `destroy`:
. ctx - the user context provided in `TaoTermShellSetContext()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellSetContext()`, `TaoTermShellGetContext()`
@*/
PetscErrorCode TaoTermShellSetContextDestroy(TaoTerm term, PetscErrorCode (*destroy)(void *ctx))
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetContextDestroy_Shell(TaoTerm mat, PetscErrorCode (*f)(void *))
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)mat->data;

  PetscFunctionBegin;
  if (shell->ctxcontainer) PetscCall(PetscContainerSetUserDestroy(shell->ctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermShellGetContext - Get the user-specified context for a `TAOTERMSHELL`

  Logically collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMSHELL`

  Output Parameter:
. ctx - a user context

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellSetContext()`, `TaoTermShellSetContextDestroy()`
@*/
PetscErrorCode TaoTermShellGetContext(TaoTerm term, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  PetscUseMethod(term, "TaoTermShellGetContext_C", (TaoTerm, void *), (term, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellGetContext_Shell(TaoTerm term, void *ctx)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  if (shell->ctxcontainer) PetscCall(PetscContainerGetPointer(shell->ctxcontainer, (void **)ctx));
  else *(void **)ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermShellSetContext - Set a user context for a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMSHELL`
- ctx  - a user context

  Level: intermediate

  Note:
  The user context can be accessed in callbacks using `TaoTermShellGetContext()`

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`
@*/
PetscErrorCode TaoTermShellSetContext(TaoTerm term, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscUseMethod(term, "TaoTermShellSetContext_C", (TaoTerm, void *), (term, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetContext_Shell(TaoTerm term, void *ctx)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  if (ctx) {
    PetscContainer ctxcontainer;

    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)term, "TaoTermShell ctx", (PetscObject)ctxcontainer));
    shell->ctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  } else {
    PetscCall(PetscObjectCompose((PetscObject)term, "TaoTermShell ctx", NULL));
    shell->ctxcontainer = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Shell(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscBool any;
    if (shell->ctxcontainer) PetscCall(PetscViewerASCIIPrintf(viewer, "User context has been set\n"));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "No user context has been set\n"));

    PetscCall(PetscViewerASCIIPrintf(viewer, "The following methods have been set:"));
    any = PETSC_FALSE;
    if (term->ops->objective) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " objective,"));
    }
    if (term->ops->gradient) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " gradient,"));
    }
    if (term->ops->objectiveandgradient) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " objectiveandgradient,"));
    }
    if (term->ops->hessian) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " hessian,"));
    }
    if (term->ops->proximalmap) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " proximalmap"));
    }
    if (!any) PetscCall(PetscViewerASCIIPrintf(viewer, " (none)"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermDestroy_Shell(TaoTerm term)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  PetscCall(PetscContainerDestroy(&shell->ctxcontainer));
  PetscCall(PetscFree(shell));
  term->data                      = NULL;
  term->ops->objective            = NULL;
  term->ops->gradient             = NULL;
  term->ops->objectiveandgradient = NULL;
  term->ops->hessian              = NULL;
  term->ops->proximalmap          = NULL;
  term->ops->view                 = TaoTermView_Shell;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContextDestroy_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContext_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellGetContext_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjectiveAndGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetHessian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetProximalMap_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetView_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetObjective - Set the objective function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term      - a `TaoTerm` of type `TAOTERMSHELL`
- objective - an objective function with the same signature as `TaoTermObjective()`

  Calling sequence of `objective`:
+ term   - the `TaoTerm`
. x      - a value of the solution variables
. params - a value of the parameters (may be NULL if the term is not parametric)
- value  - value that will be returned in `TaoTermObjective()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetProximalMap()`,
          `TaoTermShellSetView()`,
@*/
PetscErrorCode TaoTermShellSetObjective(TaoTerm term, PetscErrorCode (*objective)(TaoTerm term, Vec x, Vec params, PetscReal *value))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetObjective_C", (TaoTerm, PetscErrorCode(*)(TaoTerm, Vec, Vec, PetscReal *)), (term, objective));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetObjective_Shell(TaoTerm term, PetscErrorCode (*objective)(TaoTerm, Vec, Vec, PetscReal *))
{
  PetscFunctionBegin;
  term->ops->objective = objective;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetGradient - Set the gradient function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term     - a `TaoTerm` of type `TAOTERMSHELL`
- gradient - a gradient function with the same signature as `TaoTermGradient()`

  Calling sequence of `gradient`:
+ term   - the `TaoTerm`
. x      - a value of the solution variables
. params - a value of the parameters (may be NULL if the term is not parametric)
- g      - gradient vector that will be set in `TaoTermGradient()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetProximalMap()`,
          `TaoTermShellSetView()`,
@*/
PetscErrorCode TaoTermShellSetGradient(TaoTerm term, PetscErrorCode (*gradient)(TaoTerm term, Vec x, Vec params, Vec g))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetGradient_C", (TaoTerm, PetscErrorCode(*)(TaoTerm, Vec, Vec, Vec)), (term, gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetGradient_Shell(TaoTerm term, PetscErrorCode (*gradient)(TaoTerm, Vec, Vec, Vec))
{
  PetscFunctionBegin;
  term->ops->gradient = gradient;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetObjectiveAndGradient - Set the gradient function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term       - a `TaoTerm` of type `TAOTERMSHELL`
- objandgrad - a function with the same signature as `TaoTermObjectiveAndGradient()`

  Calling sequence of `objandgrad`:
+ term   - the `TaoTerm`
. x      - a value of the solution variables
. params - a value of the parameters (may be NULL if the term is not parametric)
. value  - value that will be returned in `TaoTermObjectiveAndGradeint()`
- g      - gradient vector that will be set in `TaoTermObjectiveAndGradient()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetProximalMap()`,
          `TaoTermShellSetView()`,
@*/
PetscErrorCode TaoTermShellSetObjectiveAndGradient(TaoTerm term, PetscErrorCode (*objandgrad)(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetObjectiveAndGradient_C", (TaoTerm, PetscErrorCode(*)(TaoTerm, Vec, Vec, PetscReal *, Vec)), (term, objandgrad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetObjectiveAndGradient_Shell(TaoTerm term, PetscErrorCode (*objandgrad)(TaoTerm, Vec, Vec, PetscReal *, Vec))
{
  PetscFunctionBegin;
  term->ops->objectiveandgradient = objandgrad;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetHessian - Set the Hessian function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSHELL`
- hessian - a Hessian function with the same signature as `TaoTermHessian()`

  Calling sequence of `hessian`:
+ term   - the `TaoTerm`
. x      - a value of the solution variables
. params - a value of the parameters (may be NULL if the term is not parametric)
. H      - Hessian matrix that will be returned in `TaoTermHessian()`
- Hpre   - preconditioning matrix that will be returned in `TaoTermHessian()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetProximalMap()`,
          `TaoTermShellSetView()`,
@*/
PetscErrorCode TaoTermShellSetHessian(TaoTerm term, PetscErrorCode (*hessian)(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetHessian_C", (TaoTerm, PetscErrorCode(*)(TaoTerm, Vec, Vec, Mat, Mat)), (term, hessian));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetHessian_Shell(TaoTerm term, PetscErrorCode (*hessian)(TaoTerm, Vec, Vec, Mat, Mat))
{
  PetscFunctionBegin;
  term->ops->hessian = hessian;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetProximalMap - Set the proximal map function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term        - a `TaoTerm` of type `TAOTERMSHELL`
- proximalmap - a proximal map function with the same signature as `TaoTermProximalMap()`

  Calling sequence of `hessian`:
+ term    - the `TaoTerm`
. params  - a value of the parameters (may be NULL if `term` is not parametric)
. scale   - the scale of `term`
. g       - a second `TaoTerm`
. gparams - the parameters of `g`
. gscale  - the scale of `g`
- sol     - the proximal map solution that will be returned in `TaoTermProximalMap()`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetView()`,
@*/
PetscErrorCode TaoTermShellSetProximalMap(TaoTerm term, PetscErrorCode (*proximalmap)(TaoTerm term, Vec params, PetscReal scale, TaoTerm g, Vec gparams, PetscReal gscale, Vec sol))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetProximalMap_C", (TaoTerm, PetscErrorCode(*)(TaoTerm, Vec, PetscReal, TaoTerm, Vec, PetscReal, Vec)), (term, proximalmap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetProximalMap_Shell(TaoTerm term, PetscErrorCode (*proximalmap)(TaoTerm, Vec, PetscReal, TaoTerm, Vec, PetscReal, Vec))
{
  PetscFunctionBegin;
  term->ops->proximalmap = proximalmap;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetView - Set the view function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMSHELL`
- view - a function with the same signature as `TaoTermView()`

  Calling sequence of `view`:
+ term   - the `TaoTerm`
- viewer - a `PetscViewer`

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetProximalMap()`,
@*/
PetscErrorCode TaoTermShellSetView(TaoTerm term, PetscErrorCode (*view)(TaoTerm term, PetscViewer viewer))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetView_C", (TaoTerm, PetscErrorCode(*)(TaoTerm, PetscViewer)), (term, view));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetView_Shell(TaoTerm term, PetscErrorCode (*view)(TaoTerm, PetscViewer))
{
  PetscFunctionBegin;
  term->ops->view = view;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMSHELL - A `TaoTerm` that collects user-defined callbacks for the operations of the term

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetProximalMap()`,
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Shell(TaoTerm term)
{
  TaoTerm_Shell *shell;

  PetscFunctionBegin;
  PetscCall(PetscNew(&shell));
  term->data = (void *)shell;

  term->ops->destroy = TaoTermDestroy_Shell;
  term->ops->view    = TaoTermView_Shell;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContextDestroy_C", TaoTermShellSetContextDestroy_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContext_C", TaoTermShellSetContext_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellGetContext_C", TaoTermShellGetContext_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjective_C", TaoTermShellSetObjective_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetGradient_C", TaoTermShellSetGradient_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjectiveAndGradient_C", TaoTermShellSetObjectiveAndGradient_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetHessian_C", TaoTermShellSetHessian_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetProximalMap_C", TaoTermShellSetProximalMap_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetView_C", TaoTermShellSetView_Shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}
