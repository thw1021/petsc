#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

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
