#include <petsc/private/nmfimpl.h> /*I "petscnmf.h" I*/

PetscBool         PetscNMFRegisterAllCalled = PETSC_FALSE;
PetscFunctionList PetscNMFList              = NULL;

PetscClassId PETSCNMF_CLASSID;

PetscLogEvent PETSCNMF_Fit;

const char *const PetscNMFInitTypes[] = {"none", "random", "nndsvd", "nndsvda", "nndsvdar", "custom", "PetscNMFInitType", "PETSCNMF_INIT_", NULL};
//TODO do i want enum option flag for this?
const char *const PetscNMFTermParameterTypes[] = {"none", "x", "w", "h", "wh", "PetscNMFTermParameterType", "PETSCNMF_PARAM_", NULL};

/*@C
  PetscNMFRegister - Adds a method to the PetscNMF package for factorization.

  Not Collective, No Fortran Support

  Input Parameters:
+ sname - name of a new user-defined fitter
- func  - routine to Create method context

  Example Usage:
.vb
   PetscNMFRegister("my_nmf", MyNMFCreate);
.ve

  Then, your fitter can be chosen with the procedural interface via
$     PetscNMFSetType(nmf, "my_nmf")
  or at runtime via the option
$     -nmf_type my_nmf_

  Level: advanced

  Note:
  `PetscNMFRegister()` may be called multiple times to add several user-defined fitters.

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFSetType()`, `PetscNMFRegisterAll()`, `PetscNMFRegisterDestroy()`
@*/
PetscErrorCode PetscNMFRegister(const char sname[], PetscErrorCode (*func)(PetscNMF))
{
  PetscFunctionBegin;
  PetscCall(PetscNMFInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscNMFList, sname, func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFSetInitializationType(PetscNMF nmf, PetscNMFInitType type)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFSetRandomContext(PetscNMF nmf, PetscRandom rctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFSetRank(PetscNMF nmf, PetscInt rank)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFCreate - Creates a PetscNMF fitter

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newnmf - the new `PetscNMF` context

  Options Database Key:
. -nmf_type - select which method PetscNMF should use

  Level: beginner

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFFit()`, `PetscNMFDestroy()`, `PetscNMFSetFromOptions()`, `PetscNMFSetType()`
@*/
PetscErrorCode PetscNMFCreate(MPI_Comm comm, PetscNMF *newnmf)
{
  PetscNMF nmf;

  PetscFunctionBegin;
  PetscAssertPointer(newnmf, 2);
  PetscCall(PetscNMFInitializePackage());

  PetscCall(PetscHeaderCreate(nmf, PETSCNMF_CLASSID, "PetscNMF", "NMF Fitter", "PetscNMF", comm, PetscNMFDestroy, PetscNMFView));

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)nmf), &nmf->tao));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)nmf->tao, "nmf_"));
  //TODO is this okay? I dont have orig_callbacks for NMF....
  //I think for tao. term[0] is orig_callbacks, and term[1] is sum
  //for NMF, term[0] is sum...

  //TODO do I actually need this? can I just create TaoTerm sum, and delete it later?
  PetscCall(TaoTermCreate(comm, &nmf->orig_sum));
  PetscCall(TaoTermSetType(nmf->orig_sum, TAOTERMSUM));
  //omitting prefix stuff for now...
  PetscCall(TaoMappedTermSetData(&nmf->objective_term, "objective_", 1.0, nmf->orig_sum, NULL));

  *newnmf = nmf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFFit - Fits an nonnegative matrix facotrization problem.

  Collective

  Input Parameter:
. nmf - the `PetscNMF` context

  Level: beginner

  Notes:
  The user must set up the `PetscNMF` object  with calls to `PetscNMFSetSolution()`, `PetscNMFSetObjective()`, `PetscNMFSetGradient()`, and (if using 2nd order method) `PetscNMFSetHessian()`.

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFCreate()`, `PetscNMFSetUp()`
 @*/
PetscErrorCode PetscNMFFit(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  PetscCall(PetscNMFSetUp(nmf));

  PetscCall(PetscLogEventBegin(PETSCNMF_Fit, nmf, 0, 0, 0));
  PetscTryTypeMethod(nmf, fit);
  PetscCall(PetscLogEventEnd(PETSCNMF_Fit, nmf, 0, 0, 0));

  if (nmf->solution_vec) PetscCall(VecViewFromOptions(nmf->solution_vec, (PetscObject)nmf, "-nmf_view_vec_solution"));
  if (nmf->solution_mat) PetscCall(MatViewFromOptions(nmf->solution_mat, (PetscObject)nmf, "-nmf_view_mat_solution"));

  PetscCall(PetscNMFViewFromOptions(nmf, NULL, "-nmf_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFTransform(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFSetUp - Sets up the internal data structures for the later use
  of a PetscNMF fitter.

  Collective

  Input Parameter:
. nmf - the `PetscNMF` context

  Level: advanced

  Note:
  The user will not need to explicitly call `PetscNMFSetUp()`, as it will
  automatically be called in `PetscNMFFit()`.  However, if the user
  desires to call it explicitly, it should come after `PetscNMFCreate()`
  and any PetscNMFSetSomething() routines, but before `PetscNMFFit()`.

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFCreate()`, `PetscNMFFit()`
@*/
PetscErrorCode PetscNMFSetUp(PetscNMF nmf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  if (nmf->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  //TODO mat vec solution stuff
  //TODO taoterm setup stuff
  PetscTryTypeMethod(nmf, setup);
  nmf->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFDestroy - Destroys the `PetscNMF` context that was created with `PetscNMFCreate()`

  Collective

  Input Parameter:
. nmf - the `PetscNMF` context

  Level: beginner

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFCreate()`, `PetscNMFFit()`
@*/
PetscErrorCode PetscNMFDestroy(PetscNMF *nmf)
{
  PetscInt num_terms;

  PetscFunctionBegin;
  if (!*nmf) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*nmf, PETSCNMF_CLASSID, 1);
  if (--((PetscObject)*nmf)->refct > 0) {
    *nmf = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod(*nmf, destroy);
  //TODO I am assuming that this is TAOTERMSUM... this has to be the case, right?
  PetscCall(TaoTermSumGetNumSubterms((*nmf)->objective_term.term, &num_terms));
  for (PetscInt i = 0; i < num_terms; i++) {
    PetscReal sub_scale;
    TaoTerm   sub_term;

    PetscCall(TaoTermSumGetSubterm((*nmf)->objective_term.term, i, NULL, &sub_scale, &sub_term, NULL));
    PetscCall(PetscObjectCompose((PetscObject)sub_term, "TaoMappedTermGetXParamType_NMF", NULL));
    PetscCall(PetscObjectCompose((PetscObject)sub_term, "TaoMappedTermGetPParamType_NMF", NULL));
  }
  PetscCall(TaoMappedTermReset(&(*nmf)->objective_term));

  PetscTryTypeMethod(*nmf, destroy);

  PetscCall(VecDestroy(&(*nmf)->solution_vec));
  PetscCall(MatDestroy(&(*nmf)->solution_mat));

  //TODO not clear what to do with monitor
  //PetscCall(PetscNMFMonitorCancel(*nmf));
  PetscCall(PetscHeaderDestroy(nmf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFSetFromOptions - Sets various PetscNMF parameters from the options database

  Collective

  Input Parameter:
. nmf - the `PetscNMF` fitter context

  Options Database Keys:
+ -nmf_type <type>         - The algorithm that PetscNMF uses (aoadmm, hals, etc.)
. -nmf_view                - prints information about the PetscNMF after solving
- -nmf_add_objective_terms - takes a list of options prefixes, a `TaoTerm` will be created for each and added to the obj TODO

  Level: beginner

  Note:
  To see all options, run your program with the `-help` option or consult the
  user's manual. Should be called after `PetscNMFCreate()` but before `PetscNMFFit()`

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFCreate()`, `PetscNMFFit()`
@*/
PetscErrorCode PetscNMFSetFromOptions(PetscNMF nmf)
{
  PetscNMFType default_type = PETSCNMFAOADMM;
  char         type[256];
  PetscBool    flg;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)nmf, &comm));

  if (((PetscObject)nmf)->type_name) default_type = ((PetscObject)nmf)->type_name;

  PetscObjectOptionsBegin((PetscObject)nmf);
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-nmf_type", "PetscNMF Fitter type", "PetscNMFSetType", PetscNMFList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(PetscNMFSetType(nmf, type));
  } else if (!((PetscObject)nmf)->type_name) {
    PetscCall(PetscNMFSetType(nmf, default_type));
  }

  PetscCall(PetscOptionsEnum("-nmf_init_type", "Initialization type", "", PetscNMFInitTypes, (PetscEnum)nmf->init_type, (PetscEnum *)&nmf->init_type, NULL));
//  PetscCall(PetscOptionsEnum("-nmf_term_parameter_type", "Initialization type", "", PetscNMFTermParameterTypes, (PetscEnum)nmf->init_type, (PetscEnum *)&nmf->init_type, NULL));a
//

  PetscTryTypeMethod(nmf, setfromoptions, PetscOptionsObject);

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)nmf, PetscOptionsObject));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFViewFromOptions - View a `PetscNMF` object based on values in the options database

  Collective

  Input Parameters:
+ A    - the  `PetscNMF` context
. obj  - Optional object that provides the prefix for the options database
- name - command line option

  Level: intermediate

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFView`, `PetscObjectViewFromOptions()`, `PetscNMFCreate()`
@*/
PetscErrorCode PetscNMFViewFromOptions(PetscNMF A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCNMF_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFView - Prints information about the `PetscNMF` object

  Collective

  Input Parameters:
+ nmf    - the `PetscNMF` context
- viewer - visualization context

  Options Database Key:
. -nmf_view - Calls `PetscNMFView()` at the end of `PetscNMFFit()`

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: [](ch_nmf), `PetscNMF`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode PetscNMFView(PetscNMF nmf, PetscViewer viewer)
{
  PetscBool    isascii, isstring;
  PetscNMFType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)nmf)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(nmf, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)nmf, viewer));

    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(nmf, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(PetscNMFGetType(nmf, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " %-3.3s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFSetComponentsMat(PetscNMF nmf, Mat components)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFGetComponentsMat(PetscNMF nmf, Mat *components)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFSetTransformMat(PetscNMF nmf, Mat transform)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFGetTransformMat(PetscNMF nmf, Mat *transform)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscNMFSetTrainingMat(PetscNMF nmf, Mat training)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO do i even need this?
PetscErrorCode PetscNMFGetTrainingMat(PetscNMF nmf, Mat *training)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFSetType - Sets the `PetscNMFType` for the minimization fitter.

  Collective

  Input Parameters:
+ nmf  - the `PetscNMF` fitter context
- type - a known method

  Options Database Key:
. -nmf_type <type> - Sets the method; use -help for a list
   of available methods (for instance, "-nmf_type aoadmm" or "-nmf_type hals")

  Level: intermediate

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFCreate()`, `PetscNMFGetType()`, `PetscNMFType`
@*/
PetscErrorCode PetscNMFSetType(PetscNMF nmf, PetscNMFType type)
{
  PetscErrorCode (*create_xxx)(PetscNMF);
  PetscBool issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);

  PetscCall(PetscObjectTypeCompare((PetscObject)nmf, type, &issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscNMFList, type, &create_xxx));
  PetscCheck(create_xxx, PetscObjectComm((PetscObject)nmf), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscNMF type %s", type);

  /* Destroy the existing fitter information */
  PetscTryTypeMethod(nmf, destroy);
  nmf->ops->setup          = NULL;
  nmf->ops->fit            = NULL;
  nmf->ops->view           = NULL;
  nmf->ops->setfromoptions = NULL;
  nmf->ops->destroy        = NULL;

  nmf->setupcalled           = PETSC_FALSE;
  //TODO destroy taoterm stuff

  PetscCall((*create_xxx)(nmf));
  PetscCall(PetscObjectChangeTypeName((PetscObject)nmf, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFGetType - Gets the current `PetscNMFType` being used in the `PetscNMF` object

  Not Collective

  Input Parameter:
. nmf - the `PetscNMF` fitter context

  Output Parameter:
. type - the `PetscNMFType`

  Level: intermediate

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFType`, `PetscNMFSetType()`
@*/
PetscErrorCode PetscNMFGetType(PetscNMF nmf, PetscNMFType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)nmf)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFSetOptionsPrefix - Sets the prefix used for searching for all
  PetscNMF options in the database.

  Logically Collective

  Input Parameters:
+ nmf - the `PetscNMF` context
- p   - the prefix string to prepend to all PetscNMF option requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  For example, to distinguish between the runtime options for two
  different PetscNMF solvers, one could call
.vb
      PetscNMFSetOptionsPrefix(nmf1,"sys1_")
      PetscNMFSetOptionsPrefix(nmf2,"sys2_")
.ve

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFSetFromOptions()`, `PetscNMFAppendOptionsPrefix()`, `PetscNMFGetOptionsPrefix()`
@*/
PetscErrorCode PetscNMFSetOptionsPrefix(PetscNMF nmf, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)nmf, p));
  if (nmf->tao) PetscCall(TaoSetOptionsPrefix(nmf->tao, p));
  //TODO taoterm options prefix? callbacks?
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFAppendOptionsPrefix - Appends to the prefix used for searching for all PetscNMF options in the database.

  Logically Collective

  Input Parameters:
+ nmf - the `PetscNMF` solver context
- p   - the prefix string to prepend to all `PetscNMF` option requests

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is automatically the hyphen.

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFSetFromOptions()`, `PetscNMFSetOptionsPrefix()`, `PetscNMFGetOptionsPrefix()`
@*/
PetscErrorCode PetscNMFAppendOptionsPrefix(PetscNMF nmf, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, TAO_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)nmf, p));
  if (nmf->tao) PetscCall(TaoAppendOptionsPrefix(nmf->tao, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFGetOptionsPrefix - Gets the prefix used for searching for all
  PetscNMF options in the database

  Not Collective

  Input Parameter:
. nmf - the `PetscNMF` context

  Output Parameter:
. p - pointer to the prefix string used is returned

  Fortran Notes:
  Pass in a string 'prefix' of sufficient length to hold the prefix.

  Level: advanced

.seealso: [](ch_nmf), `PetscNMF`, `PetscNMFSetFromOptions()`, `PetscNMFSetOptionsPrefix()`, `PetscNMFAppendOptionsPrefix()`
@*/
PetscErrorCode PetscNMFGetOptionsPrefix(PetscNMF nmf, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, TAO_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)nmf, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NMFParamContainerDestroy(void **ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscNMFAddObjectiveTerm - Add a term to the NMF objective.

  Collective

  Input Parameters:
+ nmf     - a `PetscNMF` object context
. prefix  - the prefix used for configuring the new term (if NULL, the index of the term will be used as a prefix, e.g. "0_", "1_", etc.)
. scale   - scaling coefficient for the new term
. term    - the real-valued function defining the new term of `TaoTerm`, f(Ax;p)
. x_param - (optional) type of parameter for main input x
. p_param - (optional) type of parameter for main input p
- map     - (optional) matrix map. Currently does nothing

  Level: beginner

.seealso: [](ch_nmf), `PetscNMF`
@*/
PetscErrorCode PetscNMFAddObjectiveTerm(PetscNMF nmf, const char prefix[], PetscReal scale, TaoTerm term, PetscNMFTermParameterType x_type, PetscNMFTermParameterType p_type, Mat map)
{
  PetscBool      is_sum;
  PetscInt       num_old_terms;
  PetscContainer x_container;
  PetscContainer p_container;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nmf, PETSCNMF_CLASSID, 1);
  if (prefix) PetscAssertPointer(prefix, 2);
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 4);
  PetscCheckSameComm(nmf, 1, term, 4);

  //Note: Almost copy-paste from taosolver.c but little caveat....
  PetscCall(PetscObjectTypeCompare((PetscObject)nmf->objective_term.term, TAOTERMSUM, &is_sum));
  if (!is_sum) {
    TaoTerm     old_sum;
    const char *prefix;
    const char *term_prefix;

    //TODO in Tao, they create initial callbacks term. but nmf has nothing... what to do?
    PetscCall(TaoTermDuplicate(nmf->objective_term.term, TAOTERM_DUPLICATE_SIZEONLY, &old_sum));
    if (nmf->objective_term.map) {
      VecType     map_vectype;
      VecType     param_vectype;
      PetscLayout cmap, param_layout;

      PetscCall(MatGetVecType(nmf->objective_term.map, &map_vectype));
      PetscCall(TaoTermGetVecTypes(old_sum, NULL, &param_vectype));
      PetscCall(TaoTermSetVecTypes(old_sum, map_vectype, param_vectype));
      PetscCall(MatGetLayouts(nmf->objective_term.map, NULL, &cmap));
      PetscCall(TaoTermGetLayouts(old_sum, NULL, &param_layout));
      PetscCall(TaoTermSetLayouts(old_sum, cmap, param_layout));
    }
    PetscCall(TaoTermSetType(old_sum, TAOTERMSUM));
    PetscCall(PetscNMFGetOptionsPrefix(nmf, &prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)old_sum, prefix));
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)old_sum, "objective_"));
    PetscCall(TaoTermSumSetNumSubterms(old_sum, 1));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)nmf->objective_term.term, &term_prefix));
    PetscCall(TaoTermSumSetSubterm(old_sum, 0, term_prefix, nmf->objective_term.scale, nmf->objective_term.term, nmf->objective_term.map));
    // No hessian stuff
    PetscCall(TaoMappedTermReset(&nmf->objective_term));
    PetscCall(TaoMappedTermSetData(&nmf->objective_term, "objective_", 1.0, old_sum, NULL));
    //No param vec stuff here
    PetscCall(TaoTermDestroy(&old_sum));
  }
  if (nmf->objective_term.scale != 1.0 || nmf->objective_term.map != NULL) {
    PetscInt num_terms;

    PetscCall(TaoTermSumGetNumSubterms(nmf->objective_term.term, &num_terms));
    for (PetscInt i = 0; i < num_terms; i++) {
      const char *sub_name;
      PetscReal   sub_scale;
      TaoTerm     sub_term;

      //TODO notsure what to do with sub_map stuff...
      PetscCall(TaoTermSumGetSubterm(nmf->objective_term.term, i, &sub_name, &sub_scale, &sub_term, NULL));
      sub_scale *= nmf->objective_term.scale;
      PetscCall(TaoTermSumSetSubterm(nmf->objective_term.term, i, sub_name, sub_scale, sub_term, NULL));
    }
    PetscCall(TaoMappedTermSetData(&nmf->objective_term, nmf->objective_term.prefix, 1.0, nmf->objective_term.term, NULL));
  }
  PetscCall(TaoTermSumGetNumSubterms(nmf->objective_term.term, &num_old_terms));
  // no paramvec unpack stuff here
  PetscCall(TaoTermSumAddSubterm(nmf->objective_term.term, prefix, scale, term, map, NULL));
  // no vec_list stuff

  //TODO whats the best way to do this????
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)nmf), &x_container));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)nmf), &p_container));
  PetscCall(PetscContainerSetPointer(x_container, &x_type));
  PetscCall(PetscContainerSetPointer(p_container, &p_type));
  PetscCall(PetscContainerSetCtxDestroy(x_container, NMFParamContainerDestroy));
  PetscCall(PetscContainerSetCtxDestroy(p_container, NMFParamContainerDestroy));
  PetscCall(PetscObjectCompose((PetscObject)term, "TaoMappedTermGetXParamType_NMF", (PetscObject)x_container));
  PetscCall(PetscObjectCompose((PetscObject)term, "TaoMappedTermGetPParamType_NMF", (PetscObject)p_container));
  PetscFunctionReturn(PETSC_SUCCESS);
}
