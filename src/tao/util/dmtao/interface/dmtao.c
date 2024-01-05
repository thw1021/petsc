#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/

PetscBool         DMTaoRegisterAllCalled = PETSC_FALSE;
PetscFunctionList DMTaoList              = NULL;

PetscClassId DMTAO_CLASSID = 0;

PetscLogEvent DMTAO_Eval;

static PetscErrorCode DMTaoDestroy(DMTao *kdm)
{
  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*kdm), DMTAO_CLASSID, 1);
  PetscCall(VecDestroy(&(*kdm)->workvec));
  if (--((PetscObject)(*kdm))->refct > 0) {
    *kdm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*kdm)->ops->destroy) PetscCall(((*kdm)->ops->destroy)(*kdm));
  PetscCall(PetscHeaderDestroy(kdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoView - Prints information about the `DMTao` object

  Collective

  Input Parameters:
+ kdm    - the `DMTao` context
- viewer - visualization context

  Options Database Key:
. -dm_tao_view - Calls `DMTaoView()` at the end of `TaoSolve()` or `DMTaoApplyProximalMap()`.

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: [](ch_tao), `DMTao`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode DMTaoView(DM kdm, PetscViewer viewer)
{
  PetscBool isascii, isbinary;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(DMGetDMTao(kdm, &tdm));
  if (isascii) {
#if defined(PETSC_SERIALIZE_FUNCTIONS)
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)tdm, viewer));

    const char *fname;

    PetscCall(PetscFPTFind(tdm->ops->computeobjective, &fname));
    if (fname) PetscCall(PetscViewerASCIIPrintf(viewer, "Objective used by DMTao: %s\n", fname));
    PetscCall(PetscFPTFind(tdm->ops->computegradient, &fname));
    if (fname) PetscCall(PetscViewerASCIIPrintf(viewer, "Gradient function used by DMTao: %s\n", fname));
    PetscCall(PetscFPTFind(tdm->ops->computeobjectiveandgradient, &fname));
    if (fname) PetscCall(PetscViewerASCIIPrintf(viewer, "Objective and Gradient function used by DMTao: %s\n", fname));
    PetscCall(PetscViewerASCIIPrintf(viewer, "DMTao scale=%g,", (double)tdm->scale));

    /* TODO pushtabascii what to print here? */
#endif
  } else if (isbinary) {
    struct {
      PetscErrorCode (*obj)(DM, Vec, PetscReal *, void *);
    } objstruct;
    struct {
      PetscErrorCode (*grad)(DM, Vec, Vec, void *);
    } gradstruct;
    struct {
      PetscErrorCode (*objgrad)(DM, Vec, PetscReal *, Vec, void *);
    } objgradstruct;
    objstruct.obj         = tdm->ops->computeobjective;
    gradstruct.grad       = tdm->ops->computegradient;
    objgradstruct.objgrad = tdm->ops->computeobjectiveandgradient;
    PetscCall(PetscViewerBinaryWrite(viewer, &objstruct, 1, PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer, &gradstruct, 1, PETSC_FUNCTION));
    PetscCall(PetscViewerBinaryWrite(viewer, &objgradstruct, 1, PETSC_FUNCTION));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoCreate(MPI_Comm comm, DMTao *kdm)
{
  DMTao dm;
  PetscFunctionBegin;
  PetscAssertPointer(kdm, 2);
  PetscCall(DMTaoInitializePackage());
  PetscCall(PetscHeaderCreate(dm, DMTAO_CLASSID, "DMTao", "DMTao", "DMTao", comm, DMTaoDestroy, DMTaoView));
  dm->scale = 1.;
  *kdm      = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMTaoCopy - copies the information in a `DMTao` to another `DMTao`

  Not Collective

  Input Parameters:
+ kdm  - Original `DMTao`
- nkdm - `DMTao` to receive the data, should have been created with `DMTaoCreate()`

  Level: developer

.seealso: `DMTao`, `DMTaoCreate()`, `DMTaoDestroy()`
*/
static PetscErrorCode DMTaoCopy(DMTao kdm, DMTao nkdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(kdm, DMTAO_CLASSID, 1);
  PetscValidHeaderSpecific(nkdm, DMTAO_CLASSID, 2);
  nkdm->ops->computeobjective            = kdm->ops->computeobjective;
  nkdm->ops->computegradient             = kdm->ops->computegradient;
  nkdm->ops->computeobjectiveandgradient = kdm->ops->computeobjectiveandgradient;
  nkdm->ops->setup                       = kdm->ops->setup;
  nkdm->ops->destroy                     = kdm->ops->destroy;
  nkdm->ops->duplicate                   = kdm->ops->duplicate;

  nkdm->userctx_func     = kdm->userctx_func;
  nkdm->userctx_grad     = kdm->userctx_grad;
  nkdm->userctx_funcgrad = kdm->userctx_funcgrad;
  nkdm->data             = kdm->data;
  nkdm->originaldm       = kdm->originaldm;

  /* implementation specific copy hooks */
  PetscTryTypeMethod(kdm, duplicate, nkdm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetDMTao - get read-only private `DMTao` context from a `DM`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameter:
. taodm - private `DMTao` context

  Level: developer

  Note:
  Use `DMGetDMTaoWrite()` if write access is needed. The DMTaoSetXXX API should be used wherever possible.

.seealso: `DMGetDMTaoWrite()`
@*/
PetscErrorCode DMGetDMTao(DM dm, DMTao *taodm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *taodm = (DMTao)dm->dmtao;
  if (!*taodm) {
    PetscCall(PetscInfo(dm, "Creating new DMTao\n"));
    PetscCall(DMTaoCreate(PetscObjectComm((PetscObject)dm), taodm));

    dm->dmtao            = (PetscObject)*taodm;
    (*taodm)->originaldm = dm;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMGetDMTaoWrite - get write access to private `DMTao` context from a `DM`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameter:
. taodm - private `DMTao` context

  Level: developer

.seealso: `DMGetDMTao()`
@*/
PetscErrorCode DMGetDMTaoWrite(DM dm, DMTao *taodm)
{
  DMTao sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &sdm));
  PetscCheck(sdm->originaldm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DMTao has a NULL originaldm");
  if (sdm->originaldm != dm) { /* Copy on write */
    DMTao oldsdm = sdm;
    PetscCall(PetscInfo(dm, "Copying DMTao due to write\n"));
    PetscCall(DMTaoCreate(PetscObjectComm((PetscObject)dm), &sdm));
    PetscCall(DMTaoCopy(oldsdm, sdm));
    PetscCall(DMTaoDestroy((DMTao *)&dm->dmtao));
    dm->dmtao       = (PetscObject)sdm;
    sdm->originaldm = dm;
  }
  *taodm = sdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMCopyDMTao - copies a `DMTao` context to a new `DM`

  Logically Collective

  Input Parameters:
+ dmsrc  - `DM` to obtain context from
- dmdest - `DM` to add context to

  Level: developer

  Note:
  The context is copied by reference. This function does not ensure that a context exists.

.seealso: `DMGetDMTao()`, `TaoSetDM()`
@*/
PetscErrorCode DMCopyDMTao(DM dmsrc, DM dmdest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmdest, DM_CLASSID, 2);
  if (!dmdest->dmtao) PetscCall(DMTaoCreate(PetscObjectComm((PetscObject)dmdest), (DMTao *)&dmdest->dmtao));
  PetscCall(DMTaoCopy((DMTao)dmsrc->dmtao, (DMTao)dmdest->dmtao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoSetObjective - set `Tao` objective function

  Not Collective

  Input Parameters:
+ dm  - DM to be used with `Tao`
. f   - objective evaluation function
- ctx - context for objective function

  Level: advanced

.seealso: `TaoSetObjective`
@*/
PetscErrorCode DMTaoSetObjective(DM dm, PetscErrorCode (*f)(DM, Vec, PetscReal *, void *), void *ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (f || ctx) PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (f) tdm->ops->computeobjective = f;
  if (ctx) tdm->userctx_func = ctx;
  PetscCall(DMTaoSetType(dm, DMTAOSHELL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
/*@C
  DMTaoGetObjective - get `Tao` objective evaluation function

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameters:
+ obj - objective evaluation function
- ctx - context for objective evaluation

  Level: advanced

.seealso: `DMTaoSetObjective()`
@*/
PetscErrorCode DMTaoGetObjective(DM dm, PetscErrorCode (**obj)(DM, Vec, PetscReal *, void *), void **ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (obj) *obj = tdm->ops->computeobjective;
  if (ctx) *ctx = tdm->userctx_func;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
  DMTaoSetGradient - set `Tao` gradient function

  Not Collective

  Input Parameters:
+ dm  - DM to be used with `Tao`
. f   - gradient evaluation function
- ctx - context for gradient function

  Level: advanced

.seealso: `TaoSetGradient`
@*/
PetscErrorCode DMTaoSetGradient(DM dm, PetscErrorCode (*f)(DM, Vec, Vec, void *), void *ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (f || ctx) PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (f) tdm->ops->computegradient = f;
  if (ctx) tdm->userctx_grad = ctx;
  PetscCall(DMTaoSetType(dm, DMTAOSHELL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
/*@C
  DMTaoGetGradient - get `Tao` gradient evaluation function

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameters:
+ obj - gradient evaluation function
- ctx - context for gradient evaluation

  Level: advanced

.seealso: `DMTaoSetGradient()`
@*/
PetscErrorCode DMTaoGetGradient(DM dm, PetscErrorCode (**obj)(DM, Vec, Vec, void *), void **ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (obj) *obj = tdm->ops->computegradient;
  if (ctx) *ctx = tdm->userctx_grad;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
  DMTaoSetObjectiveAndGradient - set `Tao` objective and gradient function

  Not Collective

  Input Parameters:
+ dm  - DM to be used with `Tao`
. f   - objective and gradient evaluation function
- ctx - context for objective and gradient function

  Level: advanced

.seealso: `TaoSetObjectiveAndGradient`
@*/
PetscErrorCode DMTaoSetObjectiveAndGradient(DM dm, PetscErrorCode (*f)(DM, Vec, PetscReal *, Vec, void *), void *ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (f || ctx) PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (f) tdm->ops->computeobjectiveandgradient = f;
  if (ctx) tdm->userctx_funcgrad = ctx;
  PetscCall(DMTaoSetType(dm, DMTAOSHELL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
/*@C
  DMTaoGetObjectiveAndGradient - get `Tao` objective and gradient evaluation function

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameters:
+ obj - objective gradient evaluation function
- ctx - context for objective and gradient evaluation

  Level: advanced

.seealso: `DMTaoSetObjectiveAndGradient()`
@*/
PetscErrorCode DMTaoGetObjectiveAndGradient(DM dm, PetscErrorCode (**obj)(DM, Vec, PetscReal *, Vec, void *), void **ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (obj) *obj = tdm->ops->computeobjectiveandgradient;
  if (ctx) *ctx = tdm->userctx_funcgrad;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
  DMTaoSetFromOptions - Sets various `DMTao` parameters from user
  options.

  Collective

  Input Parameter:
. dm - the `DM` context

  Options Database Keys:
+ -dmtao_type <type>   - The type of `DMTao` (L1,L2,KL,SimplexUSER)
. -dmtao_scale <scale> - scalar scale for DMTao
- -dmtao_view          - display line-search results to standard output

  Level: beginner

.seealso: `DMTao`
@*/
PetscErrorCode DMTaoSetFromOptions(DM dm)
{
  const char *default_type = DMTAOL2;
  char        type[256];
  PetscBool   flg;
  DMTao       tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)dm);
  if (((PetscObject)dm)->type_name) default_type = ((PetscObject)dm)->type_name;
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-dmtao_type", "DMTao type", "DMTaoSetType", DMTaoList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(DMTaoSetType(dm, type));
  } else if (!((PetscObject)dm)->type_name) {
    PetscCall(DMTaoSetType(dm, default_type));
  }
  PetscCall(DMGetDMTaoWrite(dm, &tdm));

  PetscCall(PetscOptionsReal("-dm_tao_scale", "DMTao scale", "DMTaoSetScale", tdm->scale, &tdm->scale, NULL));
  PetscTryTypeMethod(dm, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoViewFromOptions - View a `DMTao` object based on values in the options database

  Collective

  Input Parameters:
+ dm   - the `DMTao` context
. obj  - Optional object
- name - command line option

  Level: intermediate

  Note:
  See `PetscObjectViewFromOptions()` for available viewer options

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoView()`, `PetscObjectViewFromOptions()`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoViewFromOptions(DM dm, PetscObject obj, const char name[])
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tdm, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetUp - Sets up the internal data structures for the later use
  of a `DMTao`

  Collective

  Input Parameter:
. dm - the `DMTao` context

  Level: developer

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`, `DMTaoApply()`
@*/
PetscErrorCode DMTaoSetUp(DM dm)
{
  const char *default_type = DMTAOL2;
  PetscBool   flg;
  DMTao       tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (!((PetscObject)dm)->type_name) PetscCall(DMTaoSetType(dm, default_type));
  PetscTryTypeMethod(dm, setup);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (tdm->usetaoroutines) {
    PetscCall(TaoIsObjectiveDefined(tdm->dm_subtao, &flg));
    tdm->hasobjective = flg;
    PetscCall(TaoIsGradientDefined(tdm->dm_subtao, &flg));
    tdm->hasgradient = flg;
    PetscCall(TaoIsObjectiveAndGradientDefined(tdm->dm_subtao, &flg));
    tdm->hasobjectiveandgradient = flg;
    /* TODO Hessian */
  } else {
    if (tdm->ops->computeobjective) {
      tdm->hasobjective = PETSC_TRUE;
    } else {
      tdm->hasobjective = PETSC_FALSE;
    }
    if (tdm->ops->computegradient) {
      tdm->hasgradient = PETSC_TRUE;
    } else {
      tdm->hasgradient = PETSC_FALSE;
    }
    if (tdm->ops->computeobjectiveandgradient) {
      tdm->hasobjectiveandgradient = PETSC_TRUE;
    } else {
      tdm->hasobjectiveandgradient = PETSC_FALSE;
    }
  }
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoSetType - Sets the algorithm used in a DMTao

  Collective

  Input Parameters:
+ dm   - the `DM` context
- type - the `DMTaoType` selection

  Options Database Key:
. -dmtao_type <type> - select which method DMTao should use at runtime

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoType`, `DMTaoCreate()`, `DMTaoGetType()`,
@*/
PetscErrorCode DMTaoSetType(DM dm, DMTaoType type)
{
  PetscErrorCode (*r)(DMTao);
  PetscBool flg;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, type, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(DMTaoList, type, (void (**)(void)) & r));
  PetscCheck(r, PetscObjectComm((PetscObject)tdm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested DMTao type %s", type);
  PetscTryTypeMethod(tdm, destroy);
  tdm->scale = 1.0;

  dm->setupcalled = PETSC_FALSE;
  PetscCall((*r)(tdm));
  PetscCall(PetscObjectChangeTypeName((PetscObject)tdm, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoGetType - Gets the current dm algorithm

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. type - the DMTao algorithm in effect

  Level: developer

.seealso: `DMTao`
@*/
PetscErrorCode DMTaoGetType(DM dm, DMTaoType *type)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  *type = ((PetscObject)tdm)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoIsUsingTaoRoutines - Checks whether the DMTao is using
  the standard `Tao` evaluation routines.

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. flg - `PETSC_TRUE` if the DMTao is using `Tao` evaluation routines,
        otherwise `PETSC_FALSE`

  Level: developer

.seealso: `DMTao`
@*/
PetscErrorCode DMTaoIsUsingTaoRoutines(DM dm, PetscBool *flg)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  *flg = tdm->usetaoroutines;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoUseTaoRoutines - Informs the `DMTao` to use the
  objective and gradient evaluation routines from the given `Tao` object.

  Logically Collective

  Input Parameters:
+ dm  - the `DM` context
- tao - the `Tao` context with defined objective/gradient evaluation routines

  Level: developer

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoUseTaoRoutines(DM dm, Tao tao)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  tdm->dm_subtao      = tao;
  tdm->usetaoroutines = PETSC_TRUE;
  tao->is_child_dm    = PETSC_TRUE;
  PetscCall(DMTaoSetType(dm, DMTAOSHELL));
  PetscCall(PetscObjectCompose((PetscObject)tao, "TaoGetParentDM", (PetscObject)dm)); //TODO unsetting compose is done in taosolver.c....
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetCentralVector - Sets a the central vector of DMTao, for metric.

  Logically Collective

  Input Parameter:
+ dm - the `DM` context
- y  - The initial point of the metric

  Level: advanced

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoGetCentralVector()`
@*/
PetscErrorCode DMTaoSetCentralVector(DM dm, Vec y)
{
  DMTao tdm;

  PetscFunctionBegin;
  if (!y) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(VecDestroy(&tdm->y));
  tdm->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetCentralVector - Gets a the central vector of DMTao, for metric.

  Logically Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. y - The central vector of the DMTao

  Level: advanced

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoSetCentralVector()`
@*/
PetscErrorCode DMTaoGetCentralVector(DM dm, Vec *y)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(y, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (y) *y = tdm->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetScale - Sets scale for the dm.
  If this value is not set then 1.0 is assumed.

  Logically Collective

  Input Parameters:
+ dm - the `DM` context
- s  - the scale

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoGetScale()`
@*/
PetscErrorCode DMTaoSetScale(DM dm, PetscReal s)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(dm, s, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  tdm->scale = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetScale - Get scale of DMTao.

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. s - the current scale

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoGetScale(DM dm, PetscReal *s)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(s, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  *s = tdm->scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetVM - Sets VM matrix for the DM.

  Logically Collective

  Input Parameters:
+ dm - the `DM` context
- vm - the VM matrix.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoGetVM()`
@*/
PetscErrorCode DMTaoSetVM(DM dm, Mat vm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (vm) PetscValidHeaderSpecific(vm, MAT_CLASSID, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  tdm->vm = vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetVM - Get VM matrix of DMTao.

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. vm - the current VM matrix.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoGetVM(DM dm, Mat *vm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(vm, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  *vm = tdm->vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetRegularizer - Sets an DMTao to Tao object.
  It treats DMTao object as a regularizer to primary objective.
  TaoSetSolution needs to be called before this routine.

  Input Parameters:
+ tao - Tao solver context
- dm  - DMTao context

  Level: advanced

.seealso: `TaoGetRegularizer()`
@*/
PetscErrorCode TaoSetRegularizer(Tao tao, DM dm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCheck(tdm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "DMTao has not been set for DM.");
  tao->reg = dm;
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "TaoSetSolution needs to be called first.");
  if (!tdm->workvec) { PetscCall(VecDuplicate(tao->solution, &tdm->workvec)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetRegularizer - Gets an DMTao which behaves as an regularizer to a Tao object

  Input Parameters:
. tao - Tao solver context

  Output Parameter:
. dm - DMTao context

  Level: advanced

.seealso: `TaoSetRegularizer()`
@*/
PetscErrorCode TaoGetRegularizer(Tao tao, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  if (tao->is_child_dm) {
    PetscCall(PetscObjectQuery((PetscObject)tao, "TaoGetParentDM", (PetscObject *)dm));
  } else {
    *dm = tao->reg;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoComputeObjective - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ dm - the `DM` context
- x  - input vector

  Output Parameter:
. f - Objective value at `x`

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoComputeGradient()`, `DMTaoComputeObjectiveAndGradient()`, `DMTaoSetObjective()`
@*/
PetscErrorCode DMTaoComputeObjective(DM dm, Vec x, PetscReal *f)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscCheckSameComm(dm, 1, x, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscLogEventBegin(DMTAO_Eval, tdm, x, NULL, NULL));
  if (tdm->usetaoroutines) {
    PetscCall(TaoComputeObjective(tdm->dm_subtao, x, f));
  } else {
    PetscCheck(tdm->ops->computeobjective || tdm->ops->computeobjectiveandgradient, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMTao does not have objective function set");
    if (tdm->ops->computeobjective) PetscCallBack("DMTao callback objective", (*tdm->ops->computeobjective)(dm, x, f, tdm->userctx_func));
    else {
      PetscCallBack("DMTao callback objective", (*tdm->ops->computeobjectiveandgradient)(dm, x, f, tdm->workvec, tdm->userctx_funcgrad));
    }
  }
  PetscCall(PetscLogEventEnd(DMTAO_Eval, dm, x, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ dm - the `DM` context
- x  - input vector

  Output Parameters:
+ f - Objective value at `x`
- g - Gradient vector at `x`

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoComputeGradient()`, `DMTaoSetObjective()`
@*/
PetscErrorCode DMTaoComputeObjectiveAndGradient(DM dm, Vec x, PetscReal *f, Vec g)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(dm, 1, x, 2);
  PetscCheckSameComm(dm, 1, g, 4);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscLogEventBegin(DMTAO_Eval, tdm, x, g, NULL));
  if (tdm->usetaoroutines) {
    PetscCall(TaoComputeObjectiveAndGradient(tdm->dm_subtao, x, f, g));
  } else {
    if (tdm->ops->computeobjectiveandgradient) PetscCallBack("DMTao callback objective/gradient", (*tdm->ops->computeobjectiveandgradient)(dm, x, f, g, tdm->userctx_funcgrad));
    else {
      PetscCallBack("DMTao callback objective", (*tdm->ops->computeobjective)(dm, x, f, tdm->userctx_func));
      PetscCallBack("DMTao callback gradient", (*tdm->ops->computegradient)(dm, x, g, tdm->userctx_grad));
    }
    PetscCall(PetscInfo(dm, "DMTao Function evaluation: %14.12e\n", (double)(*f)));
  }
  PetscCall(PetscLogEventEnd(DMTAO_Eval, tdm, x, g, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoComputeGradient - Computes the gradient of the objective function

  Collective

  Input Parameters:
+ dm - the `DM` context
- x  - input vector

  Output Parameter:
. g - gradient vector

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoComputeObjective()`, `DMTaoComputeObjectiveAndGradient()`, `DMTaoSetGradient()`
@*/
PetscErrorCode DMTaoComputeGradient(DM dm, Vec x, Vec g)
{
  PetscReal fdummy;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheckSameComm(dm, 1, x, 2);
  PetscCheckSameComm(dm, 1, g, 3);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscLogEventBegin(DMTAO_Eval, tdm, x, g, NULL));
  if (tdm->usetaoroutines) {
    PetscCall(TaoComputeGradient(tdm->dm_subtao, x, g));
  } else {
    if (tdm->ops->computegradient) PetscCallBack("DMTao callback gradient", (*tdm->ops->computegradient)(dm, x, g, tdm->userctx_grad));
    else PetscCallBack("DMTao callback gradient", (*tdm->ops->computeobjectiveandgradient)(dm, x, &fdummy, g, tdm->userctx_funcgrad));
  }
  PetscCall(PetscLogEventEnd(DMTAO_Eval, tdm, x, g, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoRegister - Adds a DMTao type to the registry

  Not Collective

  Input Parameters:
+ sname - name of a new user-defined dmtAO
- func  - routine to Create method context

  Your solver can be chosen with the procedural interface via
$    DMTaoSetType(pd, "my_dmtao")
  or at runtime via the option
$    -dmtao_type my_dmtao

  Level: developer

  Note:
  `DMTaoRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoRegister(const char sname[], PetscErrorCode (*func)(DM))
{
  PetscFunctionBegin;
  PetscCall(DMTaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&DMTaoList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoApplyProximalMap - Solves proximal algorithm, based on provided DMTao routines.
  dm0 will serve as primary objective, and dm1 will serve as regularizer.

  Collective

  Input Parameters:
+ dm0    - the `DM` primary  context
. dm1    - the `DM` regularizer context
. lambda - the scale of regularizer
. y      - the central vector of regularizer
. x      - the solution vector
- ctx    - pointer to context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoApplyProximalMap(DM dm0, DM dm1, PetscReal lambda, Vec y, Vec x, void *ctx)
{
  PetscInt low1, low2, high1, high2;
  DMTao    tdm0, tdm1;

  PetscFunctionBegin;
  /*TODO boilerplates */
  PetscValidHeaderSpecific(dm0, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm1, DM_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscCall(DMGetDMTao(dm0, &tdm0));
  PetscCall(DMGetDMTao(dm1, &tdm1));
  PetscTryTypeMethod(tdm0, applyproximalmap, tdm1, lambda, y, x, ctx);
  PetscCheckSameComm(dm0, 1, y, 4);
  PetscCheckSameTypeAndComm(y, 4, x, 5);
  PetscCall(VecGetOwnershipRange(y, &low1, &high1));
  PetscCall(VecGetOwnershipRange(x, &low2, &high2));
  PetscCheck(low1 == low2 && high1 == high2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible vector local lengths");

  /* TODO DMTAO_Eval vs. DMTAO_APPLY ? */
  PetscCall(PetscLogEventBegin(DMTAO_Eval, dm0, dm1, y, x));
  /* TODO do we want view for both, or just the primary objective? */
  PetscCall(DMTaoViewFromOptions(dm0, NULL, "-dm_tao_view"));
  PetscCall(DMTaoViewFromOptions(dm1, NULL, "-dm_tao_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMTaoGetParentDM(DMTao tdm, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdm, DMTAO_CLASSID, 1);
  *dm = tdm->originaldm;
  PetscFunctionReturn(PETSC_SUCCESS);
}
