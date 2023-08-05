#include <petsc/private/petscimpl.h>
#include <petsc/private/bmimpl.h> /*I  "petscbm.h"   I*/
#include <petscviewer.h>

PetscClassId             BM_CLASSID;
static PetscBool         PetscBMPackageInitialized = PETSC_FALSE;
static PetscFunctionList PetscBMList               = NULL;

// PetscClangLinter pragma disable: -fdoc-internal-linkage
/*@C
  PetscBMFinalizePackage - This function destroys everything in the `PetscBM` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscFinalize()`, `PetscBMInitializePackage()`, `PetscBMCreate()`, `PetscBM`, `PetscBMType`
@*/
static PetscErrorCode PetscBMFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscBMList));
  PetscBMPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMInitializePackage - This function initializes everything in the `PetscBM` package.

  Level: developer

.seealso: `PetscInitialize()`, `PetscBMCreate()`, `PetscBM`, `PetscBMType`
@*/
PetscErrorCode PetscBMInitializePackage(void)
{
  PetscFunctionBegin;
  if (PetscBMPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscBMPackageInitialized = PETSC_TRUE;
  PetscCall(PetscClassIdRegister("PetscBM", &BM_CLASSID));
  PetscCall(PetscRegisterFinalize(PetscBMFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMRegister -  Adds a benchmark test, `PetscBMType`, to the `PetscBM` package

  Not Collective

  Input Parameters:
+ sname    - name of a new benchmark
- function - routine to create benchmark

  Level: advanced

  Note:
  `PetscBMRegister()` may be called multiple times

.seealso: `PetscBMInitializePackage()`, `PetscBMCreate()`, `PetscBM`, `PetscBMType`, `PetscBMSetType()`, `PetscBMGetType()`
@*/
PetscErrorCode PetscBMRegister(const char sname[], PetscErrorCode (*function)(PetscBM))
{
  PetscFunctionBegin;
  PetscCall(PetscBMInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscBMList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMReset - removes all the intermediate data structures in a `PetscBM`

  Collective

  Input Parameter:
. bm - the `PetscBM`

  Level: advanced

.seealso: `PetscBM`, `PetscBMView()`, `PetscBMSetFromOptions()`, `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`
@*/
PetscErrorCode PetscBMReset(PetscBM bm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscCall(PetscLogHandlerDestroy(&bm->lhdlr)); // Temporarily here until PetscLogHanderReset() exists
  PetscTryTypeMethod(bm, reset);
  bm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMDestroy - Destroys a `PetscBM`

  Collective

  Input Parameter:
. bm - the `PetscBM`

  Level: advanced

.seealso: `PetscBM`, `PetscBMView()`, `PetscBMSetFromOptions()`, `PetscBMCreate()`
@*/
PetscErrorCode PetscBMDestroy(PetscBM *bm)
{
  PetscFunctionBegin;
  PetscAssertPointer(bm, 1);
  if (!*bm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*bm), BM_CLASSID, 1);
  if (--((PetscObject)(*bm))->refct > 0) {
    *bm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscBMReset(*bm));
  PetscTryTypeMethod(*bm, destroy);
  PetscCall(PetscHeaderDestroy(bm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBMSetUp - sets up the `PetscBM`

  Collective

  Input Parameter:
. bm - the `PetscBM`

  Level: advanced

.seealso: `PetscBM`, `PetscBMView()`, `PetscBMSetFromOptions()`, `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetType()`,
          `PetscBMRun()`, `PetscBMSetSize()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMSetUp(PetscBM bm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  if (bm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerCreate(PETSC_COMM_WORLD, &bm->lhdlr)); // Temporarily here until PetscLogHandlerReset() exists
  PetscCall(PetscLogHandlerSetType(bm->lhdlr, PETSCLOGHANDLERDEFAULT));
  PetscTryTypeMethod(bm, setup);
  bm->setupcalled = PETSC_TRUE;
  PetscTryTypeMethod(bm, run);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBMRun - runs the `PetscBM`

  Collective

  Input Parameter:
. bm - the `PetscBM`

  Level: advanced

.seealso: `PetscBM`, `PetscBMView()`, `PetscBMSetFromOptions()`, `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetSize()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMRun(PetscBM bm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  if (!bm->setupcalled) PetscCall(PetscBMSetUp(bm));
  PetscCall(PetscLogHandlerStart(bm->lhdlr));
  PetscTryTypeMethod(bm, run);
  PetscCall(PetscLogHandlerStop(bm->lhdlr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscBMSetFromOptions - Sets options to a `PetscBM` using the options database

  Collective

  Input Parameter:
. bm - the `PetscBM`

  Level: advanced

.seealso: `PetscBM`, `PetscBMView()`, `PetscBMRun()`, `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetSize()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMSetFromOptions(PetscBM bm)
{
  char      type[256];
  PetscBool flg;
  PetscInt  m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)bm);
  PetscCall(PetscOptionsFList("-petscbm_type", "PetscBM", "PetscBMSetType", PetscBMList, ((PetscObject)bm)->type_name, type, sizeof(type), &flg));
  if (flg) { PetscCall(PetscBMSetType(bm, type)); }
  PetscCheck(((PetscObject)bm)->type_name, PetscObjectComm((PetscObject)bm), PETSC_ERR_ARG_WRONGSTATE, "No PetscBMType provided for PetscBM");
  PetscCall(PetscOptionsInt("-bm_size", "Size of benchmark", "PetscBMSetSize", bm->size, &m, &flg));
  if (flg) PetscCall(PetscBMSetSize(bm, m));
  PetscTryTypeMethod(bm, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMView - Views a PETSc benchmark `PetscBM`

  Collective

  Input Parameters:
+ bm     - the `PetscBM`
- viewer - location to view `bm`

  Level: advanced

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetSize()`, `PetscBMGetSize()`, `PetscBMViewFromOptions()`
@*/
PetscErrorCode PetscBMView(PetscBM bm, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscTryTypeMethod(bm, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMViewFromOptions - Processes command line options to determine if/how a `PetscBM` is to be viewed.

  Collective

  Input Parameters:
+ bm         - the object
. bobj       - optional other object that provides prefix (if `NULL` then the prefix in `bm` is used)
- optionname - option to activate viewing

  Level: advanced

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMCreate()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetSize()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMViewFromOptions(PetscBM bm, PetscObject bobj, const char optionname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)bm, bobj, optionname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMCreate - Create a PETSc benchmark `PetscBM` object

  Collective

  Input Parameters:
. comm - communicator to share the `PetscBM`

  Output Parameter:
. bm - the `PetscBM`

  Level: advanced

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMViewFromOptions()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetSize()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMCreate(MPI_Comm comm, PetscBM *bm)
{
  PetscFunctionBegin;
  PetscAssertPointer(bm, 2);
  *bm = NULL;
  PetscCall(PetscBMInitializePackage());
  PetscCall(PetscHeaderCreate(*bm, BM_CLASSID, "BM", "PetscBM", "BM", comm, PetscBMDestroy, PetscBMView));
  (*bm)->size = PETSC_DECIDE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMSetOptionsPrefix - Sets the prefix used for searching for all `PetscBM` items in the options database.

  Logically Collective

  Input Parameters:
+ bm  - the `PetscBM`
- pre - the prefix to prepend all `PetscBM` option names

  Level: advanced

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMViewFromOptions()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetSize()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMSetOptionsPrefix(PetscBM bm, const char pre[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)bm, pre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMSetSize - Sets the size of the `PetscBM` benchmark to run

  Logically Collective

  Input Parameters:
+ bm - the `PetscBM`
- n  - the size

  Level: advanced

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMViewFromOptions()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetOptionsPrefix()`, `PetscBMGetSize()`
@*/
PetscErrorCode PetscBMSetSize(PetscBM bm, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  if (bm->size > 0 && bm->size != n && bm->setupcalled) {
    PetscCall(PetscBMReset(bm));
    bm->setupcalled = PETSC_FALSE;
  }
  PetscCheck(n > 0, PetscObjectComm((PetscObject)bm), PETSC_ERR_ARG_OUTOFRANGE, "Illegal value of n. Must be > 0");
  bm->size = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMGetSize - Gets the size of the `PetscBM` benchmark to run

  Logically Collective

  Input Parameter:
. bm - the `PetscBM`

  Output Parameter:
. n - the size

  Level: advanced

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMViewFromOptions()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMSetType()`,
          `PetscBMSetOptionsPrefix()`, `PetscBMSetSize()`
@*/
PetscErrorCode PetscBMGetSize(PetscBM bm, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscAssertPointer(n, 2);
  *n = bm->size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMSetType - set the type of `PetscBM` benchmark to run

  Collective

  Input Parameters:
+ bm   - the `PetscBM`
- type - a known method

  Options Database Key:
. -bm_type <type> - Sets `PetscBM` type

  Level: advanced

  Developer Note:
  `PetscBMRegister()` is used to add new benchmark types

.seealso: `PetscBM`, `PetscBMSetFromOptions()`, `PetscBMRun()`, `PetscBMViewFromOptions()`, `PetscBMDestroy()`, `PetscBMSetUp()`, `PetscBMGetSize()`,
          `PetscBMSetOptionsPrefix()`, `PetscBMSetSize()`, `PetscBMGetType()`, `PetscBMCreate()`
@*/
PetscErrorCode PetscBMSetType(PetscBM bm, PetscBMType type)
{
  PetscBool match;
  PetscErrorCode (*r)(PetscBM);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)bm, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscBMList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)bm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscBM type %s", type);
  /* Destroy the previous private PC context */
  PetscTryTypeMethod(bm, destroy);
  bm->ops->destroy = NULL;
  bm->data         = NULL;

  PetscCall(PetscFunctionListDestroy(&((PetscObject)bm)->qlist));
  /* Reinitialize function pointers in PCOps structure */
  PetscCall(PetscMemzero(bm->ops, sizeof(struct _PetscBMOps)));

  PetscCall(PetscObjectChangeTypeName((PetscObject)bm, type));
  PetscCall((*r)(bm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscBMGetType - Gets the `PetscBMType` (as a string) from the `PetscBM`
  context.

  Not Collective

  Input Parameter:
. bm - the preconditioner context

  Output Parameter:
. type - name of preconditioner method

  Level: intermediate

.seealso: `PetscBM`, `PetscBMType`, `PetscBMSetType()`, `PetscBMCreate()`
@*/
PetscErrorCode PetscBMGetType(PetscBM bm, PetscBMType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(bm, BM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)bm)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}
