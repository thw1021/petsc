#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscis.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscsection.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <petscdmsolveadaptor.h> /*I "petscdmsolveadaptor.h" I*/
#include <petscdmplex.h>
#include <petscksp.h>

#include <petsc/private/dmsolveadaptorimpl.h>
#include <petsc/private/pcimpl.h> // For setfromoptionscalled

PetscClassId DMSOLVEADAPTOR_CLASSID;

PetscFunctionList DMSolveAdaptorList              = NULL;
PetscBool         DMSolveAdaptorRegisterAllCalled = PETSC_FALSE;

/*@
  DMSolveAdaptorRegister - Adds a new solve adaptor component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine

  Example Usage:
.vb
  DMSolveAdaptorRegister("my_adaptor", MyAdaptorCreate);
.ve

  Then, your adaptor type can be chosen with the procedural interface via
.vb
  DMSolveAdaptorCreate(MPI_Comm, DMSolveAdaptor *);
  DMSolveAdaptorSetType(DMSolveAdaptor, "my_adaptor");
.ve
  or at runtime via the option
.vb
  -solve_adaptor_type my_adaptor
.ve

  Level: advanced

  Note:
  `DMSolveAdaptorRegister()` may be called multiple times to add several user-defined adaptors

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMSolveAdaptor`, `DMSolveAdaptorRegisterAll()`, `DMSolveAdaptorRegisterDestroy()`
@*/
PetscErrorCode DMSolveAdaptorRegister(const char name[], PetscErrorCode (*create_func)(DMSolveAdaptor))
{
  PetscFunctionBegin;
  PetscCall(DMInitializePackage());
  PetscCall(PetscFunctionListAdd(&DMSolveAdaptorList, name, create_func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMSolveAdaptorCreate_Residual(DMSolveAdaptor);

/*@
  DMSolveAdaptorRegisterAll - Registers all of the solve adaptor components in the `DM` package.

  Not Collective

  Level: advanced

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMSolveAdaptorType`, `DMRegisterAll()`, `DMSolveAdaptorRegisterDestroy()`
@*/
PetscErrorCode DMSolveAdaptorRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMSolveAdaptorRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DMSolveAdaptorRegisterAllCalled = PETSC_TRUE;

  PetscCall(DMSolveAdaptorRegister(DMSOLVEADAPTORRESIDUAL, DMSolveAdaptorCreate_Residual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorRegisterDestroy - This function destroys the registered `DMSolveAdaptorType`. It is called from `PetscFinalize()`.

  Not collective

  Level: developer

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMSolveAdaptorRegisterAll()`, `DMSolveAdaptorType`, `PetscFinalize()`
@*/
PetscErrorCode DMSolveAdaptorRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMSolveAdaptorList));
  DMSolveAdaptorRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorCreate - Create a `DMSolveAdaptor` object. Its purpose is to modify a PETSc solver using information from the solve and the associated `DM`.

  Collective

  Input Parameter:
. comm - The communicator for the `DMSolveAdaptor` object

  Output Parameter:
. adaptor - The `DMSolveAdaptor` object

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptor`, `DMSolveAdaptorDestroy()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorCreate(MPI_Comm comm, DMSolveAdaptor *adaptor)
{
  VecTaggerBox wrongBox;

  PetscFunctionBegin;
  PetscAssertPointer(adaptor, 2);
  PetscCall(PetscSysInitializePackage());

  PetscCall(PetscHeaderCreate(*adaptor, DMSOLVEADAPTOR_CLASSID, "DMSolveAdaptor", "DM Solve Adaptor", "DMSolveAdaptor", comm, DMSolveAdaptorDestroy, DMSolveAdaptorView));
  (*adaptor)->numSeq = 1;

  wrongBox.min = 0.2;
  wrongBox.max = 1.0;
  PetscCall(VecTaggerCreate(PetscObjectComm((PetscObject)*adaptor), &(*adaptor)->wrongTag));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*adaptor)->wrongTag, "wrong_"));
  PetscCall(VecTaggerSetType((*adaptor)->wrongTag, VECTAGGERRELATIVE));
  PetscCall(VecTaggerRelativeSetBox((*adaptor)->wrongTag, &wrongBox));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorDestroy - Destroys a `DMSolveAdaptor` object

  Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptor`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorDestroy(DMSolveAdaptor *adaptor)
{
  PetscFunctionBegin;
  if (!*adaptor) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  if (--((PetscObject)*adaptor)->refct > 0) {
    *adaptor = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecTaggerDestroy(&(*adaptor)->wrongTag));
  PetscCall(PetscHeaderDestroy(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetType - Sets the particular implementation for a adaptor.

  Collective

  Input Parameters:
+ adaptor - The `DMSolveAdaptor`
- method  - The name of the adaptor type

  Options Database Key:
. -solve_adaptor_type type - Sets the solve adaptor type; see `DMSolveAdaptorType`

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMSolveAdaptor`, `DMSolveAdaptorType`, `DMSolveAdaptorGetType()`, `DMSolveAdaptorCreate()`
@*/
PetscErrorCode DMSolveAdaptorSetType(DMSolveAdaptor adaptor, DMSolveAdaptorType method)
{
  PetscErrorCode (*r)(DMSolveAdaptor);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)adaptor, method, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMSolveAdaptorRegisterAll());
  PetscCall(PetscFunctionListFind(DMSolveAdaptorList, method, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)adaptor), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown DMSolveAdaptor type: %s", method);

  PetscTryTypeMethod(adaptor, destroy);
  PetscCall(PetscMemzero(adaptor->ops, sizeof(*adaptor->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)adaptor, method));
  PetscCall((*r)(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorGetType - Gets the type name (as a string) from the adaptor.

  Not Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor`

  Output Parameter:
. type - The `DMSolveAdaptorType` name

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMSolveAdaptor`, `DMSolveAdaptorType`, `DMSolveAdaptorSetType()`, `DMSolveAdaptorCreate()`
@*/
PetscErrorCode DMSolveAdaptorGetType(DMSolveAdaptor adaptor, DMSolveAdaptorType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(DMSolveAdaptorRegisterAll());
  *type = ((PetscObject)adaptor)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetOptionsPrefix - Sets the prefix used for searching for all `DMSolveAdaptor` options in the database.

  Logically Collective

  Input Parameters:
+ adaptor - the `DMSolveAdaptor`
- prefix  - the prefix to prepend to all option names

  Level: advanced

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](ch_snes), `DMSolveAdaptor`, `SNESSetOptionsPrefix()`, `DMSolveAdaptorSetFromOptions()`
@*/
PetscErrorCode DMSolveAdaptorSetOptionsPrefix(DMSolveAdaptor adaptor, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adaptor, prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)adaptor->wrongTag, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)adaptor->wrongTag, "wrong_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetFromOptions - Sets properties of a `DMSolveAdaptor` object from values in the options database

  Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Options Database Keys:
. -solve_adaptor_sequence_num num - Number of adaptations to generate an optimal solver

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptor`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorSetFromOptions(DMSolveAdaptor adaptor)
{
  char        typeName[PETSC_MAX_PATH_LEN];
  const char *defName = DMSOLVEADAPTORRESIDUAL;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin((PetscObject)adaptor);
  PetscCall(PetscOptionsFList("-solve_adaptor_type", "DMSolveAdaptor", "DMSolveAdaptorSetType", DMSolveAdaptorList, defName, typeName, 1024, &flg));
  if (flg) PetscCall(DMSolveAdaptorSetType(adaptor, typeName));
  else if (!((PetscObject)adaptor)->type_name) PetscCall(DMSolveAdaptorSetType(adaptor, defName));
  PetscCall(PetscOptionsInt("-solve_adaptor_sequence_num", "Number of adaptations to generate an optimal grid", "DMAdaptorSetSequenceLength", adaptor->numSeq, &adaptor->numSeq, NULL));
  PetscOptionsEnd();
  PetscCall(VecTaggerSetFromOptions(adaptor->wrongTag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorView - Views a `DMSolveAdaptor` object

  Collective

  Input Parameters:
+ adaptor - The `DMSolveAdaptor` object
- viewer  - The `PetscViewer` object

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptor`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorView(DMSolveAdaptor adaptor, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)adaptor, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "DM Adaptor\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  sequence length: %" PetscInt_FMT "\n", adaptor->numSeq));
  PetscCall(VecTaggerView(adaptor->wrongTag, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorGetLinearSolver - Gets the linear solver to be optimized

  Not Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Output Parameter:
. ksp - The solver

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptor`, `DMSolveAdaptorSetLinearSolver()`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorGetLinearSolver(DMSolveAdaptor adaptor, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscAssertPointer(ksp, 2);
  *ksp = adaptor->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetLinearSolver - Sets the linear solver to be optimized

  Not Collective

  Input Parameters:
+ adaptor - The `DMSolveAdaptor` object
- ksp     - The solver, this MUST have an attached `DM`

  Level: intermediate

.seealso: [](ch_dmbase), `DMSolveAdaptor`, `DMSolveAdaptorGetLinearSolver()`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorSetLinearSolver(DMSolveAdaptor adaptor, KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  adaptor->ksp = ksp;
  PetscCall(KSPGetDM(adaptor->ksp, &adaptor->dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorGetNonlinearSolver - Gets the nonlinear solver to be optimized

  Not Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Output Parameter:
. snes - The solver

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptor`, `DMSolveAdaptorSetNonlinearSolver()`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorGetNonlinearSolver(DMSolveAdaptor adaptor, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscAssertPointer(snes, 2);
  *snes = adaptor->snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetNonlinearSolver - Sets the nonlinear solver to be optimized

  Not Collective

  Input Parameters:
+ adaptor - The `DMSolveAdaptor` object
- snes    - The solver, this MUST have an attached `DM`

  Level: intermediate

.seealso: [](ch_dmbase), `DMSolveAdaptor`, `DMSolveAdaptorGetNonlinearSolver()`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorSetNonlinearSolver(DMSolveAdaptor adaptor, SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  //PetscValidHeaderSpecific(snes, SNES_CLASSID, 2);
  adaptor->snes = snes;
  //PetscCall(SNESGetDM(adaptor->snes, &adaptor->dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorGetSequenceLength - Gets the number of sequential adaptations used by an adapter

  Not Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Output Parameter:
. num - The number of adaptations

  Level: intermediate

.seealso: [](ch_dmbase), `DMSolveAdaptor`, `DMSolveAdaptorSetSequenceLength()`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorGetSequenceLength(DMSolveAdaptor adaptor, PetscInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscAssertPointer(num, 2);
  *num = adaptor->numSeq;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetSequenceLength - Sets the number of sequential adaptations

  Not Collective

  Input Parameters:
+ adaptor - The `DMSolveAdaptor` object
- num     - The number of adaptations

  Level: intermediate

.seealso: [](ch_dmbase), `DMSolveAdaptorGetSequenceLength()`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorSetSequenceLength(DMSolveAdaptor adaptor, PetscInt num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  adaptor->numSeq = num;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorSetUp - After the solver is specified, creates data structures for controlling adaptivity

  Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Level: beginner

.seealso: [](ch_dmbase), `DMSolveAdaptor`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorAdapt()`
@*/
PetscErrorCode DMSolveAdaptorSetUp(DMSolveAdaptor adaptor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscCall(VecTaggerSetUp(adaptor->wrongTag));
  PetscTryTypeMethod(adaptor, setup);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexCreateDofHalo - Mark all mesh points in a halo around the points associated with the given global dofs.

  Input Parameters:
+ dm    - The `DM`
- dofIS - An `IS` holding global dofs

  Output Parameter:
. label - A `DMLabel` marking mesh points in the halo of points associated with the global dofs

  Level: developer

  Note:
  Each contiguous halo is marked with a sequence number

.seealso: [](ch_unstructured), `DMPlex`, `DMSolveAdaptor`
 */
static PetscErrorCode DMPlexCreateDofHalo(DM dm, IS dofIS, DMLabel label)
{
  PetscSection    gs;
  const PetscInt *dofs;
  PetscInt       *adj = NULL;
  PetscInt        pStart, pEnd, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMPLEX);
  PetscValidHeaderSpecific(dofIS, IS_CLASSID, 2);
  PetscValidHeaderSpecific(label, DMLABEL_CLASSID, 3);
  PetscCall(DMGetGlobalSection(dm, &gs));
  PetscCall(PetscSectionGetChart(gs, &pStart, &pEnd));
  PetscCall(ISGetLocalSize(dofIS, &n));
  PetscCall(ISGetIndices(dofIS, &dofs));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt ind     = dofs[i];
    PetscInt       adjSize = PETSC_DETERMINE;
    PetscInt       p, val;

    // TODO Make a function for dof to point
    //   Fast path using Find if no permutation
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(gs, p, &dof));
      PetscCall(PetscSectionGetOffset(gs, p, &off));
      if (ind >= off && ind < off + dof) break;
    }
    PetscCheck(p < pEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh point could not be located for dof %" PetscInt_FMT, ind);
    PetscCall(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
    // Check for overlap with other halos
    PetscCall(DMLabelGetValue(label, p, &val));
    for (PetscInt a = 0; a < adjSize; ++a) {
      if (val >= 0) break;
      PetscCall(DMLabelGetValue(label, adj[a], &val));
    }
    // If a point is already labeled, use that label instead
    val = val < 0 ? i : val;
    PetscCall(DMLabelSetValue(label, p, val));
    for (PetscInt a = 0; a < adjSize; ++a) {
      PetscInt oval;

      PetscCall(DMLabelGetValue(label, adj[a], &oval));
      PetscCall(DMLabelSetValue(label, adj[a], val));
      // Encode value to convert patch after iteration
      if (oval >= 0) adj[a] = -(oval + 1);
    }
    for (PetscInt a = 0; a < adjSize; ++a) {
      if (adj[a] < 0) {
        const PetscInt  oval = -(adj[a] + 1);
        IS              pointIS;
        const PetscInt *points;
        PetscInt        n;

        PetscCall(DMLabelGetStratumIS(label, oval, &pointIS));
        PetscCall(ISGetLocalSize(pointIS, &n));
        PetscCall(ISGetIndices(pointIS, &points));
        for (PetscInt op = 0; op < n; ++op) {
          const PetscInt point = points[op];

          PetscCall(DMLabelClearValue(label, point, oval));
          PetscCall(DMLabelSetValue(label, point, val));
        }
        PetscCall(ISRestoreIndices(pointIS, &points));
        PetscCall(ISDestroy(&pointIS));
      }
    }
  }
  PetscCall(ISRestoreIndices(dofIS, &dofs));
  PetscCall(PetscFree(adj));
  // TODO Use LabelPropagation to mark parallel halo overlap
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateRestrictionOperator_Private(DM dm, Vec residual, DMLabel label, Mat *R)
{
  PetscSection    gs;
  MatType         matType;
  IS              valueIS;
  const PetscInt *values;
  PetscInt       *dnnz, *onnz, *dnnzu, *onnzu;
  PetscInt        Nv, m = 0, n, moff = 0;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(residual, &n));
  PetscCall(DMGetGlobalSection(dm, &gs));
  // Count local dofs
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  for (PetscInt v = 0; v < Nv; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        Np;

    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &Np));
    PetscCall(ISGetIndices(pointIS, &points));
    for (PetscInt p = 0; p < Np; ++p) {
      PetscInt dof, cdof;

      PetscCall(PetscSectionGetDof(gs, points[p], &dof));
      PetscCall(PetscSectionGetConstraintDof(gs, points[p], &cdof));
      m += dof - cdof;
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  // Create operator
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), R));
  PetscCall(MatSetSizes(*R, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(DMGetMatType(dm, &matType));
  PetscCall(MatSetType(*R, matType));
  PetscCall(PetscMalloc4(m, &dnnz, m, &onnz, m, &dnnzu, m, &onnzu));
  for (PetscInt i = 0; i < m; ++i) dnnz[i] = onnz[i] = dnnzu[i] = onnzu[i] = 1;
  PetscCall(MatXAIJSetPreallocation(*R, 1, dnnz, onnz, dnnzu, onnzu));
  PetscCall(PetscFree4(dnnz, onnz, dnnzu, onnzu));
  PetscCall(MatSetUp(*R));
  // R is a boolean matrix, set the values
  for (PetscInt v = 0; v < Nv; ++v) {
    IS              pointIS;
    const PetscInt *points;
    PetscInt        Np;

    PetscCall(DMLabelGetStratumIS(label, values[v], &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &Np));
    PetscCall(ISGetIndices(pointIS, &points));
    for (PetscInt p = 0; p < Np; ++p) {
      const PetscInt *cind;
      PetscInt        dof, cdof, off, i = 0, e = 0;

      PetscCall(PetscSectionGetDof(gs, points[p], &dof));
      PetscCall(PetscSectionGetConstraintDof(gs, points[p], &cdof));
      PetscCall(PetscSectionGetConstraintIndices(gs, points[p], &cind));
      PetscCall(PetscSectionGetOffset(gs, points[p], &off));
      for (PetscInt d = 0; d < dof; ++d) {
        if (i < cdof && d == cind[i]) {
          ++i;
          continue;
        }
        PetscCall(MatSetValue(*R, moff++, off + e++, 1., INSERT_VALUES));
      }
      PetscCheck(i == cdof, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Found constraints %" PetscInt_FMT " != %" PetscInt_FMT " total constraints", i, cdof);
      PetscCheck(i + e == dof, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Constraints + dofs %" PetscInt_FMT " != %" PetscInt_FMT " total dofs", i + e, dof);
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCheck(moff == m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local dofs %" PetscInt_FMT " != %" PetscInt_FMT " local restriction operator size", moff, m);
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscCall(MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSolveAdaptorAdapt_Residual_Linear(DMSolveAdaptor adaptor)
{
  DM       dm = adaptor->dm;
  DMLabel  subsolveLabel;
  KSP      ksp;
  PC       origpc, comppc, gpc;
  Mat      R, mat, pmat;
  Vec      residual, work;
  IS       wrongIS;
  PetscInt nWrong, its;

  PetscFunctionBegin;
  PetscCall(DMSolveAdaptorGetLinearSolver(adaptor, &ksp));
  PetscCall(DMGetGlobalVector(dm, &residual));
  PetscCall(DMGetGlobalVector(dm, &work));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (!its) {
    Vec rhs;

    PetscCall(KSPGetRhs(ksp, &rhs));
    PetscCall(VecCopy(rhs, residual));
  } else {
    PetscCall(KSPBuildResidual(ksp, work, residual, &residual));
  }
  // We want to select based on residual magnitude
  PetscCall(VecAbs(residual));
  PetscCall(VecViewFromOptions(residual, NULL, "-residual_view"));
  PetscCall(DMRestoreGlobalVector(dm, &work));
  // Do we want to have separate boes for each field?
  PetscCall(VecTaggerComputeIS(adaptor->wrongTag, residual, &wrongIS, NULL));
  PetscCall(ISViewFromOptions(wrongIS, (PetscObject)adaptor->wrongTag, "-is_view"));
  PetscCall(ISGetSize(wrongIS, &nWrong));
  PetscCall(PetscInfo(dm, "DMSolveAdaptor: num large residual entries %" PetscInt_FMT "\n", nWrong));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "subsolve", &subsolveLabel));
  PetscCall(DMPlexCreateDofHalo(dm, wrongIS, subsolveLabel));
  PetscCall(ISDestroy(&wrongIS));
  PetscCall(DMLabelViewFromOptions(subsolveLabel, NULL, "-subsolve_view"));
  PetscCall(CreateRestrictionOperator_Private(dm, residual, subsolveLabel, &R));
  PetscCall(MatViewFromOptions(R, NULL, "-restriction_view"));
  PetscCall(DMRestoreGlobalVector(dm, &residual));
  PetscCall(DMLabelDestroy(&subsolveLabel));

  // TODO Add a patch to make a PCPatch instead
  PetscCall(KSPGetPC(ksp, &origpc));
  PetscCall(PCGetOperators(origpc, &mat, &pmat));

  PetscCall(PCCreate(PetscObjectComm((PetscObject)adaptor), &gpc));
  PetscCall(PCSetType(gpc, PCGALERKIN));
  PetscCall(PCGalerkinSetRestriction(gpc, R));
  PetscCall(PCSetOperators(gpc, mat, pmat));
  PetscCall(MatDestroy(&R));

  PetscCall(PCCreate(PetscObjectComm((PetscObject)adaptor), &comppc));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)comppc, "comp_"));
  PetscCall(PCSetType(comppc, PCCOMPOSITE));
  PetscCall(PCCompositeSetType(comppc, PC_COMPOSITE_MULTIPLICATIVE));
  PetscCall(PCCompositeAddPC(comppc, gpc));
  PetscCall(PCCompositeAddPC(comppc, origpc));
  PetscCall(PCSetOperators(comppc, mat, pmat));
  if (origpc->setfromoptionscalled) PetscCall(PCSetFromOptions(comppc));
  PetscCall(PCSetUp(comppc));

  PetscCall(KSPSetPC(ksp, comppc));
  PetscCall(PCDestroy(&gpc));
  PetscCall(PCDestroy(&comppc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSolveAdaptorAdapt_Residual(DMSolveAdaptor adaptor)
{
  PetscFunctionBegin;
  if (adaptor->ksp) {
    PetscCall(DMSolveAdaptorAdapt_Residual_Linear(adaptor));
  } else if (adaptor->snes) {
    SETERRQ(PetscObjectComm((PetscObject)adaptor), PETSC_ERR_SUP, "Nonlinear solver not currently supported");
  } else SETERRQ(PetscObjectComm((PetscObject)adaptor), PETSC_ERR_ARG_WRONG, "No linear or nonlinear solver was specified for optimization");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSolveAdaptorAdapt - Creates a new `DM` that is adapted to the problem

  Not Collective

  Input Parameter:
. adaptor - The `DMSolveAdaptor` object

  Level: intermediate

  Note:
  The solver held internally is modified by this routine.

.seealso: [](ch_dmbase), `DMSolveAdaptor`, `DMAdaptationStrategy`, `DMSolveAdaptorSetNonlinearSolver()`, `DMSolveAdaptorCreate()`
@*/
PetscErrorCode DMSolveAdaptorAdapt(DMSolveAdaptor adaptor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  PetscUseTypeMethod(adaptor, adapt);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSolveAdaptorInitialize_Residual(DMSolveAdaptor adaptor)
{
  PetscFunctionBegin;
  adaptor->ops->adapt = DMSolveAdaptorAdapt_Residual;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMSolveAdaptorCreate_Residual(DMSolveAdaptor adaptor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DMSOLVEADAPTOR_CLASSID, 1);
  adaptor->data = NULL;

  PetscCall(DMSolveAdaptorInitialize_Residual(adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}
