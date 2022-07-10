#include <petsc/private/benchimpl.h>     /*I  "petscbench.h"   I*/
#include <petscviewer.h>

/*@
   PetscBenchDestroy - Destroys a benchmark object

   Collective on PetscBench

   Input Parameter:
.  ben - the benchmark object

   Level: developer

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchSetFromOptions()`, `PetscBenchCreate()`, `PetscBenchRun()`

@*/
PetscErrorCode  PetscBenchDestroy(PetscBench *ben)
{
  PetscFunctionBegin;
  if (!*ben) PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}

/*@
   PetscBenchSetFromOptions - Allows setting options from a benchmark object

   Collective on PetscBench

   Input Parameter:
.  ben - the benchmark object

   Level: developer

.seealso: `PetscBench`, `PetscBenchView()`, `PetscBenchDestroy()`, `PetscBenchCreate()`, `PetscBenchRun()`

@*/
PetscErrorCode  PetscBenchSetFromOptions(PetscBench ben)
{
  PetscFunctionBegin;
  PetscValidPointer(ben,1);
  PetscObjectOptionsBegin((PetscObject)ben);
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
   PetscBenchView - Views a benchmark

   Collective on PetscBench

   Input Parameters:
+  ben - the benchmark
-  viewer - location to view the values

   Level: developer

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchDestroy()`, `PetscBenchCreate()`, `PetscBenchRun()`
@*/
PetscErrorCode  PetscBenchView(PetscBench ben,PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidPointer(ben,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)ben,viewer));
  }
  PetscFunctionReturn(0);
}

/*@
   PetscBenchRun - runs the benchmark in a benchmark object

   Collective on PetscBench

   Input Parameters:
.  ben - the benchmark

   Level: developer

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchDestroy()`, `PetscBenchCreate()`, `PetscBenchRun()`
@*/
PetscErrorCode  PetscBenchRun(PetscBench ben)
{
  static PetscLogStage stage = 0;

  PetscFunctionBegin;
  if (!stage) PetscCall(PetscLogStageRegister("Micro-benchmark",&stage));
  PetscCall(PetscLogStagePush(stage));
  PetscCall((*ben->ops->run)(ben));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(0);
}

/*@C
    PetscBenchCreate - Create a benchmark

  Collective

  Level: Developer

  Input Parameters:
.  comm - communicator to share benchmark

  Output Parameter:
.   ben - the benchmark object

  Note:
  Unlike most other PETSc objects, this object does not have a set type method, rather this routine is called by a specific constructor such
  as `PetscBenchCreateVecStreams()`.

.seealso: `PetscBench`, `PetscBenchSetFromOptions()`, `PetscBenchDestroy()`, `PetscBenchDestroy()`, `PetscBenchRun()` `PetscBenchCreateVecStreams()`
@*/
PetscErrorCode PetscBenchCreate(MPI_Comm comm, PetscBench *ben)
{
  PetscFunctionBegin;
  PetscValidPointer(ben,2);

  PetscCall(PetscHeaderCreate(*ben,PETSC_BENCH_CLASSID,"PetscBench","PETSc Benchmark","PetscBench",comm,PetscBenchDestroy,PetscBenchView));
  PetscFunctionReturn(0);
}
