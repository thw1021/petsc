#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/

/*@C
  PetscLogDefaultBegin - Turns on logging of objects and events using the default logging functions `PetscLogEventBeginDefault()` and `PetscLogEventEndDefault()`. This logs flop
  rates and object creation and should not slow programs down too much.
  This routine may be called more than once.

  Logically Collective over `PETSC_COMM_WORLD`

  Options Database Key:
. -log_view [viewertype:filename:viewerformat] - Prints summary of flop and timing information to the
                  screen (for code configured with --with-log=1 (which is the default))

  Usage:
.vb
      PetscInitialize(...);
      PetscLogDefaultBegin();
       ... code ...
      PetscLogView(viewer); or PetscLogDump();
      PetscFinalize();
.ve

  Level: advanced

  Note:
  `PetscLogView()` or `PetscLogDump()` actually cause the printing of
  the logging information.

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogTraceBegin()`
@*/
PetscErrorCode PetscLogDefaultBegin(void)
{
  PetscFunctionBegin;
  PetscCall(PetscLogSet(PetscLogEventBeginDefault, PetscLogEventEndDefault));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogAllBegin - Turns on extensive logging of objects and events. Logs
  all events. This creates large log files and slows the program down.

  Logically Collective on `PETSC_COMM_WORLD`

  Options Database Key:
. -log_all - Prints extensive log information

  Usage:
.vb
     PetscInitialize(...);
     PetscLogAllBegin();
     ... code ...
     PetscLogDump(filename);
     PetscFinalize();
.ve

  Level: advanced

  Note:
  A related routine is `PetscLogDefaultBegin()` (with the options key -log_view), which is
  intended for production runs since it logs only flop rates and object
  creation (and shouldn't significantly slow the programs).

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogDefaultBegin()`, `PetscLogTraceBegin()`
@*/
PetscErrorCode PetscLogAllBegin(void)
{
  PetscFunctionBegin;
  PetscCall(PetscLogSet(PetscLogEventBeginComplete, PetscLogEventEndComplete));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscLogTraceBegin - Activates trace logging.  Every time a PETSc event
  begins or ends, the event name is printed.

  Logically Collective on `PETSC_COMM_WORLD`

  Input Parameter:
. file - The file to print trace in (e.g. stdout)

  Options Database Key:
. -log_trace [filename] - Activates `PetscLogTraceBegin()`

  Level: intermediate

  Notes:
  `PetscLogTraceBegin()` prints the processor number, the execution time (sec),
  then "Event begin:" or "Event end:" followed by the event name.

  `PetscLogTraceBegin()` allows tracing of all PETSc calls, which is useful
  to determine where a program is hanging without running in the
  debugger.  Can be used in conjunction with the -info option.

.seealso: [](ch_profiling), `PetscLogDump()`, `PetscLogAllBegin()`, `PetscLogView()`, `PetscLogDefaultBegin()`
@*/
PetscErrorCode PetscLogTraceBegin(FILE *file)
{
  PetscFunctionBegin;
  petsc_tracefile = file;

  PetscCall(PetscLogSet(PetscLogEventBeginTrace, PetscLogEventEndTrace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

