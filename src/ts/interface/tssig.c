#include <petsc/private/tsimpl.h> /*I "petscts.h"  I*/
#include <signal.h>

/*@
   TSSetConvergeOnSignal - make `TSSolve()` watch for specified signal and set `TS_CONVERGED_SIGNAL` if received

   Logically collective

   Input Parameters:
+  ts - the `TS` that will watch for signals
-  sig - the signal to watch for, such as SIGTERM or SIGUSR1, or `PETSC_DEFAULT`. Pass 0 to disable this feature.

   Level: intermediate

   Notes:
   Each call to `TSSolve()` will push the signal handler and pop it on completion. The handler will inform `TSSolve()`
   that a signal has been received so it can exit gracefully at the end of each step. If signals other than the
   specified signal are received, this function calls `PetscSignalHandlerDefault()`.

   Resource managers have different ways to send signals to jobs. Use `scancel --signal HUP $JOBID` to send specified
   signals with Slurm and `qsig -s HUP $JOBID` with PBS Pro. A list of signal names and numbers can be obtained with the
   command `kill -l`.

   Developer Notes:

   Open MPI uses SIGTERM to clean up child processes, therefore PETSc signal handlers do not attempt to catch SIGTERM
   when using Open MPI. PBS automatically sends SIGTERM to processes a short time in advance of hard killing it, thus
   it's desirable to catch SIGTERM with PBS.

.seealso: `PetscPushSignalHandler()`, `PetscSignalHandlerDefault()`, `TSSolve()`
@*/
PetscErrorCode TSSetConvergeOnSignal(TS ts, int sig)
{
  PetscFunctionBegin;
  if (sig == PETSC_DECIDE || sig == PETSC_DEFAULT) sig = SIGTERM;
  ts->signal.watch = sig;
  PetscFunctionReturn(0);
}
