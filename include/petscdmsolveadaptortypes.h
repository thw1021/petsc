#pragma once

/* SUBMANSEC = DM */

/*S
  DMSolveAdaptor - An object that modifies a `KSP`, `SNES`, `TS`, or `TAO` using information from the associated `DM` and the solve itself.

  Level: developer

.seealso: [](ch_dmbase), `DM`, `DMSolveAdaptorCreate()`, `DMSolveAdaptorSetLinearSolver()`, `DMSolveAdaptorSetNonlinearSolver()`, `DMSolveAdaptorSetTimestepper()`, `DMSolveAdaptorSetOptimizer()`,
          `DMSolveAdaptorSetFromOptions()`, `DMSolveAdaptorSetUp()`, `DMSolveAdaptorAdapt()`, `DMSolveAdaptorDestroy()`, `DMAdaptorCreate()`
S*/
typedef struct _p_DMSolveAdaptor *DMSolveAdaptor;
