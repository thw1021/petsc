
#include <petsc/private/dmswarmimpl.h> /*I "petscdmswarmx.h" I*/
#include <petsc/private/tsimpl.h>      /*I "petscts.h" I*/

/*@
  DMSwarmTSCreateSolution - This function creates phase space solution for the `DMSwarm` attached to the `TS`

  Collective on ts

  Input Parameter:
. ts - The `TS` object

  Level: developer

.seealso: `DMSwarmTSRedistribute()`, `DMSwarmMigrate()`, `DMSWARM`, 'TS'
@*/
PetscErrorCode DMSwarmTSCreateSolution(TS ts)
{
  DM       sw;
  Vec      u;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, dim));
  PetscCall(VecSetSizes(u, 2 * Np * dim, PETSC_DECIDE));
  PetscCall(VecSetUp(u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmTSRedistribute - This function resets the `TS` object, and then replaces the integration information, after a `DMSwarm` migration.

  Collective on ts

  Input Parameter:
. ts - The `TS` object

  Level: developer

.seealso: `DMSwarmTSCreateSolution()`, `DMSwarmMigrate()`, `DMSWARM`, 'TS'
@*/
PetscErrorCode DMSwarmTSRedistribute(TS ts)
{
  DM        sw;
  Vec       u;
  PetscReal t, maxt, dt;
  PetscInt  n, maxn;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetMaxTime(ts, &maxt));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetMaxSteps(ts, &maxn));

  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTime(ts, t));
  PetscCall(TSSetMaxTime(ts, maxt));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetStepNumber(ts, n));
  PetscCall(TSSetMaxSteps(ts, maxn));

  PetscCall(DMSwarmTSCreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
