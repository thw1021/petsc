/*
  Explicit Euler batched time stepping. The purpose is to batch solves over systems
  consisting of non coupled systems with varying rates of stepping per system on the GPU,
  all leveraging the same kernel but with different solver parameters.
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdevice_hip.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

typedef struct {
  Vec update; /* work vector where new solution is formed  */
  void (*func)(TS, PetscReal, Vec, Vec, void *);
} TS_Device;

PETSC_EXTERN PetscErrorCode TSDeviceSetRHSFunction(TS ts, Vec v, TSDeviceRHSFunctionFn *func, void *ctx){
    TS_Device *device = (TS_Device*)ts->data;

    PetscFunctionBegin;
    device->func = func;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_Device(TS ts)
{
    PetscFunctionBegin;
    PetscFunctionReturn(PETSC_SUCCESS);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetUp_Device(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_Device(TS ts)
{

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_Device(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_Device(TS ts, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_Device(TS ts, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_Device(TS ts, PetscReal t, Vec X)
{

  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeLinearStability_Device(TS ts, PetscReal xr, PetscReal xi, PetscReal *yr, PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* ------------------------------------------------------------ */

/*MC
      TSDEVICEEULER - Device implementation for forward Euler

  Level: beginner

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSBEULER`, `TSType`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_Device(TS ts)
{
  TS_Device *device;

  PetscFunctionBegin;
  PetscCall(PetscNew(&device));
  ts->data = (void *)device;

  ts->ops->setup           = TSSetUp_Device;
  ts->ops->step            = TSStep_Device;
  ts->ops->reset           = TSReset_Device;
  ts->ops->destroy         = TSDestroy_Device;
  ts->ops->setfromoptions  = TSSetFromOptions_Device;
  ts->ops->view            = TSView_Device;
  ts->ops->interpolate     = TSInterpolate_Device;
  ts->ops->linearstability = TSComputeLinearStability_Device;
  ts->default_adapt_type   = TSADAPTNONE;
  ts->usessnes             = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
