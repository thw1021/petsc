/*
  Explicit Euler batched time stepping. The purpose is to batch solves over systems
  consisting of non coupled systems with varying rates of stepping per system on the GPU,
  all leveraging the same kernel but with different solver parameters.
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petsctsdevice.hpp>
#include <petscdevice_hip.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

typedef struct {
  Vec update; /* work vector where new solution is formed  */
  PetscInt n_des;
  void (*func)(TS, PetscReal, PetscScalar*, PetscScalar*, void *);
} TS_Device;

PETSC_EXTERN PetscErrorCode TSDeviceSetRHSFunction(TS ts, Vec v, TSDeviceRHSFunctionFn *func, void *ctx){
    TS_Device *device = (TS_Device*)ts->data;

    PetscFunctionBegin;
    device->func = func;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_Device(TS ts)
{
    //TS_Device *device = (TS_Device *)ts->data;
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
  PetscCall(TSReset_Device(ts));
  PetscCall(PetscFree(ts->data));
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

__device__ static void TSStep_DeviceFn(){

}

template<typename func>
__global__ void TSSolve_DeviceFn(TS ts, PetscScalar *sol, PetscScalar *res, PetscReal dt){

    int i = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
    // pre stage
    // computerhsfunction
    func(ts, dt, sol, res, NULL);
    // vecaypx
    // poststage
    // adaptcheckstage
    // checkdiverged
    // domainerror
    // checkstageok
    // adapter?
    // copy update to solution
    // increment ptime w/ time step
    // time_step = nex time step
    printf("TS_Device thread %d\n", i);
    return;
}

template<typename func>
PetscErrorCode TSSolve_Device(TS ts){
  PetscScalar *sol, *res, dt;
  TS_Device *ts_device = (TS_Device*)ts->data;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD, "TSSolve_Device\n");
  hipLaunchKernelGGL(HIP_KERNEL_NAME(TSSolve_DeviceFn<func>), dim3(256), dim3(256), 0, PetscDefaultHipStream, ts, sol, res, dt);
  PetscCallHIP(hipPeekAtLastError());
  PetscCallHIP(hipDeviceSynchronize());
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
