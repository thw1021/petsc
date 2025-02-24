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
static PetscErrorCode TSDeviceGetNumEquations(TS ts, PetscInt *numEq){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    *numEq = device->n_des;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDeviceSetNumEquations(TS ts, PetscInt numEq){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    device->n_des = numEq;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_Device(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_Device(TS ts)
{
  TS_Device *device = (TS_Device *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&device->update));
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
  PetscInt  Ne;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)ts), "", "TSDevice Options", "TSDEVICE");
  PetscCall(PetscOptionsInt("-ts_device_num_equations", "The size of the system in terms of separable equations, default is 1.", "", Ne, &Ne, &flg));
  if (!flg) Ne = 1;
  PetscCall(TSDeviceSetNumEquations(ts, Ne));
  PetscOptionsEnd();
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

template
typedef struct {

} TSDevice_Euler;

template<typename func>
__device__ void TSStep_DeviceFn(TS *ts, PetscInt i, PetscScalar *sol, PetscScalar *res, PetscReal dt, func rhsfunc){
    return;
}

template<typename func>
__global__ void TSSolve_DeviceFn(TS *ts, PetscInt N, PetscScalar *sol, PetscScalar *res, PetscReal dt, func rhsfunc){
    int i = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
    if (i < N){
      // pre stage
      // computerhsfunction
      // TODO: Generalize to multi dimensional problems.
      while (!converged) {
        TSStep_DeviceFn
        rhsfunc(ts, dt, sol, res, NULL);
        res[i] = res[i] * dt + sol[i];// come back and leave to rocblas?
      }
      // copy update to solution
      for (int d = 0; d < elem; ++d ) sol[i*elem + d] = res[i*elem + d];
      // increment ptime w/ time step
      // time_step = nex time step
      printf("TS_Device thread %d\n", i);
    }
    return;
}

template<typename func>
PetscErrorCode TSSolve_Device(TS ts, func rhsfunc){
  Vec solution, residual;
  PetscScalar *sol;
  PetscScalar *res, dt;
  PetscInt     neq;
  //TS_Device *ts_device = (TS_Device*)ts->data;

  PetscFunctionBegin;
  dt = ts->time_step;
  PetscCall(TSDeviceGetNumEquations(ts, &neq));
  PetscPrintf(PETSC_COMM_WORLD, "TSSolve_Device for %" PetscInt_FMT " equations.\n", neq);
  PetscCall(TSGetSolution(ts, &solution));
  PetscCall(VecDuplicate(solution, &residual));
  PetscCall(VecZeroEntries(residual));
  PetscCall(VecGetArrayWrite(residual, &res));
  PetscCall(VecGetArrayWrite(solution, &sol));
  //The default device configuration should be a size of the system of 1, although anyone that
  //uses it that way would be hamstringing their efficiency over the classic TS and it is highly
  //not recommended to do so
  hipLaunchKernelGGL(HIP_KERNEL_NAME(TSSolve_DeviceFn), dim3(256), dim3(256), 0, PetscDefaultHipStream, ts, neq, sol, res, dt, rhsfunc);
  PetscCallHIP(hipPeekAtLastError());
  PetscCallHIP(hipDeviceSynchronize());

  PetscCall(VecRestoreArrayWrite(residual, &res));
  PetscCall(VecRestoreArrayWrite(solution, &sol));
  PetscCall(VecDestroy(&residual));
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
