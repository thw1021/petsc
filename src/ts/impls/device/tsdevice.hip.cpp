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

const char *TSDeviceTypeNames[]   = {"tsdevice_euler", "tsdevice_rk4", NULL};

typedef struct {
  Vec update; /* work vector where new solution is formed  */
  PetscInt n_des;
  PetscInt elements;
  void *ts_device_data;
  TSDeviceType type;
  void (*func)(TS, PetscReal, PetscScalar*, PetscScalar*, void *);
  PetscInt maxSteps;
} TS_Device;

PETSC_EXTERN PetscErrorCode TSDeviceSetRHSFunction(TS ts, Vec v, TSDeviceRHSFunctionFn *func, void *ctx){
    TS_Device *device = (TS_Device*)ts->data;

    PetscFunctionBegin;
    device->func = func;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDeviceSetType(TS ts, TSDeviceType type) {
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    device->type = type;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSDeviceGetType(TS ts, TSDeviceType *type) {
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    *type = device->type;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_Device(TS ts)
{
    //TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDeviceGetNumEquations(TS ts, PetscInt *numEq){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    *numEq = device->n_des;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TSDeviceSetNumEquations(TS ts, PetscInt numEq){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    device->n_des = numEq;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TSDeviceSetNumElements(TS ts, PetscInt numEle){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    device->elements = numEle;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDeviceGetNumElements(TS ts, PetscInt *numEle){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    *numEle = device->elements;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TSDeviceSetMaxSteps(TS ts, PetscInt maxSteps){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    device->maxSteps = maxSteps;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDeviceGetMaxSteps(TS ts, PetscInt *maxSteps){
    TS_Device *device = (TS_Device *)ts->data;
    PetscFunctionBegin;
    *maxSteps = device->maxSteps;
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_Device(TS ts)
{
  TS_Device *device = (TS_Device *)ts->data;
  PetscInt elements, size;
  TSDeviceType type;
  PetscFunctionBegin;
  PetscCall(TSDeviceGetNumEquations(ts, &size));
  PetscCall(TSDeviceGetNumElements(ts, &elements));
  PetscCall(TSDeviceGetType(ts, &type));
  // Allocate the internal array of device memory.
  // Each element points to an on device struct with
  // its own step size, convergence history, and in the future
  // method for rhs updates.
  switch (type) {
    case TSDEVICE_EULER:
      PetscCallHIP(hipMalloc(&(device->ts_device_data), size*sizeof(TSDevice_Euler)));
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_WRONG, "Unknown or unsupported ts device type.");
      break;
  }
  PetscPrintf(PETSC_COMM_WORLD, "Device has been set up\n");
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
  PetscInt  Ne, Nele = 1, maxSteps = 10;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)ts), "", "TSDevice Options", "TSDEVICE");
  PetscCall(PetscOptionsInt("-ts_device_num_equations", "The size of the system in terms of separable equations, default is 1.", "", Ne, &Ne, &flg));
  if (!flg) Ne = 1;
  PetscCall(TSDeviceSetNumEquations(ts, Ne));flg=PETSC_FALSE;
  PetscCall(PetscOptionsInt("-ts_device_num_elements", "Number of elements in each individual system. Default is 1.", "", Nele, &Nele, &flg));
  PetscCall(TSDeviceSetNumElements(ts, Nele));
  PetscCall(PetscOptionsInt("-ts_max_steps", "Max number of steps to take before declaring convergence.", "", maxSteps, &maxSteps, NULL));
  PetscCall(TSDeviceSetMaxSteps(ts, maxSteps));
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

__device__ void TSDevice_SetUpDevice_Euler(TS_Device *ts, TSDevice_Euler *tsDevice, PetscInt i, PetscReal dt){
    //hipMalloc(&tsDevice, sizeof(TSDevice_Euler));
    tsDevice->i = i;
    tsDevice->elements = ts->elements;
    tsDevice->converged = PETSC_FALSE;
    tsDevice->time = 0.;
    tsDevice->dt = dt;
    tsDevice->maxSteps = ts->maxSteps;
    tsDevice->step = 0;
}

template<typename postEventFunc>
__device__ void TSDevice_PostEventFunction(TSDevice_Euler ts, PetscReal t, PetscReal *sol, postEventFunc func){
    func(ts, t, sol);
    return;
}

template<typename func>
__device__ void TSDevice_Event(func rhsfunction, PetscReal *update, PetscReal *indicator){
    // Overloaded functor measures indicator to determine direction of zero crossing.
    rhsfunction(update, indicator);// later make this take a TSDevice?
    return;
}

// Need to decide if I want to pass the reference or the pointer eventually, the mix and matching is a mess
template<typename func>
__device__ void TSStep_DeviceFn(TSDevice_Euler ts, PetscScalar *sol, PetscScalar *res, func rhsfunc){
    rhsfunc(ts, ts.dt, sol, res, NULL);
    return;
}

/*
  Initial implementation to just output the solution. Mostly for debugging and plotting purposes
  and not necessarily performant.
*/
__device__ void TSDevice_SolutionMonitor(TSDevice_Euler tsdevice, PetscReal *solution, int i){
    printf("Thread %d at time %g with dt %g solution %g %g %g %g\n", i, tsdevice.time, tsdevice.dt, solution[i*tsdevice.elements + 0], solution[i*tsdevice.elements + 1], solution[i*tsdevice.elements + 2], solution[i*tsdevice.elements + 3]);
}

__device__ void TSDevice_View(TSDevice_Euler tsdevice){
    printf("TSDevice type: Euler at step: %d with dt %g, max step %d, on thread %d\n", tsdevice.step, tsdevice.dt, tsdevice.maxSteps, tsdevice.i);
}

/*
  TODOs:
    1) Allow one initial \delta t, but implement adaptation for individual sub TS
    2) Generalized struct set up and assembly so each thread can generate its own type
       (aligns with general \delta t for adaptivity in the method as well as step size)
    3) Generate a viewer so each thread can output its own TSDevice_TYPE information and
       ponder a way to parse the mess that will be in large problem domains
    4) Interface to DMPlex/DMSwarm for computation of field data (DMTS but will not
       hold the whole thing, just device allocated pointers for field data and some
       device functions for operating on it.)
*/
template<typename func>
__global__ void TSSolve_DeviceFn(TS_Device *ts, PetscInt N, PetscScalar *sol, PetscScalar *res, PetscReal dt, func rhsfunc){
    int i = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
    TSDevice_Euler *ldctx = (TSDevice_Euler*)ts->ts_device_data;

    // Check thread index less than global thread count
    if (i < N){
      PetscReal update[4];// TODO: update needs to be generalized in size to avoid weird overflows
      int elem = ts->elements;
      PetscReal indicator;

      // Set up the on device context
      TSDevice_SetUpDevice_Euler(ts, &ldctx[i], i, dt);
      // Call main viewer, or any user provided viewers if applicable
      TSDevice_View(ldctx[i]);

      printf("Thread %d iterating to convergence.\n", i);
      while (!ldctx[i].converged) {
        PetscReal ptime = ldctx[i].dt;
        // Perform rhs function evaluation
        TSStep_DeviceFn(ldctx[i], sol, res, rhsfunc);

        // Store update to check for events and divergence
        for (int e = 0; e < ldctx[i].elements; e++) {
            update[e] = (res[i*elem+e] * dt) + sol[i*elem +e];
        }

        // Check if the update triggers an event, update to leverage the device struct
        // to customize how the zero crossing is calculated and customize its handling
        TSDevice_Event(rhsfunc, update, &indicator);
        // Right now, just mark converged if the direction is negative
        if (indicator < 0.0) {
            PetscBool found = PETSC_FALSE;
            PetscReal rightBound, leftBound;

            rightBound = ldctx[i].dt;
            leftBound = rightBound/2.0;
            while (!found) {
                for (int e = 0; e < ldctx[i].elements; e++) update[e] = (res[i*elem+e] * leftBound) + sol[i*elem +e];
                if (update[1] < 0.0){
                    rightBound = leftBound;
                    leftBound = rightBound/2;
                }
                if (update[1] > 0.0 && update[1] < 1000*PETSC_SMALL) {
                    found = PETSC_TRUE;
                    ptime = leftBound;
                    break;
                }
                if (update[1] > 0.0){
                    leftBound = (leftBound + rightBound)/2.0;
                }
            }
            TSDevice_PostEventFunction(ldctx[i], ldctx[i].time, update, rhsfunc);
            //ldctx[i].converged = PETSC_TRUE;
        }
        for (int e = 0; e < ldctx[i].elements; e++) sol[i*elem+e] = update[e];
        ldctx[i].time += ptime;
        ldctx[i].step += 1;

        TSDevice_SolutionMonitor(ldctx[i], sol, i);
        if (ldctx[i].step >= ldctx[i].maxSteps) ldctx[i].converged = PETSC_TRUE;//Call device function to check convergence for solver type.
        if (ldctx[i].converged) printf("Thread %d converged;", i);
      }
      // increment ptime w/ time step
      // time_step = nex time step
      printf("TS_Device thread %d done.\n", i);
    }
    return;
}

template<typename func>
PetscErrorCode TSSolve_Device(TS ts, func rhsfunc){
  Vec solution, residual;
  PetscScalar *sol, *res, dt;
  PetscInt     neq, size;
  TS_Device   *ts_device;// = (TS_Device*)ts->data;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  dt = ts->time_step;
  PetscCall(TSSetUp(ts));
  PetscCall(TSDeviceGetNumEquations(ts, &neq));
  PetscPrintf(PETSC_COMM_WORLD, "TSSolve_Device for %" PetscInt_FMT " equations.\n", neq);
  PetscCall(TSGetSolution(ts, &solution));
  PetscCall(VecViewFromOptions(solution, NULL, "-solution_view"));
  PetscCall(VecGetSize(solution, &size));
  PetscCall(VecDuplicate(solution, &residual));
  PetscCall(VecZeroEntries(residual));
  PetscCall(VecViewFromOptions(residual, NULL, "-res_view"));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArray(residual, &res));
  PetscCall(VecGetArray(solution, &sol));

  //The default device configuration should be a size of the system of 1, although anyone that
  //uses it that way would be hamstringing their efficiency over the classic TS and it is highly
  //not recommended to do so
  PetscCallHIP(hipMalloc((void**)&ts_device, sizeof(TS_Device)));
  PetscCallHIP(hipMemcpy(ts_device, ts->data, sizeof(TS_Device), hipMemcpyHostToDevice));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(TSSolve_DeviceFn), dim3(256), dim3(256), 0, PetscDefaultHipStream, ts_device, neq, sol, res, dt, rhsfunc);
  PetscCallHIP(hipFree(ts_device));//clenup
  PetscCallHIP(hipPeekAtLastError());
  PetscCallHIP(hipDeviceSynchronize());

  PetscCall(VecRestoreArray(residual, &res));
  PetscCall(VecRestoreArray(solution, &sol));
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
