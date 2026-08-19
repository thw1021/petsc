#pragma once

#include <petscsnestypes.h>
#include <petscdmdatypes.h>

/* SUBMANSEC = TS */

/*S
   TS - Abstract PETSc object that manages integrating an ODE.

   Level: beginner

.seealso: [](integrator_table), [](ch_ts), `TSCreate()`, `TSSetType()`, `TSType`, `SNES`, `KSP`, `PC`, `TSDestroy()`
S*/
typedef struct _p_TS *TS;

/*J
   TSType - String with the name of a PETSc `TS` method. These are all the time/ODE integrators that PETSc provides.

   Level: beginner

   Note:
   Use `TSSetType()` or the options database key `-ts_type` to set the ODE integrator method to use with a given `TS` object

.seealso: [](integrator_table), [](ch_ts), `TSSetType()`, `TS`, `TSRegister()`
J*/
typedef const char *TSType;
#define TSEULER           "euler"
#define TSBEULER          "beuler"
#define TSBASICSYMPLECTIC "basicsymplectic"
#define TSPSEUDO          "pseudo"
#define TSCN              "cn"
#define TSSUNDIALS        "sundials"
#define TSRK              "rk"
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSALPHA           "alpha"
#define TSALPHA2          "alpha2"
#define TSGLLE            "glle"
#define TSGLEE            "glee"
#define TSSSP             "ssp"
#define TSARKIMEX         "arkimex"
#define TSROSW            "rosw"
#define TSEIMEX           "eimex"
#define TSMIMEX           "mimex"
#define TSBDF             "bdf"
#define TSRADAU5          "radau5"
#define TSMPRK            "mprk"
#define TSDISCGRAD        "discgrad"
#define TSIRK             "irk"
#define TSDIRK            "dirk"

/*E
   TSProblemType - Determines the type of problem this `TS` object is to be used to solve

   Values:
 + `TS_LINEAR`    - a linear ODE or DAE
 - `TS_NONLINEAR` - a nonlinear ODE or DAE

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSCreate()`
E*/
typedef enum {
  TS_LINEAR,
  TS_NONLINEAR
} TSProblemType;

/*E
   TSEquationType - type of `TS` problem that is solved

   Values:
+  `TS_EQ_UNSPECIFIED` - (default)
.  `TS_EQ_EXPLICIT`    - {ODE and DAE index 1, 2, 3, HI} F(t,U,U_t) := M(t) U_t - G(U,t) = 0
-  `TS_EQ_IMPLICIT`    - {ODE and DAE index 1, 2, 3, HI} F(t,U,U_t) = 0

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSGetEquationType()`, `TSSetEquationType()`
E*/
typedef enum {
  TS_EQ_UNSPECIFIED               = -1,
  TS_EQ_EXPLICIT                  = 0,
  TS_EQ_ODE_EXPLICIT              = 1,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX1  = 100,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX2  = 200,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX3  = 300,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI = 500,
  TS_EQ_IMPLICIT                  = 1000,
  TS_EQ_ODE_IMPLICIT              = 1001,
  TS_EQ_DAE_IMPLICIT_INDEX1       = 1100,
  TS_EQ_DAE_IMPLICIT_INDEX2       = 1200,
  TS_EQ_DAE_IMPLICIT_INDEX3       = 1300,
  TS_EQ_DAE_IMPLICIT_INDEXHI      = 1500
} TSEquationType;

/*E
   TSConvergedReason - reason a `TS` method has converged (integrated to the requested time) or not

   Values:
+  `TS_CONVERGED_ITERATING`          - this only occurs if `TSGetConvergedReason()` is called during the `TSSolve()`
.  `TS_CONVERGED_TIME`               - the final time was reached
.  `TS_CONVERGED_ITS`                - the maximum number of iterations (time-steps) was reached prior to the final time
.  `TS_CONVERGED_USER`               - user requested termination
.  `TS_CONVERGED_EVENT`              - user requested termination on event detection
.  `TS_CONVERGED_PSEUDO_FATOL`       - stops when function norm decreased by a set amount, used only for `TSPSEUDO`
.  `TS_CONVERGED_PSEUDO_FRTOL`       - stops when function norm decreases below a set amount, used only for `TSPSEUDO`
.  `TS_DIVERGED_NONLINEAR_SOLVE`     - too many nonlinear solve failures have occurred
.  `TS_DIVERGED_STEP_REJECTED`       - too many steps were rejected
.  `TSFORWARD_DIVERGED_LINEAR_SOLVE` - tangent linear solve failed
-  `TSADJOINT_DIVERGED_LINEAR_SOLVE` - transposed linear solve failed

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSGetConvergedReason()`
E*/
typedef enum {
  TS_CONVERGED_ITERATING          = 0,
  TS_CONVERGED_TIME               = 1,
  TS_CONVERGED_ITS                = 2,
  TS_CONVERGED_USER               = 3,
  TS_CONVERGED_EVENT              = 4,
  TS_CONVERGED_PSEUDO_FATOL       = 5,
  TS_CONVERGED_PSEUDO_FRTOL       = 6,
  TS_DIVERGED_NONLINEAR_SOLVE     = -1,
  TS_DIVERGED_STEP_REJECTED       = -2,
  TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
  TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
} TSConvergedReason;
PETSC_EXTERN const char *const *TSConvergedReasons;

/*E
   TSExactFinalTimeOption - option for handling of final time step

   Values:
+  `TS_EXACTFINALTIME_STEPOVER`    - Don't do anything if requested final time is exceeded
.  `TS_EXACTFINALTIME_INTERPOLATE` - Interpolate back to final time
-  `TS_EXACTFINALTIME_MATCHSTEP`   - Adapt final time step to match the final time requested

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSGetConvergedReason()`, `TSSetExactFinalTime()`, `TSGetExactFinalTime()`
E*/
typedef enum {
  TS_EXACTFINALTIME_UNSPECIFIED = 0,
  TS_EXACTFINALTIME_STEPOVER    = 1,
  TS_EXACTFINALTIME_INTERPOLATE = 2,
  TS_EXACTFINALTIME_MATCHSTEP   = 3
} TSExactFinalTimeOption;
PETSC_EXTERN const char *const TSExactFinalTimeOptions[];

/*S
   TSTrajectory - Abstract PETSc object that stores the trajectory (solution of ODE/DAE at each time step)

   Level: advanced

.seealso: [](ch_ts), `TS`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectorySetType()`, `TSTrajectoryDestroy()`, `TSTrajectoryReset()`
S*/
typedef struct _p_TSTrajectory *TSTrajectory;

/*J
   TSTrajectoryType - String with the name of a PETSc `TS` trajectory storage method

   Level: intermediate

.seealso: [](ch_ts), `TS`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`
J*/
typedef const char *TSTrajectoryType;
#define TSTRAJECTORYBASIC         "basic"
#define TSTRAJECTORYSINGLEFILE    "singlefile"
#define TSTRAJECTORYMEMORY        "memory"
#define TSTRAJECTORYVISUALIZATION "visualization"

/*E
   TSTrajectoryMemoryType - Selects the in-memory checkpointing scheme used by `TSTRAJECTORYMEMORY` to store the forward states needed for an adjoint or sensitivity computation

   Values:
+   `TJ_REVOLVE` - the Revolve binomial checkpointing schedule of Griewank & Walther
.   `TJ_CAMS`    - the CAMS (cache-aware multistage) checkpointing schedule
-   `TJ_PETSC`   - PETSc's own in-memory checkpointing implementation

   Level: advanced

.seealso: `TSTrajectory`, `TSTRAJECTORYMEMORY`, `TSTrajectoryMemorySetType()`, `TSSetSaveTrajectory()`
E*/
typedef enum {
  TJ_REVOLVE,
  TJ_CAMS,
  TJ_PETSC
} TSTrajectoryMemoryType;
PETSC_EXTERN const char *const TSTrajectoryMemoryTypes[];

/*S
  TSMonitorDrawCtx - Context object for the `TS` graphical monitor routines that draw the solution, phase plot or error using a `PetscDraw`

  Level: developer

.seealso: `TS`, `TSMonitorDrawCtxCreate()`, `TSMonitorDrawCtxDestroy()`, `TSMonitorDrawSolution()`, `TSMonitorDrawSolutionPhase()`, `TSMonitorDrawError()`, `TSMonitorDrawSolutionFunction()`
S*/
typedef struct _n_TSMonitorDrawCtx *TSMonitorDrawCtx;

/*S
  TSMonitorSolutionCtx - Context object for the `TS` `TSMonitorSolution()` monitor that views the solution at each time step using a `PetscViewer`

  Level: developer

.seealso: `TS`, `TSMonitorSet()`, `TSMonitorSolution()`, `TSMonitorSolutionSetup()`
S*/
typedef struct _n_TSMonitorSolutionCtx *TSMonitorSolutionCtx;

/*S
  TSMonitorVTKCtx - Context object for the `TS` `TSMonitorSolutionVTK()` monitor that dumps the solution to VTK files at each time step

  Level: developer

.seealso: `TS`, `TSMonitorSet()`, `TSMonitorSolutionVTK()`, `TSMonitorSolutionVTKCtxCreate()`, `TSMonitorSolutionVTKDestroy()`
S*/
typedef struct _n_TSMonitorVTKCtx *TSMonitorVTKCtx;

/*S
  TSRHSFunctionFn - A prototype of a `TS` right-hand-side evaluation function that would be passed to `TSSetRHSFunction()`

  Calling Sequence:
+ ts  - timestep context
. t   - current time
. u   - input vector
. F   - function vector
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  The deprecated `TSRHSFunction` still works as a replacement for `TSRHSFunctionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetRHSFunction()`, `DMTSSetRHSFunction()`, `TSIFunctionFn`,
`TSIJacobianFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSRHSFunctionFn(TS ts, PetscReal t, Vec u, Vec F, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSRHSFunctionFn *TSRHSFunction;

/*S
  TSRHSJacobianFn - A prototype of a `TS` right-hand-side Jacobian evaluation function that would be passed to `TSSetRHSJacobian()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - current time
. u    - input vector
. Amat - (approximate) Jacobian matrix
. Pmat - matrix from which preconditioner is to be constructed (usually the same as `Amat`)
- ctx  - [optional] user-defined context for matrix evaluation routine

  Level: beginner

  Note:
  The deprecated `TSRHSJacobian` still works as a replacement for `TSRHSJacobianFn` *.

.seealso: [](ch_ts), `TS`, `TSSetRHSJacobian()`, `DMTSSetRHSJacobian()`, `TSRHSFunctionFn`,
`TSIFunctionFn`, `TSIJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSRHSJacobianFn(TS ts, PetscReal t, Vec u, Mat Amat, Mat Pmat, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSRHSJacobianFn *TSRHSJacobian;

/*S
  TSRHSJacobianPFn - A prototype of a function that computes the Jacobian of G w.r.t. the parameters P where
  U_t = G(U,P,t), as well as the location to store the matrix that would be passed to `TSSetRHSJacobianP()`

  Calling Sequence:
+ ts  - the `TS` context
. t   - current timestep
. U   - input vector (current ODE solution)
. A   - output matrix
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  The deprecated `TSRHSJacobianP` still works as a replacement for `TSRHSJacobianPFn` *.

.seealso: [](ch_ts), `TS`, `TSSetRHSJacobianP()`, `TSGetRHSJacobianP()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSRHSJacobianPFn(TS ts, PetscReal t, Vec U, Mat A, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSRHSJacobianPFn *TSRHSJacobianP;

/*S
  TSSolutionFn - A prototype of a `TS` solution evaluation function that would be passed to `TSSetSolutionFunction()`

  Calling Sequence:
+ ts  - timestep context
. t   - current time
. u   - output vector
- ctx - [optional] user-defined function context

  Level: advanced

  Note:
  The deprecated `TSSolutionFunction` still works as a replacement for `TSSolutionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetSolutionFunction()`, `DMTSSetSolutionFunction()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSSolutionFn(TS ts, PetscReal t, Vec u, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSSolutionFn *TSSolutionFunction;

/*S
  TSForcingFn - A prototype of a `TS` forcing function evaluation function that would be passed to `TSSetForcingFunction()`

  Calling Sequence:
+ ts  - timestep context
. t   - current time
. f   - output vector
- ctx - [optional] user-defined function context

  Level: advanced

  Note:
  The deprecated `TSForcingFunction` still works as a replacement for `TSForcingFn` *.

.seealso: [](ch_ts), `TS`, `TSSetForcingFunction()`, `DMTSSetForcingFunction()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSForcingFn(TS ts, PetscReal t, Vec f, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSForcingFn *TSForcingFunction;

/*S
  TSIFunctionFn - A prototype of a `TS` implicit function evaluation function that would be passed to `TSSetIFunction()

  Calling Sequence:
+ ts  - the `TS` context obtained from `TSCreate()`
. t   - time at step/stage being solved
. U   - state vector
. U_t - time derivative of state vector
. F   - function vector
- ctx - [optional] user-defined context for function

  Level: beginner

  Note:
  The deprecated `TSIFunction` still works as a replacement for `TSIFunctionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetIFunction()`, `DMTSSetIFunction()`, `TSIJacobianFn`, `TSRHSFunctionFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSIFunctionFn(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSIFunctionFn *TSIFunction;

/*S
  TSIJacobianFn - A prototype of a `TS` Jacobian evaluation function that would be passed to `TSSetIJacobian()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - time at step/stage being solved
. U    - state vector
. U_t  - time derivative of state vector
. a    - shift
. Amat - (approximate) Jacobian of F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
. Pmat - matrix used for constructing preconditioner, usually the same as `Amat`
- ctx  - [optional] user-defined context for Jacobian evaluation routine

  Level: beginner

  Note:
  The deprecated `TSIJacobian` still works as a replacement for `TSIJacobianFn` *.

.seealso: [](ch_ts), `TSSetIJacobian()`, `DMTSSetIJacobian()`, `TSIFunctionFn`, `TSRHSFunctionFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSIJacobianFn(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal a, Mat Amat, Mat Pmat, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSIJacobianFn *TSIJacobian;

/*S
  TSI2FunctionFn - A prototype of a `TS` implicit function evaluation function for 2nd order systems that would be passed to `TSSetI2Function()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - time at step/stage being solved
. U    - state vector
. U_t  - time derivative of state vector
. U_tt - second time derivative of state vector
. F    - function vector
- ctx  - [optional] user-defined context for matrix evaluation routine (may be `NULL`)

  Level: advanced

  Note:
  The deprecated `TSI2Function` still works as a replacement for `TSI2FunctionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetI2Function()`, `DMTSSetI2Function()`, `TSIFunctionFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSI2FunctionFn(TS ts, PetscReal t, Vec U, Vec U_t, Vec U_tt, Vec F, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSI2FunctionFn *TSI2Function;

/*S
  TSI2JacobianFn - A prototype of a `TS` implicit Jacobian evaluation function for 2nd order systems that would be passed to `TSSetI2Jacobian()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - time at step/stage being solved
. U    - state vector
. U_t  - time derivative of state vector
. U_tt - second time derivative of state vector
. v    - shift for U_t
. a    - shift for U_tt
. J    - Jacobian of G(U) = F(t,U,W+v*U,W'+a*U), equivalent to dF/dU + v*dF/dU_t  + a*dF/dU_tt
. jac  - matrix from which to construct the preconditioner, may be same as `J`
- ctx  - [optional] user-defined context for matrix evaluation routine

  Level: advanced

  Note:
  The deprecated `TSI2Jacobian` still works as a replacement for `TSI2JacobianFn` *.

.seealso: [](ch_ts), `TS`, `TSSetI2Jacobian()`, `DMTSSetI2Jacobian()`, `TSIFunctionFn`, `TSIJacobianFn`, `TSRHSFunctionFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSI2JacobianFn(TS ts, PetscReal t, Vec U, Vec U_t, Vec U_tt, PetscReal v, PetscReal a, Mat J, Mat Jac, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSI2JacobianFn *TSI2Jacobian;

/*S
  TSTransientVariableFn - A prototype of a function to transform from state to transient variables that would be passed to `TSSetTransientVariable()`

  Calling Sequence:
+ ts  - timestep context
. p   - input vector (primitive form)
. c   - output vector, transient variables (conservative form)
- ctx - [optional] user-defined function context

  Level: advanced

  Note:
  The deprecated `TSTransientVariable` still works as a replacement for `TSTransientVariableFn` *.

.seealso: [](ch_ts), `TS`, `TSSetTransientVariable()`, `DMTSSetTransientVariable()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSTransientVariableFn(TS ts, Vec p, Vec c, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSTransientVariableFn *TSTransientVariable;

/*S
  DMDATSRHSFunctionLocalFn - A prototype of a local `TS` right-hand side residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetRHSFunctionLocal()`

  Calling Sequence:
+ info - defines the subdomain to evaluate the residual on
. t    - time at which to evaluate residual
. x    - array of local state information
. f    - output array of local residual information
- ctx  - optional application context

  Level: beginner

  Note:
  The deprecated `DMDATSRHSFunctionLocal` still works as a replacement for `DMDATSRHSFunctionLocalFn` *.

.seealso: `DMDA`, `DMDATSSetRHSFunctionLocal()`, `TSRHSFunctionFn`, `DMDATSRHSJacobianLocalFn`, `DMDATSIJacobianLocalFn`, `DMDATSIFunctionLocalFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSRHSFunctionLocalFn(DMDALocalInfo *info, PetscReal t, void *x, void *f, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSRHSFunctionLocalFn *DMDATSRHSFunctionLocal;

/*S
  DMDATSRHSJacobianLocalFn - A prototype of a local residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetRHSJacobianLocal()`

  Calling Sequence:
+ info - defines the subdomain to evaluate the residual on
. t    - time at which to evaluate residual
. x    - array of local state information
. J    - Jacobian matrix
. B    - matrix from which to construct the preconditioner; often same as `J`
- ctx  - optional context

  Level: beginner

  Note:
  The deprecated `DMDATSRHSJacobianLocal` still works as a replacement for `DMDATSRHSJacobianLocalFn` *.

.seealso: `DMDA`, `DMDATSSetRHSJacobianLocal()`, `TSRHSJacobianFn`, `DMDATSRHSFunctionLocalFn`, `DMDATSIJacobianLocalFn`, `DMDATSIFunctionLocalFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSRHSJacobianLocalFn(DMDALocalInfo *info, PetscReal t, void *x, Mat J, Mat B, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSRHSJacobianLocalFn *DMDATSRHSJacobianLocal;

/*S
  DMDATSIFunctionLocalFn - A prototype of a local residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetIFunctionLocal()`

  Calling Sequence:
+ info  - defines the subdomain to evaluate the residual on
. t     - time at which to evaluate residual
. x     - array of local state information
. xdot  - array of local time derivative information
. imode - output array of local function evaluation information
- ctx   - optional context

  Level: beginner

  Note:
  The deprecated `DMDATSIFunctionLocal` still works as a replacement for `DMDATSIFunctionLocalFn` *.

.seealso: `DMDA`, `DMDATSSetIFunctionLocal()`, `DMDATSIJacobianLocalFn`, `TSIFunctionFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSIFunctionLocalFn(DMDALocalInfo *info, PetscReal t, void *x, void *xdot, void *imode, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSIFunctionLocalFn *DMDATSIFunctionLocal;

/*S
  DMDATSIJacobianLocalFn - A prototype of a local residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetIJacobianLocal()`

  Calling Sequence:
+ info  - defines the subdomain to evaluate the residual on
. t     - time at which to evaluate the jacobian
. x     - array of local state information
. xdot  - time derivative at this state
. shift - see `TSSetIJacobian()` for the meaning of this parameter
. J     - Jacobian matrix
. B     - matrix from which to construct the preconditioner; often same as `J`
- ctx   - optional context

  Level: beginner

  Note:
  The deprecated `DMDATSIJacobianLocal` still works as a replacement for `DMDATSIJacobianLocalFn` *.

.seealso: `DMDA`, `DMDATSSetIJacobianLocal()`, `TSIJacobianFn`, `DMDATSIFunctionLocalFn`, `DMDATSRHSFunctionLocalFn`, `DMDATSRHSJacobianlocal()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSIJacobianLocalFn(DMDALocalInfo *info, PetscReal t, void *x, void *xdot, PetscReal shift, Mat J, Mat B, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSIJacobianLocalFn *DMDATSIJacobianLocal;

/*S
  TSMonitorLGCtx - Context object for `TS` line-graph monitor routines that plot residuals, iteration counts or solution components at each time step on a `PetscDrawLG`

  Level: developer

.seealso: `TS`, `TSMonitorLGCtxCreate()`, `TSMonitorLGCtxDestroy()`, `TSMonitorLGSolution()`, `TSMonitorLGTimeStep()`, `TSMonitorLGError()`
S*/
typedef struct _n_TSMonitorLGCtx *TSMonitorLGCtx;

/*S
  TSMonitorDMDARayCtx - Context object for `TSMonitorDMDARay()`

  Level: developer

.seealso: `TS`, `TSSetMonitor()`, `TSMonitorDMDARay()`, `TSMonitorDMDARayDestroy()`
S*/
typedef struct {
  Vec            ray;
  VecScatter     scatter;
  PetscViewer    viewer;
  TSMonitorLGCtx lgctx;
} TSMonitorDMDARayCtx;

/*S
  TSMonitorLGCtxNetwork - Context object for the `TSMonitorLGCtxNetworkSolution()` line-graph monitor that plots solution components on each subnetwork of a `DMNETWORK`

  Level: developer

.seealso: `TS`, `DMNETWORK`, `TSMonitorLGCtxNetworkCreate()`, `TSMonitorLGCtxNetworkDestroy()`, `TSMonitorLGCtxNetworkSolution()`
S*/
typedef struct _n_TSMonitorLGCtxNetwork *TSMonitorLGCtxNetwork;

/*S
  TSMonitorEnvelopeCtx - Context object for the `TSMonitorEnvelope()` monitor that tracks the per-component min/max envelope of the solution over a time integration

  Level: developer

.seealso: `TS`, `TSMonitorEnvelopeCtxCreate()`, `TSMonitorEnvelopeCtxDestroy()`, `TSMonitorEnvelope()`, `TSMonitorEnvelopeGetBounds()`
S*/
typedef struct _n_TSMonitorEnvelopeCtx *TSMonitorEnvelopeCtx;

/*S
  TSMonitorSPEigCtx - Context object for the `TSMonitorSPEig()` monitor that displays an estimate of the spectrum of the operator using a `PetscDrawSP` scatter plot

  Level: developer

.seealso: `TS`, `TSMonitorSPEigCtxCreate()`, `TSMonitorSPEigCtxDestroy()`, `TSMonitorSPEig()`
S*/
typedef struct _n_TSMonitorSPEigCtx *TSMonitorSPEigCtx;

/*S
  TSMonitorSPCtx - Context object for the `TSMonitorSPSwarmSolution()` scatter-plot monitor that draws the swarm particle positions at each time step on a `PetscDrawSP`

  Level: developer

.seealso: `TS`, `DMSWARM`, `TSMonitorSPCtxCreate()`, `TSMonitorSPCtxDestroy()`, `TSMonitorSPSwarmSolution()`
S*/
typedef struct _n_TSMonitorSPCtx *TSMonitorSPCtx;

/*S
  TSMonitorHGCtx - Context object for the `TSMonitorHGSwarmSolution()` histogram monitor that displays a histogram of `DMSWARM` particle quantities at each time step

  Level: developer

.seealso: `TS`, `DMSWARM`, `TSMonitorHGCtxCreate()`, `TSMonitorHGCtxDestroy()`, `TSMonitorHGSwarmSolution()`
S*/
typedef struct _n_TSMonitorHGCtx *TSMonitorHGCtx;

/*M
   TSEvent - Abstract object to handle event detection in `TS` time integrator

   Level: intermediate

   Note:
   See `TSSetEventHandler()` for the management of events.

.seealso: [](sec_ts_event), `TS`, `TSSetEventHandler()`, `TSSetPostEventStep()`, `TSSetPostEventSecondStep()`, `TSSetEventTolerances()`, `TSGetNumEvents()`
M*/
typedef struct _n_TSEvent *TSEvent;

/*J
   TSSSPType - string with the name of a `TSSSP` scheme.

   Level: beginner

.seealso: [](ch_ts), `TSSSPSetType()`, `TS`, `TSSSP`
J*/
typedef const char *TSSSPType;
#define TSSSPRKS2  "rks2"
#define TSSSPRKS3  "rks3"
#define TSSSPRK104 "rk104"

/*S
   TSAdapt - Abstract object that manages time-step adaptivity

   Level: beginner

.seealso: [](ch_ts), [](sec_ts_error_control), `TS`, `TSGetAdapt()`, `TSAdaptCreate()`, `TSAdaptType`
S*/
typedef struct _p_TSAdapt *TSAdapt;

/*J
   TSAdaptType - String with the name of `TSAdapt` scheme.

   Level: beginner

.seealso: [](ch_ts), [](sec_ts_error_control), `TSGetAdapt()`, `TSAdaptSetType()`, `TS`, `TSAdapt`
J*/
typedef const char *TSAdaptType;
#define TSADAPTNONE    "none"
#define TSADAPTBASIC   "basic"
#define TSADAPTDSP     "dsp"
#define TSADAPTCFL     "cfl"
#define TSADAPTGLEE    "glee"
#define TSADAPTHISTORY "history"

/*S
   TSGLLEAdapt - Abstract object that manages time-step adaptivity for `TSGLLE`

   Level: beginner

   Developer Note:
   This functionality should be replaced by the `TSAdapt`.

.seealso: [](ch_ts), `TS`, `TSGLLE`, `TSGLLEAdaptCreate()`, `TSGLLEAdaptType`
S*/
typedef struct _p_TSGLLEAdapt *TSGLLEAdapt;

/*J
   TSGLLEAdaptType - String with the name of `TSGLLEAdapt` scheme

   Level: beginner

   Developer Note:
   This functionality should be replaced by the `TSAdaptType`.

.seealso: [](ch_ts), `TSGLLEAdaptSetType()`, `TS`
J*/
typedef const char *TSGLLEAdaptType;
#define TSGLLEADAPT_NONE "none"
#define TSGLLEADAPT_SIZE "size"
#define TSGLLEADAPT_BOTH "both"

/*J
   TSGLLEAcceptType - String with the name of `TSGLLEAccept` scheme

   Level: beginner

.seealso: [](ch_ts), `TSGLLESetAcceptType()`, `TS`, `TSGLLEAccept`
J*/
typedef const char *TSGLLEAcceptType;
#define TSGLLEACCEPT_ALWAYS "always"

/*S
  TSGLLEAcceptFn - A prototype of a `TS` accept function that would be passed to `TSGLLEAcceptRegister()`

  Calling Sequence:
+ ts     - timestep context
. nt     - time to end of solution time
. h      - the proposed step-size
. enorm  - unknown
- accept - output, if the proposal is accepted

  Level: beginner

  Note:
  The deprecated `TSGLLEAcceptFunction` still works as a replacement for `TSGLLEAcceptFn` *

.seealso: [](ch_ts), `TS`, `TSSetRHSFunction()`, `DMTSSetRHSFunction()`, `TSIFunctionFn`,
`TSIJacobianFn`, `TSRHSJacobianFn`, `TSGLLEAcceptRegister()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSGLLEAcceptFn(TS ts, PetscReal nt, PetscReal h, const PetscReal enorm[], PetscBool *accept);

PETSC_EXTERN_TYPEDEF typedef TSGLLEAcceptFn *TSGLLEAcceptFunction;

/*J
  TSGLLEType - string with the name of a General Linear `TSGLLE` type

  Level: beginner

.seealso: [](ch_ts), `TS`, `TSGLLE`, `TSGLLESetType()`, `TSGLLERegister()`, `TSGLLEAccept`
J*/
typedef const char *TSGLLEType;
#define TSGLLE_IRKS "irks"

/*J
   TSEIMEXType - String with the name of an Extrapolated IMEX `TSEIMEX` type

   Level: beginner

.seealso: [](ch_ts), `TSEIMEXSetType()`, `TS`, `TSEIMEX`, `TSEIMEXRegister()`
J*/
#define TSEIMEXType char *

/*J
   TSRKType - String with the name of a Runge-Kutta `TSRK` type

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSRKSetType()`, `TSRK`, `TSRKRegister()`
J*/
typedef const char *TSRKType;
#define TSRK1FE "1fe"
#define TSRK2A  "2a"
#define TSRK2B  "2b"
#define TSRK3   "3"
#define TSRK3BS "3bs"
#define TSRK4   "4"
#define TSRK5F  "5f"
#define TSRK5DP "5dp"
#define TSRK5BS "5bs"
#define TSRK6VR "6vr"
#define TSRK7VR "7vr"
#define TSRK8VR "8vr"

/*J
   TSMPRKType - String with the name of a partitioned Runge-Kutta `TSMPRK` type

   Level: beginner

.seealso: [](ch_ts), `TSMPRKSetType()`, `TS`, `TSMPRK`, `TSMPRKRegister()`
J*/
typedef const char *TSMPRKType;
#define TSMPRK2A22 "2a22"
#define TSMPRK2A23 "2a23"
#define TSMPRK2A32 "2a32"
#define TSMPRK2A33 "2a33"
#define TSMPRKP2   "p2"
#define TSMPRKP3   "p3"

/*J
   TSIRKType - String with the name of an implicit Runge-Kutta `TSIRK` type

   Level: beginner

.seealso: [](ch_ts), `TSIRKSetType()`, `TS`, `TSIRK`, `TSIRKRegister()`
J*/
typedef const char *TSIRKType;
#define TSIRKGAUSS "gauss"

/*J
   TSGLEEType - String with the name of a General Linear with Error Estimation `TSGLEE` type

   Level: beginner

.seealso: [](ch_ts), `TSGLEESetType()`, `TS`, `TSGLEE`, `TSGLEERegister()`
J*/
typedef const char *TSGLEEType;
#define TSGLEEi1      "BE1"
#define TSGLEE23      "23"
#define TSGLEE24      "24"
#define TSGLEE25I     "25i"
#define TSGLEE35      "35"
#define TSGLEEEXRK2A  "exrk2a"
#define TSGLEERK32G1  "rk32g1"
#define TSGLEERK285EX "rk285ex"

/*J
  TSARKIMEXType - String with the name of an Additive Runge-Kutta IMEX `TSARKIMEX` type

  Options Database Key:
. -ts_arkimex_type (1bee|a2|l2|ars122|2c|2d|2e|prssp2|3|bpr3|ars443|4|5) - set `TSARKIMEX` scheme type, see `TSARKIMEXType`

  Level: beginner

.seealso: [](ch_ts), `TSARKIMEXSetType()`, `TS`, `TSARKIMEX`, `TSARKIMEXRegister()`
J*/
typedef const char *TSARKIMEXType;
#define TSARKIMEX1BEE   "1bee"
#define TSARKIMEXA2     "a2"
#define TSARKIMEXL2     "l2"
#define TSARKIMEXARS122 "ars122"
#define TSARKIMEX2C     "2c"
#define TSARKIMEX2D     "2d"
#define TSARKIMEX2E     "2e"
#define TSARKIMEXPRSSP2 "prssp2"
#define TSARKIMEX3      "3"
#define TSARKIMEXBPR3   "bpr3"
#define TSARKIMEXARS443 "ars443"
#define TSARKIMEX4      "4"
#define TSARKIMEX5      "5"

/*J
   TSDIRKType - String with the name of a Diagonally Implicit Runge-Kutta `TSDIRK` type

   Level: beginner

.seealso: [](ch_ts), `TSDIRKSetType()`, `TS`, `TSDIRK`, `TSDIRKRegister()`
J*/
typedef const char *TSDIRKType;
#define TSDIRKS212      "s212"
#define TSDIRKES122SAL  "es122sal"
#define TSDIRKES213SAL  "es213sal"
#define TSDIRKES324SAL  "es324sal"
#define TSDIRKES325SAL  "es325sal"
#define TSDIRK657A      "657a"
#define TSDIRKES648SA   "es648sa"
#define TSDIRK658A      "658a"
#define TSDIRKS659A     "s659a"
#define TSDIRK7510SAL   "7510sal"
#define TSDIRKES7510SA  "es7510sa"
#define TSDIRK759A      "759a"
#define TSDIRKS7511SAL  "s7511sal"
#define TSDIRK8614A     "8614a"
#define TSDIRK8616SAL   "8616sal"
#define TSDIRKES8516SAL "es8516sal"

/*J
   TSRosWType - String with the name of a Rosenbrock-W `TSROSW` type

   Level: beginner

.seealso: [](ch_ts), `TSRosWSetType()`, `TS`, `TSROSW`, `TSRosWRegister()`
J*/
typedef const char *TSRosWType;
#define TSROSW2M          "2m"
#define TSROSW2P          "2p"
#define TSROSWRA3PW       "ra3pw"
#define TSROSWRA34PW2     "ra34pw2"
#define TSROSWR34PRW      "r34prw"
#define TSROSWR3PRL2      "r3prl2"
#define TSROSWRODAS3      "rodas3"
#define TSROSWRODASPR     "rodaspr"
#define TSROSWRODASPR2    "rodaspr2"
#define TSROSWSANDU3      "sandu3"
#define TSROSWASSP3P3S1C  "assp3p3s1c"
#define TSROSWLASSP3P4S2C "lassp3p4s2c"
#define TSROSWLLSSP3P4S2C "llssp3p4s2c"
#define TSROSWARK3        "ark3"
#define TSROSWTHETA1      "theta1"
#define TSROSWTHETA2      "theta2"
#define TSROSWGRK4T       "grk4t"
#define TSROSWSHAMP4      "shamp4"
#define TSROSWVELDD4      "veldd4"
#define TSROSW4L          "4l"

/*J
  TSBasicSymplecticType - String with the name of a basic symplectic integration `TSBASICSYMPLECTIC` type

  Level: beginner

.seealso: [](ch_ts), `TSBasicSymplecticSetType()`, `TS`, `TSBASICSYMPLECTIC`, `TSBasicSymplecticRegister()`
J*/
typedef const char *TSBasicSymplecticType;
#define TSBASICSYMPLECTICSIEULER   "1"
#define TSBASICSYMPLECTICVELVERLET "2"
#define TSBASICSYMPLECTIC3         "3"
#define TSBASICSYMPLECTIC4         "4"

/*E
   TSDGType - Selects the discrete-gradient flavor used by `TSDISCGRAD` when integrating gradient-system ODEs

   Values:
+   `TS_DG_GONZALEZ` - Gonzalez's mid-point-style discrete gradient
.   `TS_DG_AVERAGE`  - average vector field (AVF) discrete gradient
-   `TS_DG_NONE`     - do not apply a discrete gradient correction; the integrator falls back to a standard mid-point rule

   Level: advanced

.seealso: `TS`, `TSDISCGRAD`, `TSDiscGradSetType()`, `TSDiscGradGetType()`, `TSDiscGradSetFormulation()`
E*/
typedef enum {
  TS_DG_GONZALEZ,
  TS_DG_AVERAGE,
  TS_DG_NONE
} TSDGType;

#if PetscDefined(HAVE_SUNDIALS2)
/*E
   TSSundialsLmmType - Selects which linear multistep method is used by the `TSSUNDIALS` interface to SUNDIALS' CVODE integrator

   Values:
+   `SUNDIALS_ADAMS` - variable-order Adams methods (non-stiff problems)
-   `SUNDIALS_BDF`   - variable-order backward differentiation formulas (stiff problems)

   Level: intermediate

.seealso: `TS`, `TSSUNDIALS`, `TSSundialsSetType()`, `TSSundialsGramSchmidtType`
E*/
typedef enum {
  SUNDIALS_ADAMS = 1,
  SUNDIALS_BDF   = 2
} TSSundialsLmmType;
PETSC_EXTERN const char *const TSSundialsLmmTypes[];

/*E
   TSSundialsGramSchmidtType - Selects the Gram--Schmidt orthogonalization variant used by SUNDIALS' internal GMRES inside `TSSUNDIALS`

   Values:
+   `SUNDIALS_MODIFIED_GS`  - modified Gram--Schmidt (more stable)
-   `SUNDIALS_CLASSICAL_GS` - classical Gram--Schmidt (cheaper, less stable)

   Level: advanced

.seealso: `TS`, `TSSUNDIALS`, `TSSundialsSetGramSchmidtType()`, `TSSundialsLmmType`
E*/
typedef enum {
  SUNDIALS_MODIFIED_GS  = 1,
  SUNDIALS_CLASSICAL_GS = 2
} TSSundialsGramSchmidtType;
PETSC_EXTERN const char *const TSSundialsGramSchmidtTypes[];
#endif

/*S
  TSAlpha2PredictorFn - A callback to set the predictor (i.e., the initial guess for the nonlinear solver) in
  a second-order generalized-alpha time integrator.

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. X0   - the previous time step's state vector
. V0   - the previous time step's first derivative of the state vector
. A0   - the previous time step's second derivative of the state vector
. X1   - the vector into which the initial guess for the current time step will be written
- ctx  - [optional] user-defined context for the predictor evaluation routine (may be `NULL`)

  Level: intermediate

  Note:
  The deprecated `TSAlpha2Predictor` still works as a replacement for `TSAlpha2PredictorFn` *.

.seealso: [](ch_ts), `TS`, `TSAlpha2SetPredictor()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSAlpha2PredictorFn(TS ts, Vec X0, Vec V0, Vec A0, Vec X1, PetscCtx ctx);

PETSC_EXTERN_TYPEDEF typedef TSAlpha2PredictorFn *TSAlpha2Predictor;
