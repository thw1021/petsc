#include <petsc/private/tsimpl.h> /*I "petscts.h" */
EXTERN_C_BEGIN
#include <cvode/cvode.h> /* prototypes for CVODE fcts. */
#include <nvector/nvector_petsc.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>
EXTERN_C_END

typedef struct {
  SUNContext         sunctx;
  void              *cvode_mem;
  int                cvtype;
  PetscBool          cvodeInited;
  N_Vector           nvecx;
  Vec                xdot;
  SUNNonlinearSolver NLS;
  PetscInt           order, max_order;
  Vec                constraints;
} TS_CVode;

/* Currently the PetscCallSUNDIALS macros are only used in this single source file. Eventually they will likely be moved to a private include file */
/*MC
  PetscCallSUNDIALSSUNErrCode - Calls a PETSc function from a SUNDIALS function that returns SUNErrCode.

  Synopsis:
  #include <petscsys.h>
  SUNErrCode PetscCallSUNDIALSSUNErrCode(functionname(args))

  Not Collective

  Input Parameter:
. PetscFunction - any PETSc function that returns an error code

  Level: developer

.seealso: `PetscCallSUNDIALSMem()`, `PetscCallSUNDIALSVoid()`, `PetscCall()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`,
           `PetscCallHYPRE()`
M*/
#define PetscCallSUNDIALSSUNErrCode(...) \
  do { \
    PetscErrorCode ierr_sun = __VA_ARGS__; \
    if (PetscUnlikely(ierr_sun != PETSC_SUCCESS)) return (PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_sun, PETSC_ERROR_REPEAT, " "), CV_UNRECOGNIZED_ERR); \
  } while (0)

/*MC
  PetscCallSUNDIALS - Calls a SUNDIALS function and then checks the resulting error code, if it is
  non-zero it calls the error handler and returns from the current function with a PETSc error code.

  Synopsis:
  #include <petscts.h>
  PetscErrorCode PetscCallSUNDIALS(functionname, args)

  Not Collective

  Input Parameter:
+ functionname - any SUNDIALS function that returns an error code
- args         - the function arguments

  Level: beginner

.seealso: `PetscCallSUNDIALSMem()`, `PetscCallSUNDIALSVoid()`, `PetscCall()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`,
           `PetscCallHYPRE()`
M*/
#define PetscCallSUNDIALS(fn, ...) \
  do { \
    SUNErrCode ierr_sundials = fn(__VA_ARGS__); \
    PetscCheck(!ierr_sundials, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in SUNDIALS: " #fn " return code %d", ierr_sundials); \
  } while (0)

/*MC
  PetscCallSUNDIALSMem - Calls a SUNDIALS function that returns an object/array and then checks the resulting object is not `NULL`, if it is
  `NULL` it calls the error handler and returns from the current function with a PETSc error code.

  Synopsis:
  #include <petscts.h>
  PetscErrorCode PetscCallSUNDIALSMem(returnarg, functionname, args)

  Not Collective

  Input Parameter:
+ returnarg    - name of the argument that is being returned by the function
. functionname - any SUNDIALS function that returns an object/array
- args         - the function arguments

  Level: beginner

.seealso: `PetscCallSUNDIALS()`, `PetscCallSUNDIALSVoid()`, `PetscCall()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallHYPRE()`
M*/
#define PetscCallSUNDIALSMem(ret, fn, ...) \
  do { \
    ret = fn(__VA_ARGS__); \
    PetscCheck(ret, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in SUNDIALS: " #fn " unexpected NULL returned"); \
  } while (0)

/*MC
  PetscCallSUNDIALSVoid - Calls a SUNDIALS function that returns void.

  Synopsis:
  #include <petscts.h>
  void PetscCallSUNDIALSVoid(functionname, args)

  Not Collective

  Input Parameter:
+ functionname - any SUNDIALS function that returns a void
- args         - the function arguments

  Level: beginner

.seealso: `PetscCallSUNDIALS()`, `PetscCallSUNDIALSMem()`, `PetscCall()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallHYPRE()`
M*/
#define PetscCallSUNDIALSVoid(fn, ...) \
  do { \
    PetscStackPushExternal(#fn); \
    fn(__VA_ARGS__); \
    PetscStackPop; \
  } while (0)

static SUNErrCode TSCVodeRHS(PetscReal t, N_Vector x, N_Vector result, void *ptr)
{
  TS             ts = (TS)ptr;
  TSIFunctionFn *ifunction;
  DM             dm;

  PetscCallSUNDIALSSUNErrCode(TSGetDM(ts, &dm));
  PetscCallSUNDIALSSUNErrCode(DMTSGetIFunction(dm, &ifunction, NULL));
  if (!ifunction) PetscCallSUNDIALSSUNErrCode(TSComputeRHSFunction(ts, t, N_VGetVector_Petsc(x), N_VGetVector_Petsc(result)));
  else {
    TS_CVode *cvode = (TS_CVode *)ts->data;

    if (!cvode->xdot) PetscCallSUNDIALSSUNErrCode(VecDuplicate(N_VGetVector_Petsc(x), &cvode->xdot));
    /* If rhsfunction is also set, this computes both parts and shifts them to the right */
    PetscCallSUNDIALSSUNErrCode(TSComputeIFunction(ts, t, N_VGetVector_Petsc(x), cvode->xdot, N_VGetVector_Petsc(result), PETSC_FALSE));
    PetscCallSUNDIALSSUNErrCode(VecScale(N_VGetVector_Petsc(result), -1.));
  }
  return CV_SUCCESS;
}

static PetscErrorCode TSReset_CVode(TS ts)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;

  PetscFunctionBegin;
  if (cvode->cvodeInited) {
    PetscCallSUNDIALS(SUNNonlinSolFree, cvode->NLS);
    PetscCallSUNDIALSVoid(CVodeFree, &cvode->cvode_mem);
    PetscCallSUNDIALSVoid(N_VDestroy, cvode->nvecx);
    PetscCallSUNDIALS(SUNContext_Free, &cvode->sunctx);
    cvode->NLS         = NULL;
    cvode->cvode_mem   = NULL;
    cvode->nvecx       = NULL;
    cvode->sunctx      = NULL;
    cvode->cvodeInited = PETSC_FALSE;
  }
  PetscCall(VecDestroy(&cvode->constraints));
  PetscCall(VecDestroy(&cvode->xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_CVode(TS ts)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeSetConstraints_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeGetConstraints_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeSetOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeGetOrder_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_CVode(TS ts)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;
  SNES      snes;
  Mat       Jmf;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCallSUNDIALS(SUNContext_Create, PetscObjectComm((PetscObject)ts), &cvode->sunctx);
  PetscCallSUNDIALSMem(cvode->cvode_mem, CVodeCreate, cvode->cvtype, cvode->sunctx);
  PetscCallSUNDIALSMem(cvode->nvecx, N_VMake_Petsc, ts->vec_sol, cvode->sunctx);
  PetscCallSUNDIALSMem(cvode->NLS, SUNNonlinSol_PetscSNES, cvode->nvecx, ts->snes, cvode->sunctx);
  PetscCallSUNDIALS(CVodeInit, cvode->cvode_mem, TSCVodeRHS, ts->ptime, cvode->nvecx);
  cvode->cvodeInited = PETSC_TRUE;
  PetscCallSUNDIALS(CVodeSetUserData, cvode->cvode_mem, (void *)ts);
  PetscCallSUNDIALS(CVodeSStolerances, cvode->cvode_mem, ts->rtol, ts->atol);
  PetscCallSUNDIALS(CVodeSetMaxOrd, cvode->cvode_mem, cvode->order);
  if (cvode->constraints) {
    N_Vector cv_constraints;

    PetscCallSUNDIALSMem(cv_constraints, N_VMake_Petsc, cvode->constraints, cvode->sunctx);
    PetscCallSUNDIALS(CVodeSetConstraints, cvode->cvode_mem, cv_constraints);
    PetscCallSUNDIALSVoid(N_VDestroy, cv_constraints);
  }
  PetscCallSUNDIALS(CVodeSetNonlinearSolver, cvode->cvode_mem, cvode->NLS);

  PetscCall(MatCreateSNESMF(snes, &Jmf));
  PetscCall(SNESSetJacobian(snes, Jmf, Jmf, MatMFFDComputeJacobian, NULL));
  PetscCall(MatDestroy(&Jmf));
  PetscCall(SNESSetFromOptions(snes));

  PetscCallSUNDIALS(CVodeSetInitStep, cvode->cvode_mem, ts->time_step);
  if (ts->exact_final_time == TS_EXACTFINALTIME_INTERPOLATE || ts->exact_final_time == TS_EXACTFINALTIME_MATCHSTEP) PetscCallSUNDIALS(CVodeSetStopTime, cvode->cvode_mem, ts->max_time);
  PetscCallSUNDIALS(CVodeSetMaxNumSteps, cvode->cvode_mem, (long int)ts->max_steps);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_CVode(TS ts)
{
  TS_CVode  *cvode = (TS_CVode *)ts->data;
  PetscReal  oldt = ts->ptime, t = ts->ptime;
  SUNErrCode err;

  PetscFunctionBegin;
  err = CVode(cvode->cvode_mem, ts->max_time, cvode->nvecx, &t, CV_ONE_STEP);
  switch (err) {
  case CV_SUCCESS:
    break;
  case CV_TSTOP_RETURN:
    PetscInfo(ts, "SUNDIALS cvode() returned with CV_TSTOP_RETURN indicating it exactly reached the final requested time");
    break;
  case CV_ILL_INPUT:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_ILL_INPUT");
    break;
  case CV_TOO_CLOSE:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_TOO_CLOSE");
    break;
  case CV_TOO_MUCH_WORK: {
    long int  nsteps;
    PetscReal tcur;
    PetscCallSUNDIALS(CVodeGetNumSteps, cvode->cvode_mem, &nsteps);
    PetscCallSUNDIALS(CVodeGetCurrentTime, cvode->cvode_mem, &tcur);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_TOO_MUCH_WORK. At t=%g, nsteps %ld exceeds maxstep %" PetscInt_FMT ". Increase '-ts_max_steps <>' or modify TSSetMaxSteps()", (double)tcur, nsteps, ts->max_steps);
  } break;
  case CV_TOO_MUCH_ACC:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_TOO_MUCH_ACC");
    break;
  case CV_ERR_FAILURE:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_ERR_FAILURE");
    break;
  case CV_CONV_FAILURE:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_CONV_FAILURE");
    break;
  case CV_LINIT_FAIL:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_LINIT_FAIL");
    break;
  case CV_LSETUP_FAIL:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_LSETUP_FAIL");
    break;
  case CV_LSOLVE_FAIL:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_LSOLVE_FAIL");
    break;
  case CV_RHSFUNC_FAIL:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_RHSFUNC_FAIL");
    break;
  case CV_FIRST_RHSFUNC_ERR:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_FIRST_RHSFUNC_ERR");
    break;
  case CV_REPTD_RHSFUNC_ERR:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_REPTD_RHSFUNC_ERR");
    break;
  case CV_UNREC_RHSFUNC_ERR:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_UNREC_RHSFUNC_ERR");
    break;
  case CV_RTFUNC_FAIL:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_RTFUNC_FAIL");
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, SUNErrCode %d", (int)err);
  }
  ts->ptime     = t;
  ts->time_step = t - oldt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCVodeSetOrder_CVode(TS ts, PetscInt order)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;

  PetscFunctionBegin;
  if (order == cvode->order) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(order >= 1, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Order %" PetscInt_FMT " not implemented", order);
  PetscCheck(order <= cvode->max_order, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Order %" PetscInt_FMT " not implemented for TSType %s", order, ((PetscObject)ts)->type_name);
  cvode->order = order;
  if (cvode->cvodeInited) PetscCallSUNDIALS(CVodeSetMaxOrd, cvode->cvode_mem, order);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCVodeGetOrder_CVode(TS ts, PetscInt *order)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;

  PetscFunctionBegin;
  *order = cvode->order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_CVode(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;
  PetscBool flg;
  PetscInt  order;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "CVode ODE solver options");
  PetscCall(PetscOptionsInt("-ts_cvode_order", "Order of the CVode method", "TSCVodeSetOrder", cvode->order, &order, &flg));
  if (flg) PetscCall(TSCVodeSetOrder(ts, order));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_CVode(TS ts, PetscViewer viewer)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  Order=%" PetscInt_FMT "\n", cvode->order));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCVodeSetConstraints_CVode(TS ts, Vec constraints)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;

  PetscFunctionBegin;
  if (cvode->cvodeInited) {
    N_Vector cv_constraints;

    PetscCallSUNDIALSMem(cv_constraints, N_VMake_Petsc, constraints, cvode->sunctx);
    PetscCallSUNDIALS(CVodeSetConstraints, cvode->cvode_mem, cv_constraints);
    PetscCallSUNDIALSVoid(N_VDestroy, cv_constraints);
  }
  PetscCall(PetscObjectReference((PetscObject)constraints));
  PetscCall(VecDestroy(&cvode->constraints));
  cvode->constraints = constraints;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCVodeGetConstraints_CVode(TS ts, Vec *constraints)
{
  TS_CVode *cvode = (TS_CVode *)ts->data;

  PetscFunctionBegin;
  *constraints = cvode->constraints;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSCreate_CVode(TS ts, int cvtype)
{
  TS_CVode *cvode;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cvode));
  ts->data                = (void *)cvode;
  ts->usessnes            = PETSC_TRUE;
  ts->ops->setup          = TSSetUp_CVode;
  ts->ops->step           = TSStep_CVode;
  ts->ops->view           = TSView_CVode;
  ts->ops->setfromoptions = TSSetFromOptions_CVode;
  ts->ops->reset          = TSReset_CVode;
  ts->ops->destroy        = TSDestroy_CVode;
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeSetConstraints_C", TSCVodeSetConstraints_CVode));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeGetConstraints_C", TSCVodeGetConstraints_CVode));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeSetOrder_C", TSCVodeSetOrder_CVode));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSCVodeGetOrder_C", TSCVodeGetOrder_CVode));
  cvode->cvtype = cvtype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSCVODEBDF - Provides access to the SUNDIALS CVode BDF integrators

  Level: beginner

  Options Database Key:
.  -ts_cvode_order order  - Maximum order of the BDF method

  Notes:
  If used in conjunction with `TSSetIFunction()` the function provided must be of the form $U_t - F(U,t)$. That is it cannot be a DAE
  or have a mass matrix.

  Currently these must be run with `-snes_mf`.

  The default maximum order is 5.

  PETSc must be configured with the option `--download-sundials` to access these integrators.

.seealso: [](ch_ts), [](sec_sundials), `TS`, `TSCreate()`, `TSSetType()`, `TSType`, `TSBDFSetOrder()`, `TSCVODEADAMS`,
          `TSCVodeSetConstraints()`, `TSCVodeGetConstraints()`, `TSCVodeSetOrder()`, `TSCVodeGetOrder()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_CVODEBDF(TS ts)
{
  TS_CVode *cvode;

  PetscFunctionBegin;
  PetscCall(TSCreate_CVode(ts, CV_BDF));
  cvode            = (TS_CVode *)ts->data;
  cvode->order     = 5;
  cvode->max_order = 5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSCVODEADAMS - Provides access to the SUNDIALS CVode Adams integrators

  Level: beginner

  Options Database Key:
.  -ts_cvode_order order  - Maximum order of the Adams method

  Notes:
  If used in conjunction with `TSSetIFunction()` the function provided must be of the form $U_t - F(U,t)$. That is it cannot be a DAE
  or have a mass matrix.

  Currently these must be run with `-snes_mf`.

  The default maximum order is 12.

  PETSc must be configured with the option `--download-sundials` to access these integrators.

.seealso: [](ch_ts), [](sec_sundials), `TS`, `TSCreate()`, `TSSetType()`, `TSType`, `TSBDFSetOrder()`, `TSCVODEBDF`,
          `TSCVodeSetConstraints()`, `TSCVodeGetConstraints()`, `TSCVodeSetOrder()`, `TSCVodeGetOrder()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_CVODEADAMS(TS ts)
{
  TS_CVode *cvode;

  PetscFunctionBegin;
  PetscCall(TSCreate_CVode(ts, CV_ADAMS));
  cvode            = (TS_CVode *)ts->data;
  cvode->order     = 12;
  cvode->max_order = 12;
  PetscFunctionReturn(PETSC_SUCCESS);
}
