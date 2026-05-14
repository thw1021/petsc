/*
    Provides a PETSc interface to version 2.5 of SUNDIALS/CVODE solver (a very old version)
    The interface to PVODE (old version of CVODE) was originally contributed
    by Liyang Xu. It has been redone by Hong Zhang and Dinesh Kaushik.

    Reference: sundials-2.4.0/examples/cvode/parallel/cvDiurnal_kry_p.c
*/
#include <../src/ts/impls/implicit/sundials2/sundials2.h> /*I "petscts.h" I*/

/*
      TSPrecond_Sundials2 - function that we provide to SUNDIALS to
                        evaluate the preconditioner.
*/
static PetscErrorCode TSPrecond_Sundials2_Petsc(realtype tn, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr, realtype _gamma, void *P_data, N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
  TS            ts    = (TS)P_data;
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  PC            pc;
  Mat           J, P;
  Vec           yy = cvode->w1, yydot = cvode->ydot;
  PetscReal     gm = (PetscReal)_gamma;
  PetscScalar  *y_data;

  PetscFunctionBegin;
  PetscCall(TSGetIJacobian(ts, &J, &P, NULL, NULL));
  y_data = (PetscScalar *)N_VGetArrayPointer(y);
  PetscCall(VecPlaceArray(yy, y_data));
  PetscCall(VecZeroEntries(yydot)); /* The Jacobian is independent of Ydot for ODE which is all that CVode works for */
  /* compute the shifted Jacobian   (1/gm)*I + Jrest */
  PetscCall(TSComputeIJacobian(ts, ts->ptime, yy, yydot, 1 / gm, J, P, PETSC_FALSE));
  PetscCall(VecResetArray(yy));
  PetscCall(MatScale(P, gm)); /* turn into I-gm*Jrest, J is not used by SUNDIALS  */
  *jcurPtr = TRUE;
  PetscCall(TSSundials2GetPC(ts, &pc));
  PetscCall(PCSetOperators(pc, J, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SUNDIALS expects an int (*)(args...) but PetscErrorCode is an enum. Instead of switching out
   all the PetscCalls in TSPrecond_Sundials_Petsc we just wrap it */
static int TSPrecond_Sundials2_Private(realtype tn, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr, realtype _gamma, void *P_data, N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
  return (int)TSPrecond_Sundials2_Petsc(tn, y, fy, jok, jcurPtr, _gamma, P_data, vtemp1, vtemp2, vtemp3);
}

/*
     TSPSolve_Sundials2 -  routine that we provide to SUNDIALS that applies the preconditioner.
*/
static PetscErrorCode TSPSolve_Sundials2_Petsc(realtype tn, N_Vector y, N_Vector fy, N_Vector r, N_Vector z, realtype _gamma, realtype delta, int lr, void *P_data, N_Vector vtemp)
{
  TS            ts    = (TS)P_data;
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  PC            pc;
  Vec           rr = cvode->w1, zz = cvode->w2;
  PetscScalar  *r_data, *z_data;

  PetscFunctionBegin;
  /* Make the PETSc work vectors rr and zz point to the arrays in the SUNDIALS vectors r and z respectively*/
  r_data = (PetscScalar *)N_VGetArrayPointer(r);
  z_data = (PetscScalar *)N_VGetArrayPointer(z);
  PetscCall(VecPlaceArray(rr, r_data));
  PetscCall(VecPlaceArray(zz, z_data));

  /* Solve the Px=r and put the result in zz */
  PetscCall(TSSundials2GetPC(ts, &pc));
  PetscCall(PCApply(pc, rr, zz));
  PetscCall(VecResetArray(rr));
  PetscCall(VecResetArray(zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* See TSPrecond_Sundials2_Private() */
static int TSPSolve_Sundials2_Private(realtype tn, N_Vector y, N_Vector fy, N_Vector r, N_Vector z, realtype _gamma, realtype delta, int lr, void *P_data, N_Vector vtemp)
{
  return (int)TSPSolve_Sundials2_Petsc(tn, y, fy, r, z, _gamma, delta, lr, P_data, vtemp);
}

/*
        TSFunction_Sundials2 - routine that we provide to SUNDIALS that applies the right-hand side.
*/
static int TSFunction_Sundials2(realtype t, N_Vector y, N_Vector ydot, PetscCtx ctx)
{
  TS             ts = (TS)ctx;
  DM             dm;
  DMTS           tsdm;
  TSIFunctionFn *ifunction;
  MPI_Comm       comm;
  TS_Sundials2  *cvode = (TS_Sundials2 *)ts->data;
  Vec            yy = cvode->w1, yyd = cvode->w2, yydot = cvode->ydot;
  PetscScalar   *y_data, *ydot_data;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  /* Make the PETSc work vectors yy and yyd point to the arrays in the SUNDIALS vectors y and ydot respectively*/
  y_data    = (PetscScalar *)N_VGetArrayPointer(y);
  ydot_data = (PetscScalar *)N_VGetArrayPointer(ydot);
  PetscCallAbort(comm, VecPlaceArray(yy, y_data));
  PetscCallAbort(comm, VecPlaceArray(yyd, ydot_data));

  /* Now compute the right-hand side function, via IFunction unless only the more efficient RHSFunction is set */
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDMTS(dm, &tsdm));
  PetscCall(DMTSGetIFunction(dm, &ifunction, NULL));
  if (!ifunction) PetscCall(TSComputeRHSFunction(ts, t, yy, yyd));
  else {
    /* If rhsfunction is also set, this computes both parts and shifts them to the right */
    PetscCall(VecZeroEntries(yydot));
    PetscCallAbort(comm, TSComputeIFunction(ts, t, yy, yydot, yyd, PETSC_FALSE));
    PetscCall(VecScale(yyd, -1.));
  }
  PetscCallAbort(comm, VecResetArray(yy));
  PetscCallAbort(comm, VecResetArray(yyd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
       TSStep_Sundials2 - Calls SUNDIALS to integrate the ODE.
*/
static PetscErrorCode TSStep_Sundials2(TS ts)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  PetscInt      flag;
  long int      nits, lits, nsteps;
  realtype      t, tout;
  PetscScalar  *y_data;
  void         *mem;

  PetscFunctionBegin;
  mem  = cvode->mem;
  tout = ts->max_time;
  PetscCall(VecGetArray(ts->vec_sol, &y_data));
  N_VSetArrayPointer((realtype *)y_data, cvode->y);
  PetscCall(VecRestoreArray(ts->vec_sol, NULL));

  /* We would like to TSPreStage() and TSPostStage()
   * before each stage solve but CVode does not appear to support this. */
  if (cvode->monitorstep) flag = CVode(mem, tout, cvode->y, &t, CV_ONE_STEP);
  else flag = CVode(mem, tout, cvode->y, &t, CV_NORMAL);

  if (flag) { /* display error message */
    switch (flag) {
    case CV_ILL_INPUT:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_ILL_INPUT");
      break;
    case CV_TOO_CLOSE:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, CV_TOO_CLOSE");
      break;
    case CV_TOO_MUCH_WORK: {
      PetscReal tcur;
      PetscCallExternal(CVodeGetNumSteps, mem, &nsteps);
      PetscCallExternal(CVodeGetCurrentTime, mem, &tcur);
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
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "CVode() fails, flag %d", flag);
    }
  }

  /* log inner nonlinear and linear iterations */
  PetscCallExternal(CVodeGetNumNonlinSolvIters, mem, &nits);
  PetscCallExternal(CVSpilsGetNumLinIters, mem, &lits);
  ts->snes_its += nits;
  ts->ksp_its = lits;

  /* copy the solution from cvode->y to cvode->update and sol */
  PetscCall(VecPlaceArray(cvode->w1, y_data));
  PetscCall(VecCopy(cvode->w1, cvode->update));
  PetscCall(VecResetArray(cvode->w1));
  PetscCall(VecCopy(cvode->update, ts->vec_sol));

  ts->time_step = t - ts->ptime;
  ts->ptime     = t;

  PetscCallExternal(CVodeGetNumSteps, mem, &nsteps);
  if (!cvode->monitorstep) ts->steps += nsteps - 1; /* TSStep() increments the step counter by one */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_Sundials2(TS ts, PetscReal t, Vec X)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  N_Vector      y;
  PetscScalar  *x_data;
  PetscInt      glosize, locsize;

  PetscFunctionBegin;
  /* get the vector size */
  PetscCall(VecGetSize(X, &glosize));
  PetscCall(VecGetLocalSize(X, &locsize));
  PetscCall(VecGetArray(X, &x_data));

  /* Initialize N_Vec y with x_data */
  if (cvode->use_dense) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "TSSUNDIALS2 only supports a dense solve in the serial case");
    y = N_VMake_Serial(locsize, (realtype *)x_data);
  } else {
    y = N_VMake_Parallel(cvode->comm_sundials, locsize, glosize, (realtype *)x_data);
  }

  PetscCheck(y, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Interpolated y is not allocated");

  PetscCallExternal(CVodeGetDky, cvode->mem, t, 0, y);
  PetscCall(VecRestoreArray(X, &x_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_Sundials2(TS ts)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cvode->update));
  PetscCall(VecDestroy(&cvode->ydot));
  PetscCall(VecDestroy(&cvode->w1));
  PetscCall(VecDestroy(&cvode->w2));
  if (cvode->mem) CVodeFree(&cvode->mem);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_Sundials2(TS ts)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  PetscCall(TSReset_Sundials2(ts));
  PetscCallMPI(MPI_Comm_free(&cvode->comm_sundials));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetMaxl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetLinearTolerance_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetGramSchmidtType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetTolerance_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetMinTimeStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetMaxTimeStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2GetPC_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2GetIterations_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2MonitorInternalSteps_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetUseDense_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_Sundials2(TS ts)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  PetscInt      glosize, locsize, i;
  PetscScalar  *y_data, *parray;
  PC            pc;
  PCType        pctype;
  PetscBool     pcnone;

  PetscFunctionBegin;
  PetscCheck(ts->exact_final_time != TS_EXACTFINALTIME_MATCHSTEP, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for exact final time option 'MATCHSTEP' when using SUNDIALS");

  /* get the vector size */
  PetscCall(VecGetSize(ts->vec_sol, &glosize));
  PetscCall(VecGetLocalSize(ts->vec_sol, &locsize));

  /* allocate the memory for N_Vec y */
  if (cvode->use_dense) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "TSSUNDIALS2 only supports a dense solve in the serial case");
    cvode->y = N_VNew_Serial(locsize);
  } else {
    cvode->y = N_VNew_Parallel(cvode->comm_sundials, locsize, glosize);
  }
  PetscCheck(cvode->y, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "cvode->y is not allocated");

  /* initialize N_Vec y: copy ts->vec_sol to cvode->y */
  PetscCall(VecGetArray(ts->vec_sol, &parray));
  y_data = (PetscScalar *)N_VGetArrayPointer(cvode->y);
  for (i = 0; i < locsize; i++) y_data[i] = parray[i];
  PetscCall(VecRestoreArray(ts->vec_sol, NULL));

  PetscCall(VecDuplicate(ts->vec_sol, &cvode->update));
  PetscCall(VecDuplicate(ts->vec_sol, &cvode->ydot));

  /*
    Create work vectors for the TSPSolve_Sundials2() routine. Note these are
    allocated with zero space arrays because the actual array space is provided
    by SUNDIALS and set using VecPlaceArray().
  */
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ts), 1, locsize, PETSC_DECIDE, NULL, &cvode->w1));
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ts), 1, locsize, PETSC_DECIDE, NULL, &cvode->w2));

  /* Call CVodeCreate to create the solver memory and the use of a Newton iteration */
  cvode->mem = CVodeCreate(cvode->cvode_type, CV_NEWTON);
  PetscCheck(cvode->mem, PETSC_COMM_SELF, PETSC_ERR_MEM, "CVodeCreate() fails");

  /* Set the pointer to user-defined data */
  PetscCallExternal(CVodeSetUserData, cvode->mem, ts);

  /* SUNDIALS may choose to use a smaller initial step, but will never use a larger step. */
  PetscCallExternal(CVodeSetInitStep, cvode->mem, (realtype)ts->time_step);
  if (cvode->mindt > 0) {
    int flag = CVodeSetMinStep(cvode->mem, (realtype)cvode->mindt);
    if (flag) {
      PetscCheck(flag != CV_MEM_NULL, PetscObjectComm((PetscObject)ts), PETSC_ERR_LIB, "CVodeSetMinStep() failed, cvode_mem pointer is NULL");
      PetscCheck(flag != CV_ILL_INPUT, PetscObjectComm((PetscObject)ts), PETSC_ERR_LIB, "CVodeSetMinStep() failed, hmin is nonpositive or it exceeds the maximum allowable step size");
      SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_LIB, "CVodeSetMinStep() failed");
    }
  }
  if (cvode->maxdt > 0) PetscCallExternal(CVodeSetMaxStep, cvode->mem, (realtype)cvode->maxdt);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right-hand side function in u'=f(t,u), the initial time T0, and
   * the initial dependent variable vector cvode->y */
  PetscCallExternal(CVodeInit, cvode->mem, TSFunction_Sundials2, ts->ptime, cvode->y);

  /* specifies scalar relative and absolute tolerances */
  PetscCallExternal(CVodeSStolerances, cvode->mem, cvode->reltol, cvode->abstol);

  /* Specify max order of BDF / ADAMS method */
  if (cvode->maxord != PETSC_DEFAULT) PetscCallExternal(CVodeSetMaxOrd, cvode->mem, cvode->maxord);

  /* Specify max num of steps to be taken by cvode in its attempt to reach the next output time */
  PetscCallExternal(CVodeSetMaxNumSteps, cvode->mem, ts->max_steps);

  if (cvode->use_dense) PetscCallExternal(CVDense, cvode->mem, locsize);
  else {
    /* call CVSpgmr to use GMRES as the linear solver.        */
    /* setup the ode integrator with the given preconditioner */
    PetscCall(TSSundials2GetPC(ts, &pc));
    PetscCall(PCGetType(pc, &pctype));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCNONE, &pcnone));
    if (pcnone) {
      PetscCallExternal(CVSpgmr, cvode->mem, PREC_NONE, 0);
    } else {
      PetscCallExternal(CVSpgmr, cvode->mem, PREC_LEFT, cvode->maxl);

      /* Set preconditioner and solve routines Precond and PSolve,
         and the pointer to the user-defined block data */
      PetscCallExternal(CVSpilsSetPreconditioner, cvode->mem, TSPrecond_Sundials2_Private, TSPSolve_Sundials2_Private);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* type of CVODE linear multistep method */
const char *const TSSundials2LmmTypes[] = {"", "ADAMS", "BDF", "TSSundials2LmmType", "SUNDIALS2_", NULL};
/* type of G-S orthogonalization used by CVODE linear solver */
const char *const TSSundials2GramSchmidtTypes[] = {"", "MODIFIED", "CLASSICAL", "TSSundials2GramSchmidtType", "SUNDIALS2_", NULL};

static PetscErrorCode TSSetFromOptions_Sundials2(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  int           indx;
  PetscBool     flag;
  PC            pc;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SUNDIALS2 ODE solver options");
  PetscCall(PetscOptionsEList("-ts_sundials2_type", "Scheme", "TSSundials2SetType", TSSundials2LmmTypes, 3, TSSundials2LmmTypes[cvode->cvode_type], &indx, &flag));
  if (flag) PetscCall(TSSundials2SetType(ts, (TSSundials2LmmType)indx));
  PetscCall(PetscOptionsEList("-ts_sundials2_gramschmidt_type", "Type of orthogonalization", "TSSundials2SetGramSchmidtType", TSSundials2GramSchmidtTypes, 3, TSSundials2GramSchmidtTypes[cvode->gtype], &indx, &flag));
  if (flag) PetscCall(TSSundials2SetGramSchmidtType(ts, (TSSundials2GramSchmidtType)indx));
  PetscCall(PetscOptionsReal("-ts_sundials2_atol", "Absolute tolerance for convergence", "TSSundials2SetTolerance", cvode->abstol, &cvode->abstol, NULL));
  PetscCall(PetscOptionsReal("-ts_sundials2_rtol", "Relative tolerance for convergence", "TSSundials2SetTolerance", cvode->reltol, &cvode->reltol, NULL));
  PetscCall(PetscOptionsReal("-ts_sundials2_mindt", "Minimum step size", "TSSundials2SetMinTimeStep", cvode->mindt, &cvode->mindt, NULL));
  PetscCall(PetscOptionsReal("-ts_sundials2_maxdt", "Maximum step size", "TSSundials2SetMaxTimeStep", cvode->maxdt, &cvode->maxdt, NULL));
  PetscCall(PetscOptionsReal("-ts_sundials2_linear_tolerance", "Convergence tolerance for linear solve", "TSSundials2SetLinearTolerance", cvode->linear_tol, &cvode->linear_tol, NULL));
  PetscCall(PetscOptionsInt("-ts_sundials2_maxord", "Max Order for BDF/Adams method", "TSSundials2SetMaxOrd", cvode->maxord, &cvode->maxord, NULL));
  PetscCall(PetscOptionsInt("-ts_sundials2_maxl", "Max dimension of the Krylov subspace", "TSSundials2SetMaxl", cvode->maxl, &cvode->maxl, NULL));
  PetscCall(PetscOptionsBool("-ts_sundials2_monitor_steps", "Monitor SUNDIALS2 internal steps", "TSSundials2MonitorInternalSteps", cvode->monitorstep, &cvode->monitorstep, NULL));
  PetscCall(PetscOptionsBool("-ts_sundials2_use_dense", "Use dense internal solver in SUNDIALS2 (serial only)", "TSSundials2SetUseDense", cvode->use_dense, &cvode->use_dense, NULL));
  PetscOptionsHeadEnd();
  PetscCall(TSSundials2GetPC(ts, &pc));
  PetscCall(PCSetFromOptions(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_Sundials2(TS ts, PetscViewer viewer)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;
  char         *type;
  char          atype[] = "Adams";
  char          btype[] = "BDF: backward differentiation formula";
  PetscBool     isascii, isstring;
  long int      nsteps, its, nfevals, nlinsetups, nfails, itmp;
  PetscInt      qlast, qcur;
  PetscReal     hinused, hlast, hcur, tcur, tolsfac;
  PC            pc;

  PetscFunctionBegin;
  if (cvode->cvode_type == SUNDIALS2_ADAMS) type = atype;
  else type = btype;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 integrator does not use SNES!\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 integrator type %s\n", type));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 maxord %" PetscInt_FMT "\n", cvode->maxord));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 abs tol %g rel tol %g\n", (double)cvode->abstol, (double)cvode->reltol));
    if (cvode->use_dense) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 integrator using a dense linear solve\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 linear solver tolerance factor %g\n", (double)cvode->linear_tol));
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 max dimension of Krylov subspace %" PetscInt_FMT "\n", cvode->maxl));
      if (cvode->gtype == SUNDIALS2_MODIFIED_GS) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 using modified Gram-Schmidt for orthogonalization in GMRES\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 using unmodified (classical) Gram-Schmidt for orthogonalization in GMRES\n"));
      }
    }
    if (cvode->mindt > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 minimum time step %g\n", (double)cvode->mindt));
    if (cvode->maxdt > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 maximum time step %g\n", (double)cvode->maxdt));

    /* Outputs from CVODE, CVSPILS */
    PetscCallExternal(CVodeGetTolScaleFactor, cvode->mem, &tolsfac);
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 suggested factor for tolerance scaling %g\n", tolsfac));
    PetscCallExternal(CVodeGetIntegratorStats, cvode->mem, &nsteps, &nfevals, &nlinsetups, &nfails, &qlast, &qcur, &hinused, &hlast, &hcur, &tcur);
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 cumulative number of internal steps %ld\n", nsteps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of calls to rhs function %ld\n", nfevals));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of calls to linear solver setup function %ld\n", nlinsetups));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of error test failures %ld\n", nfails));

    PetscCallExternal(CVodeGetNonlinSolvStats, cvode->mem, &its, &nfails);
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of nonlinear solver iterations %ld\n", its));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of nonlinear convergence failure %ld\n", nfails));
    if (!cvode->use_dense) {
      PetscCallExternal(CVSpilsGetNumLinIters, cvode->mem, &its); /* its = no. of calls to TSPrecond_Sundials2() */
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of linear iterations %ld\n", its));
      PetscCallExternal(CVSpilsGetNumConvFails, cvode->mem, &itmp);
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of linear convergence failures %ld\n", itmp));

      PetscCall(TSSundials2GetPC(ts, &pc));
      PetscCall(PCView(pc, viewer));
      PetscCallExternal(CVSpilsGetNumPrecEvals, cvode->mem, &itmp);
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of preconditioner evaluations %ld\n", itmp));
      PetscCallExternal(CVSpilsGetNumPrecSolves, cvode->mem, &itmp);
      PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of preconditioner solves %ld\n", itmp));
    }
    PetscCallExternal(CVSpilsGetNumJtimesEvals, cvode->mem, &itmp);
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of Jacobian-vector product evaluations %ld\n", itmp));
    PetscCallExternal(CVSpilsGetNumRhsEvals, cvode->mem, &itmp);
    PetscCall(PetscViewerASCIIPrintf(viewer, "SUNDIALS2 no. of rhs calls for finite diff. Jacobian-vector evals %ld\n", itmp));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer, "SUNDIALS2 type %s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetType_Sundials2(TS ts, TSSundials2LmmType type)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->cvode_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetMaxl_Sundials2(TS ts, PetscInt maxl)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->maxl = maxl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetLinearTolerance_Sundials2(TS ts, PetscReal tol)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->linear_tol = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetGramSchmidtType_Sundials2(TS ts, TSSundials2GramSchmidtType type)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->gtype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetTolerance_Sundials2(TS ts, PetscReal aabs, PetscReal rel)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  if (aabs != PETSC_DECIDE) cvode->abstol = aabs;
  if (rel != PETSC_DECIDE) cvode->reltol = rel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetMinTimeStep_Sundials2(TS ts, PetscReal mindt)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->mindt = mindt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetMaxTimeStep_Sundials2(TS ts, PetscReal maxdt)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->maxdt = maxdt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2SetUseDense_Sundials2(TS ts, PetscBool use_dense)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->use_dense = use_dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2GetPC_Sundials2(TS ts, PC *pc)
{
  SNES snes;
  KSP  ksp;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2GetIterations_Sundials2(TS ts, int *nonlin, int *lin)
{
  PetscFunctionBegin;
  if (nonlin) *nonlin = ts->snes_its;
  if (lin) *lin = ts->ksp_its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSundials2MonitorInternalSteps_Sundials2(TS ts, PetscBool s)
{
  TS_Sundials2 *cvode = (TS_Sundials2 *)ts->data;

  PetscFunctionBegin;
  cvode->monitorstep = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2GetIterations - Gets the number of nonlinear and linear iterations used so far by `TSSUNDIALS2`.

  Not Collective

  Input Parameter:
. ts - the time-step context

  Output Parameters:
+ nonlin - number of nonlinear iterations
- lin    - number of linear iterations

  Level: advanced

  Note:
  These return the number since the creation of the `TS` object

.seealso: [](ch_ts), `TSSundials2SetType()`, `TSSundials2SetMaxl()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`, `TSSundials2SetTolerance()`,
          `TSSundials2GetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2GetIterations(TS ts, int *nonlin, int *lin)
{
  PetscFunctionBegin;
  PetscUseMethod(ts, "TSSundials2GetIterations_C", (TS, int *, int *), (ts, nonlin, lin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetType - Sets the method that `TSSUNDIALS2` will use for integration.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
- type - one of  `SUNDIALS2_ADAMS` or `SUNDIALS2_BDF`

  Level: intermediate

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetMaxl()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`,
          `TSSundials2SetTolerance()`, `TSSundials2GetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2SetType(TS ts, TSSundials2LmmType type)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSSundials2SetType_C", (TS, TSSundials2LmmType), (ts, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetMaxord - Sets the maximum order for BDF/Adams method used by `TSSUNDIALS2`.

  Logically Collective

  Input Parameters:
+ ts     - the time-step context
- maxord - maximum order of BDF / Adams method

  Level: advanced

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`,
          `TSSundials2SetTolerance()`, `TSSundials2GetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2SetMaxord(TS ts, PetscInt maxord)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts, maxord, 2);
  PetscTryMethod(ts, "TSSundials2SetMaxOrd_C", (TS, PetscInt), (ts, maxord));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetMaxl - Sets the dimension of the Krylov space used by
  GMRES in the linear solver in `TSSUNDIALS2`. `TSSUNDIALS2` DOES NOT use restarted GMRES so
  this is the maximum number of GMRES steps that will be used.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
- maxl - number of direction vectors (the dimension of Krylov subspace).

  Level: advanced

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`, `TSSundials2SetTolerance()`,
          `TSSundials2GetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2SetMaxl(TS ts, PetscInt maxl)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts, maxl, 2);
  PetscTryMethod(ts, "TSSundials2SetMaxl_C", (TS, PetscInt), (ts, maxl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetLinearTolerance - Sets the tolerance used to solve the linear
  system by `TSSUNDIALS2`.

  Logically Collective

  Input Parameters:
+ ts  - the time-step context
- tol - the factor by which the tolerance on the nonlinear solver is
             multiplied to get the tolerance on the linear solver, .05 by default.

  Level: advanced

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`, `TSSundials2SetMaxl()`,
          `TSSundials2SetGramSchmidtType()`, `TSSundials2SetTolerance()`,
          `TSSundials2GetPC()`,
          `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2SetLinearTolerance(TS ts, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ts, tol, 2);
  PetscTryMethod(ts, "TSSundials2SetLinearTolerance_C", (TS, PetscReal), (ts, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetGramSchmidtType - Sets type of orthogonalization used
  in GMRES method by `TSSUNDIALS2` linear solver.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
- type - either `SUNDIALS2_MODIFIED_GS` or `SUNDIALS2_CLASSICAL_GS`

  Level: advanced

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`, `TSSundials2SetMaxl()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetTolerance()`,
          `TSSundials2GetPC()`,
          `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2SetGramSchmidtType(TS ts, TSSundials2GramSchmidtType type)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSSundials2SetGramSchmidtType_C", (TS, TSSundials2GramSchmidtType), (ts, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetTolerance - Sets the absolute and relative tolerance used by
  `TSSUNDIALS2` for error control.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
. aabs - the absolute tolerance
- rel  - the relative tolerance

     See the CVODE/SUNDIALS users manual for exact details on these parameters. Essentially
    these regulate the size of the error for a SINGLE timestep.

  Level: intermediate

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`, `TSSundials2SetGMRESMaxl()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`,
          `TSSundials2GetPC()`,
          `TSSetExactFinalTime()`
@*/
PetscErrorCode TSSundials2SetTolerance(TS ts, PetscReal aabs, PetscReal rel)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSSundials2SetTolerance_C", (TS, PetscReal, PetscReal), (ts, aabs, rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2GetPC - Extract the PC context from a time-step context for `TSSUNDIALS2`.

  Input Parameter:
. ts - the time-step context

  Output Parameter:
. pc - the preconditioner context

  Level: advanced

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`, `TSSundials2SetMaxl()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`, `TSSundials2SetTolerance()`
@*/
PetscErrorCode TSSundials2GetPC(TS ts, PC *pc)
{
  PetscFunctionBegin;
  PetscUseMethod(ts, "TSSundials2GetPC_C", (TS, PC *), (ts, pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetMinTimeStep - Smallest time step to be chosen by the adaptive controller.

  Input Parameters:
+ ts    - the time-step context
- mindt - lowest time step if positive, negative to deactivate

  Note:
  `TSSUNDIALS2` will error if it is not possible to keep the estimated truncation error below
  the tolerance set with `TSSundials2SetTolerance()` without going below this step size.

  Level: beginner

.seealso: [](ch_ts), `TSSundials2SetType()`, `TSSundials2SetTolerance()`
@*/
PetscErrorCode TSSundials2SetMinTimeStep(TS ts, PetscReal mindt)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSSundials2SetMinTimeStep_C", (TS, PetscReal), (ts, mindt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetMaxTimeStep - Largest time step to be chosen by the adaptive controller.

  Input Parameters:
+ ts    - the time-step context
- maxdt - lowest time step if positive, negative to deactivate

  Level: beginner

.seealso: [](ch_ts), `TSSundials2SetType()`, `TSSundials2SetTolerance()`
@*/
PetscErrorCode TSSundials2SetMaxTimeStep(TS ts, PetscReal maxdt)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSSundials2SetMaxTimeStep_C", (TS, PetscReal), (ts, maxdt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2MonitorInternalSteps - Monitor `TSSUNDIALS2` internal steps (Defaults to false).

  Input Parameters:
+ ts - the time-step context
- ft - `PETSC_TRUE` if monitor, else `PETSC_FALSE`

  Level: beginner

.seealso: [](ch_ts), `TSSundials2GetIterations()`, `TSSundials2SetType()`, `TSSundials2SetMaxl()`,
          `TSSundials2SetLinearTolerance()`, `TSSundials2SetGramSchmidtType()`, `TSSundials2SetTolerance()`,
          `TSSundials2GetPC()`
@*/
PetscErrorCode TSSundials2MonitorInternalSteps(TS ts, PetscBool ft)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSSundials2MonitorInternalSteps_C", (TS, PetscBool), (ts, ft));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSSundials2SetUseDense - Set a flag to use a dense linear solver in `TSSUNDIALS2` (serial only)

  Logically Collective

  Input Parameters:
+ ts        - the time-step context
- use_dense - `PETSC_TRUE` to use the dense solver

  Level: advanced

.seealso: [](ch_ts), `TSSUNDIALS2`
@*/
PetscErrorCode TSSundials2SetUseDense(TS ts, PetscBool use_dense)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts, use_dense, 2);
  PetscTryMethod(ts, "TSSundials2SetUseDense_C", (TS, PetscBool), (ts, use_dense));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
      TSSUNDIALS2 - ODE solver using a very old version of the LLNL CVODE/SUNDIALS package, version 2.5. Requires ./configure --download-sundials2

   Options Database Keys:
+    -ts_sundials2_type (bdf|adams)                      - integrator type
.    -ts_sundials2_gramschmidt_type (modified|classical) - type of orthogonalization inside GMRES
.    -ts_sundials2_atol atol                             - Absolute tolerance for convergence
.    -ts_sundials2_rtol rtol                             - Relative tolerance for convergence
.    -ts_sundials2_linear_tolerance ltol                 - convergence tolerance for linear solver
.    -ts_sundials2_maxl maxl                             - Max dimension of the Krylov subspace
.    -ts_sundials2_monitor_steps                         - Monitor SUNDIALS2 internal steps
-    -ts_sundials2_use_dense                             - Use a dense linear solver within CVODE (serial only)

    Level: beginner

    Note:
    This uses its own nonlinear solver and Krylov method so PETSc `SNES` and `KSP` options do not apply,
    only PETSc `PC` options.

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSSundials2SetType()`, `TSSundials2SetMaxl()`, `TSSundials2SetLinearTolerance()`, `TSType`,
          `TSSundials2SetGramSchmidtType()`, `TSSundials2SetTolerance()`, `TSSundials2GetPC()`, `TSSundials2GetIterations()`, `TSSetExactFinalTime()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_Sundials2(TS ts)
{
  TS_Sundials2 *cvode;
  PC            pc;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Sundials2;
  ts->ops->destroy        = TSDestroy_Sundials2;
  ts->ops->view           = TSView_Sundials2;
  ts->ops->setup          = TSSetUp_Sundials2;
  ts->ops->step           = TSStep_Sundials2;
  ts->ops->interpolate    = TSInterpolate_Sundials2;
  ts->ops->setfromoptions = TSSetFromOptions_Sundials2;
  ts->default_adapt_type  = TSADAPTNONE;

  PetscCall(PetscNew(&cvode));

  ts->usessnes = PETSC_TRUE;

  ts->data           = (void *)cvode;
  cvode->cvode_type  = SUNDIALS2_BDF;
  cvode->gtype       = SUNDIALS2_CLASSICAL_GS;
  cvode->maxl        = 5;
  cvode->maxord      = PETSC_DEFAULT;
  cvode->linear_tol  = .05;
  cvode->monitorstep = PETSC_TRUE;
  cvode->use_dense   = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)ts), &cvode->comm_sundials));

  cvode->mindt = -1.;
  cvode->maxdt = -1.;

  /* set tolerance for SUNDIALS2 */
  cvode->reltol = 1e-6;
  cvode->abstol = 1e-6;

  /* set PCNONE as default pctype */
  PetscCall(TSSundials2GetPC_Sundials2(ts, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetType_C", TSSundials2SetType_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetMaxl_C", TSSundials2SetMaxl_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetLinearTolerance_C", TSSundials2SetLinearTolerance_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetGramSchmidtType_C", TSSundials2SetGramSchmidtType_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetTolerance_C", TSSundials2SetTolerance_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetMinTimeStep_C", TSSundials2SetMinTimeStep_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetMaxTimeStep_C", TSSundials2SetMaxTimeStep_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2GetPC_C", TSSundials2GetPC_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2GetIterations_C", TSSundials2GetIterations_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2MonitorInternalSteps_C", TSSundials2MonitorInternalSteps_Sundials2));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSSundials2SetUseDense_C", TSSundials2SetUseDense_Sundials2));
  PetscFunctionReturn(PETSC_SUCCESS);
}
