/*
    Provides a PETSc interface to version 2.5 of SUNDIALS/CVODE solver (a very old version)
    The interface to PVODE (old version of CVODE) was originally contributed
    by Liyang Xu. It has been redone by Hong Zhang and Dinesh Kaushik.

    Reference: sundials-2.4.0/examples/cvode/parallel/cvDiurnal_kry_p.c
*/
#include <../src/ts/impls/implicit/sundials2/sundials2.h> /*I "petscts.h" I*/

/*
      TSPrecond_AncientSundials - function that we provide to SUNDIALS to
                        evaluate the preconditioner.
*/
static PetscErrorCode TSPrecond_AncientSundials_Petsc(realtype tn, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr, realtype _gamma, void *P_data, N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
  TS                  ts    = (TS)P_data;
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  PC                  pc;
  Mat                 J, P;
  Vec                 yy = cvode->w1, yydot = cvode->ydot;
  PetscReal           gm = (PetscReal)_gamma;
  PetscScalar        *y_data;

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
  PetscCall(TSAncientSundialsGetPC(ts, &pc));
  PetscCall(PCSetOperators(pc, J, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SUNDIALS expects an int (*)(args...) but PetscErrorCode is an enum. Instead of switching out
   all the PetscCalls in TSPrecond_Sundials_Petsc we just wrap it */
static int TSPrecond_AncientSundials_Private(realtype tn, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr, realtype _gamma, void *P_data, N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
  return (int)TSPrecond_AncientSundials_Petsc(tn, y, fy, jok, jcurPtr, _gamma, P_data, vtemp1, vtemp2, vtemp3);
}

/*
     TSPSolve_AncientSundials -  routine that we provide to SUNDIALS that applies the preconditioner.
*/
static PetscErrorCode TSPSolve_AncientSundials_Petsc(realtype tn, N_Vector y, N_Vector fy, N_Vector r, N_Vector z, realtype _gamma, realtype delta, int lr, void *P_data, N_Vector vtemp)
{
  TS                  ts    = (TS)P_data;
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  PC                  pc;
  Vec                 rr = cvode->w1, zz = cvode->w2;
  PetscScalar        *r_data, *z_data;

  PetscFunctionBegin;
  /* Make the PETSc work vectors rr and zz point to the arrays in the SUNDIALS vectors r and z respectively*/
  r_data = (PetscScalar *)N_VGetArrayPointer(r);
  z_data = (PetscScalar *)N_VGetArrayPointer(z);
  PetscCall(VecPlaceArray(rr, r_data));
  PetscCall(VecPlaceArray(zz, z_data));

  /* Solve the Px=r and put the result in zz */
  PetscCall(TSAncientSundialsGetPC(ts, &pc));
  PetscCall(PCApply(pc, rr, zz));
  PetscCall(VecResetArray(rr));
  PetscCall(VecResetArray(zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* See TSPrecond_AncientSundials_Private() */
static int TSPSolve_AncientSundials_Private(realtype tn, N_Vector y, N_Vector fy, N_Vector r, N_Vector z, realtype _gamma, realtype delta, int lr, void *P_data, N_Vector vtemp)
{
  return (int)TSPSolve_AncientSundials_Petsc(tn, y, fy, r, z, _gamma, delta, lr, P_data, vtemp);
}

/*
        TSFunction_AncientSundials - routine that we provide to SUNDIALS that applies the right-hand side.
*/
static int TSFunction_AncientSundials(realtype t, N_Vector y, N_Vector ydot, PetscCtx ctx)
{
  TS                  ts = (TS)ctx;
  DM                  dm;
  DMTS                tsdm;
  TSIFunctionFn      *ifunction;
  MPI_Comm            comm;
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  Vec                 yy = cvode->w1, yyd = cvode->w2, yydot = cvode->ydot;
  PetscScalar        *y_data, *ydot_data;

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
       TSStep_AncientSundials - Calls SUNDIALS to integrate the ODE.
*/
static PetscErrorCode TSStep_AncientSundials(TS ts)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  PetscInt            flag;
  long int            nits, lits, nsteps;
  realtype            t, tout;
  PetscScalar        *y_data;
  void               *mem;

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
  ts->ksp_its = (PetscInt)lits;

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

static PetscErrorCode TSInterpolate_AncientSundials(TS ts, PetscReal t, Vec X)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  N_Vector            y;
  PetscScalar        *x_data;
  PetscInt            glosize, locsize;

  PetscFunctionBegin;
  /* get the vector size */
  PetscCall(VecGetSize(X, &glosize));
  PetscCall(VecGetLocalSize(X, &locsize));
  PetscCall(VecGetArray(X, &x_data));

  /* Initialize N_Vec y with x_data */
  if (cvode->use_dense) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "TSANCIENTSUNDIALS only supports a dense solve in the serial case");
    y = N_VMake_Serial(locsize, (realtype *)x_data);
  } else {
    y = N_VMake_Parallel(cvode->comm_sundials, locsize, glosize, (realtype *)x_data);
  }

  PetscCheck(y, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Interpolated y is not allocated");

  PetscCallExternal(CVodeGetDky, cvode->mem, t, 0, y);
  PetscCall(VecRestoreArray(X, &x_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_AncientSundials(TS ts)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cvode->update));
  PetscCall(VecDestroy(&cvode->ydot));
  PetscCall(VecDestroy(&cvode->w1));
  PetscCall(VecDestroy(&cvode->w2));
  if (cvode->mem) CVodeFree(&cvode->mem);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_AncientSundials(TS ts)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  PetscCall(TSReset_AncientSundials(ts));
  PetscCallMPI(MPI_Comm_free(&cvode->comm_sundials));
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetMaxl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetLinearTolerance_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetGramSchmidtType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetTolerance_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetMinTimeStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetMaxTimeStep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsGetPC_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsGetIterations_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsMonitorInternalSteps_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetUseDense_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_AncientSundials(TS ts)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  PetscInt            glosize, locsize, i;
  PetscScalar        *y_data, *parray;
  PC                  pc;
  PCType              pctype;
  PetscBool           pcnone;

  PetscFunctionBegin;
  PetscCheck(ts->exact_final_time != TS_EXACTFINALTIME_MATCHSTEP, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for exact final time option 'MATCHSTEP' when using SUNDIALS version 2.5");

  /* get the vector size */
  PetscCall(VecGetSize(ts->vec_sol, &glosize));
  PetscCall(VecGetLocalSize(ts->vec_sol, &locsize));

  /* allocate the memory for N_Vec y */
  if (cvode->use_dense) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "TSANCIENTSUNDIALS only supports a dense solve in the serial case");
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
    Create work vectors for the TSPSolve_AncientSundials() routine. Note these are
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
  PetscCallExternal(CVodeInit, cvode->mem, TSFunction_AncientSundials, ts->ptime, cvode->y);

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
    PetscCall(TSAncientSundialsGetPC(ts, &pc));
    PetscCall(PCGetType(pc, &pctype));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCNONE, &pcnone));
    if (pcnone) {
      PetscCallExternal(CVSpgmr, cvode->mem, PREC_NONE, 0);
    } else {
      PetscCallExternal(CVSpgmr, cvode->mem, PREC_LEFT, cvode->maxl);

      /* Set preconditioner and solve routines Precond and PSolve,
         and the pointer to the user-defined block data */
      PetscCallExternal(CVSpilsSetPreconditioner, cvode->mem, TSPrecond_AncientSundials_Private, TSPSolve_AncientSundials_Private);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* type of CVODE linear multistep method */
const char *const TSAncientSundialsLmmTypes[] = {"", "ADAMS", "BDF", "TSAncientSundialsLmmType", "ANCIENTSUNDIALS_", NULL};
/* type of G-S orthogonalization used by CVODE linear solver */
const char *const TSAncientSundialsGramSchmidtTypes[] = {"", "MODIFIED", "CLASSICAL", "TSAncientSundialsGramSchmidtType", "ANCIENTSUNDIALS_", NULL};

static PetscErrorCode TSSetFromOptions_AncientSundials(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  int                 indx;
  PetscBool           flag;
  PC                  pc;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "ANCIENTSUNDIALS ODE solver options");
  PetscCall(PetscOptionsEList("-ts_ancientsundials_type", "Scheme", "TSAncientSundialsSetType", TSAncientSundialsLmmTypes, 3, TSAncientSundialsLmmTypes[cvode->cvode_type], &indx, &flag));
  if (flag) PetscCall(TSAncientSundialsSetType(ts, (TSAncientSundialsLmmType)indx));
  PetscCall(PetscOptionsEList("-ts_ancientsundials_gramschmidt_type", "Type of orthogonalization", "TSAncientSundialsSetGramSchmidtType", TSAncientSundialsGramSchmidtTypes, 3, TSAncientSundialsGramSchmidtTypes[cvode->gtype], &indx, &flag));
  if (flag) PetscCall(TSAncientSundialsSetGramSchmidtType(ts, (TSAncientSundialsGramSchmidtType)indx));
  PetscCall(PetscOptionsReal("-ts_ancientsundials_atol", "Absolute tolerance for convergence", "TSAncientSundialsSetTolerance", cvode->abstol, &cvode->abstol, NULL));
  PetscCall(PetscOptionsReal("-ts_ancientsundials_rtol", "Relative tolerance for convergence", "TSAncientSundialsSetTolerance", cvode->reltol, &cvode->reltol, NULL));
  PetscCall(PetscOptionsReal("-ts_ancientsundials_mindt", "Minimum step size", "TSAncientSundialsSetMinTimeStep", cvode->mindt, &cvode->mindt, NULL));
  PetscCall(PetscOptionsReal("-ts_ancientsundials_maxdt", "Maximum step size", "TSAncientSundialsSetMaxTimeStep", cvode->maxdt, &cvode->maxdt, NULL));
  PetscCall(PetscOptionsReal("-ts_ancientsundials_linear_tolerance", "Convergence tolerance for linear solve", "TSAncientSundialsSetLinearTolerance", cvode->linear_tol, &cvode->linear_tol, NULL));
  PetscCall(PetscOptionsInt("-ts_ancientsundials_maxord", "Max Order for BDF/Adams method", "TSAncientSundialsSetMaxOrd", cvode->maxord, &cvode->maxord, NULL));
  PetscCall(PetscOptionsInt("-ts_ancientsundials_maxl", "Max dimension of the Krylov subspace", "TSAncientSundialsSetMaxl", cvode->maxl, &cvode->maxl, NULL));
  PetscCall(PetscOptionsBool("-ts_ancientsundials_monitor_steps", "Monitor ANCIENTSUNDIALS internal steps", "TSAncientSundialsMonitorInternalSteps", cvode->monitorstep, &cvode->monitorstep, NULL));
  PetscCall(PetscOptionsBool("-ts_ancientsundials_use_dense", "Use dense internal solver in ANCIENTSUNDIALS (serial only)", "TSAncientSundialsSetUseDense", cvode->use_dense, &cvode->use_dense, NULL));
  PetscOptionsHeadEnd();
  PetscCall(TSAncientSundialsGetPC(ts, &pc));
  PetscCall(PCSetFromOptions(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_AncientSundials(TS ts, PetscViewer viewer)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;
  char               *type;
  char                atype[] = "Adams";
  char                btype[] = "BDF: backward differentiation formula";
  PetscBool           isascii, isstring;
  long int            nsteps, its, nfevals, nlinsetups, nfails, itmp;
  PetscInt            qlast, qcur;
  PetscReal           hinused, hlast, hcur, tcur, tolsfac;
  PC                  pc;

  PetscFunctionBegin;
  if (cvode->cvode_type == ANCIENTSUNDIALS_ADAMS) type = atype;
  else type = btype;

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS integrator does not use SNES!\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS integrator type %s\n", type));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS maxord %" PetscInt_FMT "\n", cvode->maxord));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS abs tol %g rel tol %g\n", (double)cvode->abstol, (double)cvode->reltol));
    if (cvode->use_dense) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS integrator using a dense linear solve\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS linear solver tolerance factor %g\n", (double)cvode->linear_tol));
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS max dimension of Krylov subspace %" PetscInt_FMT "\n", cvode->maxl));
      if (cvode->gtype == ANCIENTSUNDIALS_MODIFIED_GS) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS using modified Gram-Schmidt for orthogonalization in GMRES\n"));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS using unmodified (classical) Gram-Schmidt for orthogonalization in GMRES\n"));
      }
    }
    if (cvode->mindt > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS minimum time step %g\n", (double)cvode->mindt));
    if (cvode->maxdt > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS maximum time step %g\n", (double)cvode->maxdt));

    /* Outputs from CVODE, CVSPILS */
    PetscCallExternal(CVodeGetTolScaleFactor, cvode->mem, &tolsfac);
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS suggested factor for tolerance scaling %g\n", tolsfac));
    PetscCallExternal(CVodeGetIntegratorStats, cvode->mem, &nsteps, &nfevals, &nlinsetups, &nfails, &qlast, &qcur, &hinused, &hlast, &hcur, &tcur);
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS cumulative number of internal steps %ld\n", nsteps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of calls to rhs function %ld\n", nfevals));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of calls to linear solver setup function %ld\n", nlinsetups));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of error test failures %ld\n", nfails));

    PetscCallExternal(CVodeGetNonlinSolvStats, cvode->mem, &its, &nfails);
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of nonlinear solver iterations %ld\n", its));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of nonlinear convergence failure %ld\n", nfails));
    if (!cvode->use_dense) {
      PetscCallExternal(CVSpilsGetNumLinIters, cvode->mem, &its); /* its = no. of calls to TSPrecond_AncientSundials() */
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of linear iterations %ld\n", its));
      PetscCallExternal(CVSpilsGetNumConvFails, cvode->mem, &itmp);
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of linear convergence failures %ld\n", itmp));

      PetscCall(TSAncientSundialsGetPC(ts, &pc));
      PetscCall(PCView(pc, viewer));
      PetscCallExternal(CVSpilsGetNumPrecEvals, cvode->mem, &itmp);
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of preconditioner evaluations %ld\n", itmp));
      PetscCallExternal(CVSpilsGetNumPrecSolves, cvode->mem, &itmp);
      PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of preconditioner solves %ld\n", itmp));
    }
    PetscCallExternal(CVSpilsGetNumJtimesEvals, cvode->mem, &itmp);
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of Jacobian-vector product evaluations %ld\n", itmp));
    PetscCallExternal(CVSpilsGetNumRhsEvals, cvode->mem, &itmp);
    PetscCall(PetscViewerASCIIPrintf(viewer, "ANCIENTSUNDIALS no. of rhs calls for finite diff. Jacobian-vector evals %ld\n", itmp));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer, "ANCIENTSUNDIALS type %s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetType_AncientSundials(TS ts, TSAncientSundialsLmmType type)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->cvode_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetMaxl_AncientSundials(TS ts, PetscInt maxl)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->maxl = maxl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetLinearTolerance_AncientSundials(TS ts, PetscReal tol)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->linear_tol = tol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetGramSchmidtType_AncientSundials(TS ts, TSAncientSundialsGramSchmidtType type)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->gtype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetTolerance_AncientSundials(TS ts, PetscReal aabs, PetscReal rel)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  if (aabs != PETSC_DECIDE) cvode->abstol = aabs;
  if (rel != PETSC_DECIDE) cvode->reltol = rel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetMinTimeStep_AncientSundials(TS ts, PetscReal mindt)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->mindt = mindt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetMaxTimeStep_AncientSundials(TS ts, PetscReal maxdt)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->maxdt = maxdt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsSetUseDense_AncientSundials(TS ts, PetscBool use_dense)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->use_dense = use_dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsGetPC_AncientSundials(TS ts, PC *pc)
{
  SNES snes;
  KSP  ksp;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsGetIterations_AncientSundials(TS ts, int *nonlin, int *lin)
{
  PetscFunctionBegin;
  if (nonlin) *nonlin = ts->snes_its;
  if (lin) *lin = ts->ksp_its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSAncientSundialsMonitorInternalSteps_AncientSundials(TS ts, PetscBool s)
{
  TS_AncientSundials *cvode = (TS_AncientSundials *)ts->data;

  PetscFunctionBegin;
  cvode->monitorstep = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsGetIterations - Gets the number of nonlinear and linear iterations used so far by `TSANCIENTSUNDIALS`.

  Not Collective

  Input Parameter:
. ts - the time-step context

  Output Parameters:
+ nonlin - number of nonlinear iterations
- lin    - number of linear iterations

  Level: advanced

  Note:
  These return the number since the creation of the `TS` object

.seealso: [](ch_ts), `TSAncientSundialsSetType()`, `TSAncientSundialsSetMaxl()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`, `TSAncientSundialsSetTolerance()`,
          `TSAncientSundialsGetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsGetIterations(TS ts, int *nonlin, int *lin)
{
  PetscFunctionBegin;
  PetscUseMethod(ts, "TSAncientSundialsGetIterations_C", (TS, int *, int *), (ts, nonlin, lin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetType - Sets the method that `TSANCIENTSUNDIALS` will use for integration.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
- type - one of  `ANCIENTSUNDIALS_ADAMS` or `ANCIENTSUNDIALS_BDF`

  Level: intermediate

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetMaxl()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`,
          `TSAncientSundialsSetTolerance()`, `TSAncientSundialsGetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsSetType(TS ts, TSAncientSundialsLmmType type)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSAncientSundialsSetType_C", (TS, TSAncientSundialsLmmType), (ts, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetMaxord - Sets the maximum order for BDF/Adams method used by `TSANCIENTSUNDIALS`.

  Logically Collective

  Input Parameters:
+ ts     - the time-step context
- maxord - maximum order of BDF / Adams method

  Level: advanced

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`,
          `TSAncientSundialsSetTolerance()`, `TSAncientSundialsGetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsSetMaxord(TS ts, PetscInt maxord)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts, maxord, 2);
  PetscTryMethod(ts, "TSAncientSundialsSetMaxOrd_C", (TS, PetscInt), (ts, maxord));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetMaxl - Sets the dimension of the Krylov space used by
  GMRES in the linear solver in `TSANCIENTSUNDIALS`. `TSANCIENTSUNDIALS` DOES NOT use restarted GMRES so
  this is the maximum number of GMRES steps that will be used.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
- maxl - number of direction vectors (the dimension of Krylov subspace).

  Level: advanced

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`, `TSAncientSundialsSetTolerance()`,
          `TSAncientSundialsGetPC()`, `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsSetMaxl(TS ts, PetscInt maxl)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts, maxl, 2);
  PetscTryMethod(ts, "TSAncientSundialsSetMaxl_C", (TS, PetscInt), (ts, maxl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetLinearTolerance - Sets the tolerance used to solve the linear
  system by `TSANCIENTSUNDIALS`.

  Logically Collective

  Input Parameters:
+ ts  - the time-step context
- tol - the factor by which the tolerance on the nonlinear solver is
             multiplied to get the tolerance on the linear solver, .05 by default.

  Level: advanced

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`, `TSAncientSundialsSetMaxl()`,
          `TSAncientSundialsSetGramSchmidtType()`, `TSAncientSundialsSetTolerance()`,
          `TSAncientSundialsGetPC()`,
          `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsSetLinearTolerance(TS ts, PetscReal tol)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveReal(ts, tol, 2);
  PetscTryMethod(ts, "TSAncientSundialsSetLinearTolerance_C", (TS, PetscReal), (ts, tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetGramSchmidtType - Sets type of orthogonalization used
  in GMRES method by `TSANCIENTSUNDIALS` linear solver.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
- type - either `ANCIENTSUNDIALS_MODIFIED_GS` or `ANCIENTSUNDIALS_CLASSICAL_GS`

  Level: advanced

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`, `TSAncientSundialsSetMaxl()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetTolerance()`,
          `TSAncientSundialsGetPC()`,
          `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsSetGramSchmidtType(TS ts, TSAncientSundialsGramSchmidtType type)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSAncientSundialsSetGramSchmidtType_C", (TS, TSAncientSundialsGramSchmidtType), (ts, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetTolerance - Sets the absolute and relative tolerance used by
  `TSANCIENTSUNDIALS` for error control.

  Logically Collective

  Input Parameters:
+ ts   - the time-step context
. aabs - the absolute tolerance
- rel  - the relative tolerance

     See the CVODE/SUNDIALS users manual for exact details on these parameters. Essentially
    these regulate the size of the error for a SINGLE timestep.

  Level: intermediate

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`, `TSAncientSundialsSetGMRESMaxl()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`,
          `TSAncientSundialsGetPC()`,
          `TSSetExactFinalTime()`
@*/
PetscErrorCode TSAncientSundialsSetTolerance(TS ts, PetscReal aabs, PetscReal rel)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSAncientSundialsSetTolerance_C", (TS, PetscReal, PetscReal), (ts, aabs, rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsGetPC - Extract the PC context from a time-step context for `TSANCIENTSUNDIALS`.

  Input Parameter:
. ts - the time-step context

  Output Parameter:
. pc - the preconditioner context

  Level: advanced

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`, `TSAncientSundialsSetMaxl()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`, `TSAncientSundialsSetTolerance()`
@*/
PetscErrorCode TSAncientSundialsGetPC(TS ts, PC *pc)
{
  PetscFunctionBegin;
  PetscUseMethod(ts, "TSAncientSundialsGetPC_C", (TS, PC *), (ts, pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetMinTimeStep - Smallest time step to be chosen by the adaptive controller.

  Input Parameters:
+ ts    - the time-step context
- mindt - lowest time step if positive, negative to deactivate

  Note:
  `TSANCIENTSUNDIALS` will error if it is not possible to keep the estimated truncation error below
  the tolerance set with `TSAncientSundialsSetTolerance()` without going below this step size.

  Level: beginner

.seealso: [](ch_ts), `TSAncientSundialsSetType()`, `TSAncientSundialsSetTolerance()`
@*/
PetscErrorCode TSAncientSundialsSetMinTimeStep(TS ts, PetscReal mindt)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSAncientSundialsSetMinTimeStep_C", (TS, PetscReal), (ts, mindt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetMaxTimeStep - Largest time step to be chosen by the adaptive controller.

  Input Parameters:
+ ts    - the time-step context
- maxdt - lowest time step if positive, negative to deactivate

  Level: beginner

.seealso: [](ch_ts), `TSAncientSundialsSetType()`, `TSAncientSundialsSetTolerance()`
@*/
PetscErrorCode TSAncientSundialsSetMaxTimeStep(TS ts, PetscReal maxdt)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSAncientSundialsSetMaxTimeStep_C", (TS, PetscReal), (ts, maxdt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsMonitorInternalSteps - Monitor `TSANCIENTSUNDIALS` internal steps (Defaults to false).

  Input Parameters:
+ ts - the time-step context
- ft - `PETSC_TRUE` if monitor, else `PETSC_FALSE`

  Level: beginner

.seealso: [](ch_ts), `TSAncientSundialsGetIterations()`, `TSAncientSundialsSetType()`, `TSAncientSundialsSetMaxl()`,
          `TSAncientSundialsSetLinearTolerance()`, `TSAncientSundialsSetGramSchmidtType()`, `TSAncientSundialsSetTolerance()`,
          `TSAncientSundialsGetPC()`
@*/
PetscErrorCode TSAncientSundialsMonitorInternalSteps(TS ts, PetscBool ft)
{
  PetscFunctionBegin;
  PetscTryMethod(ts, "TSAncientSundialsMonitorInternalSteps_C", (TS, PetscBool), (ts, ft));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSAncientSundialsSetUseDense - Set a flag to use a dense linear solver in `TSANCIENTSUNDIALS` (serial only)

  Logically Collective

  Input Parameters:
+ ts        - the time-step context
- use_dense - `PETSC_TRUE` to use the dense solver

  Level: advanced

.seealso: [](ch_ts), `TSANCIENTSUNDIALS`
@*/
PetscErrorCode TSAncientSundialsSetUseDense(TS ts, PetscBool use_dense)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ts, use_dense, 2);
  PetscTryMethod(ts, "TSAncientSundialsSetUseDense_C", (TS, PetscBool), (ts, use_dense));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   TSANCIENTSUNDIALS - ODE solver using a very old version of the LLNL CVODE/SUNDIALS package, version 2.5.
   Requires `./configure --download-ancientsundials`

   Options Database Keys:
+    -ts_ancientsundials_type (bdf|adams)                      - integrator type
.    -ts_ancientsundials_gramschmidt_type (modified|classical) - type of orthogonalization inside GMRES
.    -ts_ancientsundials_atol atol                             - Absolute tolerance for convergence
.    -ts_ancientsundials_rtol rtol                             - Relative tolerance for convergence
.    -ts_ancientsundials_linear_tolerance ltol                 - convergence tolerance for linear solver
.    -ts_ancientsundials_maxl maxl                             - Max dimension of the Krylov subspace
.    -ts_ancientsundials_monitor_steps                         - Monitor ANCIENTSUNDIALS internal steps
-    -ts_ancientsundials_use_dense                             - Use a dense linear solver within CVODE (serial only)

    Level: beginner

    Note:
    This uses its own nonlinear solver and Krylov method so PETSc `SNES` and `KSP` options do not apply,
    only PETSc `PC` options.

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSAncientSundialsSetType()`, `TSAncientSundialsSetMaxl()`, `TSAncientSundialsSetLinearTolerance()`, `TSType`,
          `TSAncientSundialsSetGramSchmidtType()`, `TSAncientSundialsSetTolerance()`, `TSAncientSundialsGetPC()`, `TSAncientSundialsGetIterations()`, `TSSetExactFinalTime()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_AncientSundials(TS ts)
{
  TS_AncientSundials *cvode;
  PC                  pc;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_AncientSundials;
  ts->ops->destroy        = TSDestroy_AncientSundials;
  ts->ops->view           = TSView_AncientSundials;
  ts->ops->setup          = TSSetUp_AncientSundials;
  ts->ops->step           = TSStep_AncientSundials;
  ts->ops->interpolate    = TSInterpolate_AncientSundials;
  ts->ops->setfromoptions = TSSetFromOptions_AncientSundials;
  ts->default_adapt_type  = TSADAPTNONE;

  PetscCall(PetscNew(&cvode));

  ts->usessnes = PETSC_TRUE;

  ts->data           = (void *)cvode;
  cvode->cvode_type  = ANCIENTSUNDIALS_BDF;
  cvode->gtype       = ANCIENTSUNDIALS_CLASSICAL_GS;
  cvode->maxl        = 5;
  cvode->maxord      = PETSC_DEFAULT;
  cvode->linear_tol  = .05;
  cvode->monitorstep = PETSC_TRUE;
  cvode->use_dense   = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)ts), &cvode->comm_sundials));

  cvode->mindt = -1.;
  cvode->maxdt = -1.;

  /* set tolerance for ANCIENTSUNDIALS */
  cvode->reltol = 1e-6;
  cvode->abstol = 1e-6;

  /* set PCNONE as default pctype */
  PetscCall(TSAncientSundialsGetPC_AncientSundials(ts, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetType_C", TSAncientSundialsSetType_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetMaxl_C", TSAncientSundialsSetMaxl_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetLinearTolerance_C", TSAncientSundialsSetLinearTolerance_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetGramSchmidtType_C", TSAncientSundialsSetGramSchmidtType_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetTolerance_C", TSAncientSundialsSetTolerance_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetMinTimeStep_C", TSAncientSundialsSetMinTimeStep_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetMaxTimeStep_C", TSAncientSundialsSetMaxTimeStep_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsGetPC_C", TSAncientSundialsGetPC_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsGetIterations_C", TSAncientSundialsGetIterations_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsMonitorInternalSteps_C", TSAncientSundialsMonitorInternalSteps_AncientSundials));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSAncientSundialsSetUseDense_C", TSAncientSundialsSetUseDense_AncientSundials));
  PetscFunctionReturn(PETSC_SUCCESS);
}
