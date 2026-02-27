/*
  Code for Timestepping with Runge-Kutta Super-time-steppers (Chebyshev, Legendre).
*/
#include <petsc/private/tsimpl.h>

const char *RKSTypes[] = {"rkc1", "rkc2", "rkl1", "rkl2", "RKSType", "RKS_", NULL};
typedef enum {
  RKS_RKC1,
  RKS_RKC2,
  RKS_RKL1,
  RKS_RKL2
} RKSType;

typedef struct {
  PetscInt  stages;
  RKSType   type;
  PetscReal epsilon;
  Vec       U0, Uprev, Ucurr, Uwork;
  Vec       F0, Fcurr;

  PetscReal *mu;
  PetscReal *nu;
  PetscReal *tilde_mu;
  PetscReal *tilde_gamma;
  PetscReal *b;
} TS_RKS;

/*
  Polynomial Functions
*/
static inline PetscReal Chebyshev(PetscInt j, PetscReal x)
{
  if (PetscAbsReal(x) <= 1.0) {
    PetscReal theta = PetscAcosReal(x);
    return PetscCosReal(j * theta);
  } else {
    /* Use hyperbolic form for x > 1 (or x < -1) */
    PetscReal y   = PetscAcoshReal(PetscAbsReal(x));
    PetscReal val = PetscCoshReal(j * y);
    if (x < 0.0 && (j % 2)) val = -val;
    return val;
  }
}

static inline PetscReal ChebyshevPrime(PetscInt j, PetscReal x)
{
  if (j == 0) return 0.0;
  if (j == 1) return 1.0;

  if (PetscAbsReal(x) <= 1.0) {
    PetscReal theta     = PetscAcosReal(x);
    PetscReal sin_theta = PetscSinReal(theta);
    if (PetscAbsReal(sin_theta) < PETSC_MACHINE_EPSILON) return PetscSignReal(sin_theta) * PETSC_MAX_REAL;
    PetscReal U = PetscSinReal(j * theta) / sin_theta;
    return j * U;
  } else {
    PetscReal y     = PetscAcoshReal(PetscAbsReal(x));
    PetscReal denom = PetscSqrtReal(x * x - 1.0);
    PetscReal U     = PetscSinhReal(j * y) / denom;
    if (x < 0.0 && (j % 2)) U = -U;
    return j * U;
  }
}

static inline PetscReal ChebyshevDoublePrime(PetscInt j, PetscReal x)
{
  if (j <= 1) return 0.0;

  PetscReal Tj = Chebyshev(j, x);
  PetscReal Uj1;
  if (PetscAbsReal(x) <= 1.0) {
    PetscReal theta     = PetscAcosReal(x);
    PetscReal sin_theta = PetscSinReal(theta);
    if (PetscAbsReal(sin_theta) < PETSC_MACHINE_EPSILON) return PETSC_MAX_REAL;
    Uj1 = PetscSinReal(j * theta) / sin_theta;
  } else {
    PetscReal y     = PetscAcoshReal(PetscAbsReal(x));
    PetscReal denom = PetscSqrtReal(x * x - 1.0);
    Uj1             = PetscSinhReal(j * y) / denom;
    if (x < 0.0 && (j % 2)) Uj1 = -Uj1;
  }
  PetscReal denom2 = x * x - 1.0;
  if (PetscAbsReal(denom2) < PETSC_MACHINE_EPSILON) return PETSC_MAX_REAL;
  return j * (x * Uj1 - j * Tj) / denom2;
}

static inline PetscReal Legendre(PetscInt j, PetscReal x)
{
  if (j == 0) return 1.0;
  if (j == 1) return x;
  PetscReal Pjm2 = 1.0, Pjm1 = x, Pj;
  for (PetscInt n = 2; n <= j; n++) {
    Pj   = ((2.0 * n - 1.0) * x * Pjm1 - (n - 1.0) * Pjm2) / n;
    Pjm2 = Pjm1;
    Pjm1 = Pj;
  }
  return Pjm1;
}

static inline PetscReal LegendrePrime(PetscInt j, PetscReal x)
{
  if (j == 0) return 0.0;
  if (j == 1) return 1.0;
  if (x == 1.0) return j * (j + 1) / 2;
  else if (x == -1.0) return PetscPowReal(-1.0,j+1) * (j * (j + 1) / 2);
  else {
    PetscReal Pj    = Legendre(j, x);
    PetscReal Pjm1  = Legendre(j - 1, x);
    PetscReal denom = x * x - 1.0;
    return j * (x * Pj - Pjm1) / denom;
  }
}

/*@
  TSRKC1ComputeCoefficients - This function calculates the coefficients for the first-order RK-Chebyshev method

  Not Collective

  Input Parameter:
+ s       - number of stages
- epsilon - stabilizing parameter

  Output Parameters:
+ mu       - mu coefficient
. nu       - nu coefficient
. tilde_mu - $\tilde{mu}$ coefficient
- b        - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `TSRKC2ComputeCoefficients()`
*/
PetscErrorCode TSRKC1ComputeCoefficients(PetscInt s, PetscReal epsilon, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *b)
{
  PetscReal w0 = 1.0 + epsilon / ((PetscReal)s * (PetscReal)s);

  PetscFunctionBegin;
  for (PetscInt j = 0; j <= s; ++j) {
    PetscReal Pj = Chebyshev(j, w0);
    b[j] = 1.0 / Pj;
  }

  PetscReal Ps, Psprime;
  Ps      = Chebyshev(s, w0);
  Psprime = ChebyshevPrime(s, w0);
  PetscReal w1 = Ps / Psprime;

  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = 0.0;

  tilde_mu[1] = w1 / w0;
  for (PetscInt j = 2; j <= s; ++j) {
    mu[j]       = 2.0 * w0 * (b[j] / b[j - 1]);
    nu[j]       = -(b[j] / b[j - 2]);
    tilde_mu[j] = 2.0 * w1 * (b[j] / b[j - 1]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRKC2ComputeCoefficients - This function calculates the coefficients for the second-order RK-Chebyshev method

  Not Collective

  Input Parameter:
+ s       - number of stages
- epsilon - stabilizing parameter

  Output Parameters:
+ mu          - mu coefficient
. nu          - nu coefficient
. tilde_mu    - $\tilde{mu}$ coefficient
. tilde_gamma - $\tilde{gamma}$ coefficient
- b           - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `TSRKC1ComputeCoefficients()`
*/
PetscErrorCode TSRKC2ComputeCoefficients(PetscInt s, PetscReal epsilon, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *tilde_gamma, PetscReal *b)
{
  PetscReal w0 = 1.0 + epsilon / ((PetscReal)s * (PetscReal)s);

  PetscFunctionBegin;
  for (PetscInt j = 2; j <= s; j++) {
    PetscReal Pp, Ppp;
    Pp  = ChebyshevPrime(j, w0);
    Ppp = ChebyshevDoublePrime(j, w0);
    b[j] = -Ppp / (Pp * Pp);
  }
  b[0] = b[2];
  b[1] = b[2];

  PetscReal Psprime, Ps2;
  Psprime = ChebyshevPrime(s, w0);
  Ps2     = ChebyshevDoublePrime(s, w0);
  PetscReal w1 = -Psprime / Ps2;

  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = tilde_gamma[j] = 0.0;
  tilde_mu[1] = b[1] * w1;

  for (PetscInt j = 2; j <= s; ++j) {
    mu[j] = 2.0 * w0 * (b[j] / b[j - 1]);
    if (j == 2) nu[j] = 0;
    else nu[j] = -(b[j] / b[j - 2]);
    tilde_mu[j] = 2.0 * w1 * (b[j] / b[j - 1]);
    PetscReal Pjm1;
    Pjm1           = Chebyshev(j - 1, w0);
    tilde_gamma[j] = -(1.0 - b[j - 1] * Pjm1) * tilde_mu[j];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRKL1ComputeCoefficients - This function calculates the coefficients for the first-order RK-Legendre method

  Not Collective

  Input Parameter:
+ s - number of stages

  Output Parameters:
+ mu       - mu coefficient
. nu       - nu coefficient
- tilde_mu - $\tilde{mu}$ coefficient

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `TSRKL2ComputeCoefficients()`
*/
PetscErrorCode TSRKL1ComputeCoefficients(PetscInt s, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu)
{
  PetscReal w1 = 2.0 / ((PetscReal)s * (PetscReal)s + (PetscReal)s);

  PetscFunctionBegin;
  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = 0.0;
  tilde_mu[1] = ((2.0 - 1.0) / 1.0) * w1;
  for (PetscInt j = 2; j <= s; ++j) {
    PetscReal j_r = (PetscReal)j;
    mu[j]         = ((2.0 * j_r - 1.0) / j_r);
    nu[j]         = ((1.0 - j_r) / j_r);
    tilde_mu[j]   = ((2.0 * j_r - 1.0) / j_r) * w1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TSRKL2ComputeCoefficients - This function calculates the coefficients for the second-order RK-Legendre method

  Not Collective

  Input Parameter:
+ s           - number of stages

  Output Parameters:
+ mu          - mu coefficient
. nu          - nu coefficient
. tilde_mu    - $\tilde{mu}$ coefficient
. tilde_gamma - $\tilde{gamma}$ coefficient
- b           - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `TSRKL1ComputeCoefficients()`
*/
PetscErrorCode TSRKL2ComputeCoefficients(PetscInt s, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *tilde_gamma, PetscReal *b)
{
  PetscReal w1;

  PetscFunctionBegin;
  w1   = 4.0 / (((PetscReal)s * (PetscReal)s + (PetscReal)s - 2.0));
  b[0] = 1.0 / 3.0;
  b[1] = 1.0 / 3.0;

  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = tilde_gamma[j] = 0.0;
  tilde_mu[1] = 4 / (3 * ((PetscReal)s * (PetscReal)s + (PetscReal)s - 2));
  for (PetscInt j = 2; j <= s; ++j) {
    PetscReal j_r   = (PetscReal)j;
    b[j]            = (j_r * j_r + j_r - 2) / (2 * j_r * (j_r + 1));
    PetscReal a_jm1 = 1 - b[j - 1];

    mu[j]          = ((2.0 * j_r - 1.0) / (j_r)) * (b[j] / b[j - 1]);
    nu[j]          = -((j_r - 1.0) / (j_r)) * (b[j] / b[j - 2]);
    tilde_mu[j]    = w1 * ((2.0 * j_r - 1.0) / j_r) * (b[j] / b[j - 1]);
    tilde_gamma[j] = -a_jm1 * tilde_mu[j];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_RKS(TS ts)
{
  TS_RKS   *rks = (TS_RKS *)ts->data;
  PetscReal dt, time;
  PetscInt  s;
  Vec       U;

  PetscFunctionBegin;
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));

  s = rks->stages;

  PetscCall(VecCopy(U, rks->Ucurr)); /* Ucurr = U^{(j-1)} */
  PetscCall(VecCopy(U, rks->Uprev)); /* Uprev = U^{(j-2)} */
  PetscCall(VecCopy(U, rks->U0));    /* U0    = U^{(0)}   */

  /* F(U^{(0)}) */
  PetscCall(TSComputeRHSFunction(ts, time, rks->U0, rks->F0));

  for (PetscInt j = 1; j <= s; ++j) {
    /* F(U^{(j-1)}) */
    if (j > 1) PetscCall(TSComputeRHSFunction(ts, time, rks->Ucurr, rks->Fcurr));

    if (j == 1) {
      /* U^{(1)} = U^{(0)} + tilde_mu[1] * dt * F(U^{(0)}) */
      PetscCall(VecWAXPY(rks->Uwork, rks->tilde_mu[1] * dt, rks->F0, rks->Ucurr));
    } else {
      /* First Order:  U^{(j)} = mu[j] U^{(j-1)} + nu_j U^{(j-2)} + tilde_mu[j] dt F(U^{(j-1)}) */
      /* Second Order: U^{(j)} = mu[j] U^{(j-1)} + nu_j U^{(j-2)} + tilde_mu[j] dt F(U^{(j-1)}) + (1 - mu[j] - nu[j]) U^{(0)} + tilde_gamma[j] dt F(U^{(0)}) */
      PetscCall(VecCopy(rks->Ucurr, rks->Uwork));  /* U^{(j)} = U^{(j-1)} */
      PetscCall(VecScale(rks->Uwork, rks->mu[j])); /* U^{(j)} = mu[j]*U^{(j-1)} */

      PetscScalar coefs[4];
      Vec         vecs[4];
      /* nu[j] * U^{(j-2)} */
      coefs[0] = rks->nu[j];
      vecs[0]  = rks->Uprev;
      /* tilde_mu[j]*dt * F(U^{(j-1)}) */
      coefs[1] = rks->tilde_mu[j] * dt;
      vecs[1]  = rks->Fcurr;

      if (rks->type == RKS_RKC2 || rks->type == RKS_RKL2) {
        coefs[2] = 1.0 - rks->mu[j] - rks->nu[j];
        vecs[2]  = rks->U0; /* (1-mu[j]-nu[j]) * U^{(0)} */
        coefs[3] = rks->tilde_gamma[j] * dt;
        vecs[3]  = rks->F0; /* tilde_gamma[j]*dt * F(U^{(0)}) */
        PetscCall(VecMAXPY(rks->Uwork, 4, coefs, vecs));
      } else {
        PetscCall(VecMAXPY(rks->Uwork, 2, coefs, vecs));
      }
    }
    {
      Vec tmp    = rks->Uprev;
      rks->Uprev = rks->Ucurr;
      rks->Ucurr = rks->Uwork;
      rks->Uwork = tmp;
    }
  }
  PetscCall(VecCopy(rks->Ucurr, U));
  PetscCall(TSSetTime(ts, time + dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetUp_RKS(TS ts)
{
  TS_RKS *rks = (TS_RKS *)ts->data;
  Vec     U;

  PetscFunctionBegin;
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecDuplicate(U, &rks->U0));
  PetscCall(VecDuplicate(U, &rks->Uprev));
  PetscCall(VecDuplicate(U, &rks->Ucurr));
  PetscCall(VecDuplicate(U, &rks->Uwork));
  PetscCall(VecDuplicate(U, &rks->F0));
  PetscCall(VecDuplicate(U, &rks->Fcurr));

  PetscCheck(rks->stages >= 2, PETSC_COMM_SELF, PETSC_ERR_FP, "RKS requires s >= 2");

  PetscCall(PetscCalloc5(rks->stages + 1, &rks->b, rks->stages + 1, &rks->mu, rks->stages + 1, &rks->nu, rks->stages + 1, &rks->tilde_mu, rks->stages + 1, &rks->tilde_gamma));
  switch (rks->type) {
  case RKS_RKC1:
    PetscCall(TSRKC1ComputeCoefficients(rks->stages, rks->epsilon, rks->mu, rks->nu, rks->tilde_mu, rks->b));
    break;
  case RKS_RKL1:
    PetscCall(TSRKL1ComputeCoefficients(rks->stages, rks->mu, rks->nu, rks->tilde_mu));
    break;
  case RKS_RKC2:
    PetscCall(TSRKC2ComputeCoefficients(rks->stages, rks->epsilon, rks->mu, rks->nu, rks->tilde_mu, rks->tilde_gamma, rks->b));
    break;
  case RKS_RKL2:
    PetscCall(TSRKL2ComputeCoefficients(rks->stages, rks->mu, rks->nu, rks->tilde_mu, rks->tilde_gamma, rks->b));
    break;
  default:
    SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown polynomial type");
  }
  PetscCheck(rks->mu && rks->nu && rks->tilde_mu && rks->b, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Coefficients not initialized");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSSetFromOptions_RKS(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_RKS *rks = (TS_RKS *)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "RKS options");
  PetscCall(PetscOptionsEnum("-ts_rks_type", "Super-time-stepper type", "", RKSTypes, (PetscEnum)rks->type, (PetscEnum *)&rks->type, NULL));
  PetscCall(PetscOptionsInt("-ts_rks_stages", "Number of stages per macro-step", NULL, rks->stages, &rks->stages, NULL));
  PetscCall(PetscOptionsReal("-ts_rks_epsilon", "RKS epsilon", "", rks->epsilon, &rks->epsilon, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_RKS(TS ts)
{
  TS_RKS *rks = (TS_RKS *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&rks->U0));
  PetscCall(VecDestroy(&rks->Uprev));
  PetscCall(VecDestroy(&rks->Ucurr));
  PetscCall(VecDestroy(&rks->Uwork));
  PetscCall(VecDestroy(&rks->F0));
  PetscCall(VecDestroy(&rks->Fcurr));
  PetscCall(PetscFree5(rks->b, rks->mu, rks->nu, rks->tilde_mu, rks->tilde_gamma));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_RKS(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_RKS(ts));
  PetscCall(PetscFree(ts->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_RKS(TS ts, PetscViewer viewer)
{
  TS_RKS   *rks = (TS_RKS *)ts->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "RKS method: %s, stages=%" PetscInt_FMT "\n", RKSTypes[rks->type], rks->stages));
    for (PetscInt j = 0; j < rks->stages; ++j) PetscCall(PetscViewerASCIIPrintf(viewer, "s:%" PetscInt_FMT ", mu=%f, nu=%f, tilde_mu=%f, tilde_gamma=%f, b=%f\n", j, rks->mu[j], rks->nu[j], rks->tilde_mu[j], rks->tilde_gamma[j], rks->b[j]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TSRKS - explicit ODE solver using Runge-Kutta Supertimestepper (RKS) schemes

  The implementations of each polynomial type come from:
  - Runge-Kutta-Chebyshev: {cite}`VerwerHundsdorferSommeijer1990`
  - Runge-Kutta-Legendre: {cite}`MeyerDinshawTariq2014`

  The user should provide the right-hand side of the equation
  using `TSSetRHSFunction()`.

  Level: beginner

  Note:
  The default is `TSRKL1`, it can be changed with `-ts_rks_type`

.seealso: [](ch_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSRK`, `TSType`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_RKS(TS ts)
{
  TS_RKS *rks;

  PetscFunctionBegin;
  PetscCall(PetscNew(&rks));
  ts->data = (void *)rks;

  ts->ops->setup          = TSSetUp_RKS;
  ts->ops->step           = TSStep_RKS;
  ts->ops->destroy        = TSDestroy_RKS;
  ts->ops->reset          = TSReset_RKS;
  ts->ops->setfromoptions = TSSetFromOptions_RKS;
  ts->ops->view           = TSView_RKS;

  rks->type    = RKS_RKL1;
  rks->stages  = 10;
  rks->epsilon = 0.05;
  PetscFunctionReturn(PETSC_SUCCESS);
}
