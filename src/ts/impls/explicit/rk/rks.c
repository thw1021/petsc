/*
  Code for Timestepping with Runge-Kutta Supertimesteppers (Chebyshev, Legendre, Gegenbauer).

  The implementations of each polynomial type come from:
  - Runge-Kutta-Chebyshev (https://doi.org/10.1007/BF01386405)
  - Runge-Kutta-Legendre (https://doi.org/10.1016/j.jcp.2013.08.021)
  - Runge-Kutta-Gegenbauer (https://doi.org/10.1016/j.jcp.2020.109879)
*/
#include <petsc/private/tsimpl.h>

const char *RKSTypes[] = {"rkc1","rkc2","rkl1","rkl2","rkg1","rkg2", "RKSType", "RKS_", NULL};
typedef enum {RKS_RKC1, RKS_RKC2, RKS_RKL1, RKS_RKL2, RKS_RKG1, RKS_RKG2} RKSType;

typedef struct {
  PetscInt   stages;
  RKSType    type;
  PetscReal  rkg_parameter;
  PetscReal  epsilon;
  Vec        U0, Uprev, Ucurr, Uwork;
  Vec        F0, Fcurr;
} TS_RKS;

/*
  Polynomial Functions
*/
static inline PetscReal Chebyshev(PetscInt j, PetscReal x)
{
  if (PetscAbsReal(x) <= 1.0) {
    PetscReal theta = PetscAcosReal(x);
    return PetscCosReal((double)j * theta);
  } else {
    /* Use hyperbolic form for x > 1 (or x < -1) */
    PetscReal y = PetscAcoshReal(PetscAbsReal(x));
    PetscReal val = PetscCoshReal((double)j * y);
    if (x < 0.0 && (j % 2)) val = -val;
    return val;
  }
}

static inline PetscReal ChebyshevPrime(PetscInt j, PetscReal x)
{
  if (j == 0) return 0.0;
  if (j == 1) return 1.0;

  if (PetscAbsReal(x) <= 1.0) {
    PetscReal theta = PetscAcosReal(x);
    PetscReal sin_theta = PetscSinReal(theta);
    if (PetscAbsReal(sin_theta) < 1e-14) return PETSC_MAX_REAL;
    PetscReal U = PetscSinReal(j * theta) / sin_theta;
    return j * U;
  } else {
    PetscReal y = PetscAcoshReal(PetscAbsReal(x));
    PetscReal denom = PetscSqrtReal(x * x - 1.0);
    PetscReal U = PetscSinhReal(j * y) / denom;
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
    PetscReal theta = PetscAcosReal(x);
    PetscReal sin_theta = PetscSinReal(theta);
    if (PetscAbsReal(sin_theta) < 1e-14) return PETSC_MAX_REAL;
    Uj1 = PetscSinReal(j * theta) / sin_theta;
  } else {
    PetscReal y = PetscAcoshReal(PetscAbsReal(x));
    PetscReal denom = PetscSqrtReal(x * x - 1.0);
    Uj1 = PetscSinhReal(j * y) / denom;
    if (x < 0.0 && (j % 2)) Uj1 = -Uj1;
  }
  PetscReal denom2 = x * x - 1.0;
  if (PetscAbsReal(denom2) < 1e-14) return PETSC_MAX_REAL;
  return j * (x * Uj1 - j * Tj) / denom2;
}

static inline PetscReal Legendre(PetscInt j, PetscReal x)
{
  if (j == 0) return 1.0;
  if (j == 1) return x;
  PetscReal Pjm2 = 1.0, Pjm1 = x, Pj;
  for (PetscInt n = 2; n <= j; n++) {
    Pj = ((2.0*n - 1.0)*x*Pjm1 - (n - 1.0)*Pjm2)/n;
    Pjm2 = Pjm1;
    Pjm1 = Pj;
  }
  return Pjm1;
}

static inline PetscReal LegendrePrime(PetscInt j, PetscReal x)
{
  if (j == 0) return 0.0;
  if (j == 1) return 1.0;
  PetscReal Pj = Legendre(j, x);
  PetscReal Pjm1 = Legendre(j-1, x);
  return j*(x*Pj - Pjm1)/(x*x - 1.0);
}

static inline PetscReal LegendreDoublePrime(PetscInt j, PetscReal x)
{
  if (j < 2) return 0.0;
  PetscReal Pj = Legendre(j, x);
  PetscReal PjP = LegendrePrime(j, x);
  return (2.0*x*PjP - j*(j+1.0)*Pj)/(x*x - 1.0);
}

static inline PetscReal Gegenbauer(PetscInt j, PetscReal gegen_param, PetscReal x)
{
  if (j == 0) return 1.0;
  if (j == 1) return 2.0*gegen_param*x;
  PetscReal Cjm2 = 1.0, Cjm1 = 2.0*gegen_param*x, Cj;
  for (PetscInt n = 2; n <= j; n++) {
    Cj = (1/(n+1)) * (2.0*(n + gegen_param)*x*Cjm1 - (n + 2.0*gegen_param - 1.0)*Cjm2);
    Cjm2 = Cjm1;
    Cjm1 = Cj;
  }
  return Cjm1;
}

static inline PetscReal GegenbauerPrime(PetscInt j, PetscReal gegen_param, PetscReal x)
{
  if (j == 0) return 0.0;
  return 2.0*gegen_param * Gegenbauer(j-1, gegen_param+1.0, x);
}

static inline PetscReal GegenbauerDoublePrime(PetscInt j, PetscReal gegen_param, PetscReal x)
{
  if (j < 2) return 0.0;
  PetscReal denom = 1.0 - x * x;
  if (PetscAbsReal(denom) < 1e-14) return PETSC_INFINITY; /* avoid singularity at |x|=1 */
  PetscReal term1 = ((2.0 * gegen_param + 1.0) * x / denom) * GegenbauerPrime(j, gegen_param, x);
  PetscReal term2 = (j * (j + 2.0 * gegen_param) / denom) * Gegenbauer(j, gegen_param, x);
  return term1 - term2;
}

/*
  Runge-Kutta-Chebyshev Coefficient Functions
*/

/*@C
  ComputeRKL1coefficients - This function calculates the coefficients for the first-order RK-Chebyshev method

  Not Collective

  Input Parameter:
+ s       - number of stages
- epsilon - stabilizing parameter

  Output Parameters:
+ mu       - mu coefficient
. nu       - nu coefficient
. tilde_mu - \tilde{mu} coefficient
- b        - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `ComputeRKC2coefficients()`
@*/
static PetscErrorCode ComputeRKC1coefficients(PetscInt s, PetscReal epsilon, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *b)
{
  PetscReal w0 = 1.0 + epsilon / ((PetscReal)s*(PetscReal)s);

  PetscFunctionBegin;
  for (PetscInt j = 0; j <= s; ++j) {
    PetscReal Pj = Chebyshev(j, w0);
    PetscCheck(PetscAbsReal(Pj) > 1e-16,PETSC_COMM_SELF, PETSC_ERR_FP,"Polynomial nearly zero at j=%d, w0=%g", j, (double)w0);
    b[j] = 1.0 / Pj;
  }

  PetscReal Ps, Psprime;
  Ps = Chebyshev(s, w0);
  Psprime = ChebyshevPrime(s, w0);
  PetscCheck(isfinite((double)Psprime) && PetscAbsReal(Psprime) >= 1e-16,PETSC_COMM_SELF, PETSC_ERR_FP,"Polynomial derivative nearly zero at w0");
  PetscReal w1 = Ps / Psprime;

  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = 0.0;

  tilde_mu[1] = w1 / w0;
  for (PetscInt j = 2; j <= s; ++j) {
    mu[j]       = 2.0 * w0 * (b[j] / b[j-1]);
    nu[j]       = - (b[j] / b[j-2]);
    tilde_mu[j] = 2.0 * w1 * (b[j] / b[j-1]);
  }
  PetscFunctionReturn(0);
}

/*@C
  ComputeRKC2coefficients - This function calculates the coefficients for the second-order RK-Chebyshev method

  Not Collective

  Input Parameter:
+ s       - number of stages
- epsilon - stabilizing parameter

  Output Parameters:
+ mu          - mu coefficient
. nu          - nu coefficient
. tilde_mu    - \tilde{mu} coefficient
. tilde_gamma - \tilde{gamma} coefficient
- b           - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `ComputeRKC1coefficients()`
@*/
static PetscErrorCode ComputeRKC2coefficients(PetscInt s, PetscReal epsilon, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *tilde_gamma, PetscReal *b)
{
  PetscReal w0 = 1.0 + epsilon / ((PetscReal)s*(PetscReal)s);

  PetscFunctionBegin;
  for (PetscInt j = 2; j <= s; j++) {
    PetscReal Pp, Ppp;
    Pp = ChebyshevPrime(j, w0);
    Ppp = ChebyshevDoublePrime(j, w0);
    PetscCheck((PetscAbsReal(Pp) > 1e-16),PETSC_COMM_SELF, PETSC_ERR_FP,"Polynomial derivative nearly zero at j=%d, w0=%g", j, (double)w0);
    b[j] = Ppp / (Pp*Pp);
  }
  b[0] = b[2];
  b[1] = b[2];

  PetscReal Psprime, Ps2;
  Psprime = ChebyshevPrime(s, w0);
  Ps2 = ChebyshevDoublePrime(s, w0);
  PetscCheck(isfinite((double)Psprime) && isfinite((double)Ps2) && PetscAbsReal(Ps2) > 1e-16,PETSC_COMM_SELF, PETSC_ERR_FP,"Polynomial derivatives nearly zero at w0");
  PetscReal w1 = Psprime / Ps2;

  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = tilde_gamma[j] = 0.0;
  tilde_mu[1] = b[1] * w1;

  for (PetscInt j = 2; j <= s; ++j) {
    mu[j] = 2.0 * w0 * (b[j] / b[j-1]);
    if (j == 2) nu[j] = 0;
    else nu[j] = - (b[j] / b[j-2]);
    tilde_mu[j] = 2.0 * w1 * (b[j] / b[j-1]);
    PetscReal Pjm1;
    Pjm1 = Chebyshev(j-1,w0);
    tilde_gamma[j] = - (1.0 - b[j-1]*Pjm1) * tilde_mu[j];
  }
  PetscFunctionReturn(0);
}

/*
  Runge-Kutta-Legendre Coefficient Functions
*/

/*@C
  ComputeRKL1coefficients - This function calculates the coefficients for the first-order RK-Legendre method

  Not Collective

  Input Parameter:
+ s - number of stages

  Output Parameters:
+ mu       - mu coefficient
. nu       - nu coefficient
. tilde_mu - \tilde{mu} coefficient
- b        - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `ComputeRKL2coefficients()`
@*/
static PetscErrorCode ComputeRKL1coefficients(PetscInt s, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *b)
{
  PetscReal w1 = 2.0 / ((PetscReal)s*(PetscReal)s + (PetscReal)s);

  PetscFunctionBegin;
  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = 0.0;
  tilde_mu[1] = ((2.0 - 1.0) / 1.0) * w1;
  for (PetscInt j = 2; j <= s; ++j) {
    PetscReal j_r = (PetscReal) j;
    mu[j] = ( (2.0*j_r - 1.0) / j_r );
    nu[j]  = ((1.0 - j_r ) / j_r );
    tilde_mu[j] = ( (2.0*j_r - 1.0) / j_r ) * w1;
  }
  PetscFunctionReturn(0);
}

/*@C
  ComputeRKL2coefficients - This function calculates the coefficients for the second-order RK-Legendre method

  Not Collective

  Input Parameter:
+ s           - number of stages

  Output Parameters:
+ mu          - mu coefficient
. nu          - nu coefficient
. tilde_mu    - \tilde{mu} coefficient
. tilde_gamma - \tilde{gamma} coefficient
- b           - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `ComputeRKL1coefficients()`
@*/
static PetscErrorCode ComputeRKL2coefficients(PetscInt s, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *tilde_gamma, PetscReal *b)
{
  PetscReal w1;

  PetscFunctionBegin;
  w1 = 4.0 / (((PetscReal)s*(PetscReal)s + (PetscReal)s - 2.0));
  b[0] = 1.0/3.0;
  b[1] =  1.0/3.0;

  for (PetscInt j = 0; j <= s; ++j) mu[j] = nu[j] = tilde_mu[j] = tilde_gamma[j] = 0.0;
  tilde_mu[1] = 4 / (3*((PetscReal)s*(PetscReal)s+(PetscReal)s-2));
  for (PetscInt j = 2; j <= s; ++j) {
    PetscReal j_r = (PetscReal) j;
    PetscReal s_r = (PetscReal) s;
    b[j] = (j_r*j_r + j_r - 2)/(2*j_r*(j_r + 1));
    PetscReal a_jm1 = 1 - b[j-1];

    mu[j] = ((2.0*j_r - 1.0) / (j_r))*(b[j]/b[j-1]);
    nu[j]  = -((j_r - 1.0) / (j_r))*(b[j]/b[j-2]);
    tilde_mu[j] = (4.0*(2.0*j_r - 1.0)/(j_r*(s_r*s_r + s_r - 2)))*(b[j]/b[j-1]);
    tilde_gamma[j] = - a_jm1 * tilde_mu[j];
  }
  PetscFunctionReturn(0);
}

/*
  Runge-Kutta-Gegenbauer Coefficient Functions
*/

/*@C
  ComputeRKG1coefficients - This function calculates the coefficients for the first-order RK-Gegenbauer method

  Not Collective

  Input Parameter:
+ s           - number of stages
- gegen_param - Gegenbauer polynomial parameter

  Output Parameters:
+ mu       - mu coefficient
. nu       - nu coefficient
. tilde_mu - \tilde{mu} coefficient
- b        - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `ComputeRKG2coefficients()`
@*/
static PetscErrorCode ComputeRKG1coefficients(PetscInt s, PetscReal gegen_param, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *b)
{
  PetscReal w1 = 4.0/((PetscReal)s*((PetscReal)s + 3.0));

  PetscFunctionBegin;
  for (PetscInt j = 0; j <= s; ++j) {
    PetscReal j_r = (PetscReal) j;
    b[j] = 2.0 / ((j_r + 1.)*(j_r + 2.));
    mu[j] = nu[j] = tilde_mu[j] = 0.0;

    if (j>=1) mu[j]       = ((2.0*j_r + 1.0)/(j_r)) * (b[j]/b[j-1]);
    if (j>=2) nu[j]       = - ((j_r + 1.0)/(j_r)) * (b[j]/b[j-2]);
    if (j>=1) tilde_mu[j] = mu[j] * w1;
  }
  PetscFunctionReturn(0);
}

/*@C
  ComputeRKG2coefficients - This function calculates the coefficients for the second-order RK-Gegenbauer method

  Not Collective

  Input Parameter:
+ s           - number of stages
- gegen_param - Gegenbauer polynomial parameter

  Output Parameters:
+ mu          - mu coefficient
. nu          - nu coefficient
. tilde_mu    - \tilde{mu} coefficient
. tilde_gamma - \tilde{gamma} coefficient
- b           - b vector

  Level: developer

.seealso: [](ch_ts), `TSRKS`, `ComputeRKG1coefficients()`
@*/
static PetscErrorCode ComputeRKG2coefficients(PetscInt s, PetscReal gegen_param, PetscReal *mu, PetscReal *nu, PetscReal *tilde_mu, PetscReal *tilde_gamma, PetscReal *b)
{
  PetscReal w1 = 6.0 / (((PetscReal)s + 4.)*((PetscReal)s - 1.));

  PetscFunctionBegin;
  b[0] = 1.;
  b[1] = 1./3.;
  for (PetscInt j = 2; j <= s; ++j) {
    PetscReal ajm1, j_r = (PetscReal) j;
    b[j] = (4*(j_r-1)*(j_r+4))/(3*j_r*(j_r+1)*(j_r+2)*(j_r+3));
    mu[j] = nu[j] = tilde_mu[j] = tilde_gamma[j] = 0.0;
    ajm1 = 1 - 0.5*(j_r+1.)*(j_r+2)*b[j];

    if (j>=2) mu[j] = ((2.*j_r + 1.)/(j_r)) * (b[j] / b[j-1]);
    if (j>=2) tilde_mu[j] = mu[j] * w1;
    if (j>=2) nu[j] = - ((j_r + 1.)/(j_r)) * (b[j] / b[j-2]);
    if (j>=2) tilde_gamma[j] = - tilde_mu[j]*ajm1;
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------ Step Functions -----------------------------------------*/
/*@C
  TSRK1Step - This function calculates the step for first-order methods

  Level: developer

.seealso: [](ch_ts), `TSRKS`
@*/
static PetscErrorCode TSRK1Step(TS ts)
{
  TS_RKS   *rks = (TS_RKS*)ts->data;
  Vec       U;
  PetscReal dt, time;
  PetscInt  s;
  PetscReal epsilon = rks->epsilon;
  PetscReal *b = NULL, *mu = NULL, *nu = NULL, *tilde_mu = NULL;

  PetscFunctionBegin;
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));

  s = rks->stages;
  PetscCheck(s>=2,PETSC_COMM_SELF, PETSC_ERR_FP,"RKS requires s>=2");

  PetscCall(PetscCalloc4(s+1, &b, s+1, &mu, s+1, &nu, s+1, &tilde_mu));

  PetscReal gegen_param = rks->rkg_parameter;
  switch (rks->type) {
    case RKS_RKC1:
      PetscCall(ComputeRKC1coefficients(s, epsilon, mu, nu, tilde_mu, b));
      break;
    case RKS_RKL1:
      PetscCall(ComputeRKL1coefficients(s, mu, nu, tilde_mu, b));
      break;
    case RKS_RKG1:
      PetscCall(ComputeRKG1coefficients(s, gegen_param, mu, nu, tilde_mu, b));
      break;
    default:
      SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown polynomial type");
  }
  PetscCall(VecCopy(U, rks->Ucurr));
  PetscCall(VecCopy(U, rks->Uprev));
  PetscCall(VecCopy(U, rks->U0));

  /* compute F(U^{(0)}) */
  PetscCall(TSComputeRHSFunction(ts, time, rks->U0, rks->F0));

  for (PetscInt j = 1; j <= s; ++j) {
    PetscCall(TSComputeRHSFunction(ts, time, rks->Ucurr, rks->Fcurr)); /* F(U^{(j-1)}) */

    if (j == 1) {
      /* U^{(1)} = U^{(0)} + tilde_mu[1] * dt F(U^{(0)}) */
      PetscCall(VecWAXPY(rks->Uwork, tilde_mu[1]*dt, rks->F0, rks->Ucurr));
    } else {
      /* U^{(j)} = mu[j] U^{(j-1)} + nu[j] U^{(j-2)} + (1 - mu[j] - nu_[j]) U^{(0)} + tilde_mu[j] dt F(U^{(j-1)}) */
      PetscCall(VecCopy(rks->Ucurr, rks->Uwork));                 /* Uwork = U^{(j-1)} */
      PetscCall(VecScale(rks->Uwork, mu[j]));                     /* mu[j] * U^{(j-1)} */
      PetscCall(VecAXPY(rks->Uwork, nu[j], rks->Uprev));          /* + nu[j] * U^{(j-2)} */
      if (rks->type != RKS_RKG1) PetscCall(VecAXPY(rks->Uwork, 1.0 - mu[j] - nu[j], rks->U0)); /* + (1-mu-nu)*U^{(0)} */
      PetscCall(VecAXPY(rks->Uwork, tilde_mu[j]*dt, rks->Fcurr)); /* + tilde_mu[j] * dt * F(U^{(j-1)}) */
    }
    PetscCall(VecCopy(rks->Ucurr, rks->Uprev));
    PetscCall(VecCopy(rks->Uwork, rks->Ucurr));
  }
  PetscCall(VecCopy(rks->Ucurr, U));
  PetscCall(PetscFree4(b, mu, nu, tilde_mu));
  PetscFunctionReturn(0);
}

/*@C
  TSRK1Step - This function calculates the step for second-order methods

  Level: developer

.seealso: [](ch_ts), `TSRKS`
@*/
static PetscErrorCode TSRK2Step(TS ts)
{
  TS_RKS    *rks = (TS_RKS*)ts->data;
  PetscReal  dt, time;
  PetscInt   s;
  PetscReal  epsilon = rks->epsilon;
  PetscReal *b = NULL, *mu = NULL, *nu = NULL, *tilde_mu = NULL, *tilde_gamma = NULL;
  Vec        U;

  PetscFunctionBegin;
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(TSGetTime(ts, &time));
  PetscCall(TSGetTimeStep(ts, &dt));

  s = rks->stages;
  PetscCheck(s>=2,PETSC_COMM_SELF, PETSC_ERR_FP,"RKS requires s>=2");

  PetscCall(PetscCalloc5(s+1, &b, s+1, &mu, s+1, &nu, s+1, &tilde_mu, s+1, &tilde_gamma));

  PetscReal gegen_param = rks->rkg_parameter;
  switch (rks->type) {
    case RKS_RKC2:
      PetscCall(ComputeRKC2coefficients(s, epsilon, mu, nu, tilde_mu, tilde_gamma, b));
      break;
    case RKS_RKL2:
      PetscCall(ComputeRKL2coefficients(s, mu, nu, tilde_mu, tilde_gamma, b));
      break;
    case RKS_RKG2:
      PetscCall(ComputeRKG2coefficients(s, gegen_param, mu, nu, tilde_mu, tilde_gamma, b));
      break;
    default:
      SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown polynomial type");
  }
  PetscCall(VecCopy(U, rks->Ucurr));
  PetscCall(VecCopy(U, rks->Uprev));
  PetscCall(VecCopy(U, rks->U0));

  PetscCall(TSComputeRHSFunction(ts, time, rks->U0, rks->F0));

  /* stage loop */
  for (PetscInt j = 1; j <= s; ++j) {
    /* compute F(U^{(j-1)}) */
    PetscCall(TSComputeRHSFunction(ts, time, rks->Ucurr, rks->Fcurr));

    if (j == 1) {
      /* U^{(1)} = U^{(0)} + tilde_mu[1] * dt * F(U^{(0)}) */
      PetscCall(VecWAXPY(rks->Uwork, tilde_mu[1]*dt, rks->F0, rks->Ucurr));
    } else {
      /* U^{(j)} = mu[j] U^{(j-1)} + nu_j U^{(j-2)} + (1 - mu[j] - nu[j]) U^{(0)} + tilde_mu[j] dt F(U^{(j-1)}) + tilde_gamma[j] dt F(U^{(0)}) */
      PetscCall(VecCopy(rks->Ucurr, rks->Uwork));                     /* Uwork = U^{(j-1)} */
      PetscCall(VecScale(rks->Uwork, mu[j]));                         /* mu[j] * U^{(j-1)} */
      PetscCall(VecAXPY(rks->Uwork, nu[j], rks->Uprev));              /* + nu[j] * U^{(j-2)} */
      PetscCall(VecAXPY(rks->Uwork, 1.0 - mu[j] - nu[j], rks->U0));   /* + (1-mu[j]-nu[j])*U^{(0)} */
      PetscCall(VecAXPY(rks->Uwork, tilde_mu[j]*dt, rks->Fcurr));     /* + tilde_mu[j] * dt * F(U^{(j-1)}) */
      PetscCall(VecAXPY(rks->Uwork, tilde_gamma[j]*dt, rks->F0));     /* + tilde_gamma[j] * dt * F(U^{(0)}) */
    }
    PetscCall(VecCopy(rks->Ucurr, rks->Uprev));
    PetscCall(VecCopy(rks->Uwork, rks->Ucurr));
  }
  PetscCall(VecCopy(rks->Ucurr, U));
  PetscCall(PetscFree5(b, mu, nu, tilde_mu, tilde_gamma));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_RKS(TS ts)
{
  TS_RKS *rks = (TS_RKS*)ts->data;

  PetscFunctionBegin;
  switch (rks->type) {
    case RKS_RKC1: PetscCall(TSRK1Step(ts)); break;
    case RKS_RKC2: PetscCall(TSRK2Step(ts)); break;
    case RKS_RKL1: PetscCall(TSRK1Step(ts)); break;
    case RKS_RKL2: PetscCall(TSRK2Step(ts)); break;
    case RKS_RKG1: PetscCall(TSRK1Step(ts)); break;
    case RKS_RKG2: PetscCall(TSRK2Step(ts)); break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown RKS type");
  }
  {
    PetscReal t, dt;
    PetscCall(TSGetTime(ts,&t));
    PetscCall(TSGetTimeStep(ts,&dt));
    PetscCall(TSSetTime(ts, t + dt));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TSSetUp_RKS(TS ts)
{
  TS_RKS *rks = (TS_RKS*)ts->data;
  Vec U;

  PetscFunctionBegin;
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecDuplicate(U, &rks->U0));
  PetscCall(VecDuplicate(U, &rks->Uprev));
  PetscCall(VecDuplicate(U, &rks->Ucurr));
  PetscCall(VecDuplicate(U, &rks->Uwork));
  PetscCall(VecDuplicate(U, &rks->F0));
  PetscCall(VecDuplicate(U, &rks->Fcurr));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_RKS(TS ts, PetscOptionItems PetscOptionsObject)
{
  TS_RKS *rks = (TS_RKS*)ts->data;
  PetscEnum meth = (PetscEnum)rks->type;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"RKS options");
  PetscCall(PetscOptionsEnum("-ts_rks_type","Super-time-stepper type","",RKSTypes,(PetscEnum)rks->type,(PetscEnum*)&meth,NULL));
  rks->type = (RKSType)meth;
  PetscCall(PetscOptionsInt("-ts_rks_stages","Number of stages per macro-step",NULL,rks->stages,&rks->stages,NULL));
  PetscCall(PetscOptionsReal("-ts_rks_rkg_parameter", "Gegenbauer parameter","",rks->rkg_parameter,&rks->rkg_parameter,NULL));
  PetscCall(PetscOptionsReal("-ts_rks_epsilon", "RKS epsilon","",rks->epsilon,&rks->epsilon,NULL));

  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
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
  TS_RKS *rks = (TS_RKS*)ts->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "RKS method: %s, stages=%d\n", RKSTypes[rks->type], (int)rks->stages));
  PetscFunctionReturn(0);
}

/*MC
      TSRKS - ODE and DAE solver using Runge-Kutta Supertimestepper (RKS) schemes

  The user should provide the right-hand side of the equation
  using `TSSetRHSFunction()`.

  Level: beginner

  Notes:
  The default is `TSRKL1`, it can be changed with -ts_rks_type

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

  rks->type          = RKS_RKL1;
  rks->stages        = 10;
  rks->rkg_parameter = 1.5;
  rks->epsilon       = 0.05;
  PetscFunctionReturn(0);
}