static char help[] = "Biological network from https://link.springer.com/article/10.1007/s42967-023-00297-3\n\n\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscts.h>

/*
    Here we solve the system of PDEs on \Omega \in R^2:

    * dC/dt - D^2 \Delta C - c^2 \nabla p \cross \nabla p + \alpha sqrt(||C||^2_F + eps)^(\gamma-2) C = 0
    * - \nabla \cdot ((r + C) \nabla p) = S

    where:
      C = symmetric 2x2 conductivity tensor
      p = pressure
      S = source

    with natural boundary conditions on \partial\Omega:
      \nabla C \cdot n  = 0
      \nabla ((r + C)\nabla p) \cdot n  = 0

    Parameters:
      D = diffusion constant
      c = activation parameter
      \alpha = metabolic coefficient
      \gamma = metabolic exponent
      r, eps are regularization parameters

    We use Lagrange elements for C_ij and P.
*/

typedef enum _fieldidx {
  C_FIELD_ID = 0,
  P_FIELD_ID,
  NUM_FIELDS
} FieldIdx;

typedef enum _constantidx {
  R_ID = 0,
  EPS_ID,
  ALPHA_ID,
  GAMMA_ID,
  D_ID,
  C2_ID,
  NUM_CONSTANTS
} ConstantIdx;

#define NORM2C(c00, c01, c11) PetscSqr(c00) + 2 * PetscSqr(c01) + PetscSqr(c11)

/* residual for C when tested against basis functions */
static void C_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal   c2       = PetscRealPart(constants[C2_ID]);
  const PetscReal   alpha    = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma    = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal   eps      = PetscRealPart(constants[EPS_ID]);
  const PetscScalar gradp[]  = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};
  const PetscScalar crossp[] = {gradp[0] * gradp[0], gradp[0] * gradp[1], gradp[1] * gradp[1]};
  const PetscScalar C00      = u[uOff[C_FIELD_ID]];
  const PetscScalar C01      = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C11      = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar norm     = NORM2C(C00, C01, C11) + eps;
  const PetscScalar nexp     = (gamma - 2.0) / 2.0;
  const PetscScalar fnorm    = PetscPowScalar(norm, nexp);

  for (PetscInt k = 0; k < 3; k++) f0[k] = u_t[uOff[C_FIELD_ID] + k] - c2 * crossp[k] + alpha * fnorm * u[uOff[C_FIELD_ID] + k];
}

/* Jacobian for C against C basis functions */
static void JC_0_c0c0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   alpha  = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma  = PetscRealPart(constants[GAMMA_ID]);
  const PetscReal   eps    = PetscRealPart(constants[EPS_ID]);
  const PetscScalar C00    = u[uOff[C_FIELD_ID]];
  const PetscScalar C01    = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C11    = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar norm   = NORM2C(C00, C01, C11) + eps;
  const PetscScalar nexp   = (gamma - 2.0) / 2.0;
  const PetscScalar fnorm  = PetscPowScalar(norm, nexp);
  const PetscScalar dfnorm = nexp * PetscPowScalar(norm, nexp - 1.0);
  const PetscScalar dC[]   = {2 * C00, 4 * C01, 2 * C11};

  for (PetscInt k = 0; k < 3; k++) {
    for (PetscInt j = 0; j < 3; j++) J[k * 3 + j] = alpha * dfnorm * dC[j] * u[uOff[C_FIELD_ID] + k];
    J[k * 3 + k] += alpha * fnorm + u_tShift;
  }
}

/* Jacobian for C against C basis functions and gradients of P basis functions */
static void JC_0_c0p1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   c2      = PetscRealPart(constants[C2_ID]);
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  J[0] = -c2 * 2 * gradp[0];
  J[1] = 0.0;
  J[2] = -c2 * gradp[1];
  J[3] = -c2 * gradp[0];
  J[4] = 0.0;
  J[5] = -c2 * 2 * gradp[1];
}

/* residual for C when tested against gradients of basis functions */
static void C_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt d = 0; d < 2; d++) f1[k * 2 + d] = PetscSqr(D) * u_x[uOff_x[C_FIELD_ID] + k * 2 + d];
}

/* Jacobian for C against gradients of C basis functions */
static void JC_1_c1c1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal D = PetscRealPart(constants[D_ID]);
  for (PetscInt k = 0; k < 3; k++)
    for (PetscInt d = 0; d < 2; d++) J[k * (3 + 1) * 2 * 2 + d * 2 + d] = PetscSqr(D);
}

/* residual for P when tested against basis functions.
   The source term always comes from the auxiliary vec because it needs to have zero mean */
static void P_0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscScalar S = a[aOff[P_FIELD_ID]];

  f0[0] = -S;
}

/* residual for P when tested against gradients of basis functions */
static void P_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal   r       = PetscRealPart(constants[R_ID]);
  const PetscScalar C00     = u[uOff[C_FIELD_ID]];
  const PetscScalar C01     = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10     = C01;
  const PetscScalar C11     = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  f1[0] = (C00 + r) * gradp[0] + C01 * gradp[1];
  f1[1] = C10 * gradp[0] + (C11 + r) * gradp[1];
}

/* Same as above for the P-only subproblem for initial conditions: the conductivity values come from the auxiliary vec */
static void P_1_aux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal   r       = PetscRealPart(constants[R_ID]);
  const PetscScalar C00     = a[aOff[C_FIELD_ID]];
  const PetscScalar C01     = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar C10     = C01;
  const PetscScalar C11     = a[aOff[C_FIELD_ID] + 2];
  const PetscScalar gradp[] = {u_x[uOff_x[0]], u_x[uOff_x[0] + 1]};

  f1[0] = (C00 + r) * gradp[0] + C01 * gradp[1];
  f1[1] = C10 * gradp[0] + (C11 + r) * gradp[1];
}

/* Jacobian for P against gradients of P basis functions */
static void JP_1_p1p1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   r   = PetscRealPart(constants[R_ID]);
  const PetscScalar C00 = u[uOff[C_FIELD_ID]];
  const PetscScalar C01 = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10 = C01;
  const PetscScalar C11 = u[uOff[C_FIELD_ID] + 2];

  J[0] = C00 + r;
  J[1] = C01;
  J[2] = C10;
  J[3] = C11 + r;
}

/* Same as above for the P-only subproblem for initial conditions */
static void JP_1_p1p1_aux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscReal   r   = PetscRealPart(constants[R_ID]);
  const PetscScalar C00 = a[aOff[C_FIELD_ID]];
  const PetscScalar C01 = a[aOff[C_FIELD_ID] + 1];
  const PetscScalar C10 = C01;
  const PetscScalar C11 = a[aOff[C_FIELD_ID] + 2];

  J[0] = C00 + r;
  J[1] = C01;
  J[2] = C10;
  J[3] = C11 + r;
}

/* Jacobian for P against gradients of P basis functions and C basis functions */
static void JP_1_p1c0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar J[])
{
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  J[0] = gradp[0];
  J[1] = 0;
  J[2] = gradp[1];
  J[3] = gradp[0];
  J[4] = 0;
  J[5] = gradp[1];
}

/* the source term S(x) = exp(-500*||x - x0||^2) */
static PetscErrorCode source_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscReal *x0 = (PetscReal *)ctx;
  PetscReal  n  = 0;
  for (PetscInt d = 0; d < dim; ++d) n += (x[d] - x0[d]) * (x[d] - x0[d]);
  u[0] = PetscExpReal(-500 * n);
  return PETSC_SUCCESS;
}

/* functionals to be integrated: average -> \int_\Omega u dx */
static void average(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = u[uOff[P_FIELD_ID]];
}

/* functionals to be integrated: energy -> D^2/2 * ||\nabla C||^2 + c^2\nabla p * (r + C) * \nabla p + \alpha/ \gamma * ||C||^\gamma */
static void energy(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  const PetscReal   D         = PetscRealPart(constants[D_ID]);
  const PetscReal   c2        = PetscRealPart(constants[C2_ID]);
  const PetscReal   r         = PetscRealPart(constants[R_ID]);
  const PetscReal   alpha     = PetscRealPart(constants[ALPHA_ID]);
  const PetscReal   gamma     = PetscRealPart(constants[GAMMA_ID]);
  const PetscScalar C00       = u[uOff[C_FIELD_ID]];
  const PetscScalar C01       = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10       = C01;
  const PetscScalar C11       = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar gradp[]   = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};
  const PetscScalar gradC00[] = {u_x[uOff_x[C_FIELD_ID] + 0], u_x[uOff_x[C_FIELD_ID] + 1]};
  const PetscScalar gradC01[] = {u_x[uOff_x[C_FIELD_ID] + 2], u_x[uOff_x[C_FIELD_ID] + 3]};
  const PetscScalar gradC11[] = {u_x[uOff_x[C_FIELD_ID] + 4], u_x[uOff_x[C_FIELD_ID] + 5]};
  const PetscScalar normC     = NORM2C(C00, C01, C11);
  const PetscScalar normgradC = NORM2C(gradC00[0], gradC01[0], gradC11[0]) + NORM2C(gradC00[1], gradC01[1], gradC11[1]);
  const PetscScalar nexp      = gamma / 2.0;

  const PetscScalar t0 = PetscSqr(D) / 2.0 * normgradC;
  const PetscScalar t1 = c2 * (gradp[0] * ((C00 + r) * gradp[0] + C01 * gradp[1]) + gradp[1] * (C10 * gradp[0] + (C11 + r) * gradp[1]));
  const PetscScalar t2 = alpha / gamma * PetscPowScalar(normC, nexp);

  obj[0] = t0 + t1 + t2;
}

/* initial conditions for C: eq. 16 */
static PetscErrorCode initial_conditions_C_0(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 1;
  u[1] = 0;
  u[2] = 1;
  return PETSC_SUCCESS;
}

/* initial conditions for C: eq. 17 */
static PetscErrorCode initial_conditions_C_1(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscReal x = xx[0];
  const PetscReal y = xx[1];

  u[0] = (2 - PetscAbsReal(x + y)) * PetscExpReal(-10 * PetscAbsReal(x - y));
  u[1] = 0;
  u[2] = (2 - PetscAbsReal(x + y)) * PetscExpReal(-10 * PetscAbsReal(x - y));
  return PETSC_SUCCESS;
}

/* initial conditions for C: eq. 18 */
static PetscErrorCode initial_conditions_C_2(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0;
  u[1] = 0;
  u[2] = 0;
  return PETSC_SUCCESS;
}

/* functionals to be sampled: C * \grad p */
static void flux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscScalar C00     = u[uOff[C_FIELD_ID]];
  const PetscScalar C01     = u[uOff[C_FIELD_ID] + 1];
  const PetscScalar C10     = C01;
  const PetscScalar C11     = u[uOff[C_FIELD_ID] + 2];
  const PetscScalar gradp[] = {u_x[uOff_x[P_FIELD_ID]], u_x[uOff_x[P_FIELD_ID] + 1]};

  f[0] = C00 * gradp[0] + C01 * gradp[1];
  f[1] = C10 * gradp[0] + C11 * gradp[1];
}

/* functionals to be sampled: zero function */
static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt d = 0; d < Nc; ++d) u[d] = 0.0;
  return PETSC_SUCCESS;
}

/* functionals to be sampled: constant function */
static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < Nc; ++d) u[d] = 1.0;
  return PETSC_SUCCESS;
}

/* application context: customizable parameters */
typedef struct {
  PetscReal r;
  PetscReal eps;
  PetscReal alpha;
  PetscReal gamma;
  PetscReal D;
  PetscReal c;
  PetscInt  ic_num;
  PetscInt  source_num;
  PetscReal x0[2];
  PetscBool restart;
  char      restart_filename[PETSC_MAX_PATH_LEN];
} AppCtx;

/* process command line options */
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt dim = PETSC_STATIC_ARRAY_LENGTH(options->x0);

  PetscFunctionBeginUser;
  options->r          = 1.e-1;
  options->eps        = 1.e-3;
  options->alpha      = 0.75;
  options->gamma      = 0.75;
  options->c          = 5;
  options->D          = 1.e-2;
  options->ic_num     = 0;
  options->source_num = 0;
  options->x0[0]      = 0.25;
  options->x0[1]      = 0.25;
  options->restart    = PETSC_FALSE;

  PetscOptionsBegin(comm, "", __FILE__, "DMPLEX");
  PetscCall(PetscOptionsReal("-alpha", "alpha", __FILE__, options->alpha, &options->alpha, NULL));
  PetscCall(PetscOptionsReal("-gamma", "gamma", __FILE__, options->gamma, &options->gamma, NULL));
  PetscCall(PetscOptionsReal("-c", "c", __FILE__, options->c, &options->c, NULL));
  PetscCall(PetscOptionsReal("-d", "D", __FILE__, options->D, &options->D, NULL));
  PetscCall(PetscOptionsReal("-eps", "eps", __FILE__, options->eps, &options->eps, NULL));
  PetscCall(PetscOptionsReal("-r", "r", __FILE__, options->r, &options->r, NULL));
  PetscCall(PetscOptionsRealArray("-x0", "x0", __FILE__, options->x0, &dim, NULL));
  PetscCall(PetscOptionsInt("-ic_num", "ic_num", __FILE__, options->ic_num, &options->ic_num, NULL));
  PetscCall(PetscOptionsInt("-source_num", "source_num", __FILE__, options->source_num, &options->source_num, NULL));
  PetscCall(PetscOptionsString("-restart", "filename with data for restarting", __FILE__, options->restart_filename, options->restart_filename, PETSC_MAX_PATH_LEN, &options->restart));
  PetscOptionsEnd();

  PetscCall(PetscPrintf(comm, "----------------------------\n"));
  PetscCall(PetscPrintf(comm, "Simulation parameters:\n"));
  PetscCall(PetscPrintf(comm, "  r    : %g\n", (double)options->r));
  PetscCall(PetscPrintf(comm, "  eps  : %g\n", (double)options->eps));
  PetscCall(PetscPrintf(comm, "  alpha: %g\n", (double)options->alpha));
  PetscCall(PetscPrintf(comm, "  gamma: %g\n", (double)options->gamma));
  PetscCall(PetscPrintf(comm, "  D    : %g\n", (double)options->D));
  PetscCall(PetscPrintf(comm, "  c    : %g\n", (double)options->c));
  PetscCall(PetscPrintf(comm, "  IC   : %" PetscInt_FMT "\n", options->ic_num));
  PetscCall(PetscPrintf(comm, "  S    : %" PetscInt_FMT "\n", options->source_num));
  PetscCall(PetscPrintf(comm, "  x0   : (%g,%g)\n", (double)options->x0[0], (double)options->x0[1]));
  PetscCall(PetscPrintf(comm, "----------------------------\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Create mesh by command line options */
static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  PetscFunctionBeginUser;
  if (ctx->restart) {
    PetscCall(DMPlexCreateFromFile(comm, ctx->restart_filename, NULL, PETSC_TRUE, dm));
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
  }
  PetscCall(DMSetFromOptions(*dm));
  {
    char      convType[256];
    PetscBool flg;
    PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");
    PetscCall(PetscOptionsFList("-dm_plex_convert_type", "Convert DMPlex to another format", __FILE__, DMList, DMPLEX, convType, 256, &flg));
    PetscOptionsEnd();
    if (flg) {
      DM dmConv;
      PetscCall(DMConvert(*dm, convType, &dmConv));
      if (dmConv) {
        PetscCall(DMDestroy(dm));
        *dm = dmConv;
        PetscCall(DMSetFromOptions(*dm));
        PetscCall(DMSetUp(*dm));
      }
    }
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* callback for the creation of the pressure null space */
static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
  Vec vec;
  PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero};

  PetscFunctionBeginUser;
  funcs[nfield] = constant;
  PetscCall(DMCreateGlobalVector(dm, &vec));
  PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec));
  PetscCall(VecNormalize(vec, NULL));
  PetscCall(PetscObjectSetName((PetscObject)vec, "Pressure Null Space"));
  PetscCall(VecViewFromOptions(vec, NULL, "-pressure_nullspace_view"));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullSpace));
  /* break ref cycles */
  PetscCall(VecSetDM(vec, NULL));
  PetscCall(VecDestroy(&vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* customize residuals and Jacobians */
static PetscErrorCode SetupProblem(DM dm, AppCtx *ctx)
{
  PetscDS     ds;
  PetscInt    cdim, dim, id = 1;
  PetscScalar constants[NUM_CONSTANTS], vals[NUM_FIELDS];
  void       *ctxs[NUM_FIELDS];
  DM          dmAux;
  Vec         u, lu;
  IS          is;
  PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

  PetscFunctionBeginUser;
  constants[R_ID]     = ctx->r;
  constants[EPS_ID]   = ctx->eps;
  constants[ALPHA_ID] = ctx->alpha;
  constants[GAMMA_ID] = ctx->gamma;
  constants[D_ID]     = ctx->D;
  constants[C2_ID]    = ctx->c * ctx->c;

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCheck(dim == 2 && cdim == 2, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only for 2D meshes");
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  PetscCall(PetscDSSetImplicit(ds, C_FIELD_ID, PETSC_TRUE));
  PetscCall(PetscDSSetImplicit(ds, P_FIELD_ID, PETSC_TRUE));
  PetscCall(PetscDSSetResidual(ds, C_FIELD_ID, C_0, C_1));
  PetscCall(PetscDSSetResidual(ds, P_FIELD_ID, P_0, P_1));
  PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, C_FIELD_ID, JC_0_c0c0, NULL, NULL, JC_1_c1c1));
  PetscCall(PetscDSSetJacobian(ds, C_FIELD_ID, P_FIELD_ID, NULL, JC_0_c0p1, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, C_FIELD_ID, NULL, NULL, JP_1_p1c0, NULL));
  PetscCall(PetscDSSetJacobian(ds, P_FIELD_ID, P_FIELD_ID, NULL, NULL, NULL, JP_1_p1p1));

  /* Project source function and make it zero-mean */
  switch (ctx->source_num) {
  case 0:
    funcs[P_FIELD_ID] = source_0;
    ctxs[P_FIELD_ID]  = ctx->x0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unknwon source");
  }
  funcs[C_FIELD_ID] = zero;
  ctxs[C_FIELD_ID]  = NULL;
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(DMProjectFunction(dm, 0, funcs, ctxs, INSERT_ALL_VALUES, u));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, average));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals, NULL));
  PetscCall(VecShift(u, -vals[P_FIELD_ID]));
  id = C_FIELD_ID;
  PetscCall(DMCreateSubDM(dm, 1, &id, &is, NULL));
  PetscCall(VecISSet(u, is, 0));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscDSSetObjective(ds, P_FIELD_ID, NULL));

  /* Attach pressure nullspace */
  PetscCall(DMSetNullSpaceConstructor(dm, P_FIELD_ID, CreatePressureNullSpace));

  /* Attach source vector as auxiliary vector:
     Use a different DM to break ref cycles */
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMCopyDisc(dm, dmAux));
  PetscCall(DMCreateLocalVector(dmAux, &lu));
  PetscCall(DMDestroy(&dmAux));
  PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, lu));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, lu));
  PetscCall(VecViewFromOptions(lu, NULL, "-aux_view"));
  PetscCall(VecDestroy(&lu));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* create discrete spaces */
static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM           plex, cdm = dm;
  PetscFE      feC, feP;
  PetscBool    simplex;
  PetscInt     dim;
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm);
  MatNullSpace nsp;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCall(DMDestroy(&plex));

  /* We model Cij in H^1 with Cij = Cji -> dim*(dim+1)/2 components */
  PetscCall(PetscFECreateDefault(comm, dim, (dim * (dim + 1)) / 2, simplex, "c_", -1, &feC));
  PetscCall(PetscObjectSetName((PetscObject)feC, "conductivity"));
  PetscCall(PetscFECreateDefault(comm, dim, 1, simplex, "p_", -1, &feP));
  PetscCall(PetscObjectSetName((PetscObject)feP, "pressure"));
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nsp));
  PetscCall(PetscObjectCompose((PetscObject)feP, "nullspace", (PetscObject)nsp));
  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(PetscFECopyQuadrature(feC, feP));

  PetscCall(DMSetNumFields(dm, 2));
  PetscCall(DMSetField(dm, C_FIELD_ID, NULL, (PetscObject)feC));
  PetscCall(DMSetField(dm, P_FIELD_ID, NULL, (PetscObject)feP));
  PetscCall(PetscFEDestroy(&feC));
  PetscCall(PetscFEDestroy(&feP));
  PetscCall(DMCreateDS(dm));

  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(SetupProblem(cdm, ctx));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute initial conditions and exclude pressure from local truncation error */
static PetscErrorCode SetInitialConditionsAndTolerances(TS ts, AppCtx *ctx)
{
  DM         dm;
  Vec        u, p, lsource, subaux, vatol, vrtol;
  PetscReal  t, atol, rtol;
  PetscInt   fields[NUM_FIELDS] = {C_FIELD_ID, P_FIELD_ID};
  IS         isp;
  DM         dmp;
  VecScatter sctp;
  PetscDS    ds;
  SNES       snes;
  PetscErrorCode (*funcs[NUM_FIELDS])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution_"));
  PetscCall(TSGetTime(ts, &t));
  switch (ctx->ic_num) {
  case 0:
    funcs[C_FIELD_ID] = initial_conditions_C_0;
    break;
  case 1:
    funcs[C_FIELD_ID] = initial_conditions_C_1;
    break;
  case 2:
    funcs[C_FIELD_ID] = initial_conditions_C_2;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Unknwon IC");
  }
  funcs[P_FIELD_ID] = zero;
  PetscCall(DMProjectFunction(dm, t, funcs, NULL, INSERT_ALL_VALUES, u));

  /* Compute initial P consistent with C: since we are solving a DAE,
     once the initial conditions for the differential
     variables are set, we need to compute the corresponding value for the
     algebraic variables. We do so by creating a subDM for the pressure only
     and solve a static problem with SNES */
  PetscCall(DMCreateSubDM(dm, NUM_FIELDS - 1, fields + 1, &isp, &dmp));
  PetscCall(DMGetDS(dmp, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, P_0, P_1_aux));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, JP_1_p1p1_aux));
  PetscCall(DMPlexSetSNESLocalFEM(dmp, NULL, NULL, NULL));

  /* pass conductivity and source information via auxiliary data */
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &lsource));
  PetscCall(DMCreateLocalVector(dm, &subaux));
  PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, subaux));
  PetscCall(VecAXPY(subaux, 1.0, lsource));
  PetscCall(VecViewFromOptions(subaux, NULL, "-initial_aux_view"));
  PetscCall(DMSetAuxiliaryVec(dmp, NULL, 0, 0, subaux));
  PetscCall(VecDestroy(&subaux));
  PetscCall(DMCreateGlobalVector(dmp, &p));
  PetscCall(VecScatterCreate(u, isp, p, NULL, &sctp));

  /* Solve the subproblem */
  PetscCall(VecSet(p, 0.0));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dmp), &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "initial_"));
  PetscCall(SNESSetDM(snes, dmp));
  PetscCall(DMDestroy(&dmp));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetUp(snes));
  PetscCall(SNESSolve(snes, NULL, p));
  PetscCall(SNESDestroy(&snes));

  /* scatter from pressure only to full space */
  PetscCall(VecScatterBegin(sctp, p, u, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(sctp, p, u, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&p));
  PetscCall(VecScatterDestroy(&sctp));
  PetscCall(VecDestroy(&u));

  /* exclude pressure from computation of the LTE */
  PetscCall(DMCreateGlobalVector(dm, &vatol));
  PetscCall(DMCreateGlobalVector(dm, &vrtol));
  PetscCall(TSGetTolerances(ts, &atol, NULL, &rtol, NULL));
  PetscCall(VecSet(vatol, atol));
  PetscCall(VecISSet(vatol, isp, -1));
  PetscCall(VecSet(vrtol, rtol));
  PetscCall(VecISSet(vrtol, isp, -1));
  PetscCall(TSSetTolerances(ts, atol, vatol, rtol, vrtol));
  PetscCall(VecDestroy(&vatol));
  PetscCall(VecDestroy(&vrtol));
  PetscCall(ISDestroy(&isp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Monitor energy functional */
static PetscErrorCode MonitorEnergy(TS ts, PetscInt steps, PetscReal time, Vec u, void *ctx)
{
  PetscScalar vals[NUM_FIELDS];
  DM          dm;
  PetscDS     ds;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, energy));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, vals, NULL));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "%" PetscInt_FMT " TS: time %g, energy %g\n", steps, time, vals[0]));
  PetscCall(PetscDSSetObjective(ds, C_FIELD_ID, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecViewFlux(Vec u, const char *opts)
{
  Vec        fluxVec;
  DM         dmFlux, dm, plex;
  PetscSpace P;
  PetscInt   dim, k;
  PetscFE    feC, feFluxC;
  PetscBool  simplex, has;

  void (*funcs[1])(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]) = {flux};

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsHasName(NULL, NULL, opts, &has));
  if (!has) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetDM(u, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetField(dm, C_FIELD_ID, NULL, (PetscObject *)&feC));
  PetscCall(PetscFEGetBasisSpace(feC, &P));
  PetscCall(PetscSpaceGetDegree(P, &k, NULL));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscFECreateLagrange(PetscObjectComm((PetscObject)dm), dim, dim, simplex, k, -1, &feFluxC));
  PetscCall(PetscFECopyQuadrature(feC, feFluxC));
  PetscCall(DMClone(dm, &dmFlux));
  PetscCall(DMSetNumFields(dmFlux, 1));
  PetscCall(DMSetField(dmFlux, 0, NULL, (PetscObject)feFluxC));
  PetscCall(DMCreateDS(dmFlux));
  PetscCall(PetscFEDestroy(&feFluxC));

  PetscCall(DMGetGlobalVector(dmFlux, &fluxVec));
  PetscCall(DMProjectField(dmFlux, 0.0, u, funcs, INSERT_VALUES, fluxVec));
  PetscCall(VecViewFromOptions(fluxVec, NULL, opts));
  PetscCall(DMRestoreGlobalVector(dmFlux, &fluxVec));
  PetscCall(DMDestroy(&dmFlux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm;
  TS     ts;
  Vec    u;
  AppCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm, &ctx));
  PetscCall(SetupDiscretization(dm, &ctx));
  PetscCall(DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL));
  PetscCall(DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL));
  PetscCall(DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(TSSetMaxTime(ts, 10.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSMonitorSet(ts, MonitorEnergy, NULL, NULL));
  PetscCall(TSSetMaxSNESFailures(ts, -1));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(SetInitialConditionsAndTolerances(ts, &ctx));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSGetSolution(ts, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-final_view"));
  PetscCall(VecViewFlux(u, "-final_flux_view"));

  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2}}
    args: -dm_plex_box_faces 3,3 -pc_type svd -dm_refine 1 -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 1 -initial_snes_test_jacobian -snes_test_jacobian -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_simplex 0

  test:
    requires: p4est
    suffix: 0_p4est
    nsize: {{1 2}}
    args: -dm_plex_box_faces 3,3 -pc_type svd -dm_refine 1 -c_petscspace_degree 1 -p_petscspace_degree 1 -ts_max_steps 1 -initial_snes_test_jacobian -snes_test_jacobian -initial_snes_type ksponly -snes_type ksponly -petscpartitioner_type simple -dm_plex_convert_type p4est -dm_plex_simplex 0

TEST*/
