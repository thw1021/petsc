static char help[] = "An underconstrained nonlinear elasticity problem in 3d with simplicial finite elements.\n\
We solve for an equilibrium configuration of a compressible Neo-Hookean solid.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef enum {
  RUN_FULL,
  RUN_TEST
} RunType;

typedef struct {
  RunType         runType;     /* Whether to run tests, or solve the full problem */
  PetscReal       lambda;      /* bulk modulus */
  PetscReal       nu;          /* Poisson's ratio */
  PetscBool       partial_bcs; /* Apply partial boundary conditions at the top and bottom */
  SNESJacobianFn *inner_jacobian_fn;
  void           *inner_jacobian_ctx;
} AppCtx;

static inline PetscReal shear_modulus(PetscReal lambda, PetscReal nu)
{
  return lambda * (1 - 2 * nu) / (2 * nu);
}

static inline PetscReal Determinant(const PetscReal J[])
{
  // clang-format off
  return J[0 * 3 + 0] * (J[1 * 3 + 1] * J[2 * 3 + 2] - J[1 * 3 + 2] * J[2 * 3 + 1])
       + J[0 * 3 + 1] * (J[1 * 3 + 2] * J[2 * 3 + 0] - J[1 * 3 + 0] * J[2 * 3 + 2])
       + J[0 * 3 + 2] * (J[1 * 3 + 0] * J[2 * 3 + 1] - J[1 * 3 + 1] * J[2 * 3 + 0]);
  // clang-format on
}

static inline void Adjugate(PetscReal Adj[], const PetscReal A[])
{
  Adj[0 * 3 + 0] = PetscRealPart(A[1 * 3 + 1] * A[2 * 3 + 2] - A[1 * 3 + 2] * A[2 * 3 + 1]);
  Adj[0 * 3 + 1] = PetscRealPart(A[0 * 3 + 2] * A[2 * 3 + 1] - A[0 * 3 + 1] * A[2 * 3 + 2]);
  Adj[0 * 3 + 2] = PetscRealPart(A[0 * 3 + 1] * A[1 * 3 + 2] - A[0 * 3 + 2] * A[1 * 3 + 1]);
  Adj[1 * 3 + 0] = PetscRealPart(A[1 * 3 + 2] * A[2 * 3 + 0] - A[1 * 3 + 0] * A[2 * 3 + 2]);
  Adj[1 * 3 + 1] = PetscRealPart(A[0 * 3 + 0] * A[2 * 3 + 2] - A[0 * 3 + 2] * A[2 * 3 + 0]);
  Adj[1 * 3 + 2] = PetscRealPart(A[0 * 3 + 2] * A[1 * 3 + 0] - A[0 * 3 + 0] * A[1 * 3 + 2]);
  Adj[2 * 3 + 0] = PetscRealPart(A[1 * 3 + 0] * A[2 * 3 + 1] - A[1 * 3 + 1] * A[2 * 3 + 0]);
  Adj[2 * 3 + 1] = PetscRealPart(A[0 * 3 + 1] * A[2 * 3 + 0] - A[0 * 3 + 0] * A[2 * 3 + 1]);
  Adj[2 * 3 + 2] = PetscRealPart(A[0 * 3 + 0] * A[1 * 3 + 1] - A[0 * 3 + 1] * A[1 * 3 + 0]);
}

static inline void Inverse(PetscReal AInv[], const PetscReal A[])
{
  PetscReal det = Determinant(A);

  Adjugate(AInv, A);
  for (PetscInt i = 0; i < 9; i++) AInv[i] /= det;
}

// det(H + I) - 1
static inline PetscReal ShiftedDeterminant(const PetscScalar H[])
{
  PetscReal u00 = PetscRealPart(H[0 * 3 + 0]);
  PetscReal u01 = PetscRealPart(H[0 * 3 + 1]);
  PetscReal u02 = PetscRealPart(H[0 * 3 + 2]);
  PetscReal u10 = PetscRealPart(H[1 * 3 + 0]);
  PetscReal u11 = PetscRealPart(H[1 * 3 + 1]);
  PetscReal u12 = PetscRealPart(H[1 * 3 + 2]);
  PetscReal u20 = PetscRealPart(H[2 * 3 + 0]);
  PetscReal u21 = PetscRealPart(H[2 * 3 + 1]);
  PetscReal u22 = PetscRealPart(H[2 * 3 + 2]);

  // clang-format off
  return u00 * u11 * u22 + u00 * u11 + u00 * u22 + u11 * u22 + u00 + u11 + u22 // p = (0,1,2) [product of diagonal terms - 1]
         - (u00 + 1) *  u12      *  u21                                        // p = (0,2,1)
         -  u01      *  u10      * (u22 + 1)                                   // p = (1,0,2)
         +  u01      *  u12      *  u20                                        // p = (1,2,0)
         +  u02      *  u10      *  u21                                        // p = (2,0,1)
         -  u02      * (u11 + 1) *  u20;                                       // p = (2,1,0)
  // clang-format on
}

static inline void Product(PetscReal AB[], const PetscReal A[], const PetscReal B[])
{
  for (PetscInt i = 0; i < 9; i++) AB[i] = 0;

  for (PetscInt k = 0; k < 3; k++) {
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) AB[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    }
  }
}

// Cauchy-Green tensor C and Green-Lagrange strain E = (1/2) * (C - I) = (1/2) (H + H^T + H^T H)
static inline void CauchyTensors(PetscReal C[], PetscReal E[], const PetscScalar H[])
{
  for (PetscInt i = 0; i < 3; i++) {
    for (PetscInt j = 0; j < 3; j++) { C[i * 3 + j] = PetscRealPart(H[i * 3 + j] + H[j * 3 + i]); }
  }

  for (PetscInt k = 0; k < 3; k++) {
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) { C[i * 3 + j] += PetscRealPart(H[k * 3 + i] * H[k * 3 + j]); }
    }
  }

  for (PetscInt i = 0; i < 9; i++) E[i] = 0.5 * C[i];
  for (PetscInt i = 0; i < 3; i++) C[i * 3 + i] += 1;
}

// For a jacobian 4-tensor C that would like to contract like d_X v : C : d_X u, it must be stored in i,k,j,l order for PetscFE's assembly routines
#define IDX(i, j, k, l) ((i * 3 + k) * 3 + j) * 3 + l

static inline void NeoHookeanFirstPiolaKirchhoffStress(PetscReal P[], PetscReal DP[], const PetscScalar H[], PetscReal lambda, PetscReal mu)
{
  PetscReal C[9], Cinv[9], E[9], F[9], S[9];
  PetscReal Jminus1;

  for (PetscInt i = 0; i < 9; i++) F[i] = PetscRealPart(H[i]);
  for (PetscInt i = 0; i < 3; i++) F[i * 3 + i] += 1;

  Jminus1 = ShiftedDeterminant(H);
  CauchyTensors(C, E, H);
  Inverse(Cinv, C);

  Product(S, Cinv, E);
  // (lambda / 2) * (J - 1)^2 * C^-1 + 2 * mu * C^-1 * E
  for (PetscInt i = 0; i < 9; i++) S[i] = 0.5 * lambda * Jminus1 * (Jminus1 + 2) * Cinv[i] + 2 * mu * S[i];

  Product(P, F, S);
  // NOTE: the 4-D tensor dP tensor has to be store in C_IKJL order
  if (DP) {
    PetscReal J = Jminus1 + 1;
    PetscReal FCinv[9];
    PetscReal alpha = mu - 0.5 * lambda * Jminus1 * (Jminus1 + 2);

    Product(FCinv, F, Cinv);

    for (PetscInt i = 0; i < 81; i++) DP[i] = 0;

    // d_X v : d_X u S
    for (PetscInt j = 0; j < 3; j++) {
      for (PetscInt l = 0; l < 3; l++) {
        PetscReal S_jl = 0.5 * (S[j * 3 + l] + S[l * 3 + j]);

        for (PetscInt ik = 0; ik < 3; ik++) DP[IDX(ik, j, ik, l)] += S_jl;
      }
    }

    // lambda * J^2 * (d_X v : F C^-1) (C^-1 : sym(F^T d_X u) = C^-1 : F^T d_X u = F C^-1 : d_X u)
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        PetscReal FCinv_ij = FCinv[i * 3 + j];

        for (PetscInt k = 0; k < 3; k++) {
          for (PetscInt l = 0; l < 3; l++) {
            PetscReal FCinv_kl = FCinv[k * 3 + l];

            DP[IDX(i, j, k, l)] += lambda * J * J * FCinv_ij * FCinv_kl;
          }
        }
      }
    }

    // alpha * d_X v : F C^-1 (d_X u)^T F C^-1 = (FC^-1)^T d_X V : ((FC^-1)^T d_X u)^T
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        for (PetscInt k = 0; k < 3; k++) {
          for (PetscInt l = 0; l < 3; l++) { DP[IDX(i, j, k, l)] += alpha * FCinv[i * 3 + l] * FCinv[k * 3 + j]; }
        }
      }
    }

    // alpha * d_X v : F C^-1 F^T d_X u C^-1 = d_X v : d_X u C^-1
    for (PetscInt j = 0; j < 3; j++) {
      for (PetscInt l = 0; l < 3; l++) {
        PetscReal Cinv_jl = 0.5 * (Cinv[j * 3 + l] + Cinv[l * 3 + j]);

        for (PetscInt ik = 0; ik < 3; ik++) DP[IDX(ik, j, ik, l)] += alpha * Cinv_jl;
      }
    }
  }
}

// a nonaffine deformation to use as the initial guess
static PetscErrorCode spherical_transformation_offset(PetscInt dim, PetscReal tim, const PetscReal x[], PetscInt nf, PetscScalar *u, void *ctx)
{
  PetscReal r     = x[0] + 1;
  PetscReal theta = x[1];
  PetscReal phi   = x[2];
  PetscReal XYZ[3];

  XYZ[0] = r * cos(theta) * cos(phi);
  XYZ[1] = r * sin(theta) * cos(phi);
  XYZ[2] = r * sin(phi);

  u[0] = XYZ[0] - x[0];
  u[1] = XYZ[1] - x[1];
  u[2] = XYZ[2] - x[2];

  return PETSC_SUCCESS;
}

// a nonaffine deformation to use as the initial guess
static PetscErrorCode polar_transformation_offset_yz(PetscInt dim, PetscReal tim, const PetscReal x[], PetscInt nf, PetscScalar *u, void *ctx)
{
  PetscReal r     = x[1] + 1;
  PetscReal theta = x[2];
  PetscReal YZ[2];

  YZ[0] = r * cos(theta);
  YZ[1] = r * sin(theta);

  u[0] = 0;
  u[1] = YZ[0] - x[1];
  u[2] = YZ[1] - x[2];

  return PETSC_SUCCESS;
}

static PetscErrorCode zero_displacement(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  for (PetscInt d = 0; d < 3; d++) u[d] = 0;
  return PETSC_SUCCESS;
}

static PetscErrorCode elasticityMaterial(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  u[0]         = user->lambda;
  u[1]         = user->nu;
  return PETSC_SUCCESS;
}

static void v_x_dot_stress(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar v_x_dot[])
{
  const PetscReal lambda = PetscRealPart(a[0]), nu = PetscRealPart(a[1]);
  PetscReal       mu;
  PetscReal       P[9];

  mu = shear_modulus(lambda, nu);
  NeoHookeanFirstPiolaKirchhoffStress(P, NULL, u_x, lambda, mu);
  for (PetscInt i = 0; i < 9; i++) v_x_dot[i] = P[i];
}

static void d_v_x_dot_stress(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar d_v_x_dot[])
{
  const PetscReal lambda = PetscRealPart(a[0]), nu = PetscRealPart(a[1]);
  PetscReal       mu;
  PetscReal       P[9], DP[81];

  mu = shear_modulus(lambda, nu);
  NeoHookeanFirstPiolaKirchhoffStress(P, DP, u_x, lambda, mu);
  for (PetscInt i = 0; i < 81; i++) d_v_x_dot[i] = DP[i];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->runType     = RUN_FULL;
  options->lambda      = 1.0;
  options->nu          = 0.25;
  options->partial_bcs = PETSC_FALSE;
  PetscOptionsBegin(comm, "", "Underconstrained nonlinear elasticity problem options", "DMPLEX");
  PetscCall(PetscOptionsReal("-bulk_modulus", "The bulk modulus", "ex79.c", options->lambda, &options->lambda, NULL));
  PetscCall(PetscOptionsReal("-poissons_ratio", "Poisson's ratio", "ex79.c", options->nu, &options->nu, NULL));
  PetscCall(PetscOptionsBool("-partial_bcs", "Impose normal boundary conditions on top and bottom faces", "ex79.c", options->partial_bcs, &options->partial_bcs, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupProblem(DM dm, PetscInt dim, AppCtx *user)
{
  PetscDS ds;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, v_x_dot_stress));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, d_v_x_dot_stress));

  if (user->partial_bcs) {
    DMLabel  faceSets;
    PetscInt faceMarkerLeft  = 6;
    PetscInt faceMarkerRight = 5;
    PetscInt x_comp          = 0;
    PetscInt values[2]       = {faceMarkerLeft, faceMarkerRight};

    PetscCall(DMGetLabel(dm, "Face Sets", &faceSets));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "fixed", faceSets, 2, values, 0, 1, &x_comp, (void (*)(void))zero_displacement, NULL, user, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupMaterial(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*matFuncs[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {elasticityMaterial};
  Vec   A;
  void *ctxs[2];

  PetscFunctionBegin;
  ctxs[0] = user;
  ctxs[1] = user;
  PetscCall(DMCreateLocalVector(dmAux, &A));
  PetscCall(DMProjectFunctionLocal(dmAux, 0.0, matFuncs, ctxs, INSERT_ALL_VALUES, A));
  PetscCall(DMSetAuxiliaryVec(dm, NULL, 0, 0, A));
  PetscCall(VecDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupAuxDM(DM dm, PetscInt NfAux, PetscFE feAux[], AppCtx *user)
{
  DM       dmAux, coordDM;
  PetscInt f;

  PetscFunctionBegin;
  /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
  PetscCall(DMGetCoordinateDM(dm, &coordDM));
  if (!feAux) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMClone(dm, &dmAux));
  PetscCall(DMSetCoordinateDM(dmAux, coordDM));
  for (f = 0; f < NfAux; ++f) PetscCall(DMSetField(dmAux, f, NULL, (PetscObject)feAux[f]));
  PetscCall(DMCreateDS(dmAux));
  PetscCall(SetupMaterial(dm, dmAux, user));
  PetscCall(DMDestroy(&dmAux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM        cdm = dm;
  PetscFE   fe, feAux;
  PetscBool simplex;
  PetscInt  dim;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  /* Create finite element */
  PetscCall(PetscFECreateDefault(comm, dim, dim, simplex, "disp_", PETSC_DEFAULT, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "displacement"));

  PetscCall(PetscFECreateDefault(comm, dim, 2, simplex, "elastMat_", PETSC_DEFAULT, &feAux));
  PetscCall(PetscObjectSetName((PetscObject)feAux, "elasticityMaterial"));
  PetscCall(PetscFECopyQuadrature(fe, feAux));

  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm, dim, user));
  while (cdm) {
    PetscCall(SetupAuxDM(cdm, 1, &feAux, user));
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFEDestroy(&feAux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeJacobian(SNES snes, Vec u, Mat A, Mat P, void *ctx)
{
  DM           dm;
  AppCtx      *user = (AppCtx *)ctx;
  MatNullSpace nullsp;

  PetscFunctionBegin;
  PetscCall((*user->inner_jacobian_fn)(snes, u, A, P, user->inner_jacobian_ctx));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMPlexCreateDisplacementRigidBody(dm, u, 0, &nullsp));
  PetscCall(MatSetNearNullSpace(A, nullsp));
  PetscCall(MatSetNearNullSpace(P, nullsp));
  PetscCall(MatNullSpaceDestroy(&nullsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES     snes; /* nonlinear solver */
  DM       dm;   /* problem definition */
  Vec      u, r; /* solution, residual vectors */
  Mat      A, J; /* Jacobian matrix */
  AppCtx   user; /* user-defined work context */
  PetscInt its;  /* iterations for convergence */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMSetApplicationContext(dm, &user));

  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));
  PetscCall(VecDuplicate(u, &r));

  PetscCall(DMSetMatType(dm, MATAIJ));
  PetscCall(DMCreateMatrix(dm, &J));
  A = J;

  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));
  PetscCall(DMSNESGetJacobian(dm, &user.inner_jacobian_fn, &user.inner_jacobian_ctx));
  PetscCall(SNESSetJacobian(snes, A, J, ComputeJacobian, &user));

  PetscCall(SNESSetFromOptions(snes));

  {
    PetscErrorCode (*initialGuess[1])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    initialGuess[0] = user.partial_bcs ? polar_transformation_offset_yz : spherical_transformation_offset;
    PetscCall(DMProjectFunction(dm, 0.0, initialGuess, NULL, INSERT_VALUES, u));
  }

  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT "\n", its));
  PetscCall(VecViewFromOptions(u, NULL, "-sol_vec_view"));

  if (A != J) PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: ctetgen !single !complex
    args: -dm_plex_dim 3 -disp_petscspace_degree 1 -elastMat_petscspace_degree 0 \
          -snes_monitor -snes_converged_reason -snes_rtol 1.e-12 -snes_ksp_ew -snes_ksp_ew_version 1 \
          -ksp_type minres -ksp_minres_qlp -ksp_converged_reason

    # observe that the full rigid body modes are not in the nullspace until convergence
    test:
      suffix: 0
      args: -dm_plex_box_faces 1,1,1 -pc_type none -ksp_view_eigenvalues_explicit

    test:
      suffix: 1
      args: -dm_plex_box_faces 4,4,4 -pc_type pbjacobi

    test:
      suffix: 2
      args: -partial_bcs -dm_plex_simplex 0 -dm_plex_box_faces 2,1,1 -pc_type none -ksp_view_eigenvalues_explicit

TEST*/
