static char help[] = "Tests mesh adaptation with DMPlex and pragmatic.\n";

#include <petsc/private/dmpleximpl.h>

#include <petscsnes.h>

typedef struct {
  PetscInt  Nr;         /* The number of refinement passes */
  PetscInt  metOpt;     /* Different choices of metric */
  PetscReal hmax, hmin; /* Max and min sizes prescribed by the metric */
  PetscBool doL2;       /* Test L2 projection */
} AppCtx;

/*
Classic hyperbolic sensor function for testing multi-scale anisotropic mesh adaptation:

  f:[-1, 1]x[-1, 1] \to R,
    f(x, y) = sin(50xy)/100 if |xy| > 2\pi/50 else sin(50xy)

(mapped to have domain [0,1] x [0,1] in this case).
*/
static PetscErrorCode sensor(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  const PetscReal xref = 2.*x[0] - 1.;
  const PetscReal yref = 2.*x[1] - 1.;
  const PetscReal xy   = xref*yref;

  PetscFunctionBeginUser;
  u[0] = PetscSinReal(50.*xy);
  if (PetscAbsReal(xy) > 2.*PETSC_PI/50.) u[0] *= 0.01;
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->Nr     = 1;
  options->metOpt = 1;
  options->hmin   = 0.05;
  options->hmax   = 0.5;
  options->doL2   = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-Nr", "Numberof refinement passes", "ex19.c", options->Nr, &options->Nr, NULL, 1);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-met", "Different choices of metric", "ex19.c", options->metOpt, &options->metOpt, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmax", "Max size prescribed by the metric", "ex19.c", options->hmax, &options->hmax, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-hmin", "Min size prescribed by the metric", "ex19.c", options->hmin, &options->hmin, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-do_L2", "Test L2 projection", "ex19.c", options->doL2, &options->doL2, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "DMinit");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-init_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMetricSensor(DM dm, AppCtx *user, Vec *metric)
{
  PetscSimplePointFunc funcs[1] = {sensor};
  DM             dmSensor, dmGrad, dmHess;
  PetscFE        fe;
  Vec            f, g, H;
  PetscBool      simplex;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);

  ierr = DMClone(dm, &dmSensor);CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, simplex, 1, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dmSensor, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dmSensor);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmSensor, &f);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmSensor, 0., funcs, NULL, INSERT_VALUES, f);CHKERRQ(ierr);
  ierr = VecViewFromOptions(f, NULL, "-sensor_view");CHKERRQ(ierr);

  // Recover the gradient of the sensor function
  ierr = DMClone(dm, &dmGrad);CHKERRQ(ierr);
  ierr = PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, simplex, 1, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dmGrad, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dmGrad);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmGrad, &g);CHKERRQ(ierr);
  ierr = DMPlexComputeGradientClementInterpolant(dmSensor, f, g);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecViewFromOptions(g, NULL, "-gradient_view");CHKERRQ(ierr);

  // Recover the Hessian of the sensor function
  ierr = DMClone(dm, &dmHess);CHKERRQ(ierr);
  ierr = DMPlexMetricCreate(dmHess, 0, &H);CHKERRQ(ierr);
  ierr = DMPlexComputeGradientClementInterpolant(dmGrad, g, H);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecViewFromOptions(H, NULL, "-hessian_view");CHKERRQ(ierr);

  // Obtain a metric by Lp normalization
  ierr = DMPlexMetricNormalize(dmHess, H, PETSC_TRUE, PETSC_TRUE, metric);CHKERRQ(ierr);
  ierr = VecDestroy(&H);CHKERRQ(ierr);
  ierr = DMDestroy(&dmHess);CHKERRQ(ierr);
  ierr = DMDestroy(&dmGrad);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSensor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMetric(DM dm, AppCtx *user, Vec *metric)
{
  PetscReal          lambda = 1/(user->hmax*user->hmax);
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  if (user->metOpt == 0) {
    /* Specify a uniform, isotropic metric */
    ierr = DMPlexMetricCreateUniform(dm, 0, lambda, metric);CHKERRQ(ierr);
  } else if (user->metOpt == 3) {
    ierr = ComputeMetricSensor(dm, user, metric);CHKERRQ(ierr);
  } else {
    DM                 cdm;
    Vec                coordinates;
    const PetscScalar *coords;
    PetscScalar       *met;
    PetscReal          h;
    PetscInt           vStart, vEnd, v;

    ierr = DMPlexMetricCreateUniform(dm, 0, lambda, metric);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecGetArray(*metric, &met);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscScalar       *vcoords;
      PetscScalar       *pmet;

      ierr = DMPlexPointLocalRead(cdm, v, coords, &vcoords);CHKERRQ(ierr);
      switch (user->metOpt) {
      case 1:
        h = user->hmax - (user->hmax-user->hmin)*PetscRealPart(vcoords[0]);
        break;
      case 2:
        h = user->hmax*PetscAbsReal(((PetscReal) 1.0)-PetscExpReal(-PetscAbsScalar(vcoords[0]-(PetscReal)0.5))) + user->hmin;
        break;
      default:
        SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "metOpt = 0, 1 or 2, cannot be %d", user->metOpt);
      }
      lambda = 1/(h*h);
      ierr = DMPlexPointLocalRef(dm, v, met, &pmet);CHKERRQ(ierr);
      pmet[0] = lambda;
    }
    ierr = VecRestoreArray(*metric, &met);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0] + x[1];
  return 0;
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode TestL2Projection(DM dm, DM dma, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  KSP              ksp;
  PetscDS          prob;
  PetscFE          fe;
  Mat              Interp, mass;
  Vec              u, ua, scaling, ones, massLumped, rhs, uproj;
  PetscReal        error;
  PetscInt         dim;
  MPI_Comm         comm;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, PETSC_TRUE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dma, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);

  funcs[0] = linear;
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &ua);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &ones);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &massLumped);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &rhs);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dma, &uproj);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Original");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-orig_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, funcs, NULL, u, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Original L2 Error: %g\n", (double) error);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dm, dma, &Interp, &scaling);CHKERRQ(ierr);
  ierr = MatInterpolate(Interp, u, ua);CHKERRQ(ierr);
  ierr = MatDestroy(&Interp);CHKERRQ(ierr);
  ierr = VecDestroy(&scaling);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ua, "Interpolation");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ua, NULL, "-interp_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dma, 0.0, funcs, NULL, ua, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Interpolated L2 Error: %g\n", (double) error);CHKERRQ(ierr);

  ierr = VecSet(ones, 1.0);CHKERRQ(ierr);
  ierr = DMSNESComputeJacobianAction(dma, ua, ones, massLumped, user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) massLumped, "Lumped mass");CHKERRQ(ierr);
  ierr = VecViewFromOptions(massLumped, NULL, "-mass_vec_view");CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(dm, dma, &mass);CHKERRQ(ierr);
  ierr = MatMult(mass, u, rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);
  ierr = VecViewFromOptions(rhs, NULL, "-lumped_rhs_view");CHKERRQ(ierr);
  ierr = VecPointwiseDivide(uproj, rhs, massLumped);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj, "Different Lumped Projection");CHKERRQ(ierr);
  ierr = VecViewFromOptions(uproj, NULL, "-lumped_rhs_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dma, 0.0, funcs, NULL, uproj, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Lumped (rhs) L2 Error: %g\n", (double) error);CHKERRQ(ierr);

  ierr = DMCreateMatrix(dma, &mass);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dma, ua, mass, mass, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mass, NULL, "-mass_mat_view");CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mass, mass);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj, "Full Projection");CHKERRQ(ierr);
  ierr = VecViewFromOptions(uproj, NULL, "-proj_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dma, 0.0, funcs, NULL, uproj, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Projected L2 Error: %g\n", (double) error);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &ua);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &ones);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &massLumped);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &rhs);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dma, &uproj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  MPI_Comm       comm;
  DM             dma, odm;
  Vec            metric;
  PetscInt       r;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm);CHKERRQ(ierr);

  odm  = dm;
  ierr = DMPlexDistributeOverlap(odm, 1, NULL, &dm);CHKERRQ(ierr);
  if (!dm) {dm = odm;}
  else     {ierr = DMDestroy(&odm);CHKERRQ(ierr);}

  for (r = 0; r < user.Nr; ++r) {
    DMLabel label;

    ierr = ComputeMetric(dm, &user, &metric);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
    ierr = DMAdaptMetric(dm, metric, label, &dma);CHKERRQ(ierr);
    ierr = VecDestroy(&metric);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dma, "DMadapt");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) dma, "adapt_");CHKERRQ(ierr);
    ierr = DMViewFromOptions(dma, NULL, "-dm_view");CHKERRQ(ierr);
    if (user.doL2) {ierr = TestL2Projection(dm, dma, &user);CHKERRQ(ierr);}
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dma;
  }
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dm, "final_");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: pragmatic

  testset:
    args: -dm_plex_box_faces 4,4,4 -met 2 -init_dm_view -adapt_dm_view

    test:
      suffix: 0
      args: -dm_plex_separate_marker 0
    test:
      suffix: 1
      args: -dm_plex_separate_marker 1
    test:
      suffix: 2
      args: -dm_plex_dim 3
    test:
      suffix: 3
      args: -dm_plex_dim 3

  # Pragmatic hangs for simple partitioner
  testset:
    requires: parmetis
    args: -dm_plex_box_faces 2,2 -dm_distribute -petscpartitioner_type parmetis -met 2 -init_dm_view -adapt_dm_view

    test:
      suffix: 4
      nsize: 2
    test:
      suffix: 5
      nsize: 4

  test:
    suffix: 6
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_box_faces 9,9,9 -dm_distribute -petscpartitioner_type parmetis \
          -met 0 -hmin 0.01 -hmax 0.03 -init_dm_view -adapt_dm_view
  test:
    suffix: 7
    nsize: 2
    args: -dm_plex_box_faces 19,19 -dm_distribute -petscpartitioner_type parmetis \
          -met 2 -hmax 0.5 -hmin 0.001 -init_dm_view -adapt_dm_view
  test:
    suffix: proj_0
    TODO: broken
    args: -dm_plex_box_faces 2,2 -dm_plex_separate_marker 0 -init_dm_view -adapt_dm_view -do_L2 -petscspace_degree 1 -petscfe_default_quadrature_order 1 -dm_plex_hash_location -pc_type lu
  test:
    suffix: proj_1
    TODO: broken
    args: -dm_plex_box_faces 4,4 -dm_plex_separate_marker 0 -init_dm_view -adapt_dm_view -do_L2 -petscspace_degree 2 -petscfe_default_quadrature_order 4 -dm_plex_hash_location -pc_type lu

  test:
    suffix: sensor
    args: -dm_plex_box_faces 9,9 -met 3 -init_dm_view -adapt_dm_view \
          -dm_plex_metric_h_min 1.e-10 -dm_plex_metric_h_max 1.0e-01 -dm_plex_metric_a_max 1.0e+05 -dm_plex_metric_p 1.0 \
            -dm_plex_metric_target_complexity 10000.0

TEST*/
