static const char help[] = "\
Minimum surface area problem in 2D using DMPLEX.\n\
It solves the unconstrained minimization problem \n\
\n\
argmin \\int_\\Omega (1 + ||u||^2), u = g on \\partial\\Omega.\n\
\n\
This example shows how to setup an optimization problem using DMPLEX FEM routines.\n\
It supports nonlinear domain decomposition and multilevel solvers.\n\
\n\n";

/*
A visualization of the adaptation can be accomplished using:

  -dm_adapt_view hdf5:$PWD/adapt.h5 -sol_adapt_view hdf5:$PWD/adapt.h5::append -dm_adapt_pre_view hdf5:$PWD/orig.h5 -sol_adapt_pre_view hdf5:$PWD/orig.h5::append

Information on refinement:

   -info :~sys,vec,is,mat,ksp,snes,ts
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

/* The pointwise evaluation routine for the objective function */
static void objective_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  const PetscReal ux2 = PetscSqr(u_x[0]);
  const PetscReal uy2 = PetscSqr(u_x[1]);

  obj[0] = PetscSqrtReal(1.0 + ux2 + uy2);
}

/* The pointwise evaluation routine for the gradient wrt the gradients of the FEM basis */
static void gradient_1_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  const PetscReal ux2 = PetscSqr(u_x[0]);
  const PetscReal uy2 = PetscSqr(u_x[1]);
  const PetscReal v   = 1. / PetscSqrtReal(1.0 + ux2 + uy2);

  f[0] = v * u_x[0];
  f[1] = v * u_x[1];
}

/* The pointwise evaluation routine for the hessian wrt the gradients of the FEM basis */
static void hessian_11_2d(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar jac[])
{
  const PetscReal ux2 = PetscSqr(u_x[0]);
  const PetscReal uy2 = PetscSqr(u_x[1]);
  const PetscReal uxy = u_x[0] * u_x[1];
  const PetscReal v1  = 1. / PetscSqrtReal(1.0 + ux2 + uy2);
  const PetscReal v2  = v1 / (1.0 + ux2 + uy2);

  jac[0] = v1 - v2 * ux2;
  jac[1] = -v2 * uxy;
  jac[2] = -v2 * uxy;
  jac[3] = v1 - v2 * uy2;
}

/* The boundary conditions we impose */
static PetscErrorCode sins_2d(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscFunctionBeginUser;
  const PetscReal pi2 = PETSC_PI * 2;
  const PetscReal x   = xx[0];
  const PetscReal y   = xx[1];

  *u = (y - 0.5) * PetscSinReal(pi2 * x) + (x - 0.5) * PetscSinReal(pi2 * y);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DM      plex;
  DMLabel label;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, name));
  PetscCall(DMGetLabel(dm, name, &label));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexMarkBoundaryFaces(plex, 1, label));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupProblem(DM dm)
{
  PetscDS        ds;
  DMLabel        label;
  PetscInt       dim;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim == 2, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only for 2-D\n");
  PetscCall(PetscDSSetObjective(ds, 0, objective_2d));
  PetscCall(PetscDSSetResidual(ds, 0, NULL, gradient_1_2d));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, hessian_11_2d));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "data", label, 1, &id, 0, 0, NULL, (void (*)(void))sins_2d, NULL, NULL, NULL));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_TRUE, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm)
{
  DM        plex, cdm = dm;
  PetscFE   fe;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "potential"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm));
  while (cdm) {
    PetscBool hasLabel;

    PetscCall(DMHasLabel(cdm, "marker", &hasLabel));
    if (!hasLabel) PetscCall(CreateBCLabel(cdm, "marker"));
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM   dm;   /* Problem specification */
  SNES snes; /* nonlinear solver */
  Vec  u;    /* solution vector */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SNESSetDM(snes, dm));

  PetscCall(SetupDiscretization(dm));

  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(SNESSolve(snes, NULL, u));

  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    requires: !single
    suffix: qn_nasm
    nsize: 4
    args: -petscspace_degree 1 -dm_refine 2 -snes_type qn -snes_npc_side {{left right}separate output} -npc_snes_type nasm -npc_snes_nasm_type restrict -npc_sub_snes_linesearch_order 1 -npc_sub_snes_linesearch_type bt -dm_plex_dd_overlap 1 -snes_linesearch_type bt -snes_linesearch_order 1 -npc_sub_pc_type lu -npc_sub_ksp_type preonly -snes_converged_reason -snes_monitor_short -petscpartitioner_type simple -npc_sub_snes_max_it 1 -dm_plex_simplex 0

TEST*/
