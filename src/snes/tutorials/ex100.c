static char help[] = "Nonlinear elimination of one field via SNESMULTIBLOCK NPC.\n\
We solve a two-field coupled nonlinear PDE on [0,1]^2:\n\
  -Delta u + u*p = f_u\n\
  -Delta p + p^3 + u = f_p\n\
The outer Newton handles u; the NPC (SNESMULTIBLOCK) pre-eliminates p.\n\
Manufactured solution: u = sin(pi*x)*sin(pi*y), p = sin(2*pi*x)*sin(2*pi*y).\n\n";

/*
  Nonlinear elimination of pressure field p via SNESMULTIBLOCK as a nonlinear preconditioner (NPC):
  - Outer Newton (SNESNEWTONLS) solves the u-equation using a Schur-complement fieldsplit.
  - NPC block 0 (field 0 = u): SNESNONE -- identity, u is left for the outer Newton.
  - NPC block 1 (field 1 = p): SNESNEWTONLS -- tightly solves -Delta p + p^3 + u = f_p given u.
  After each NPC call, p satisfies its equation to tight tolerance, so the outer Newton's
  p-residual is near zero and the Schur complement is effectively in the u-space alone.

  Illustrative options for the nonlinear elimination test:
    -snes_type newtonls -npc_snes_type multiblock
      -npc_snes_multiblock_0_fields 0 -npc_snes_multiblock_1_fields 1
      -npc_multiblock_0_snes_type none
      -npc_multiblock_1_snes_type newtonls
    -ksp_type preonly -pc_type fieldsplit -pc_fieldsplit_type schur
      -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full
      -fieldsplit_u_ksp_type preonly -fieldsplit_u_pc_type lu
      -fieldsplit_p_ksp_type preonly -fieldsplit_p_pc_type lu

  F_p is nonlinear in p, so -Delta p + p^3 = f_p - u is a well-posed sub-problem for p
  given u, making p a suitable field for nonlinear elimination.
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

/* Exact solutions for MMS verification */
static PetscErrorCode u_exact_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(PETSC_PI * x[0]) * PetscSinReal(PETSC_PI * x[1]);
  return PETSC_SUCCESS;
}

static PetscErrorCode p_exact_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *p, void *ctx)
{
  p[0] = PetscSinReal(2.0 * PETSC_PI * x[0]) * PetscSinReal(2.0 * PETSC_PI * x[1]);
  return PETSC_SUCCESS;
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  for (PetscInt c = 0; c < Nc; ++c) u[c] = 0.0;
  return PETSC_SUCCESS;
}

/*
  Residual for F_u = -Delta u + u*p - f_u where
  f_u = 2*pi^2*sin(pi*x)*sin(pi*y) + sin(pi*x)*sin(pi*y)*sin(2*pi*x)*sin(2*pi*y)
*/
static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal sxy  = PetscSinReal(PETSC_PI * x[0]) * PetscSinReal(PETSC_PI * x[1]);
  const PetscReal s2xy = PetscSinReal(2.0 * PETSC_PI * x[0]) * PetscSinReal(2.0 * PETSC_PI * x[1]);

  f0[0] = u[uOff[0]] * u[uOff[1]] - (2.0 * PETSC_PI * PETSC_PI * sxy + sxy * s2xy);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[uOff_x[0] + d];
}

/*
  Residual for F_p = -Delta p + p^3 + u - f_p where
  f_p = 8*pi^2*sin(2*pi*x)*sin(2*pi*y) + sin^3(2*pi*x)*sin^3(2*pi*y) + sin(pi*x)*sin(pi*y)

  Note: uOff[0] = u-field offset, uOff[1] = p-field offset.  When called from the p sub-SNES
  (dsIn mechanism), the same indices are valid because dsIn uses the parent DS field layout.
*/
static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal   sxy  = PetscSinReal(PETSC_PI * x[0]) * PetscSinReal(PETSC_PI * x[1]);
  const PetscReal   s2xy = PetscSinReal(2.0 * PETSC_PI * x[0]) * PetscSinReal(2.0 * PETSC_PI * x[1]);
  const PetscScalar p    = u[uOff[1]];

  f0[0] = p * p * p + u[uOff[0]] - (8.0 * PETSC_PI * PETSC_PI * s2xy + s2xy * s2xy * s2xy + sxy);
}

static void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d] = u_x[uOff_x[1] + d];
}

/* Jacobians: dF_u/du = p (mass-like), dF_u/d(grad u) = I */
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u[uOff[1]]; /* p */
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

/* dF_u/dp = u (coupling block) */
static void g0_up(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = u[uOff[0]]; /* u */
}

/* dF_p/du = 1 (coupling block) */
static void g0_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

/* dF_p/dp = 3*p^2, dF_p/d(grad p) = I */
static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  const PetscScalar p = u[uOff[1]];
  g0[0] = 3.0 * p * p;
}

static void g3_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
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
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  /* Residuals */
  PetscCall(PetscDSSetResidual(ds, 0, f0_u, f1_u));
  PetscCall(PetscDSSetResidual(ds, 1, f0_p, f1_p));
  /* Jacobians: all four blocks to form the coupled MATNEST */
  PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetJacobian(ds, 0, 1, g0_up, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 0, g0_pu, NULL, NULL, NULL));
  PetscCall(PetscDSSetJacobian(ds, 1, 1, g0_pp, NULL, NULL, g3_pp));
  /* Exact solutions for dmsnes_check */
  PetscCall(PetscDSSetExactSolution(ds, 0, u_exact_2d, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 1, p_exact_2d, NULL));
  /* Homogeneous Dirichlet on both fields (exact solutions vanish on boundary) */
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall_u", label, 1, &id, 0, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall_p", label, 1, &id, 1, 0, NULL, (void (*)(void))zero, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm)
{
  DM        cdm = dm;
  PetscFE   fe[2];
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  /* Two scalar P_k fields; degree controlled via -u_petscspace_degree / -p_petscspace_degree */
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "u_", PETSC_DEFAULT, &fe[0]));
  PetscCall(PetscObjectSetName((PetscObject)fe[0], "u"));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "p_", PETSC_DEFAULT, &fe[1]));
  PetscCall(PetscObjectSetName((PetscObject)fe[1], "p"));
  PetscCall(PetscFECopyQuadrature(fe[0], fe[1]));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe[0]));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe[1]));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupProblem(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe[0]));
  PetscCall(PetscFEDestroy(&fe[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM   dm;
  SNES snes;
  Vec  u;
  Mat  J;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution"));
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(DMPlexSetSNESLocalFEM(dm, NULL, NULL, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(VecSet(u, 0.0));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&J));
  PetscCall(DMDestroy(&dm));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p1_check
    requires: !complex double triangle
    args: -dm_plex_box_faces 4,4 -u_petscspace_degree 1 -p_petscspace_degree 1 \
          -dmsnes_check 0.0001

  test:
    suffix: 2d_p1_elimination
    requires: !complex double triangle
    args: -dm_plex_box_faces 4,4 -u_petscspace_degree 1 -p_petscspace_degree 1 \
          -snes_type newtonls -snes_converged_reason \
          -npc_snes_type multiblock \
            -npc_snes_multiblock_0_fields 0 -npc_snes_multiblock_1_fields 1 \
            -npc_multiblock_0_snes_type none \
            -npc_multiblock_1_snes_type newtonls \
          -ksp_type preonly \
          -pc_type fieldsplit -pc_fieldsplit_type schur \
            -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full \
            -fieldsplit_u_ksp_type preonly -fieldsplit_u_pc_type lu \
            -fieldsplit_p_ksp_type preonly -fieldsplit_p_pc_type lu

TEST*/
