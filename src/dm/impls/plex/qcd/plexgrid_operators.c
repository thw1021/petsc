/*
 * Contains well known matrix free operators for lattice gauge theories. This
 * includes the Wilson operator, domain wall fermions, and Pauli-Villar "preconditioned"
 * domain wall fermions
 */
#include <petscconf.h>
#include <petscgrid.h>
#include <petsc/private/dmpleximpl.h>

// Apply \gamma_\mu
static PetscErrorCode ComputeGamma(PetscInt d, PetscInt ldx, PetscScalar f[])
{
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (d) {
  case 0:
    f[0 * ldx] = PETSC_i * fin[3];
    f[1 * ldx] = PETSC_i * fin[2];
    f[2 * ldx] = -PETSC_i * fin[1];
    f[3 * ldx] = -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] = -fin[3];
    f[1 * ldx] = fin[2];
    f[2 * ldx] = fin[1];
    f[3 * ldx] = -fin[0];
    break;
  case 2:
    f[0 * ldx] = PETSC_i * fin[2];
    f[1 * ldx] = -PETSC_i * fin[3];
    f[2 * ldx] = -PETSC_i * fin[0];
    f[3 * ldx] = PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] = fin[2];
    f[1 * ldx] = fin[3];
    f[2 * ldx] = fin[0];
    f[3 * ldx] = fin[1];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 4)", d);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Apply (1 \pm \gamma_\mu)/2
static inline PetscErrorCode ComputeGammaFactor(PetscInt d, PetscBool forward, PetscInt ldx, PetscScalar f[])
{
  const PetscReal   sign   = forward ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (d) {
  case 0:
    f[0 * ldx] += sign * PETSC_i * fin[3];
    f[1 * ldx] += sign * PETSC_i * fin[2];
    f[2 * ldx] += sign * -PETSC_i * fin[1];
    f[3 * ldx] += sign * -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] += -sign * fin[3];
    f[1 * ldx] += sign * fin[2];
    f[2 * ldx] += sign * fin[1];
    f[3 * ldx] += -sign * fin[0];
    break;
  case 2:
    f[0 * ldx] += sign * PETSC_i * fin[2];
    f[1 * ldx] += sign * -PETSC_i * fin[3];
    f[2 * ldx] += sign * -PETSC_i * fin[0];
    f[3 * ldx] += sign * PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] += sign * fin[2];
    f[1 * ldx] += sign * fin[3];
    f[2 * ldx] += sign * fin[0];
    f[3 * ldx] += sign * fin[1];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 4)", d);
  }
  f[0 * ldx] *= 0.5;
  f[1 * ldx] *= 0.5;
  f[2 * ldx] *= 0.5;
  f[3 * ldx] *= 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void TwoSpinProject(PetscInt mu, PetscBool minus, PetscInt ldx, const PetscScalar f[], PetscScalar o[])
{
  const PetscReal   sign   = (minus) ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  //  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    o[0 * ldx] = fin[0] + sign * PETSC_i * fin[3];
    o[1 * ldx] = fin[1] + sign * PETSC_i * fin[2];
    break;
  case 1:
    o[0 * ldx] = fin[0] - sign * fin[3];
    o[1 * ldx] = fin[1] + sign * fin[2];
    break;
  case 2:
    o[0 * ldx] = fin[0] + sign * PETSC_i * fin[2];
    o[1 * ldx] = fin[1] - sign * PETSC_i * fin[3];
    break;
  case 3:
    o[0 * ldx] = fin[0] + sign * fin[2];
    o[1 * ldx] = fin[1] + sign * fin[3];
    break;
  case 4:
    if (sign == 1) {
      o[0 * ldx] = fin[0];
      o[1 * ldx] = fin[1];
    } else {
      o[0 * ldx] = fin[2];
      o[1 * ldx] = fin[3];
    }
    break;
  default:
    //    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
    break;
  }
  o[0 * ldx] *= 0.5;
  o[1 * ldx] *= 0.5;
  //  PetscFunctionReturn(0);
}

// Apply (1 \pm \gamma_\mu)/2
static inline void TwoSpinAccumulate(PetscInt mu, PetscBool minus, PetscInt ldx, const PetscScalar f[], PetscScalar p[])
{
  const PetscReal   sign   = (minus) ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  switch (mu) {
  case 0:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] += sign * PETSC_i * fin[1];
    p[3 * ldx] += sign * PETSC_i * fin[0];
    break;
  case 1:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] -= sign * fin[1];
    p[3 * ldx] += sign * fin[0];
    break;
  case 2:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] += sign * PETSC_i * fin[0];
    p[3 * ldx] -= sign * PETSC_i * fin[1];
    break;
  case 3:
    p[0 * ldx] -= fin[0];
    p[1 * ldx] -= fin[1];
    p[2 * ldx] -= sign * fin[0];
    p[3 * ldx] -= sign * fin[1];
    break;
  case 4:
    if (sign == 1) {
      p[0 * ldx] += 2 * fin[0]; // Not sure of this 2x
      p[1 * ldx] += 2 * fin[1];
    } else {
      p[2 * ldx] += 2 * fin[0];
      p[3 * ldx] += 2 * fin[1];
    }
    break;
  default:
    //SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
    break;
  }
}

// ComputeAction() sums the action of 1/2 (1 \pm \gamma_\mu) U \psi into f
static PetscErrorCode ComputeAction(PetscInt d, PetscBool forward, PetscBool dag, const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12], utmp[12];
  PetscBool   gforward;

  PetscFunctionBeginHot;
  if (dag) {
    if (forward) gforward = PETSC_FALSE;
    else gforward = PETSC_TRUE;
  } else {
    gforward = forward;
  }

  for (PetscInt c = 0; c < 3; ++c) TwoSpinProject(d, gforward, 3, &psi[c], &tmp[c]);
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &tmp[beta * 3], &utmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &tmp[beta * 3], &utmp[beta * 3]);
  }
  for (PetscInt c = 0; c < 3; ++c) TwoSpinAccumulate(d, gforward, 3, &utmp[c], &f[c]);
  PetscFunctionReturn(0);
}

/*
-- From Peter Boyle at https://github.com/paboyle/PETSc-Grid/blob/main/petsc_fermion.h --
*/

// Apply (1 \pm \gamma_\mu)/2
static PetscErrorCode SpinProject(PetscInt mu, PetscBool minus, PetscInt ldx, PetscScalar f[])
{
  const PetscReal   sign   = (minus) ? -1. : 1.;
  const PetscScalar fin[4] = {f[0 * ldx], f[1 * ldx], f[2 * ldx], f[3 * ldx]};

  PetscFunctionBeginHot;
  switch (mu) {
  case 0:
    f[0 * ldx] += sign * PETSC_i * fin[3];
    f[1 * ldx] += sign * PETSC_i * fin[2];
    f[2 * ldx] += sign * -PETSC_i * fin[1];
    f[3 * ldx] += sign * -PETSC_i * fin[0];
    break;
  case 1:
    f[0 * ldx] += -sign * fin[3];
    f[1 * ldx] += sign * fin[2];
    f[2 * ldx] += sign * fin[1];
    f[3 * ldx] += -sign * fin[0];
    break;
  case 2:
    f[0 * ldx] += sign * PETSC_i * fin[2];
    f[1 * ldx] += sign * -PETSC_i * fin[3];
    f[2 * ldx] += sign * -PETSC_i * fin[0];
    f[3 * ldx] += sign * PETSC_i * fin[1];
    break;
  case 3:
    f[0 * ldx] += sign * fin[2];
    f[1 * ldx] += sign * fin[3];
    f[2 * ldx] += sign * fin[0];
    f[3 * ldx] += sign * fin[1];
    break;
  case 4:
    f[0 * ldx] += sign * fin[0];
    f[1 * ldx] += sign * fin[1];
    f[2 * ldx] -= sign * fin[2];
    f[3 * ldx] -= sign * fin[3];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Direction for gamma %" PetscInt_FMT " not in [0, 5)", mu);
  }
  f[0 * ldx] *= 0.5;
  f[1 * ldx] *= 0.5;
  f[2 * ldx] *= 0.5;
  f[3 * ldx] *= 0.5;
  PetscFunctionReturn(0);
}

// Wilson quark mass
const PetscReal Wmass = -0.92;//TODO: Put in user accessible struct

/*
  Compute the Wilson operator
*/
static PetscErrorCode ComputeWilson(Mat F, Vec u, Vec f, PetscBool dag)
{
  DM                 dm, dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;

  PetscFunctionBegin;
  PetscCall(MatGetDM(F, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));

  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecViewFromOptions(gauge, NULL, "-residual_gauge_view"));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));
  // Loop over y
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));
    // Diagonal
    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (Wmass + 4) * ua[xoff + i];
    // Loop over mu
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;

      // Left action -(1 + \gamma_\mu)/2 \otimes U^\dagger_\mu(y) \delta_{x - \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(ComputeAction(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));
      // Right edge -(1 - \gamma_\mu)/2 \otimes U_\mu(x) \delta_{x + \mu,y} \psi(y)
      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(ComputeAction(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Configure a MatShell to represent the matrix free Wilson operator using a Gauge field contained
 * in a DMPlex.
 */
static PetscErrorCode ComputeDWilson_Forward(Mat F, Vec u, Vec f)
{
  PetscFunctionBeginHot;
  PetscCall(ComputeWilson(F, u, f, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeDWilson_Dagger(Mat F, Vec u, Vec f)
{
  PetscFunctionBeginHot;
  PetscCall(ComputeWilson(F, u, f, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscGridSetUpWilson(DM dm, PetscBool setGauge, Mat *WilsonOperator)
{
  Vec      u;
  PetscInt locSize;
  PetscBool isPV = PETSC_FALSE;
  PetscInt Ls;
  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, NULL, WilsonOperator));
  PetscCall(MatShellSetOperation(*WilsonOperator, MATOP_MULT, (void (*)(void))ComputeDWilson_Forward));
  PetscCall(MatShellSetOperation(*WilsonOperator, MATOP_MULT_TRANSPOSE, (void (*)(void))ComputeDWilson_Dagger));
  PetscCall(PetscObjectSetName((PetscObject)*WilsonOperator, "Wilson Operator"));
  PetscCall(MatSetDM(*WilsonOperator, dm));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
  Configure a MatShell to represent the matrix free Domain Wall Fermion operator
 */
static PetscErrorCode DdwfDhop(PetscInt d, PetscBool forward, PetscBool dag, const PetscScalar U[], const PetscScalar psi[], PetscScalar f[])
{
  PetscScalar tmp[12];
  int         gamma[] = {4, 0, 1, 2, 3};
  PetscBool   gforward;

  PetscFunctionBeginHot;
  for (PetscInt beta = 0; beta < 4; ++beta) {
    if (forward) DMPlex_Mult3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
    else DMPlex_MultTranspose3D_Internal(U, 1, &psi[beta * 3], &tmp[beta * 3]);
  }
  if (dag) gforward = forward ? PETSC_FALSE : PETSC_TRUE;
  else     gforward = forward;
  for (PetscInt c = 0; c < 3; ++c) PetscCall(SpinProject(gamma[d], gforward, 3, &tmp[c]));
  for (PetscInt i = 0; i < 12; ++i) f[i] -= tmp[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Domain-wall fermion application over a 5D Hypercubic DMPlex. */
static PetscErrorCode Ddwf(Mat M, Vec u, Vec f, PetscBool dag)
{
  DM                 dm, dmAux;
  Vec                gauge;
  PetscSection       s, sGauge;
  const PetscScalar *ua;
  PetscScalar       *fa, *link;
  PetscInt           dim, vStart, vEnd;
  PetscScalar        M5;

  PetscFunctionBeginHot;
  PetscCall(MatGetDM(M, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(VecGetArrayRead(u, &ua));
  PetscCall(VecGetArray(f, &fa));
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &gauge));
  PetscCall(VecGetDM(gauge, &dmAux));
  PetscCall(DMGetLocalSection(dmAux, &sGauge));
  PetscCall(VecGetArray(gauge, &link));

  M5 = 1.8;//TODO: Put in user accessible struct
  for (PetscInt v = vStart; v < vEnd; ++v) {
    const PetscInt *supp;
    PetscInt        xdof, xoff;

    PetscCall(DMPlexGetSupport(dm, v, &supp));
    PetscCall(PetscSectionGetDof(s, v, &xdof));
    PetscCall(PetscSectionGetOffset(s, v, &xoff));

    for (PetscInt i = 0; i < xdof; ++i) fa[xoff + i] = (5.0 - M5) * ua[xoff + i];
    for (PetscInt d = 0; d < dim; ++d) {
      const PetscInt *cone;
      PetscInt        yoff, goff;

      PetscCall(DMPlexGetCone(dm, supp[2 * d + 0], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[0], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 0], &goff));
      PetscCall(DdwfDhop(d, PETSC_FALSE, dag, &link[goff], &ua[yoff], &fa[xoff]));

      PetscCall(DMPlexGetCone(dm, supp[2 * d + 1], &cone));
      PetscCall(PetscSectionGetOffset(s, cone[1], &yoff));
      PetscCall(PetscSectionGetOffset(sGauge, supp[2 * d + 1], &goff));
      PetscCall(DdwfDhop(d, PETSC_TRUE, dag, &link[goff], &ua[yoff], &fa[xoff]));
    }
  }
  PetscCall(VecRestoreArray(f, &fa));
  PetscCall(VecRestoreArray(gauge, &link));
  PetscCall(VecRestoreArrayRead(u, &ua));
  PetscCall(VecViewFromOptions(f, NULL, "-residual_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Ddwf_Forward(Mat M, Vec u, Vec f)
{
  PetscFunctionBeginHot;
  PetscCall(Ddwf(M, u, f, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Ddwf_Dag(Mat M, Vec u, Vec f)
{
  PetscFunctionBeginHot;
  PetscCall(Ddwf(M, u, f, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscGridSetUpDdwf(DM dm, PetscBool setGauge, Mat *DdwfOperator)
{
  Vec      u;
  PetscInt locSize;

  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, NULL, DdwfOperator));
  PetscCall(MatSetBlockSize(*DdwfOperator, 12));
  PetscCall(MatShellSetOperation(*DdwfOperator, MATOP_MULT,           (void (*)(void)) Ddwf_Forward));
  PetscCall(MatShellSetOperation(*DdwfOperator, MATOP_MULT_TRANSPOSE, (void (*)(void)) Ddwf_Dag));
  PetscCall(PetscObjectSetName((PetscObject) *DdwfOperator, "Ddwf Operator"));
  PetscCall(MatSetDM(*DdwfOperator, dm));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO
#if 0
static PetscErrorCode SetUpPV_5DPlex(DM dm, Mat *DWOperator, PetscBool isPV, PetscInt Ls, PetscBool setGauge, AppCtx *user)
{
  Vec      u;
  PetscInt locSize;

  PetscFunctionBegin;
  if (setGauge) PetscCall(PetscSetGauge_Grid5D(dm, GRID_LATTICE_FILE, isPV, Ls, user->fixGauge, user->transform, user->gridFile));
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, user, DWOperator));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT, (void (*)(void))PrecOp));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT_TRANSPOSE, (void (*)(void))PrecOpDag));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetUpSqrDW(DM dm, Mat *DWOperator, PetscInt Ls, AppCtx *user)
{
  Vec      u;
  PetscInt locSize;

  PetscFunctionBegin;
  PetscCall(PetscSetGauge_Grid5D(dm, GRID_LATTICE_FILE, PETSC_FALSE, Ls, user->fixGauge, user->transform, user->gridFile));
  PetscCall(DMCreateLocalVector(dm, &u));
  PetscCall(VecGetLocalSize(u, &locSize));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, locSize, locSize, PETSC_DECIDE, PETSC_DECIDE, user, DWOperator));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT, (void (*)(void))DdwfDagDdwf));
  PetscCall(MatShellSetOperation(*DWOperator, MATOP_MULT_TRANSPOSE, (void (*)(void))DdwfDagDdwf));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
