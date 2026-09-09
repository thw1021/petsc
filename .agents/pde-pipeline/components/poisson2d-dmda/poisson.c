/*
  poisson.c  -- code-generation output for numerical-plan `poisson2d-plan`.

  Solves the 2-D Poisson problem   -Laplacian(u) = f   on the unit square
  with homogeneous Dirichlet boundary conditions, discretized with a standard
  second-order 5-point finite-difference stencil on a structured DMDA and
  solved with CG + GAMG (overridable from the command line).

  Verification is by the method of manufactured solutions:
      u_exact(x,y) = sin(pi x) sin(pi y)
      f(x,y)       = 2 pi^2 sin(pi x) sin(pi y)
  Boundary data g = u_exact = 0 on the boundary.

  Run a single grid:
      mpiexec -n 4 ./poisson -da_grid_x 65 -da_grid_y 65
  Run the whole convergence study in one shot (base grid refined `-mms_levels` times):
      mpiexec -n 4 ./poisson -mms_base 17 -mms_levels 4 -mms_csv convergence.csv

  NOTE: this program was generated for the pipeline dry-run and has NOT been
  compiled in this environment (no PETSc build was available). Build with the
  accompanying makefile once PETSC_DIR/PETSC_ARCH are set.
*/

#include <petscksp.h>
#include <petscdmda.h>
#include <math.h>

/* ------- manufactured solution, forcing, and boundary data ------- */
static inline PetscScalar uExact(PetscReal x, PetscReal y)
{
  return PetscSinReal(PETSC_PI * x) * PetscSinReal(PETSC_PI * y);
}
static inline PetscScalar forcing(PetscReal x, PetscReal y)
{
  return 2.0 * PETSC_PI * PETSC_PI * PetscSinReal(PETSC_PI * x) * PetscSinReal(PETSC_PI * y);
}
static inline PetscScalar gBC(PetscReal x, PetscReal y)
{
  return uExact(x, y); /* = 0 on the boundary of the unit square */
}

/* ------- RHS assembly (called by KSP via the DM) ------- */
static PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  DM            da;
  DMDALocalInfo info;
  PetscScalar **arr;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  const PetscReal hx = 1.0 / (info.mx - 1), hy = 1.0 / (info.my - 1);
  PetscCall(DMDAVecGetArray(da, b, &arr));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      const PetscReal x = i * hx, y = j * hy;
      if (i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1) {
        arr[j][i] = gBC(x, y); /* identity row: value is the Dirichlet datum */
      } else {
        PetscScalar val = forcing(x, y);
        /* lift known Dirichlet neighbours to the RHS (zero here, kept general) */
        if (i == 1)            val += gBC(0.0, y)  / (hx * hx);
        if (i == info.mx - 2)  val += gBC(1.0, y)  / (hx * hx);
        if (j == 1)            val += gBC(x, 0.0)  / (hy * hy);
        if (j == info.my - 2)  val += gBC(x, 1.0)  / (hy * hy);
        arr[j][i] = val;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, b, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- operator assembly (called by KSP via the DM) ------- */
static PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
  DM            da;
  DMDALocalInfo info;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  const PetscReal hx2 = 1.0 / ((info.mx - 1) * (info.mx - 1));
  const PetscReal hy2 = 1.0 / ((info.my - 1) * (info.my - 1));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      MatStencil row = {0};
      row.i = i; row.j = j;
      if (i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1) {
        PetscScalar v = 1.0; /* identity row keeps the operator SPD */
        PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, &v, INSERT_VALUES));
      } else {
        MatStencil  col[5];
        PetscScalar v[5];
        PetscInt    n = 0;
        col[n].i = i;     col[n].j = j;     v[n++] = 2.0 / hx2 + 2.0 / hy2;
        /* couple only to interior neighbours; boundary coupling is lifted to RHS */
        if (i - 1 > 0)          { col[n].i = i - 1; col[n].j = j;     v[n++] = -1.0 / hx2; }
        if (i + 1 < info.mx - 1) { col[n].i = i + 1; col[n].j = j;     v[n++] = -1.0 / hx2; }
        if (j - 1 > 0)          { col[n].i = i;     col[n].j = j - 1; v[n++] = -1.0 / hy2; }
        if (j + 1 < info.my - 1) { col[n].i = i;     col[n].j = j + 1; v[n++] = -1.0 / hy2; }
        PetscCall(MatSetValuesStencil(jac, 1, &row, n, col, v, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  if (J != jac) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- discrete error norms against the manufactured solution ------- */
static PetscErrorCode ComputeError(DM da, Vec u, PetscReal *l2, PetscReal *linf)
{
  DMDALocalInfo info;
  PetscScalar **arr;
  PetscReal     locSum = 0.0, locMax = 0.0;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da, &info));
  const PetscReal hx = 1.0 / (info.mx - 1), hy = 1.0 / (info.my - 1);
  PetscCall(DMDAVecGetArrayRead(da, u, &arr));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      const PetscReal x = i * hx, y = j * hy;
      const PetscReal e = PetscAbsScalar(arr[j][i] - uExact(x, y));
      locSum += e * e;
      locMax  = PetscMax(locMax, e);
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, u, &arr));
  MPI_Comm comm = PetscObjectComm((PetscObject)da);
  PetscReal gSum, gMax;
  PetscCallMPI(MPI_Allreduce(&locSum, &gSum, 1, MPIU_REAL, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&locMax, &gMax, 1, MPIU_REAL, MPI_MAX, comm));
  *l2   = PetscSqrtReal(hx * hy * gSum); /* discrete L2 norm */
  *linf = gMax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- solve one refinement level ------- */
static PetscErrorCode SolveLevel(PetscInt M, PetscReal *h, PetscReal *l2,
                                 PetscReal *linf, PetscInt *its, PetscInt *dof)
{
  DM  da;
  KSP ksp;
  Vec u;

  PetscFunctionBeginUser;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_STAR, M, M, PETSC_DECIDE, PETSC_DECIDE,
                         1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, NULL));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, NULL));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &u));
  PetscCall(KSPGetIterationNumber(ksp, its));

  PetscCall(ComputeError(da, u, l2, linf));
  *h   = 1.0 / (M - 1);
  *dof = (M - 2) * (M - 2); /* interior unknowns */

  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt  base = 17, levels = 1;
  char      csv[PETSC_MAX_PATH_LEN] = "";
  PetscBool haveCsv = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL,
    "2-D Poisson MMS verification on a structured DMDA.\n"));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mms_base", &base, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mms_levels", &levels, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mms_csv", csv, sizeof(csv), &haveCsv));

  PetscViewer vcsv = NULL;
  if (haveCsv) {
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, csv, &vcsv));
    PetscCall(PetscViewerASCIIPrintf(vcsv, "level,h,dof,L2,Linf,iterations\n"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "%-6s %-12s %-8s %-14s %-8s %-14s %-8s %-6s\n",
    "level", "h", "dof", "L2", "rateL2", "Linf", "rateLinf", "its"));

  PetscReal l2prev = 0.0, linfPrev = 0.0, hprev = 0.0;
  for (PetscInt l = 0; l < levels; l++) {
    const PetscInt M = (base - 1) * (1 << l) + 1;
    PetscReal h, l2, linf;
    PetscInt  its, dof;
    PetscCall(SolveLevel(M, &h, &l2, &linf, &its, &dof));

    PetscReal rL2 = 0.0, rLinf = 0.0;
    if (l > 0) {
      rL2   = PetscLogReal(l2prev / l2)     / PetscLogReal(hprev / h);
      rLinf = PetscLogReal(linfPrev / linf) / PetscLogReal(hprev / h);
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "%-6" PetscInt_FMT " %-12.3e %-8" PetscInt_FMT " %-14.6e %-8.3f %-14.6e %-8.3f %-6" PetscInt_FMT "\n",
      l, (double)h, dof, (double)l2, (double)rL2, (double)linf, (double)rLinf, its));
    if (haveCsv)
      PetscCall(PetscViewerASCIIPrintf(vcsv, "%" PetscInt_FMT ",%.6e,%" PetscInt_FMT ",%.6e,%.6e,%" PetscInt_FMT "\n",
                                       l, (double)h, dof, (double)l2, (double)linf, its));
    l2prev = l2; linfPrev = linf; hprev = h;
  }

  if (haveCsv) PetscCall(PetscViewerDestroy(&vcsv));
  PetscCall(PetscFinalize());
  return 0;
}
