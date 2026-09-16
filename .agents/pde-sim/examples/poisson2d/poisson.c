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
        if (i == 1) val += gBC(0.0, y) / (hx * hx);
        if (i == info.mx - 2) val += gBC(1.0, y) / (hx * hx);
        if (j == 1) val += gBC(x, 0.0) / (hy * hy);
        if (j == info.my - 2) val += gBC(x, 1.0) / (hy * hy);
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
      row.i          = i;
      row.j          = j;
      if (i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1) {
        PetscScalar v = 1.0; /* identity row keeps the operator SPD */
        PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, &v, INSERT_VALUES));
      } else {
        MatStencil  col[5];
        PetscScalar v[5];
        PetscInt    n = 0;
        col[n].i      = i;
        col[n].j      = j;
        v[n++]        = 2.0 / hx2 + 2.0 / hy2;
        /* couple only to interior neighbours; boundary coupling is lifted to RHS */
        if (i - 1 > 0) {
          col[n].i = i - 1;
          col[n].j = j;
          v[n++]   = -1.0 / hx2;
        }
        if (i + 1 < info.mx - 1) {
          col[n].i = i + 1;
          col[n].j = j;
          v[n++]   = -1.0 / hx2;
        }
        if (j - 1 > 0) {
          col[n].i = i;
          col[n].j = j - 1;
          v[n++]   = -1.0 / hy2;
        }
        if (j + 1 < info.my - 1) {
          col[n].i = i;
          col[n].j = j + 1;
          v[n++]   = -1.0 / hy2;
        }
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

static PetscErrorCode ComputeError(DM da, Vec u, PetscReal *l2, PetscReal *linf);

/* ------- in-situ NaN/Inf detection on a solution field -------
   Cheap, global: a NaN anywhere makes the 2-norm NaN; an Inf makes it Inf. */
static PetscErrorCode CheckNanInf(Vec v, const char *name)
{
  PetscReal nrm;
  MPI_Comm  comm = PetscObjectComm((PetscObject)v);

  PetscFunctionBeginUser;
  PetscCall(VecNorm(v, NORM_2, &nrm));
  PetscCheck(!PetscIsInfOrNanReal(nrm), comm, PETSC_ERR_FP, "NaN/Inf detected in field '%s'", name);
  PetscCall(PetscPrintf(comm, "[nan/inf check] field '%s' OK (||.||_2 = %.6e)\n", name, (double)nrm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- fill the exact-solution and pointwise-error fields ------- */
static PetscErrorCode FillExactAndError(DM da, Vec u, Vec exact, Vec err)
{
  DMDALocalInfo info;
  PetscScalar **ua, **ea, **ra;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da, &info));
  const PetscReal hx = 1.0 / (info.mx - 1), hy = 1.0 / (info.my - 1);
  PetscCall(DMDAVecGetArrayRead(da, u, &ua));
  PetscCall(DMDAVecGetArray(da, exact, &ea));
  PetscCall(DMDAVecGetArray(da, err, &ra));
  for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
      const PetscReal   x = i * hx, y = j * hy;
      const PetscScalar ue = uExact(x, y);
      ea[j][i]             = ue;
      ra[j][i]             = PetscAbsScalar(ua[j][i] - ue); /* pointwise |u_h - u_exact| */
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, u, &ua));
  PetscCall(DMDAVecRestoreArray(da, exact, &ea));
  PetscCall(DMDAVecRestoreArray(da, err, &ra));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- gather a DMDA global vector to natural (row-major) order on
   rank 0 and dump it as a headerless raw little-endian double array,
   which the XDMF wrapper references (this build has no HDF5). ------- */
static PetscErrorCode WriteFieldBinary(DM da, Vec g, const char *fname)
{
  Vec         natural, seq;
  VecScatter  scat;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(DMDACreateNaturalVector(da, &natural));
  PetscCall(DMDAGlobalToNaturalBegin(da, g, INSERT_VALUES, natural));
  PetscCall(DMDAGlobalToNaturalEnd(da, g, INSERT_VALUES, natural));
  PetscCall(VecScatterCreateToZero(natural, &scat, &seq));
  PetscCall(VecScatterBegin(scat, natural, seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, natural, seq, INSERT_VALUES, SCATTER_FORWARD));

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)da), &rank));
  if (rank == 0) {
    const PetscScalar *a;
    PetscInt           n;
    FILE              *fp;
    PetscCall(VecGetSize(seq, &n));
    PetscCall(VecGetArrayRead(seq, &a));
    fp = fopen(fname, "wb");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s for writing", fname);
    for (PetscInt k = 0; k < n; k++) {
      double d = (double)PetscRealPart(a[k]);
      size_t w = fwrite(&d, sizeof(double), 1, fp);
      PetscCheck(w == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Short write to %s", fname);
    }
    PetscCheck(fclose(fp) == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Error closing %s", fname);
    PetscCall(VecRestoreArrayRead(seq, &a));
  }
  PetscCall(VecScatterDestroy(&scat));
  PetscCall(VecDestroy(&seq));
  PetscCall(VecDestroy(&natural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- write an XDMF wrapper for a uniform 2-D grid over the raw
   binary field dumps so ParaView can open the solution. ------- */
static PetscErrorCode WriteXDMF(MPI_Comm comm, const char *xdmf, PetscInt mx, PetscInt my, PetscReal hx, PetscReal hy, const char *ubin, const char *exbin, const char *errbin)
{
  FILE *fp;

  PetscFunctionBeginUser;
  /* PetscFOpen/PetscFPrintf/PetscFClose are collective; only rank 0 touches the file. */
  PetscCall(PetscFOpen(comm, xdmf, "w", &fp));
  PetscCall(PetscFPrintf(comm, fp, "<?xml version=\"1.0\" ?>\n"));
  PetscCall(PetscFPrintf(comm, fp, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"));
  PetscCall(PetscFPrintf(comm, fp, "<Xdmf Version=\"2.0\">\n  <Domain>\n"));
  PetscCall(PetscFPrintf(comm, fp, "    <Grid Name=\"poisson2d\" GridType=\"Uniform\">\n"));
  PetscCall(PetscFPrintf(comm, fp, "      <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"%" PetscInt_FMT " %" PetscInt_FMT "\"/>\n", my, mx));
  PetscCall(PetscFPrintf(comm, fp, "      <Geometry GeometryType=\"ORIGIN_DXDY\">\n"));
  PetscCall(PetscFPrintf(comm, fp, "        <DataItem Dimensions=\"2\" Format=\"XML\">0.0 0.0</DataItem>\n"));
  PetscCall(PetscFPrintf(comm, fp, "        <DataItem Dimensions=\"2\" Format=\"XML\">%.10g %.10g</DataItem>\n", (double)hy, (double)hx));
  PetscCall(PetscFPrintf(comm, fp, "      </Geometry>\n"));
  const char *names[3] = {"u", "u_exact", "error"};
  const char *bins[3]  = {ubin, exbin, errbin};
  for (int f = 0; f < 3; f++) {
    PetscCall(PetscFPrintf(comm, fp, "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Node\">\n", names[f]));
    PetscCall(PetscFPrintf(comm, fp, "        <DataItem Dimensions=\"%" PetscInt_FMT " %" PetscInt_FMT "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Endian=\"Little\">%s</DataItem>\n", my, mx, bins[f]));
    PetscCall(PetscFPrintf(comm, fp, "      </Attribute>\n"));
  }
  PetscCall(PetscFPrintf(comm, fp, "    </Grid>\n  </Domain>\n</Xdmf>\n"));
  PetscCall(PetscFClose(comm, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- single-resolution solve that writes solution output ------- */
static PetscErrorCode OutputSolve(PetscInt M, const char *prefix)
{
  DM        da;
  KSP       ksp;
  Vec       u, exact, err;
  PetscReal l2, linf;
  MPI_Comm  comm = PETSC_COMM_WORLD;
  char      ubin[PETSC_MAX_PATH_LEN], exbin[PETSC_MAX_PATH_LEN], errbin[PETSC_MAX_PATH_LEN];
  char      xdmf[PETSC_MAX_PATH_LEN], vts[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, M, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetDM(ksp, da));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, NULL));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, NULL));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetTolerances(ksp, 1e-10, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "u"));

  /* in-situ NaN/Inf detection (vis-spec: nan_inf_detection, in_situ) */
  PetscCall(CheckNanInf(u, "u"));

  PetscCall(DMCreateGlobalVector(da, &exact));
  PetscCall(DMCreateGlobalVector(da, &err));
  PetscCall(PetscObjectSetName((PetscObject)exact, "u_exact"));
  PetscCall(PetscObjectSetName((PetscObject)err, "error"));
  PetscCall(FillExactAndError(da, u, exact, err));

  PetscCall(ComputeError(da, u, &l2, &linf));
  PetscCall(PetscPrintf(comm, "[output solve] %" PetscInt_FMT "x%" PetscInt_FMT " grid  L2=%.6e  Linf=%.6e\n", M, M, (double)l2, (double)linf));

  /* raw binary field dumps + XDMF wrapper (no HDF5 in this PETSc build) */
  PetscCall(PetscSNPrintf(ubin, sizeof(ubin), "%s_u.bin", prefix));
  PetscCall(PetscSNPrintf(exbin, sizeof(exbin), "%s_u_exact.bin", prefix));
  PetscCall(PetscSNPrintf(errbin, sizeof(errbin), "%s_error.bin", prefix));
  PetscCall(PetscSNPrintf(xdmf, sizeof(xdmf), "%s.xdmf", prefix));
  PetscCall(WriteFieldBinary(da, u, ubin));
  PetscCall(WriteFieldBinary(da, exact, exbin));
  PetscCall(WriteFieldBinary(da, err, errbin));
  {
    const PetscReal hx = 1.0 / (M - 1), hy = 1.0 / (M - 1);
    PetscCall(WriteXDMF(comm, xdmf, M, M, hx, hy, ubin, exbin, errbin));
  }

  /* also emit a native VTK structured-grid file (ParaView reads .vts directly) */
  {
    PetscViewer vtk;
    PetscCall(PetscSNPrintf(vts, sizeof(vts), "%s.vts", prefix));
    PetscCall(PetscViewerVTKOpen(comm, vts, FILE_MODE_WRITE, &vtk));
    PetscCall(VecView(u, vtk));
    PetscCall(VecView(exact, vtk));
    PetscCall(VecView(err, vtk));
    PetscCall(PetscViewerDestroy(&vtk));
  }

  PetscCall(PetscPrintf(comm, "[output solve] wrote %s (+ .bin fields) and %s\n", xdmf, vts));

  PetscCall(VecDestroy(&exact));
  PetscCall(VecDestroy(&err));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&da));
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
      locMax = PetscMax(locMax, e);
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da, u, &arr));
  MPI_Comm  comm = PetscObjectComm((PetscObject)da);
  PetscReal gSum, gMax;
  PetscCallMPI(MPIU_Allreduce(&locSum, &gSum, 1, MPIU_REAL, MPI_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(&locMax, &gMax, 1, MPIU_REAL, MPI_MAX, comm));
  *l2   = PetscSqrtReal(hx * hy * gSum); /* discrete L2 norm */
  *linf = gMax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------- solve one refinement level ------- */
static PetscErrorCode SolveLevel(PetscInt M, PetscReal *h, PetscReal *l2, PetscReal *linf, PetscInt *its, PetscInt *dof)
{
  DM  da;
  KSP ksp;
  Vec u;

  PetscFunctionBeginUser;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, M, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
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
  PetscInt  base = 17, levels = 1, outGrid = 0;
  char      csv[PETSC_MAX_PATH_LEN]       = "";
  char      outPrefix[PETSC_MAX_PATH_LEN] = "poisson_solution";
  PetscBool haveCsv                       = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "2-D Poisson MMS verification on a structured DMDA.\n"));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mms_base", &base, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mms_levels", &levels, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mms_csv", csv, sizeof(csv), &haveCsv));
  /* -out_grid M (>0) triggers a single-resolution solve that writes solution
     output (XDMF + raw binary + VTK) with an in-situ NaN/Inf check. */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-out_grid", &outGrid, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-out_prefix", outPrefix, sizeof(outPrefix), NULL));

  PetscViewer vcsv = NULL;
  if (haveCsv) {
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, csv, &vcsv));
    PetscCall(PetscViewerASCIIPrintf(vcsv, "level,h,dof,L2,Linf,iterations\n"));
  }

  if (levels > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-6s %-12s %-8s %-14s %-8s %-14s %-8s %-6s\n", "level", "h", "dof", "L2", "rateL2", "Linf", "rateLinf", "its"));

  PetscReal l2prev = 0.0, linfPrev = 0.0, hprev = 0.0;
  for (PetscInt l = 0; l < levels; l++) {
    const PetscInt M = (base - 1) * (1 << l) + 1;
    PetscReal      h, l2, linf;
    PetscInt       its, dof;
    PetscCall(SolveLevel(M, &h, &l2, &linf, &its, &dof));

    PetscReal rL2 = 0.0, rLinf = 0.0;
    if (l > 0) {
      rL2   = PetscLogReal(l2prev / l2) / PetscLogReal(hprev / h);
      rLinf = PetscLogReal(linfPrev / linf) / PetscLogReal(hprev / h);
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-6" PetscInt_FMT " %-12.3e %-8" PetscInt_FMT " %-14.6e %-8.3f %-14.6e %-8.3f %-6" PetscInt_FMT "\n", l, (double)h, dof, (double)l2, (double)rL2, (double)linf, (double)rLinf, its));
    if (haveCsv) PetscCall(PetscViewerASCIIPrintf(vcsv, "%" PetscInt_FMT ",%.6e,%" PetscInt_FMT ",%.6e,%.6e,%" PetscInt_FMT "\n", l, (double)h, dof, (double)l2, (double)linf, its));
    l2prev   = l2;
    linfPrev = linf;
    hprev    = h;
  }

  if (haveCsv) PetscCall(PetscViewerDestroy(&vcsv));

  if (outGrid > 0) PetscCall(OutputSolve(outGrid, outPrefix));

  PetscCall(PetscFinalize());
  return 0;
}
