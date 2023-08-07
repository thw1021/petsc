/*
(vector-)Poisson in N-D. Modeled by the PDE:

  - delta u = f on Omega
  u = 0 on partial Omega

u can be a vector field, in that case the linear system solved is a separate laplacian per field.

This benchmark is intended to measure bandwidth and flop rates.
We are not interested in the PDE we are solving.
We thus always fill the nonzero values for a standard FD stencil.
We assume full coupling between fields, with blocks fill by zeros.

Exampe usage:

  Run on GPU (requires respective backends installed):
    ./bench_kspsolve_dmda -dm_vec_type cuda -dm_mat_type aijcusparse
    ./bench_kspsolve_dmda -dm_vec_type hip -dm_mat_type aijhipsparse
    ./bench_kspsolve_dmda -dm_vec_type kokkos -dm_mat_type aijkokkos

  Test only MatMult:
    ./bench_kspsolve_dmda -matmult

  Test MatMult over 1000 iterations:
    ./bench_kspsolve_dmda -matmult -its 1000

  Change size of problem (e.g., use a 128x128x128 grid):
    ./bench_kspsolve_dmda -da_grid_x 128 -da_grid_y 128 -da_grid_z 128
*/
static char help[] = "Solves ND Laplacian with various stencils.\n";

#include <petscksp.h>
#include <petscdmda.h>

PetscErrorCode FillSystem(Mat A, Vec b)
{
  PetscInt     dim, dof, mx, my, mz, xm, ym, zm, xs, ys, zs;
  PetscScalar *v, Hx, Hy, Hz, dHx, dHy, dHz;
  MatStencil   row, col[7];
  DM           da;

  PetscFunctionBeginUser;
  PetscCall(MatGetDM(A, &da));

  PetscCall(DMDAGetInfo(da, &dim, &mx, &my, &mz, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  Hx  = 1.0 / (PetscReal)(mx - 1);
  Hy  = my > 1 ? 1.0 / (PetscReal)(my - 1) : 1.0;
  Hz  = mz > 1 ? 1.0 / (PetscReal)(mz - 1) : 1.0;
  dHx = Hy * Hz / Hx;
  dHy = Hx * Hz / Hy;
  dHz = Hx * Hy / Hz;
  if (my == 1) dHy = 0.0;
  if (mz == 1) dHz = 0.0;

  PetscCall(VecSet(b, Hx * Hy * Hz));

  PetscCall(PetscMalloc1(7 * dof, &v));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; k++) {
    for (PetscInt j = ys; j < ys + ym; j++) {
      for (PetscInt i = xs; i < xs + xm; i++) {
        for (PetscInt d = 0; d < dof; d++) {
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = d;
          if (i == 0 || (dim > 1 && j == 0) || (dim > 2 && k == 0) || i == mx - 1 || (dim > 1 && j == my - 1) || (dim > 2 && k == mz - 1)) {
            v[0] = 2.0 * (dHx + dHy + dHz);
            PetscCall(MatSetValuesStencil(A, 1, &row, 1, &row, v, INSERT_VALUES));
          } else {
            PetscInt n = 0;

            v[n]     = 2.0 * (dHx + dHy + dHz);
            col[n].i = i;
            col[n].j = j;
            col[n].k = k;
            col[n].c = d;
            n++;

            v[n]     = -dHx;
            col[n].i = i - 1;
            col[n].j = j;
            col[n].k = k;
            col[n].c = d;
            n++;

            v[n]     = -dHx;
            col[n].i = i + 1;
            col[n].j = j;
            col[n].k = k;
            col[n].c = d;
            n++;

            if (dim > 1) {
              v[n]     = -dHy;
              col[n].i = i;
              col[n].j = j - 1;
              col[n].k = k;
              col[n].c = d;
              n++;

              v[n]     = -dHy;
              col[n].i = i;
              col[n].j = j + 1;
              col[n].k = k;
              col[n].c = d;
              n++;
            }
            if (dim > 2) {
              v[n]     = -dHz;
              col[n].i = i;
              col[n].j = j;
              col[n].k = k - 1;
              col[n].c = d;
              n++;

              v[n]     = -dHz;
              col[n].i = i;
              col[n].j = j;
              col[n].k = k + 1;
              col[n].c = d;
              n++;
            }
            PetscCall(MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES));
          }
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscLogDouble             time_start, time_mid1 = 0.0, time_mid2 = 0.0, time_end, time_avg, floprate;
  PETSC_UNUSED PetscLogStage stage;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Benchmark options */
  PetscInt  its         = 100;         /* No of matmult_iterations */
  PetscBool matmult     = PETSC_FALSE; /* Do MatMult() only */
  PetscBool splitksp    = PETSC_FALSE; /* Split KSPSolve and PCSetUp */
  PetscBool printTiming = PETSC_TRUE;  /* If run in CI, do not print timing result */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-its", &its, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-matmult", &matmult, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-split_ksp", &splitksp, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_timing", &printTiming, NULL));

  /* Topological dimension and default grid */
  PetscInt dim = 3, grid[3] = {64, 64, 64};
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));

  /* PDE options */
  PetscInt  dof        = 1;           /* No of dofs per point */
  PetscBool fd_stencil = PETSC_FALSE; /* Use FD type stencil */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof", &dof, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fd_stencil", &fd_stencil, NULL));

  /* Create structured grid */
  DM              dm;
  DMDAStencilType stencil_type = fd_stencil ? DMDA_STENCIL_STAR : DMDA_STENCIL_BOX;
  PetscInt        stencil_pts;
  switch (dim) {
  case 3:
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, stencil_type, grid[0], grid[1], grid[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, 1, NULL, NULL, NULL, &dm));
    stencil_pts = fd_stencil ? 7 : 27;
    break;
  case 2:
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, stencil_type, grid[0], grid[1], PETSC_DECIDE, PETSC_DECIDE, dof, 1, NULL, NULL, &dm));
    stencil_pts = fd_stencil ? 5 : 9;
    break;
  case 1:
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, grid[0], dof, 1, NULL, &dm));
    stencil_pts = 3;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDASetUniformCoordinates(dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  /* Create matrix and get some info */
  PetscInt N;
  MatInfo  info;
  Mat      A;
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
  PetscCall(MatGetSize(A, &N, NULL));

  /* Due to errors in MatSolve_SeqAIJHIPSPARSE_ICC0() */
  PetscBool iship;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iship, MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, ""));
  if (!iship) {
    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SPD_ETERNAL, PETSC_TRUE));
  }

  /* Print banner */
  PetscCall(DMDAGetInfo(dm, &dim, &grid[0], &grid[1], &grid[2], NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCount global_nnz = (PetscCount)info.nz_allocated;
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "===========================================\n"));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Test: %s performance\n", matmult ? "MatMult" : "KSP"));
  PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Grid: %" PetscInt_FMT "x%" PetscInt_FMT "x%" PetscInt_FMT "\n", grid[0], grid[1], grid[2]));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Fields: %" PetscInt_FMT "\n", dof));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Input matrix: %" PetscInt_FMT "-pt stencil\n", stencil_pts));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "DoFs: %" PetscInt_FMT "\n", N));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Number of nonzeros: %" PetscCount_FMT "\n", global_nnz));
  PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*  Create the Vecs and Mat                                            */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nStep1  - creating Vecs and Mat...\n"));
  PetscCall(PetscLogStageRegister("Step1  - Vecs and Mat", &stage));
  PetscCall(PetscLogStagePush(stage));

  /* Allocate vectors*/
  Vec x, b;
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &b));

  /* Fill matrix and rhs */
  PetscCall(FillSystem(A, b));

  PetscCall(PetscLogStagePop());

  if (matmult) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  MatMult                                                            */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(VecSetRandom(x, NULL));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Step2  - running MatMult() %" PetscInt_FMT " times...\n", its));
    PetscCall(PetscLogStageRegister("Step2  - MatMult", &stage));
    PetscCall(PetscLogStagePush(stage));
    PetscCall(PetscTime(&time_start));
    for (PetscInt i = 0; i < its; i++) PetscCall(MatMult(A, x, b));
    PetscCall(PetscTime(&time_end));
    PetscCall(PetscLogStagePop());
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  Calculate Performance metrics                                      */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    time_avg = (time_end - time_start) / ((PetscLogDouble)its);
    floprate = 2 * global_nnz / time_avg * 1e-9;
    if (printTiming) {
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\n%-15s%-7.5f seconds\n", "Average time:", time_avg));
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-9.3e Gflops/sec\n", "FOM:", floprate)); /* figure of merit */
    }
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "===========================================\n"));
  } else {
    KSP ksp;
    if (!splitksp) {
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      /*  Solve the linear system of equations                               */
      /*  Measure only time of PCSetUp() and KSPSolve()                      */
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Step2  - running KSPSolve()...\n"));
      PetscCall(PetscLogStageRegister("Step2  - KSPSolve", &stage));
      PetscCall(PetscLogStagePush(stage));
      PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(PetscTime(&time_start));
      PetscCall(KSPSolve(ksp, b, x));
      PetscCall(PetscTime(&time_end));
      PetscCall(PetscLogStagePop());
    } else {
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      /*  Solve the linear system of equations                               */
      /*  Measure only time of PCSetUp() and KSPSolve()                      */
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      PC pc; /* Preconditioner */
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Step2a - running PCSetUp()...\n"));
      PetscCall(PetscLogStageRegister("Step2a - PCSetUp", &stage));
      PetscCall(PetscLogStagePush(stage));
      PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PetscTime(&time_start));
      PetscCall(PCSetUp(pc));
      PetscCall(PetscTime(&time_mid1));
      PetscCall(PetscLogStagePop());
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Step2b - running KSPSolve()...\n"));
      PetscCall(PetscLogStageRegister("Step2b - KSPSolve", &stage));
      PetscCall(PetscLogStagePush(stage));
      PetscCall(PetscTime(&time_mid2));
      PetscCall(KSPSolve(ksp, b, x));
      PetscCall(PetscTime(&time_end));
      PetscCall(PetscLogStagePop());
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*  Summary                                                            */
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(KSPGetIterationNumber(ksp, &its));
    if (printTiming) {
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-3" PetscInt_FMT "\n", "KSP iters:", its));
      if (splitksp) {
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-7.5f seconds\n", "PCSetUp:", time_mid1 - time_start));
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-7.5f seconds\n", "KSPSolve:", time_end - time_mid2));
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-7.5f seconds\n", "Total Solve:", time_end - time_start));
      } else {
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-7.5f seconds\n", "KSPSolve:", time_end - time_start));
      }
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "%-15s%-1.3e DoFs/sec\n", "FOM:", N / (time_end - time_start))); /* figure of merit */
    }
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "===========================================\n"));
    PetscCall(KSPDestroy(&ksp));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*  Free up memory                                                     */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -print_timing false -matmult -its 10 -da_grid_x 8 -da_grid_y 8 -da_grid_z 8
    nsize: {{1 3}}
    output_file: output/bench_kspsolve_dmda_matmult.out

    test:
      suffix: matmult

    test:
      suffix: hip_matmult
      requires: hip
      args: -dm_vec_type hip -dm_mat_type aijhipsparse

    test:
      suffix: cuda_matmult
      requires: cuda
      args: -dm_vec_type cuda -dm_mat_type aijcusparse

    test:
      suffix: kok_matmult
      requires: kokkos_kernels
      args: -dm_vec_type kokkos -dm_mat_type aijkokkos

  testset:
    args: -print_timing false -its 10 -da_grid_x 8 -da_grid_y 8 -da_grid_z 8
    nsize: {{1 3}}
    output_file: output/bench_kspsolve_dmda_ksp.out

    test:
      suffix: ksp

    test:
      suffix: hip_ksp
      requires: hip
      args: -dm_vec_type hip -dm_mat_type aijhipsparse

    test:
      suffix: cuda_ksp
      requires: cuda
      args: -dm_vec_type cuda -dm_mat_type aijcusparse

    test:
      suffix: kok_ksp
      requires: kokkos_kernels
      args: -dm_vec_type kokkos -dm_mat_type aijkokkos
TEST*/
