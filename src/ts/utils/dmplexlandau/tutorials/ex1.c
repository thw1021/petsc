static char help[] = "Landau collision operator driver\n\n";

#include <petscts.h>
#include <petsclandau.h>
#include <petscdmcomposite.h>

/*
 call back method for DMPlexLandauAccess:

Input Parameters:
 .   dm - a DM for this field
 -   local_field - the local index in the grid for this field
 .   grid - the grid index
 +   b_id - the batch index
 -   vctx - a user context

 Input/Output Parameters:
 +   x - Vector to data to

 */
PetscErrorCode landau_field_print_access_callback(DM dm, Vec x, PetscInt local_field, PetscInt grid, PetscInt b_id, void *vctx)
{
  LandauCtx  *ctx;
  PetscScalar val;
  PetscInt species;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  species = ctx->species_offset[grid] + local_field;
  val = (PetscScalar)(LAND_PACK_IDX(b_id, grid) + (species + 1) * 10);
  PetscCall(VecSet(x, val));
  PetscCall(PetscInfo(dm, "DMPlexLandauAccess user 'add' method to grid %" PetscInt_FMT ", batch %" PetscInt_FMT " and local field %" PetscInt_FMT " with %" PetscInt_FMT " grids\n", grid, b_id, local_field, ctx->num_grids));

  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts, PetscInt stepi, PetscReal time, Vec X, void *actx)
{
  LandauCtx *ctx   = (LandauCtx *)actx; /* user-defined application context */
  PetscInt nDMs, id, grid_view_idx = ctx->verbose;
  DM pack;
  PetscReal time2;
  Vec           *XsubArray = NULL;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &pack));
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
  PetscCall(DMGetOutputSequenceNumber(ctx->plex[grid_view_idx], &id, &time2));
  PetscCall(DMSetOutputSequenceNumber(ctx->plex[grid_view_idx], id+1, time));
  PetscCall(PetscInfo(pack, "ex1 plot step %d, grid %d, time = %g\n", (int)id, (int)grid_view_idx, (double)time));
  PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(VecViewFromOptions(XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, grid_view_idx)], NULL, "-ex1_vec_view"));
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));
  PetscCall(PetscFree(XsubArray));
  PetscCall(DMPlexLandauPrintNorms(X, id+1));

  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             pack;
  Vec            X;
  PetscInt       dim = 2, nDMs, grid_view_idx = 0;
  TS             ts;
  Mat            J;
  Vec           *XsubArray = NULL;
  LandauCtx     *ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  /* Create a mesh */
  PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack));
  PetscCall(DMSetUp(pack));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
  PetscCall(DMPlexLandauPrintNorms(X, 0));
  ctx->verbose = grid_view_idx; //co-opt 'verbose'. Not used in Landau after setup
  /* output plot */
  PetscCall(PetscMalloc(sizeof(*XsubArray) * nDMs, &XsubArray));
  PetscCall(DMCompositeGetAccessArray(pack, X, nDMs, NULL, XsubArray)); // read only
  PetscCall(PetscObjectSetName((PetscObject)XsubArray[LAND_PACK_IDX(ctx->batch_view_idx, grid_view_idx)], grid_view_idx == 0 ? "ue" : "ui"));
  PetscCall(DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, XsubArray));
  PetscCall(PetscFree(XsubArray));
  PetscCall(DMSetOutputSequenceNumber(ctx->plex[grid_view_idx], -1, 0.0));
  PetscCall(DMViewFromOptions(ctx->plex[grid_view_idx], NULL, "-ex1_dm_view"));
  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetDM(ts, pack));
  PetscCall(TSSetIFunction(ts, NULL, DMPlexLandauIFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, DMPlexLandauIJacobian, NULL));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSMonitorSet(ts, Monitor, ctx, NULL));
  PetscCall(TSSolve(ts, X));
  PetscCall(DMPlexLandauPrintNorms(X, 1));
  /* test add field method & output */
  /* PetscCall(DMPlexLandauAccess(pack, X, landau_field_print_access_callback, NULL)); */
  /* PetscCall(Monitor(ts, -1, 1.0, X, ctx)); */
  /* clean up */
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    requires: p4est !complex double defined(PETSC_USE_DMLANDAU_2D)
    output_file: output/ex1_0.out
    filter: grep -v "%  type: seq"
    args: -dm_landau_num_species_grid 1,2 -petscspace_degree 3 -petscspace_poly_tensor 1 -dm_landau_type p4est -dm_landau_ion_masses 2,4 -dm_landau_ion_charges 1,18 -dm_landau_thermal_temps 5,5,.5 -dm_landau_n 1.00018,1,1e-5 -dm_landau_n_0 1e20 -ts_monitor -snes_rtol 1.e-14 -snes_stol 1.e-14 -snes_monitor -snes_converged_reason -ts_type arkimex -ts_arkimex_type 1bee -ts_max_snes_failures -1 -ts_rtol 1e-1 -ts_dt 1.e-1 -ts_max_time 1 -ts_adapt_clip .5,1.25 -ts_adapt_scale_solve_failed 0.75 -ts_adapt_time_step_increase_delay 5 -ts_max_steps 1 -pc_type lu -ksp_type preonly -dm_landau_amr_levels_max 2,1 -ex1_dm_view  -ex1_vec_view ::ascii_matlab -dm_landau_num_cells 2,2 -dm_landau_domain_max_par 6,6 -dm_landau_domain_max_perp 4,4
    test:
      suffix: cpu
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos
    test:
      suffix: cuda
      requires: cuda
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda -mat_cusparse_use_cpu_solve

TEST*/
