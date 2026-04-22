static char help[] = "Conservation test for the excess pitch-angle scattering operator (Option D).\n\n"
                     "Verifies that LandauPitchAngleCreateMatrix produces a matrix whose row sums\n"
                     "are zero (discrete particle conservation) and whose energy-weighted row sums\n"
                     "are zero (discrete energy conservation):\n\n"
                     "   integral Delta C_pitch[f] dv = 0          (particle conservation)\n"
                     "   integral v^2 Delta C_pitch[f] dv = 0      (energy conservation)\n\n"
                     "The stiffness matrix Apitch satisfies:\n"
                     "   Apitch * 1    = 0  (particle conservation, row-sum test)\n"
                     "   Apitch^T * v^2 = 0  (energy conservation, since Apitch is symmetric)\n\n"
                     "Run with:\n"
                     "  mpirun -n 1 ./ex3 \\\n"
                     "    -dim 3 -dm_landau_num_species_grid 1 -dm_landau_thermal_temps 1 \\\n"
                     "    -dm_landau_device_type cpu -dm_landau_domain_radius 6 \\\n"
                     "    -petscspace_degree 2 -dm_landau_num_cells 4,4,4 \\\n"
                     "    -dm_landau_type p8est\n\n"
                     "Note: when using the pitch-angle operator inside a TS/SNES solve, pass\n"
                     "  -snes_rtol 1.e-14 -snes_stol 1.e-14 for 13-digit energy conservation.\n\n";

#include <petscdmplex.h>
#include <petsclandau.h>
#include <petscts.h>
#include <petscds.h>
#include <petscdmcomposite.h>

/* ------------------------------------------------------------------ */
/* Callback for DMProjectFunction: returns v^2 = x[0]^2+x[1]^2+x[2]^2 */
/* (3D Cartesian) or v^2 = x[0]^2+x[1]^2 (2D cylindrical).            */
/* This projects the function v^2 onto the FEM space to get v2_vec,    */
/* the FEM coefficient vector for v^2.                                  */
/* ------------------------------------------------------------------ */
static PetscErrorCode v2_func(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscFunctionBegin;
  (void)time;
  (void)Nf;
  (void)ctx;
  if (dim == 2) u[0] = x[0] * x[0] + x[1] * x[1];      /* v_perp^2 + v_par^2 */
  else u[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; /* vx^2 + vy^2 + vz^2 */
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM            pack;
  Vec           X, J_vec, F_pitch, ones_vec, Apitch_ones, v2_vec, Apitch_v2;
  Mat           Apitch;
  LandauCtx    *ctx;
  PitchAngleCtx pa_ctx;
  PetscInt      dim = 3, nDMs;
  PetscReal     row_sum_inf, row_sum_abs;
  PetscReal     F_norm, particle_err;
  PetscReal     energy_inf, v2_norm;
  PetscBool     pass = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));

  /* ---------------------------------------------------------------- */
  /* 1. Create velocity space DM and initial Maxwellian distribution  */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "=== Option D Conservation Test (ex3) ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Creating velocity space (dim=%d)...\n", (int)dim));
  {
    Mat J_mat_tmp;
    PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_WORLD, dim, "", &J_vec, &J_mat_tmp, &pack));
    /* J_mat_tmp is stored as ctx->J and destroyed by DMPlexLandauDestroyVelocitySpace */
  }
  /* J_vec is the initial distribution (Maxwellian).
     Both J_vec and J_mat are owned by the DM and will be destroyed by
     DMPlexLandauDestroyVelocitySpace.  We keep X as an alias -- do NOT
     destroy it separately. */
  X = J_vec;
  PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
  PetscCall(DMGetApplicationContext(pack, &ctx));
  PetscCall(DMSetUp(pack));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  num_grids=%d  batch_sz=%d  nDMs=%d\n", (int)ctx->num_grids, (int)ctx->batch_sz, (int)nDMs));

  /* ---------------------------------------------------------------- */
  /* 2. Build the excess pitch-angle matrix                           */
  /*    Use Ar2+ parameters from Hesslow et al. 2018 as a test case  */
  /* ---------------------------------------------------------------- */
  pa_ctx.lnL2_factor  = 150.0;  /* Ar2+ enhancement factor */
  pa_ctx.lnL_standard = 17.0;   /* standard Coulomb logarithm */
  pa_ctx.T_e_eV       = 5.0;    /* bulk electron temperature [eV] */
  pa_ctx.n_e          = 2.0e20; /* electron density [m^-3] */
  pa_ctx.Z_eff        = 2.0;    /* effective charge */
  pa_ctx.Apitch       = NULL;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Building pitch-angle matrix (lnL2_factor=%.0f)...\n", (double)pa_ctx.lnL2_factor));
  PetscCall(LandauPitchAngleCreateMatrix(pack, &pa_ctx));
  Apitch = pa_ctx.Apitch;

  /* ---------------------------------------------------------------- */
  /* 3. Conservation check 1: row-sum test (particle conservation)    */
  /*                                                                  */
  /* The stiffness matrix Apitch satisfies Apitch * 1 = 0 because    */
  /* the constant function is in the null space of any elliptic       */
  /* operator with natural (Neumann) boundary conditions.             */
  /*                                                                  */
  /* Equivalently: 1^T * Apitch * X = VecSum(Apitch * X) = 0         */
  /* for any X, which is the discrete particle conservation law.      */
  /* ---------------------------------------------------------------- */
  PetscCall(VecDuplicate(X, &ones_vec));
  PetscCall(VecSet(ones_vec, 1.0));
  PetscCall(VecDuplicate(X, &Apitch_ones));
  PetscCall(MatMult(Apitch, ones_vec, Apitch_ones));
  PetscCall(VecNorm(Apitch_ones, NORM_INFINITY, &row_sum_inf));
  {
    PetscScalar row_sum_sc;
    PetscCall(VecSum(Apitch_ones, &row_sum_sc));
    row_sum_abs = PetscAbsReal(PetscRealPart(row_sum_sc));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--- Row-sum test: Apitch * ones ---\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||Apitch * ones||_inf = %10.3e  (should be ~0, roundoff only)\n", (double)row_sum_inf));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  |sum(Apitch * ones)|  = %10.3e  (should be ~0)\n", (double)row_sum_abs));

  /* Threshold: for a stiffness matrix, row sums should be at floating-point
     roundoff level relative to the matrix norm. Accept up to 1e-6. */
  if (row_sum_inf > 1e-6) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL: ||Apitch*ones||_inf = %10.3e > 1e-6\n", (double)row_sum_inf));
    pass = PETSC_FALSE;
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS: row sums are near zero (particle conservation)\n"));
  }

  /* ---------------------------------------------------------------- */
  /* 4. Conservation check 2: apply to Maxwellian                    */
  /*                                                                  */
  /* F_pitch = Apitch * X  (X is the initial Maxwellian)             */
  /* VecSum(F_pitch) = 1^T * Apitch * X should be ~0                 */
  /* This is the discrete integral Delta C_pitch[f_Maxwellian] dv = 0 check. */
  /* ---------------------------------------------------------------- */
  PetscCall(VecDuplicate(X, &F_pitch));
  PetscCall(MatMult(Apitch, X, F_pitch));
  PetscCall(VecNorm(F_pitch, NORM_2, &F_norm));

  {
    PetscScalar particle_sum_sc;
    PetscCall(VecSum(F_pitch, &particle_sum_sc));
    particle_err = PetscAbsReal(PetscRealPart(particle_sum_sc));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--- Particle conservation: VecSum(Apitch * X_Maxwellian) ---\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||Apitch * X||_2          = %10.3e\n", (double)F_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  |VecSum(Apitch * X)|      = %10.3e  (should be ~0)\n", (double)particle_err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  relative error            = %10.3e  (|sum|/||F||)\n", (double)(F_norm > 0 ? particle_err / F_norm : particle_err)));

  if (particle_err / (F_norm + 1e-300) > 1e-10) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL: relative particle error %10.3e > 1e-10\n", (double)(particle_err / (F_norm + 1e-300))));
    pass = PETSC_FALSE;
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS: particle conservation\n"));
  }

  /* ---------------------------------------------------------------- */
  /* 5. Conservation check 3: energy conservation (numerical)        */
  /*                                                                  */
  /* The pitch-angle operator conserves kinetic energy because        */
  /* P_ij v_j = (v^2 delta_ij - v_i v_j)/v^3 * v_j = 0 (traceless in v). */
  /*                                                                  */
  /* Correct discrete check: Apitch * v2_vec ~= 0                   */
  /* where v2_vec is the FEM coefficient vector for v^2,             */
  /* obtained by projecting v^2 onto the FEM space via               */
  /* DMProjectFunction.                                               */
  /*                                                                  */
  /* Since Apitch is symmetric, energy conservation means v^2 is in  */
  /* the null space of Apitch: Apitch * v2_vec = 0 for all f.       */
  /* This is the energy-weighted row-sum test, analogous to the      */
  /* particle row-sum test (Apitch * ones = 0).                      */
  /*                                                                  */
  /* Threshold: ||Apitch * v2_vec||_inf / ||v2_vec||_inf < 1e-6      */
  /* (same relative threshold as the particle row-sum test).         */
  /* ---------------------------------------------------------------- */
  PetscCall(VecDuplicate(X, &v2_vec));
  {
    /* Project v^2 onto the FEM space using DMProjectFunction.
       The pack DM is a composite; we need the sub-DM for grid 0.
       Use DMCompositeGetAccessArray to get the sub-vector, project
       onto the sub-DM, then restore. */
    Vec *XsubArray = NULL;
    DM   dm0;
    Vec  v2_sub;
    PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *) = {v2_func};
    void *ctxs[1]                                                                                       = {NULL};

    PetscCall(PetscMalloc1(nDMs, &XsubArray));
    PetscCall(DMCompositeGetAccessArray(pack, v2_vec, nDMs, NULL, XsubArray));

    /* Grid 0 sub-DM */
    PetscCall(DMCompositeGetEntriesArray(pack, &dm0));
    v2_sub = XsubArray[LAND_PACK_IDX(0, 0)];

    /* Project v^2(x) onto the FEM space of dm0 */
    PetscCall(DMProjectFunction(dm0, 0.0, funcs, ctxs, INSERT_ALL_VALUES, v2_sub));

    PetscCall(DMCompositeRestoreAccessArray(pack, v2_vec, nDMs, NULL, XsubArray));
    PetscCall(PetscFree(XsubArray));
  }

  /* Energy-weighted row-sum: Apitch * v2_vec should be ~0 */
  PetscCall(VecDuplicate(X, &Apitch_v2));
  PetscCall(MatMult(Apitch, v2_vec, Apitch_v2));
  PetscCall(VecNorm(Apitch_v2, NORM_INFINITY, &energy_inf));
  PetscCall(VecNorm(v2_vec, NORM_INFINITY, &v2_norm));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--- Energy conservation: ||Apitch * v2_vec||_inf ---\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||v2_vec||_inf            = %10.3e\n", (double)v2_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||Apitch * v2_vec||_inf   = %10.3e  (should be ~0)\n", (double)energy_inf));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  relative error            = %10.3e  (||Apitch*v2||/||v2||)\n", (double)(energy_inf / (v2_norm + 1e-300))));

  if (energy_inf / (v2_norm + 1e-300) > 1e-6) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  FAIL: relative energy error %10.3e > 1e-6\n", (double)(energy_inf / (v2_norm + 1e-300))));
    pass = PETSC_FALSE;
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  PASS: energy conservation (v^2 in null space of Apitch)\n"));
  }

  /* ---------------------------------------------------------------- */
  /* 6. Summary                                                        */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== Summary ===\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||Apitch * ones||_inf   = %10.3e  (particle conservation, row-sum test)\n", (double)row_sum_inf));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  |VecSum(Apitch*X)|      = %10.3e  (particle conservation, Maxwellian)\n", (double)particle_err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  ||Apitch*v2||/||v2||    = %10.3e  (energy conservation, v^2 null-space test)\n", (double)(energy_inf / (v2_norm + 1e-300))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n=== Result: %s ===\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED"));

  /* ---------------------------------------------------------------- */
  /* 7. Clean up                                                       */
  /* ---------------------------------------------------------------- */
  PetscCall(LandauPitchAngleDestroy(&pa_ctx));
  PetscCall(VecDestroy(&ones_vec));
  PetscCall(VecDestroy(&Apitch_ones));
  PetscCall(VecDestroy(&F_pitch));
  PetscCall(VecDestroy(&v2_vec));
  PetscCall(VecDestroy(&Apitch_v2));
  /* DMPlexLandauDestroyVelocitySpace destroys ctx->J (the Jacobian matrix)
     internally, so do NOT call MatDestroy(&J_mat) afterward.
     The Vec J_vec is NOT owned by the DM -- destroy it after. */
  PetscCall(DMPlexLandauDestroyVelocitySpace(&pack));
  PetscCall(VecDestroy(&J_vec));
  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}

/*TEST

  test:
    requires: double !complex !defined(PETSC_USE_DMLANDAU_2D) p4est
    suffix: 3d_cpu_amr1
    nsize: 1
    args: -dim 3 -dm_landau_num_species_grid 1 -dm_landau_thermal_temps 1 \
          -dm_landau_device_type cpu -dm_landau_domain_radius 6 \
          -petscspace_degree 2 -dm_landau_num_cells 4,4,4 \
          -dm_landau_type p8est -dm_landau_amr_levels_max 1 -dm_landau_verbose 0 \
          -snes_rtol 1.e-14 -snes_stol 1.e-14
    filter: grep -E "(PASS|FAIL|Result)"
    output_file: output/ex3_3d_cpu_amr1.out

TEST*/
