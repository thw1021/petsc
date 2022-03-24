static char help[] = "Grid based Landau collision operator with PIC interface with OpenMP setup. (one species per grid)\n";

/*
   Support 2D with axisymmetric coordinates
     - r,z coordinates
     - Domain and species data input by Landau operator
     - "radius" for each grid, normalized with electron thermal velocity
     - Domain: (0,radius) x (-radius,radius), thus first coordinate x[0] is perpendicular velocity and 2pi*x[0] term is added for axisymmetric
 */

#include "petscdmplex.h"
#include "petscds.h"
#include "petscdmswarm.h"
#include "petscksp.h"
#include <petsc/private/petscimpl.h>
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
#include <omp.h>
#endif
#include <petsclandau.h>
#include <petscdmcomposite.h>

typedef struct {
  Mat MpTrans;
  Mat Mp;
  Vec ff;
  Vec uu;
} MatShellCtx;

PetscErrorCode MatMultMtM_SeqAIJ(Mat MtM,Vec xx,Vec yy)
{
  MatShellCtx    *matshellctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(MtM,&matshellctx);CHKERRQ(ierr);
  PetscCheck(matshellctx,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  ierr = MatMult(matshellctx->Mp, xx, matshellctx->ff);CHKERRQ(ierr);
  ierr = MatMult(matshellctx->MpTrans, matshellctx->ff, yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAddMtM_SeqAIJ(Mat MtM,Vec xx, Vec yy, Vec zz)
{
  MatShellCtx    *matshellctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(MtM,&matshellctx);CHKERRQ(ierr);
  PetscCheck(matshellctx,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "No context");
  ierr = MatMult(matshellctx->Mp, xx, matshellctx->ff);CHKERRQ(ierr);
  ierr = MatMultAdd(matshellctx->MpTrans, matshellctx->ff, yy, zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode createSwarm(const DM dm, DM *sw)
{
  PetscErrorCode ierr;
  PetscInt       Nc = 1, dim = 2;

  PetscFunctionBeginUser;
  ierr = DMCreate(PETSC_COMM_SELF, sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", Nc, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode gridToParticles(const DM dm, DM sw, Vec rhs, Vec work, Mat M_p, Mat Mass)
{
  PetscBool      is_lsqr;
  KSP            ksp;
  Mat            PM_p=NULL,MtM,D;
  Vec            ff;
  PetscErrorCode ierr;
  PetscInt       N, M, nzl;
  MatShellCtx    *matshellctx;

  PetscFunctionBeginUser;
  ierr = MatMult(Mass, rhs, work);CHKERRQ(ierr);
  ierr = VecCopy(work, rhs);CHKERRQ(ierr);
  // pseudo-inverse
  ierr = KSPCreate(PETSC_COMM_SELF, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp, "ftop_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPLSQR,&is_lsqr);
  if (!is_lsqr) {
    ierr = MatGetLocalSize(M_p, &M, &N);CHKERRQ(ierr);
    if (N>M) {
      PC        pc;
      ierr = PetscInfo(ksp, " M (%" PetscInt_FMT ") < M (%" PetscInt_FMT ") -- skip revert to lsqr\n",M,N);CHKERRQ(ierr);
      is_lsqr = PETSC_TRUE;
      ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr); // could put in better solver -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
    } else {
      ierr = PetscNew(&matshellctx);CHKERRQ(ierr);
      ierr = MatCreateShell(PetscObjectComm((PetscObject)dm),N,N,PETSC_DECIDE,PETSC_DECIDE,matshellctx,&MtM);CHKERRQ(ierr);
      ierr = MatTranspose(M_p,MAT_INITIAL_MATRIX,&matshellctx->MpTrans);CHKERRQ(ierr);
      matshellctx->Mp = M_p;
      ierr = MatShellSetOperation(MtM, MATOP_MULT, (void (*)(void))MatMultMtM_SeqAIJ);CHKERRQ(ierr);
      ierr = MatShellSetOperation(MtM, MATOP_MULT_ADD, (void (*)(void))MatMultAddMtM_SeqAIJ);CHKERRQ(ierr);
      ierr = MatCreateVecs(M_p,&matshellctx->uu,&matshellctx->ff);CHKERRQ(ierr);
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,1,NULL,&D);CHKERRQ(ierr);
      ierr = MatViewFromOptions(matshellctx->MpTrans,NULL,"-ftop2_Mp_mat_view");CHKERRQ(ierr);
      for (int i=0 ; i<N ; i++) {
        const PetscScalar *vals;
        const PetscInt    *cols;
        PetscScalar dot = 0;
        ierr = MatGetRow(matshellctx->MpTrans,i,&nzl,&cols,&vals);CHKERRQ(ierr);
        for (int ii=0 ; ii<nzl ; ii++) dot += PetscSqr(vals[ii]);
        PetscCheck(dot!=0.0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Row %" PetscInt_FMT " is empty", i);
        ierr = MatSetValue(D,i,i,dot,INSERT_VALUES);
      }
      ierr = MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = PetscInfo(M_p,"createMtMKSP Have %" PetscInt_FMT " eqs, nzl = %" PetscInt_FMT "\n",N,nzl);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp, MtM, D);CHKERRQ(ierr);
      ierr = MatViewFromOptions(D,NULL,"-ftop2_D_mat_view");CHKERRQ(ierr);
      ierr = MatViewFromOptions(M_p,NULL,"-ftop2_Mp_mat_view");CHKERRQ(ierr);
      ierr = MatViewFromOptions(matshellctx->MpTrans,NULL,"-ftop2_MpTranspose_mat_view");CHKERRQ(ierr);
    }
  }
  if (is_lsqr) {
    PC        pc;
    PetscBool is_bjac;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&is_bjac);
    if (is_bjac) {
      ierr = DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp, M_p, PM_p);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(ksp, M_p, M_p);CHKERRQ(ierr);
    }
  }
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr); // this grabs access
  if (!is_lsqr) {
    ierr = KSPSolve(ksp, rhs, matshellctx->uu);CHKERRQ(ierr);
    ierr = MatMult(M_p, matshellctx->uu, ff);CHKERRQ(ierr);
    ierr = MatDestroy(&matshellctx->MpTrans);CHKERRQ(ierr);
    ierr = VecDestroy(&matshellctx->ff);CHKERRQ(ierr);
    ierr = VecDestroy(&matshellctx->uu);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&MtM);CHKERRQ(ierr);
    ierr = PetscFree(matshellctx);CHKERRQ(ierr);
  } else {
    ierr = KSPSolveTranspose(ksp, rhs, ff);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  /* Visualize particle field */
  ierr = VecViewFromOptions(ff, NULL, "-weights_view");CHKERRQ(ierr);
  ierr = MatDestroy(&PM_p);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode particlesToGrid(const DM dm, DM sw, const PetscInt Np, const PetscInt a_tid, const PetscInt dim,
                               const PetscReal xx[], const PetscReal yy[], const PetscReal a_wp[], Vec rho, Mat *Mp_out)
{

  PetscBool      removePoints = PETSC_TRUE;
  PetscReal      *wq, *coords;
  PetscDataType  dtype;
  Mat            M_p;
  Vec            ff;
  PetscErrorCode ierr;
  PetscInt       bs,p,zero=0;

  PetscFunctionBeginUser;
  ierr = DMSwarmSetLocalSizes(sw, Np, zero);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  for (p=0;p<Np;p++) {
    coords[p*2+0]  = xx[p];
    coords[p*2+1]  = yy[p];
    wq[p]          = a_wp[p];
    //if (coords[p*2+1]==0.5 || coords[p*2+0]==.25) ierr = PetscInfo(dm,"%D/%D) x = %14.7e, y = %14.7e, w = %14.7e\n", p, Np, coords[p*2+0], coords[p*2+1], a_wp[p]);
  }
  ierr = DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wq);CHKERRQ(ierr);
  ierr = DMSwarmMigrate(sw, removePoints);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)sw, "Particle Grid");CHKERRQ(ierr);

  /* This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE */
  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)rho, "rho");CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)ff, "weights");CHKERRQ(ierr);
  ierr = MatMultTranspose(M_p, ff, rho);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &ff);CHKERRQ(ierr);

  // output
  *Mp_out = M_p;

  PetscFunctionReturn(0);
}
static void maxwellian(PetscInt dim, const PetscReal x[], PetscReal kt_m, PetscReal n, PetscScalar *u)
{
  PetscInt      i;
  PetscReal     v2 = 0, theta = 2.0*kt_m; /* theta = 2kT/mc^2 */

  /* compute the exponents, v^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  /* evaluate the Maxwellian */
  u[0] = n*PetscPowReal(PETSC_PI*theta,-1.5)*(PetscExpReal(-v2/theta));
}

#define MAX_NUM_THRDS 12
PetscErrorCode go(TS ts, Vec X, const PetscInt NUserV, const PetscInt a_Np, const PetscInt dim2, const PetscInt b_target, const PetscInt g_target)
{
  DM              pack, *globSwarmArray, grid_dm[LANDAU_MAX_GRIDS];
  Mat             *globMpArray, g_Mass[LANDAU_MAX_GRIDS];
  KSP             t_ksp[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
  Vec             t_fhat[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
  PetscInt        nDMs, glb_b_id;
  PetscErrorCode  ierr;
#if defined(PETSC_HAVE_OPENMP) && defined(PETSC_HAVE_THREADSAFETY)
  PetscInt        numthreads = PetscNumOMPThreads;
#else
  PetscInt        numthreads = 1;
#endif
  LandauCtx      *ctx;
  Vec            *globXArray;
  PetscReal       moments_0[3], moments_1[3];

  PetscFunctionBeginUser;
  PetscCheck(numthreads<=MAX_NUM_THRDS,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Too many threads %" PetscInt_FMT " > %" PetscInt_FMT "", numthreads, MAX_NUM_THRDS);
  PetscCheck(numthreads>0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number threads %" PetscInt_FMT " > %" PetscInt_FMT " ", numthreads,  MAX_NUM_THRDS);
  ierr = TSGetDM(ts,&pack);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(pack, &ctx);CHKERRQ(ierr);
  PetscCheck(ctx->batch_sz%numthreads==0,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "batch size (-dm_landau_batch_size) %" PetscInt_FMT "  mod #threads %" PetscInt_FMT " must equal zero", ctx->batch_sz, numthreads);
  ierr = DMCompositeGetNumberDM(pack,&nDMs);CHKERRQ(ierr);
  ierr = PetscInfo(pack,"Have %" PetscInt_FMT " total grids, with %" PetscInt_FMT " Landau local batched and %" PetscInt_FMT " global items (vertices)\n",ctx->num_grids,ctx->batch_sz,NUserV);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(*globXArray)*nDMs, &globXArray);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(*globMpArray)*nDMs, &globMpArray);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(*globSwarmArray)*nDMs, &globSwarmArray);CHKERRQ(ierr);
  ierr = DMViewFromOptions(ctx->plex[g_target],NULL,"-ex29_dm_view");CHKERRQ(ierr);
  // create mass matrices
  ierr = DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr); // just to duplicate
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
    Vec  subX = globXArray[LAND_PACK_IDX(0,grid)];
    DM   dm = ctx->plex[grid];
    PetscSection s;
    grid_dm[grid] = dm;
    ierr = DMCreateMassMatrix(dm,dm, &g_Mass[grid]);CHKERRQ(ierr);
    //
    ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
    ierr = DMPlexCreateClosureIndex(dm, s);CHKERRQ(ierr);
    for (int tid=0; tid<numthreads; tid++) {
      ierr = VecDuplicate(subX,&t_fhat[grid][tid]);CHKERRQ(ierr);
      ierr = KSPCreate(PETSC_COMM_SELF, &t_ksp[grid][tid]);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(t_ksp[grid][tid], "ptof_");CHKERRQ(ierr);
      ierr = KSPSetOperators(t_ksp[grid][tid], g_Mass[grid], g_Mass[grid]);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(t_ksp[grid][tid]);CHKERRQ(ierr);
    }
  }
  ierr = DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
  // create particle raw data. could use OMP with a thread safe malloc, but this is just the fake user
  for (int i=0;i<3;i++) moments_0[i] = moments_1[i] = 0;
  for (PetscInt global_batch_id=0 ; global_batch_id < NUserV ; global_batch_id += ctx->batch_sz) {
    ierr = DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
    if (b_target >= global_batch_id && b_target < global_batch_id+ctx->batch_sz) {
      ierr = PetscObjectSetName((PetscObject)globXArray[LAND_PACK_IDX(b_target%ctx->batch_sz,g_target)], "rho");CHKERRQ(ierr);
      //ierr = VecViewFromOptions(globXArray[LAND_PACK_IDX(b_target%ctx->batch_sz,g_target)],NULL,"-ex29_vec_view");CHKERRQ(ierr);
    }
    ierr = VecZeroEntries(X);CHKERRQ(ierr);
    // create fake particles
    for (PetscInt b_id_0 = 0 ; b_id_0 < ctx->batch_sz ; b_id_0 += numthreads) {
      PetscReal *xx_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS], *yy_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS], *wp_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
      PetscInt  Np_t[LANDAU_MAX_GRIDS][MAX_NUM_THRDS];
      // make particles
      for (int tid=0; tid<numthreads; tid++) {
        const PetscInt b_id = b_id_0 + tid;
        if ((glb_b_id = global_batch_id + b_id) < NUserV) { // the ragged edge of the last batch
          PetscInt Npp0 = a_Np + (glb_b_id%a_Np), NN; // fake user: number of particels in each dimension with add some load imbalance and diff (<2x)
          for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
            const PetscReal kT_m = ctx->k*ctx->thermal_temps[ctx->species_offset[grid]]/ctx->masses[ctx->species_offset[grid]]/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 per species -- TODO */;
            PetscReal       lo[3] = {-ctx->radius[grid],-ctx->radius[grid],-ctx->radius[grid]}, hi[3] = {ctx->radius[grid],ctx->radius[grid],ctx->radius[grid]}, hp[3], vole; // would be nice to get box from DM
            const PetscInt  Npi=Npp0,Npj=2*Npp0;
            if (dim2==2) lo[0] = 0; // Landau coordinate (r,z)
            // User: use glb_b_id to index into your data
            NN = Npi*Npj; // make a regular grid of particles Npp x Npp
            Np_t[grid][tid] = NN;
            ierr = PetscMalloc3(NN,&xx_t[grid][tid],NN,&yy_t[grid][tid],NN,&wp_t[grid][tid]);CHKERRQ(ierr);
            hp[0] = (hi[0] - lo[0])/Npi;
            hp[1] = (hi[1] - lo[1])/Npj;
            //ierr = PetscInfo(pack," lo = %14.7e, hi = %14.7e; hp = %14.7e, %14.7e; kT_m = %g\n",lo[1],hi[1], hp[0], hp[1],kT_m);CHKERRQ(ierr); // temp
            vole = hp[0]*hp[1];
            ierr = PetscInfo(pack,"*** Particle data: batch item ('glob' index) %" PetscInt_FMT ", grid %" PetscInt_FMT " with %" PetscInt_FMT " particles in each dimension\n",glb_b_id,grid,Npp0);CHKERRQ(ierr);
            for (int pj=0, pp=0 ; pj < Npj ; pj++) {
              for (int pi=0 ; pi < Npi ; pi++, pp++) {
                xx_t[grid][tid][pp] = lo[0] + hp[0]/2.0 + pi*hp[0];
                yy_t[grid][tid][pp] = lo[1] + hp[1]/2.0 + pj*hp[1];
                {
                  PetscReal x[] = {xx_t[grid][tid][pp], yy_t[grid][tid][pp]};
                  maxwellian(2, x, kT_m, vole, &wp_t[grid][tid][pp]);
                  //wp_t[grid][tid][pp] = vole;
                  //ierr = PetscInfo(pack,"%D) x = %14.7e, %14.7e, n = %14.7e, w = %14.7e\n", pp, x[0], x[1], vole, vole, wp_t[grid][tid][pp]);CHKERRQ(ierr); // temp
                  if (glb_b_id==b_target) {
                    moments_0[0] += 2.0*PETSC_PI*x[0]*wp_t[grid][tid][pp]*ctx->n_0                  *ctx->charges[ctx->species_offset[grid]];
                    moments_0[1] += 2.0*PETSC_PI*x[0]*wp_t[grid][tid][pp]*ctx->n_0*ctx->v_0         *ctx->masses[ctx->species_offset[grid]] * x[1]; // z-momentum
                    moments_0[2] +=     PETSC_PI*x[0]*wp_t[grid][tid][pp]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ctx->species_offset[grid]] * (PetscSqr(x[0]) + PetscSqr(x[1]));
                  }
                }
              }
            }
          } // grid
        } // active
      } // fake threads
      /* Create particle swarm */
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) { // the ragged edge of the last batch
            //ierr = PetscInfo(pack,"Create swarms for 'glob' index %" PetscInt_FMT " create swarm\n",glb_b_id);CHKERRQ(ierr);
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode  ierr_t;
              PetscSection    section;
              PetscInt        Nf;
              DM              dm = grid_dm[grid];
              ierr_t = DMGetLocalSection(dm, &section);
              ierr_t = PetscSectionGetNumFields(section, &Nf);
              if (Nf != 1) ierr_t = 9999;
              else {
                ierr_t = DMViewFromOptions(dm,NULL,"-dm_view");
                ierr_t = PetscInfo(pack,"call createSwarm [%" PetscInt_FMT ".%" PetscInt_FMT "] local batch index %" PetscInt_FMT "\n",b_id,grid,LAND_PACK_IDX(b_id,grid));
                ierr_t = createSwarm(dm, &globSwarmArray[LAND_PACK_IDX(b_id,grid)]);
              }
              if (ierr_t) ierr = ierr_t;
            }
          } // active
        }
      PetscCheckFalse(ierr == 9999, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Only support one species per grid");
      CHKERRQ(ierr);
      // p --> g: make globMpArray & set X
      ierr = PetscInfo(pack,"particlesToGrid for batch %" PetscInt_FMT " to %" PetscInt_FMT "\n",global_batch_id,global_batch_id+ctx->batch_sz);CHKERRQ(ierr);
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) {
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode ierr_t;
              DM             dm = grid_dm[grid];
              DM             sw = globSwarmArray[LAND_PACK_IDX(b_id,grid)];
              Vec            subX = globXArray[LAND_PACK_IDX(b_id,grid)];
              PetscInfo(pack,"particlesToGrid %" PetscInt_FMT ".%" PetscInt_FMT ") particlesToGrid for local batch %" PetscInt_FMT "\n",b_id,grid,LAND_PACK_IDX(b_id,grid));
              ierr_t = particlesToGrid(dm, sw, Np_t[grid][tid], tid, dim2, xx_t[grid][tid], yy_t[grid][tid], wp_t[grid][tid], subX, &globMpArray[LAND_PACK_IDX(b_id,grid)]);
              if (ierr_t) ierr = ierr_t;
            }
          }
        }
      CHKERRQ(ierr);
      ierr = PetscInfo(pack,"Mass solve batch %" PetscInt_FMT " to %" PetscInt_FMT "\n",global_batch_id,global_batch_id+ctx->batch_sz);CHKERRQ(ierr); // can merge
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) {
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode ierr_t;
              Vec            subX = globXArray[LAND_PACK_IDX(b_id,grid)], work = t_fhat[grid][tid];
              KSP            ksp = t_ksp[grid][tid];
              // u = M^_1 f_w
              ierr_t = VecCopy(subX, work);
              ierr_t = KSPSolve(ksp, work, subX);
              if (ierr_t) ierr = ierr_t;
            }
          }
        }
      CHKERRQ(ierr);
      /* Cleanup */
      for (int tid=0; tid<numthreads; tid++) {
        const PetscInt b_id = b_id_0 + tid;
        if ((glb_b_id = global_batch_id + b_id) < NUserV) {
          ierr = PetscInfo(pack,"Free for global batch %" PetscInt_FMT "\n",glb_b_id);CHKERRQ(ierr);
          for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
            ierr = PetscFree3(xx_t[grid][tid],yy_t[grid][tid],wp_t[grid][tid]);CHKERRQ(ierr);
          }
        } // active
      }
    } // Landau
    if (b_target >= global_batch_id && b_target < global_batch_id+ctx->batch_sz) {
      ierr = VecViewFromOptions(globXArray[LAND_PACK_IDX(b_target%ctx->batch_sz,g_target)],NULL,"-ex29_vec_view");CHKERRQ(ierr);
    }
    ierr = DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
    ierr = DMPlexLandauPrintNorms(X,0);CHKERRQ(ierr);
    // advance
    ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
    ierr = TSSolve(ts,X);CHKERRQ(ierr);
    ierr = DMPlexLandauPrintNorms(X,1);CHKERRQ(ierr);
    ierr = DMCompositeGetAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
    if (b_target >= global_batch_id && b_target < global_batch_id+ctx->batch_sz) {
      //ierr = VecViewFromOptions(globXArray[LAND_PACK_IDX(b_target%ctx->batch_sz,g_target)],NULL,"-ex29_vec_view");CHKERRQ(ierr);
    }
    // map back to particles
    for (PetscInt b_id_0 = 0 ; b_id_0 < ctx->batch_sz ; b_id_0 += numthreads) {
      ierr = PetscInfo(pack,"g2p: global batch %" PetscInt_FMT " of %" PetscInt_FMT ", Landau batch %" PetscInt_FMT " of %" PetscInt_FMT ": map back to particles\n",global_batch_id+1,NUserV,b_id_0+1,ctx->batch_sz);CHKERRQ(ierr);
      PetscPragmaOMP(parallel for)
        for (int tid=0; tid<numthreads; tid++) {
          const PetscInt b_id = b_id_0 + tid;
          if ((glb_b_id = global_batch_id + b_id) < NUserV) {
            for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
              PetscErrorCode  ierr_t;
              PetscInfo(pack,"gridToParticles: global batch %" PetscInt_FMT ", local batch b=%" PetscInt_FMT ", grid g=%" PetscInt_FMT ", index(b,g) %" PetscInt_FMT "\n",global_batch_id,b_id,grid,LAND_PACK_IDX(b_id,grid));
              ierr_t = gridToParticles(grid_dm[grid], globSwarmArray[LAND_PACK_IDX(b_id,grid)], globXArray[LAND_PACK_IDX(b_id,grid)], t_fhat[grid][tid], globMpArray[LAND_PACK_IDX(b_id,grid)], g_Mass[grid]);
              if (ierr_t) ierr = ierr_t;
            }
          }
        }
      CHKERRQ(ierr);
      /* Cleanup, and get data */
      ierr = PetscInfo(pack,"Cleanup batches %" PetscInt_FMT " to %" PetscInt_FMT "\n",b_id_0,b_id_0+numthreads);CHKERRQ(ierr);
      for (int tid=0; tid<numthreads; tid++) {
        const PetscInt b_id = b_id_0 + tid;
        if ((glb_b_id = global_batch_id + b_id) < NUserV) {
          for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) {
            PetscDataType dtype;
            PetscReal     *wp,*coords;
            DM            sw = globSwarmArray[LAND_PACK_IDX(b_id,grid)];
            PetscInt      npoints,bs=1;
            ierr = DMSwarmGetField(sw, "w_q", &bs, &dtype, (void**)&wp);CHKERRQ(ierr); // take data out here
            if (glb_b_id==b_target) {
              ierr = DMSwarmGetField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
              ierr = DMSwarmGetLocalSize(sw,&npoints);CHKERRQ(ierr);
              for (int p=0;p<npoints;p++) {
                moments_1[0] += 2.0*PETSC_PI*coords[p*2+0]*wp[p]*ctx->n_0                  *ctx->charges[ctx->species_offset[grid]];
                moments_1[1] += 2.0*PETSC_PI*coords[p*2+0]*wp[p]*ctx->n_0*ctx->v_0         *ctx->masses[ctx->species_offset[grid]] * coords[p*2+1]; // z-momentum
                moments_1[2] +=     PETSC_PI*coords[p*2+0]*wp[p]*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ctx->species_offset[grid]] * (PetscSqr(coords[p*2+0]) + PetscSqr(coords[p*2+1]));
              }
              ierr = DMSwarmRestoreField(sw, "DMSwarmPIC_coor", &bs, &dtype, (void**)&coords);CHKERRQ(ierr);
            }
            ierr = DMSwarmRestoreField(sw, "w_q", &bs, &dtype, (void**)&wp);CHKERRQ(ierr);
            ierr = DMDestroy(&globSwarmArray[LAND_PACK_IDX(b_id,grid)]);CHKERRQ(ierr);
            ierr = MatDestroy(&globMpArray[LAND_PACK_IDX(b_id,grid)]);CHKERRQ(ierr);
          }
        }
      }
    } // thread batch
    ierr = DMCompositeRestoreAccessArray(pack, X, nDMs, NULL, globXArray);CHKERRQ(ierr);
  } // user batch
  /* Cleanup */
  ierr = PetscFree(globXArray);CHKERRQ(ierr);
  ierr = PetscFree(globSwarmArray);CHKERRQ(ierr);
  ierr = PetscFree(globMpArray);CHKERRQ(ierr);
  // clean up mass matrices
  for (PetscInt grid=0 ; grid<ctx->num_grids ; grid++) { // add same particels for all grids
    ierr = MatDestroy(&g_Mass[grid]);
    for (int tid=0; tid<numthreads; tid++) {
      ierr = VecDestroy(&t_fhat[grid][tid]);
      ierr = KSPDestroy(&t_ksp[grid][tid]);
    }
  }
  ierr = PetscInfo(pack,"Total number density: %20.12e (%20.12e); x-momentum = %20.12e (%20.12e); energy = %20.12e (%20.12e) error = %e, %D particles. Use %D threads\n", moments_1[0], moments_0[0], moments_1[1], moments_0[1], moments_1[2],  moments_0[2], (moments_1[2]-moments_0[2])/moments_0[2],a_Np,numthreads);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             pack;
  Vec            X;
  PetscErrorCode ierr;
  PetscInt       dim=2,nvert=1,Np=10,btarget=0,gtarget=0;
  TS             ts;
  Mat            J;
  LandauCtx      *ctx;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Create a mesh */
  ierr = DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &pack);CHKERRQ(ierr);
  ierr = DMSetUp(pack);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(pack, 0, 0.0);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(pack, &ctx);CHKERRQ(ierr);
  // process args
  ierr = PetscOptionsBegin(PETSC_COMM_SELF, "", "Collision Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-number_spatial_vertices", "Number of user spatial vertices to be batched for Landau", "ex29.c", nvert, &nvert, NULL);CHKERRQ(ierr);
  PetscCheck(nvert >= ctx->batch_sz, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %" PetscInt_FMT "should be <= batch size %" PetscInt_FMT,nvert,ctx->batch_sz);
  ierr = PetscOptionsInt("-number_particles_per_dimension", "Number of particles per grid, with slight modification per spatial vertex, in each dimension of base Cartesian grid", "ex29.c", Np, &Np, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-view_vertex_target", "Batch to view with diagnostics", "ex29.c", btarget, &btarget, NULL);CHKERRQ(ierr);
  PetscCheck(btarget < nvert, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Batch to view %" PetscInt_FMT " should be < number of vertices %" PetscInt_FMT,btarget,nvert);
  ierr = PetscOptionsInt("-view_grid_target", "Grid to view with diagnostics", "ex29.c", gtarget, &gtarget, NULL);CHKERRQ(ierr);
  PetscCheck(gtarget < ctx->num_grids, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Grid to view %" PetscInt_FMT " should be < number of grids %" PetscInt_FMT,gtarget,ctx->num_grids);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,pack);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,DMPlexLandauIFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,DMPlexLandauIJacobian,NULL);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X, "X");CHKERRQ(ierr);
  // do particle advance
  ierr = go(ts,X,nvert,Np,dim,btarget,gtarget);CHKERRQ(ierr);
  /* clean up */
  ierr = DMPlexLandauDestroyVelocitySpace(&pack);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: !complex hdf5

  testset:
    requires: double
    output_file: output/ex29_0.out
    args: -dm_landau_amr_levels_max 1 \
          -dm_landau_amr_post_refine 0 \
          -dm_landau_batch_size 1 \
          -dm_landau_device_type cpu \
          -dm_landau_n 1 \
          -dm_landau_thermal_temps 1 \
          -dm_preallocate_only false \
          -ex29_dm_view hdf5:local3.h5 \
          -ex29_vec_view hdf5:local3.h5::append \
          -ftop_ksp_converged_reason \
          -ftop_ksp_rtol 1e-14\
          -ftop_ksp_type lsqr \
          -ftop_pc_type bjacobi \
          -ftop_sub_pc_factor_shift_type nonzero \
          -ftop_sub_pc_type lu \
          -ksp_type preonly \
          -number_particles_per_dimension 10 \
          -pc_type lu \
          -petscspace_degree 3 \
          -ptof_ksp_converged_reason \
          -ptof_ksp_rtol 1e-14\
          -snes_converged_reason \
          -snes_monitor \
          -snes_rtol 1e-14\
          -snes_stol 1e-14\
          -ts_dt 1 \
          -ts_exact_final_time stepover \
          -ts_max_snes_failures -1 \
          -ts_max_steps 1 \
          -ts_monitor \
          -ts_type beuler

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
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda

TEST*/
