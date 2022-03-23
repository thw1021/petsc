static char help[] = "Solves the incompressible, variable-viscosity Stokes equation in 2D or 3D, driven by buoyancy variations.\n"
                     "-dim: set dimension (2 or 3)\n"
                     "-nondimensional: replace dimensional domain and coefficients with nondimensional ones\n"
                     "-isoviscous: use constant viscosity\n"
                     "-levels: number of grids to create, by coarsening\n"
                     "-rediscretize: create operators for all grids and set up a FieldSplit/MG solver\n\n";

#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>
#include <petscdmda.h>


/* Application context - grid-based data*/
typedef struct LevelCtxData_ {
  DM          dm_stokes,dm_coefficients,dm_faces;
  Vec         coeff;
  PetscInt    cells_x,cells_y,cells_z; /* redundant with DMs */
  PetscReal   hx_characteristic,hy_characteristic,hz_characteristic;
  PetscScalar K_bound,K_cont;
} LevelCtxData;
typedef LevelCtxData* LevelCtx;

/* Application context - problem and grid(s) (but not solver-specific data) */
typedef struct CtxData_ {
  MPI_Comm    comm;
  PetscInt    dim;                       /* redundant with DMs */
  PetscInt    cells_x, cells_y, cells_z; /* Redundant with finest DMs */
  PetscReal   xmax,ymax,xmin,ymin,zmin,zmax;
  PetscScalar eta1,eta2,rho1,rho2,gy,eta_characteristic;
  PetscBool   pin_pressure;
  PetscScalar (*GetEta)(struct CtxData_*,PetscScalar,PetscScalar,PetscScalar);
  PetscScalar (*GetRho)(struct CtxData_*,PetscScalar,PetscScalar,PetscScalar);
  PetscInt    n_levels;
  LevelCtx    *levels;
} CtxData;
typedef CtxData* Ctx;

/* Helper to pass system-creation parameters */
typedef struct SystemParameters_ {
  Ctx       ctx;
  PetscInt  level;
  PetscBool include_inverse_visc, faces_only;
} SystemParametersData;
typedef SystemParametersData* SystemParameters;


/* Main logic */
static PetscErrorCode AttachNullspace(DM,Mat);
static PetscErrorCode CreateAuxiliaryOperator(Ctx,PetscInt,Mat*);
static PetscErrorCode CreateSystem(SystemParameters,Mat*,Vec*);
static PetscErrorCode CtxCreateAndSetFromOptions(Ctx*);
static PetscErrorCode CtxDestroy(Ctx*);
static PetscErrorCode DumpOperator(Mat,const char*);
static PetscErrorCode DumpSolution(Ctx,PetscInt,Vec);
static PetscErrorCode OperatorInsertInverseViscosityPressureTerms(DM,DM,Vec,PetscScalar,Mat);
static PetscErrorCode PopulateCoefficientData(Ctx,PetscInt);
static PetscErrorCode SystemParametersCreate(SystemParameters*,Ctx);
static PetscErrorCode SystemParametersDestroy(SystemParameters*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Ctx            ctx;
  Mat            A,*A_faces = NULL,S_hat = NULL, P = NULL;
  Vec            x,b;
  KSP            ksp;
  DM             dm_stokes,dm_coefficients;
  PetscBool      dump_solution, dump_operators, build_auxiliary_operator, rediscretize, custom_pc_mat;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Accept options for program behavior */
  dump_solution = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump_solution",&dump_solution,NULL);CHKERRQ(ierr);
  dump_operators = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump_operators",&dump_operators,NULL);CHKERRQ(ierr);
  rediscretize = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-rediscretize",&rediscretize,NULL);CHKERRQ(ierr);
  build_auxiliary_operator = rediscretize;
  ierr = PetscOptionsGetBool(NULL,NULL,"-build_auxiliary_operator",&build_auxiliary_operator,NULL);CHKERRQ(ierr);
  custom_pc_mat = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-custom_pc_mat",&custom_pc_mat,NULL);CHKERRQ(ierr);


  /* Populate application context */
  ierr = CtxCreateAndSetFromOptions(&ctx);CHKERRQ(ierr);

  /* Create two DMStag objects, corresponding to the same domain and parallel
     decomposition ("topology"). Each defines a different set of fields on
     the domain ("section"); the first the solution to the Stokes equations
     (x- and y-velocities and scalar pressure), and the second holds coefficients
     (viscosities on elements and viscosities+densities on corners/edges in 2d/3d) */
  if (ctx->dim == 2) {
  ierr = DMStagCreate2d(
      ctx->comm,
      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
      ctx->cells_x,ctx->cells_y,         /* Global element counts */
      PETSC_DECIDE,PETSC_DECIDE,         /* Determine parallel decomposition automatically */
      0,1,1,                             /* dof: 0 per vertex, 1 per edge, 1 per face/element */
      DMSTAG_STENCIL_BOX,
      1,                                 /* elementwise stencil width */
      NULL,NULL,
      &ctx->levels[ctx->n_levels-1]->dm_stokes);CHKERRQ(ierr);
  } else if (ctx->dim == 3) {
    ierr = DMStagCreate3d(
        ctx->comm,
        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
        ctx->cells_x,ctx->cells_y,ctx->cells_z,
        PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
        0,0,1,1,                        /* dof: 1 per face, 1 per element */
        DMSTAG_STENCIL_BOX,
        1,
        NULL,NULL,NULL,
        &ctx->levels[ctx->n_levels-1]->dm_stokes);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension: %d",ctx->dim);
  dm_stokes = ctx->levels[ctx->n_levels-1]->dm_stokes;
  ierr = DMSetFromOptions(dm_stokes);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm_stokes,&ctx->cells_x,&ctx->cells_y, &ctx->cells_z);CHKERRQ(ierr);
  ierr = DMSetUp(dm_stokes);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm_stokes,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax);CHKERRQ(ierr);

  if (ctx->dim == 2) {
    ierr = DMStagCreateCompatibleDMStag(dm_stokes,2,0,1,0,&ctx->levels[ctx->n_levels-1]->dm_coefficients);CHKERRQ(ierr);
  } else if (ctx->dim == 3) {
    ierr = DMStagCreateCompatibleDMStag(dm_stokes,0,2,0,1,&ctx->levels[ctx->n_levels-1]->dm_coefficients);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension: %d",ctx->dim);
  dm_coefficients = ctx->levels[ctx->n_levels-1]->dm_coefficients;
  ierr = DMSetUp(dm_coefficients);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm_coefficients,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax);CHKERRQ(ierr);

  /* Create additional DMs by coarsening. 0 is the coarsest level */
  for (PetscInt level=ctx->n_levels-1; level>0; --level) {
    ierr = DMCoarsen(ctx->levels[level]->dm_stokes,      MPI_COMM_NULL,&ctx->levels[level-1]->dm_stokes      );CHKERRQ(ierr);
    ierr = DMCoarsen(ctx->levels[level]->dm_coefficients,MPI_COMM_NULL,&ctx->levels[level-1]->dm_coefficients);CHKERRQ(ierr);
  }

  /* Compute scaling constants, knowing grid spacing */
  ctx->eta_characteristic = PetscMin(PetscRealPart(ctx->eta1),PetscRealPart(ctx->eta2));
  for (PetscInt level=0; level<ctx->n_levels; ++level) {
    PetscInt  N[3];
    PetscReal hx_avg_inv;

    ierr = DMStagGetGlobalSizes(ctx->levels[level]->dm_stokes,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
    ctx->levels[level]->hx_characteristic = (ctx->xmax-ctx->xmin)/N[0];
    ctx->levels[level]->hy_characteristic = (ctx->ymax-ctx->ymin)/N[1];
    ctx->levels[level]->hz_characteristic = (ctx->zmax-ctx->zmin)/N[2];
    if (ctx->dim == 2) {
      hx_avg_inv = 2.0/(ctx->levels[level]->hx_characteristic + ctx->levels[level]->hy_characteristic);
    } else if (ctx->dim == 3) {
      hx_avg_inv = 3.0/(ctx->levels[level]->hx_characteristic + ctx->levels[level]->hy_characteristic + ctx->levels[level]->hz_characteristic);
    } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %d",ctx->dim);
    //ctx->levels[level]->K_cont = ctx->eta_characteristic*hx_avg_inv;
    ctx->levels[level]->K_cont = 1.0; // !!
    //ctx->levels[level]->K_bound = ctx->eta_characteristic*hx_avg_inv*hx_avg_inv;
    ctx->levels[level]->K_bound = 1.0; // !!
  }

  /* Populate coefficient data */
  for (PetscInt level=0; level<ctx->n_levels; ++level) {
    ierr = PopulateCoefficientData(ctx,level);CHKERRQ(ierr);
  }

  /* Construct main system */
  {
    SystemParameters system_parameters;

    ierr = SystemParametersCreate(&system_parameters,ctx);CHKERRQ(ierr);
    ierr = CreateSystem(system_parameters,&A,&b);CHKERRQ(ierr);
    ierr = SystemParametersDestroy(&system_parameters);CHKERRQ(ierr);
  }

  /* Attach a constant-pressure nullspace to the fine-level operator */
  if (!ctx->pin_pressure) {
    ierr = AttachNullspace(dm_stokes,A);CHKERRQ(ierr);
  }

  /* Set up solver */
  ierr = KSPCreate(ctx->comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPFGMRES);CHKERRQ(ierr);
  {
    /* Default to a direct solver, if a package is available */
    PetscMPIInt size;

    ierr = MPI_Comm_size(ctx->comm,&size);CHKERRQ(ierr);
    if (PetscDefined(HAVE_SUITESPARSE) && size == 1) {
      PC pc;

      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERUMFPACK);CHKERRQ(ierr); /* A default, requires SuiteSparse */
    }
    if (PetscDefined(HAVE_MUMPS) && size > 1) {
      PC pc;

      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);CHKERRQ(ierr); /* A default, requires MUMPS */
    }
  }

  ierr = PetscOptionsSetValue(NULL,"-ksp_converged_reason","");CHKERRQ(ierr);

  /* Create and set a custom preconditioning matrix */
  if (custom_pc_mat) {
    {
      SystemParameters system_parameters;

      ierr = SystemParametersCreate(&system_parameters,ctx);CHKERRQ(ierr);
      system_parameters->include_inverse_visc = PETSC_TRUE;
      ierr = CreateSystem(system_parameters,&P,NULL);CHKERRQ(ierr);
      ierr = SystemParametersDestroy(&system_parameters);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ksp,A,P);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  }

  ierr = KSPSetDM(ksp,dm_stokes);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_FALSE);CHKERRQ(ierr);

  /* Finish setting up solver (can override options set above) */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Additional solver configuration that can involve setting up and CANNOT be
     overridden from the command line */

  /* Construct an auxiliary operator for use a Schur complement preconditioner,
     and tell PCFieldSplit to use it (which has no effect if not using that PC) */
  if (build_auxiliary_operator && !rediscretize) {
    PC pc;

    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = CreateAuxiliaryOperator(ctx,ctx->n_levels-1,&S_hat);CHKERRQ(ierr);
    ierr = PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_USER,S_hat);CHKERRQ(ierr);
  }

  if (rediscretize) {
    /* Set up an ABF solver with rediscretized geometric multigrid on the velocity-velocity block */
    PC  pc,pc_faces;
    KSP ksp_faces;

    if (ctx->n_levels < 2) {
      ierr = PetscPrintf(ctx->comm,"Warning: not using multiple levels!\n");CHKERRQ(ierr);
    }

    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR);CHKERRQ(ierr);
    ierr = PCFieldSplitSetSchurFactType(pc,PC_FIELDSPLIT_SCHUR_FACT_UPPER);CHKERRQ(ierr);
    if (build_auxiliary_operator) {
      ierr = CreateAuxiliaryOperator(ctx,ctx->n_levels-1,&S_hat);CHKERRQ(ierr);
      ierr = PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_USER,S_hat);CHKERRQ(ierr);
    }

    /* Create rediscretized velocity-only DMs and operators */
    ierr = PetscCalloc1(ctx->n_levels,&A_faces);CHKERRQ(ierr);
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      if (ctx->dim == 2) {
        ierr = DMStagCreateCompatibleDMStag(ctx->levels[level]->dm_stokes,0,1,0,0,&ctx->levels[level]->dm_faces);CHKERRQ(ierr);
      } else if (ctx->dim == 3) {
        ierr = DMStagCreateCompatibleDMStag(ctx->levels[level]->dm_stokes,0,0,1,0,&ctx->levels[level]->dm_faces);CHKERRQ(ierr);
      } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %D",ctx->dim);
      {
        SystemParameters system_parameters;

        ierr = SystemParametersCreate(&system_parameters,ctx);CHKERRQ(ierr);
        system_parameters->faces_only = PETSC_TRUE;
        system_parameters->level = level;
        ierr = CreateSystem(system_parameters,&A_faces[level],NULL);CHKERRQ(ierr);
        ierr = SystemParametersDestroy(&system_parameters);CHKERRQ(ierr);
      }
    }

    /* Set up to populate enough to define the sub-solver */
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    /* Set multigrid components and settings */
    {
      KSP *sub_ksp;

      ierr = PCFieldSplitSchurGetSubKSP(pc,NULL,&sub_ksp);CHKERRQ(ierr);
      ksp_faces = sub_ksp[0];
      ierr = PetscFree(sub_ksp);CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp_faces,KSPGCR);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_faces,&pc_faces);CHKERRQ(ierr);
    ierr = PCSetType(pc_faces,PCMG);CHKERRQ(ierr);
    ierr = PCMGSetLevels(pc_faces,ctx->n_levels,NULL);CHKERRQ(ierr);
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      KSP ksp_level;
      PC  pc_level;

      /* Smoothers */
      ierr = PCMGGetSmoother(pc_faces,level,&ksp_level);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp_level,&pc_level);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp_level,A_faces[level],A_faces[level]);CHKERRQ(ierr);
      if (level > 0) {
        ierr = PCSetType(pc_level,PCJACOBI);CHKERRQ(ierr);
      }

      /* Transfer Operators */
      if (level > 0) {
        Mat restriction,interpolation;
        DM  dm_level=ctx->levels[level]->dm_faces;
        DM  dm_coarser=ctx->levels[level-1]->dm_faces;

        ierr = DMCreateInterpolation(dm_coarser,dm_level,&interpolation,NULL);CHKERRQ(ierr);
        ierr = PCMGSetInterpolation(pc_faces,level,interpolation);CHKERRQ(ierr);
        ierr = MatDestroy(&interpolation);CHKERRQ(ierr);
        ierr = DMCreateRestriction(dm_coarser,dm_level,&restriction);CHKERRQ(ierr);
        ierr = PCMGSetRestriction(pc_faces,level,restriction);CHKERRQ(ierr);
        ierr = MatDestroy(&restriction);CHKERRQ(ierr);
      }
    }
  }


  /* For diagnostic purposes, dump operators */
  if (dump_operators) {
    ierr = DumpOperator(A,"A");CHKERRQ(ierr);
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      if (A_faces && A_faces[level]) {
        char filename[PETSC_MAX_PATH_LEN];

        ierr = PetscSNPrintf(filename,sizeof(filename),"A_faces_%D",level);CHKERRQ(ierr);
        ierr = DumpOperator(A_faces[level],filename);CHKERRQ(ierr);
      }
    }
    if (build_auxiliary_operator) {
      ierr = DumpOperator(S_hat,"S_hat");CHKERRQ(ierr);
    }
    if (custom_pc_mat) {
      ierr = DumpOperator(P,"P");CHKERRQ(ierr);
    }
  }

  /* Solve */
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  {
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
    if (reason < 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Linear solve failed");
  }

  /* Dump solution by converting to DMDAs and dumping */
  if (dump_solution) {
    ierr = DumpSolution(ctx,ctx->n_levels-1,x);CHKERRQ(ierr);
  }

  /* Destroy PETSc objects and finalize */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(A);CHKERRQ(ierr);
  if (A_faces) {
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      if (A_faces[level]) {
        ierr = MatDestroy(&A_faces[level]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(A_faces);CHKERRQ(ierr);
  }
  if (P) {
    ierr = MatDestroy(&P);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&S_hat);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = CtxDestroy(&ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscScalar GetEta_constant(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z)
{
  (void) ctx;
  (void) x;
  (void) y;
  (void) z;
  return 1.0;
}

static PetscScalar GetRho_layers(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z)
{
  (void) y;
  (void) z;
  return x < (ctx->xmax - ctx->xmin) / 2.0 ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_layers(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z)
{
  (void) y;
  (void) z;
  return x < (ctx->xmax - ctx->xmin) / 2.0 ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetRho_sinker_box2(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  (void) z;
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  return (xx*xx > 0.15*0.15 || yy*yy > 0.15*0.15) ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_sinker_box2(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  (void) z;
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  return (xx*xx > 0.15*0.15 || yy*yy > 0.15*0.15) ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetRho_sinker_box3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  const PetscScalar zz = z/d - 0.5;
  const PetscScalar half_width =  0.15;
  return (PetscAbsScalar(xx) > half_width || PetscAbsScalar(yy) > half_width || PetscAbsScalar(zz) > half_width) ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_sinker_box3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  const PetscScalar zz = z/d - 0.5;
  const PetscScalar half_width = 0.15;
  return (PetscAbsScalar(xx) > half_width || PetscAbsScalar(yy) > half_width || PetscAbsScalar(zz) > half_width) ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetRho_sinker_sphere3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  const PetscScalar zz = z/d - 0.5;
  const PetscScalar half_width =  0.3;
  return (xx*xx + yy*yy + zz*zz > half_width*half_width) ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_sinker_sphere3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  const PetscScalar zz = z/d - 0.5;
  const PetscScalar half_width = 0.3;
  return (xx*xx + yy*yy + zz*zz > half_width*half_width) ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetEta_blob3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  const PetscScalar zz = z/d - 0.5;

  return ctx->eta1 + ctx->eta2 * PetscExpScalar(-20.0 * (xx*xx + yy*yy + zz*zz));
}

static PetscScalar GetRho_blob3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscScalar d = ctx->xmax - ctx->xmin;
  const PetscScalar xx = x/d - 0.5;
  const PetscScalar yy = y/d - 0.5;
  const PetscScalar zz = z/d - 0.5;

  return ctx->rho1 + ctx->rho2 * PetscExpScalar(-20.0 * (xx*xx + yy*yy + zz*zz));
}

static PetscErrorCode LevelCtxCreate(LevelCtx *p_level_ctx)
{
  PetscErrorCode ierr;
  LevelCtx       level_ctx;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(1,p_level_ctx);CHKERRQ(ierr);
  level_ctx = *p_level_ctx;
  level_ctx->dm_stokes = NULL;
  level_ctx->dm_coefficients = NULL;
  level_ctx->dm_faces = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode LevelCtxDestroy(LevelCtx *p_level_ctx)
{
  PetscErrorCode ierr;
  LevelCtx       level_ctx;

  PetscFunctionBeginUser;
  level_ctx = *p_level_ctx;
  if (level_ctx->dm_stokes) {
    ierr = DMDestroy(&level_ctx->dm_stokes);CHKERRQ(ierr);
  }
  if (level_ctx->dm_coefficients) {
    ierr = DMDestroy(&level_ctx->dm_coefficients);CHKERRQ(ierr);
  }
  if (level_ctx->dm_faces) {
    ierr = DMDestroy(&level_ctx->dm_faces);CHKERRQ(ierr);
  }
  if (level_ctx->coeff) {
    ierr = VecDestroy(&level_ctx->coeff);CHKERRQ(ierr);
  }
  ierr = PetscFree(*p_level_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CtxCreateAndSetFromOptions(Ctx *p_ctx)
{
  PetscErrorCode ierr;
  Ctx            ctx;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(1,p_ctx);CHKERRQ(ierr);
  ctx = *p_ctx;

  ctx->comm = PETSC_COMM_WORLD;
  ctx->pin_pressure = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-pin_pressure",&ctx->pin_pressure,NULL);CHKERRQ(ierr);
  ctx->dim = 3;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&ctx->dim,NULL);CHKERRQ(ierr);
  if (ctx->dim <= 2) {
    ctx->cells_x = 32;
  } else {
    ctx->cells_x = 16;
  }
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&ctx->cells_x,NULL);CHKERRQ(ierr);  /* shortcut. Usually, use -stag_grid_x etc. */
  ctx->cells_z = ctx->cells_y = ctx->cells_x;
  ctx->xmin = ctx->ymin = ctx->zmin = 0.0;
  {
    PetscBool nondimensional = PETSC_TRUE;

    ierr = PetscOptionsGetBool(NULL,NULL,"-nondimensional",&nondimensional,NULL);CHKERRQ(ierr);
    if (nondimensional) {
      ctx->xmax = ctx->ymax = ctx->zmax = 1.0;
      ctx->rho1 = 0.0;
      ctx->rho2 = 1.0;
      ctx->eta1 = 1.0;
      ctx->eta2 = 1e2;
      ctx->gy   = -1.0; /* downwards */
    } else {
      ctx->xmax = 1e6;
      ctx->ymax = 1.5e6;
      ctx->zmax = 1e6;
      ctx->rho1 = 3200;
      ctx->rho2 = 3300;
      ctx->eta1 = 1e20;
      ctx->eta2 = 1e22;
      ctx->gy   = -10.0; /* downwards */
    }
  }
  {
    PetscBool isoviscous;

    isoviscous = PETSC_FALSE;
    ierr = PetscOptionsGetReal(NULL,NULL,"-eta1",&ctx->eta1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-isoviscous",&isoviscous,NULL);CHKERRQ(ierr);
    if (isoviscous) {
      ctx->eta2 = ctx->eta1;
      ctx->GetEta = GetEta_constant; /* override */
    } else {
      ierr = PetscOptionsGetReal(NULL,NULL,"-eta2",&ctx->eta2,NULL);CHKERRQ(ierr);
    }
  }
  {
    char      mode[1024] = "sinker";
    PetscBool is_layers,is_blob,is_sinker_box,is_sinker_sphere;

    ierr = PetscOptionsGetString(NULL,NULL,"-coefficients",mode,sizeof(mode),NULL);CHKERRQ(ierr);
    ierr = PetscStrncmp(mode,"layers",sizeof(mode),&is_layers);CHKERRQ(ierr);
    ierr = PetscStrncmp(mode,"sinker",sizeof(mode),&is_sinker_box);CHKERRQ(ierr);
    if (!is_sinker_box) {
      ierr = PetscStrncmp(mode,"sinker_box",sizeof(mode),&is_sinker_box);CHKERRQ(ierr);
    }
    ierr = PetscStrncmp(mode,"sinker_sphere",sizeof(mode),&is_sinker_sphere);CHKERRQ(ierr);
    ierr = PetscStrncmp(mode,"blob",sizeof(mode),&is_blob);CHKERRQ(ierr);

    if (is_layers) {
        ctx->GetRho = GetRho_layers;
        ctx->GetEta = GetEta_layers;
    }
    if (is_blob) {
        if (ctx->dim == 3) {
          ctx->GetRho = GetRho_blob3;
          ctx->GetEta = GetEta_blob3;
        } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %d",ctx->dim);
    }
    if (is_sinker_box) {
        if (ctx->dim == 2) {
          ctx->GetRho = GetRho_sinker_box2;
          ctx->GetEta = GetEta_sinker_box2;
        } else if (ctx->dim == 3) {
          ctx->GetRho = GetRho_sinker_box3;
          ctx->GetEta = GetEta_sinker_box3;
        } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %d",ctx->dim);
      }
    if (is_sinker_sphere) {
        if (ctx->dim == 3) {
          ctx->GetRho = GetRho_sinker_sphere3;
          ctx->GetEta = GetEta_sinker_sphere3;
        } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %d",ctx->dim);
      }
  }

  /* Per-level data */
  ctx->n_levels = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-levels",&ctx->n_levels,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->n_levels,&ctx->levels);CHKERRQ(ierr);
  for (PetscInt i=0; i<ctx->n_levels; ++i) {
    ierr = LevelCtxCreate(&ctx->levels[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CtxDestroy(Ctx *p_ctx)
{
  PetscErrorCode ierr;
  Ctx            ctx;

  PetscFunctionBeginUser;
  ctx = *p_ctx;
  for (PetscInt i=0; i<ctx->n_levels; ++i) {
    ierr = LevelCtxDestroy(&ctx->levels[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->levels);CHKERRQ(ierr);
  ierr = PetscFree(*p_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SystemParametersCreate(SystemParameters* parameters,Ctx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1,parameters);CHKERRQ(ierr);
  (*parameters)->ctx = ctx;
  (*parameters)->level = ctx->n_levels-1;
  PetscFunctionReturn(0);
}

static PetscErrorCode SystemParametersDestroy(SystemParameters* parameters)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(*parameters);CHKERRQ(ierr);
  *parameters = NULL;
  PetscFunctionReturn(0);
}


static PetscErrorCode CreateSystem2d(SystemParameters parameters, Mat *pA,Vec *pRhs)
{
  PetscErrorCode ierr;
  PetscInt       N[2];
  PetscInt       ex,ey,startx,starty,nx,ny;
  Mat            A;
  Vec            rhs;
  PetscReal      hx,hy,dv;
  Vec            coefficients_local;
  PetscBool      build_rhs;
  DM             dm_main, dm_coefficients;
  PetscScalar    K_cont, K_bound;
  Ctx            ctx = parameters->ctx;
  PetscInt       level = parameters->level;

  PetscFunctionBeginUser;
  if (parameters->faces_only) {
    dm_main = ctx->levels[level]->dm_faces;
  } else {
    dm_main = ctx->levels[level]->dm_stokes;
  }
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  K_cont = ctx->levels[level]->K_cont;
  K_bound = ctx->levels[level]->K_bound;
  ierr = DMCreateMatrix(dm_main,pA);CHKERRQ(ierr);
  A = *pA;
  build_rhs = pRhs != NULL;
  if (parameters->faces_only && build_rhs) SETERRQ(PetscObjectComm((PetscObject)dm_main),PETSC_ERR_SUP,"RHS for faces-only not supported");
  if (build_rhs) {
    ierr = DMCreateGlobalVector(dm_main,pRhs);CHKERRQ(ierr);
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  ierr = DMStagGetCorners(dm_main,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm_main,&N[0],&N[1],NULL);CHKERRQ(ierr);
  hx = ctx->levels[level]->hx_characteristic;
  hy = ctx->levels[level]->hy_characteristic;
  dv = hx*hy;
  ierr = DMGetLocalVector(dm_coefficients,&coefficients_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm_coefficients,ctx->levels[level]->coeff,INSERT_VALUES,coefficients_local);CHKERRQ(ierr);

  /* Loop over all local elements */
  for (ey = starty; ey<starty+ny; ++ey) { /* With DMStag, always iterate x fastest, y second fastest, z slowest */
    for (ex = startx; ex<startx+nx; ++ex) {
      const PetscBool left_boundary   = ex == 0;
      const PetscBool right_boundary  = ex == N[0]-1;
      const PetscBool bottom_boundary = ey == 0;
      const PetscBool top_boundary    = ey == N[1]-1;

      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil     row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_UP; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil     row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_DOWN; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y
           includes non-zero forcing and free-slip boundary conditions */
        PetscInt      count;
        DMStagStencil row,col[11];
        PetscScalar   val_A[11];
        DMStagStencil rhoPoint[2];
        PetscScalar   rho[2],val_rhs;
        DMStagStencil etaPoint[4];
        PetscScalar   eta[4],eta_left,eta_right,eta_up,eta_down;

        row.i = ex; row.j = ey; row.loc = DMSTAG_DOWN; row.c = 0;

        /* get rho values  and compute rhs value*/
        rhoPoint[0].i = ex; rhoPoint[0].j = ey; rhoPoint[0].loc = DMSTAG_DOWN_LEFT;  rhoPoint[0].c = 1;
        rhoPoint[1].i = ex; rhoPoint[1].j = ey; rhoPoint[1].loc = DMSTAG_DOWN_RIGHT; rhoPoint[1].c = 1;
        ierr = DMStagVecGetValuesStencil(dm_coefficients,coefficients_local,2,rhoPoint,rho);CHKERRQ(ierr);
        val_rhs = -ctx->gy * dv * 0.5 * (rho[0] + rho[1]);

        /* Get eta values */
        etaPoint[0].i = ex; etaPoint[0].j = ey;   etaPoint[0].loc = DMSTAG_DOWN_LEFT;  etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex; etaPoint[1].j = ey;   etaPoint[1].loc = DMSTAG_DOWN_RIGHT; etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex; etaPoint[2].j = ey;   etaPoint[2].loc = DMSTAG_ELEMENT;    etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex; etaPoint[3].j = ey-1; etaPoint[3].loc = DMSTAG_ELEMENT;    etaPoint[3].c = 0; /* Down  */
        ierr = DMStagVecGetValuesStencil(dm_coefficients,coefficients_local,4,etaPoint,eta);CHKERRQ(ierr);
        eta_left = eta[0]; eta_right = eta[1]; eta_up = eta[2]; eta_down = eta[3];

        count = 0;

        col[count] = row;
        val_A[count] = -2.0 * dv * (eta_down + eta_up) / (hy*hy);
        if (!left_boundary)  val_A[count] += -1.0 * dv * eta_left  / (hx*hx);
        if (!right_boundary) val_A[count] += -1.0 * dv * eta_right / (hx*hx);
        ++count;

        col[count].i   = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_down  / (hy*hy); ++count;
        col[count].i   = ex;   col[count].j = ey+1; col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_up    / (hy*hy); ++count;
        if (!left_boundary) {
          col[count].i = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =        dv * eta_left  / (hx*hx); ++count;
        }
        if (!right_boundary) {
          col[count].i = ex+1; col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =        dv * eta_right / (hx*hx); ++count;
        }
        col[count].i   = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =        dv * eta_left  / (hx*hy); ++count; /* down left x edge */
        col[count].i   = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_RIGHT;   col[count].c  = 0; val_A[count] = -1.0 * dv * eta_right / (hx*hy); ++count; /* down right x edge */
        col[count].i   = ex;   col[count].j = ey;   col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] = -1.0 * dv * eta_left  / (hx*hy); ++count; /* up left x edge */
        col[count].i   = ex;   col[count].j = ey;   col[count].loc = DMSTAG_RIGHT;   col[count].c  = 0; val_A[count] =        dv * eta_right / (hx*hy); ++count; /* up right x edge */
        if (!parameters->faces_only) {
          col[count].i = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] =        K_cont * dv / hy;         ++count;
          col[count].i = ex;   col[count].j = ey;   col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] = -1.0 * K_cont * dv / hy;         ++count;
        }

        ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        /* Redundant in the corner */
        DMStagStencil     row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_RIGHT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_LEFT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x
          zero RHS, including free-slip boundary conditions */
        PetscInt          count;
        DMStagStencil     row,col[11];
        PetscScalar       val_A[11];
        DMStagStencil     etaPoint[4];
        PetscScalar       eta[4],eta_left,eta_right,eta_up,eta_down;
        const PetscScalar val_rhs = 0.0;

        row.i = ex; row.j = ey; row.loc = DMSTAG_LEFT; row.c = 0;

        /* Get eta values */
        etaPoint[0].i = ex-1; etaPoint[0].j = ey; etaPoint[0].loc = DMSTAG_ELEMENT;   etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex;   etaPoint[1].j = ey; etaPoint[1].loc = DMSTAG_ELEMENT;   etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex;   etaPoint[2].j = ey; etaPoint[2].loc = DMSTAG_UP_LEFT;   etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex;   etaPoint[3].j = ey; etaPoint[3].loc = DMSTAG_DOWN_LEFT; etaPoint[3].c = 0; /* Down  */
        ierr = DMStagVecGetValuesStencil(dm_coefficients,coefficients_local,4,etaPoint,eta);CHKERRQ(ierr);
        eta_left = eta[0]; eta_right = eta[1]; eta_up = eta[2]; eta_down = eta[3];

        count = 0;
        col[count] = row;
        val_A[count] = -2.0 * dv * (eta_left + eta_right) / (hx*hx);
        if (!bottom_boundary) val_A[count] += -1.0 * dv * eta_down / (hy*hy);
        if (!top_boundary)    val_A[count] += -1.0 * dv * eta_up   / (hy*hy);
        ++count;

        if (!bottom_boundary) {
          col[count].i  = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =        dv * eta_down  / (hy*hy); ++count;
        }
        if (!top_boundary) {
          col[count].i  = ex;   col[count].j = ey+1; col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =        dv * eta_up    / (hy*hy); ++count;
        }
        col[count].i    = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_left  / (hx*hx); ++count;
        col[count].i    = ex+1; col[count].j = ey;   col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_right / (hx*hx); ++count;
        col[count].i    = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =        dv * eta_down  / (hx*hy); ++count; /* down left */
        col[count].i    = ex;   col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] = -1.0 * dv * eta_down  / (hx*hy); ++count; /* down right */
        col[count].i    = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_UP;      col[count].c  = 0; val_A[count] = -1.0 * dv * eta_up    / (hx*hy); ++count; /* up left */
        col[count].i    = ex;   col[count].j = ey;   col[count].loc = DMSTAG_UP;      col[count].c  = 0; val_A[count] =        dv * eta_up    / (hx*hy); ++count; /* up right */
        if (!parameters->faces_only) {
          col[count].i  = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] =        K_cont * dv / hx;         ++count;
          col[count].i  = ex;   col[count].j = ey;   col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] = -1.0 * K_cont * dv / hx;         ++count;
        }

        ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
        if (build_rhs) {
          ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      /* P equation : u_x + v_y = 0

         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace)

        Note: the scaling by dv is not chosen in a principled way and is likely sub-optimal
       */
      if (!parameters->faces_only) {
        if (ctx->pin_pressure && ex == 0 && ey == 0) { /* Pin the first pressure node to zero, if requested */
          DMStagStencil     row;
          const PetscScalar val_A = K_bound;
          const PetscScalar val_rhs = 0.0;

          row.i = ex; row.j = ey; row.loc = DMSTAG_ELEMENT; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
          if (build_rhs) {
            ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
          }
        } else {
          DMStagStencil     row,col[5];
          PetscScalar       val_A[5];
          const PetscScalar val_rhs = 0.0;

          row.i    = ex; row.j    = ey; row.loc    = DMSTAG_ELEMENT; row.c    = 0;
          col[0].i = ex; col[0].j = ey; col[0].loc = DMSTAG_LEFT;    col[0].c = 0; val_A[0] = -1.0 * K_cont * dv / hx;
          col[1].i = ex; col[1].j = ey; col[1].loc = DMSTAG_RIGHT;   col[1].c = 0; val_A[1] =        K_cont * dv / hx;
          col[2].i = ex; col[2].j = ey; col[2].loc = DMSTAG_DOWN;    col[2].c = 0; val_A[2] = -1.0 * K_cont * dv / hy;
          col[3].i = ex; col[3].j = ey; col[3].loc = DMSTAG_UP;      col[3].c = 0; val_A[3] =        K_cont * dv / hy;
          col[4] = row;                                                            val_A[4] =  0.0;
          ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,5,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
          if (build_rhs) {
            ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm_coefficients,&coefficients_local);CHKERRQ(ierr);

  /* Add additional inverse viscosity terms (for use in building a preconditioning matrix) */
  if (parameters->include_inverse_visc) {
    if (parameters->faces_only) SETERRQ(PetscObjectComm((PetscObject)dm_main),PETSC_ERR_SUP,"Does not make sense with faces only");
    ierr = OperatorInsertInverseViscosityPressureTerms(dm_main, dm_coefficients, ctx->levels[level]->coeff, 1.0, A);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (build_rhs) {
    ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  }
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (build_rhs) {
    ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSystem3d(SystemParameters parameters,Mat *pA,Vec *pRhs)
{
  PetscErrorCode  ierr;
  PetscInt        N[3];
  PetscInt        ex,ey,ez,startx,starty,startz,nx,ny,nz;
  Mat             A;
  PetscReal       hx,hy,hz,dv;
  PetscInt        pinx,piny,pinz;
  Vec             coeff_local,rhs;
  PetscBool       build_rhs;
  DM              dm_main, dm_coefficients;
  PetscScalar     K_cont, K_bound;
  Ctx             ctx = parameters->ctx;
  PetscInt        level = parameters->level;

  PetscFunctionBeginUser;
  if (parameters->faces_only) {
    dm_main = ctx->levels[level]->dm_faces;
  } else {
    dm_main = ctx->levels[level]->dm_stokes;
  }
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  K_cont = ctx->levels[level]->K_cont;
  K_bound = ctx->levels[level]->K_bound;
  ierr = DMCreateMatrix(dm_main,pA);CHKERRQ(ierr);
  A = *pA;
  build_rhs = pRhs != NULL;
  if (build_rhs) {
    ierr = DMCreateGlobalVector(dm_main,pRhs);CHKERRQ(ierr);
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  ierr = DMStagGetCorners(dm_main,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm_main,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  hx = ctx->levels[level]->hx_characteristic;
  hy = ctx->levels[level]->hy_characteristic;
  hz = ctx->levels[level]->hz_characteristic;
  dv = hx*hy*hz;
  ierr = DMGetLocalVector(dm_coefficients,&coeff_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm_coefficients,ctx->levels[level]->coeff,INSERT_VALUES,coeff_local);CHKERRQ(ierr);

  if (N[0] < 2 || N[1] < 2 || N[2] < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for less than 2 elements in any direction");
  pinx = 1; piny = 0; pinz = 0; /* Depends on assertion above that there are at least two element in the x direction */

  /* Loop over all local elements.

     For each element, fill 4-7 rows of the matrix, corresponding to
     - the pressure degree of freedom (dof), centered on the element
     - the 3 velocity dofs on left, bottom, and back faces of the element
     - velocity dof on the right, top, and front faces of the element (only on domain boundaries)

   */
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        const PetscBool left_boundary   = ex == 0;
        const PetscBool right_boundary  = ex == N[0]-1;
        const PetscBool bottom_boundary = ey == 0;
        const PetscBool top_boundary    = ey == N[1]-1;
        const PetscBool back_boundary   = ez == 0;
        const PetscBool front_boundary  = ez == N[2]-1;

        /* Note that below, we depend on the check above that there is never one
           element (globally) in a given direction.  Thus, for example, an
           element is never both on the left and right boundary */

        /* X-faces - right boundary */
        if (right_boundary) {
          /* Right x-velocity Dirichlet */
          DMStagStencil     row;
          const PetscScalar val_rhs = 0.0;
          const PetscScalar val_A = K_bound;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_RIGHT; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
          if (build_rhs) {
            ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
          }
        }

        /* X faces - left*/
        {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_LEFT; row.c = 0;

          if (left_boundary) {
            /* Left x-velocity Dirichlet */
            const PetscScalar val_rhs = 0.0;
            const PetscScalar val_A = K_bound;

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          } else {
            /* X-momentum equation */
            PetscInt      count;
            DMStagStencil col[17];
            PetscScalar   val_A[17];
            DMStagStencil eta_point[6];
            PetscScalar   eta[6],eta_left,eta_right,eta_up,eta_down,eta_back,eta_front; /* relative to the left face */
            const PetscScalar val_rhs = 0.0;

            /* Get eta values */
            eta_point[0].i = ex-1; eta_point[0].j = ey; eta_point[0].k = ez; eta_point[0].loc = DMSTAG_ELEMENT;    eta_point[0].c = 0; /* Left  */
            eta_point[1].i = ex;   eta_point[1].j = ey; eta_point[1].k = ez; eta_point[1].loc = DMSTAG_ELEMENT;    eta_point[1].c = 0; /* Right */
            eta_point[2].i = ex;   eta_point[2].j = ey; eta_point[2].k = ez; eta_point[2].loc = DMSTAG_DOWN_LEFT;  eta_point[2].c = 0; /* Down  */
            eta_point[3].i = ex;   eta_point[3].j = ey; eta_point[3].k = ez; eta_point[3].loc = DMSTAG_UP_LEFT;    eta_point[3].c = 0; /* Up    */
            eta_point[4].i = ex;   eta_point[4].j = ey; eta_point[4].k = ez; eta_point[4].loc = DMSTAG_BACK_LEFT;  eta_point[4].c = 0; /* Back  */
            eta_point[5].i = ex;   eta_point[5].j = ey; eta_point[5].k = ez; eta_point[5].loc = DMSTAG_FRONT_LEFT; eta_point[5].c = 0; /* Front  */
            ierr = DMStagVecGetValuesStencil(dm_coefficients,coeff_local,6,eta_point,eta);CHKERRQ(ierr);
            eta_left = eta[0]; eta_right = eta[1]; eta_down = eta[2]; eta_up = eta[3]; eta_back = eta[4]; eta_front = eta[5];

            count = 0;

            col[count] = row;
            val_A[count] = -2.0 * dv * (eta_left + eta_right) / (hx*hx);
            if (!top_boundary)    val_A[count] += -1.0 * dv * eta_up    / (hy*hy);
            if (!bottom_boundary) val_A[count] += -1.0 * dv * eta_down  / (hy*hy);
            if (!back_boundary)   val_A[count] += -1.0 * dv * eta_back  / (hz*hz);
            if (!front_boundary)  val_A[count] += -1.0 * dv * eta_front / (hz*hz);
            ++count;

            col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_left  / (hx*hx); ++count;
            col[count].i = ex+1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_right  / (hx*hx); ++count;
            if (!bottom_boundary) {
              col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_down / (hy*hy); ++count;
            }
            if (!top_boundary) {
              col[count].i = ex; col[count].j = ey+1; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_up / (hy*hy); ++count;
            }
            if (!back_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_back / (hz*hz); ++count;
            }
            if (!front_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez+1; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_front / (hz*hz); ++count;
            }

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_DOWN;  col[count].c = 0;
            val_A[count]  =        dv * eta_down  / (hx*hy); ++count; /* down left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_DOWN;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_down  / (hx*hy); ++count; /* down right */

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_UP;    col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_up    / (hx*hy); ++count; /* up left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_UP;    col[count].c = 0;
            val_A[count]  =        dv * eta_up    / (hx*hy); ++count; /* up right */

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  =        dv * eta_back  / (hx*hz); ++count; /* back left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_back  / (hx*hz); ++count; /* back right */

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_front / (hx*hz); ++count; /* front left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  =        dv * eta_front / (hx*hz); ++count; /* front right */

            if (!parameters->faces_only) {
              col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] =             K_cont * dv / hx; ++count;
              col[count].i = ex;   col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] = -1.0 *      K_cont * dv / hx; ++count;
            }

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }

        /* Y faces - top boundary */
        if (top_boundary) {
          /* Top y-velocity Dirichlet */
          DMStagStencil     row;
          const PetscScalar val_rhs = 0.0;
          const PetscScalar val_A = K_bound;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_UP; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
          if (build_rhs) {
            ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
          }
        }

        /* Y faces - down */
        {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_DOWN; row.c = 0;

          if (bottom_boundary) {
            /* Bottom y-velocity Dirichlet */
            const PetscScalar val_rhs = 0.0;
            const PetscScalar val_A = K_bound;

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          } else {
            /* Y-momentum equation (including non-zero forcing) */
            PetscInt      count;
            DMStagStencil col[17];
            PetscScalar   val_rhs,val_A[17];
            DMStagStencil eta_point[6],rho_point[4];
            PetscScalar   eta[6],rho[4],eta_left,eta_right,eta_up,eta_down,eta_back,eta_front; /* relative to the bottom face */

            if (build_rhs) {
              /* get rho values  (note .c = 1) */
              /* Note that we have rho at perhaps strange points (edges not corners) */
              rho_point[0].i = ex; rho_point[0].j = ey; rho_point[0].k = ez; rho_point[0].loc = DMSTAG_DOWN_LEFT;  rho_point[0].c = 1;
              rho_point[1].i = ex; rho_point[1].j = ey; rho_point[1].k = ez; rho_point[1].loc = DMSTAG_DOWN_RIGHT; rho_point[1].c = 1;
              rho_point[2].i = ex; rho_point[2].j = ey; rho_point[2].k = ez; rho_point[2].loc = DMSTAG_BACK_DOWN;  rho_point[2].c = 1;
              rho_point[3].i = ex; rho_point[3].j = ey; rho_point[3].k = ez; rho_point[3].loc = DMSTAG_FRONT_DOWN; rho_point[3].c = 1;
              ierr = DMStagVecGetValuesStencil(dm_coefficients,coeff_local,4,rho_point,rho);CHKERRQ(ierr);

              /* Compute forcing */
              val_rhs = ctx->gy * dv * (0.25 * (rho[0] + rho[1] + rho[2] + rho[3]));
            }

            /* Get eta values */
            eta_point[0].i = ex; eta_point[0].j = ey;   eta_point[0].k = ez; eta_point[0].loc = DMSTAG_DOWN_LEFT;  eta_point[0].c = 0; /* Left  */
            eta_point[1].i = ex; eta_point[1].j = ey;   eta_point[1].k = ez; eta_point[1].loc = DMSTAG_DOWN_RIGHT; eta_point[1].c = 0; /* Right */
            eta_point[2].i = ex; eta_point[2].j = ey-1; eta_point[2].k = ez; eta_point[2].loc = DMSTAG_ELEMENT;    eta_point[2].c = 0; /* Down  */
            eta_point[3].i = ex; eta_point[3].j = ey;   eta_point[3].k = ez; eta_point[3].loc = DMSTAG_ELEMENT;    eta_point[3].c = 0; /* Up    */
            eta_point[4].i = ex; eta_point[4].j = ey;   eta_point[4].k = ez; eta_point[4].loc = DMSTAG_BACK_DOWN;  eta_point[4].c = 0; /* Back  */
            eta_point[5].i = ex; eta_point[5].j = ey;   eta_point[5].k = ez; eta_point[5].loc = DMSTAG_FRONT_DOWN; eta_point[5].c = 0; /* Front  */
            ierr = DMStagVecGetValuesStencil(dm_coefficients,coeff_local,6,eta_point,eta);CHKERRQ(ierr);
            eta_left = eta[0]; eta_right = eta[1]; eta_down = eta[2]; eta_up = eta[3]; eta_back = eta[4]; eta_front = eta[5];

            count = 0;

            col[count] = row;
            val_A[count] = -2.0 * dv * (eta_up + eta_down) / (hy*hy);
            if (!left_boundary)  val_A[count] += -1.0 * dv * eta_left  / (hx*hx);
            if (!right_boundary) val_A[count] += -1.0 * dv * eta_right / (hx*hx);
            if (!back_boundary)  val_A[count] += -1.0 * dv * eta_back  / (hz*hz);
            if (!front_boundary) val_A[count] += -1.0 * dv * eta_front / (hz*hz);
            ++count;

            col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_down / (hy*hy); ++count;
            col[count].i = ex; col[count].j = ey+1; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_up   / (hy*hy); ++count;

            if (!left_boundary) {
              col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_left / (hx*hx); ++count;
            }
            if (!right_boundary) {
              col[count].i = ex+1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_right / (hx*hx); ++count;
            }
            if (!back_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_back / (hz*hz); ++count;
            }
            if (!front_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez+1; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_front / (hz*hz); ++count;
            }

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc = DMSTAG_LEFT;  col[count].c = 0;
            val_A[count]  =        dv * eta_left  / (hx*hy); ++count; /* down left*/
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc = DMSTAG_LEFT;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_left  / (hx*hy); ++count; /* up left*/

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_right / (hx*hy); ++count; /* down right*/
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  =        dv * eta_right / (hx*hy); ++count; /* up right*/

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  =        dv * eta_back  / (hy*hz); ++count; /* back down */
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_back  / (hy*hz); ++count;/* back up */

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_front / (hy*hz); ++count; /* front down */
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  =        dv * eta_front / (hy*hz); ++count;/* front up */

            if (!parameters->faces_only) {
              col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] =             K_cont * dv / hy; ++count;
              col[count].i = ex; col[count].j = ey;   col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] = -1.0 *      K_cont * dv / hy; ++count;
            }

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }

        if (front_boundary) {
          /* Front z-velocity Dirichlet */
          DMStagStencil     row;
          const PetscScalar val_rhs = 0.0;
          const PetscScalar val_A = K_bound;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_FRONT; row.c = 0;
          ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
          if (build_rhs) {
            ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
          }
        }

        /* Z faces - back */
        {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_BACK; row.c = 0;

          if (back_boundary) {
            /* Back z-velocity Dirichlet */
            const PetscScalar val_rhs = 0.0;
            const PetscScalar val_A = K_bound;

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          } else {
            /* Z-momentum equation */
            PetscInt          count;
            DMStagStencil     col[17];
            PetscScalar       val_A[17];
            DMStagStencil     eta_point[6];
            PetscScalar       eta[6],eta_left,eta_right,eta_up,eta_down,eta_back,eta_front; /* relative to the back face */
            const PetscScalar val_rhs = 0.0;

            /* Get eta values */
            eta_point[0].i = ex; eta_point[0].j = ey; eta_point[0].k = ez;   eta_point[0].loc = DMSTAG_BACK_LEFT;  eta_point[0].c = 0; /* Left  */
            eta_point[1].i = ex; eta_point[1].j = ey; eta_point[1].k = ez;   eta_point[1].loc = DMSTAG_BACK_RIGHT; eta_point[1].c = 0; /* Right */
            eta_point[2].i = ex; eta_point[2].j = ey; eta_point[2].k = ez;   eta_point[2].loc = DMSTAG_BACK_DOWN;  eta_point[2].c = 0; /* Down  */
            eta_point[3].i = ex; eta_point[3].j = ey; eta_point[3].k = ez;   eta_point[3].loc = DMSTAG_BACK_UP;    eta_point[3].c = 0; /* Up    */
            eta_point[4].i = ex; eta_point[4].j = ey; eta_point[4].k = ez-1; eta_point[4].loc = DMSTAG_ELEMENT;    eta_point[4].c = 0; /* Back  */
            eta_point[5].i = ex; eta_point[5].j = ey; eta_point[5].k = ez;   eta_point[5].loc = DMSTAG_ELEMENT;    eta_point[5].c = 0; /* Front  */
            ierr = DMStagVecGetValuesStencil(dm_coefficients,coeff_local,6,eta_point,eta);CHKERRQ(ierr);
            eta_left = eta[0]; eta_right = eta[1]; eta_down = eta[2]; eta_up = eta[3]; eta_back = eta[4]; eta_front = eta[5];

            count = 0;

            col[count] = row;
            val_A[count] = -2.0 * dv * (eta_back + eta_front) / (hz*hz);
            if (!left_boundary)   val_A[count] += -1.0 * dv * eta_left  / (hx*hx);
            if (!right_boundary)  val_A[count] += -1.0 * dv * eta_right / (hx*hx);
            if (!top_boundary)    val_A[count] += -1.0 * dv * eta_up    / (hy*hy);
            if (!bottom_boundary) val_A[count] += -1.0 * dv * eta_down  / (hy*hy);
            ++count;

            col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_BACK; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_back  / (hz*hz); ++count;
            col[count].i = ex; col[count].j = ey; col[count].k = ez+1; col[count].loc = DMSTAG_BACK; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_front / (hz*hz); ++count;

            if (!left_boundary) {
              col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_left / (hx*hx); ++count;
            }
            if (!right_boundary) {
              col[count].i = ex+1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_right / (hx*hx); ++count;
            }
            if (!bottom_boundary) {
              col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_down / (hy*hy); ++count;
            }
            if (!top_boundary) {
              col[count].i = ex; col[count].j = ey+1; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_up  / (hy*hy); ++count;
            }

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count]  =        dv * eta_left  / (hx*hz); ++count; /* back left*/
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_left  / (hx*hz); ++count; /* front left*/

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_right / (hx*hz); ++count; /* back right */
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  =        dv * eta_right / (hx*hz); ++count; /* front right*/

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count]  =        dv * eta_down  / (hy*hz); ++count; /* back down */
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_down  / (hy*hz); ++count; /* back down */

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_UP; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_up    / (hy*hz); ++count; /* back up */
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_UP; col[count].c = 0;
            val_A[count]  =        dv * eta_up    / (hy*hz); ++count; /* back up */

            if (!parameters->faces_only) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] =             K_cont * dv / hz; ++count;
              col[count].i = ex; col[count].j = ey;   col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] = -1.0 *      K_cont * dv / hz; ++count;
            }

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }

        /* Elements */
        if (!parameters->faces_only) {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_ELEMENT; row.c = 0;

          if (ctx->pin_pressure && ex == pinx && ey == piny && ez == pinz) {
            /* Pin a pressure node to zero, if requested */
            const PetscScalar val_A = K_bound;
            const PetscScalar val_rhs = 0.0;

            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          } else {
            /* Continuity equation */
            /* Note that this includes an explicit zero on the diagonal. This is only needed for
               some direct solvers (not required if using an iterative solver and setting a constant-pressure nullspace) */
            /* Note: the scaling by dv is not chosen in a principled way and is likely sub-optimal */
            DMStagStencil     col[7];
            PetscScalar       val_A[7];
            const PetscScalar val_rhs = 0.0;

            col[0].i = ex; col[0].j = ey; col[0].k = ez; col[0].loc = DMSTAG_LEFT;    col[0].c = 0; val_A[0] = - 1.0 * K_cont * dv / hx;
            col[1].i = ex; col[1].j = ey; col[1].k = ez; col[1].loc = DMSTAG_RIGHT;   col[1].c = 0; val_A[1] =         K_cont * dv / hx;
            col[2].i = ex; col[2].j = ey; col[2].k = ez; col[2].loc = DMSTAG_DOWN;    col[2].c = 0; val_A[2] = - 1.0 * K_cont * dv / hy;
            col[3].i = ex; col[3].j = ey; col[3].k = ez; col[3].loc = DMSTAG_UP;      col[3].c = 0; val_A[3] =         K_cont * dv / hy;
            col[4].i = ex; col[4].j = ey; col[4].k = ez; col[4].loc = DMSTAG_BACK;    col[4].c = 0; val_A[4] = - 1.0 * K_cont * dv / hz;
            col[5].i = ex; col[5].j = ey; col[5].k = ez; col[5].loc = DMSTAG_FRONT;   col[5].c = 0; val_A[5] =         K_cont * dv / hz;
            col[6] = row;                                                                           val_A[6] = 0.0;
            ierr = DMStagMatSetValuesStencil(dm_main,A,1,&row,7,col,val_A,INSERT_VALUES);CHKERRQ(ierr);
            if (build_rhs) {
              ierr = DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  }
  ierr = DMRestoreLocalVector(dm_coefficients,&coeff_local);CHKERRQ(ierr);

  /* Add additional inverse viscosity terms (for use in building a preconditioning matrix) */
  if (parameters->include_inverse_visc) {
    if (parameters->faces_only) SETERRQ(PetscObjectComm((PetscObject)dm_main),PETSC_ERR_SUP,"Does not make sense with faces only");
    ierr = OperatorInsertInverseViscosityPressureTerms(dm_main, dm_coefficients, ctx->levels[level]->coeff, 1.0, A);CHKERRQ(ierr);  // TODO not sure of scale
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (build_rhs) {
    ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  }
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (build_rhs) {
    ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSystem(SystemParameters parameters,Mat *pA,Vec *pRhs)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (parameters->ctx->dim == 2) {
    ierr = CreateSystem2d(parameters,pA,pRhs);CHKERRQ(ierr);
  } else if (parameters->ctx->dim == 3) {
    ierr = CreateSystem3d(parameters,pA,pRhs);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %d",parameters->ctx->dim);
  PetscFunctionReturn(0);
}


PetscErrorCode PopulateCoefficientData(Ctx ctx,PetscInt level)
{
  PetscErrorCode ierr;
  PetscInt       dim;
  PetscInt       N[3];
  PetscInt       ex,ey,ez,startx,starty,startz,nx,ny,nz;
  PetscInt       slot_prev,slot_center;
  PetscInt       slot_rho_downleft,slot_rho_backleft,slot_rho_backdown,slot_eta_element,slot_eta_downleft,slot_eta_backleft,slot_eta_backdown;
  Vec            coeff_local;
  PetscReal      **arr_coordinates_x,**arr_coordinates_y,**arr_coordinates_z;
  DM             dm_coefficients;
  Vec            coeff;

  PetscFunctionBeginUser;
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  ierr = DMGetDimension(dm_coefficients,&dim);CHKERRQ(ierr);

  /* Create global coefficient vector */
  ierr = DMCreateGlobalVector(dm_coefficients,&ctx->levels[level]->coeff);CHKERRQ(ierr);
  coeff = ctx->levels[level]->coeff;

  /* Get temporary access to a local representation of the coefficient data */
  ierr = DMGetLocalVector(dm_coefficients,&coeff_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm_coefficients,coeff,INSERT_VALUES,coeff_local);CHKERRQ(ierr);

  /* Use direct array acccess to coefficient and coordinate arrays, to popoulate coefficient data */
  ierr = DMStagGetGhostCorners(dm_coefficients,&startx,&starty,&startz,&nx,&ny,&nz);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm_coefficients,&N[0],&N[1],&N[2]);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm_coefficients,&arr_coordinates_x,&arr_coordinates_y,&arr_coordinates_z);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_coefficients,DMSTAG_ELEMENT,&slot_center);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm_coefficients,DMSTAG_LEFT,&slot_prev);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_ELEMENT,  0,&slot_eta_element);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_DOWN_LEFT,0,&slot_eta_downleft);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_DOWN_LEFT,1,&slot_rho_downleft);CHKERRQ(ierr);
  if (dim == 2) {
    PetscScalar ***arr_coefficients;

    ierr = DMStagVecGetArray(dm_coefficients,coeff_local,&arr_coefficients);CHKERRQ(ierr);
    /* Note that these ranges are with respect to the local representation */
    for (ey =starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        arr_coefficients[ey][ex][slot_eta_element]  = ctx->GetEta(ctx,arr_coordinates_x[ex][slot_center],arr_coordinates_y[ey][slot_center],0.0);
        arr_coefficients[ey][ex][slot_eta_downleft] = ctx->GetEta(ctx,arr_coordinates_x[ex][slot_prev],  arr_coordinates_y[ey][slot_prev],  0.0);
        arr_coefficients[ey][ex][slot_rho_downleft] = ctx->GetRho(ctx,arr_coordinates_x[ex][slot_prev],  arr_coordinates_y[ey][slot_prev],  0.0);
      }
    }
    ierr = DMStagVecRestoreArray(dm_coefficients,coeff_local,&arr_coefficients);CHKERRQ(ierr);
  } else if (dim == 3) {
    PetscScalar ****arr_coefficients;

    ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_LEFT,0,&slot_eta_backleft);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_LEFT,1,&slot_rho_backleft);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_DOWN,0,&slot_eta_backdown);CHKERRQ(ierr);
    ierr = DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_DOWN,1,&slot_rho_backdown);CHKERRQ(ierr);
    ierr = DMStagVecGetArray(dm_coefficients,coeff_local,&arr_coefficients);CHKERRQ(ierr);
    /* Note that these are with respect to the entire local representation, including ghosts */
    for (ez = startz; ez<startz+nz; ++ez) {
      for (ey = starty; ey<starty+ny; ++ey) {
        for (ex = startx; ex<startx+nx; ++ex) {
          const PetscScalar x_prev = arr_coordinates_x[ex][slot_prev];
          const PetscScalar y_prev = arr_coordinates_y[ey][slot_prev];
          const PetscScalar z_prev = arr_coordinates_z[ez][slot_prev];
          const PetscScalar x_center = arr_coordinates_x[ex][slot_center];
          const PetscScalar y_center = arr_coordinates_y[ey][slot_center];
          const PetscScalar z_center = arr_coordinates_z[ez][slot_center];

          arr_coefficients[ez][ey][ex][slot_eta_element]  = ctx->GetEta(ctx,x_center,y_center,z_center);
          arr_coefficients[ez][ey][ex][slot_eta_downleft] = ctx->GetEta(ctx,x_prev,  y_prev,  z_center);
          arr_coefficients[ez][ey][ex][slot_rho_downleft] = ctx->GetRho(ctx,x_prev,  y_prev,  z_center);
          arr_coefficients[ez][ey][ex][slot_eta_backleft] = ctx->GetEta(ctx,x_prev,  y_center,z_prev  );
          arr_coefficients[ez][ey][ex][slot_rho_backleft] = ctx->GetRho(ctx,x_prev,  y_center,z_prev  );
          arr_coefficients[ez][ey][ex][slot_eta_backdown] = ctx->GetEta(ctx,x_center,y_prev,  z_prev  );
          arr_coefficients[ez][ey][ex][slot_rho_backdown] = ctx->GetRho(ctx,x_center,y_prev,  z_prev  );
        }
      }
    }
    ierr = DMStagVecRestoreArray(dm_coefficients,coeff_local,&arr_coefficients);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)dm_coefficients),PETSC_ERR_SUP,"Unsupported dimension %d",dim);
  ierr = DMStagRestoreProductCoordinateArraysRead(dm_coefficients,&arr_coordinates_x,&arr_coordinates_y,&arr_coordinates_z);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm_coefficients,coeff_local,INSERT_VALUES,coeff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm_coefficients,&coeff_local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAuxiliaryOperator(Ctx ctx,PetscInt level, Mat *p_S_hat)
{
  PetscErrorCode ierr;
  DM             dm_element;
  Mat            S_hat;
  DM             dm_stokes,dm_coefficients;
  Vec            coeff;

  PetscFunctionBeginUser;
  dm_stokes = ctx->levels[level]->dm_stokes;
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  coeff = ctx->levels[level]->coeff;
  if (ctx->dim == 2) {
    ierr = DMStagCreateCompatibleDMStag(dm_stokes,0,0,1,0,&dm_element);CHKERRQ(ierr);
  } else if (ctx->dim == 3) {
    ierr = DMStagCreateCompatibleDMStag(dm_stokes,0,0,0,1,&dm_element);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %d",ctx->dim);
  ierr = DMCreateMatrix(dm_element,p_S_hat);CHKERRQ(ierr);
  S_hat = *p_S_hat;
  ierr = OperatorInsertInverseViscosityPressureTerms(dm_element,dm_coefficients,coeff,1.0,S_hat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(S_hat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S_hat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode OperatorInsertInverseViscosityPressureTerms(DM dm, DM dm_coefficients, Vec coefficients, PetscScalar scale, Mat mat)
{
  PetscErrorCode ierr;
  PetscInt       dim,ex,ey,ez,startx,starty,startz,nx,ny,nz;
  Vec            coeff_local;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm_coefficients,&coeff_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm_coefficients,coefficients,INSERT_VALUES,coeff_local);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  if (dim == 2) { /* Trick to have one loop nest */
    startz = 0;
    nz = 1;
  }
  for (ez = startz; ez<startz+nz; ++ez) {
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil  from,to;
        PetscScalar    val;

        /* component 0 on element is viscosity */
        from.i = ex; from.j = ey; from.k = ez; from.c = 0; from.loc = DMSTAG_ELEMENT;
        ierr = DMStagVecGetValuesStencil(dm_coefficients,coeff_local,1,&from,&val);CHKERRQ(ierr);
        val = scale/val; /* inverse viscosity, scaled */
        to = from;
        ierr = DMStagMatSetValuesStencil(dm,mat,1,&to,1,&to,&val,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMRestoreLocalVector(dm_coefficients,&coeff_local);CHKERRQ(ierr);
  /* Note that this function does not call MatAssembly{Begin,End} */
  PetscFunctionReturn(0);
}

/* Create a pressure-only DMStag and use it to generate a nullspace vector
   - Create a compatible DMStag with one dof per element (and nothing else).
   - Create a constant vector and normalize it
   - Migrate it to a vector on the original dmSol (making use of the fact
   that this will fill in zeros for "extra" dof)
   - Set the nullspace for the operator
   - Destroy everything (the operator keeps the references it needs) */
static PetscErrorCode AttachNullspace(DM dmSol,Mat A)
{
  PetscErrorCode ierr;
  DM             dmPressure;
  Vec            constantPressure,basis;
  PetscReal      nrm;
  MatNullSpace   matNullSpace;

  PetscFunctionBeginUser;
  ierr = DMStagCreateCompatibleDMStag(dmSol,0,0,1,0,&dmPressure);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmPressure,&constantPressure);CHKERRQ(ierr);
  ierr = VecSet(constantPressure,1.0);CHKERRQ(ierr);
  ierr = VecNorm(constantPressure,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = VecScale(constantPressure,1.0/nrm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dmSol,&basis);CHKERRQ(ierr);
  ierr = DMStagMigrateVec(dmPressure,constantPressure,dmSol,basis);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dmSol),PETSC_FALSE,1,&basis,&matNullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&basis);CHKERRQ(ierr);
  ierr = MatSetNullSpace(A,matNullSpace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&matNullSpace);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmPressure,&constantPressure);CHKERRQ(ierr);
  ierr = DMDestroy(&dmPressure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DumpSolution(Ctx ctx,PetscInt level, Vec x)
{
  PetscErrorCode ierr;
  DM             dm_stokes,dm_coefficients;
  Vec            coeff;
  DM             dm_vel_avg;
  Vec            vel_avg;
  DM             da_vel_avg,da_p,da_eta_element;
  Vec            vec_vel_avg,vec_p,vec_eta_element;
  DM             da_eta_down_left,da_rho_down_left,da_eta_back_left,da_rho_back_left,da_eta_back_down,da_rho_back_down;
  Vec            vec_eta_down_left,vec_rho_down_left,vec_eta_back_left,vec_rho_back_left,vec_eta_back_down,vec_rho_back_down;
  PetscInt       ex,ey,ez,startx,starty,startz,nx,ny,nz;
  Vec            stokesLocal;

  PetscFunctionBeginUser;
  dm_stokes = ctx->levels[level]->dm_stokes;
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  coeff = ctx->levels[level]->coeff;

  /* For convenience, create a new DM and Vec which will hold averaged velocities
     Note that this could also be accomplished with direct array access, using
     DMStagVecGetArray() and related functions */
  if (ctx->dim == 2) {
    ierr = DMStagCreateCompatibleDMStag(dm_stokes,0,0,2,0,&dm_vel_avg);CHKERRQ(ierr); /* 2 dof per element */
  } else if (ctx->dim == 3) {
    ierr = DMStagCreateCompatibleDMStag(dm_stokes,0,0,0,3,&dm_vel_avg);CHKERRQ(ierr); /* 3 dof per element */
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %d",ctx->dim);
  ierr = DMSetUp(dm_vel_avg);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dm_vel_avg,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm_vel_avg,&vel_avg);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm_stokes,&stokesLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm_stokes,x,INSERT_VALUES,stokesLocal);CHKERRQ(ierr);
    ierr = DMStagGetCorners(dm_vel_avg,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL);CHKERRQ(ierr);
  if (ctx->dim == 2) {
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4],to[2];
        PetscScalar   valFrom[4],valTo[2];

        from[0].i = ex; from[0].j = ey; from[0].loc = DMSTAG_UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DMSTAG_DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = DMSTAG_LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = DMSTAG_RIGHT; from[3].c = 0;
        ierr = DMStagVecGetValuesStencil(dm_stokes,stokesLocal,4,from,valFrom);CHKERRQ(ierr);
        to[0].i = ex; to[0].j = ey; to[0].loc = DMSTAG_ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = DMSTAG_ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        ierr = DMStagVecSetValuesStencil(dm_vel_avg,vel_avg,2,to,valTo,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else if (ctx->dim == 3) {
    for (ez = startz; ez<startz+nz; ++ez) {
      for (ey = starty; ey<starty+ny; ++ey) {
        for (ex = startx; ex<startx+nx; ++ex) {
          DMStagStencil from[6],to[3];
          PetscScalar   valFrom[6],valTo[3];

          from[0].i = ex; from[0].j = ey; from[0].k = ez; from[0].loc = DMSTAG_UP;    from[0].c = 0;
          from[1].i = ex; from[1].j = ey; from[1].k = ez; from[1].loc = DMSTAG_DOWN;  from[1].c = 0;
          from[2].i = ex; from[2].j = ey; from[2].k = ez; from[2].loc = DMSTAG_LEFT;  from[2].c = 0;
          from[3].i = ex; from[3].j = ey; from[3].k = ez; from[3].loc = DMSTAG_RIGHT; from[3].c = 0;
          from[4].i = ex; from[4].j = ey; from[4].k = ez; from[4].loc = DMSTAG_BACK;  from[4].c = 0;
          from[5].i = ex; from[5].j = ey; from[5].k = ez; from[5].loc = DMSTAG_FRONT; from[5].c = 0;
          ierr = DMStagVecGetValuesStencil(dm_stokes,stokesLocal,6,from,valFrom);CHKERRQ(ierr);
          to[0].i = ex; to[0].j = ey; to[0].k = ez; to[0].loc = DMSTAG_ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
          to[1].i = ex; to[1].j = ey; to[1].k = ez; to[1].loc = DMSTAG_ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
          to[2].i = ex; to[2].j = ey; to[2].k = ez; to[2].loc = DMSTAG_ELEMENT;    to[2].c = 2; valTo[2] = 0.5 * (valFrom[4] + valFrom[5]);
          ierr = DMStagVecSetValuesStencil(dm_vel_avg,vel_avg,3,to,valTo,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = VecAssemblyBegin(vel_avg);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vel_avg);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm_stokes,&stokesLocal);CHKERRQ(ierr);

  /* Create individual DMDAs for sub-grids of our DMStag objects. This is
     somewhat inefficient, but allows use of the DMDA API without re-implementing
     all utilities for DMStag */

  ierr = DMStagVecSplitToDMDA(dm_stokes,x,DMSTAG_ELEMENT,0,&da_p,&vec_p);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec_p,"p (scaled)");CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_ELEMENT,0, &da_eta_element,&vec_eta_element);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec_eta_element,"eta");CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_DOWN_LEFT,0,&da_eta_down_left,&vec_eta_down_left);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec_eta_down_left,"eta");CHKERRQ(ierr);

  ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_DOWN_LEFT,1,&da_rho_down_left,&vec_rho_down_left);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vec_rho_down_left,"density");CHKERRQ(ierr);

  if (ctx->dim == 3) {
    ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_LEFT,0,&da_eta_back_left,&vec_eta_back_left);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec_eta_back_left,"eta");CHKERRQ(ierr);

    ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_LEFT,1,&da_rho_back_left,&vec_rho_back_left);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec_rho_back_left,"rho");CHKERRQ(ierr);

    ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_DOWN,0,&da_eta_back_down,&vec_eta_back_down);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec_eta_back_down,"eta");CHKERRQ(ierr);

    ierr = DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_DOWN,1,&da_rho_back_down,&vec_rho_back_down);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec_rho_back_down,"rho");CHKERRQ(ierr);
  }

  ierr = DMStagVecSplitToDMDA(dm_vel_avg,vel_avg,DMSTAG_ELEMENT,-3,&da_vel_avg,&vec_vel_avg);CHKERRQ(ierr); /* note -3 : pad with zero */
  ierr = PetscObjectSetName((PetscObject)vec_vel_avg,"Velocity (Averaged)");CHKERRQ(ierr);

  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;

    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_vel_avg),"ex4_element.vtr",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vec_vel_avg,viewer);CHKERRQ(ierr);
    ierr = VecView(vec_p,viewer);CHKERRQ(ierr);
    ierr = VecView(vec_eta_element,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Dump vertex- or edge-based fields to a second .vtr file */
  {
    PetscViewer viewer;

    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_eta_down_left),"ex4_down_left.vtr",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vec_eta_down_left,viewer);CHKERRQ(ierr);
    ierr = VecView(vec_rho_down_left,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (ctx->dim == 3) {
    PetscViewer viewer;

    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_eta_back_left),"ex4_back_left.vtr",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vec_eta_back_left,viewer);CHKERRQ(ierr);
    ierr = VecView(vec_rho_back_left,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (ctx->dim == 3) {
    PetscViewer viewer;

    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_eta_back_down),"ex4_back_down.vtr",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vec_eta_back_down,viewer);CHKERRQ(ierr);
    ierr = VecView(vec_rho_back_down,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Destroy DMDAs and Vecs */
  ierr = VecDestroy(&vec_vel_avg);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_p);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_eta_element);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_eta_down_left);CHKERRQ(ierr);
  if (ctx->dim == 3) {
    ierr = VecDestroy(&vec_eta_back_left);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_eta_back_down);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vec_rho_down_left);CHKERRQ(ierr);
  if (ctx->dim == 3) {
    ierr = VecDestroy(&vec_rho_back_left);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_rho_back_down);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&da_vel_avg);CHKERRQ(ierr);
  ierr = DMDestroy(&da_p);CHKERRQ(ierr);
  ierr = DMDestroy(&da_eta_element);CHKERRQ(ierr);
  ierr = DMDestroy(&da_eta_down_left);CHKERRQ(ierr);
  if (ctx->dim == 3) {
    ierr = DMDestroy(&da_eta_back_left);CHKERRQ(ierr);
    ierr = DMDestroy(&da_eta_back_down);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&da_rho_down_left);CHKERRQ(ierr);
  if (ctx->dim == 3) {
    ierr = DMDestroy(&da_rho_back_left);CHKERRQ(ierr);
    ierr = DMDestroy(&da_rho_back_down);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vel_avg);CHKERRQ(ierr);
  ierr = DMDestroy(&dm_vel_avg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DumpOperator(Mat A, const char *name)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"%s.pbin",name);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)A),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*TEST

   test:
      suffix: direct_umfpack
      requires: suitesparse !complex
      nsize: 1
      args: -dim 2 -coefficients layers -nondimensional 0 -stag_grid_x 12 -stag_grid_y 7 -pc_type lu -pc_factor_mat_solver_type umfpack

   test:
      suffix: direct_mumps
      requires: mumps !complex
      nsize: 9
      args: -dim 2 -coefficients layers -nondimensional 0 -stag_grid_x 13 -stag_grid_y 8 -pc_type lu -pc_factor_mat_solver_type mumps

   test:
      suffix: isovisc_nondim_abf_mg
      nsize: 1
      args: -dim 2 -coefficients layers -nondimensional 1 -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -stag_grid_x 24 -stag_grid_y 24 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_ksp_converged_reason -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper -isoviscous

   test:
      suffix: isovisc_nondim_abf_mg_2
      nsize: 1
      args: -dim 2 -coefficients layers -nondimensional -isoviscous -eta1 1.0 -stag_grid_x 32 -stag_grid_y 32 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper -build_auxiliary_operator -fieldsplit_element_ksp_type preonly -fieldsplit_element_pc_type jacobi -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_mg_levels_pc_type jacobi -fieldsplit_face_mg_levels_ksp_type chebyshev

   test:
      suffix: nondim_abf_mg
      requires: suitesparse
      nsize: 4
      args: -dim 2 -coefficients layers -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_mg_coarse_pc_type redundant -fieldsplit_face_mg_coarse_redundant_pc_type lu -fieldsplit_face_mg_coarse_redundant_pc_factor_mat_solver_type umfpack  -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper -nondimensional -eta1 1e-2 -eta2 1.0 -ksp_monitor -fieldsplit_face_mg_levels_pc_type jacobi -fieldsplit_face_mg_levels_ksp_type gmres -fieldsplit_element_pc_type jacobi -pc_fieldsplit_schur_precondition selfp   -stag_grid_x 32 -stag_grid_y 32 -fieldsplit_face_ksp_monitor

   test:
      suffix: nondim_abf_lu
      requires: suitesparse
      nsize: 1
      args: -dim 2 -coefficients layers -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -ksp_type fgmres -fieldsplit_element_pc_type none -pc_fieldsplit_schur_fact_type upper -nondimensional -eta1 1e-2 -eta2 1.0 -isoviscous 0 -ksp_monitor -fieldsplit_element_pc_type jacobi -build_auxiliary_operator -fieldsplit_face_pc_type lu -fieldsplit_face_pc_factor_mat_solver_type umfpack -stag_grid_x 32 -stag_grid_y 32

   test:
      suffix: 3d_nondim_isovisc_abf_mg
      nsize: 1
      args: -dim 3 -coefficients layers -isoviscous -nondimensional -build_auxiliary_operator -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -s 16 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_ksp_converged_reason -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper

   test:
      suffix: 3d_nondim_isovisc_sinker_abf_mg
      nsize: 1
      args: -dim 3 -coefficients sinker -isoviscous -nondimensional -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -s 16 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_ksp_converged_reason -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper

   test:
      suffix: 3d_nondim_mono_mg_lamemstyle
      nsize: 1
      requires: suitesparse
      args: -dim 3 -coefficients layers -nondimensional -s 16 -custom_pc_mat -pc_type mg -pc_mg_galerkin -pc_mg_levels 2 -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_levels_ksp_richardson_scale 0.5 -mg_levels_ksp_max_it 20 -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack

   test:
      suffix: 3d_nondim_mono_mg_lamemstyle_par
      nsize: 8
      requires: mumps
      args: -dim 3 -coefficients layers -nondimensional -s 16 -custom_pc_mat -pc_type mg -pc_mg_galerkin -pc_mg_levels 2 -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_levels_ksp_richardson_scale 0.5 -mg_levels_ksp_max_it 20 -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type mumps
TEST*/
