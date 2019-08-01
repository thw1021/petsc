static char help[] = "Test: Solve a toy 2D problem on a staggered grid using 2-level multigrid. \n"
                      "The solve is only applied to the u-u block.\n\n";
/*

  Solves an isoviscous incompressible Stokes problem on a rectangular 2D
  domain, using a manufactured solution.

  u_xx + u_yy - p_x = f^x
  v_xx + v_yy - p_y = f^y
  u_x + v_y         = g

  g is 0 in the physical case.

  Boundary conditions give prescribed flow perpendicular to the boundaries,
  and zero derivative perpendicular to them (free slip).

  Supply the -analyze flag to activate a custom KSP monitor. Note that
  this does an additional direct solve of the system to have an "exact"
  solution to the discrete system available (see KSPSetOptionsPrefix
  call below for the prefix).

  This is for testing purposes, and uses some routines to make
  sure that transfer operators are consistent with extrating submatrices.

  -extractTransferOperators (true by default) defines transfer operators for
  the velocity-velocity system by extracing submatrices from the operators for
  the full system.

*/
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h> /* Includes petscdmproduct.h */

/* Shorter, more convenient names for DMStagStencilLocation entries */
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

static PetscErrorCode CreateSystem(DM,Mat*,Vec*);
static PetscErrorCode CreateNumericalReferenceSolution(Mat,Vec,Vec*);
static PetscErrorCode DMStagAnalysisKSPMonitor(KSP,PetscInt,PetscReal,void*);
typedef struct {
  DM  dm;
  Vec solRef,solRefNum,solPrev;
} DMStagAnalysisKSPMonitorContext;

/* Manufactured solution. Chosen to be higher order than can be solved exactly,
and to have a zero derivative for flow parallel to the boundaries. That is,
d(ux)/dy = 0 at the top and bottom boundaries, and d(uy)/dx = 0 at the right
and left boundaries. */
static PetscScalar uxRef(PetscScalar x,PetscScalar y) {return 0.0*x + y*y - 2.0*y*y*y + y*y*y*y;}    /* no x-dependence  */
static PetscScalar uyRef(PetscScalar x,PetscScalar y) {return x*x - 2.0*x*x*x + x*x*x*x + 0.0*y;}    /* no y-dependence  */
static PetscScalar fx   (PetscScalar x,PetscScalar y) {return 0.0*x + 2.0 -12.0*y + 12.0*y*y + 1.0;} /* no x-dependence  */
static PetscScalar fy   (PetscScalar x,PetscScalar y) {return 2.0 -12.0*x + 12.0*x*x + 3.0*y;}
static PetscScalar g    (PetscScalar x,PetscScalar y) {return 0.0*x*y;}                              /* identically zero */

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dmSol,dmSolc,dmuu,dmuuc;
  KSP            ksp,kspc;
  PC             pc;
  Mat            IIu,Ru,Auu,Auuc;
  Vec            su,xu,fu;
  IS             isuf,ispf,isuc,ispc;
  PetscInt       cnt = 0;
  DMStagStencil  stencil_set[1+1+1];
  PetscBool      extractTransferOperators,extractSystem;

  DMStagAnalysisKSPMonitorContext mctx;
  PetscBool                       analyze;

  /* Initialize PETSc and process command line arguments */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  analyze = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-analyze",&analyze,NULL);CHKERRQ(ierr);
  extractTransferOperators = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-extractTransferOperators",&extractTransferOperators,NULL);CHKERRQ(ierr);
  extractSystem = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-extractSystem",&extractSystem,NULL);CHKERRQ(ierr);

  /* Create 2D DMStag for the solution, and set up. */
  {
    const PetscInt dof0 = 0, dof1 = 1,dof2 = 1; /* 1 dof on each edge and element center */
    const PetscInt stencilWidth = 1;
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,8,8,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,&dmSol);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmSol);CHKERRQ(ierr);
    ierr = DMSetUp(dmSol);CHKERRQ(ierr);
  }

  /* Create coarse DMStag (which we may or may not directly use) */
  ierr = DMCoarsen(dmSol,MPI_COMM_NULL,&dmSolc);CHKERRQ(ierr);
  ierr = DMSetUp(dmSolc);CHKERRQ(ierr);

  /* Create compatible DMStags with only velocity dof (which we may or may not
     directly use) */
  ierr = DMStagCreateCompatibleDMStag(dmSol,0,1,0,0,&dmuu);CHKERRQ(ierr); /* vel-only */
  ierr = DMSetUp(dmuu);CHKERRQ(ierr);
  ierr = DMCoarsen(dmuu,MPI_COMM_NULL,&dmuuc);CHKERRQ(ierr);
  ierr = DMSetUp(dmuuc);CHKERRQ(ierr);

  /* Define uniform coordinates as a product of 1D arrays */
  ierr = DMStagSetUniformCoordinatesProduct(dmSol, 0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmSolc,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmuu,  0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(dmuuc, 0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);

  /* Create ISs for the velocity and pressure blocks */
  /* i,j will be ignored */
  stencil_set[cnt].loc = DMSTAG_DOWN;    stencil_set[cnt].c = 0;   cnt++; /* u */
  stencil_set[cnt].loc = DMSTAG_LEFT;    stencil_set[cnt].c = 0;   cnt++; /* v */
  stencil_set[cnt].loc = DMSTAG_ELEMENT; stencil_set[cnt].c = 0;   cnt++; /* p */

  ierr = DMStagCreateISFromStencils(dmSol, 2,stencil_set,    &isuf);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMStagCreateISFromStencils(dmSol, 1,&stencil_set[2],&ispf);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMStagCreateISFromStencils(dmSolc,2,stencil_set,    &isuc);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMStagCreateISFromStencils(dmSolc,1,&stencil_set[2],&ispc);CHKERRQ(ierr);CHKERRQ(ierr);

  /* Assemble velocity-velocity system */
  if (extractSystem) {
    Mat A,Ac;
    Vec tmp,rhs;

    ierr = CreateSystem(dmSol,&A,&rhs);CHKERRQ(ierr);
    ierr = CreateSystem(dmSolc,&Ac,NULL);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(A,isuf,isuf,MAT_INITIAL_MATRIX,&Auu);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(Ac,isuc,isuc,MAT_INITIAL_MATRIX,&Auuc);CHKERRQ(ierr);
    ierr = MatCreateVecs(Auu,&xu,&fu);CHKERRQ(ierr);
    ierr = VecGetSubVector(rhs,isuf,&tmp);CHKERRQ(ierr);
    ierr = VecCopy(tmp,fu);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(rhs,isuf,&tmp);CHKERRQ(ierr);
    ierr = MatDestroy(&Ac);CHKERRQ(ierr);
    ierr = VecDestroy(&rhs);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  } else {
    ierr = CreateSystem(dmuu,&Auu,&fu);CHKERRQ(ierr);
    ierr = CreateSystem(dmuuc,&Auuc,NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(Auu,&xu,NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)Auu,"Auu");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Auuc,"Auuc");CHKERRQ(ierr);

  /* Create Transfer Operators and scaling for the velocity-velocity block */
  if (extractTransferOperators) {
    Mat II,R;
    Vec s,tmp;

    ierr = DMCreateInterpolation(dmSolc,dmSol,&II,NULL);CHKERRQ(ierr);
    ierr = DMCreateRestriction(dmSolc,dmSol,&R);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(II,isuf,isuc,MAT_INITIAL_MATRIX,&IIu);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(R,isuc,isuf,MAT_INITIAL_MATRIX,&Ru);CHKERRQ(ierr);
    ierr = DMCreateInterpolationScale(dmSolc,dmSol,II,&s);CHKERRQ(ierr);
    ierr = MatCreateVecs(IIu,&su,NULL);CHKERRQ(ierr);
    ierr = VecGetSubVector(s,isuc,&tmp);CHKERRQ(ierr);
    ierr = VecCopy(tmp,su);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(s,isuc,&tmp);CHKERRQ(ierr);
    ierr = MatDestroy(&R);CHKERRQ(ierr);
    ierr = MatDestroy(&II);CHKERRQ(ierr);
    ierr = VecDestroy(&s);CHKERRQ(ierr);
  } else {
    ierr = DMCreateInterpolation(dmuuc,dmuu,&IIu,NULL);CHKERRQ(ierr);
    ierr = DMCreateInterpolationScale(dmuuc,dmuu,IIu,&su);CHKERRQ(ierr);
    ierr = DMCreateRestriction(dmuuc,dmuu,&Ru);CHKERRQ(ierr);
  }

  /* Create and configure solver */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,Auu,Auu);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,NULL);CHKERRQ(ierr);
  ierr = PCMGSetInterpolation(pc,1,IIu);CHKERRQ(ierr);
  /* ierr = PCMGSetRestriction(pc,1,Ru);CHKERRQ(ierr); */
   ierr = PCMGSetRScale(pc,1,su);CHKERRQ(ierr);

  ierr = PCMGGetCoarseSolve(pc,&kspc);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspc,Auuc,Auuc);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  if (analyze) {
    mctx.dm = dmuu;
    mctx.solRef = NULL; /* Reference solution not computed for u-u only */
    mctx.solPrev = NULL; /* Populated automatically */
    ierr = CreateNumericalReferenceSolution(Auu,fu,&mctx.solRefNum);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,DMStagAnalysisKSPMonitor,&mctx,NULL);CHKERRQ(ierr);
  }

  /* Solve */
  ierr = KSPSolve(ksp,fu,xu);CHKERRQ(ierr);

  /* Clean up and finalize PETSc */
  if (analyze) {
    ierr = VecDestroy(&mctx.solPrev);CHKERRQ(ierr);
    ierr = VecDestroy(&mctx.solRef);CHKERRQ(ierr);
    ierr = VecDestroy(&mctx.solRefNum);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dmuu);CHKERRQ(ierr);
  ierr = DMDestroy(&dmuuc);CHKERRQ(ierr);
  ierr = VecDestroy(&su);CHKERRQ(ierr);
  ierr = ISDestroy(&ispc);CHKERRQ(ierr);
  ierr = ISDestroy(&ispf);CHKERRQ(ierr);
  ierr = ISDestroy(&isuc);CHKERRQ(ierr);
  ierr = ISDestroy(&isuf);CHKERRQ(ierr);
  ierr = MatDestroy(&Ru);CHKERRQ(ierr);
  ierr = MatDestroy(&IIu);CHKERRQ(ierr);
  ierr = MatDestroy(&Auuc);CHKERRQ(ierr);
  ierr = MatDestroy(&Auu);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  ierr = VecDestroy(&fu);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSolc);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSol);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
Note: in this system all stencil coefficients which are not related to the Dirichlet boundary are scaled by dv = dx*dy.
This scaling is necessary for multigrid to converge.
*/
static PetscErrorCode CreateSystem(DM dm,Mat *pA,Vec *pRhs)
{
  PetscErrorCode ierr;
  PetscInt       N[2],dof[3];
  PetscBool      isLastRankx,isLastRanky,isFirstRankx,isFirstRanky;
  PetscInt       ex,ey,startx,starty,nx,ny;
  PetscInt       iprev,icenter,inext;
  Mat            A;
  Vec            rhs;
  PetscReal      hx,hy,dv,bogusScale;
  PetscScalar    **cArrX,**cArrY;
  PetscBool      hasPressure;

  PetscFunctionBeginUser;

  /* Determine whether or not to create system including pressure dof (on elements) */
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],&dof[2],NULL);CHKERRQ(ierr);
  if (dof[0] !=0 || dof[1] != 1 || (dof[2] != 1 && dof[2] !=0)) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"CreateSystem only implemented for velocity-only or velocity+pressure grids");
  hasPressure = (PetscBool) (dof[2] == 1);

  bogusScale = 1.0;
  ierr = PetscOptionsGetReal(NULL,NULL,"-bogus",&bogusScale,NULL);CHKERRQ(ierr); /* Use this to break MG (try small values) */

  ierr = DMCreateMatrix(dm,pA);CHKERRQ(ierr);
  A = *pA;
  if (pRhs) {
    ierr = DMCreateGlobalVector(dm,pRhs);CHKERRQ(ierr);
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  ierr = DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetIsLastRank(dm,&isLastRankx,&isLastRanky,NULL);CHKERRQ(ierr);
  ierr = DMStagGetIsFirstRank(dm,&isFirstRankx,&isFirstRanky,NULL);CHKERRQ(ierr);
  hx = 1.0/N[0]; hy = 1.0/N[1];
  dv = hx * hy;
  ierr = DMStagGetProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  /* Loop over all local elements. Note that it may be more efficient in real
     applications to loop over each boundary separately */
  for (ey = starty; ey<starty+ny; ++ey) {
    for (ex = startx; ex<startx+nx; ++ex) {

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;

        const PetscScalar valA = bogusScale * 1.0;
        row.i = ex; row.j = ey; row.loc = RIGHT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = bogusScale * uxRef(cArrX[ex][inext],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = UP; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = uyRef(cArrX[ex][icenter],cArrY[ey][inext]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil     row;
        PetscScalar       valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DOWN; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = uyRef(cArrX[ex][icenter],cArrY[ey][iprev]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;

        row.i    = ex  ; row.j    = ey  ; row.loc    = DOWN;    row.c     = 0;
        if (ex == 0) {
          nEntries = 4;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -dv*1.0 / (hx*hx) -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          /* Missing left element */
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          if (hasPressure) {
            nEntries += 2;
            col[4].i = ex  ; col[4].j = ey-1; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hy;
            col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hy;
          }
        } else if (ex == N[0]-1) {
          /* Right boundary y velocity stencil */
          nEntries = 4;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -dv*1.0 / (hx*hx) -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          /* Missing right element */
          if (hasPressure) {
            nEntries += 2;
            col[4].i = ex  ; col[4].j = ey-1; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hy;
            col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hy;
          }
        } else {
          nEntries = 5;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DOWN;    col[0].c  = 0; valA[0] = -dv*2.0 / (hx*hx) -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DOWN;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DOWN;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DOWN;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = DOWN;    col[4].c  = 0; valA[4] =  dv*1.0 / (hx*hx);
          if (hasPressure) {
            nEntries += 2;
            col[5].i = ex  ; col[5].j = ey-1; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  dv*1.0 / hy;
            col[6].i = ex  ; col[6].j = ey  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -dv*1.0 / hy;
          }
        }
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = dv*fy(cArrX[ex][icenter],cArrY[ey][iprev]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = LEFT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = uxRef(cArrX[ex][iprev],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;
        row.i    = ex  ; row.j    = ey  ; row.loc    = LEFT;    row.c     = 0;

        if (ey == 0) {
          nEntries = 4;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -dv*2.0 /(hx*hx) -dv*1.0 /(hy*hy);
          /* missing term from element below */
          col[1].i = ex  ; col[1].j = ey+1; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  dv*1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          if (hasPressure) {
            nEntries += 2;
            col[4].i = ex-1; col[4].j = ey  ; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hx;
            col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hx;
          }
        } else if (ey == N[1]-1) {
          /* Top boundary x velocity stencil */
          nEntries = 4;
          row.i    = ex  ; row.j    = ey  ; row.loc    = LEFT;    row.c     = 0;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -dv*2.0 / (hx*hx) -dv*1.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          /* Missing element above term */
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  dv*1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          if (hasPressure) {
            nEntries += 2;
            col[4].i = ex-1; col[4].j = ey  ; col[4].loc = ELEMENT; col[4].c  = 0; valA[4] =  dv*1.0 / hx;
            col[5].i = ex  ; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] = -dv*1.0 / hx;
          }
        } else {
          /* Note how this is identical to the stencil for U_y, with "DOWN" replaced by "LEFT" and the pressure derivative in the other direction */
          nEntries = 5;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = LEFT;    col[0].c  = 0; valA[0] = -dv*2.0 / (hx*hx) + -dv*2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = LEFT;    col[1].c  = 0; valA[1] =  dv*1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = LEFT;    col[2].c  = 0; valA[2] =  dv*1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = LEFT;    col[3].c  = 0; valA[3] =  dv*1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = LEFT;    col[4].c  = 0; valA[4] =  dv*1.0 / (hx*hx);
          if (hasPressure) {
            nEntries += 2;
            col[5].i = ex-1; col[5].j = ey  ; col[5].loc = ELEMENT; col[5].c  = 0; valA[5] =  dv*1.0 / hx;
            col[6].i = ex  ; col[6].j = ey  ; col[6].loc = ELEMENT; col[6].c  = 0; valA[6] = -dv*1.0 / hx;
          }

        }
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = dv*fx(cArrX[ex][iprev],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      /* P equation : u_x + v_y = g
         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
      if (hasPressure) {
        DMStagStencil row,col[5];
        PetscScalar   valA[5],valRhs;

        row.i    = ex; row.j    = ey; row.loc    = ELEMENT; row.c    = 0;
        /* Note: the scaling by dv here may not be optimal (but this test isn't concerned with these equations) */
        col[0].i = ex; col[0].j = ey; col[0].loc = LEFT;    col[0].c = 0; valA[0] = -dv*1.0 / hx;
        col[1].i = ex; col[1].j = ey; col[1].loc = RIGHT;   col[1].c = 0; valA[1] =  dv*1.0 / hx;
        col[2].i = ex; col[2].j = ey; col[2].loc = DOWN;    col[2].c = 0; valA[2] = -dv*1.0 / hy;
        col[3].i = ex; col[3].j = ey; col[3].loc = UP;      col[3].c = 0; valA[3] =  dv*1.0 / hy;
        col[4] = row;                                                     valA[4] = dv*0.0;
        ierr = DMStagMatSetValuesStencil(dm,A,1,&row,5,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        if (rhs) {
          valRhs = dv*g(cArrX[ex][icenter],cArrY[ey][icenter]);
          ierr = DMStagVecSetValuesStencil(dm,rhs,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (rhs) {
    ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


/* A custom monitor function for analysis purposes. Computes and dumps
   residuals and errors for each KSP iteration */
PetscErrorCode DMStagAnalysisKSPMonitor(KSP ksp,PetscInt it,PetscReal rnorm,void *mctx)
{
  PetscErrorCode                  ierr;
  DM                              dm;
  Vec                             r,sol;
  DMStagAnalysisKSPMonitorContext *ctx = (DMStagAnalysisKSPMonitorContext*)mctx;

  PetscFunctionBeginUser;
  ierr = KSPBuildSolution(ksp,NULL,&sol);CHKERRQ(ierr); /* don't destroy sol */
  ierr = KSPBuildResidual(ksp,NULL,NULL,&r);CHKERRQ(ierr);
  dm = ctx->dm; /* Would typically get this with VecGetDM(), KSPGetDM() */
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Analaysis monitor requires a DM which is properly associated with the solution Vec");
  ierr = PetscPrintf(PetscObjectComm((PetscObject)dm)," *** DMStag Analysis KSP Monitor (it. %" PetscInt_FMT ") ***\n",it);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)dm)," Residual Norm: %g\n",(double)rnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)dm)," Dumping files...\n");CHKERRQ(ierr);

  /* Note: these blocks are almost entirely duplicated */
  {
    const DMStagStencilLocation loc = DMSTAG_LEFT;
    const PetscInt              c = 0;
    Vec                         vec;
    DM                          da;
    PetscViewer                 viewer;
    char                        filename[PETSC_MAX_PATH_LEN];

    ierr = DMStagVecSplitToDMDA(dm,r,loc,c,&da,&vec);CHKERRQ(ierr);
    ierr = PetscSNPrintf(filename,sizeof(filename),"res_vx_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vec,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
  }

  {
    const DMStagStencilLocation loc = DMSTAG_DOWN;
    const PetscInt              c = 0;
    Vec                         vec;
    DM                          da;
    PetscViewer                 viewer;
    char                        filename[PETSC_MAX_PATH_LEN];

    ierr = DMStagVecSplitToDMDA(dm,r,loc,c,&da,&vec);CHKERRQ(ierr);
    ierr = PetscSNPrintf(filename,sizeof(filename),"res_vy_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(vec,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
  }

  if (ctx->solRef) {
    Vec e;
    ierr = VecDuplicate(ctx->solRef,&e);CHKERRQ(ierr);
    ierr = VecCopy(ctx->solRef,e);CHKERRQ(ierr);
    ierr = VecAXPY(e,-1.0,sol);CHKERRQ(ierr);

    {
      const DMStagStencilLocation loc = DMSTAG_LEFT;
      const PetscInt              c = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      ierr = DMStagVecSplitToDMDA(dm,e,loc,c,&da,&vec);CHKERRQ(ierr);
      ierr = PetscSNPrintf(filename,sizeof(filename),"err_vx_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
      ierr = DMDestroy(&da);CHKERRQ(ierr);
    }

    {
      const DMStagStencilLocation loc = DMSTAG_DOWN;
      const PetscInt              c = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      ierr = DMStagVecSplitToDMDA(dm,e,loc,c,&da,&vec);CHKERRQ(ierr);
      ierr = PetscSNPrintf(filename,sizeof(filename),"err_vy_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
      ierr = DMDestroy(&da);CHKERRQ(ierr);
    }

    ierr = VecDestroy(&e);CHKERRQ(ierr);
  }

  /* Repeat error computations wrt an "exact" solution to the discrete equations */
  if (ctx->solRefNum) {
    Vec e;
    ierr = VecDuplicate(ctx->solRefNum,&e);CHKERRQ(ierr);
    ierr = VecCopy(ctx->solRefNum,e);CHKERRQ(ierr);
    ierr = VecAXPY(e,-1.0,sol);CHKERRQ(ierr);

    {
      const DMStagStencilLocation loc = DMSTAG_LEFT;
      const PetscInt              c = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      ierr = DMStagVecSplitToDMDA(dm,e,loc,c,&da,&vec);CHKERRQ(ierr);
      ierr = PetscSNPrintf(filename,sizeof(filename),"err_num_vx_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
      ierr = DMDestroy(&da);CHKERRQ(ierr);
    }

    {
      const DMStagStencilLocation loc = DMSTAG_DOWN;
      const PetscInt              c = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      ierr = DMStagVecSplitToDMDA(dm,e,loc,c,&da,&vec);CHKERRQ(ierr);
      ierr = PetscSNPrintf(filename,sizeof(filename),"err_num_vy_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
      ierr = DMDestroy(&da);CHKERRQ(ierr);
    }

    ierr = VecDestroy(&e);CHKERRQ(ierr);
  }

  /* Step */
  if (!ctx->solPrev) {
    ierr = VecDuplicate(sol,&ctx->solPrev);CHKERRQ(ierr);
  } else {
    Vec diff;
    ierr = VecDuplicate(sol,&diff);CHKERRQ(ierr);
    ierr = VecCopy(sol,diff);CHKERRQ(ierr);
    ierr = VecAXPY(diff,-1.0,ctx->solPrev);CHKERRQ(ierr);
    {
      const DMStagStencilLocation loc = DMSTAG_LEFT;
      const PetscInt              c = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      ierr = DMStagVecSplitToDMDA(dm,diff,loc,c,&da,&vec);CHKERRQ(ierr);
      ierr = PetscSNPrintf(filename,sizeof(filename),"diff_vx_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
      ierr = DMDestroy(&da);CHKERRQ(ierr);
    }

    {
      const DMStagStencilLocation loc = DMSTAG_DOWN;
      const PetscInt              c = 0;
      Vec                         vec;
      DM                          da;
      PetscViewer                 viewer;
      char                        filename[PETSC_MAX_PATH_LEN];

      ierr = DMStagVecSplitToDMDA(dm,diff,loc,c,&da,&vec);CHKERRQ(ierr);
      ierr = PetscSNPrintf(filename,sizeof(filename),"diff_vy_%" PetscInt_FMT ".vtr",it);CHKERRQ(ierr);
      ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)r),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      ierr = VecView(vec,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
      ierr = DMDestroy(&da);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&diff);CHKERRQ(ierr);
  }
  ierr = VecCopy(sol,ctx->solPrev);CHKERRQ(ierr);

  ierr = VecDestroy(&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Use a direct solver to create an "exact" solution to the discrete system
   useful for testing solvers (in that it doesn't include discretization error) */
static PetscErrorCode CreateNumericalReferenceSolution(Mat A,Vec rhs,Vec *px)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  Vec            x;

  PetscFunctionBeginUser;
  ierr = KSPCreate(PetscObjectComm((PetscObject)A),&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverType(pc,MATSOLVERUMFPACK);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp,"numref_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = VecDuplicate(rhs,px);CHKERRQ(ierr);
  x = *px;
  ierr = KSPSolve(ksp,rhs,x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: gmg_1
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason

   test:
      suffix: gmg_1_b
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractTransferOperators false

   test:
      suffix: gmg_1_c
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false

   test:
      suffix: gmg_1_d
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false -extractTransferOperators false

   test:
      suffix: gmg_1_bigger
      nsize: 1
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -stag_grid_x 32 -stag_grid_y 32

   test:
      suffix: gmg_8
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason

   test:
      suffix: gmg_8_b
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractTransferOperators false

   test:
      suffix: gmg_8_c
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false

   test:
      suffix: gmg_8_d
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false -extractTransferOperators false

   test:
      suffix: gmg_8_galerkin
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -extractSystem false -extractTransferOperators false -pc_mg_galerkin

   test:
      suffix: gmg_8_bigger
      nsize: 8
      args: -ksp_type fgmres -mg_levels_pc_type jacobi -ksp_converged_reason -stag_grid_x 32 -stag_grid_y 32

TEST*/
