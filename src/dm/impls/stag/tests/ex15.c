static char help[] = "Test DMStag default MG components, on a Stokes-style system.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscksp.h>

PetscErrorCode CreateSystem(DM dm,Mat *A,Vec *b);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim;
  PetscBool      flg,dump;
  KSP            ksp;
  Mat            A;
  Vec            x,b;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option\n");CHKERRQ(ierr);
    return 1;
  }
  if (dim == 1) {
    ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,1,1,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,PETSC_DECIDE,PETSC_DECIDE,0,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
  } else if (dim == 3) {
    ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,4,4,4,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,0,0,1,1,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n");CHKERRQ(ierr);
      return 1;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  /* Flags to dump binary or ASCII output */
  dump = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump",&dump,NULL);CHKERRQ(ierr);

  /* Directly create a coarsened DM and transfer operators */
  {
    DM dmCoarse;
    ierr = DMCoarsen(dm,MPI_COMM_NULL,&dmCoarse);CHKERRQ(ierr);
    {
      Mat Ai;
      ierr = DMCreateInterpolation(dmCoarse,dm,&Ai,NULL);CHKERRQ(ierr);
      if (dump) {
        PetscViewer viewer;
        ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matI.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = MatView(Ai,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&Ai);CHKERRQ(ierr);
    }
    {
      Mat Ar;
      ierr = DMCreateRestriction(dmCoarse,dm,&Ar);CHKERRQ(ierr);
      if (dump) {
        PetscViewer viewer;
        ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matR.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = MatView(Ar,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&Ar);CHKERRQ(ierr);
    }
    ierr = DMDestroy(&dmCoarse);CHKERRQ(ierr);
  }

  /* Create and solve a system */
  ierr = CreateSystem(dm,&A,&b);CHKERRQ(ierr);
  if (dump) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"matA.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (dump) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"vecb.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(b,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = KSPCreate(PetscObjectComm((PetscObject)dm),&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  if (dump) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)dm),"vecx.pbin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(x,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* Note: unlike in the 2D case, this does not include reasonable scaling and so will not work well */
PetscErrorCode CreateSystem1d(DM dm,Mat *A,Vec *b)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm,A);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,b);CHKERRQ(ierr);
  if (dim == 1) {
    PetscInt    e,start,n,N;
    PetscBool   isFirstRank,isLastRank;
    PetscScalar h;
    ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DMStagGetGlobalSizes(dm,&N,NULL,NULL);CHKERRQ(ierr);
    h = 1.0/N;
    ierr = DMStagGetIsFirstRank(dm,&isFirstRank,NULL,NULL);CHKERRQ(ierr);
    ierr = DMStagGetIsLastRank(dm,&isLastRank,NULL,NULL);CHKERRQ(ierr);
    for (e = start; e<start+n; ++e) {
      DMStagStencil pos[3];
      PetscScalar   val[3];
      PetscInt      idxLoc;

      if (isFirstRank && e == start) {
        /* Fix first pressure node to eliminate nullspace */
        idxLoc = 0;
        pos[idxLoc].i   = e;
        pos[idxLoc].loc = DMSTAG_ELEMENT;
        pos[idxLoc].c   = 0;
        val[idxLoc]     = 1.0; /* 0 pressure forcing term (physical) */
        ++idxLoc;
      } else {
        idxLoc = 0;
        pos[idxLoc].i   = e;
        pos[idxLoc].loc = DMSTAG_ELEMENT;
        pos[idxLoc].c   = 0;
        val[idxLoc]     = 0.0; /* 0 pressure forcing term (physical) */
        ++idxLoc;
      }

      if (isFirstRank && e == start) {
        pos[idxLoc].i   = e;
        pos[idxLoc].loc = DMSTAG_LEFT;
        pos[idxLoc].c   = 0;
        val[idxLoc]     = 3.0; /* fixed left BC */
        ++idxLoc;
      } else {
        pos[idxLoc].i   = e;
        pos[idxLoc].loc = DMSTAG_LEFT;
        pos[idxLoc].c   = 0;
        val[idxLoc]     = 1.0; /* constant forcing */
        ++idxLoc;
      }
      if (isLastRank && e == start+n-1) {
        /* Special case on right boundary (in addition to usual case) */
        pos[idxLoc].i   = e;     /* This element in the 1d ordering */
        pos[idxLoc].loc = DMSTAG_RIGHT;
        pos[idxLoc].c   = 0;
        val[idxLoc]     = 3.0; /* fixed right BC */
        ++idxLoc;
      }
      ierr = DMStagVecSetValuesStencil(dm,*b,idxLoc,pos,val,INSERT_VALUES);CHKERRQ(ierr);
    }

    for (e = start; e<start+n; ++e) {
      if (isFirstRank && e == start) {
        DMStagStencil row;
        PetscScalar   val;

        row.i   = e;
        row.loc = DMSTAG_LEFT;
        row.c   = 0;
        val     = 1.0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        DMStagStencil row,col[5];
        PetscScalar   val[5];

        row.i      = e;
        row.loc    = DMSTAG_LEFT;
        row.c      = 0;

        col[0].i   = e;
        col[0].loc = DMSTAG_ELEMENT;
        col[0].c   = 0;

        col[1].i   = e-1;
        col[1].loc = DMSTAG_ELEMENT;
        col[1].c   = 0;

        val[0]     = -1.0/h;
        val[1]     =  1.0/h;

        col[2].i   = e;
        col[2].loc = DMSTAG_LEFT;
        col[2].c   = 0;
        val[2]     = -2.0/(h*h);

        col[3].i   = e-1;
        col[3].loc = DMSTAG_LEFT;
        col[3].c   = 0;
        val[3]     = 1.0/(h*h);

        col[4].i   = e+1;
        col[4].loc = DMSTAG_LEFT;
        col[4].c   = 0;
        val[4]     = 1.0/(h*h);

        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,5,col,val,INSERT_VALUES);CHKERRQ(ierr);
      }

      /* Additional velocity point (BC) on the right */
      if (isLastRank && e == start+n-1) {
        DMStagStencil row;
        PetscScalar   val;

        row.i = e;
        row.loc = DMSTAG_RIGHT;
        row.c = 0;
        val = 1.0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
      }

      /* Equation on pressure (element) variables */
      if (isFirstRank && e == 0) {
        /* Fix first pressure node to eliminate nullspace */
        DMStagStencil row;
        PetscScalar   val;

        row.i      = e;
        row.loc    = DMSTAG_ELEMENT;
        row.c      = 0;
        val        = 1.0;

        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        DMStagStencil row,col[2];
        PetscScalar   val[2];

        row.i      = e;
        row.loc    = DMSTAG_ELEMENT;
        row.c      = 0;

        col[0].i = e;
        col[0].loc = DMSTAG_LEFT;
        col[0].c   = 0;

        col[1].i = e;
        col[1].loc = DMSTAG_RIGHT;
        col[1].c   = 0;

        val[0]     = -1.0/h;
        val[1]     = 1.0/h;

        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSystem2d(DM dm,Mat *A,Vec *b)
{
  PetscErrorCode ierr;
  PetscInt       N[2];
  PetscBool      isLastRankx,isLastRanky,isFirstRankx,isFirstRanky;
  PetscInt       ex,ey,startx,starty,nx,ny;
  PetscReal      hx,hy,dv;

  PetscFunctionBeginUser;
  ierr = DMCreateMatrix(dm,A);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,b);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N[0],&N[1],NULL);CHKERRQ(ierr);
  ierr = DMStagGetIsLastRank(dm,&isLastRankx,&isLastRanky,NULL);CHKERRQ(ierr);
  ierr = DMStagGetIsFirstRank(dm,&isFirstRankx,&isFirstRanky,NULL);CHKERRQ(ierr);
  hx = 1.0/N[0]; hy = 1.0/N[1];
  dv = hx*hy;

  for (ey = starty; ey<starty+ny; ++ey) {
    for (ex = startx; ex<startx+nx; ++ex) {

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_RIGHT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = 0.0; /* zero Dirichlet */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }
      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_UP; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = 0.0; /* zero Diri */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_DOWN; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = 0.0; /* zero Diri */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;

        row.i    = ex  ; row.j    = ey  ; row.loc    = DMSTAG_DOWN;    row.c     = 0;
        if (ex == 0) {
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_DOWN;    col[0].c  = 0; valA[0] = -dv * 1.0 / (hx*hx) -dv * 2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_DOWN;    col[1].c  = 0; valA[1] =  dv * 1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_DOWN;    col[2].c  = 0; valA[2] =  dv * 1.0 / (hy*hy);
          /* Missing left element */
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DMSTAG_DOWN;    col[3].c  = 0; valA[3] =  dv * 1.0 / (hx*hx);
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv * 1.0 / hy;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv * 1.0 / hy;
        } else if (ex == N[0]-1) {
          /* Right boundary y velocity stencil */
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_DOWN;    col[0].c  = 0; valA[0] = -dv * 1.0 / (hx*hx) - dv * 2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_DOWN;    col[1].c  = 0; valA[1] =  dv * 1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_DOWN;    col[2].c  = 0; valA[2] =  dv * 1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DMSTAG_DOWN;    col[3].c  = 0; valA[3] =  dv * 1.0 / (hx*hx);
          /* Missing right element */
          col[4].i = ex  ; col[4].j = ey-1; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv * 1.0 / hy;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv * 1.0 / hy;
        } else {
          nEntries = 7;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_DOWN;    col[0].c  = 0; valA[0] = -dv * 2.0 / (hx*hx) -dv * 2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_DOWN;    col[1].c  = 0; valA[1] =  dv * 1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_DOWN;    col[2].c  = 0; valA[2] =  dv * 1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DMSTAG_DOWN;    col[3].c  = 0; valA[3] =  dv * 1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = DMSTAG_DOWN;    col[4].c  = 0; valA[4] =  dv * 1.0 / (hx*hx);
          col[5].i = ex  ; col[5].j = ey-1; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] =  dv * 1.0 / hy;
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = DMSTAG_ELEMENT; col[6].c  = 0; valA[6] = -dv * 1.0 / hy;
        }
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = dv * 1.0; /* non-zero */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }

      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        PetscScalar   valRhs;
        const PetscScalar valA = 1.0;
        row.i = ex; row.j = ey; row.loc = DMSTAG_LEFT; row.c = 0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = 0.0; /* zero Diri */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x */
        DMStagStencil row,col[7];
        PetscScalar   valA[7],valRhs;
        PetscInt      nEntries;
        row.i    = ex  ; row.j    = ey  ; row.loc    = DMSTAG_LEFT;    row.c     = 0;

        if (ey == 0) {
          nEntries = 6;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_LEFT;    col[0].c  = 0; valA[0] = -dv * 2.0 /(hx*hx) -dv * 1.0 /(hy*hy);
          /* missing term from element below */
          col[1].i = ex  ; col[1].j = ey+1; col[1].loc = DMSTAG_LEFT;    col[1].c  = 0; valA[1] =  dv * 1.0 / (hy*hy);
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = DMSTAG_LEFT;    col[2].c  = 0; valA[2] =  dv * 1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DMSTAG_LEFT;    col[3].c  = 0; valA[3] =  dv * 1.0 / (hx*hx);
          col[4].i = ex-1; col[4].j = ey  ; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv * 1.0 / hx;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv * 1.0 / hx;
        } else if (ey == N[1]-1) {
          /* Top boundary x velocity stencil */
          nEntries = 6;
          row.i    = ex  ; row.j    = ey  ; row.loc    = DMSTAG_LEFT;    row.c     = 0;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_LEFT;    col[0].c  = 0; valA[0] = -dv * 2.0 / (hx*hx) -dv * 1.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_LEFT;    col[1].c  = 0; valA[1] =  dv * 1.0 / (hy*hy);
          /* Missing element above term */
          col[2].i = ex-1; col[2].j = ey  ; col[2].loc = DMSTAG_LEFT;    col[2].c  = 0; valA[2] =  dv * 1.0 / (hx*hx);
          col[3].i = ex+1; col[3].j = ey  ; col[3].loc = DMSTAG_LEFT;    col[3].c  = 0; valA[3] =  dv * 1.0 / (hx*hx);
          col[4].i = ex-1; col[4].j = ey  ; col[4].loc = DMSTAG_ELEMENT; col[4].c  = 0; valA[4] =  dv * 1.0 / hx;
          col[5].i = ex  ; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] = -dv * 1.0 / hx;
        } else {
          nEntries = 7;
          col[0].i = ex  ; col[0].j = ey  ; col[0].loc = DMSTAG_LEFT;    col[0].c  = 0; valA[0] = -dv * 2.0 / (hx*hx) + -dv * 2.0 / (hy*hy);
          col[1].i = ex  ; col[1].j = ey-1; col[1].loc = DMSTAG_LEFT;    col[1].c  = 0; valA[1] =  dv * 1.0 / (hy*hy);
          col[2].i = ex  ; col[2].j = ey+1; col[2].loc = DMSTAG_LEFT;    col[2].c  = 0; valA[2] =  dv * 1.0 / (hy*hy);
          col[3].i = ex-1; col[3].j = ey  ; col[3].loc = DMSTAG_LEFT;    col[3].c  = 0; valA[3] =  dv * 1.0 / (hx*hx);
          col[4].i = ex+1; col[4].j = ey  ; col[4].loc = DMSTAG_LEFT;    col[4].c  = 0; valA[4] =  dv * 1.0 / (hx*hx);
          col[5].i = ex-1; col[5].j = ey  ; col[5].loc = DMSTAG_ELEMENT; col[5].c  = 0; valA[5] =  dv * 1.0 / hx;
          col[6].i = ex  ; col[6].j = ey  ; col[6].loc = DMSTAG_ELEMENT; col[6].c  = 0; valA[6] = -dv * 1.0 / hx;

        }
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,nEntries,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = dv * 0.0; /* zero */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }

      /* P equation : u_x + v_y = g
         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace) */
      if (ex == 0 && ey == 0) { /* Pin the first pressure node */
        DMStagStencil row;
        PetscScalar valA,valRhs;
        row.i = ex; row.j = ey; row.loc  = DMSTAG_ELEMENT; row.c = 0;
        valA = 1.0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,1,&row,&valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = 0.0; /* zero pinned pressure */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        DMStagStencil row,col[5];
        PetscScalar   valA[5],valRhs;

        /* Note: the scaling by dv here is not principled and likely suboptimal */
        row.i    = ex; row.j    = ey; row.loc    = DMSTAG_ELEMENT; row.c    = 0;
        col[0].i = ex; col[0].j = ey; col[0].loc = DMSTAG_LEFT;    col[0].c = 0; valA[0] = -dv * 1.0 / hx;
        col[1].i = ex; col[1].j = ey; col[1].loc = DMSTAG_RIGHT;   col[1].c = 0; valA[1] =  dv * 1.0 / hx;
        col[2].i = ex; col[2].j = ey; col[2].loc = DMSTAG_DOWN;    col[2].c = 0; valA[2] = -dv * 1.0 / hy;
        col[3].i = ex; col[3].j = ey; col[3].loc = DMSTAG_UP;      col[3].c = 0; valA[3] =  dv * 1.0 / hy;
        col[4] = row;                                                            valA[4] = 0.0;
        ierr = DMStagMatSetValuesStencil(dm,*A,1,&row,5,col,valA,INSERT_VALUES);CHKERRQ(ierr);
        valRhs = dv * 0.0; /* zero pressure forcing */
        ierr = DMStagVecSetValuesStencil(dm,*b,1,&row,&valRhs,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(*b);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSystem(DM dm,Mat *A,Vec *b)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim == 1) {
    ierr = CreateSystem1d(dm,A,b);CHKERRQ(ierr);
  } else if (dim == 2) {
    ierr = CreateSystem2d(dm,A,b);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1d_directsmooth_seq
      nsize: 1
      requires: suitesparse
      args: -dim 1 -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_levels_pc_type lu -mg_levels_pc_factor_mat_solver_type umfpack -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack -ksp_converged_reason

   test:
      suffix: 1d_directsmooth_par
      nsize: 4
      requires: mumps
      args: -dim 1 /ex15 -dim 1 -stag_grid_x 16 -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_levels_pc_type lu -mg_levels_pc_factor_mat_solver_type mumps -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type mumps -ksp_converged_reason

   test:
      suffix: 1d_fssmooth_seq
      nsize: 1
      requires: suitesparse
      args: -dim 1 -stag_grid_x 256 -ksp_converged_reason -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack -mg_levels_pc_type fieldsplit -mg_levels_pc_fieldsplit_detect_saddle_point

   test:
      suffix: 1d_fssmooth_par
      nsize: 1
      requires: mumps
      args: -dim 1 -stag_grid_x 256 -ksp_converged_reason -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type mumps -mg_levels_pc_type fieldsplit -mg_levels_pc_fieldsplit_detect_saddle_point

   test:
      suffix: 2d_directsmooth_seq
      nsize: 1
      requires: suitesparse
      args: -dim 2 -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_levels_pc_type lu -mg_levels_pc_factor_mat_solver_type umfpack -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack -ksp_converged_reason

   test:
      suffix: 2d_directsmooth_par
      nsize: 4
      requires: mumps
      args: -dim 1 /ex15 -dim 1 -stag_grid_x 16 -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_levels_pc_type lu -mg_levels_pc_factor_mat_solver_type mumps -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type mumps -ksp_converged_reason

   test:
      suffix: 2d_fssmooth_seq
      nsize: 1
      requires: suitesparse
      args: -dim 2 -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_levels_pc_type fieldsplit -mg_levels_pc_fieldsplit_detect_saddle_point -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack -ksp_converged_reason

   test:
      suffix: 2d_fssmooth_par
      nsize: 1
      requires: mumps
      args: -dim 2 -ksp_type fgmres -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin -mg_levels_pc_type fieldsplit -mg_levels_pc_fieldsplit_detect_saddle_point -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type mumps -ksp_converged_reason

   test:
      suffix: 2d_mono_mg
      nsize: 1
      requires: suitesparse
      args: -dim 2 -pc_type mg -pc_mg_galerkin -mg_levels_pc_type fieldsplit -mg_levels_pc_fieldsplit_type SCHUR -ksp_monitor_true_residual -ksp_converged_reason -mg_levels_fieldsplit_element_pc_type jacobi -mg_levels_fieldsplit_face_pc_type jacobi -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack -pc_mg_levels 3 -mg_levels_pc_fieldsplit_schur_fact_type UPPER  -mg_levels_fieldsplit_face_pc_type sor -mg_levels_fieldsplit_face_ksp_type richardson -mg_levels_fieldsplit_face_pc_sor_symmetric -mg_levels_fieldsplit_element_ksp_type richardson  -mg_levels_fieldsplit_element_pc_type none -ksp_type fgmres -stag_grid_x 48 -stag_grid_y 48 -mg_levels_fieldsplit_face_ksp_max_it 1 -mg_levels_ksp_max_it 4 -mg_levels_fieldsplit_element_ksp_type preonly -mg_levels_fieldsplit_element_pc_type none -pc_mg_cycle_type w -mg_levels_ksp_max_it 4 -mg_levels_2_ksp_max_it 8

TEST*/
