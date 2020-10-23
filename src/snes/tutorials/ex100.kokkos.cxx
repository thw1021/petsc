static char help[] = "Newton methods to solve u'' + u^{2} = f in parallel.\n";

/*T
   Concepts: SNES^basic parallel example
   Concepts: SNES^setting a user-defined monitoring routine
   Processors: n
T*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
   User-defined application context
*/
typedef struct {
  DM          da;      /* distributed array */
  Vec         f;       /* right-hand-side of PDE */
  PetscReal   h;       /* mesh spacing */
} ApplicationCtx;

/* ------------------------------------------------------------------- */
/*
   computeInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode computeInitialGuess(Vec x,ApplicationCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    pfive = .50;

  PetscFunctionBeginUser;
  ierr = VecSet(x,pfive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   computeInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode computeForcing(Vec f,ApplicationCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    xp,*F;
  PetscInt       i,M,xs,xm;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(user->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayWrite(user->da,f,&F);CHKERRQ(ierr);

  /*
     Compute local vector entries
  */
  xp = user->h*xs;
  for (i=xs; i<xs+xm; i++) {
    F[i] = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    xp   += user->h;
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayWrite(user->da,f,&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   computeInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode computeExactSolution(Vec x,ApplicationCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    xp,*X;
  PetscInt       i,M,xs,xm;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(user->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayWrite(user->da,x,&X);CHKERRQ(ierr);

  /*
     Compute local vector entries
  */
  xp = user->h*xs;
  for (i=xs; i<xs+xm; i++) {
    X[i] = xp*xp*xp;
    xp   += user->h;
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayWrite(user->da,x,&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   computeFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  f - function vector

   Note:
   The user-defined context can contain any application-specific
   data needed for the function evaluation.
*/
PetscErrorCode computeFunction(SNES snes,Vec x,Vec r,void *ctx)
{
  ApplicationCtx     *user = (ApplicationCtx*) ctx;
  DM                 da = user->da;
  const PetscScalar  *X,*F;
  PetscScalar        *R,d;
  PetscErrorCode     ierr;
  PetscInt           i,M,xs,xm;
  Vec                xlocal;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(da,&xlocal);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can
     be done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);

  /*
     Get pointers to vector data.
       - The vector xlocal includes ghost point; the vectors x and f do
         NOT include ghost points.
       - Using DMDAVecGetArray() allows accessing the values using global ordering
  */
  ierr = DMDAVecGetArrayRead(da,xlocal,&X);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayWrite(da,r,&R);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,user->f,&F);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Set function values for boundary points; define local interior grid point range:
        xsi - starting interior grid index
        xei - ending interior grid index
  */
  if (xs == 0) { /* left boundary */
    R[0] = X[0];
    xs++;xm--;
  }
  if (xs+xm == M) {  /* right boundary */
    R[xs+xm-1] = X[xs+xm-1] - 1.0;
    xm--;
  }

  /*
     Compute function over locally owned part of the grid (interior points only)
  */
  d = 1.0/(user->h*user->h);
  for (i=xs; i<xs+xm; i++) R[i] = d*(X[i-1] - 2.0*X[i] + X[i+1]) + X[i]*X[i] - F[i];

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,xlocal,&X);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayWrite(da,r,&R);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,user->f,&F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

PetscErrorCode KokkosComputeFunction(SNES snes,Vec x,Vec r,void *ctx)
{
  ApplicationCtx    *user = (ApplicationCtx*) ctx;
  DM                da    = user->da;
  PetscScalar       d;
  PetscErrorCode    ierr;
  PetscInt          M,xs,xm,gxs,gxm;
  Vec               xlocal;
  PetscScalar       *Rk;
  const PetscScalar *Xk,*Fk;


  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(da,&xlocal);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can
     be done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,NULL,NULL,&gxm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = VecGetArrayRead(xlocal,&Xk);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(r,&Rk);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->f,&Fk);CHKERRQ(ierr);

  Kokkos::Experimental::OffsetView<const PetscScalar*> F(Kokkos::View<const PetscScalar*>(Fk,xm),{xs});
  Kokkos::Experimental::OffsetView<PetscScalar*> R(Kokkos::View<PetscScalar*>(Rk,xm),{xs});
  Kokkos::Experimental::OffsetView<const PetscScalar*> X(Kokkos::View<const PetscScalar*>(Xk,gxm),{gxs});

  /*
     Set function values for boundary points; define local interior grid point range:
        xsi - starting interior grid index
        xei - ending interior grid index
  */
  if (xs == 0) { /* left boundary */
    R[0] = X[0];
    xs++;xm--;
  }
  if (xs+xm == M) {  /* right boundary */
    R[xs+xm-1] = X[xs+xm-1] - 1.0;
    xm--;
  }

  d = 1.0/(user->h*user->h);
  Kokkos:: parallel_for (Kokkos::RangePolicy<> (xs,xm+1), KOKKOS_LAMBDA ( int i) {
    R[i] = d*(X[i-1] - 2.0*X[i] + X[i+1]) + X[i]*X[i] - F[i];
  });

  ierr = VecRestoreArrayRead(xlocal,&Xk);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(r,&Rk);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->f,&Fk);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   computeJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode computeJacobian(SNES snes,Vec x,Mat jac,Mat B,void *ctx)
{
  ApplicationCtx    *user = (ApplicationCtx*) ctx;
  const PetscScalar *X;
  PetscScalar       d;
  PetscErrorCode    ierr;
  PetscInt          i,M,xs,xm;
  DM                da = user->da;

  PetscFunctionBeginUser;

  /*
     Get pointer to vector data
  */
  ierr = DMDAVecGetArrayRead(da,x,&X);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
    Get range of locally owned matrix
  */
  ierr = DMDAGetInfo(user->da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Determine starting and ending local indices for interior grid points.
     Set Jacobian entries for boundary points.
  */

  if (xs == 0) {  /* left boundary */
    PetscInt i = 0;
    PetscScalar A = 1.0;

    ierr = MatSetValues(jac,1,&i,1,&i,&A,INSERT_VALUES);CHKERRQ(ierr);
    xs++;xm--;
  }
  if (xs+xm == M) { /* right boundary */
    PetscInt i    = M-1;
    PetscScalar A = 1.0;
    ierr = MatSetValues(jac,1,&i,1,&i,&A,INSERT_VALUES);CHKERRQ(ierr);
    xm--;
  }

  /*
     Interior grid points
      - Note that in this case we set all elements for a particular
        row at once.
  */
  d = 1.0/(user->h*user->h);
  for (i=xs; i<xs+xm; i++) {
    PetscInt j[] = {i - 1,i,i + 1};
    PetscScalar A[] = {d, -2.0*d + 2.0*X[i],d};
    ierr = MatSetValues(jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.

     Also, restore vector.
  */

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,x,&X);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#include "ex100.h"


#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

PetscErrorCode MatGetDeviceCSRWrite_Kokkos_Seq(Mat A, PetscSplitCSRDataStructure **B)
{
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  A->was_assembled = PETSC_TRUE;
  if (!A->spptr) {
    PetscSplitCSRDataStructure  *mat;
    Mat_SeqAIJ                  *aij = (Mat_SeqAIJ*)A->data;

    ierr = PetscNew(&mat);CHKERRQ(ierr);
    mat->diag.i = aij->i;
    mat->diag.j = aij->j;
    mat->diag.a = aij->a;
    mat->rstart = 0; mat->rend = A->rmap->n;
    mat->cstart = 0; mat->cend = A->cmap->n;
    mat->diag.ignorezeroentries = aij->ignorezeroentries;
    A->spptr = *B = mat;
  } else *B = (PetscSplitCSRDataStructure *)A->spptr;
  A->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDeviceCSRWrite(Mat A, PetscSplitCSRDataStructure **B)
{
  return MatGetDeviceCSRWrite_Kokkos_Seq(A, B);
}

PetscErrorCode MatRestoreDeviceCSRWrite(Mat A, PetscSplitCSRDataStructure **B)
{
  return 0;
}

PetscErrorCode KokkosComputeJacobian(SNES snes,Vec x,Mat jac,Mat B,void *ctx)
{
  ApplicationCtx             *user = (ApplicationCtx*) ctx;
  const PetscScalar          *Xk;
  PetscScalar                d;
  PetscErrorCode             ierr;
  PetscInt                   M,xs,xm,gxs,gxm;
  DM                         da = user->da;
  PetscSplitCSRDataStructure *csrmat;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,NULL,NULL,&gxm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = VecGetArrayRead(x,&Xk);CHKERRQ(ierr);
  Kokkos::Experimental::OffsetView<const PetscScalar*> X(Kokkos::View<const PetscScalar*>(Xk,gxm),{gxs});

  if (xs == 0) {  /* left boundary */
    PetscInt i = 0; PetscScalar A = 1.0;

    ierr = MatSetValues(jac,1,&i,1,&i,&A,INSERT_VALUES);CHKERRQ(ierr);
    xs++;xm--;
  }
  if (xs+xm == M) { /* right boundary */
    PetscInt i = M-1;
    PetscScalar A = 1.0;
    ierr = MatSetValues(jac,1,&i,1,&i,&A,INSERT_VALUES);CHKERRQ(ierr);
    xm--;
  }

  d = 1.0/(user->h*user->h);
  csrmat = NULL;
  ierr = MatGetDeviceCSRWrite(jac,&csrmat);CHKERRQ(ierr);

  Kokkos:: parallel_for (Kokkos::RangePolicy<> (xs,xm+1), KOKKOS_LAMBDA (int i)
  {
    PetscInt j[] = {i-1,i,i+1};
    PetscScalar A[] = {d, -2.0*d + 2.0*X[i],d};

    MatSetValuesKokkos(csrmat,1,&i,3,j,A,INSERT_VALUES);
  });

  ierr = MatRestoreDeviceCSRWrite(jac,&csrmat);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&Xk);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  SNES           snes;
  Mat            J;
  ApplicationCtx user;               /* user-defined context */
  Vec            x,r,u;              /* vectors */
  PetscScalar    none = -1.0;
  PetscErrorCode ierr;
  PetscInt       N = 5;
  PetscReal      norm;
  PetscBool      useKokkos = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr  = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsGetBool(NULL,NULL,"-use_Kokkos",&useKokkos,NULL);CHKERRQ(ierr);
  user.h = 1.0/(N-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,1,1,NULL,&user.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da);CHKERRQ(ierr);
  ierr = DMSetUp(user.da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&user.f);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateMatrix(user.da,&J);CHKERRQ(ierr);

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.  Whenever the nonlinear solver needs to compute the
     Jacobian matrix, it will call this routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        Jacobian evaluation routine.
  */
  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  if (useKokkos) {
    ierr = SNESSetFunction(snes,r,KokkosComputeFunction,&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,KokkosComputeJacobian,&user);CHKERRQ(ierr);
  } else {
    ierr = SNESSetFunction(snes,r,computeFunction,&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,computeJacobian,&user);CHKERRQ(ierr);
  }

  /*
     Set names for some vectors to facilitate monitoring (optional)
  */
  ierr = PetscObjectSetName((PetscObject)x,"Approximate Solution");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"Exact Solution");CHKERRQ(ierr);

  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = computeForcing(user.f,&user);CHKERRQ(ierr);
  ierr = computeExactSolution(u,&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = computeInitialGuess(x,&user);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  ierr = VecAXPY(x,none,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g\n",(double)norm);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&user.f);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: kokkos

   test:
     args: -snes_monitor

   test:
      suffix: 2
      nsize: 3
      args: -snes_monitor

TEST*/
