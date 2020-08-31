static char help[] =
"Simple ODE system to test implicit and IMEX methods when *both* RHSFunction()\n"
"and IFunction() and *both* RHSJacobian() and IJacobian() are supplied.  The\n"
"problem has form\n"
"   F(t,u,dudt) = G(t,u)\n"
"where  u(t) = (x(t),y(t))  and dF/d(dudt) is invertible, so the problem is a\n"
"2D ODE IVP.  Functions F (IFunction) and G (RHSFunction) are actually linear,\n"
"but the ProblemType is set to TS_NONLINEAR type.  By default the system is\n"
"   x' + y' +  x + 2y =  x + 8y\n"
"        y' + 3x + 4y = 4x + 4y\n"
"The implicit methods like BEULER, THETA, CN, BDF should be able to handle\n"
"this system using the Jacobians.  Apparently ROSW can also handle it.\n"
"However, ARKIMEX and EIMEX require dF/d(dudt) to be the identity.\n"
"(But consider option -ts_arkimex_fully_implicit.)  If option -identity_in_F\n"
"is given then an equivalent system is formed,\n"
"   x'      +  x + 2y = 8y\n"
"        y' + 3x + 4y = 4x + 4y\n"
"(Both of these systems are trivial transformations of the system\n"
"x'=6y-x, y'=x.)  With the initial conditions chosen below, x(0)=1 and y(0)=3,\n"
"the exact solution to the above systems is x(t) = -3 e^{-3t} + 4 e^{2t},\n"
"y(t) = e^{-3t} + 2 e^{2t}, and we compute the final numerical error\n"
"accordingly.  Default type is BDF.  See the manual pages for the various\n"
"methods (e.g. TSROSW, TSARKIMEX, TSEIMEX, ...) for further information.\n\n";

#include <petsc.h>

typedef struct {
  PetscBool  identity_in_F;
} Ctx;

extern PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec F,void*);
extern PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            u, uexact;
  Mat            JF,JG;
  Ctx            user;
  PetscMPIInt    size;
  PetscReal      tf,xf,yf,errnorm;
  PetscInt       steps;

  ierr = PetscInitialize(&argc,&argv,NULL,help);  if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Uniprocessor example only!");

  user.identity_in_F = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Simple F(t,u,u')=G(t,u) ODE system.","TS");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-identity_in_F","set up system so the dF/d(dudt) = I","ex54.c",user.identity_in_F,&(user.identity_in_F),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,2); CHKERRQ(ierr);
  ierr = VecSetFromOptions(u); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&JF); CHKERRQ(ierr);
  ierr = MatSetSizes(JF,PETSC_DECIDE,PETSC_DECIDE,2,2); CHKERRQ(ierr);
  ierr = MatSetFromOptions(JF); CHKERRQ(ierr);
  ierr = MatSetUp(JF); CHKERRQ(ierr);
  ierr = MatDuplicate(JF,MAT_DO_NOT_COPY_VALUES,&JG); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,&user); CHKERRQ(ierr);

  ierr = TSSetIFunction(ts,NULL,FormIFunction,&user); CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,JF,JF,FormIJacobian,&user); CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user); CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,JG,JG,FormRHSJacobian,&user); CHKERRQ(ierr);

  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBDF); CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = VecSetValue(u,0,1.0,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecSetValue(u,1,3.0,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u); CHKERRQ(ierr);
  ierr = TSSolve(ts,u); CHKERRQ(ierr);

  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
  xf = -3.0 * PetscExpReal(-3.0*tf) + 4.0 * PetscExpReal(2.0*tf);
  yf = PetscExpReal(-3.0*tf) + 2.0 * PetscExpReal(2.0*tf);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
  ierr = VecSetValue(uexact,0,xf,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecSetValue(uexact,1,yf,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr); // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_2,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"error norm at tf = %.6f from %D steps:  |u-u_exact| =  %.5e\n",(double)tf,steps,(double)errnorm);CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  MatDestroy(&JF);  MatDestroy(&JG);
  TSDestroy(&ts);
  return PetscFinalize();
}


PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec u, Vec dudt,Vec F, void *user)
{
    PetscErrorCode  ierr;
    const PetscReal *au, *adudt;
    PetscReal       *aF;
    PetscBool       flag = ((Ctx*)user)->identity_in_F;

    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArrayRead(dudt,&adudt); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF);
    if (flag) {
      aF[0] = adudt[0] + au[0] + 2.0 * au[1];
      aF[1] = adudt[1] + 3.0 * au[0] + 4.0 * au[1];
    } else {
      aF[0] = adudt[0] + adudt[1] + au[0] + 2.0 * au[1];
      aF[1] = adudt[1] + 3.0 * au[0] + 4.0 * au[1];
    }
    ierr = VecRestoreArrayRead(u,&au);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(dudt,&adudt);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
    return 0;
}

// computes  J = dF/du + a dF/d(dudt)
PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec u, Vec dudt,PetscReal a, Mat J, Mat P, void *user)
{
    PetscErrorCode ierr;
    PetscInt       row[2] = {0, 1},  col[2] = {0, 1};
    PetscReal      v[4];
    PetscBool      flag = ((Ctx*)user)->identity_in_F;

    if (flag) {
      v[0] = a + 1.0;    v[1] = 2.0;
      v[2] = 3.0;        v[3] = a + 4.0;
    } else {
      v[0] = a + 1.0;    v[1] = a + 2.0;
      v[2] = 3.0;        v[3] = a + 4.0;
    }
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
      ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec u,Vec G, void *user)
{
    PetscErrorCode  ierr;
    const PetscReal *au;
    PetscReal       *aG;
    PetscBool       flag = ((Ctx*)user)->identity_in_F;

    ierr = VecGetArrayRead(u,&au);CHKERRQ(ierr);
    ierr = VecGetArray(G,&aG);CHKERRQ(ierr);
    if (flag) {
      aG[0] = 8.0 * au[1];
      aG[1] = 4.0 * au[0] + 4.0 * au[1];
    } else {
      aG[0] = au[0] + 8.0 * au[1];
      aG[1] = 4.0 * au[0] + 4.0 * au[1];
    }
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&aG); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormRHSJacobian(TS ts, PetscReal t, Vec u, Mat J, Mat P,void *user)
{
    PetscErrorCode ierr;
    PetscInt       row[2] = {0, 1},  col[2] = {0, 1};
    PetscReal      v[4];
    PetscBool      flag = ((Ctx*)user)->identity_in_F;

    if (flag) {
      v[0] = 0.0;    v[1] = 8.0;
      v[2] = 4.0;    v[3] = 4.0;
    } else {
      v[0] = 1.0;    v[1] = 8.0;
      v[2] = 4.0;    v[3] = 4.0;
    }
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  return 0;
}

/*TEST

    test:
      suffix: 1
      args: 
      requires: !single

    test:
      suffix: 2
      args: -ts_type arkimex -identity_in_F
      requires: !single

    test:
      suffix: 3
      args: -ts_rtol 1.0e-10 -ts_atol 1.0e-10 -ts_max_time 0.001
      requires: !single

TEST*/

