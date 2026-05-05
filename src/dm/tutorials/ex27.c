static char help[] = "Example adapted from Ed Bueler - petsc4pdes.\n"
                     "Solves structured-grid Poisson problem in 2D.\n"
                     "Equation is\n"
                     "    - cx u_xx - cy u_yy = f,\n"
                     "subject to Dirichlet boundary conditions.\n"
                     "Defaults to a SNESType of KSPONLY and a KSPType of CG.\n\n";

// run with -da_use_section

#include <petscsnes.h>
#include <petscdmda.h>
#include <petscsection.h>

typedef struct {
  // domain dimensions
  PetscScalar Lx, Ly;
  // coefficients in  - cx u_xx - cy u_yy = f
  PetscScalar cx, cy;
  // right-hand-side f(x,y)
  PetscScalar (*f_rhs)(PetscScalar x, PetscScalar y, void *ctx);
  // Dirichlet boundary condition g(x,y)
  PetscScalar (*g_bdry)(PetscScalar x, PetscScalar y, void *ctx);
} PoissonCtx;

static PetscScalar u_exact_2D(PetscScalar x, PetscScalar y, void *ctx)
{
  return -x * PetscExpReal(y);
}

// right-hand-side function  f(x,y) = - laplacian u
static PetscScalar f_rhs_2D(PetscScalar x, PetscScalar y, void *ctx)
{
  return x * PetscExpReal(y); // note  f = - (u_xx + u_yy) = - u
}

PetscErrorCode InsertBoundary(DM da, Vec u, PoissonCtx *user);
PetscErrorCode JacMult(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunctionGlobal(SNES snes, Vec X, Vec F, void *dummy);
PetscErrorCode FormExact(DMDALocalInfo *info, Vec u, PoissonCtx *user);
PetscErrorCode SetupBCs(DM da);

int main(int argc, char **argv)
{
  Mat           J;
  DM            da;
  KSP           ksp;
  PC            pc;
  SNES          snes;
  Vec           uglobal, u_exact, u_exact_local;
  PetscSection  lsection, gsection;
  PetscSF       sf;
  DMDALocalInfo info;
  PetscScalar   errinf, normconst2h, err2h;
  char          gridstr[99];
  PetscInt      N = 9;
  PoissonCtx    user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  user.Lx     = 1.0;
  user.Ly     = 1.0;
  user.cx     = 1.0;
  user.cy     = 1.0;
  user.g_bdry = &u_exact_2D;

  user.f_rhs  = &f_rhs_2D;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, N, N, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, user.Lx, 0.0, user.Ly, 0.0, 1.0));
  PetscCall(DMSetMatType(da, MATSHELL));
  PetscCall(SetupBCs(da));
  // create the local section - run with -da_use_section
  PetscCall(DMGetLocalSection(da, &lsection));
  PetscCall(DMGetPointSF(da, &sf));
  // create global section
  PetscCall(PetscSectionCreateGlobalSection(lsection, sf, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &gsection));
  PetscCall(DMSetGlobalSection(da, gsection));
  PetscCall(DMCreateSectionSF(da, lsection, gsection));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(SNESSetDM(snes, da));
  // Since we have constrained the boundaries, the matrix should be size 7^2 x 7^2 not 9^2 x 9^2
  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, NULL));
  // Set the matrix-free matmult action for J
  PetscCall(MatShellSetOperation(J, MATOP_MULT, (void (*)(void))JacMult));
  PetscCall(SNESSetFunction(snes, NULL, FormFunctionGlobal, (void *)(da)));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(MatSetDM(J, da));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(DMCreateGlobalVector(da, &uglobal));
  PetscCall(SNESSolve(snes, NULL, uglobal));
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMCreateLocalVector(da, &u_exact_local));
  PetscCall(DMCreateGlobalVector(da, &u_exact));
  PetscCall(FormExact(&info, u_exact_local, &user));
  PetscCall(DMLocalToGlobal(da, u_exact_local, INSERT_VALUES, u_exact));
  PetscCall(VecAXPY(uglobal, -1.0, u_exact)); // u <- u + (-1.0) uexact
  PetscCall(VecDestroy(&u_exact));            // no longer needed
  PetscCall(VecDestroy(&u_exact_local));
  PetscCall(VecNorm(uglobal, NORM_INFINITY, &errinf));
  PetscCall(VecNorm(uglobal, NORM_2, &err2h));
  normconst2h = PetscSqrtReal((PetscScalar)(info.mx - 1) * (info.my - 1));
  snprintf(gridstr, 99, "%d x %d point 2D", info.mx, info.my);

  err2h /= normconst2h; // like continuous L2
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "problem on %s grid:\n"
                        "  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e\n",
                        gridstr, errinf, err2h));
  PetscCall(PetscSectionDestroy(&gsection));
  PetscCall(VecDestroy(&uglobal));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());

  return 0;
}

PetscErrorCode FormFunctionGlobal(SNES snes, Vec u, Vec F, void *dummy)
{
  DM            dm = (DM)(dummy);
  PetscInt      i, j;
  DMDALocalInfo info;
  PetscScalar   scdiag, hx, hy, darea, x, y, ue, uw, un, us;
  PetscScalar **aF, **au, xymin[2], xymax[2], scx, scy;
  PoissonCtx   *user;
  Vec           u_local, F_local;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(VecSet(F, 0.0));

  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(DMGetLocalVector(dm, &F_local));
  PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, u_local));
  PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, u_local));

  // insert bc values in ghosted local vector
  PetscCall(InsertBoundary(dm, u_local, user));

  PetscCall(DMDAVecGetArray(dm, u_local, &au));
  PetscCall(DMDAVecGetArray(dm, F_local, &aF));

  PetscCall(DMGetBoundingBox(dm, xymin, xymax));
  hx     = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy     = (xymax[1] - xymin[1]) / (info.my - 1);
  darea  = hx * hy;
  scx    = user->cx * hy / hx;
  scy    = user->cy * hx / hy;
  scdiag = 2.0 * (scx + scy); // diagonal scaling

  for (j = info.ys; j < info.ys + info.ym; j++) {
    y = j * hy;
    for (i = info.xs; i < info.xs + info.xm; i++) {
      x = i * hx;
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        continue;
      } else {
        ue       = (i + 1 == info.mx - 1) ? user->g_bdry(x + hx, y, user) : au[j][i + 1];
        uw       = (i - 1 == 0) ? user->g_bdry(x - hx, y, user) : au[j][i - 1];
        un       = (j + 1 == info.my - 1) ? user->g_bdry(x, y + hy, user) : au[j + 1][i];
        us       = (j - 1 == 0) ? user->g_bdry(x, y - hy, user) : au[j - 1][i];
        aF[j][i] = scdiag * au[j][i] - (uw + ue) - (us + un) - darea * user->f_rhs(x, y, user);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(dm, u_local, &au));
  PetscCall(DMDAVecRestoreArray(dm, F_local, &aF));

  PetscCall(DMLocalToGlobalBegin(dm, F_local, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(dm, F_local, ADD_VALUES, F));

  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscCall(DMRestoreLocalVector(dm, &F_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode JacMult(Mat J, Vec X, Vec Y)
{
  DM            dm;
  DMDALocalInfo info;
  PoissonCtx   *user;
  Vec           xloc, yloc;
  PetscScalar   xymin[2], xymax[2], hx, hy, scx, scy, scdiag;
  PetscInt      i, j, xs, ys, xm, ym;
  PetscScalar   ue, uw, un, us;
  PetscScalar **x_u;
  PetscScalar **y_u;
  PetscFunctionBeginUser;
  PetscCall(VecSet(Y, 0.0));
  PetscCall(MatGetDM(J, &dm));
  PetscCall(DMGetApplicationContext(dm, &user));
  PetscCall(DMDAGetLocalInfo(dm, &info));

  PetscCall(DMGetLocalVector(dm, &xloc));
  PetscCall(DMGetLocalVector(dm, &yloc));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, xloc));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, xloc));
  PetscCall(VecSet(yloc, 0.0));
  PetscCall(DMDAVecGetArray(dm, yloc, &y_u));
  PetscCall(DMDAVecGetArray(dm, xloc, &x_u));
  PetscCall(DMGetBoundingBox(dm, xymin, xymax));
  hx     = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy     = (xymax[1] - xymin[1]) / (info.my - 1);
  scx    = user->cx * hy / hx;
  scy    = user->cy * hx / hy;
  scdiag = 2.0 * (scx + scy); // diagonal scaling

  DMDAGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL);

  for (j = info.ys; j < info.ys + info.ym; j++) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        continue;
      } else {
        ue        = (i + 1 == info.mx - 1) ? 0.0 : x_u[j][i + 1];
        uw        = (i - 1 == 0) ? 0.0 : x_u[j][i - 1];
        un        = (j + 1 == info.my - 1) ? 0.0 : x_u[j + 1][i];
        us        = (j - 1 == 0) ? 0.0 : x_u[j - 1][i];
        y_u[j][i] = scdiag * x_u[j][i] - (uw + ue) - (us + un);
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(dm, yloc, &y_u));
  PetscCall(DMDAVecRestoreArray(dm, xloc, &x_u));

  PetscCall(DMLocalToGlobalBegin(dm, yloc, ADD_VALUES, Y));
  PetscCall(DMLocalToGlobalEnd(dm, yloc, ADD_VALUES, Y));

  PetscCall(DMRestoreLocalVector(dm, &xloc));
  PetscCall(DMRestoreLocalVector(dm, &yloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InsertBoundary(DM da, Vec u, PoissonCtx *user)
{
  DMDALocalInfo info;
  PetscInt      i, j;
  PetscScalar   xymin[2], xymax[2], hx, hy, x, y, **au;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da, &info));
  PetscCall(DMDAVecGetArray(da, u, &au));
  PetscCall(DMGetBoundingBox(da, xymin, xymax));
  hx = (xymax[0] - xymin[0]) / (info.mx - 1);
  hy = (xymax[1] - xymin[1]) / (info.my - 1);
  // loop over ghosted region to insert bcs
  for (j = info.gys; j < info.gys + info.gym; j++) {
    y = xymin[1] + j * hy;
    for (i = info.gxs; i < info.gxs + info.gxm; i++) {
      if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
        x        = xymin[0] + i * hx;
        au[j][i] = user->g_bdry(x, y, user);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, u, &au));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormExact(DMDALocalInfo *info, Vec u, PoissonCtx *user)
{
  PetscInt    i, j;
  PetscScalar xymin[2], xymax[2], hx, hy, x, y, **au;

  PetscFunctionBeginUser;
  PetscCall(DMGetBoundingBox(info->da, xymin, xymax));
  hx = (xymax[0] - xymin[0]) / (info->mx - 1);
  hy = (xymax[1] - xymin[1]) / (info->my - 1);
  PetscCall(DMDAVecGetArray(info->da, u, &au));
  for (j = info->gys; j < info->gys + info->gym; j++) {
    y = xymin[1] + j * hy;
    for (i = info->gxs; i < info->gxs + info->gxm; i++) {
      x        = xymin[0] + i * hx;
      au[j][i] = user->g_bdry(x, y, user);
    }
  }
  PetscCall(DMDAVecRestoreArray(info->da, u, &au));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupBCs(DM da)
{
  PetscInt x, y, m, n, gx, gy, gm, gn, M, N, dim, dof, numBC = 0, *bcPointsArray;
  IS       bcPointsIS;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, &dim, &M, &N, NULL, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));

  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAGetCorners(da, &x, &y, NULL, &m, &n, NULL));
  PetscCall(DMDAGetGhostCorners(da, &gx, &gy, NULL, &gm, &gn, NULL));

  for (PetscInt j = info.gys; j < info.gys + info.gym; j++) {
      for (PetscInt i = info.gxs; i < info.gxs + info.gxm; i++) {
          PetscBool isBoundary = (i==0 || i==info.mx-1 || j==0 || j==info.my-1); 
          if (isBoundary) numBC++;
      }
  }
  // PetscCall(PetscPrintf(PetscObjectComm((PetscObject)da), "Number of boundary points: %d\n", numBC));
  // create an array of points to constrain
  PetscCall(PetscMalloc1(numBC, &bcPointsArray));
  PetscInt k = 0;
  for (PetscInt j = info.gys; j < info.gys + info.gym; j++) {
      for (PetscInt i = info.gxs; i < info.gxs + info.gxm; i++) {
          PetscBool isBoundary = (i==0 || i==info.mx-1 || j==0 || j==info.my-1); 
          if (isBoundary) bcPointsArray[k++] = (j - gy) * gm + (i - gx);
      }
  }
  // create an IS of boundary points
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)da), numBC, bcPointsArray, PETSC_OWN_POINTER, &bcPointsIS));
  //view the IS
  PetscCall(ISView(bcPointsIS,PETSC_VIEWER_STDOUT_WORLD));
  IS bcPoints[1] = {bcPointsIS};

  PetscCall(DMDASetPointBC(da, 1, bcPoints, NULL));

  PetscCall(ISDestroy(&bcPointsIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
