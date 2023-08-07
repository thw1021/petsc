static constexpr const char help[] = "PetscManagedType Benchmark\n";

#include <petscdmda.h>
#include <petscksp.h>
#include <petscdevice_cupm.h>
#include <petsc/private/deviceimpl.h>

#include <vector>
#include <string>
#include <algorithm>
#if PetscDefined(USE_NVTX)
  #include <cstdlib>
#endif

extern "C" const char *__asan_default_options()
{
  return PetscDefined(USE_DEBUG) ? "protect_shadow_gap=0" : "";
}

static PetscErrorCode InitializePetscDeviceContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_DEFAULT()));
  PetscCall(PetscDeviceContextGetCurrentContext(dctx));
  PetscCall(PetscDeviceContextSetUp(*dctx));
  if (PETSC_DEVICE_DEFAULT() != PETSC_DEVICE_HOST) {
    void *handle;

    // initializes the handles apriori
    PetscCall(PetscDeviceContextGetBLASHandle_Internal(*dctx, &handle));
    PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(*dctx, &handle));
  }
  PetscCall(PetscDeviceContextSynchronize(*dctx));
  PetscCallCUPM(hipDeviceSynchronize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillSystem(Mat A, Vec b)
{
  PetscInt     dim, dof, mx, my, mz, xm, ym, zm, xs, ys, zs;
  PetscScalar *v, Hx, Hy, Hz, dHx, dHy, dHz;
  MatStencil   row, col[7];
  DM           da;

  PetscFunctionBeginUser;
  PetscCall(MatGetDM(A, &da));

  PetscCall(DMDAGetInfo(da, &dim, &mx, &my, &mz, nullptr, nullptr, nullptr, &dof, nullptr, nullptr, nullptr, nullptr, nullptr));
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

static PetscErrorCode CreateDMDAOperators(MPI_Comm comm, PetscInt n, Mat *Aout, Vec *bout, Vec *xout)
{
  // Topological dimension and default grid
  PetscInt dim = 3, grid[3] = {n, n, n};
  // PDE options
  PetscInt  dof        = 1;           // No. of dofs per point
  PetscBool fd_stencil = PETSC_FALSE; // Use FD type stencil

  PetscFunctionBegin;
  PetscOptionsBegin(comm, nullptr, "Create DMDA Options", nullptr);
  PetscCall(PetscOptionsInt("-dim", "Dimension of DMDA", "", dim, &dim, nullptr));
  PetscCall(PetscOptionsInt("-dof", "DOF of DMDA", "", dof, &dof, nullptr));
  PetscCall(PetscOptionsBool("-fd_stencil", "Use FD stencil for DMDA", "DMDAStencilType", fd_stencil, &fd_stencil, nullptr));
  PetscOptionsEnd();

  /* Create structured grid */
  DM              dm;
  DMDAStencilType stencil_type = fd_stencil ? DMDA_STENCIL_STAR : DMDA_STENCIL_BOX;
  PetscInt        stencil_pts;
  switch (dim) {
  case 3:
    PetscCall(DMDACreate3d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, stencil_type, grid[0], grid[1], grid[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, 1, nullptr, nullptr, nullptr, &dm));
    stencil_pts = fd_stencil ? 7 : 27;
    break;
  case 2:
    PetscCall(DMDACreate2d(comm, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, stencil_type, grid[0], grid[1], PETSC_DECIDE, PETSC_DECIDE, dof, 1, nullptr, nullptr, &dm));
    stencil_pts = fd_stencil ? 5 : 9;
    break;
  case 1:
    PetscCall(DMDACreate1d(comm, DM_BOUNDARY_NONE, grid[0], dof, 1, nullptr, &dm));
    stencil_pts = 3;
    break;
  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, dim);
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
  PetscCall(MatGetSize(A, &N, nullptr));

  /* Due to errors in MatSolve_SeqAIJHIPSPARSE_ICC0() */
  PetscBool iship;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &iship, MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, ""));
  if (!iship) {
    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
    PetscCall(MatSetOption(A, MAT_SPD_ETERNAL, PETSC_TRUE));
  }

  /* Print banner */
  PetscCall(DMDAGetInfo(dm, &dim, &grid[0], &grid[1], &grid[2], nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCount global_nnz = (PetscCount)info.nz_allocated;

  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "===========================================\n"));
  PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(comm)));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "Grid: %" PetscInt_FMT "x%" PetscInt_FMT "x%" PetscInt_FMT "\n", grid[0], grid[1], grid[2]));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "Fields: %" PetscInt_FMT "\n", dof));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "Input matrix: %" PetscInt_FMT "-pt stencil\n", stencil_pts));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "DoFs: %" PetscInt_FMT "\n", N));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(comm), "Number of nonzeros: %" PetscCount_FMT "\n", global_nnz));
  PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(comm)));

  Vec b, x;
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &x));

  PetscCall(PetscObjectSetName((PetscObject)b, "RHS"));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecSetFromOptions(x));

  /* Fill matrix and rhs */
  PetscCall(FillSystem(A, b));

  *Aout = A;
  *bout = b;
  *xout = x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct SolveData {
  std::string name{};
  PetscInt    its_before{-1};
  PetscReal   norm_before{PETSC_MIN_REAL};
  KSP         ksp{};
  Vec         b{};
  Vec         x{};
  Vec         x_cpy{};

  PetscDeviceContext         dctx{};
  mutable PetscDeviceContext cur{};

  PetscErrorCode push_ctx() const noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceContextGetCurrentContext(&this->cur));
    PetscCall(PetscDeviceContextSetCurrentContext(this->dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode pop_ctx() const noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceContextSetCurrentContext(this->cur));
    this->cur = nullptr;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  SolveData(std::string name, PetscDeviceContext dctx, KSP ksp, Vec b, Vec x) noexcept : name{std::move(name)}, ksp{ksp}
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->init_(dctx, b, x));
    PetscFunctionReturnVoid();
  }

  PetscErrorCode init_(PetscDeviceContext dctx, Vec b, Vec x) noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscDeviceContextCreate(&this->dctx));
    PetscCall(PetscObjectSetName((PetscObject)this->dctx, this->name.c_str()));
    PetscCall(PetscDeviceContextSetStreamType(this->dctx, dctx->streamType));
    PetscCall(PetscDeviceContextSetUp(this->dctx));
    PetscCall(this->push_ctx());
    PetscCall(VecDuplicate(b, &this->b));
    PetscCall(VecCopy(b, this->b));
    PetscCall(VecDuplicate(x, &this->x));
    PetscCall(VecCopy(x, this->x));
    PetscCall(VecDuplicate(x, &this->x_cpy));
    PetscCall(VecCopy(x, this->x_cpy));
    PetscCall(this->pop_ctx());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode destroy() noexcept
  {
    PetscFunctionBegin;
    PetscCall(this->push_ctx());
    PetscCall(VecDestroy(&this->b));
    PetscCall(VecDestroy(&this->x));
    PetscCall(VecDestroy(&this->x_cpy));
    PetscCall(KSPDestroy(&this->ksp));
    PetscCall(this->pop_ctx());
    PetscCall(PetscDeviceContextDestroy(&this->dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode pre_solve() const noexcept
  {
    PetscFunctionBegin;
    PetscCall(this->push_ctx());
    PetscCall(VecCopy(this->x_cpy, this->x));
    PetscCall(PetscDeviceContextSynchronize(this->dctx));
    PetscCall(PetscDeviceContextSynchronize(nullptr));
    PetscCallCUPM(hipDeviceSynchronize());
    PetscCall(PetscBarrier(nullptr));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode solve() const noexcept
  {
    PetscFunctionBegin;
    PetscCall(this->push_ctx());
    PetscCall(KSPSolve(this->ksp, this->b, this->x));
    PetscCall(this->pop_ctx());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode post_solve(PetscInt i) noexcept
  {
    PetscInt  its;
    PetscReal norm;

    // must be above because nvtx range push-pop semantics...
    PetscFunctionBegin;
    PetscCallCUPM(hipDeviceSynchronize());
    PetscCall(KSPGetIterationNumber(this->ksp, &its));
    PetscCheck(its >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of iterations %" PetscInt_FMT " < 0", its);
    if (i) {
      PetscCheck(its == this->its_before, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of iterations changed, before: %" PetscInt_FMT " (solve %" PetscInt_FMT ") != current: %" PetscInt_FMT " (solve %" PetscInt_FMT ")", this->its_before, i - 1, its, i);
    }
    this->its_before = its;
    PetscCall(KSPGetResidualNorm(this->ksp, &norm));
    if (i) PetscCheck(norm == this->norm_before, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Norm changed, before: %g (solve %" PetscInt_FMT ") != after: %g (solve %" PetscInt_FMT ")", (double)this->norm_before, i - 1, (double)norm, i);
    this->norm_before = norm;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

int main(int argc, char **args)
{
  Vec                x, b; /* approx solution, RHS */
  Mat                A;    /* linear system matrix */
  PetscInt           n = 10, nwarmup = 2, nit = 1000, nsolve = 1;
  PetscDeviceContext dctx;
  MPI_Comm           comm;

#if PetscDefined(USE_NVTX)
  PetscGlobalNVTXDomain = nvtxDomainCreateA("bench");
  std::atexit([] { nvtxDomainDestroy(PetscGlobalNVTXDomain); });
#endif
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, nullptr, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, nullptr, "Benchmark Options", nullptr);
  PetscCall(PetscOptionsInt("-n", "Global size of matrix", "MatSetSizes()", n, &n, nullptr));
  PetscCall(PetscOptionsInt("-n_warmup", "Number of iterations in the warmup loop", nullptr, nwarmup, &nwarmup, nullptr));
  PetscCall(PetscOptionsInt("-n_it", "Number of iterations in the timing loop", nullptr, nit, &nit, nullptr));
  PetscCall(PetscOptionsBoundedInt("-n_solve", "Number of solves in the timing loop", nullptr, nsolve, &nsolve, nullptr, 0));
  PetscOptionsEnd();

  PetscCall(InitializePetscDeviceContext(&dctx));
  PetscCall(CreateDMDAOperators(comm, n, &A, &b, &x));

  std::vector<SolveData> sd_vec;

  PetscCallCXX(sd_vec.reserve(nsolve));
  for (PetscInt i = 0; i < nsolve; ++i) {
    std::string name = "sub_stream_" + std::to_string(i);
    KSP         ksp;
    PC          pc;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create the linear solver and set various options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCallCXX(sd_vec.emplace_back(std::move(name), dctx, ksp, b, x));
  }
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));

  PetscLogStage warmup, timing;

  PetscCall(PetscLogStageRegister("Warmup", &warmup));
  PetscCall(PetscLogStageRegister("Timing", &timing));

  PetscCall(PetscLogStagePush(warmup));
  for (PetscInt i = 0; i < nwarmup; ++i) {
    for (auto &&sd : sd_vec) {
      PetscCall(sd.pre_solve());
      PetscCall(sd.solve());
      PetscCall(sd.post_solve(i));
    }
  }
  PetscCall(PetscLogStagePop());

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  std::vector<PetscLogDouble> times;

  PetscCallCXX(times.reserve(nit));
  for (PetscInt i = 0; i < nit; ++i) {
    PetscLogDouble begin, end;

    for (auto &&sd : sd_vec) PetscCall(sd.pre_solve());
    PetscCall(PetscLogStagePush(timing));
    PetscCallCUPM(cupmProfilerStart());
    PetscCall(PetscTime(&begin));
    for (auto &&sd : sd_vec) PetscCall(sd.solve());
    PetscCallCUPM(hipDeviceSynchronize());
    PetscCall(PetscTime(&end));
    PetscCallCUPM(cupmProfilerStop());
    PetscCall(PetscLogStagePop());
    for (auto &&sd : sd_vec) PetscCall(sd.post_solve(i));
    PetscCallCXX(times.emplace_back(end - begin));
  }

  PetscLogDouble tmin = PETSC_MAX_REAL, tmax = PETSC_MIN_REAL, ttotal = 0;
  for (auto &&time : times) {
    ttotal += time;
    tmin = std::min(time, tmin);
    tmax = std::max(time, tmax);
  }

  KSPType type;

  PetscCall(KSPGetType(sd_vec[0].ksp, &type));
  PetscCall(PetscPrintf(comm, "KSP type: '%s', nit %" PetscInt_FMT ", nsolve %" PetscInt_FMT " total time %gs, min %gs, max %gs, avg. %gs\n", type, nit, nsolve, ttotal, tmin, tmax, ttotal / nit));
  PetscCall(PetscPrintf(comm, "Norm of error %g, Iterations %" PetscInt_FMT "\n", static_cast<double>(sd_vec[0].norm_before), sd_vec[0].its_before));

  DM dm;
  PetscCall(MatGetDM(A, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(MatDestroy(&A));
  for (auto &&sd : sd_vec) PetscCall(sd.destroy());
  PetscCall(PetscDeviceContextSetCurrentContext(dctx));
  PetscCall(PetscFinalize());
  return 0;
}
