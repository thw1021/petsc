#include <petscdmda.h>

namespace
{

constexpr const char help[] = "2D Jacobi Stencil with PETSc.\n\n";

struct Args {
  PetscInt n_iter   = 100;
  PetscInt n_warmup = 5;
};

PetscErrorCode parse_args(Args *args)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(PETSC_COMM_WORLD, nullptr, "Benchmark Options", "");
  PetscCall(PetscOptionsInt("-iter", "Number of iterations", nullptr, args->n_iter, &args->n_iter, nullptr));
  PetscCall(PetscOptionsInt("-warmup", "Number of warmup iterations", nullptr, args->n_warmup, &args->n_warmup, nullptr));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode init_boundary_conditions(DM da, Vec global)
{
  PetscScalar **xy;
  PetscInt      xs, ys, xm, ym, Nx, Ny;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetInfo(da, nullptr, &Nx, &Ny, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));

  PetscCall(VecSet(global, 0.0));
  PetscCall(DMDAVecGetArray(da, global, &xy));
  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      // North boundary
      if (j == 0) xy[j][i] = -273.15;
      // West boundary
      if (i == 0) xy[j][i] = -273.15;
      // East boundary
      if (xs + xm == Nx) xy[j][xs + xm - 1] = -273.15;
      // South boundary
      if (ys + ym == Ny) xy[ys + ym - 1][i] = 40.0;
    }
  }
  PetscCall(DMDAVecRestoreArray(da, global, &xy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode kernel(const Args &args, DM da, Vec local, Vec global)
{
  PetscInt xm, ym, xs, ys, gxm, gym, gxs, gys, Nx, Ny;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, nullptr, &gxm, &gym, nullptr));
  PetscCall(DMDAGetInfo(da, nullptr, &Nx, &Ny, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));

  const PetscScalar *xy_local;
  PetscScalar       *xy_global;

  PetscCall(VecGetArrayWrite(global, &xy_global));
  PetscCall(VecGetArrayRead(local, &xy_local));

  const auto IDX_LOCAL  = [&](PetscInt i, PetscInt j) { return xy_local[(j - gys) * gxm + (i - gxs)]; };
  const auto IDX_GLOBAL = [&](PetscInt i, PetscInt j) -> PetscScalar & { return xy_global[(j - ys) * xm + (i - xs)]; };

  const PetscInt jstart = ys == 0 ? ys + 1 : ys;
  const PetscInt jend   = ys + ym == Ny ? ym - 1 : ym;
  const PetscInt istart = xs == 0 ? xs + 1 : xs;
  const PetscInt iend   = xs + xm == Nx ? xm - 1 : xm;

  for (PetscInt j = jstart; j < jend; ++j) {
    for (PetscInt i = istart; i < iend; ++i) {
      const auto center  = IDX_LOCAL(i, j);
      const auto north   = IDX_LOCAL(i, j - 1);
      const auto east    = IDX_LOCAL(i + 1, j);
      const auto west    = IDX_LOCAL(i - 1, j);
      const auto south   = IDX_LOCAL(i, j + 1);
      const auto average = center + north + east + west + south;

      IDX_GLOBAL(i, j) = 0.2 * average;
    }
  }

  PetscCall(VecRestoreArrayRead(local, &xy_local));
  PetscCall(VecRestoreArrayWrite(global, &xy_global));

  PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(HAVE_KOKKOS_KERNELS)
PetscErrorCode kernel(const Args &args, DM da, Vec local, Vec global)
{
  PetscInt xm, ym, xs, ys, Nx, Ny;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCall(DMDAGetInfo(da, nullptr, &Nx, &Ny, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));

  ConstPetscScalarKokkosOffsetView2D xy_local;
  PetscScalarKokkosOffsetView2D      xy_global;

  PetscCall(DMDAVecGetKokkosOffsetView(da, local, &xy_local));
  PetscCall(DMDAVecGetKokkosOffsetView(da, global, &xy_global));

  const PetscInt jstart = ys == 0 ? ys + 1 : ys;
  const PetscInt jend   = ys + ym == Ny ? ym - 1 : ym;
  const PetscInt istart = xs == 0 ? xs + 1 : xs;
  const PetscInt iend   = xs + xm == Nx ? xm - 1 : xm;

  Kokkos::parallel_for(
    "stencil", MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Rank<2, Iterate::Right, Iterate::Right>>({jstart, istart}, {jend, iend}), KOKKOS_LAMBDA(PetscInt j, PetscInt i) {
      const auto center  = xy_local(i, j);
      const auto north   = xy_local(i, j - 1);
      const auto east    = xy_local(i + 1, j);
      const auto west    = xy_local(i - 1, j);
      const auto south   = xy_local(i, j + 1);
      const auto average = center + north + east + west + south;

      xy_global(i, j) = 0.2 * average;
    });

  PetscCall(DMDAVecRestoreKokkosOffsetView(da, local, &xy_local));
  PetscCall(DMDAVecRestoreKokkosOffsetView(da, global, &xy_global));

  PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

} // namespace

int main(int argc, char *argv[])
{
  Args args{};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(parse_args(&args));

  DM da;

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, 100, 100, PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  Vec local, global;

  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMCreateLocalVector(da, &local));

  PetscCall(init_boundary_conditions(da, global));

  PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));

  PetscCall(kernel(args, da, local, global));

  PetscCall(VecViewFromOptions(global, reinterpret_cast<PetscObject>(da), "-vec_view_global"));

  PetscCall(PetscFinalize());
  return 0;
}
