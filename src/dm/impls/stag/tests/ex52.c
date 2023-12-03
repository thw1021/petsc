static char help[] = "Test CGNS viewer for DMStag.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

static PetscErrorCode FillValues1d(DM dm, Vec v)
{
  PetscInt      M, x, m, nExtrax, dof[2], i, d;
  PetscInt      ileft = 0, ielem = 0, count = 0;
  PetscScalar **arr;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetGlobalSizes(dm, &M, NULL, NULL));
  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, &nExtrax, NULL, NULL));
  PetscCall(DMStagGetDOF(dm, &dof[0], &dof[1], NULL, NULL));
  if (dof[0] > 0) PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, 0, &ileft));
  if (dof[1] > 0) PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagVecGetArray(dm, v, &arr));

  for (d = 0; d < dof[0]; ++d)
    for (i = x; i < x + m + nExtrax; ++i) arr[i][ileft + d] = count++;
  for (d = 0; d < dof[1]; ++d)
    for (i = x; i < x + m; ++i) arr[i][ielem + d] = count++;

  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillValues2d(DM dm, Vec v)
{
  PetscInt       M, N, x, y, m, n, nExtrax, nExtray, dof[3], i, j, d;
  PetscInt       idownleft = 0, ileft = 0, idown = 0, ielem = 0, count = 0;
  PetscScalar ***arr;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));
  PetscCall(DMStagGetDOF(dm, &dof[0], &dof[1], &dof[2], NULL));
  if (dof[0] > 0) PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT, 0, &idownleft));
  if (dof[1] > 0) {
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, 0, &ileft));
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, 0, &idown));
  }
  if (dof[2] > 0) PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagVecGetArray(dm, v, &arr));

  for (d = 0; d < dof[0]; ++d)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m + nExtrax; ++i) arr[j][i][idownleft + d] = count++;
  for (d = 0; d < dof[1]; ++d)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m + nExtrax; ++i) arr[j][i][ileft + d] = count++;
  for (d = 0; d < dof[1]; ++d)
    for (j = y; j < y + n + nExtray; ++j)
      for (i = x; i < x + m; ++i) arr[j][i][idown + d] = count++;
  for (d = 0; d < dof[2]; ++d)
    for (j = y; j < y + n; ++j)
      for (i = x; i < x + m; ++i) arr[j][i][ielem + d] = count++;

  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillValues3d(DM dm, Vec v)
{
  PetscInt        M, N, P, x, y, z, m, n, p, nExtrax, nExtray, nExtraz, dof[4], i, j, k, d;
  PetscInt        ibackdownleft = 0, idownleft = 0, ibackleft = 0, ibackdown = 0, ileft = 0, idown = 0, iback = 0, ielem = 0, count = 0;
  PetscScalar ****arr;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, &P));
  PetscCall(DMStagGetCorners(dm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz));
  PetscCall(DMStagGetDOF(dm, &dof[0], &dof[1], &dof[2], &dof[3]));
  if (dof[0] > 0) PetscCall(DMStagGetLocationSlot(dm, DMSTAG_BACK_DOWN_LEFT, 0, &ibackdownleft));
  if (dof[1] > 0) {
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN_LEFT, 0, &idownleft));
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_BACK_LEFT, 0, &ibackleft));
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_BACK_DOWN, 0, &ibackdown));
  }
  if (dof[2] > 0) {
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, 0, &ileft));
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, 0, &idown));
    PetscCall(DMStagGetLocationSlot(dm, DMSTAG_BACK, 0, &iback));
  }
  if (dof[3] > 0) PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagVecGetArray(dm, v, &arr));

  for (d = 0; d < dof[0]; ++d)
    for (k = z; k < z + p + nExtraz; ++k)
      for (j = y; j < y + n + nExtray; ++j)
        for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][ibackdownleft + d] = count++;
  for (d = 0; d < dof[1]; ++d)
    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n + nExtray; ++j)
        for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][idownleft + d] = count++;
  for (d = 0; d < dof[1]; ++d)
    for (k = z; k < z + p + nExtraz; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][ibackleft + d] = count++;
  for (d = 0; d < dof[1]; ++d)
    for (k = z; k < z + p + nExtraz; ++k)
      for (j = y; j < y + n + nExtray; ++j)
        for (i = x; i < x + m; ++i) arr[k][j][i][ibackdown + d] = count++;
  for (d = 0; d < dof[2]; ++d)
    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m + nExtrax; ++i) arr[k][j][i][ileft + d] = count++;
  for (d = 0; d < dof[2]; ++d)
    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n + nExtray; ++j)
        for (i = x; i < x + m; ++i) arr[k][j][i][idown + d] = count++;
  for (d = 0; d < dof[2]; ++d)
    for (k = z; k < z + p + nExtraz; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) arr[k][j][i][iback + d] = count++;
  for (d = 0; d < dof[3]; ++d)
    for (k = z; k < z + p; ++k)
      for (j = y; j < y + n; ++j)
        for (i = x; i < x + m; ++i) arr[k][j][i][ielem + d] = count++;

  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  PetscInt  dim;
  PetscBool flg;
  Vec       v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Supply -dim option");

  if (dim == 1) PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm));
  else if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 12, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dm));
  else if (dim == 3) PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 12, 16, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 0, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, NULL, &dm));
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "dim must be 1, 2, or 3");
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2., 0., 3., 0., 4.));

  PetscCall(DMCreateLocalVector(dm, &v));
  PetscCall(PetscObjectSetName((PetscObject)v, "Vec"));
  if (dim == 1) PetscCall(FillValues1d(dm, v));
  else if (dim == 2) PetscCall(FillValues2d(dm, v));
  else if (dim == 3) PetscCall(FillValues3d(dm, v));
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "dim must be 1, 2, or 3");
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));

  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1d
      requires: cgns
      args: -dim 1 -vec_view cgns:vec.cgns

   test:
      suffix: 1d_par
      nsize: 2
      requires: cgns
      args: -dim 1 -vec_view cgns:vec.cgns
      output_file: output/ex52_1d.out

   test:
      suffix: 1d_dof
      requires: cgns
      args: -dim 1 -vec_view cgns:vec.cgns -stag_dof_0 2 -stag_dof_1 3
      output_file: output/ex52_1d.out

   test:
      suffix: 2d
      requires: cgns
      args: -dim 2 -vec_view cgns:vec.cgns
      output_file: output/ex52_1d.out

   test:
      suffix: 2d_par
      nsize: 4
      requires: cgns
      args: -dim 2 -vec_view cgns:vec.cgns
      output_file: output/ex52_1d.out

   test:
      suffix: 2d_dof
      requires: cgns
      args: -dim 2 -vec_view cgns:vec.cgns -stag_dof_0 2 -stag_dof_1 3 -stag_dof_2 2
      output_file: output/ex52_1d.out

   test:
      suffix: 3d
      requires: cgns
      args: -dim 3 -vec_view cgns:vec.cgns
      output_file: output/ex52_1d.out

   test:
      suffix: 3d_par
      nsize: 8
      requires: cgns
      args: -dim 3 -vec_view cgns:vec.cgns
      output_file: output/ex52_1d.out

   test:
      suffix: 3d_dof
      requires: cgns
      args: -dim 3 -vec_view cgns:vec.cgns -stag_dof_0 2 -stag_dof_2 3 -stag_dof_3 2
      output_file: output/ex52_1d.out

TEST*/
