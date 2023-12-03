static char help[] = "Test CGNS viewer\n\n";

#include <petscdm.h>
#include <petscdmstag.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM              dm;
  Vec             v;
  PetscScalar ****arr;
  PetscInt        N[3], s[3], n[3], dof, ielem, c, i, j, k;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 2, 3, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 0, NULL, NULL, NULL, &dm));
  PetscCall(PetscObjectSetName((PetscObject)dm, "DMStag"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2., -1., 2., -3., 1.));

  PetscCall(DMStagGetGlobalSizes(dm, &N[0], &N[1], &N[2]));
  PetscCall(DMStagGetCorners(dm, &s[0], &s[1], &s[2], &n[0], &n[1], &n[2], NULL, NULL, NULL));
  PetscCall(DMStagGetDOF(dm, NULL, NULL, NULL, &dof));

  PetscCall(DMCreateLocalVector(dm, &v));
  PetscCall(PetscObjectSetName((PetscObject)v, "Vec"));
  PetscCall(DMStagVecGetArray(dm, v, &arr));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  for (k = s[2]; k < s[2] + n[2]; ++k)
    for (j = s[1]; j < s[1] + n[1]; ++j)
      for (i = s[0]; i < s[0] + n[0]; ++i)
        for (c = 0; c < dof; ++c) arr[k][j][i][ielem + c] = N[0] * N[1] * k + N[0] * j + i + c;
  PetscCall(DMStagVecRestoreArray(dm, v, &arr));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));

  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      requires: !complex cgns
      args: -vec_view cgns:vec.cgns

   test:
      suffix: 2
      requires: !complex cgns
      args: -vec_view cgns:vec.cgns -stag_dof_3 3

   test:
      suffix: 3
      requires: !complex cgns
      nsize: 4
      args: -vec_view cgns:vec.cgns -stag_grid_x 4 -stag_grid_y 5 -stag_grid_z 6

TEST*/
