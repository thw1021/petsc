static char help[] = "Compares PCPATCH star patches with color-star patches.\n\n";

#include <petscksp.h>
#include <petscdmplex.h>
#include <petscpc.h>

typedef struct {
  PetscInt    cells[2];
  PetscInt    nodesPerCell;
  PetscInt    numDofs;
  PetscInt   *cellNodeMap;
  PetscScalar elemMat[16];
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  PetscInt  n = 2;
  PetscBool flg;

  PetscFunctionBeginUser;
  user->cells[0]    = 4;
  user->cells[1]    = 4;
  user->cellNodeMap = NULL;
  PetscOptionsBegin(comm, NULL, "PCPATCH color-star test options", "PC");
  PetscCall(PetscOptionsIntArray("-cells", "Number of mesh cells in each direction", "ex13.c", user->cells, &n, &flg));
  PetscOptionsEnd();
  PetscCheck(!flg || n == 2, comm, PETSC_ERR_ARG_SIZ, "Expected two entries for -cells");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, user->cells, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, dm));
  PetscCall(DMSetBasicAdjacency(*dm, PETSC_FALSE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  PetscSection section;
  PetscInt     pStart, pEnd, vStart, vEnd, cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &section));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) PetscCall(PetscSectionSetDof(section, v, 1));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionGetStorageSize(section, &user->numDofs));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCheck(cStart < cEnd, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Mesh has no cells");
  {
    PetscInt *closure = NULL;
    PetscInt  closureSize;

    user->nodesPerCell = 0;
    PetscCall(DMPlexGetTransitiveClosure(dm, cStart, PETSC_TRUE, &closureSize, &closure));
    for (PetscInt cl = 0; cl < closureSize * 2; cl += 2) {
      PetscInt dof;

      PetscCall(PetscSectionGetDof(section, closure[cl], &dof));
      if (dof) user->nodesPerCell++;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cStart, PETSC_TRUE, &closureSize, &closure));
  }
  PetscCheck(user->nodesPerCell <= 4, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Expected at most four nodes per cell");
  PetscCall(PetscMalloc1((cEnd - cStart) * user->nodesPerCell, &user->cellNodeMap));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, n = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    for (PetscInt cl = 0; cl < closureSize * 2; cl += 2) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(section, closure[cl], &dof));
      if (!dof) continue;
      PetscCall(PetscSectionGetOffset(section, closure[cl], &off));
      user->cellNodeMap[(c - cStart) * user->nodesPerCell + n++] = off;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    PetscCheck(n == user->nodesPerCell, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell has %" PetscInt_FMT " nodes, expected %" PetscInt_FMT, n, user->nodesPerCell);
  }
  for (PetscInt i = 0; i < user->nodesPerCell; ++i) {
    for (PetscInt j = 0; j < user->nodesPerCell; ++j) user->elemMat[i * user->nodesPerCell + j] = (i == j) ? (PetscScalar)user->nodesPerCell : -1.0;
  }
  PetscCall(PetscSectionDestroy(&section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperator(DM dm, AppCtx *user, Mat *A)
{
  PetscInt cStart, cEnd, dNz;

  PetscFunctionBeginUser;
  dNz = PetscMin(user->numDofs, user->nodesPerCell * user->nodesPerCell);
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, user->numDofs, user->numDofs, dNz, NULL, dNz, NULL, A));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt *idx = &user->cellNodeMap[(c - cStart) * user->nodesPerCell];

    PetscCall(MatSetValues(*A, user->nodesPerCell, idx, user->nodesPerCell, idx, user->elemMat, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputePatchOperator(PC pc, PetscInt point, Vec x, Mat mat, IS cellIS, PetscInt n, const PetscInt dofsArray[], const PetscInt dofsArrayWithAll[], void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt ncell;

  PetscFunctionBeginUser;
  PetscCheck(n % user->nodesPerCell == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Patch dof map size is not a multiple of the cell dof count");
  ncell = n / user->nodesPerCell;
  PetscCall(MatZeroEntries(mat));
  for (PetscInt c = 0; c < ncell; ++c) {
    const PetscInt *idx = &dofsArray[c * user->nodesPerCell];

    PetscCall(MatSetValues(mat, user->nodesPerCell, idx, user->nodesPerCell, idx, user->elemMat, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  One stratum per vertex, each holding the star of that vertex, which is the same decomposition that
  -pc_patch_construct_type star builds internally. Handing it to PCPatchSetPatchLabel() must therefore reproduce the
  patches that PCPATCH constructs itself. When `active` is given, only the vertices it marks get a patch, which is the
  decomposition that PCPatchSetConstructLabel() must select.
*/
static PetscErrorCode CreateVertexPatchLabel(DM dm, DMLabel active, DMLabel *label)
{
  PetscInt vStart, vEnd, n = 0;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "patches", label));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscInt val = 1;

    if (active) PetscCall(DMLabelGetValue(active, v, &val));
    if (val == 1) PetscCall(DMLabelSetValue(*label, v, n++));
  }
  PetscCall(DMPlexLabelCompleteStar(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Marks the closure of the first `nCells` cells, or of every cell when `nCells` is negative, which is the set of points
  whose star contains one of those cells. Selecting every cell must leave the patches unchanged.
*/
static PetscErrorCode CreateActiveLabel(DM dm, PetscInt nCells, DMLabel *label)
{
  PetscInt cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  if (nCells >= 0) cEnd = PetscMin(cStart + nCells, cEnd);
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "active", label));
  for (PetscInt c = cStart; c < cEnd; ++c) PetscCall(DMLabelSetValue(*label, c, 1));
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveWithPatch(DM dm, Mat A, const char prefix[], AppCtx *user, DMLabel patchLabel, DMLabel constructLabel, PetscInt *its)
{
  KSP             ksp;
  PC              pc;
  Vec             x, b;
  DM              dms[1];
  PetscInt        bs[1] = {1}, nodesPerCell[1], subspaceOffsets[2];
  const PetscInt *cellNodeMaps[1];

  PetscFunctionBeginUser;
  nodesPerCell[0]    = user->nodesPerCell;
  subspaceOffsets[0] = 0;
  subspaceOffsets[1] = user->numDofs;
  cellNodeMaps[0]    = user->cellNodeMap;
  dms[0]             = dm;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetTolerances(ksp, 1.0e-10, PETSC_CURRENT, PETSC_CURRENT, 100));
  PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPATCH));
  PetscCall(PCSetDM(pc, dm));
  PetscCall(PCPatchSetDiscretisationInfo(pc, 1, dms, bs, nodesPerCell, cellNodeMaps, subspaceOffsets, 0, NULL, 0, NULL));
  PetscCall(PCPatchSetComputeOperator(pc, ComputePatchOperator, user));
  if (patchLabel) PetscCall(PCPatchSetPatchLabel(pc, patchLabel));
  if (constructLabel) PetscCall(PCPatchSetConstructLabel(pc, constructLabel, 1));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(x, 0.0));
  PetscCall(VecSet(b, 1.0));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx   user;
  DM       dm;
  DMLabel  patchLabel, allLabel, activeLabel;
  Mat      A;
  PetscInt standardIts, coloredIts, labeledIts, allIts, activeIts, namedIts;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(CreateOperator(dm, &user, &A));
  PetscCall(SolveWithPatch(dm, A, "standard_", &user, NULL, NULL, &standardIts));
  PetscCall(SolveWithPatch(dm, A, "colored_", &user, NULL, NULL, &coloredIts));
  PetscCheck(standardIts == coloredIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard star patches took %" PetscInt_FMT " iterations, but colored star patches took %" PetscInt_FMT, standardIts, coloredIts);

  /* Supplying the star patches as the strata of a label must reproduce the standard star patches */
  PetscCall(CreateVertexPatchLabel(dm, NULL, &patchLabel));
  PetscCall(SolveWithPatch(dm, A, "labeled_", &user, patchLabel, NULL, &labeledIts));
  PetscCheck(standardIts == labeledIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard star patches took %" PetscInt_FMT " iterations, but labeled patches took %" PetscInt_FMT, standardIts, labeledIts);
  PetscCall(DMLabelDestroy(&patchLabel));

  /* Restricting the patches to every mesh point must also leave them unchanged */
  PetscCall(CreateActiveLabel(dm, -1, &allLabel));
  PetscCall(SolveWithPatch(dm, A, "restricted_", &user, NULL, allLabel, &allIts));
  PetscCheck(standardIts == allIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard star patches took %" PetscInt_FMT " iterations, but patches restricted to every point took %" PetscInt_FMT, standardIts, allIts);
  PetscCall(DMLabelDestroy(&allLabel));

  /*
    A genuinely restricted set leaves the dofs outside every patch untouched, so the preconditioner is singular and the
    solve converges only in the preconditioned norm. That is what local smoothing is: it is one level of a multigrid
    hierarchy, where the coarse grid handles the rest, and it says nothing on its own. So compare the restriction
    against the same patches listed explicitly, which must select exactly the same decomposition.
  */
  PetscCall(CreateActiveLabel(dm, 2, &activeLabel));
  PetscCall(CreateVertexPatchLabel(dm, activeLabel, &patchLabel));
  PetscCall(SolveWithPatch(dm, A, "restricted_", &user, NULL, activeLabel, &activeIts));
  PetscCall(SolveWithPatch(dm, A, "labeled_", &user, patchLabel, NULL, &labeledIts));
  PetscCheck(activeIts == labeledIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Patches restricted to two cells took %" PetscInt_FMT " iterations, but the same patches listed explicitly took %" PetscInt_FMT, activeIts, labeledIts);
  PetscCall(DMLabelDestroy(&patchLabel));

  /* Naming the same label on the DM must select the same restriction as passing it in */
  PetscCall(DMAddLabel(dm, activeLabel));
  PetscCall(SolveWithPatch(dm, A, "named_", &user, NULL, NULL, &namedIts));
  PetscCheck(activeIts == namedIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Patches restricted by a label took %" PetscInt_FMT " iterations, but naming that label on the DM took %" PetscInt_FMT, activeIts, namedIts);
  PetscCall(DMRemoveLabel(dm, "active", NULL));
  PetscCall(DMLabelDestroy(&activeLabel));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(user.cellNodeMap));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -standard_pc_patch_construct_type star -standard_sub_ksp_type preonly -standard_sub_pc_type lu \
          -colored_pc_patch_construct_type star -colored_pc_patch_use_coloring -colored_sub_ksp_type preonly -colored_sub_pc_type lu \
          -labeled_sub_ksp_type preonly -labeled_sub_pc_type lu \
          -restricted_pc_patch_construct_type star -restricted_sub_ksp_type preonly -restricted_sub_pc_type lu \
          -named_pc_patch_construct_type star -named_pc_patch_construct_label active -named_sub_ksp_type preonly -named_sub_pc_type lu
    output_file: output/empty.out
    test:
      suffix: patch_color_star
      args: -cells {{4,4 8,8}}
    # The restricted patches must be colored over the selected points only
    test:
      suffix: patch_color_star_restricted
      args: -cells {{4,4 8,8}} -restricted_pc_patch_use_coloring

TEST*/
