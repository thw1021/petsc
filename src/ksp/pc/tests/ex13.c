static char help[] = "Compares PCPATCH star and Vanka patches with the colored, labeled and restricted patches that must reproduce them.\n\n";

#include <petscksp.h>
#include <petscdmplex.h>

typedef struct {
  DM          cellDM; /* Carries the section of the cellwise subspace, NULL unless mixed */
  PetscInt    cells[2];
  PetscInt    nodesPerCell; /* Nodes of a cell in the vertex subspace */
  PetscInt    dofsPerCell;  /* Nodes of a cell over all the subspaces */
  PetscInt    numCells;
  PetscInt    numVertexDofs;
  PetscInt    numDofs;
  PetscInt   *cellNodeMap;
  PetscInt   *cellDofMap;
  PetscScalar elemMat[25];
  PetscBool   mixed;
  PetscBool   unevenColors;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *user)
{
  PetscInt  n = 2;
  PetscBool flg;

  PetscFunctionBeginUser;
  user->cells[0]     = 4;
  user->cells[1]     = 4;
  user->cellDM       = NULL;
  user->cellNodeMap  = NULL;
  user->cellDofMap   = NULL;
  user->mixed        = PETSC_FALSE;
  user->unevenColors = PETSC_FALSE;
  PetscOptionsBegin(comm, NULL, "PCPATCH color-star test options", "PC");
  PetscCall(PetscOptionsIntArray("-cells", "Number of mesh cells in each direction", "ex13.c", user->cells, &n, &flg));
  PetscCall(PetscOptionsBool("-mixed", "Add a cellwise subspace, which a Vanka patch may take at its base entity alone", "ex13.c", user->mixed, &user->mixed, NULL));
  PetscCall(PetscOptionsBool("-check_uneven_colors", "Check that the processes color their own vertices with different numbers of colors", "ex13.c", user->unevenColors, &user->unevenColors, NULL));
  PetscOptionsEnd();
  PetscCheck(flg == PETSC_FALSE || n == 2, comm, PETSC_ERR_ARG_SIZ, "Expected two entries for -cells");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM               pdm = NULL;
  PetscPartitioner part;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, user->cells, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, dm));
  PetscCall(DMSetBasicAdjacency(*dm, PETSC_FALSE, PETSC_TRUE));
  /* A star patch holds every cell around the vertex it is built around, so each process needs one layer of overlap */
  PetscCall(DMPlexGetPartitioner(*dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(DMPlexDistribute(*dm, 1, NULL, &pdm));
  if (pdm != NULL) {
    PetscCall(DMDestroy(dm));
    *dm = pdm;
  }
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
  PetscCall(PetscSectionGetStorageSize(section, &user->numVertexDofs));

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
  PetscCall(PetscSectionDestroy(&section));

  /* The cellwise subspace holds one node per cell, whose section offsets are the cell numbers themselves */
  user->numCells    = cEnd - cStart;
  user->numDofs     = user->numVertexDofs;
  user->dofsPerCell = user->nodesPerCell;
  if (user->mixed == PETSC_TRUE) {
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &section));
    PetscCall(PetscSectionSetChart(section, pStart, pEnd));
    for (PetscInt c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(section, c, 1));
    PetscCall(PetscSectionSetUp(section));
    PetscCall(DMClone(dm, &user->cellDM));
    PetscCall(DMSetLocalSection(user->cellDM, section));
    PetscCall(PetscSectionDestroy(&section));
    PetscCall(PetscMalloc1(user->numCells, &user->cellDofMap));
    for (PetscInt c = cStart; c < cEnd; ++c) user->cellDofMap[c - cStart] = c - cStart;
    user->numDofs += user->numCells;
    user->dofsPerCell += 1;
  }
  for (PetscInt i = 0; i < user->dofsPerCell; ++i) {
    for (PetscInt j = 0; j < user->dofsPerCell; ++j) user->elemMat[i * user->dofsPerCell + j] = (i == j) ? (PetscScalar)user->dofsPerCell : -1.0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperator(DM dm, AppCtx *user, Mat *A)
{
  ISLocalToGlobalMapping ltog;
  IS                     cellNumbers;
  const PetscInt        *numbers;
  PetscInt              *idx;
  PetscInt               cStart, cEnd, dNz, globalVertices, globalCells = 0, localCellMax = -1, range[2], globalRange[2];

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepthStratumGlobalSize(dm, 0, &globalVertices));
  PetscCall(DMPlexCreateCellNumbering(dm, PETSC_TRUE, &cellNumbers));
  PetscCall(ISGetIndices(cellNumbers, &numbers));
  if (user->mixed == PETSC_TRUE) {
    for (PetscInt c = 0; c < user->numCells; ++c) {
      if (numbers[c] >= 0) localCellMax = PetscMax(localCellMax, numbers[c]);
    }
    range[0] = 0;
    range[1] = localCellMax;
    PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)dm), range, globalRange));
    globalCells = globalRange[1] + 1;
  }
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  dNz = PetscMin(globalVertices + globalCells, user->dofsPerCell * user->dofsPerCell);
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, globalVertices + globalCells, globalVertices + globalCells, dNz, NULL, dNz, NULL, A));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscMalloc1(user->dofsPerCell, &idx));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscCall(PetscArraycpy(idx, &user->cellNodeMap[(c - cStart) * user->nodesPerCell], user->nodesPerCell));
    if (numbers[c - cStart] < 0) continue;
    PetscCall(ISLocalToGlobalMappingApply(ltog, user->nodesPerCell, idx, idx));
    if (user->mixed == PETSC_TRUE) idx[user->nodesPerCell] = globalVertices + numbers[c - cStart];
    PetscCall(MatSetValues(*A, user->dofsPerCell, idx, user->dofsPerCell, idx, user->elemMat, ADD_VALUES));
  }
  PetscCall(PetscFree(idx));
  PetscCall(ISRestoreIndices(cellNumbers, &numbers));
  PetscCall(ISDestroy(&cellNumbers));
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputePatchOperator(PC pc, PetscInt point, Vec x, Mat mat, IS cellIS, PetscInt n, const PetscInt dofsArray[], const PetscInt dofsArrayWithAll[], void *ctx)
{
  AppCtx  *user = (AppCtx *)ctx;
  PetscInt ncell;

  PetscFunctionBeginUser;
  PetscCheck(n % user->dofsPerCell == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Patch dof map size is not a multiple of the cell dof count");
  ncell = n / user->dofsPerCell;
  PetscCall(MatZeroEntries(mat));
  /* The patch dof map holds a cell's nodes over all the subspaces together, and a node left out of the patch comes
     back as -1, which MatSetValues() drops */
  for (PetscInt c = 0; c < ncell; ++c) {
    const PetscInt *idx = &dofsArray[c * user->dofsPerCell];

    PetscCall(MatSetValues(mat, user->dofsPerCell, idx, user->dofsPerCell, idx, user->elemMat, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Colors the vertices as PCPATCH does for star patches and checks that the processes needed different numbers of
  colors, so that a process that needed fewer has to pad the patch label with empty strata
*/
static PetscErrorCode CheckUnevenColors(DM dm)
{
  ISColoring coloring;
  PetscInt   ncolors, range[2], globalRange[2];

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateColoringLabel(dm, 0, 1, NULL, 0, &coloring));
  PetscCall(ISColoringGetColors(coloring, NULL, &ncolors, NULL));
  PetscCall(ISColoringDestroy(&coloring));
  range[0] = ncolors;
  range[1] = ncolors;
  PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)dm), range, globalRange));
  PetscCheck(globalRange[0] < globalRange[1], PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Every process colored its vertices with %" PetscInt_FMT " colors, so no process pads the patch label", globalRange[0]);
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

    if (active != NULL) PetscCall(DMLabelGetValue(active, v, &val));
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
  DM              dms[2];
  PetscInt        bs[2]      = {1, 1}, nodesPerCell[2], subspaceOffsets[3];
  PetscInt        nsubspaces = user->mixed == PETSC_TRUE ? 2 : 1;
  const PetscInt *cellNodeMaps[2];

  PetscFunctionBeginUser;
  nodesPerCell[0]    = user->nodesPerCell;
  nodesPerCell[1]    = 1;
  subspaceOffsets[0] = 0;
  subspaceOffsets[1] = user->numVertexDofs;
  subspaceOffsets[2] = user->numDofs;
  cellNodeMaps[0]    = user->cellNodeMap;
  cellNodeMaps[1]    = user->cellDofMap;
  dms[0]             = dm;
  dms[1]             = user->cellDM;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetTolerances(ksp, PETSC_SMALL, PETSC_CURRENT, PETSC_CURRENT, 100));
  PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPATCH));
  PetscCall(PCSetDM(pc, dm));
  PetscCall(PCPatchSetDiscretisationInfo(pc, nsubspaces, dms, bs, nodesPerCell, cellNodeMaps, subspaceOffsets, 0, NULL, 0, NULL));
  PetscCall(PCPatchSetComputeOperator(pc, ComputePatchOperator, user));
  if (patchLabel != NULL) PetscCall(PCPatchSetPatchLabel(pc, patchLabel));
  if (constructLabel != NULL) PetscCall(PCPatchSetConstructLabel(pc, constructLabel, 1));
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

/*
  The label-based comparisons number the patches of each process by themselves and include the vertices it does not
  own, so they hold on one process only
*/
static PetscErrorCode CompareLabeledPatches(DM dm, Mat A, AppCtx *user, PetscInt standardIts)
{
  DMLabel  patchLabel, allLabel, activeLabel;
  PetscInt labeledIts, allIts, activeIts, namedIts;

  PetscFunctionBeginUser;
  /* Supplying the star patches as the strata of a label must reproduce the standard star patches */
  PetscCall(CreateVertexPatchLabel(dm, NULL, &patchLabel));
  PetscCall(SolveWithPatch(dm, A, "labeled_", user, patchLabel, NULL, &labeledIts));
  PetscCheck(standardIts == labeledIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard star patches took %" PetscInt_FMT " iterations, but labeled patches took %" PetscInt_FMT, standardIts, labeledIts);
  PetscCall(DMLabelDestroy(&patchLabel));

  /* Restricting the patches to every mesh point must also leave them unchanged */
  PetscCall(CreateActiveLabel(dm, -1, &allLabel));
  PetscCall(SolveWithPatch(dm, A, "restricted_", user, NULL, allLabel, &allIts));
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
  PetscCall(SolveWithPatch(dm, A, "restricted_", user, NULL, activeLabel, &activeIts));
  PetscCall(SolveWithPatch(dm, A, "labeled_", user, patchLabel, NULL, &labeledIts));
  PetscCheck(activeIts == labeledIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Patches restricted to two cells took %" PetscInt_FMT " iterations, but the same patches listed explicitly took %" PetscInt_FMT, activeIts, labeledIts);
  PetscCall(DMLabelDestroy(&patchLabel));

  /* Naming the same label on the DM must select the same restriction as passing it in */
  PetscCall(DMAddLabel(dm, activeLabel));
  PetscCall(SolveWithPatch(dm, A, "named_", user, NULL, NULL, &namedIts));
  PetscCheck(activeIts == namedIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Patches restricted by a label took %" PetscInt_FMT " iterations, but naming that label on the DM took %" PetscInt_FMT, activeIts, namedIts);
  PetscCall(DMRemoveLabel(dm, "active", NULL));
  PetscCall(DMLabelDestroy(&activeLabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      user;
  DM          dm;
  Mat         A;
  PetscInt    standardIts, coloredIts, vankaIts, coloredVankaIts;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(CreateOperator(dm, &user, &A));
  if (user.unevenColors == PETSC_TRUE) PetscCall(CheckUnevenColors(dm));
  PetscCall(SolveWithPatch(dm, A, "standard_", &user, NULL, NULL, &standardIts));
  PetscCall(SolveWithPatch(dm, A, "colored_", &user, NULL, NULL, &coloredIts));
  PetscCheck(standardIts == coloredIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard star patches took %" PetscInt_FMT " iterations, but colored star patches took %" PetscInt_FMT, standardIts, coloredIts);
  if (size == 1) PetscCall(CompareLabeledPatches(dm, A, &user, standardIts));

  /*
    A Vanka patch spans the closure of the star, and every cell holding one of those points assembles into it, so
    grouping them takes a coloring of wider reach than a star patch needs. Reaching too little leaves a cell coupling
    two patches of a color, which stops the grouped operator from being block diagonal and shows up here as a
    different iteration count.
  */
  PetscCall(SolveWithPatch(dm, A, "vanka_", &user, NULL, NULL, &vankaIts));
  PetscCall(SolveWithPatch(dm, A, "colored_vanka_", &user, NULL, NULL, &coloredVankaIts));
  PetscCheck(vankaIts == coloredVankaIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard Vanka patches took %" PetscInt_FMT " iterations, but colored Vanka patches took %" PetscInt_FMT, vankaIts, coloredVankaIts);

  /*
    Excluding a subspace keeps its dofs at the point a patch was built around and nowhere else, so the patch is no
    longer the dofs of a point set and only the seeds say which points those are. Building the patches around cells
    also colors the cell stratum, which the finite-element adjacency leaves edgeless until the reach passes a cell's
    own closure.
  */
  if (user.mixed == PETSC_TRUE) {
    PetscCall(SolveWithPatch(dm, A, "excluded_vanka_", &user, NULL, NULL, &vankaIts));
    PetscCall(SolveWithPatch(dm, A, "colored_excluded_vanka_", &user, NULL, NULL, &coloredVankaIts));
    PetscCheck(vankaIts == coloredVankaIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard Vanka patches with an excluded subspace took %" PetscInt_FMT " iterations, but colored ones took %" PetscInt_FMT, vankaIts, coloredVankaIts);
  }

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(user.cellNodeMap));
  PetscCall(PetscFree(user.cellDofMap));
  PetscCall(DMDestroy(&user.cellDM));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Grouping patches reproduces the patches it groups only when every patch is solved exactly, so the comparisons
  # below measure the decomposition rather than the accuracy of the default inexact patch solve
  testset:
    args: -standard_pc_patch_construct_type star -standard_sub_ksp_type preonly -standard_sub_pc_type lu \
          -colored_pc_patch_construct_type star -colored_pc_patch_use_coloring -colored_sub_ksp_type preonly -colored_sub_pc_type lu \
          -labeled_sub_ksp_type preonly -labeled_sub_pc_type lu \
          -restricted_pc_patch_construct_type star -restricted_sub_ksp_type preonly -restricted_sub_pc_type lu \
          -named_pc_patch_construct_type star -named_pc_patch_construct_label active -named_sub_ksp_type preonly -named_sub_pc_type lu \
          -vanka_pc_patch_construct_type vanka -vanka_pc_patch_construct_dim 0 -vanka_sub_ksp_type preonly -vanka_sub_pc_type lu \
          -colored_vanka_pc_patch_construct_type vanka -colored_vanka_pc_patch_construct_dim 0 -colored_vanka_pc_patch_use_coloring -colored_vanka_sub_ksp_type preonly -colored_vanka_sub_pc_type lu
    output_file: output/empty.out
    test:
      suffix: patch_color_star
      args: -cells {{4,4 8,8}}
    # The Vanka patches of a cellwise subspace, which is what -pc_patch_exclude_subspaces is for. Building them
    # around cells colors the cell stratum, which the finite-element adjacency leaves edgeless at a reach of one.
    test:
      suffix: patch_color_vanka_mixed
      args: -cells {{4,4 8,8}} -mixed \
            -excluded_vanka_pc_patch_construct_type vanka -excluded_vanka_pc_patch_construct_codim 0 -excluded_vanka_pc_patch_exclude_subspaces 1 -excluded_vanka_sub_ksp_type preonly -excluded_vanka_sub_pc_type lu \
            -colored_excluded_vanka_pc_patch_construct_type vanka -colored_excluded_vanka_pc_patch_construct_codim 0 -colored_excluded_vanka_pc_patch_exclude_subspaces 1 -colored_excluded_vanka_pc_patch_use_coloring -colored_excluded_vanka_sub_ksp_type preonly -colored_excluded_vanka_sub_pc_type lu
    # Vanka patches that leave out every vertex but the one they are built around. The colored patches must keep
    # each seed vertex and drop the other vertices from every color that reaches them.
    test:
      suffix: patch_color_vanka_dim
      args: -cells {{4,4 8,8}} -vanka_pc_patch_vanka_dim 0 -colored_vanka_pc_patch_vanka_dim 0
    # The restricted patches must be colored over the selected points only
    test:
      suffix: patch_color_star_restricted
      args: -cells {{4,4 8,8}} -restricted_pc_patch_use_coloring
    # Coloring each process's own points must give the same patches. This runs on one process,
    # where the two colorings coincide, so it checks that the option reaches PCPATCH
    test:
      suffix: patch_color_star_local
      args: -cells {{4,4 8,8}} -restricted_pc_patch_use_coloring -dm_plex_coloring_local

  # On three processes, the simple partition of this mesh leaves the processes coloring their own vertices with
  # different numbers of colors, which -check_uneven_colors confirms, so a process that needed fewer must pad the
  # patch label with empty strata for the colored patches to match the standard ones
  test:
    suffix: patch_color_star_local_parallel
    nsize: 3
    args: -cells 8,2 -petscpartitioner_type simple -dm_plex_coloring_local -check_uneven_colors \
          -standard_pc_patch_construct_type star -standard_sub_ksp_type preonly -standard_sub_pc_type lu \
          -colored_pc_patch_construct_type star -colored_pc_patch_use_coloring -colored_sub_ksp_type preonly -colored_sub_pc_type lu
    output_file: output/empty.out

TEST*/
