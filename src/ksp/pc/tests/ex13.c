static char help[] = "Compares PCPATCH star patches with patches given as a DMLabel.\n\n";

#include <petscksp.h>
#include <petscdmplex.h>

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
  PetscOptionsBegin(comm, NULL, "PCPATCH patch label test options", "PC");
  PetscCall(PetscOptionsIntArray("-cells", "Number of mesh cells in each direction", "ex13.c", user->cells, &n, &flg));
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
  ISLocalToGlobalMapping ltog;
  PetscSection           globalSection;
  IS                     cellNumbers;
  const PetscInt        *numbers;
  PetscInt               cStart, cEnd, numOwned, dNz;

  PetscFunctionBeginUser;
  /* The cell node maps hold local dof numbers, so assemble through the local-to-global map of the section */
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(PetscSectionGetConstrainedStorageSize(globalSection, &numOwned));
  dNz = PetscMin(numOwned, user->nodesPerCell * user->nodesPerCell);
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, numOwned, numOwned, PETSC_DETERMINE, PETSC_DETERMINE, dNz, NULL, dNz, NULL, A));
  PetscCall(MatSetLocalToGlobalMapping(*A, ltog, ltog));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* A cell in the overlap lies on several processes, and only the process that owns it adds its element matrix */
  PetscCall(DMPlexCreateCellNumbering(dm, PETSC_TRUE, &cellNumbers));
  PetscCall(ISGetIndices(cellNumbers, &numbers));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt *idx = &user->cellNodeMap[(c - cStart) * user->nodesPerCell];

    if (numbers[c - cStart] < 0) continue;
    PetscCall(MatSetValuesLocal(*A, user->nodesPerCell, idx, user->nodesPerCell, idx, user->elemMat, ADD_VALUES));
  }
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
  patches that PCPATCH constructs itself.
*/
static PetscErrorCode CreateVertexPatchLabel(DM dm, DMLabel *label)
{
  PetscInt vStart, vEnd, n = 0;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "patches", label));
  for (PetscInt v = vStart; v < vEnd; ++v) PetscCall(DMLabelSetValue(*label, v, n++));
  PetscCall(DMPlexLabelCompleteStar(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveWithPatch(DM dm, Mat A, const char prefix[], AppCtx *user, DMLabel patchLabel, PetscInt *its)
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
  PetscCall(KSPSetTolerances(ksp, PETSC_SMALL, PETSC_CURRENT, PETSC_CURRENT, 100));
  PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCPATCH));
  PetscCall(PCSetDM(pc, dm));
  PetscCall(PCPatchSetDiscretisationInfo(pc, 1, dms, bs, nodesPerCell, cellNodeMaps, subspaceOffsets, 0, NULL, 0, NULL));
  PetscCall(PCPatchSetComputeOperator(pc, ComputePatchOperator, user));
  if (patchLabel != NULL) PetscCall(PCPatchSetPatchLabel(pc, patchLabel));
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
  The label-based comparison numbers the patches of each process by themselves and includes the vertices it does not
  own, so it holds on one process only
*/
static PetscErrorCode CompareLabeledPatches(DM dm, Mat A, AppCtx *user, PetscInt standardIts)
{
  DMLabel  patchLabel;
  PetscInt labeledIts;

  PetscFunctionBeginUser;
  /* Supplying the star patches as the strata of a label must reproduce the standard star patches */
  PetscCall(CreateVertexPatchLabel(dm, &patchLabel));
  PetscCall(SolveWithPatch(dm, A, "labeled_", user, patchLabel, &labeledIts));
  PetscCheck(standardIts == labeledIts, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Standard star patches took %" PetscInt_FMT " iterations, but labeled patches took %" PetscInt_FMT, standardIts, labeledIts);
  PetscCall(DMLabelDestroy(&patchLabel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      user;
  DM          dm;
  Mat         A;
  PetscInt    standardIts;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(CreateOperator(dm, &user, &A));
  PetscCall(SolveWithPatch(dm, A, "standard_", &user, NULL, &standardIts));
  if (size == 1) PetscCall(CompareLabeledPatches(dm, A, &user, standardIts));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(user.cellNodeMap));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: patch_label_star
    args: -cells {{4,4 8,8}} \
          -standard_pc_patch_construct_type star -standard_sub_ksp_type preonly -standard_sub_pc_type lu \
          -labeled_sub_ksp_type preonly -labeled_sub_pc_type lu
    output_file: output/empty.out

TEST*/
