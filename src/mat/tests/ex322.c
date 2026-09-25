static char help[] = "Test compact COO assembly against the original COO stream.\nThe -repeats option controls duplication; fixtures are limited to 1000000 entries.\n";

#include <petscmat.h>

static PetscErrorCode CheckValues(Mat A, Mat B, PetscBool inverse)
{
  Mat                D;
  Vec                x, y, z;
  const PetscScalar *adiag, *bdiag;
  PetscScalar       *xa;
  PetscInt           m, start, end;
  PetscReal          norm;
  PetscBool          assembled;

  PetscFunctionBeginUser;
  PetscCall(MatAssembled(A, &assembled));
  PetscCheck(assembled, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact values were not assembled");
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, B, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(D, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SMALL, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Compact COO matrix error %g", (double)norm);
  PetscCall(MatDestroy(&D));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(y, &z));
  PetscCall(VecGetOwnershipRange(x, &start, &end));
  PetscCall(VecGetArray(x, &xa));
  for (PetscInt i = start; i < end; i++) xa[i - start] = 1.0 + 0.25 * (i % 7);
  PetscCall(VecRestoreArray(x, &xa));
  PetscCall(MatMult(A, x, y));
  PetscCall(MatMult(B, x, z));
  PetscCall(VecAXPY(y, -1.0, z));
  PetscCall(VecNorm(y, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SMALL, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Compact COO matrix action error %g", (double)norm);
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  if (inverse) {
    // Repeated calls prime and then check the cached block-diagonal inverse across value updates.
    PetscCall(MatGetLocalSize(A, &m, NULL));
    PetscCall(MatInvertBlockDiagonal(A, &adiag));
    PetscCall(MatInvertBlockDiagonal(B, &bdiag));
    for (PetscInt i = 0; i < m; i++) PetscCheck(PetscAbsScalar(adiag[i] - bdiag[i]) <= PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stale compact COO diagonal inverse");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckInterleavedValues(Mat A, Mat B, PetscCount ncoo, PetscCount ncompact, const PetscInt map[], PetscScalar values[], PetscScalar compact[], PetscBool inverse)
{
  const InsertMode modes[9] = {INSERT_VALUES, ADD_VALUES, ADD_VALUES, INSERT_VALUES, INSERT_VALUES, ADD_VALUES, ADD_VALUES, ADD_VALUES, INSERT_VALUES};
  PetscMPIInt      rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  // Ordinary and compact updates share the COO metadata and communication buffers.
  for (PetscInt step = 0; step < 9; step++) {
    PetscBool ordinary = (PetscBool)(step == 0 || step == 2 || step == 4 || step == 7);
    PetscBool zero     = (PetscBool)(step == 3 || step == 6);
    PetscBool same;
    MatState  before, after;

    PetscCall(PetscArrayzero(compact, ncompact));
    for (PetscCount k = 0; k < ncoo; k++) {
      values[k] = zero ? 0.0 : (step + 2) * (0.5 + 0.125 * (k % 5) + rank);
#if PetscDefined(USE_COMPLEX)
      if (!zero) values[k] += PETSC_i * (0.25 * (step + 1));
#endif
      if (map[k] != ncompact - 1) compact[map[k]] += values[k];
    }
    compact[ncompact - 1] = PETSC_MAX_REAL;
    PetscCall(MatGetState(A, &before));
    if (ordinary) PetscCall(MatSetValuesCOO(A, values, modes[step]));
    else PetscCall(MatSetValuesCOOCompact(A, zero || !ncoo ? NULL : compact, modes[step]));
    PetscCall(MatGetState(A, &after));
    PetscCall(MatStateCompare(before, after, &same));
    PetscCheck(!same, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interleaved COO assembly did not update the matrix state");
    PetscCall(MatSetValuesCOO(B, values, modes[step]));
    PetscCall(CheckValues(A, B, (PetscBool)(inverse && step != 3)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckGraph(Mat A, Mat B)
{
  const PetscInt *acols, *bcols;
  PetscInt        start, end, an, bn;
  PetscBool       assembled;

  PetscFunctionBeginUser;
  PetscCall(MatAssembled(A, &assembled));
  PetscCheck(assembled, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact structure-only matrix was not assembled");
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt row = start; row < end; row++) {
    PetscCall(MatGetRow(A, row, &an, &acols, NULL));
    PetscCall(MatGetRow(B, row, &bn, &bcols, NULL));
    PetscCheck(an == bn, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact structure-only row length differs");
    for (PetscInt k = 0; k < an; k++) PetscCheck(acols[k] == bcols[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact structure-only column differs");
    PetscCall(MatRestoreRow(A, row, &an, &acols, NULL));
    PetscCall(MatRestoreRow(B, row, &bn, &bcols, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckStructureOnly(MPI_Comm comm, MatType type)
{
  Mat          A, B;
  PetscInt     start, N, rows[6], cols[6], it[6], jt[6];
  PetscInt    *map;
  PetscCount   ncompact;
  PetscScalar *compact;
  PetscBool    nooffproc;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, 2, 2, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(A, type));
  PetscCall(MatSetOption(A, MAT_STRUCTURE_ONLY, PETSC_TRUE));
  PetscCall(MatSetUp(A));
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatGetOwnershipRange(A, &start, NULL));
  PetscCall(MatGetSize(A, &N, NULL));
  rows[0] = rows[1] = rows[2] = start;
  cols[0] = cols[1] = start;
  cols[2]           = (start + 2) % N;
  rows[3] = rows[4] = (start + 2) % N;
  cols[3] = cols[4] = start;
  rows[5]           = -1;
  cols[5]           = 0;
  PetscCall(PetscArraycpy(it, rows, 6));
  PetscCall(PetscArraycpy(jt, cols, 6));
  PetscCall(MatSetPreallocationCOO(A, 6, it, jt));
  PetscCall(MatSetPreallocationCOO(B, 6, rows, cols));
  PetscCall(MatGetValuesCOOCompactMap(A, &ncompact, &map));
  PetscCall(PetscMalloc1(ncompact, &compact));
  for (PetscCount k = 0; k < ncompact; k++) compact[k] = PETSC_MAX_REAL;
  for (PetscInt step = 0; step < 4; step++) {
    InsertMode mode    = step % 2 ? ADD_VALUES : INSERT_VALUES;
    PetscBool  oldflag = (PetscBool)(step >= 2);

    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, oldflag));
    PetscCall(MatSetValuesCOOCompact(A, step < 2 ? NULL : compact, mode));
    PetscCall(MatSetValuesCOO(B, NULL, mode));
    PetscCall(CheckGraph(A, B));
    PetscCall(MatGetOption(A, MAT_NO_OFF_PROC_ENTRIES, &nooffproc));
    PetscCheck(nooffproc == oldflag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Structure-only assembly changed MAT_NO_OFF_PROC_ENTRIES");
  }
  PetscCall(PetscFree(map));
  PetscCall(PetscFree(compact));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckGraphChanges(MPI_Comm comm, MatType type)
{
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  for (PetscInt scenario = 0; scenario < (size > 1 ? 4 : 2); scenario++) {
    Mat          A;
    MatInfo      before, after;
    PetscInt     start, N, rows[2], cols[2];
    PetscInt    *map = NULL;
    PetscCount   ncompact;
    PetscScalar  values[2] = {2.0, 3.0}, value;
    PetscScalar *compact;

    PetscCall(MatCreate(comm, &A));
    PetscCall(MatSetSizes(A, 2, 2, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetType(A, type));
    PetscCall(MatSetUp(A));
    PetscCall(MatGetOwnershipRange(A, &start, NULL));
    PetscCall(MatGetSize(A, &N, NULL));
    rows[0] = start;
    cols[0] = (start + (scenario >= 2 ? 2 : 0)) % N;
    rows[1] = cols[1] = start + 1;
    PetscCall(MatSetPreallocationCOO(A, 2, rows, cols));
    PetscCall(MatSetValuesCOO(A, values, INSERT_VALUES));
    if (scenario % 2) PetscCall(MatGetValuesCOOCompactMap(A, &ncompact, &map));
    PetscCall(MatGetInfo(A, MAT_LOCAL, &before));
    PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_FALSE));
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatZeroRows(A, 1, &start, 0.0, NULL, NULL));
    PetscCall(MatSetValue(A, start, (start + (scenario >= 2 ? 3 : 1)) % N, 4.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatGetInfo(A, MAT_LOCAL, &after));
    PetscCheck(before.nz_used == after.nz_used, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Graph-change fixture must preserve the nonzero count");
    PetscCall(PetscFree(map));
    rows[0] = start;
    cols[0] = (start + (scenario >= 2 ? 2 : 0)) % N;
    rows[1] = cols[1] = start + 1;
    PetscCall(MatSetPreallocationCOO(A, 2, rows, cols));
    PetscCall(MatGetValuesCOOCompactMap(A, &ncompact, &map));
    PetscCall(PetscCalloc1(ncompact, &compact));
    for (PetscInt k = 0; k < 2; k++) compact[map[k]] += values[k];
    PetscCall(MatSetValuesCOOCompact(A, compact, INSERT_VALUES));
    for (PetscInt k = 0; k < 2; k++) {
      PetscInt row = start + k, col = k ? row : (start + (scenario >= 2 ? 2 : 0)) % N;

      PetscCall(MatGetValues(A, 1, &row, 1, &col, &value));
      PetscCheck(value == values[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Repeated preallocation did not restore compact assembly");
    }
    PetscCall(PetscFree(map));
    PetscCall(PetscFree(compact));
    PetscCall(MatDestroy(&A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDuplicateLifetime(MPI_Comm comm, MatType type)
{
  PetscFunctionBeginUser;
  for (PetscInt copy = 0; copy < 2; copy++) {
    Mat          A, B, D;
    PetscInt     start, N, rows[3], cols[3], it[3], jt[3];
    PetscInt    *map, *duplicate_map;
    PetscCount   ncompact, nduplicate;
    PetscScalar  values[3] = {2.0, 3.0, 5.0};
    PetscScalar *compact;
    PetscBool    nooffproc;

    PetscCall(MatCreate(comm, &A));
    PetscCall(MatSetSizes(A, 2, 2, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetType(A, type));
    PetscCall(MatSetUp(A));
    PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
    PetscCall(MatGetOwnershipRange(A, &start, NULL));
    PetscCall(MatGetSize(A, &N, NULL));
    rows[0] = cols[0] = start;
    rows[1] = cols[1] = start + 1;
    rows[2] = cols[2] = (start + 2) % N;
    PetscCall(PetscArraycpy(it, rows, 3));
    PetscCall(PetscArraycpy(jt, cols, 3));
    PetscCall(MatSetPreallocationCOO(A, 3, it, jt));
    PetscCall(MatSetPreallocationCOO(B, 3, rows, cols));
    PetscCall(MatSetValuesCOO(A, values, INSERT_VALUES));
    PetscCall(MatSetValuesCOO(B, values, INSERT_VALUES));
    PetscCall(MatGetValuesCOOCompactMap(A, &ncompact, &map));
    PetscCall(PetscCalloc1(ncompact, &compact));
    for (PetscInt k = 0; k < 3; k++) compact[map[k]] += values[k];
    PetscCall(MatDuplicate(A, copy ? MAT_COPY_VALUES : MAT_DO_NOT_COPY_VALUES, &D));
    PetscCall(MatGetValuesCOOCompactMap(D, &nduplicate, &duplicate_map));
    PetscCheck(nduplicate == ncompact, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Fresh duplicate changed compact size");
    PetscCall(PetscFree(duplicate_map));
    rows[0] = start;
    cols[0] = start + 1;
    PetscCall(MatSetPreallocationCOO(A, 1, rows, cols));
    PetscCall(MatDestroy(&A));
    PetscCall(MatGetValuesCOOCompactMap(D, &nduplicate, &duplicate_map));
    PetscCheck(nduplicate == ncompact, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Source re-preallocation changed duplicate compact size");
    for (PetscInt k = 0; k < 3; k++) PetscCheck(duplicate_map[k] == map[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Source destruction changed duplicate compact map");
    for (PetscInt flag = 0; flag < 2; flag++) {
      PetscCall(MatSetOption(D, MAT_NO_OFF_PROC_ENTRIES, (PetscBool)flag));
      PetscCall(MatSetValuesCOOCompact(D, compact, INSERT_VALUES));
      PetscCall(MatGetOption(D, MAT_NO_OFF_PROC_ENTRIES, &nooffproc));
      PetscCheck(nooffproc == (PetscBool)flag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact assembly changed MAT_NO_OFF_PROC_ENTRIES");
      PetscCall(CheckValues(D, B, PETSC_FALSE));
    }
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&B));
    PetscCall(PetscFree(map));
    PetscCall(PetscFree(duplicate_map));
    PetscCall(PetscFree(compact));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckCancellation(MPI_Comm comm, MatType type)
{
  Mat          A, B, D;
  Vec          x, y, z;
  PetscInt     start, end, N, rows[24], cols[24], rr[4], cc[4];
  PetscInt    *map;
  PetscCount   ncompact;
  PetscScalar  values[24], weights[4], unit = 1.0;
  PetscScalar *compact, *xa;
  PetscReal    terms[6] = {(PetscReal)1 / 10, (PetscReal)1 / 3, -(PetscReal)1 / 3, -(PetscReal)1 / 5, (PetscReal)1 / 7, (PetscReal)2 / 5};
  PetscReal    norm, scale, xnorm, tolerance;
  PetscMPIInt  rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, 2, 2, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(A, type));
  PetscCall(MatSetUp(A));
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatGetOwnershipRange(A, &start, NULL));
  PetscCall(MatGetSize(A, &N, NULL));
#if PetscDefined(USE_COMPLEX)
  unit += 0.25 * PETSC_i;
#endif
  rr[0] = cc[0] = start;
  rr[1]         = start;
  cc[1]         = (start + 2) % N;
  rr[2]         = (start + 2) % N;
  cc[2]         = start;
  rr[3] = cc[3] = start + 1;
  weights[0]    = (rank + 1) * unit;
  weights[1]    = -0.5 * (rank + 2) * unit;
  weights[2]    = 0.25 * (rank + 1) * unit;
  weights[3]    = 2.0 * (rank + 1) * unit;
  for (PetscInt k = 0; k < 6; k++) {
    for (PetscInt j = 0; j < 4; j++) {
      rows[4 * k + j]   = rr[j];
      cols[4 * k + j]   = cc[j];
      values[4 * k + j] = weights[j] * terms[k];
    }
  }
  PetscCall(MatSetPreallocationCOO(A, 24, rows, cols));
  PetscCall(MatGetValuesCOOCompactMap(A, &ncompact, &map));
  PetscCall(PetscCalloc1(ncompact, &compact));
  for (PetscInt k = 0; k < 24; k++) compact[map[k]] += values[k];
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(y, &z));
  PetscCall(VecGetOwnershipRange(x, &start, &end));
  PetscCall(VecGetArray(x, &xa));
  for (PetscInt i = start; i < end; i++) xa[i - start] = (PetscScalar)((i % 5) - 2) + (PetscReal)1 / 9;
  PetscCall(VecRestoreArray(x, &xa));
  PetscCall(VecNorm(x, NORM_INFINITY, &xnorm));
  for (PetscInt step = 0; step < 2; step++) {
    PetscCall(MatSetValuesCOOCompact(A, compact, step ? ADD_VALUES : INSERT_VALUES));
    // The six coefficients sum to 31/70; this oracle does not use the compact map or COO metadata.
    for (PetscInt j = 0; j < 4; j++) PetscCall(MatSetValue(B, rr[j], cc[j], weights[j] * ((PetscReal)31 / 70), ADD_VALUES));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatNorm(B, NORM_INFINITY, &scale));
    // Allow roundoff from cancellation and the different distributed summation orders.
    tolerance = 128 * PETSC_MACHINE_EPSILON * (1.0 + scale);
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &D));
    PetscCall(MatAXPY(D, -1.0, B, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(D, NORM_INFINITY, &norm));
    PetscCheck(norm <= tolerance, comm, PETSC_ERR_PLIB, "Cancellation matrix error %g exceeds roundoff bound %g", (double)norm, (double)tolerance);
    PetscCall(MatDestroy(&D));
    PetscCall(MatMult(A, x, y));
    PetscCall(MatMult(B, x, z));
    PetscCall(VecAXPY(y, -1.0, z));
    PetscCall(VecNorm(y, NORM_INFINITY, &norm));
    tolerance = 128 * PETSC_MACHINE_EPSILON * (1.0 + scale * xnorm);
    PetscCheck(norm <= tolerance, comm, PETSC_ERR_PLIB, "Cancellation matrix action error %g exceeds roundoff bound %g", (double)norm, (double)tolerance);
  }
  PetscCall(PetscFree(map));
  PetscCall(PetscFree(compact));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                    A, B;
  MatType                type;
  IS                     is;
  ISLocalToGlobalMapping l2g;
  PetscInt               m, N, start, end, repeats = 16;
  PetscInt              *rows, *cols, *it, *jt, *map;
  PetscCount             ncoo, ncompact, capacity, counts[2], totals[2];
  PetscScalar           *values, *compact;
  PetscMPIInt            rank, size;
  PetscBool              localapi = PETSC_FALSE, receive_only = PETSC_FALSE, empty_rank = PETSC_FALSE, empty_stream = PETSC_FALSE;
  PetscBool              empty_matrix = PETSC_FALSE, ignore_offproc = PETSC_FALSE, view_counts = PETSC_FALSE;
  PetscBool              nooffproc, equalstate;
  MatState               oldstate, newstate;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-localapi", &localapi, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-receive_only", &receive_only, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank", &empty_rank, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_stream", &empty_stream, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_matrix", &empty_matrix, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ignore_offproc", &ignore_offproc, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_counts", &view_counts, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-repeats", &repeats, NULL));
  PetscCheck(repeats > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Must use positive repeat count");
  m = empty_matrix || (empty_rank && rank == size - 1) ? 0 : 2;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, m, m, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatGetSize(A, &N, NULL));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is, &l2g));
  PetscCall(ISDestroy(&is));
  PetscCall(MatSetLocalToGlobalMapping(A, l2g, l2g));
  PetscCall(MatSetLocalToGlobalMapping(B, l2g, l2g));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  PetscCall(MatSetOption(A, MAT_IGNORE_OFF_PROC_ENTRIES, ignore_offproc));
  PetscCall(MatSetOption(B, MAT_IGNORE_OFF_PROC_ENTRIES, ignore_offproc));
  PetscCheck((PetscCount)N <= (PETSC_COUNT_MAX - 5) / 2, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Matrix size exceeds COO test capacity");
  capacity = (PetscCount)2 * N + 5;
  PetscCheck((PetscCount)repeats <= (PETSC_COUNT_MAX - 2) / capacity, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Repeat count exceeds COO test capacity");
  capacity = capacity * repeats + 2;
  PetscCheck(capacity <= 1000000, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "COO test is limited to 1000000 entries; reduce -repeats or MPI ranks");
  PetscCall(PetscMalloc4(capacity, &rows, capacity, &cols, capacity, &it, capacity, &jt));
  PetscCall(PetscMalloc1(capacity, &values));
  for (PetscInt pass = 0; pass < 2; pass++) {
    ncoo = 0;
    if (!empty_stream && N) {
      if (receive_only) {
        if (!rank)
          for (PetscInt row = 0; row < N; row++) {
            for (PetscInt j = 0; j < repeats; j++) {
              rows[ncoo]   = row;
              cols[ncoo++] = row;
              rows[ncoo]   = row;
              cols[ncoo++] = (row + pass + 1) % N;
            }
          }
      } else if (m)
        for (PetscInt j = 0; j < repeats; j++) {
          rows[ncoo]   = start;
          cols[ncoo++] = start;
          rows[ncoo]   = start + 1;
          cols[ncoo++] = start + 1;
          rows[ncoo]   = start;
          cols[ncoo++] = (start + 2 + pass) % N;
          rows[ncoo]   = (start + 2) % N;
          cols[ncoo++] = (start + 2) % N;
          rows[ncoo]   = 0;
          cols[ncoo++] = 0;
        }
      if (ncoo) {
        rows[ncoo]   = -1;
        cols[ncoo++] = 0;
        rows[ncoo]   = start;
        cols[ncoo++] = -1;
      }
    }
    PetscCall(PetscArraycpy(it, rows, ncoo));
    PetscCall(PetscArraycpy(jt, cols, ncoo));
    if (localapi) PetscCall(MatSetPreallocationCOOLocal(A, ncoo, it, jt));
    else PetscCall(MatSetPreallocationCOO(A, ncoo, it, jt));
    PetscCall(PetscArraycpy(it, rows, ncoo));
    PetscCall(PetscArraycpy(jt, cols, ncoo));
    if (localapi) PetscCall(MatSetPreallocationCOOLocal(B, ncoo, it, jt));
    else PetscCall(MatSetPreallocationCOO(B, ncoo, it, jt));
    PetscCall(MatGetValuesCOOCompactMap(A, &ncompact, &map));
    PetscCheck(ncompact >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing discarded compact slot");
    for (PetscCount k = 0; k < ncoo; k++) {
      PetscCheck(map[k] >= 0 && map[k] < ncompact, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid compact map index");
      PetscCheck(!(rows[k] < 0 || cols[k] < 0 || (ignore_offproc && (rows[k] < start || rows[k] >= end))) || map[k] == ncompact - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Ignored COO entry did not map to discarded slot");
    }
    PetscCall(PetscCalloc1(ncompact, &compact));
    if (view_counts) {
      counts[0] = ncoo;
      counts[1] = ncompact;
      PetscCallMPI(MPIU_Allreduce(counts, totals, 2, MPIU_COUNT, MPI_SUM, PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "pass %" PetscInt_FMT ": original=%" PetscCount_FMT " compact=%" PetscCount_FMT " scalar slots (including discarded slots)\n", pass, totals[0], totals[1]));
    }
    for (PetscInt step = 0; step < 6; step++) {
      InsertMode mode = step == 0 || step == 2 || step == 4 ? INSERT_VALUES : ADD_VALUES;
      PetscBool  zero = (PetscBool)(step == 3 || step == 4);

      PetscCall(PetscArrayzero(compact, ncompact));
      for (PetscCount k = 0; k < ncoo; k++) {
        values[k] = zero ? 0.0 : (step + 1) * (1.0 + 0.25 * (k % 7) + rank);
#if PetscDefined(USE_COMPLEX)
        if (!zero) values[k] += PETSC_i * (0.125 * (step + 1));
#endif
        // Ignored input must not overflow while the producer accumulates values.
        if (rows[k] < 0 || cols[k] < 0 || (ignore_offproc && (rows[k] < start || rows[k] >= end))) values[k] = PETSC_MAX_REAL;
        if (map[k] != ncompact - 1) compact[map[k]] += values[k];
      }
      // Poison the discarded slot to ensure ignored values never enter the matrix.
      compact[ncompact - 1] = PETSC_MAX_REAL;
      PetscCall(MatGetState(A, &oldstate));
      PetscCall(MatSetValuesCOOCompact(A, zero || !ncoo ? NULL : compact, mode));
      PetscCall(MatGetState(A, &newstate));
      PetscCall(MatStateCompare(oldstate, newstate, &equalstate));
      PetscCheck(!equalstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact assembly did not update the matrix state");
      PetscCall(MatSetValuesCOO(B, values, mode));
      PetscCall(CheckValues(A, B, (PetscBool)(!empty_stream && !empty_matrix && step != 4 && !(receive_only && ignore_offproc))));
      PetscCall(MatGetOption(A, MAT_NO_OFF_PROC_ENTRIES, &nooffproc));
      PetscCheck(!nooffproc, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Compact assembly did not restore MAT_NO_OFF_PROC_ENTRIES");
    }
    PetscCall(CheckInterleavedValues(A, B, ncoo, ncompact, map, values, compact, (PetscBool)(!empty_stream && !empty_matrix && !(receive_only && ignore_offproc))));
    {
      Mat        D;
      PetscCount nduplicate;
      PetscInt  *duplicate_map;

      PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &D));
      PetscCall(MatGetValuesCOOCompactMap(D, &nduplicate, &duplicate_map));
      PetscCheck(nduplicate == ncompact, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Duplicate changed compact size");
      for (PetscCount k = 0; k < ncoo; k++) PetscCheck(duplicate_map[k] == map[k], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Duplicate changed compact map");
      PetscCall(MatSetValuesCOOCompact(D, NULL, INSERT_VALUES));
      PetscCall(MatSetValuesCOOCompact(D, !ncoo ? NULL : compact, INSERT_VALUES));
      PetscCall(CheckValues(D, A, PETSC_FALSE));
      PetscCall(PetscFree(duplicate_map));
      PetscCall(MatDestroy(&D));
    }
    PetscCall(PetscFree(compact));
    PetscCall(PetscFree(map));
  }
  PetscCall(MatGetType(A, &type));
  PetscCall(CheckStructureOnly(PETSC_COMM_WORLD, type));
  PetscCall(CheckGraphChanges(PETSC_COMM_WORLD, type));
  PetscCall(CheckDuplicateLifetime(PETSC_COMM_WORLD, type));
  PetscCall(CheckCancellation(PETSC_COMM_WORLD, type));
  PetscCall(PetscFree(values));
  PetscCall(PetscFree4(rows, cols, it, jt));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Compact COO checks passed\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: seq
    args: -mat_type {{seqaij mpiaij}} -localapi {{0 1}}
    output_file: output/ex322.out
  test:
    suffix: mpi
    nsize: 3
    args: -mat_type mpiaij -localapi {{0 1}} -receive_only {{0 1}} -ignore_offproc {{0 1}}
    output_file: output/ex322.out
  test:
    suffix: empty_rank
    nsize: 3
    args: -mat_type mpiaij -empty_rank -localapi {{0 1}}
    output_file: output/ex322.out
  test:
    suffix: empty_stream
    nsize: {{1 3}}
    args: -mat_type aij -empty_stream
    output_file: output/ex322.out
  test:
    suffix: empty_matrix
    nsize: {{1 3}}
    args: -mat_type aij -empty_matrix
    output_file: output/ex322.out
TEST*/
