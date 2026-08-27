#include "petscdmlabel.h"
#include "petscsys.h"
#include "petscsystypes.h"
static char help[] = "Frame transport over manifolds using DMPlex.\n\n";


#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>

typedef enum {
  SINGLE_PHASE, // transport strike
  TWO_PHASE, // up x normal (if not horiz), transport strike
  THREE_PHASE, // seed with transport strike, up x normal (if aligned and not flat), transport strike
  NUM_STRIKE_ALGS
} StrikeAlg;
const char *strikeAlgs[NUM_STRIKE_ALGS + 1] = {"single_phase", "two_phase", "three_phase", "unknown"};

typedef struct {
  PetscReal up[3];      // The up direction
  PetscReal start[3];   // Coordinates of the starting point
  PetscBool haveStart;  // Whether the user specified a starting point
  StrikeAlg strikeAlg;  // Algorithm used to determine the strike direction
  PetscReal horizTol;   // Tolerance for detecting horizontal surfaces (up . n > horizTol)
  PetscReal alignedTol; // Tolerance for alignment of up x n with frame transport along-strike direction ((up x n) . as0 > alignedTol)
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt  n = 3, alg;
  PetscBool flg;

  PetscFunctionBeginUser;
  options->up[0]      = 0.;
  options->up[1]      = 0.;
  options->up[2]      = 1.;
  options->start[0]   = 0.;
  options->start[1]   = 0.;
  options->start[2]   = 0.;
  options->haveStart  = PETSC_FALSE;
  options->strikeAlg  = SINGLE_PHASE;
  options->horizTol   = 0.97;
  options->alignedTol = 0.85;

  PetscOptionsBegin(comm, "", "Frame Transport Options", "DMPLEX");
  PetscCall(PetscOptionsRealArray("-start_point", "Coordinates of the starting point.", __FILE__, options->start, &n, &options->haveStart));
  PetscCheck(!options->haveStart || n == 3, comm, PETSC_ERR_ARG_WRONG, "Starting point must be a 3-vector, not %" PetscInt_FMT, n);
  PetscCall(PetscOptionsRealArray("-up", "The up direction", __FILE__, options->up, &n, &flg));
  PetscCheck(!flg || n == 3, comm, PETSC_ERR_ARG_WRONG, "Up direction must be a 3-vector, not %" PetscInt_FMT, n);
  PetscCall(PetscOptionsEList("-strike_alg", "Algorithm to determine the strike direction", __FILE__, strikeAlgs, NUM_STRIKE_ALGS, strikeAlgs[options->strikeAlg], &alg, NULL));
  options->strikeAlg = (StrikeAlg)alg;
  PetscCall(PetscOptionsReal("-horiz_tol", "Tolerance for detecting surface is horizontal.", __FILE__, options->horizTol, &options->horizTol, &flg));
  PetscCall(PetscOptionsReal("-aligned_tol", "Tolerance for alignment of up x n with along-strike seed direction.", __FILE__, options->alignedTol, &options->alignedTol, &flg));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupFE(DM dm)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cdim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, cdim, ct, "as_", -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "strike_dir"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CrossProductUnitVector(const PetscReal a[], const PetscReal b[], PetscReal result[])
{
  PetscReal mag;

  PetscFunctionBeginUser;
  result[0] = a[1]*b[2] - a[2]*b[1];
  result[1] = a[2]*b[0] - a[0]*b[2];
  result[2] = a[0]*b[1] - a[1]*b[0];
  mag = PetscSqrtReal(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);
  if (mag <= PETSC_SMALL) {
      printf("a=(%.4f, %.4f, %.4f) b=(%.4f, %4.f, %.4f)\n", a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  PetscCheck(mag > PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cross product of parallel vectors");
  for (PetscInt i = 0; i < 3; ++i) result[i] /= mag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Find the cell whose centroid is closest to the target point, across all ranks.
// Returns the local cell number of the winner on the owning rank, and -1 on others.
static PetscErrorCode FindClosestCell(DM dm, PetscInt cdim, const PetscReal point[], PetscInt *cTarget)
{
  PetscInt    cStart, cEnd, cBest = -1;
  PetscReal   dBest = PETSC_MAX_REAL;
  PetscMPIInt rank, ownerRank;
  struct {
    PetscReal dist;
    int rank;
  } locVal, glbVal;
  #if defined(PETSC_USE_REAL_SINGLE)
    MPI_Datatype pairtype = MPI_FLOAT_INT;
  #else
    MPI_Datatype pairtype = MPI_DOUBLE_INT;
  #endif

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3], vol, d2 = 0.;

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, centroid, NULL));
    for (PetscInt i = 0; i < cdim; ++i) d2 += PetscSqr(centroid[i] - point[i]);
    if (d2 < dBest) {
      dBest = d2;
      cBest = c;
    }
  }
  printf("cTarget=%d, dist=%.2f\n", cBest, dBest);
  // Determine the globally closest cell via a min-loc reduction
  locVal.dist = dBest;
  locVal.rank = (int)rank;
  PetscCallMPI(MPIU_Allreduce(&locVal, &glbVal, 1, pairtype, MPI_MINLOC, PetscObjectComm((PetscObject)dm)));
  ownerRank = (PetscMPIInt)glbVal.rank;
  *cTarget  = (rank == ownerRank) ? cBest : -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeFrame(Vec n, Vec up, Vec as0, Vec as, DMLabel validFrame, AppCtx *ctx)
{
  DM           dm;
  PetscScalar *an, *aas0, *aas, *aup;
  PetscInt     cdim, cStart, cEnd, cTarget;
  PetscMPIInt  rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)n), &rank));
  PetscCall(VecGetDM(as, &dm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(VecGetArray(n, &an));
  PetscCall(VecGetArray(up, &aup));
  PetscCall(VecGetArray(as0, &aas0));
  PetscCall(VecGetArray(as, &aas));

  cTarget = cStart;
  if (ctx->haveStart) {
      PetscCall(FindClosestCell(dm, cdim, ctx->start, &cTarget));
  }
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscScalar *nv, *upv, *as0v, *asv;
    PetscReal    normal[3], vol;

    PetscCall(DMPlexPointGlobalRef(dm, c, an, &nv));
    if (nv) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, normal));
      for (PetscInt d = 0; d < cdim; ++d) nv[d] = normal[d];
    }
    PetscCall(DMPlexPointGlobalRef(dm, c, aup, &upv));
    PetscCall(DMPlexPointGlobalRef(dm, c, aas0, &as0v));
    PetscCall(DMPlexPointGlobalRef(dm, c, aas, &asv));
    for (PetscInt d = 0; d < cdim; ++d) upv[d] = ctx->up[d];

    if (!rank && (c == cTarget) && upv && asv && nv) {
      PetscCall(CrossProductUnitVector(upv, nv, as0v));
      for (PetscInt d = 0; d < cdim; ++d) asv[d] = as0v[d];
      PetscCall(DMLabelSetValue(validFrame, c, 1));
    }
  }
  PetscCall(VecRestoreArray(n, &an));
  PetscCall(VecRestoreArray(up, &aup));
  PetscCall(VecRestoreArray(as0, &aas0));
  PetscCall(VecRestoreArray(as, &aas));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AnalyticCellFrame(PetscInt face, PetscInt source, PetscInt target, DMLabel validFrame, Vec iframe, Vec n, Vec up, Vec as0, Vec as, PetscBool *updated, AppCtx *ctx)
{
  DM                 dm, idm;
  const PetscScalar *nvs = NULL, *nvt = NULL, *upvt = NULL, *aif = NULL, *an = NULL, *aup = NULL, *asvs = NULL, *aas0 = NULL, *as0vt = NULL;
  PetscScalar       *aas = NULL, *asvt = NULL, tmp[3];
  PetscReal          norm = 0.;
  PetscReal          cth, asdot;
  PetscInt           cdim, seen;

  PetscFunctionBeginUser;
  *updated = PETSC_FALSE;
  PetscCall(DMLabelGetValue(validFrame, target, &seen));
  if (seen == 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetDM(n, &dm));
  PetscCall(VecGetDM(iframe, &idm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(VecGetArrayRead(iframe, &aif));
  PetscCall(VecGetArrayRead(n, &an));
  PetscCall(VecGetArrayRead(up, &aup));
  if (as0) PetscCall(VecGetArrayRead(as0, &aas0));
  PetscCall(VecGetArray(as, &aas));
  // Normals are normalized
  if (source < 0) PetscCall(DMPlexPointLocalFieldRead(idm, face, 0, aif, &nvs));
  else PetscCall(DMPlexPointGlobalRead(dm, source, an, &nvs));
  PetscCall(DMPlexPointGlobalRead(dm, target, an, &nvt));
  PetscCall(DMPlexPointGlobalRead(dm, target, aup, &upvt));
  if (!nvs || !nvt) goto end;
  // Compute cos(th) = nvs . nvt
  cth = DMPlex_DotD_Internal(cdim, nvs, nvt);
  PetscCheck(cth >= -1. - PETSC_SMALL && cth <= 1. + PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source %" PetscInt_FMT " and Target %" PetscInt_FMT " had normal cosine %g", source, target, (double)cth);

  if (source < 0) PetscCall(DMPlexPointLocalFieldRead(idm, face, 1, aif, &asvs));
  else PetscCall(DMPlexPointGlobalRead(dm, source, aas, &asvs));
  PetscCall(DMPlexPointGlobalRef(dm, target, aas, &asvt));
  PetscCheck(asvs && asvt, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing strike direction");
  PetscCall(CrossProductUnitVector(upvt, nvt, tmp));

  if (as0) {
    PetscCall(DMPlexPointGlobalRead(dm, target, aas0, &as0vt));
    asdot = DMPlex_DotD_Internal(cdim, as0vt, tmp);
    if (fabs(asdot) >= ctx->alignedTol) {
        if (asdot > 0.0) {
            for (PetscInt d = 0; d < cdim; ++d) asvt[d] = tmp[d];
        } else {
            for (PetscInt d = 0; d < cdim; ++d) asvt[d] = -tmp[d];
        }
    } else {
        goto end;
    }
  } else {
      cth = DMPlex_DotD_Internal(cdim, asvs, tmp);
      if (cth > 0.) for (PetscInt d = 0; d < cdim; ++d) asvt[d] = tmp[d];
      else if (cth < 0.) for (PetscInt d = 0; d < cdim; ++d) asvt[d] = -tmp[d];
      else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Computed strike direction orthogonal to source strike");
  }

  // Check that source along strike is normalized
  norm = DMPlex_DotD_Internal(cdim, asvs, asvs);
  PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source along strike vector is not normalized %g", (double)norm);
  // Check that target along strike is normalized
  norm = DMPlex_DotD_Internal(cdim, asvt, asvt);
  PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Target along strike vector is not normalized %g", (double)norm);
  PetscCall(DMLabelSetValue(validFrame, target, 1));
  *updated = PETSC_TRUE;
end:
  PetscCall(VecRestoreArrayRead(iframe, &aif));
  PetscCall(VecRestoreArrayRead(up, &aup));
  PetscCall(VecRestoreArrayRead(n, &an));
  PetscCall(VecRestoreArray(as, &aas));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// We take a pair of 2D cells joined by an edge. We rotate the normal of the source cell into the normal of the target cell,
// around the dividing edge. We use this rotation to transform the source along-strike vector into the target one.
// Negative source mean that the frame is in the interface vector
// https://math.stackexchange.com/questions/4496301/finding-rotation-matrix-from-two-3d-vectors
static PetscErrorCode PropagateCellFrame(PetscInt face, PetscInt source, PetscInt target, DMLabel validFrame, Vec iframe, Vec n, Vec as, PetscBool *updated, AppCtx *ctx)
{
  DM                 dm, idm;
  const PetscScalar *nvs = NULL, *nvt = NULL, *aif = NULL, *an = NULL, *asvs = NULL, *coords;
  PetscScalar       *aas = NULL, *asvt = NULL, *ecoords, R[9];
  PetscReal          norm = 0.;
  PetscReal          cth, sth, e[3];
  PetscBool          isDG;
  PetscInt           cdim, Nc, seen;

  PetscFunctionBeginUser;
  *updated = PETSC_FALSE;
  PetscCall(DMLabelGetValue(validFrame, target, &seen));
  if (seen == 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetDM(n, &dm));
  PetscCall(VecGetDM(iframe, &idm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(VecGetArrayRead(iframe, &aif));
  PetscCall(VecGetArrayRead(n, &an));
  PetscCall(VecGetArray(as, &aas));
  // Normals are normalized
  if (source < 0) PetscCall(DMPlexPointLocalFieldRead(idm, face, 0, aif, &nvs));
  else PetscCall(DMPlexPointGlobalRead(dm, source, an, &nvs));
  PetscCall(DMPlexPointGlobalRead(dm, target, an, &nvt));
  if (!nvs || !nvt) goto end;
  // Compute cos(th) = nvs . nvt
  cth = DMPlex_DotD_Internal(cdim, nvs, nvt);
  PetscCheck(cth >= -1. - PETSC_SMALL && cth <= 1. + PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source %" PetscInt_FMT " and Target %" PetscInt_FMT " had normal cosine %g", source, target, (double)cth);
  // No rotation needed for the same normals
  if (PetscAbsReal(1. - cth) < PETSC_SMALL) {
    if (source < 0) PetscCall(DMPlexPointLocalFieldRead(idm, face, 1, aif, &asvs));
    else PetscCall(DMPlexPointGlobalRead(dm, source, aas, &asvs));
    PetscCall(DMPlexPointGlobalRef(dm, target, aas, &asvt));
    PetscCheck(asvs && asvt, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing strike direction");
    for (PetscInt d = 0; d < cdim; ++d) asvt[d] = asvs[d];
    goto check;
  }
  // Compute edge vector for rotation axis
  PetscCall(DMPlexGetCellCoordinates(dm, face, &isDG, &Nc, &coords, &ecoords));
  PetscCheck(Nc == cdim * 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge had %" PetscInt_FMT " != %" PetscInt_FMT " coordinates", Nc, cdim * 2);
  for (PetscInt d = 0; d < cdim; ++d) e[d] = ecoords[d] - ecoords[d + cdim];
  PetscCall(DMPlexRestoreCellCoordinates(dm, face, &isDG, &Nc, &coords, &ecoords));
  // Normalized edge vector
  for (PetscInt d = 0; d < cdim; ++d) norm += PetscSqr(e[d]);
  norm = PetscSqrtReal(norm);
  for (PetscInt d = 0; d < cdim; ++d) e[d] /= norm;
  // Compute sin(th) = (nvs x nvt) . e
  for (PetscInt d = 0; d < cdim; ++d) {
    R[cdim * 0 + d] = nvs[d];
    R[cdim * 1 + d] = nvt[d];
    R[cdim * 2 + d] = e[d];
  }
  DMPlex_Det3D_Scalar_Internal(&sth, R);
  {
    // Check that edge is normalized
    norm = DMPlex_DotD_Internal(cdim, e, e);
    PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge vector is not normalized %g", (double)norm);
    // Check that angles are compatible
    PetscCheck(PetscAbsReal(1. - PetscSqr(cth) - PetscSqr(sth)) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cos %g Sin %g are incompatible", (double)cth, (double)sth);
  }
  // Rodrigues' Rotation formula: R = I + sin(th) e_x + (1 - cos(th)) e^2_sk, where e_sk is the skew-symmetric matrix built from e
  //   e_sk = eps_{ikj} e_k
  R[0] = cth + PetscSqr(e[0]) * (1. - cth);
  R[1] = e[0] * e[1] * (1. - cth) - e[2] * sth;
  R[2] = e[0] * e[2] * (1. - cth) + e[1] * sth;
  R[3] = e[1] * e[0] * (1. - cth) + e[2] * sth;
  R[4] = cth + PetscSqr(e[1]) * (1. - cth);
  R[5] = e[1] * e[2] * (1. - cth) - e[0] * sth;
  R[6] = e[2] * e[0] * (1. - cth) - e[1] * sth;
  R[7] = e[2] * e[1] * (1. - cth) + e[0] * sth;
  R[8] = cth + PetscSqr(e[2]) * (1. - cth);
  if (source < 0) PetscCall(DMPlexPointLocalFieldRead(idm, face, 1, aif, &asvs));
  else PetscCall(DMPlexPointGlobalRead(dm, source, aas, &asvs));
  PetscCall(DMPlexPointGlobalRef(dm, target, aas, &asvt));
  PetscCheck(asvs && asvt, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Missing strike direction");
  DMPlex_Mult3D_Internal(R, 1, asvs, asvt);

check:
  // Check that source along strike is normalized
  norm = DMPlex_DotD_Internal(cdim, asvs, asvs);
  PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source along strike vector is not normalized %g", (double)norm);
  // Check that target along strike is normalized
  norm = DMPlex_DotD_Internal(cdim, asvt, asvt);
  PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Target along strike vector is not normalized %g", (double)norm);
  PetscCall(DMLabelSetValue(validFrame, target, 1));
  *updated = PETSC_TRUE;
end:
  PetscCall(VecRestoreArrayRead(iframe, &aif));
  PetscCall(VecRestoreArrayRead(n, &an));
  PetscCall(VecRestoreArray(as, &aas));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PropagatePush(DMPlexPointQueue queue, DMLabel validFrame, Vec iframe, Vec n, Vec as, MPI_Op merge, AppCtx *ctx)
{
  DM              dm;
  PetscSF         sf, pointSF;
  const PetscInt *leaves, *degree;
  PetscInt        fStart, fEnd, Nl;
  PetscBool       updated;
  PetscInt        Nr;
  PetscMPIInt     size;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(iframe, &dm));
  PetscCall(DMGetSectionSF(dm, &sf));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
  PetscCall(PetscSFGetGraph(sf, &Nr, NULL, NULL, NULL));
  if (size > 1 && Nr >= 0) {
    Vec          v;
    PetscScalar *aif, *a;
    PetscMemType lmtype, gmtype;

    PetscCall(DMGetGlobalVector(dm, &v));
    PetscCall(VecZeroEntries(v));
    PetscCall(VecGetArrayAndMemType(iframe, &aif, &lmtype));
    PetscCall(VecGetArrayAndMemType(v, &a, &gmtype));
    PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_SCALAR, lmtype, aif, gmtype, a, merge));
    PetscCall(PetscSFReduceEnd(sf, MPIU_SCALAR, aif, a, merge));
    PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, gmtype, a, lmtype, aif, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, a, aif, MPI_REPLACE));
    PetscCall(VecRestoreArrayAndMemType(iframe, &aif));
    PetscCall(VecRestoreArrayAndMemType(v, &a));
    PetscCall(DMRestoreGlobalVector(dm, &v));
  }

  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(PetscSFGetGraph(pointSF, &Nr, &Nl, &leaves, NULL));
  for (PetscInt l = 0; l < Nl; ++l) {
    const PetscInt p = leaves[l];

    if (p >= fStart && p < fEnd) {
      const PetscInt *supp;
      PetscInt        sS;

      PetscCall(DMPlexGetSupport(dm, p, &supp));
      PetscCall(DMPlexGetSupportSize(dm, p, &sS));
      PetscCheck(sS == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Support size of boundary face %" PetscInt_FMT " is %" PetscInt_FMT " != 1", p, sS);
      PetscCall(PropagateCellFrame(p, -1, supp[0], validFrame, iframe, n, as, &updated, ctx));
      if (updated) PetscCall(DMPlexPointQueueEnqueue(queue, supp[0]));
    }
  }
  if (Nr >= 0) {
    PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
    PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
    for (PetscInt r = 0; r < Nr; ++r) {
      if (degree[r] && (r >= fStart && r < fEnd)) {
        const PetscInt *supp;
        PetscInt        sS;

        PetscCall(DMPlexGetSupport(dm, r, &supp));
        PetscCall(DMPlexGetSupportSize(dm, r, &sS));
        PetscCheck(sS == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Support size of boundary face %" PetscInt_FMT " is %" PetscInt_FMT " != 1", r, sS);
        PetscCall(PropagateCellFrame(r, -1, supp[0], validFrame, iframe, n, as, &updated, ctx));
        if (updated) PetscCall(DMPlexPointQueueEnqueue(queue, supp[0]));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Propagate field on local process.
 *
 * phase 0: Transport algorithm
 *
 * phase 1: Up x normal.
 *   - If as0 is null, keep same direction as source cell and apply everywhere up x normal > horizTol.
 *   - If as0 is not null, then spply everywhere up x normal is aligned with transport: dot(up x normal, as0) > alignedTOl
 */
static PetscErrorCode PropagateLocal(Vec iframe, DMPlexPointQueue queue, DMLabel validFrame, Vec n, Vec up, Vec as0, Vec as, PetscInt phase, AppCtx *ctx)
{
  DM                 dm, idm;
  PetscSection       s;
  const PetscScalar *an, *anv = NULL, *aup, *aupv = NULL, *aasv = NULL;
  const PetscScalar       *aas;
  PetscReal          cth;
  PetscBool          updated;
  PetscInt           cStart, cEnd, cdim;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(n, &dm));
  PetscCall(VecGetDM(iframe, &idm));
  PetscCall(DMGetCoordinateDim(idm, &cdim));
  PetscCall(DMGetLocalSection(idm, &s));
  PetscCall(DMPlexGetHeightStratum(idm, 0, &cStart, &cEnd));
  while (!DMPlexPointQueueEmpty(queue)) {
    PetscInt p = -1;

    PetscCall(DMPlexPointQueueDequeue(queue, &p));
    if (p >= cStart && p < cEnd) {
      const PetscInt *cone;
      PetscInt        cS;

      PetscCall(DMPlexGetCone(idm, p, &cone));
      PetscCall(DMPlexGetConeSize(idm, p, &cS));
      for (PetscInt c = 0; c < cS; ++c) {
        const PetscInt  face = cone[c];
        const PetscInt *supp;
        PetscInt        sS, q, val;

        PetscCall(DMPlexGetSupport(idm, face, &supp));
        PetscCall(DMPlexGetSupportSize(idm, face, &sS));
        if (sS == 1) {
          PetscScalar *aif;
          PetscInt     dof, offN, offAS;

          PetscCall(PetscSectionGetFieldDof(s, face, 0, &dof));
          if (!dof) continue;
          PetscCall(PetscSectionGetFieldOffset(s, face, 0, &offN));
          PetscCall(PetscSectionGetFieldOffset(s, face, 1, &offAS));
          PetscCheck(dof == cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Face %" PetscInt_FMT " has %" PetscInt_FMT " != %" PetscInt_FMT " dof", face, dof, cdim);
          PetscCall(VecGetArrayRead(n, &an));
          PetscCall(VecGetArrayRead(as, &aas));
          PetscCall(VecGetArray(iframe, &aif));
          PetscCall(DMPlexPointGlobalRead(dm, p, an, &anv));
          PetscCall(DMPlexPointGlobalRead(dm, p, aas, &aasv));
          if (!anv || !aasv) continue;
          for (PetscInt i = 0; i < dof; ++i) {
            aif[offN + i]  = anv[i];
            aif[offAS + i] = aasv[i];
          }
          PetscCall(VecRestoreArrayRead(n, &an));
          PetscCall(VecRestoreArrayRead(as, &aas));
          PetscCall(VecRestoreArray(iframe, &aif));
          continue;
        }
        PetscCheck(sS == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Face %" PetscInt_FMT " has %" PetscInt_FMT " != 2 support points", face, sS);
        q = supp[0] == p ? supp[1] : supp[0];
        PetscCall(DMLabelGetValue(validFrame, q, &val));
        if (val < 0) {
          switch (phase) {
          case 0:
            // Propagate along-strike field using frame transport.
            PetscCall(PropagateCellFrame(face, p, q, validFrame, iframe, n, as, &updated, ctx));
            break;
          case 1:
            // Calculate up x n
            PetscCall(VecGetArrayRead(n, &an));
            PetscCall(VecGetArrayRead(up, &aup));
            PetscCall(DMPlexPointGlobalRead(dm, q, an, &anv));
            PetscCall(DMPlexPointGlobalRead(dm, q, aup, &aupv));
            cth = DMPlex_DotD_Internal(cdim, anv, aupv);
            PetscCall(VecRestoreArrayRead(n, &an));
            PetscCall(VecRestoreArrayRead(up, &aup));

            if (cth <= ctx->horizTol) PetscCall(AnalyticCellFrame(face, p, q, validFrame, iframe, n, up, as0, as, &updated, ctx));
            else updated = PETSC_FALSE;
            break;
            default:
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported phase: %" PetscInt_FMT, phase);
          }
          if (updated) PetscCall(DMPlexPointQueueEnqueue(queue, q));
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void MPIAPI VectorMerge3D_Private(void *a, void *b, int *len, MPI_Datatype *datatype)
{
  const int N = *len / 3;

  if (*datatype == MPIU_SCALAR) {
    PetscScalar *A = (PetscScalar *)a;
    PetscScalar *B = (PetscScalar *)b;

    for (int i = 0; i < N; i++) {
      PetscScalar norm = 0;

      for (int j = 0; j < 3; j++) {
        const int idx = i * 3 + j;

        B[idx] += A[idx];
        norm += PetscSqr(B[idx]);
      }
      norm = PetscSqrtScalar(norm);
      if (norm != 0)
        for (int j = 0; j < 3; j++) B[i * 3 + j] /= norm;
    }
  }
}

static PetscErrorCode PropagateFrame(Vec n, Vec up, Vec as0, Vec as, DMLabel validFrame, PetscInt phase, AppCtx *ctx)
{
  DM               dm, idm;
  Vec              iframe;
  PetscSection     s;
  PetscSF          pointSF;
  const PetscInt  *leaves;
  const PetscInt  *degree;
  DMPlexPointQueue queue = NULL;
  PetscBool        empty;
  IS               cellIS;
  const PetscInt  *cells;
  PetscInt         cdim, fStart, fEnd, Nr, Nl, Nc;
  MPI_Op           reduceop;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Op_create(VectorMerge3D_Private, 0, &reduceop));
  PetscCall(VecGetDM(as, &dm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCheck(cdim == 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Frame transport requires coordinate dimension 3, not %" PetscInt_FMT, cdim);

  // Create section for frame propagation along the parallel interface
  PetscCall(DMClone(dm, &idm));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(PetscSectionSetNumFields(s, 2));
  PetscCall(PetscSectionSetFieldName(s, 0, "normal"));
  PetscCall(PetscSectionSetFieldName(s, 1, "along_strike"));
  PetscCall(PetscSectionSetChart(s, fStart, fEnd));
  PetscCall(PetscSFGetGraph(pointSF, &Nr, &Nl, &leaves, NULL));
  for (PetscInt l = 0; l < Nl; ++l) {
    const PetscInt p = leaves[l];

    if (p >= fStart && p < fEnd) {
      PetscCall(PetscSectionAddDof(s, p, cdim * 2));
      PetscCall(PetscSectionAddFieldDof(s, p, 0, cdim));
      PetscCall(PetscSectionAddFieldDof(s, p, 1, cdim));
    }
  }
  if (Nr >= 0) {
    PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
    PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
    for (PetscInt r = 0; r < Nr; ++r) {
      if (degree[r] && (r >= fStart && r < fEnd)) {
        PetscCall(PetscSectionAddDof(s, r, cdim * 2));
        PetscCall(PetscSectionAddFieldDof(s, r, 0, cdim));
        PetscCall(PetscSectionAddFieldDof(s, r, 1, cdim));
      }
    }
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(idm, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(DMGetLocalVector(idm, &iframe));
  PetscCall(VecZeroEntries(iframe));
  PetscCall(PetscObjectSetName((PetscObject)iframe, "Interface Frame"));

  PetscCall(DMPlexPointQueueCreate(1024, &queue));
  PetscCall(DMLabelGetStratumIS(validFrame, 1, &cellIS));
  if (cellIS) {
    PetscCall(ISGetLocalSize(cellIS, &Nc));
    PetscCall(ISGetIndices(cellIS, &cells));
    for (PetscInt i = 0; i < Nc; ++i) { PetscCall(DMPlexPointQueueEnqueue(queue, cells[i])); }
    PetscCall(ISRestoreIndices(cellIS, &cells));
  }
  PetscCall(ISDestroy(&cellIS));
  PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  while (!empty) {
    PetscCall(PropagateLocal(iframe, queue, validFrame, n, up, as0, as, phase, ctx));
    PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)iframe, (PetscObject)n, "-iframe_view"));
    PetscCall(PropagatePush(queue, validFrame, iframe, n, as, reduceop, ctx));
    PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)iframe, (PetscObject)n, "-iframe_view"));
    PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  }
  PetscCall(DMPlexPointQueueDestroy(&queue));

  PetscCall(DMRestoreLocalVector(idm, &iframe));
  PetscCall(DMDestroy(&idm));
  PetscCallMPI(MPI_Op_free(&reduceop));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * 1. Propagate along-strike using transport.
 */
static PetscErrorCode SinglePhasePropagation(Vec n, Vec up, Vec as0, Vec as, DMLabel validFrame, AppCtx *ctx)
{
  DM       dm;
  PetscInt cStart, cEnd, Nc, Nvc = 0, NvcOld = 0;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(n, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  Nc = cEnd - cStart;
  while (Nvc < Nc) {
    PetscCall(PropagateFrame(n, up, NULL, as, validFrame, 0, ctx));
    DMLabelGetStratumSize(validFrame, 1, &Nvc);
    PetscCheck(Nvc != NvcOld, PetscObjectComm((PetscObject)n), PETSC_ERR_ARG_WRONGSTATE, "Propagation made no progress");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * 1. Set along-strike field (as) to up x n at locations where up x n is not horizontal up x n < horizTol.
 * 2. Propagate along-strike using transport.
 */
static PetscErrorCode TwoPhasePropagation(Vec n, Vec up, Vec as0, Vec as, DMLabel validFrame, AppCtx *ctx)
{
  DM       dm;
  PetscInt cStart, cEnd, Nc, Nvc = 0, NvcOld = 0;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(n, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  Nc = cEnd - cStart;
  while (Nvc < Nc) {
    PetscCall(PropagateFrame(n, up, NULL, as, validFrame, 1, ctx));
    PetscCall(PropagateFrame(n, up, NULL, as, validFrame, 0, ctx));
    DMLabelGetStratumSize(validFrame, 1, &Nvc);
    PetscCheck(Nvc != NvcOld, PetscObjectComm((PetscObject)n), PETSC_ERR_ARG_WRONGSTATE, "Propagation made no progress");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * 1. Set background along-strike field (as0) using tranport, with starting point direction set to up x n.
 * 2. Set along-strike field (as) to up x n at locations where up x n is aligned with background along-strike field (up x n) * as0 > alignedTol and not horizontal up x n < horizTol.
 * 3. Propagate along-strike using transport.
 */
static PetscErrorCode ThreePhasePropagation(Vec n, Vec up, Vec as0, Vec as, DMLabel validFrame, AppCtx *ctx)
{
  DM       dm;
  DMLabel seedFrame;
  PetscInt cStart, cEnd, Nc, Nvc = 0, NvcOld = 0;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(n, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelDuplicate(validFrame, &seedFrame));

  PetscCall(PropagateFrame(n, up, NULL, as0, seedFrame, 0, ctx));
  PetscCall(PropagateFrame(n, up, as0, as, validFrame, 1, ctx));

  PetscCall(DMLabelDestroy(&seedFrame));
  Nc = cEnd - cStart;
  while (Nvc < Nc) {
    PetscCall(PropagateFrame(n, up, as0, as, validFrame, 0, ctx));
    DMLabelGetStratumSize(validFrame, 1, &Nvc);
    PetscCheck(Nvc != NvcOld, PetscObjectComm((PetscObject)n), PETSC_ERR_ARG_WRONGSTATE, "Propagation made no progress");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM      dm;
  DMLabel validFrame;
  Vec     n, as, as0, up;
  AppCtx  ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SetupFE(dm));
  PetscCall(DMCreateLabel(dm, "Valid Frame"));
  PetscCall(DMGetLabel(dm, "Valid Frame", &validFrame));

  PetscCall(DMCreateGlobalVector(dm, &n));
  PetscCall(PetscObjectSetName((PetscObject)n, "normal_dir"));
  PetscCall(VecZeroEntries(n));

  PetscCall(DMCreateGlobalVector(dm, &up));
  PetscCall(PetscObjectSetName((PetscObject)up, "up_dir"));
  PetscCall(VecZeroEntries(up));

  PetscCall(DMCreateGlobalVector(dm, &as0));
  PetscCall(PetscObjectSetName((PetscObject)as0, "strike_seed"));
  PetscCall(VecZeroEntries(as0));

  PetscCall(DMCreateGlobalVector(dm, &as));
  PetscCall(PetscObjectSetName((PetscObject)as, "strike_dir"));
  PetscCall(VecZeroEntries(as));

  PetscCall(InitializeFrame(n, up, as0, as, validFrame, &ctx));
  PetscCall(VecViewFromOptions(n, NULL, "-n_view"));
  PetscCall(VecViewFromOptions(up, NULL, "-up_view"));
  PetscCall(VecViewFromOptions(as0, NULL, "-as0_view"));
  PetscCall(VecViewFromOptions(as, NULL, "-as_view"));
  switch (ctx.strikeAlg) {
  case SINGLE_PHASE:
    PetscCall(SinglePhasePropagation(n, up, NULL, as, validFrame, &ctx));
    break;
  case TWO_PHASE:
    PetscCall(TwoPhasePropagation(n, up, NULL, as, validFrame, &ctx));
    break;
  case THREE_PHASE:
    PetscCall(ThreePhasePropagation(n, up, as0, as, validFrame, &ctx));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported algorithm: %s", strikeAlgs[ctx.strikeAlg]);
  }
  PetscCall(VecViewFromOptions(as, NULL, "-as_view"));
  PetscCall(VecViewFromOptions(as0, NULL, "-as0_view"));
  PetscCall(PetscObjectViewSynchronizedFromOptions((PetscObject)validFrame, (PetscObject)dm, "-valid_frame_view"));
  PetscCall(VecDestroy(&n));
  PetscCall(VecDestroy(&up));
  PetscCall(VecDestroy(&as0));
  PetscCall(VecDestroy(&as));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: !complex

  # Use -view_pyvista_glyph_scale 0.3 for better visualization
  test:
    suffix: 0
    nsize: {{1 2}}
    args: -cdm_dm_plex_coordinate_dim 3 -dm_plex_simplex 0
    output_file: output/empty.out

  test:
    suffix: 1
    requires: triangle
    args: -cdm_dm_plex_coordinate_dim 3 \
          -dm_coord_remap -dm_coord_map rotate -dm_coord_map_params 0,0,0,1,0,0,-0.7853981
    output_file: output/empty.out

  test:
    suffix: 2
    requires: muparser
    args: -cdm_dm_plex_coordinate_dim 3 \
          -dm_plex_simplex 0 -dm_refine 3 \
          -dm_coord_remap -dm_coord_map muparser -dm_coord_map_func "x,y,0.1*sin(2*_pi*x)"
    output_file: output/empty.out

TEST*/
