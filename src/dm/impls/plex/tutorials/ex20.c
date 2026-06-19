#include "petscdmlabel.h"
#include "petscis.h"
static char help[] = "Frame transport over manifolds using DMPlex\n\n";

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
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
  PetscCall(PetscObjectSetName((PetscObject)fe, "Along Strike"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeFrame(Vec n, Vec as, DMLabel validFrame)
{
  DM           dm;
  PetscScalar *an, *aas;
  PetscInt     cdim, cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(as, &dm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(VecSet(as, 0.));
  PetscCall(VecGetArrayWrite(n, &an));
  PetscCall(VecGetArrayWrite(as, &aas));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscScalar *nv;
    PetscScalar *asv;
    PetscReal    normal[3], vol;

    PetscCall(DMPlexPointGlobalRef(dm, c, an, &nv));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, normal));
    for (PetscInt d = 0; d < cdim; ++d) nv[d] = normal[d];
    PetscCall(DMPlexPointGlobalRef(dm, c, aas, &asv));
    if (!c) {
      asv[0] = 1.;
      for (PetscInt d = 1; d < cdim; ++d) asv[d] = 0.;
      PetscCall(DMLabelSetValue(validFrame, c, 1));
    }
  }
  PetscCall(VecRestoreArrayWrite(n, &an));
  PetscCall(VecRestoreArrayWrite(as, &aas));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// We take a pair of 2D cells joined by an edge. We rotate the normal of the source cell into the normal of the target cell,
// around the dividing edge. We use this rotation to transform the source along-strike vector into the target one.
// https://math.stackexchange.com/questions/4496301/finding-rotation-matrix-from-two-3d-vectors
static PetscErrorCode PropagateCellFrame(PetscInt face, PetscInt source, PetscInt target, Vec n, Vec as)
{
  DM                 dm;
  const PetscScalar *nvs = NULL, *nvt = NULL, *an, *asvs, *coords;
  PetscScalar       *aas, *asvt, *ecoords, R[9];
  PetscReal          cth, sth, e[3];
  PetscBool          isDG;
  PetscInt           cdim, Nc;

  PetscFunctionBegin;
  PetscCall(VecGetDM(n, &dm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(VecGetArrayRead(n, &an));
  PetscCall(VecGetArray(as, &aas));
  // Normals are normalized
  PetscCall(DMPlexPointGlobalRead(dm, source, an, &nvs));
  PetscCall(DMPlexPointGlobalRead(dm, target, an, &nvt));
  // Compute cos(th) = nvs . nvt
  cth = DMPlex_DotD_Internal(cdim, nvs, nvt);
  // No rotation needed for the same normals
  // if (PetscAbsReal(1. - cth) < PETSC_SMALL) continue;
  // Compute edge vector for rotation axis
  PetscCall(DMPlexGetCellCoordinates(dm, face, &isDG, &Nc, &coords, &ecoords));
  PetscCheck(Nc = cdim * 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Edge had %" PetscInt_FMT " != %" PetscInt_FMT " coordinates", Nc, cdim * 2);
  for (PetscInt d = 0; d < cdim; ++d) e[d] = ecoords[d] - ecoords[d + cdim];
  PetscCall(DMPlexRestoreCellCoordinates(dm, face, &isDG, &Nc, &coords, &ecoords));
  // Normalized edge vector
  PetscReal norm = 0.;
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
  PetscCall(DMPlexPointGlobalRead(dm, source, aas, &asvs));
  PetscCall(DMPlexPointGlobalRef(dm, target, aas, &asvt));
  DMPlex_Mult3D_Internal(R, 1, asvs, asvt);
  {
    // Check that source along strike is normalized
    norm = DMPlex_DotD_Internal(cdim, asvs, asvs);
    PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source along strike vector is not normalized %g", (double)norm);
    // Check that target along strike is normalized
    norm = DMPlex_DotD_Internal(cdim, asvt, asvt);
    PetscCheck(PetscAbsReal(1. - norm) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Target along strike vector is not normalized %g", (double)norm);
  }
  PetscCall(VecRestoreArrayRead(n, &an));
  PetscCall(VecRestoreArray(as, &aas));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PropagatePush(Vec lv, MPI_Op merge)
{
  DM          dm;
  PetscSF     sf;
  PetscInt    Nr;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(VecGetDM(lv, &dm));
  PetscCall(DMGetSectionSF(dm, &sf));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
  PetscCall(PetscSFGetGraph(sf, &Nr, NULL, NULL, NULL));
  if (size > 1 && Nr >= 0) {
    Vec          v;
    PetscScalar *la, *a;
    PetscMemType lmtype, gmtype;

    PetscCall(DMGetGlobalVector(dm, &v));
    PetscCall(VecGetArrayAndMemType(lv, &la, &lmtype));
    PetscCall(VecGetArrayAndMemType(v, &a, &gmtype));
    PetscCall(PetscSFReduceWithMemTypeBegin(sf, MPIU_SCALAR, lmtype, la, gmtype, a, merge));
    PetscCall(PetscSFReduceEnd(sf, MPIU_SCALAR, la, a, merge));
    PetscCall(PetscSFBcastWithMemTypeBegin(sf, MPIU_SCALAR, gmtype, a, lmtype, la, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, a, la, MPI_REPLACE));
    PetscCall(VecRestoreArrayAndMemType(lv, &la));
    PetscCall(VecRestoreArrayAndMemType(v, &a));
    PetscCall(DMRestoreGlobalVector(dm, &v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PropagateLocal(Vec lv, DMPlexPointQueue queue, DMLabel seen, Vec n, Vec as)
{
  DM                 idm;
  PetscSection       s;
  const PetscScalar *an;
  PetscScalar       *aas;
  PetscInt           cStart, cEnd, cdim;

  PetscFunctionBegin;
  PetscCall(VecGetDM(lv, &idm));
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
          PetscScalar *la;
          PetscInt     dof, offN, offAS;

          PetscCall(PetscSectionGetFieldDof(s, face, 0, &dof));
          if (!dof) continue;
          PetscCall(PetscSectionGetFieldOffset(s, face, 0, &offN));
          PetscCall(PetscSectionGetFieldOffset(s, face, 1, &offAS));
          PetscCheck(dof == cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Face %" PetscInt_FMT " has %" PetscInt_FMT " != %" PetscInt_FMT " dof", face, dof, cdim);
          PetscCall(VecGetArrayRead(n, &an));
          PetscCall(VecGetArray(as, &aas));
          PetscCall(VecGetArrayWrite(lv, &la));
          for (PetscInt i = 0; i < dof; ++i) {
            const PetscInt coff = (c - cStart) * dof;

            la[offN + i]  = an[coff + i];
            la[offAS + i] = aas[coff + i];
          }
          PetscCall(VecRestoreArrayRead(n, &an));
          PetscCall(VecRestoreArray(as, &aas));
          PetscCall(VecRestoreArrayWrite(lv, &la));
          continue;
        }
        PetscCheck(sS == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Face %" PetscInt_FMT " has %" PetscInt_FMT " != 2 support points", face, sS);
        q = supp[0] == p ? supp[1] : supp[0];
        PetscCall(DMLabelGetValue(seen, q, &val));
        if (val < 0) {
          PetscCall(PropagateCellFrame(face, p, q, n, as));
          PetscCall(DMLabelSetValue(seen, q, 1));
          PetscCall(DMPlexPointQueueEnqueue(queue, q));
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PropagateFrame(Vec n, Vec as, DMLabel validFrame)
{
  DM               dm, idm;
  Vec              lv;
  PetscSection     s;
  PetscSF          pointSF;
  const PetscInt  *leaves;
  const PetscInt  *degree;
  DMPlexPointQueue queue = NULL;
  PetscBool        empty;
  IS               cellIS;
  const PetscInt  *cells;
  PetscInt         cdim, fStart, fEnd, Nr, Nl, Nc;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(as, &dm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMGetPointSF(dm, &pointSF));

  // Create section for frame propagation along the parallel interface
  PetscCall(DMClone(dm, &idm));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s));
  PetscCall(PetscSectionSetNumFields(s, 2));
  PetscCall(PetscSectionSetFieldName(s, 0, "normal"));
  PetscCall(PetscSectionSetFieldName(s, 1, "along strike"));
  PetscCall(PetscSectionSetChart(s, fStart, fEnd));
  PetscCall(PetscSFGetGraph(pointSF, &Nr, &Nl, &leaves, NULL));
  for (PetscInt l = 0; l < Nl; ++l) {
    const PetscInt p = leaves[l];

    if (p >= fStart && p < fEnd) {
      PetscCall(PetscSectionAddFieldDof(s, p, 0, cdim));
      PetscCall(PetscSectionAddFieldDof(s, p, 1, cdim));
    }
  }
  if (Nr >= 0) {
    PetscCall(PetscSFComputeDegreeBegin(pointSF, &degree));
    PetscCall(PetscSFComputeDegreeEnd(pointSF, &degree));
    for (PetscInt r = 0; r < Nr; ++r) {
      if (degree[r] && (r >= fStart && r < fEnd)) {
        PetscCall(PetscSectionAddFieldDof(s, r, 0, cdim));
        PetscCall(PetscSectionAddFieldDof(s, r, 1, cdim));
      }
    }
  }
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(idm, s));
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(DMGetLocalVector(idm, &lv));

  PetscCall(DMLabelGetStratumIS(validFrame, 1, &cellIS));
  PetscCall(ISGetLocalSize(cellIS, &Nc));
  PetscCall(ISGetIndices(cellIS, &cells));
  PetscCall(DMPlexPointQueueCreate(1024, &queue));
  for (PetscInt i = 0; i < Nc; ++i) {
    PetscCall(DMPlexPointQueueEnqueue(queue, cells[i]));
  }
  PetscCall(ISRestoreIndices(cellIS, &cells));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  while (!empty) {
    PetscCall(PropagateLocal(lv, queue, validFrame, n, as));
    PetscCall(PropagatePush(lv, MPI_SUM));
    PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  }
  PetscCall(DMPlexPointQueueDestroy(&queue));

  PetscCall(DMRestoreLocalVector(idm, &lv));
  PetscCall(DMDestroy(&idm));

#if 0
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt *supp;
    PetscInt        sS;

    PetscCall(DMPlexGetSupport(dm, f, &supp));
    PetscCall(DMPlexGetSupportSize(dm, f, &sS));
    if (sS == 1) continue;
    PetscCheck(sS == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Face must separate 2 cells, not %" PetscInt_FMT, sS);
    PetscCall(PropagateCellFrame(dm, f, supp[0], supp[1], an, aas));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM      dm;
  DMLabel validFrame;
  Vec     n, as;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SetupFE(dm));
  PetscCall(DMCreateLabel(dm, "Valid Frame"));
  PetscCall(DMGetLabel(dm, "Valid Frame", &validFrame));
  PetscCall(DMCreateGlobalVector(dm, &n));
  PetscCall(PetscObjectSetName((PetscObject)n, "Normal"));
  PetscCall(DMCreateGlobalVector(dm, &as));
  PetscCall(PetscObjectSetName((PetscObject)as, "Along Strike"));
  PetscCall(InitializeFrame(n, as, validFrame));
  PetscCall(VecViewFromOptions(n, NULL, "-n_view"));
  PetscCall(VecViewFromOptions(as, NULL, "-as_view"));
  PetscCall(PropagateFrame(n, as, validFrame));
  PetscCall(VecViewFromOptions(as, NULL, "-as_view"));
  PetscCall(VecDestroy(&as));
  PetscCall(VecDestroy(&n));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Use -view_pyvista_glyph_scale 0.3 for better visualization
  test:
    suffix: 0
    args: -cdm_dm_plex_coordinate_dim 3
    output_file: output/empty.out

  test:
    suffix: 1
    args: -cdm_dm_plex_coordinate_dim 3 \
          -dm_coord_remap -dm_coord_map rotate -dm_coord_map_params 0,0,0,1,0,0,0.7853981
    output_file: output/empty.out

  test:
    suffix: 2
    args: -cdm_dm_plex_coordinate_dim 3 \
          -dm_plex_simplex 0 -dm_refine 3 \
          -dm_coord_remap -dm_coord_map muparser -dm_coord_map_func "x,y,0.1*sin(2*_pi*x)"
    output_file: output/empty.out

TEST*/
