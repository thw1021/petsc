const char help[] = "Example of constraining nodal quadratic finite element to be linear.\n"
                    "This is accomplished by constraints that average vertex values, which\n"
                    "works because the discretization is nodal and because the constrained\n"
                    "degrees of freedom are at the centroids of their cells.\n";

#include <petscdmplex.h>
#include <petscsection.h>

static PetscErrorCode DetermineConstraint(DM dm, PetscInt i, PetscInt pStart, PetscBool3 constrained_point[])
{
  DMPolytopeType type;
  PetscInt  dim, cdim;

  PetscFunctionBegin;
  if (constrained_point[i - pStart] != PETSC_BOOL3_UNKNOWN) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetCellType(dm, i, &type));
  switch (type) {
  case DM_POLYTOPE_POINT:
    constrained_point[i - pStart] = PETSC_BOOL3_FALSE;
    break;
  case DM_POLYTOPE_SEGMENT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
    {
      PetscReal xpart, norm;
      PetscReal v[3];
      PetscReal J[9];
      PetscReal invJ[9];
      PetscReal detJ;
      PetscInt       constraint_direction = 0; // 0 = x, 1 = y, 2 = z
      PetscReal      thresh = PETSC_SMALL;

      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, i, v, J, invJ, &detJ));

      xpart = PetscAbsReal(J[dim * constraint_direction]);
      norm = 0.0;
      for (PetscInt i = 0; i < cdim; i++) norm += J[i * dim] * J[i * dim];
      norm = PetscSqrtReal(norm);
      if (PetscSqrtReal(1.0 - xpart / norm) < thresh) constrained_point[i - pStart] = PETSC_BOOL3_TRUE;
      else constrained_point[i - pStart] = PETSC_BOOL3_FALSE;
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
  case DM_POLYTOPE_HEXAHEDRON: {
    PetscInt        nCone;
    const PetscInt *cone;
    PetscBool       any_constrained = PETSC_FALSE;

    PetscCall(DMPlexGetConeSize(dm, i, &nCone));
    PetscCall(DMPlexGetCone(dm, i, &cone));
    for (PetscInt j = 0; j < nCone; j++) {
      PetscInt p = cone[j];
      PetscCall(DetermineConstraint(dm, p, pStart, constrained_point));
      if (constrained_point[p - pStart] == PETSC_BOOL3_TRUE) any_constrained = PETSC_TRUE;
    }
    if (any_constrained) constrained_point[i - pStart] = PETSC_BOOL3_TRUE;
    else constrained_point[i - pStart] = PETSC_BOOL3_FALSE;
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported element type for this example");
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm        comm;
  DM              dm;
  PetscInt        dim         = 2;
  PetscInt        cStart, cEnd, pStart, pEnd;
  PetscInt        polytope_type_int = -1;
  PetscInt        Nc = 3;
  DMPolytopeType  cell_type;
  PetscFE         fe;
  PetscSection    local_section, global_section;
  PetscBool3      *constrained_point;
  PetscSection    anchorSection;
  PetscInt        *anchors;
  IS               anchorIS;
  PetscInt         n_anchors;
  Mat              constraint_mat;
  PetscSection     constraint_section;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  if (cEnd > cStart) {
    DMPolytopeType type;

    PetscCall(DMPlexGetCellType(dm, cStart, &type));
    polytope_type_int = (PetscInt) type;
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &polytope_type_int, 1, MPIU_INT, MPI_MAX, comm));
  cell_type = (DMPolytopeType) polytope_type_int;
  PetscCall(PetscFECreateLagrangeByCell(comm, dim, Nc, cell_type, 2, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "Quadratic"));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(PetscFEDestroy(&fe));

  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

  PetscCall(PetscMalloc1(pEnd - pStart, &constrained_point));


  for (PetscInt i = 0; i < pEnd - pStart; i++) constrained_point[i] = PETSC_BOOL3_UNKNOWN;

  for (PetscInt i = pStart; i < pEnd; i++) PetscCall(DetermineConstraint(dm, i, pStart, constrained_point));

  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &anchorSection));
  PetscCall(PetscSectionSetChart(anchorSection, pStart, pEnd));

  for (PetscInt i = pStart; i < pEnd; i++) {
    PetscInt nCone;
    PetscInt n_unconstrained = 0;
    const PetscInt *cone;

    if (constrained_point[i - pStart] != PETSC_BOOL3_TRUE) {
      PetscCall(PetscSectionSetDof(anchorSection, i, 0));
      continue;
    }

    PetscCall(DMPlexGetConeSize(dm, i, &nCone));
    PetscCall(DMPlexGetCone(dm, i, &cone));

    for (PetscInt j = 0; j < nCone; j++) {
      PetscInt p = cone[j];

      if (constrained_point[p - pStart] != PETSC_BOOL3_TRUE) n_unconstrained++;
    }
    PetscCall(PetscSectionSetDof(anchorSection, i, n_unconstrained));
  }

  PetscCall(PetscSectionSetUp(anchorSection));
  PetscCall(PetscSectionGetStorageSize(anchorSection, &n_anchors));
  PetscCall(PetscMalloc1(n_anchors, &anchors));

  for (PetscInt i = pStart; i < pEnd; i++) {
    PetscInt nCone;
    PetscInt n_unconstrained = 0;
    PetscInt off;
    const PetscInt *cone;

    if (constrained_point[i - pStart] != PETSC_BOOL3_TRUE) {
      continue;
    }

    PetscCall(DMPlexGetConeSize(dm, i, &nCone));
    PetscCall(DMPlexGetCone(dm, i, &cone));
    PetscCall(PetscSectionGetOffset(anchorSection, i, &off));

    for (PetscInt j = 0; j < nCone; j++) {
      PetscInt p = cone[j];

      if (constrained_point[p - pStart] != PETSC_BOOL3_TRUE) {
        anchors[off + n_unconstrained++] = p;
      }
    }
  }

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n_anchors, anchors, PETSC_COPY_VALUES, &anchorIS));

  PetscCall(DMPlexSetAnchors(dm, anchorSection, anchorIS));

  PetscCall(ISDestroy(&anchorIS));

  PetscCall(PetscFree(constrained_point));

  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_(comm)));
  PetscCall(DMGetLocalSection(dm, &local_section));
  PetscCall(PetscSectionView(local_section, PETSC_VIEWER_STDOUT_(comm)));
  PetscCall(DMGetGlobalSection(dm, &global_section));
  PetscCall(PetscSectionView(global_section, PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(DMGetDefaultConstraints(dm, &constraint_section, &constraint_mat, NULL));
  PetscCall(PetscSectionView(constraint_section, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(MatZeroEntries(constraint_mat));

  for (PetscInt i = pStart; i < pEnd; i++) {
    PetscInt dof, off;
    PetscInt adof, aoff;

    PetscCall(PetscSectionGetDof(constraint_section, i, &dof));
    if (!dof) continue;
    PetscCall(PetscSectionGetOffset(constraint_section, i, &off));
    PetscCall(PetscSectionGetDof(anchorSection, i, &adof));
    PetscCall(PetscSectionGetOffset(anchorSection, i, &aoff));
    for (PetscInt j = 0; j < adof; j++) {
      PetscInt p = anchors[aoff + j];
      PetscInt ldof, loff;

      PetscCall(PetscSectionGetDof(local_section, p, &ldof));
      PetscCall(PetscSectionGetOffset(local_section, p, &loff));
      PetscCheck(ldof == dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong number of dofs at anchor point");
      for (PetscInt k = 0; k < dof; k++) {
        PetscCall(MatSetValue(constraint_mat, off + k, loff + k, 1.0 / adof, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(constraint_mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(constraint_mat, MAT_FINAL_ASSEMBLY));

  PetscCall(MatView(constraint_mat, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscFree(anchors));
  PetscCall(PetscSectionDestroy(&anchorSection));

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 1
    args: -dm_distribute -dm_plex_dim 2 -dm_plex_box_faces 3,3,3 -dm_plex_simplex 0

TEST*/
