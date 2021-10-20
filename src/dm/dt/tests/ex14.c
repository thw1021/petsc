const char help[] = "Test FEEC projections.\n\n";

#include <petscdt.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>

static PetscErrorCode createDualSpace(DMPolytopeType tope, PetscInt degree, PetscInt formDegree, PetscBool trimmed, PetscBool useMoments, PetscInt origDegree, PetscDualSpace *ds_out)
{
  DM             refCell;
  PetscDualSpace ds;
  PetscInt       dim, Nf;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  dim = DMPolytopeTypeGetDim(tope);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nf);CHKERRQ(ierr);
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF, tope, &refCell);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreate(PETSC_COMM_SELF, &ds);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(ds, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(ds, refCell);CHKERRQ(ierr);
  ierr = DMDestroy(&refCell);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(ds, Nf);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(ds, degree);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFormDegree(ds, formDegree);CHKERRQ(ierr);
  switch (tope) {
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_TRI_PRISM:
    ierr = PetscDualSpaceLagrangeSetTensor(ds, PETSC_TRUE);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  ierr = PetscDualSpaceLagrangeSetTrimmed(ds, trimmed);CHKERRQ(ierr);
  if (useMoments) {
    ierr = PetscDualSpaceLagrangeSetUseMoments(ds, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetMomentOrder(ds, origDegree);CHKERRQ(ierr);
  }
  ierr = PetscDualSpaceSetUp(ds);CHKERRQ(ierr);
  *ds_out = ds;
  PetscFunctionReturn(0);
}

static PetscErrorCode computeProjector(Mat V, Mat *Proj)
{
  Mat            VVT, S;
  PetscInt       m, n;
  PetscBool      isDense;
  Mat            Vdense = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatTransposeMult(V, V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VVT);CHKERRQ(ierr);
  ierr = MatConvert(VVT, MATSEQDENSE, MAT_INPLACE_MATRIX, &VVT);CHKERRQ(ierr);
  ierr = MatLUFactor(VVT, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(V, &m, &n);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, m, n, NULL, &S);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)V, MATSEQDENSE, &isDense);CHKERRQ(ierr);
  if (isDense) {
    ierr = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
    Vdense = V;
  } else {
    ierr = MatConvert(V, MATSEQDENSE, MAT_INITIAL_MATRIX, &Vdense);CHKERRQ(ierr);
  }
  ierr = MatMatSolve(VVT, Vdense, S);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(V, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Proj);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = MatDestroy(&VVT);CHKERRQ(ierr);
  ierr = MatDestroy(&Vdense);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test(DMPolytopeType tope, PetscInt degree, PetscInt formDegree, PetscBool trimmed, PetscBool useMoments, PetscInt origDegree, PetscReal *error)
{
  PetscDualSpace   X, dX;
  Mat              Proj1, Proj2;
  Mat              E1;
  Mat              V; // vandermonde matrices
  Mat              D2; // differential matrix
  Mat              DP, PD;
  PetscInt         Nb, Nform1, Nform2;
  PetscReal       *pScalar1, *pScalar2;
  PetscQuadrature  q1, q2;
  Mat              P1, P2; // projectors
  PetscInt         nPoints1, nPoints2;
  PetscInt         dim;
  const PetscReal *points1, *points2;

  PetscErrorCode   ierr;

  PetscFunctionBegin;
  dim = DMPolytopeTypeGetDim(tope);
  // create the dual spaces for the two polynomials spaces in the complex
  ierr = createDualSpace(tope, degree, formDegree, trimmed, useMoments, origDegree, &X);CHKERRQ(ierr);
  ierr = createDualSpace(tope, trimmed ? degree : degree - 1, PetscAbsInt(formDegree) + 1, trimmed, useMoments, origDegree, &dX);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(X, &q1, &P1);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(dX, &q2, &P2);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q1, NULL, NULL, &nPoints1, &points1, NULL);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q2, NULL, NULL, &nPoints2, &points2, NULL);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + origDegree, dim, &Nb);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nform1);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree) + 1, &Nform2);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nb * nPoints1, &pScalar1, Nb * (1 + dim) * nPoints2, &pScalar2);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, nPoints1, points1, origDegree, 0, pScalar1);CHKERRQ(ierr);

  // E1: basis evaluated at points1
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, nPoints1 * Nform1, Nb * Nform1, NULL, &E1);CHKERRQ(ierr);
  {
    PetscScalar *a;
    PetscInt b_strl = Nform1 * nPoints1;
    PetscInt f_strl = 1 + b_strl * Nb;
    PetscInt pt_strl = Nform1;

    PetscInt b_strr = nPoints1;
    PetscInt pt_strr = 1;
    ierr = MatDenseGetArrayWrite(E1, &a);CHKERRQ(ierr);
    for (PetscInt b = 0; b < Nb; b++) {
      for (PetscInt f = 0; f < Nform1; f++) {
        for (PetscInt pt = 0; pt < nPoints1; pt++) {
          a[b * b_strl + f * f_strl + pt * pt_strl] = pScalar1[b * b_strr + pt * pt_strr];
        }
      }
    }
    ierr = MatDenseRestoreArrayWrite(E1, &a);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)E1, "Evaluation");CHKERRQ(ierr);
  ierr = MatMatMult(P1, E1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &V);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)V, "Vandermonde");CHKERRQ(ierr);
  ierr = computeProjector(V, &Proj1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Proj1, "Vandermonde projector");CHKERRQ(ierr);


  ierr = PetscDTPKDEvalJet(dim, nPoints2, points2, origDegree, 1, pScalar2);CHKERRQ(ierr);
  // D2: the differential of the basis evaluated at points2
  ierr = MatCreateSeqDense(PETSC_COMM_SELF, Nform2 * nPoints2, Nb * Nform1, NULL, &D2);CHKERRQ(ierr);
  {
    PetscScalar *a;
    PetscInt (*pattern)[3] = NULL;
    PetscInt nnz = Nform2 * (PetscAbsInt(formDegree) + 1);
    PetscInt b_strl = Nform2 * nPoints2;
    PetscInt fj_strl = b_strl * Nform1;
    PetscInt fi_strl = 1;
    PetscInt pt_strl = Nform2;

    PetscInt b_strr = nPoints2 * (1 + dim);
    PetscInt jet_strr = nPoints2;
    PetscInt pt_strr = 1;

    ierr = PetscMalloc1(nnz, &pattern);CHKERRQ(ierr);
    ierr = PetscDTAltVDifferentialPattern(dim, formDegree, PetscAbsInt(formDegree) + 1, pattern);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(D2, &a);CHKERRQ(ierr);
    for (PetscInt l = 0; l < nnz; l++) {
      PetscInt jetc = pattern[l][0];
      PetscInt fj = pattern[l][1];
      PetscInt fi = pattern[l][2];
      PetscInt scale = 1.;
      if (fi < 0) {
        scale = -1.;
        fi = -(fj+1);
      }
      if (formDegree < 0) {
        fj = Nform1 - 1 - fj;
        if (fj & 1) {
          scale *= -1;
        }
      }
      for (PetscInt b = 0; b < Nb; b++) {
        for (PetscInt pt = 0; pt < nPoints2; pt++) {
          a[b * b_strl + fi * fi_strl + fj * fj_strl + pt * pt_strl] += scale * pScalar2[b * b_strr + (1 + jetc) * jet_strr + pt * pt_strr];
        }
      }
    }
    ierr = MatDenseRestoreArrayWrite(D2, &a);CHKERRQ(ierr);
    ierr = PetscFree(pattern);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject)D2, "Differential evaluation");CHKERRQ(ierr);
  ierr = PetscQuadratureView(q2, NULL);CHKERRQ(ierr);
  ierr = computeProjector(P2, &Proj2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Proj2, "Second projector");CHKERRQ(ierr);

  ierr = MatMatMult(D2, Proj1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DP);CHKERRQ(ierr);
  ierr = MatMatMult(Proj2, D2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PD);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)DP, "DP");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)PD, "PD");CHKERRQ(ierr);
  ierr = MatAXPY(DP, -1.0, PD, SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatNorm(DP, NORM_INFINITY, error);CHKERRQ(ierr);

  ierr = MatDestroy(&PD);CHKERRQ(ierr);
  ierr = MatDestroy(&DP);CHKERRQ(ierr);
  ierr = MatDestroy(&Proj2);CHKERRQ(ierr);
  ierr = MatDestroy(&D2);CHKERRQ(ierr);
  ierr = MatDestroy(&Proj1);CHKERRQ(ierr);
  ierr = MatDestroy(&V);CHKERRQ(ierr);
  ierr = MatDestroy(&E1);CHKERRQ(ierr);
  ierr = PetscFree2(pScalar1, pScalar2);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&dX);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DMPolytopeType topes[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_TETRAHEDRON, DM_POLYTOPE_HEXAHEDRON, DM_POLYTOPE_TRI_PRISM};
  PetscInt       ntopes = sizeof(topes) / sizeof (DMPolytopeType);
  PetscInt       degree = 2;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  for (PetscInt t = 0; t < ntopes; t++) {
    DMPolytopeType tope = topes[t];
    PetscInt dim = DMPolytopeTypeGetDim(tope);
    for (PetscInt formDegree = -dim+1; formDegree < dim; formDegree++) {
      for (PetscInt trimmed = 0; trimmed <= (formDegree == 0 ? 0 : 1); trimmed++) {
        for (PetscInt useMoments = 0; useMoments <= 1; useMoments++) {
          PetscReal error;
          PetscInt  origDegree = degree + 1;
          switch (tope) {
          case DM_POLYTOPE_QUADRILATERAL:
          case DM_POLYTOPE_TRI_PRISM:
            origDegree = 2* degree + 1;
            break;
          case DM_POLYTOPE_HEXAHEDRON:
            origDegree = 3 * degree + 1;
            break;
          default:
            break;
          }

          ierr = test(tope, degree, formDegree, trimmed, useMoments, origDegree, &error);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD, "%s, form degree %D, %s, degree %D, origDegree %D, %s, projection commutator error %g\n", DMPolytopeTypes[tope], formDegree, trimmed ? "trimmed" : "full", degree, origDegree, useMoments ? "modal" : "nodal", (double) error);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    args:

TEST*/
