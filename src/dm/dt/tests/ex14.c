const char help[] = "Test FEEC projections.\n\n";

#include <petscdt.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>

static PetscErrorCode createDualSpace(PetscInt dim, DMPolytopeType tope, PetscInt degree, PetscInt formDegree, PetscBool trimmed, PetscInt origDegree, PetscDualSpace *ds_out)
{
  DM             refCell;
  PetscDualSpace ds;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF, tope, &refCell);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreate(PETSC_COMM_SELF, &ds);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(ds, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(ds, refCell);CHKERRQ(ierr);
  ierr = DMDestroy(&refCell);CHKERRQ(ierr);
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
  ierr = PetscDualSpaceLagrangeSetUseMoments(ds, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetMomentOrder(ds, origDegree);CHKERRQ(ierr);
  *ds_out = ds;
  PetscFunctionReturn(0);
}

static PetscErrorCode test(PetscInt dim, DMPolytopeType tope, PetscInt degree, PetscInt formDegree, PetscBool trimmed, PetscInt origDegree)
{
  PetscDualSpace   V, dV;
  PetscRandom      rand;
  PetscInt         Nb, Nform1, Nform2;
  PetscReal       *coeffs, *pScalar1, *pScalar2;
  PetscReal       *eval1, *eval2;
  PetscQuadrature  q1, q2;
  Mat              P1, P2; // projectors
  PetscInt         nPoints1, nPoints2;
  Vec              x1, x2, y1, y2;
  const PetscReal *points1, *points2;

  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = createDualSpace(dim, tope, degree, formDegree, trimmed, origDegree, &V);CHKERRQ(ierr);
  ierr = createDualSpace(dim, tope, trimmed ? degree : degree - 1, PetscAbsInt(formDegree) + 1, trimmed, origDegree, &V);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(V, &q1, &P1);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(dV, &q2, &P2);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q1, NULL, NULL, &nPoints1, &points1, NULL);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q2, NULL, NULL, &nPoints2, &points2, NULL);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand, -1., 1.);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + origDegree, dim, &Nb);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + PetscAbsInt(formDegree), dim, &Nform1);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + PetscAbsInt(formDegree) + 1, dim, &Nform2);CHKERRQ(ierr);
  ierr = PetscMalloc5(Nb * Nform1, &coeffs, Nb * nPoints1, &pScalar1, Nb * (1 + dim) * nPoints2, &pScalar2, nPoints1 * Nform1, &eval1, nPoints2 * Nform2, &eval2);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, nPoints1, points1, origDegree, 0, pScalar1);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, nPoints2, points2, origDegree, 1, pScalar2);CHKERRQ(ierr);
  for (PetscInt i = 0; i < Nb * Nform1; i++) {
    PetscReal val;

    ierr = PetscRandomGetValueReal(rand, &val);CHKERRQ(ierr);
    coeffs[i] = val;
  }
  ierr = PetscArrayzero(eval1, nPoints1 * Nform1);CHKERRQ(ierr);
  {
    PetscBLASInt m = Nform1;
    PetscBLASInt n = nPoints1;
    PetscBLASInt k = Nb;
    PetscBLASInt lda = m;
    PetscBLASInt ldb = nPoints1;
    PetscBLASInt ldc = m;
    PetscReal alpha = 1.;
    PetscReal beta = 0.;
    PetscStackCallBLAS("BLASREALgemm",BLASREALgemm_("N","T",&m,&n,&k,&alpha,coeffs,&lda,pScalar1,&ldb,&beta,eval1,&ldc));
  }
  ierr = MatCreateVecs(P1, &x1, &y1);CHKERRQ(ierr);
  {
    PetscScalar *xa;
    ierr = VecGetArrayWrite(x1, &xa);CHKERRQ(ierr);
    for (PetscInt i = 0; i < nPoints1 * Nform1; i++) {
      xa[i] = eval1[i];
    }
    ierr = VecRestoreArrayWrite(x1, &xa);CHKERRQ(ierr);
  }
  ierr = MatMult(P1, x1, y1);CHKERRQ(ierr);

  ierr = PetscArrayzero(eval2, nPoints2 * Nform2);CHKERRQ(ierr);
  {
    PetscInt (*pattern)[3] = NULL;

    ierr = PetscMalloc1(Nform2 * (PetscAbsInt(formDegree) + 1), &pattern);CHKERRQ(ierr);
    ierr = PetscDTAltVDifferentialPattern(dim, formDegree, PetscAbsInt(formDegree) + 1, pattern);CHKERRQ(ierr);
    for (PetscInt l = 0; l < Nform2 * (PetscAbsInt(formDegree) + 1); l++) {
      PetscInt target = pattern[l][2] < 0 ? -(pattern[l][2] + 1) : pattern[l][2];
      PetscBLASInt m = 1;
      PetscBLASInt n = nPoints2;
      PetscBLASInt k = Nb;
      PetscBLASInt lda = Nform1;
      PetscBLASInt ldb = (1 + dim) * nPoints2;
      PetscBLASInt ldc = Nform2;
      PetscReal alpha = pattern[l][2] < 0 ? -1. : 1.;
      PetscReal beta = 1.;
      PetscStackCallBLAS("BLASREALgemm",BLASREALgemm_("N","T",&m,&n,&k,&alpha,&coeffs[pattern[l][1]],&lda,&pScalar2[(pattern[l][0]+1)*nPoints2],&ldb,&beta,&eval2[target],&ldc));
    }
    ierr = PetscFree(pattern);CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(P2, &x2, &y2);CHKERRQ(ierr);
  {
    PetscScalar *xa;
    ierr = VecGetArrayWrite(x2, &xa);CHKERRQ(ierr);
    for (PetscInt i = 0; i < nPoints2 * Nform2; i++) {
      xa[i] = eval2[i];
    }
    ierr = VecRestoreArrayWrite(x2, &xa);CHKERRQ(ierr);
  }
  ierr = MatMult(P2, x2, y2);CHKERRQ(ierr);

  ierr = VecDestroy(&y2);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&y1);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = PetscFree5(coeffs, pScalar1, pScalar2, eval1, eval2);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&dV);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    args:

TEST*/
