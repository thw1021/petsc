const char help[] = "Test PETSCFEVECTOR";

#include <petscfe.h>

static PetscErrorCode PetscFEVectorTest(PetscFE orig_fe, PetscInt n_copies, PetscBool interleave_basis, PetscBool interleave_components)
{
  PetscFE          vec_fe, dup_fe;
  PetscQuadrature  quad;
  PetscInt         num_points;
  const PetscReal *points;
  PetscViewer      viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)orig_fe));
  PetscTabulation  orig_T, vec_T, dup_T;
  PetscSpace       space;
  PetscInt         Nb, vNb, Nc, vNc, cdim;
  PetscDualSpace   dual_space;

  PetscFunctionBegin;
  PetscCall(PetscFEGetQuadrature(orig_fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &num_points, &points, NULL));
  PetscCall(PetscFECreateVector(orig_fe, n_copies, interleave_basis, interleave_components, &vec_fe));
  PetscCall(PetscFEView(vec_fe, viewer));
  PetscCall(PetscFECreateTabulation(orig_fe, 1, num_points, points, 1, &orig_T));
  PetscCall(PetscFECreateTabulation(vec_fe, 1, num_points, points, 1, &vec_T));
  PetscCall(PetscFEGetDimension(orig_fe, &Nb));
  PetscCall(PetscFEGetDimension(vec_fe, &vNb));
  PetscCall(PetscFEGetNumComponents(orig_fe, &Nc));
  PetscCall(PetscFEGetNumComponents(vec_fe, &vNc));
  PetscCall(PetscFEGetSpatialDimension(orig_fe, &cdim));
  {
    PetscInt *pre_image;
    PetscInt  c_stride = interleave_components ? n_copies : 1;
    PetscInt  c_incr   = interleave_components ? 1 : Nc;

    PetscCall(PetscMalloc1(vNb, &pre_image));
    for (PetscInt e = 0; e < vNb; e++) pre_image[e] = -1;
    for (PetscInt copy = 0, coffset = 0; copy < n_copies; copy++, coffset += c_incr) {
      for (PetscInt b = 0; b < Nb; b++) {
        for (PetscInt e = 0; e < vNb; e++) {
          PetscReal err = 0.0;

          for (PetscInt k = 0; k <= orig_T->K; k++) {
            const PetscReal *s_Tk   = orig_T->T[k];
            const PetscReal *v_Tk   = vec_T->T[k];
            PetscInt         dblock = PetscPowInt(cdim, k);

            for (PetscInt p = 0; p < num_points; p++) {
              const PetscReal *s_Tp = &s_Tk[(p * Nb + b) * Nc * dblock];
              const PetscReal *v_Tp = &v_Tk[(p * vNb + e) * vNc * dblock];
              for (PetscInt c = 0; c < Nc; c++) {
                PetscInt         vc   = coffset + c * c_stride;
                const PetscReal *s_Tc = &s_Tp[c * dblock];
                const PetscReal *v_Tc = &v_Tp[vc * dblock];
                for (PetscInt d = 0; d < PetscPowInt(cdim, k); d++) err = PetscMax(err, PetscAbsReal(s_Tc[d] - v_Tc[d]));
              }
            }
          }
          if (err < PETSC_SMALL) {
            PetscCheck(pre_image[e] == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Original basis %d and %d both match to vector basis %d\n", (int)pre_image[e], (int)b, (int)e);
            pre_image[e] = b;
          }
        }
      }
    }
    for (PetscInt e = 0; e < vNb; e++) PetscCheck(pre_image[e] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No original basis matched to %d\n", (int)e);
    PetscCall(PetscViewerASCIIPrintf(viewer, "Vector basis to original basis:"));
    for (PetscInt e = 0; e < vNb; e++) {
      if (!(e % 16)) PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, " %3d", (int)pre_image[e]));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscFree(pre_image));
  }
  PetscCall(PetscFEGetBasisSpace(vec_fe, &space));
  PetscCall(PetscFEGetDualSpace(vec_fe, &dual_space));
  PetscCall(PetscObjectReference((PetscObject)space));
  PetscCall(PetscObjectReference((PetscObject)dual_space));
  PetscCall(PetscFECreateFromSpaces(space, dual_space, NULL, NULL, &dup_fe));
  PetscCall(PetscFECreateTabulation(dup_fe, 1, num_points, points, 1, &dup_T));
  {
    PetscReal err = 0.0;

    for (PetscInt k = 0; k <= vec_T->K; k++) {
      PetscInt dblock = PetscPowInt(cdim, k);
      PetscInt size   = num_points * vNb * vNc * dblock;
      for (PetscInt i = 0; i < size; i++) err = PetscMax(err, PetscAbsReal(vec_T->T[k][i] - dup_T->T[k][i]));
    }
    PetscCheck(err < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error between direct tabulation and indirect tabulation: %g\n", (double)err);
  }
  PetscCall(PetscTabulationDestroy(&dup_T));
  PetscCall(PetscTabulationDestroy(&vec_T));
  PetscCall(PetscTabulationDestroy(&orig_T));
  PetscCall(PetscFEDestroy(&dup_fe));
  PetscCall(PetscFEDestroy(&vec_fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFE     scalar, vector;
  PetscViewer viewer;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, 3, 1, PETSC_TRUE, 3, PETSC_DETERMINE, &scalar));
  viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscCall(PetscFEView(scalar, viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  for (PetscInt n_copies = 1; n_copies <= 3; n_copies++) {
    PetscCall(PetscFEVectorTest(scalar, n_copies, PETSC_FALSE, PETSC_FALSE));
    PetscCall(PetscFEVectorTest(scalar, n_copies, PETSC_FALSE, PETSC_TRUE));
    PetscCall(PetscFEVectorTest(scalar, n_copies, PETSC_TRUE, PETSC_FALSE));
    PetscCall(PetscFEVectorTest(scalar, n_copies, PETSC_TRUE, PETSC_TRUE));
  }
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscFEDestroy(&scalar));
  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, 3, 3, PETSC_TRUE, 3, PETSC_DETERMINE, &vector));
  PetscCall(PetscFEView(vector, viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  for (PetscInt n_copies = 1; n_copies <= 3; n_copies++) {
    PetscCall(PetscFEVectorTest(vector, n_copies, PETSC_FALSE, PETSC_FALSE));
    PetscCall(PetscFEVectorTest(vector, n_copies, PETSC_FALSE, PETSC_TRUE));
    PetscCall(PetscFEVectorTest(vector, n_copies, PETSC_TRUE, PETSC_FALSE));
    PetscCall(PetscFEVectorTest(vector, n_copies, PETSC_TRUE, PETSC_TRUE));
  }
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscFEDestroy(&vector));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
