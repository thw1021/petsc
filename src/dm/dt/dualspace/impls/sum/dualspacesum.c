#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@
  PetscDualSpaceSumGetNumSubspaces - Get the number of spaces in the sum space

  Input Parameter:
. sp - the dual space object

  Output Parameter:
. numSumSpaces - the number of spaces

  Level: intermediate

  Note:
  The name NumSubspaces is slightly misleading because it is actually getting the number of defining spaces of the sum, not a number of Subspaces of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumSetNumSubspaces()`
@*/
PetscErrorCode PetscDualSpaceSumGetNumSubspaces(PetscDualSpace sp, PetscInt *numSumSpaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscAssertPointer(numSumSpaces, 2);
  PetscTryMethod(sp, "PetscDualSpaceSumGetNumSubspaces_C", (PetscDualSpace, PetscInt *), (sp, numSumSpaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetNumSubspaces - Set the number of spaces in the sum space

  Input Parameters:
+ sp           - the dual space object
- numSumSpaces - the number of spaces

  Level: intermediate

  Note:
  The name NumSubspaces is slightly misleading because it is actually setting the number of defining spaces of the sum, not a number of Subspaces of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumGetNumSubspaces()`
@*/
PetscErrorCode PetscDualSpaceSumSetNumSubspaces(PetscDualSpace sp, PetscInt numSumSpaces)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumSetNumSubspaces_C", (PetscDualSpace, PetscInt), (sp, numSumSpaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumGetConcatenate - Get the concatenate flag for this space.

  Input Parameter:
. sp - the dual space object

  Output Parameter:
. concatenate - flag indicating whether subspaces are concatenated.

  Level: intermediate

  Note:
  A concatenated sum space will have the number of components equal to the sum of the number of
  components of all subspaces. A non-concatenated, or direct sum space will have the same
  number of components as its subspaces.

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumSetConcatenate()`
@*/
PetscErrorCode PetscDualSpaceSumGetConcatenate(PetscDualSpace sp, PetscBool *concatenate)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumGetConcatenate_C", (PetscDualSpace, PetscBool *), (sp, concatenate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetConcatenate - Sets the concatenate flag for this space.

  Input Parameters:
+ sp          - the dual space object
- concatenate - are subspaces concatenated components (true) or direct summands (false)

  Level: intermediate

  Notes:
  A concatenated sum space will have the number of components equal to the sum of the number of
  components of all subspaces. A non-concatenated, or direct sum space will have the same
  number of components as its subspaces .

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumGetConcatenate()`
@*/
PetscErrorCode PetscDualSpaceSumSetConcatenate(PetscDualSpace sp, PetscBool concatenate)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumSetConcatenate_C", (PetscDualSpace, PetscBool), (sp, concatenate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumGetSubspace - Get a space in the sum space

  Input Parameters:
+ sp - the dual space object
- s  - The space number

  Output Parameter:
. subsp - the `PetscDualSpace`

  Level: intermediate

  Note:
  The name GetSubspace is slightly misleading because it is actually getting one of the defining spaces of the sum, not a Subspace of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumSetSubspace()`
@*/
PetscErrorCode PetscDualSpaceSumGetSubspace(PetscDualSpace sp, PetscInt s, PetscDualSpace *subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscAssertPointer(subsp, 3);
  PetscTryMethod(sp, "PetscDualSpaceSumGetSubspace_C", (PetscDualSpace, PetscInt, PetscDualSpace *), (sp, s, subsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetSubspace - Set a space in the sum space

  Input Parameters:
+ sp    - the dual space object
. s     - The space number
- subsp - the number of spaces

  Level: intermediate

  Note:
  The name SetSubspace is slightly misleading because it is actually setting one of the defining spaces of the sum, not a Subspace of it

.seealso: `PETSCDUALSPACESUM`, `PetscDualSpace`, `PetscDualSpaceSumGetSubspace()`
@*/
PetscErrorCode PetscDualSpaceSumSetSubspace(PetscDualSpace sp, PetscInt s, PetscDualSpace subsp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (subsp) PetscValidHeaderSpecific(subsp, PETSCDUALSPACE_CLASSID, 3);
  PetscTryMethod(sp, "PetscDualSpaceSumSetSubspace_C", (PetscDualSpace, PetscInt, PetscDualSpace), (sp, s, subsp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetNumSubspaces_Sum(PetscDualSpace space, PetscInt *numSumSpaces)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;

  PetscFunctionBegin;
  *numSumSpaces = sum->numSumSpaces;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetNumSubspaces_Sum(PetscDualSpace space, PetscInt numSumSpaces)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;
  PetscInt            Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(!sum->setupCalled, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Cannot change number of subspaces after setup called");
  if (numSumSpaces == Ns) PetscFunctionReturn(PETSC_SUCCESS);
  if (Ns >= 0) {
    PetscInt s;
    for (s = 0; s < Ns; ++s) PetscCall(PetscDualSpaceDestroy(&sum->sumspaces[s]));
    PetscCall(PetscFree(sum->sumspaces));
  }

  Ns = sum->numSumSpaces = numSumSpaces;
  PetscCall(PetscCalloc1(Ns, &sum->sumspaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetConcatenate_Sum(PetscDualSpace sp, PetscBool *concatenate)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  *concatenate = sum->concatenate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetConcatenate_Sum(PetscDualSpace sp, PetscBool concatenate)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;

  PetscFunctionBegin;
  PetscCheck(!sum->setupCalled, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change space concatenation after setup called.");

  sum->concatenate = concatenate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetSubspace_Sum(PetscDualSpace space, PetscInt s, PetscDualSpace *subspace)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;
  PetscInt            Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Must call PetscDualSpaceSumSetNumSubspaces() first");
  PetscCheck(s >= 0 && s < Ns, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_OUTOFRANGE, "Invalid subspace number %" PetscInt_FMT, s);

  *subspace = sum->sumspaces[s];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetSubspace_Sum(PetscDualSpace space, PetscInt s, PetscDualSpace subspace)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)space->data;
  PetscInt            Ns  = sum->numSumSpaces;

  PetscFunctionBegin;
  PetscCheck(!sum->setupCalled, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Cannot change subspace after setup called");
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_WRONGSTATE, "Must call PetscDualSpaceSumSetNumSubspaces() first");
  PetscCheck(s >= 0 && s < Ns, PetscObjectComm((PetscObject)space), PETSC_ERR_ARG_OUTOFRANGE, "Invalid subspace number %" PetscInt_FMT, s);

  PetscCall(PetscObjectReference((PetscObject)subspace));
  PetscCall(PetscDualSpaceDestroy(&sum->sumspaces[s]));
  sum->sumspaces[s] = subspace;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceDuplicate_Sum(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscInt       num_subspaces, Nc;
  PetscBool      concatenate, interleave_basis, interleave_components;
  PetscDualSpace subsp_first     = NULL;
  PetscDualSpace subsp_dup_first = NULL;
  DM             K;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &num_subspaces));
  PetscCall(PetscDualSpaceSumSetNumSubspaces(spNew, num_subspaces));
  PetscCall(PetscDualSpaceSumGetConcatenate(sp, &concatenate));
  PetscCall(PetscDualSpaceSumSetConcatenate(spNew, concatenate));
  PetscCall(PetscDualSpaceSumGetInterleave(sp, &interleave_basis, &interleave_components));
  PetscCall(PetscDualSpaceSumSetInterleave(spNew, interleave_basis, interleave_components));
  PetscCall(PetscDualSpaceGetDM(sp, &K));
  PetscCall(PetscDualSpaceSetDM(spNew, K));
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceSetNumComponents(spNew, Nc));
  for (PetscInt s = 0; s < num_subspaces; s++) {
    PetscDualSpace subsp, subspNew;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
    if (s == 0) {
      subsp_first = subsp;
      PetscCall(PetscDualSpaceDuplicate(subsp, &subsp_dup_first));
      subspNew = subsp_dup_first;
    } else if (subsp == subsp_first) {
      PetscCall(PetscObjectReference((PetscObject)subsp_dup_first));
      subspNew = subsp_dup_first;
    } else {
      PetscCall(PetscDualSpaceDuplicate(subsp, &subspNew));
    }
    PetscCall(PetscDualSpaceSumSetSubspace(spNew, s, subspNew));
    PetscCall(PetscDualSpaceDestroy(&subspNew));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscQuadratureConcatenate(PetscInt num_quads, PetscQuadrature subquads[], PetscQuadrature *fullquad)
{
  PetscFunctionBegin;
  PetscInt        cdim;
  PetscInt        Npoints;
  PetscReal      *points;
  PetscQuadrature quad;

  PetscCall(PetscQuadratureGetData(subquads[0], &cdim, NULL, NULL, NULL, NULL));
  Npoints = 0;
  for (PetscInt s = 0; s < num_quads; s++) {
    PetscInt sNpoints;

    PetscCall(PetscQuadratureGetData(subquads[s], NULL, NULL, &sNpoints, NULL, NULL));
    Npoints += sNpoints;
  }
  PetscCall(PetscMalloc1(cdim * Npoints, &points));
  for (PetscInt s = 0, offset = 0; s < num_quads; s++) {
    PetscInt         sNpoints;
    const PetscReal *sPoints;

    PetscCall(PetscQuadratureGetData(subquads[s], NULL, NULL, &sNpoints, &sPoints, NULL));
    PetscCall(PetscArraycpy(&points[offset], sPoints, cdim * sNpoints));
    offset += cdim * sNpoints;
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, fullquad));
  quad = *fullquad;
  PetscCall(PetscQuadratureSetData(quad, cdim, 0, Npoints, points, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumCreateMatrix(PetscInt Nc, PetscInt num_subspaces, PetscBool uniform, PetscBool uniform_points, PetscBool concatenate, PetscBool interleave_basis, PetscBool interleave_components, PetscQuadrature subquads[], Mat submats[], PetscQuadrature *fullquad, Mat *fullmat)
{
  Mat          mat;
  PetscInt    *i = NULL, *j = NULL;
  PetscScalar *v = NULL;
  PetscInt     nrows, ncols, nnz;

  PetscFunctionBegin;
  nrows = 0;
  ncols = 0;
  nnz   = 0;
  for (PetscInt s = 0, roffset = 0, coffset = 0; s < num_subspaces; s++) {
    // Get the COO data for each matrix, map the is and js, and append to growing COO data
    PetscInt               sNb, sNc, sNpoints, sNcols;
    Mat                    smat;
    const PetscInt        *si;
    const PetscInt        *sj;
    PetscScalar           *sv;
    PetscMemType           memtype;
    PetscInt               snz;
    PetscInt               snz_actual;
    PetscInt              *cooi;
    PetscInt              *cooj;
    PetscScalar           *coov;
    PetscScalar           *v_new;
    PetscInt              *i_new;
    PetscInt              *j_new;
    IS                     is_row, is_col;
    ISLocalToGlobalMapping isl2g_row, isl2g_col;

    if (!submats[s]) continue;
    PetscCall(MatGetSize(submats[s], &sNb, &sNcols));
    nrows += sNb;
    PetscCall(MatConvert(submats[s], MATSEQAIJ, MAT_INITIAL_MATRIX, &smat));
    PetscCall(MatBindToCPU(smat, PETSC_TRUE));
    PetscCall(MatSeqAIJGetCSRAndMemType(smat, &si, &sj, &sv, &memtype));
    PetscCheck(memtype == PETSC_MEMTYPE_HOST, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not convert matrix to host memory");
    snz = si[sNb];

    PetscCall(PetscMalloc1(nnz + snz, &v_new));
    PetscCall(PetscArraycpy(v_new, v, nnz));
    PetscCall(PetscFree(v));
    v = v_new;

    PetscCall(PetscMalloc1(nnz + snz, &i_new));
    PetscCall(PetscArraycpy(i_new, i, nnz));
    PetscCall(PetscFree(i));
    i = i_new;

    PetscCall(PetscMalloc1(nnz + snz, &j_new));
    PetscCall(PetscArraycpy(j_new, j, nnz));
    PetscCall(PetscFree(j));
    j = j_new;

    PetscCall(PetscMalloc2(snz, &cooi, snz, &cooj));

    coov = &v[nnz];

    snz_actual = 0;
    for (PetscInt row = 0; row < sNb; row++) {
      for (PetscInt k = si[row]; k < si[row + 1]; k++, snz_actual++) {
        cooi[snz_actual] = row;
        cooj[snz_actual] = sj[k];
        coov[snz_actual] = sv[k];
      }
    }
    PetscCall(MatDestroy(&smat));

    PetscCall(PetscQuadratureGetData(subquads[s], NULL, NULL, &sNpoints, NULL, NULL));
    sNc = sNcols / sNpoints;

    // There are only two possibilities for the rows: each subspaces get a contiguous block
    PetscCall(ISCreateStride(PETSC_COMM_SELF, sNb, roffset, (uniform && interleave_basis) ? num_subspaces : 1, &is_row));
    roffset += (uniform && interleave_basis) ? 1 : sNb;

    /* There are four possibilities for the cols:
       - (!concatenate) && uniform_points: each matrix gets a contiguous block the full width of the matrix
       - (!concatenate) && !uniform_points: each matrix gets an contiguous block for only its points
       - concatenate && uniform_points & interleave_components: each matrix gets a stride-num_subspaces block of all the points
       - concatenate && uniform_points & !interleave_components: each matrix gets an ISGeneral subset
       - concatenate && !uniform_points & interleave_components: Not considered (only interleave_basis of uniform and thus uniform_points)
       - concatenate && !uniform_points & !interleave_components: each matrix gets an ISGeneral subset
     */
    if (!concatenate) {
      PetscCall(ISCreateStride(PETSC_COMM_SELF, sNcols, coffset, 1, &is_col));
      coffset += uniform_points ? 0 : sNcols;
      ncols += ((!uniform_points) || s == 0) ? sNcols : 0;
    } else {
      if (uniform_points && interleave_components) {
        PetscCall(ISCreateStride(PETSC_COMM_SELF, sNcols, coffset, num_subspaces, &is_col));
        coffset += 1;
        ncols += sNcols;
      } else {
        PetscInt *cols;

        PetscCall(PetscMalloc1(sNcols, &cols));
        for (PetscInt p = 0, r = 0; p < sNpoints; p++) {
          for (PetscInt c = 0; c < sNc; c++, r++) { cols[r] = coffset + p * Nc + c; }
        }
        coffset += uniform_points ? sNc : Nc * sNpoints + sNc;
        ncols += uniform_points ? sNc * sNpoints : Nc * sNpoints;
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, sNcols, cols, PETSC_OWN_POINTER, &is_col));
      }
    }

    PetscCall(ISLocalToGlobalMappingCreateIS(is_row, &isl2g_row));
    PetscCall(ISLocalToGlobalMappingCreateIS(is_col, &isl2g_col));
    PetscCall(ISLocalToGlobalMappingApply(isl2g_row, snz_actual, cooi, &i[nnz]));
    PetscCall(ISLocalToGlobalMappingApply(isl2g_col, snz_actual, cooj, &j[nnz]));
    nnz += snz_actual;
    PetscCall(ISLocalToGlobalMappingDestroy(&isl2g_col));
    PetscCall(ISLocalToGlobalMappingDestroy(&isl2g_row));
    PetscCall(ISDestroy(&is_col));
    PetscCall(ISDestroy(&is_row));
    PetscCall(PetscFree2(cooi, cooj));
  }
  PetscCall(MatCreate(PETSC_COMM_SELF, fullmat));
  mat = *fullmat;
  PetscCall(MatSetSizes(mat, nrows, ncols, nrows, ncols));
  PetscCall(MatSetType(mat, MATSEQAIJ));
  PetscCall(MatSetPreallocationCOO(mat, nnz, i, j));
  PetscCall(MatSetValuesCOO(mat, v, INSERT_VALUES));
  PetscCall(PetscFree(i));
  PetscCall(PetscFree(j));
  PetscCall(PetscFree(v));
  if (fullquad) {
    if (uniform_points) {
      PetscCall(PetscObjectReference((PetscObject)subquads[0]));
      *fullquad = subquads[0];
    } else {
      PetscCall(PetscQuadratureConcatenate(num_subspaces, subquads, fullquad));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSetUp_Sum(PetscDualSpace sp)
{
  PetscDualSpace_Sum *sum         = (PetscDualSpace_Sum *)sp->data;
  PetscBool           concatenate = PETSC_TRUE;
  PetscBool           uniform;
  PetscInt            Ns, Nc, i, sum_Nc = 0;
  PetscInt            minNc, maxNc;
  DM                  K;
  PetscQuadrature    *all_quads = NULL;
  PetscQuadrature    *int_quads = NULL;
  Mat                *all_mats  = NULL;
  Mat                *int_mats  = NULL;

  PetscFunctionBegin;
  if (sum->setupCalled) PetscFunctionReturn(PETSC_SUCCESS);
  sum->setupCalled = PETSC_TRUE;

  PetscCall(PetscDualSpaceSumGetNumSubspaces(sp, &Ns));
  if (Ns == PETSC_DEFAULT) {
    Ns = 1;
    PetscCall(PetscDualSpaceSumSetNumSubspaces(sp, Ns));
  }
  PetscCheck(Ns >= 0, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have %" PetscInt_FMT " subspaces", Ns);

  // step 1: make sure they share a DM
  PetscCall(PetscDualSpaceGetDM(sp, &K));
  uniform = PETSC_TRUE;
  {
    PetscDualSpace first_subsp = NULL;

    for (PetscInt s = 0; s < Ns; s++) {
      PetscDualSpace subsp;
      DM             sub_K;

      PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
      PetscCall(PetscDualSpaceSetUp(subsp));
      PetscCall(PetscDualSpaceGetDM(subsp, &sub_K));
      if (s == 0 && K == NULL) {
        PetscCall(PetscDualSpaceSetDM(sp, sub_K));
        K = sub_K;
      }
      PetscCheck(sub_K == K, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Subspace %d does not have the same DM as the sum space", (int)s);
      if (!s) first_subsp = subsp;
      else if (subsp != first_subsp) uniform = PETSC_FALSE;
    }
  }
  sum->uniform = uniform;

  // step 2: count components
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceSumGetConcatenate(sp, &concatenate));
  minNc = Nc;
  maxNc = Nc;
  for (i = 0; i < Ns; ++i) {
    PetscInt       sNc;
    PetscDualSpace si;

    PetscCall(PetscDualSpaceSumGetSubspace(sp, i, &si));
    PetscCall(PetscDualSpaceSetUp(si));
    PetscCall(PetscDualSpaceGetNumComponents(si, &sNc));
    if (sNc != Nc) PetscCheck(concatenate, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Subspace as a different number of components but space does not concatenate components");
    minNc = PetscMin(minNc, sNc);
    maxNc = PetscMax(maxNc, sNc);
    sum_Nc += sNc;
  }

  if (concatenate) PetscCheck(sum_Nc == Nc, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Total number of subspace components (%" PetscInt_FMT ") does not match number of target space components (%" PetscInt_FMT ").", sum_Nc, Nc);
  else PetscCheck(minNc == Nc && maxNc == Nc, PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Subspaces must have same number of components as the target space.");

  sum->uniform = uniform;
  PetscCall(PetscMalloc4(Ns, &all_quads, Ns, &all_mats, Ns, &int_quads, Ns, &int_mats));
  {
    // test for uniform all points and uniform interior points
    PetscBool       uniform_all         = PETSC_TRUE;
    PetscBool       uniform_interior    = PETSC_TRUE;
    PetscQuadrature quad_all_first      = NULL;
    PetscQuadrature quad_interior_first = NULL;
    for (PetscInt s = 0; s < Ns; s++) {
      PetscDualSpace  subsp;
      PetscQuadrature subquad_all;
      PetscQuadrature subquad_interior;
      Mat             submat_all;
      Mat             submat_interior;

      PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
      PetscCall(PetscDualSpaceGetAllData(subsp, &subquad_all, &submat_all));
      PetscCall(PetscDualSpaceGetInteriorData(subsp, &subquad_interior, &submat_interior));
      if (!s) {
        quad_all_first      = subquad_all;
        quad_interior_first = subquad_interior;
      } else {
        if (subquad_all != quad_all_first) uniform_all = PETSC_FALSE;
        if (subquad_interior != quad_interior_first) uniform_interior = PETSC_FALSE;
      }
      PetscCall(PetscObjectReference((PetscObject)subquad_all));
      PetscCall(PetscObjectReference((PetscObject)subquad_interior));
      PetscCall(PetscObjectReference((PetscObject)submat_all));
      PetscCall(PetscObjectReference((PetscObject)submat_interior));
      all_quads[s] = subquad_all;
      int_quads[s] = subquad_interior;
      all_mats[s]  = submat_all;
      int_mats[s]  = submat_interior;
    }
    sum->uniform_all_points      = uniform_all;
    sum->uniform_interior_points = uniform_interior;
    PetscCall(PetscDualSpaceSumCreateMatrix(Nc, Ns, uniform, uniform_interior, concatenate, sum->interleave_basis, sum->interleave_components, int_quads, int_mats, &sp->intNodes, &sp->intMat));
    {
      PetscInt        pStart, pEnd;
      PetscInt        np;
      Mat            *sub_mats;
      Mat            *nest_mats;
      PetscSection    full_section;
      Mat             matnest;
      IS              perm_is;
      const PetscInt *perm = NULL;

      PetscCall(DMPlexGetChart(K, &pStart, &pEnd));
      np = pEnd - pStart;
      PetscCall(PetscMalloc1(Ns, &sub_mats));
      PetscCall(PetscCalloc1(np, &nest_mats));
      PetscCall(PetscDualSpaceSectionCreate_Internal(sp, &full_section));
      PetscCall(PetscSectionGetPermutation(full_section, &perm_is));
      if (perm_is) { PetscCall(ISGetIndices(perm_is, &perm)); }
      for (PetscInt p = pStart; p < pEnd; p++) {
        PetscInt point    = perm ? perm[p - pStart] + pStart : p;
        PetscInt full_dof = 0;
        for (PetscInt s = 0; s < Ns; s++) {
          PetscDualSpace subsp;
          PetscSection   subsection;
          PetscInt       off, dof;
          IS             isrow;

          PetscCall(PetscDualSpaceSumGetSubspace(sp, s, &subsp));
          PetscCall(PetscDualSpaceGetSection(subsp, &subsection));
          PetscCall(PetscSectionGetOffset(subsection, point, &off));
          PetscCall(PetscSectionGetDof(subsection, point, &dof));
          full_dof += dof;
          PetscCall(ISCreateStride(PETSC_COMM_SELF, dof, off, 1, &isrow));
          if (all_mats[s]) {
            PetscCall(MatCreateSubMatrix(all_mats[s], isrow, NULL, MAT_INITIAL_MATRIX, &sub_mats[s]));
          } else {
            sub_mats[s] = NULL;
          }
          PetscCall(ISDestroy(&isrow));
        }
        PetscCall(PetscSectionSetDof(full_section, point, full_dof));
        PetscCall(PetscDualSpaceSumCreateMatrix(Nc, Ns, uniform, uniform_all, concatenate, sum->interleave_basis, sum->interleave_components, all_quads, sub_mats, (p == pStart) ? &sp->allNodes : NULL, &nest_mats[p - pStart]));
        for (PetscInt s = 0; s < Ns; s++) { PetscCall(MatDestroy(&sub_mats[s])); }
      }
      PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, full_section));
      PetscCall(MatCreateNest(PETSC_COMM_SELF, np, NULL, 1, NULL, nest_mats, &matnest));
      PetscCall(MatConvert(matnest, MATSEQAIJ, MAT_INITIAL_MATRIX, &sp->allMat));
      PetscCall(MatDestroy(&matnest));
      for (PetscInt p = 0; p < pEnd - pStart; p++) { PetscCall(MatDestroy(&nest_mats[p - pStart])); }
      PetscCall(PetscFree(nest_mats));
      PetscCall(PetscFree(sub_mats));
      sp->pointSection = full_section;
    }
  }
  for (PetscInt s = 0; s < Ns; s++) {
    PetscCall(MatDestroy(&all_mats[s]));
    PetscCall(MatDestroy(&int_mats[s]));
    PetscCall(PetscQuadratureDestroy(&all_quads[s]));
    PetscCall(PetscQuadratureDestroy(&int_quads[s]));
  }
  PetscCall(PetscFree4(all_quads, all_mats, int_quads, int_mats));
  PetscCall(PetscDualSpaceComputeFunctionalsFromAllData(sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumView_Ascii(PetscDualSpace sp, PetscViewer v)
{
  PetscDualSpace_Sum *sum         = (PetscDualSpace_Sum *)sp->data;
  PetscBool           concatenate = sum->concatenate;
  PetscInt            i, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  if (concatenate) PetscCall(PetscViewerASCIIPrintf(v, "Sum space of %" PetscInt_FMT " concatenated subspaces%s\n", Ns, sum->uniform ? " (all identical)" : ""));
  else PetscCall(PetscViewerASCIIPrintf(v, "Sum space of %" PetscInt_FMT " subspaces%s\n", Ns, sum->uniform ? " (all identical)" : ""));
  for (i = 0; i < (sum->uniform ? (Ns > 0 ? 1 : 0) : Ns); ++i) {
    PetscCall(PetscViewerASCIIPushTab(v));
    PetscCall(PetscDualSpaceView(sum->sumspaces[i], v));
    PetscCall(PetscViewerASCIIPopTab(v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceView_Sum(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscDualSpaceSumView_Ascii(sp, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceDestroy_Sum(PetscDualSpace sp)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;
  PetscInt            i, Ns = sum->numSumSpaces;

  PetscFunctionBegin;
  for (i = 0; i < Ns; ++i) PetscCall(PetscDualSpaceDestroy(&sum->sumspaces[i]));
  PetscCall(PetscFree(sum->sumspaces));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetSubspace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetSubspace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetNumSubspaces_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetNumSubspaces_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetConcatenate_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetConcatenate_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetInterleave_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetInterleave_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetUseMoments_C", NULL));
  PetscCall(PetscFree(sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumSetInterleave - Set whether the basis functions and components of a uniform sum are interleaved

  Logically collective

  Input Parameters:
+ sp                    - a `PetscDualSpace` of type `PETSCDUALSPACESUM`
. interleave_basis      - if `PETSC_TRUE`, the basis vectors of the subspaces are interleaved
- interleave_components - if `PETSC_TRUE` and the space concatenates components (`PetscDualSpaceSumGetConcatenate()`),
                          interleave the concatenated components

  Level: developer

.seealso: `PetscDualSpace`, `PETSCDUALSPACESUM`, `PETSCFEVECTOR`, `PetscDualSpaceSumGetInterleave()`
@*/
PetscErrorCode PetscDualSpaceSumSetInterleave(PetscDualSpace sp, PetscBool interleave_basis, PetscBool interleave_components)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp, "PetscDualSpaceSumSetInterleave_C", (PetscDualSpace, PetscBool, PetscBool), (sp, interleave_basis, interleave_components));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumSetInterleave_Sum(PetscDualSpace sp, PetscBool interleave_basis, PetscBool interleave_components)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;
  PetscFunctionBegin;
  sum->interleave_basis      = interleave_basis;
  sum->interleave_components = interleave_components;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceSumGetInterleave - Get whether the basis functions and components of a uniform sum are interleaved

  Logically collective

  Input Parameter:
. sp - a `PetscDualSpace` of type `PETSCDUALSPACESUM`

  Output Parameters:
+ interleave_basis      - if `PETSC_TRUE`, the basis vectors of the subspaces are interleaved
- interleave_components - if `PETSC_TRUE` and the space concatenates components (`PetscDualSpaceSumGetConcatenate()`),
                          interleave the concatenated components

  Level: developer

.seealso: `PetscDualSpace`, `PETSCDUALSPACESUM`, `PETSCFEVECTOR`, `PetscDualSpaceSumSetInterleave()`
@*/
PetscErrorCode PetscDualSpaceSumGetInterleave(PetscDualSpace sp, PetscBool *interleave_basis, PetscBool *interleave_components)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (interleave_basis) PetscAssertPointer(interleave_basis, 2);
  if (interleave_components) PetscAssertPointer(interleave_components, 3);
  PetscTryMethod(sp, "PetscDualSpaceSumGetInterleave_C", (PetscDualSpace, PetscBool *, PetscBool *), (sp, interleave_basis, interleave_components));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceSumGetInterleave_Sum(PetscDualSpace sp, PetscBool *interleave_basis, PetscBool *interleave_components)
{
  PetscDualSpace_Sum *sum = (PetscDualSpace_Sum *)sp->data;
  PetscFunctionBegin;
  if (interleave_basis) *interleave_basis = sum->interleave_basis;
  if (interleave_components) *interleave_components = sum->interleave_components;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceLagrangeGetUseMoments_Sum(PetscDualSpace sp, PetscBool *use_moments)
{
  PetscFunctionBegin;
  *use_moments = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDualSpaceInitialize_Sum(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->destroy              = PetscDualSpaceDestroy_Sum;
  sp->ops->view                 = PetscDualSpaceView_Sum;
  sp->ops->setfromoptions       = NULL;
  sp->ops->duplicate            = PetscDualSpaceDuplicate_Sum;
  sp->ops->setup                = PetscDualSpaceSetUp_Sum;
  sp->ops->createheightsubspace = NULL;
  sp->ops->createpointsubspace  = NULL;
  sp->ops->getsymmetries        = NULL;
  sp->ops->apply                = PetscDualSpaceApplyDefault;
  sp->ops->applyall             = PetscDualSpaceApplyAllDefault;
  sp->ops->applyint             = PetscDualSpaceApplyInteriorDefault;
  sp->ops->createalldata        = PetscDualSpaceCreateAllDataDefault;
  sp->ops->createintdata        = PetscDualSpaceCreateInteriorDataDefault;

  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetNumSubspaces_C", PetscDualSpaceSumGetNumSubspaces_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetNumSubspaces_C", PetscDualSpaceSumSetNumSubspaces_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetSubspace_C", PetscDualSpaceSumGetSubspace_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetSubspace_C", PetscDualSpaceSumSetSubspace_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetConcatenate_C", PetscDualSpaceSumGetConcatenate_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetConcatenate_C", PetscDualSpaceSumSetConcatenate_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumGetInterleave_C", PetscDualSpaceSumGetInterleave_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceSumSetInterleave_C", PetscDualSpaceSumSetInterleave_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscDualSpaceLagrangeGetUseMoments_C", PetscDualSpaceLagrangeGetUseMoments_Sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCDUALSPACESUM = "sum" - A `PetscDualSpace` object that encapsulates a sum of subspaces.

  Level: intermediate

  Note:
  That sum can either be direct or a concatenation. For example if A and B are spaces each with 2 components,
  the direct sum of A and B will also have 2 components while the concatenated sum will have 4 components. In both cases A and B must be defined over the
  same reference element.

.seealso: `PetscDualSpace`, `PetscDualSpaceType`, `PetscDualSpaceCreate()`, `PetscDualSpaceSetType()`, `PetscDualSpaceSumGetNumSubspaces()`, `PetscDualSpaceSumSetNumSubspaces()`,
          `PetscDualSpaceSumGetConcatenate()`, `PetscDualSpaceSumSetConcatenate()`, `PetscDualSpaceSumSetInterleave()`, `PetscDualSpaceSumGetInterleave()`
M*/
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Sum(PetscDualSpace sp)
{
  PetscDualSpace_Sum *sum;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscCall(PetscNew(&sum));
  sum->numSumSpaces = PETSC_DEFAULT;
  sp->data          = sum;
  PetscCall(PetscDualSpaceInitialize_Sum(sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDualSpaceCreateSum - Create a finite element dual basis that is the sum of other dual bases

  Collective

  Input Parameters:
+ numSubspaces - the number of spaces that will be added together
. subspaces    - an array of length `numSubspaces` of spaces
- concatenate  - if `PETSC_FALSE`, the sum-space has the same components as the individual dual spaces (`PetscDualSpaceGetNumComponents()`); if `PETSC_TRUE`, the individual components are concatenated to create a dual space with more components

  Output Parameter:
. sumSpace - a `PetscDualSpace` of type `PETSCDUALSPACESUM`

  Level: advanced

.seealso: `PetscDualSpace`, `PETSCDUALSPACESUM`, `PETSCSPACESUM`
@*/
PetscErrorCode PetscDualSpaceCreateSum(PetscInt numSubspaces, const PetscDualSpace subspaces[], PetscBool concatenate, PetscDualSpace *sumSpace)
{
  PetscInt i, Nc = 0;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]), sumSpace));
  PetscCall(PetscDualSpaceSetType(*sumSpace, PETSCDUALSPACESUM));
  PetscCall(PetscDualSpaceSumSetNumSubspaces(*sumSpace, numSubspaces));
  PetscCall(PetscDualSpaceSumSetConcatenate(*sumSpace, concatenate));
  for (i = 0; i < numSubspaces; ++i) {
    PetscInt sNc;

    PetscCall(PetscDualSpaceSumSetSubspace(*sumSpace, i, subspaces[i]));
    PetscCall(PetscDualSpaceGetNumComponents(subspaces[i], &sNc));
    if (concatenate) Nc += sNc;
    else Nc = sNc;

    if (i == 0) {
      DM dm;

      PetscCall(PetscDualSpaceGetDM(subspaces[i], &dm));
      PetscCall(PetscDualSpaceSetDM(*sumSpace, dm));
    }
  }
  PetscCall(PetscDualSpaceSetNumComponents(*sumSpace, Nc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
