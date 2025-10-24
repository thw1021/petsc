#include <petsc/private/taoimpl.h>

PETSC_INTERN PetscErrorCode TaoTermMappingSetData(TaoTermMapping *mt, const char *prefix, PetscReal scale, TaoTerm term, Mat map)
{
  PetscBool same_name;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(prefix, mt->prefix, &same_name));
  if (!same_name) {
    PetscCall(PetscFree(mt->prefix));
    PetscCall(PetscStrallocpy(prefix, &mt->prefix));
  }
  if (term != mt->term) {
    PetscCall(VecDestroy(&mt->_unmapped_gradient));
    PetscCall(MatDestroy(&mt->_unmapped_H));
    PetscCall(MatDestroy(&mt->_unmapped_Hpre));
    PetscCall(MatDestroy(&mt->_mapped_H));
    PetscCall(MatDestroy(&mt->_mapped_Hpre));
    PetscCall(MatDestroy(&mt->_mapped_H_work));
    PetscCall(MatDestroy(&mt->_mapped_Hpre_work));
  }
  PetscCall(PetscObjectReference((PetscObject)term));
  PetscCall(TaoTermDestroy(&mt->term));
  mt->term  = term;
  mt->scale = scale;
  if (map != mt->map) PetscCall(VecDestroy(&mt->_map_output));
  PetscCall(PetscObjectReference((PetscObject)map));
  PetscCall(MatDestroy(&mt->map));
  mt->map = map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingReset(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermMappingSetData(mt, NULL, 0.0, NULL, NULL));
  PetscCall(VecDestroy(&mt->_mapped_gradient));
  PetscCall(MatDestroy(&mt->_unmapped_H));
  PetscCall(MatDestroy(&mt->_unmapped_Hpre));
  PetscCall(MatDestroy(&mt->_mapped_H));
  PetscCall(MatDestroy(&mt->_mapped_Hpre));
  PetscCall(MatDestroy(&mt->_mapped_H_work));
  PetscCall(MatDestroy(&mt->_mapped_Hpre_work));
  mt->mask = TAOTERM_MASK_NONE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingGetData(TaoTermMapping *mt, const char **prefix, PetscReal *scale, TaoTerm *term, Mat *map)
{
  PetscFunctionBegin;
  if (prefix) *prefix = mt->prefix;
  if (term) *term = mt->term;
  if (scale) *scale = mt->scale;
  if (map) *map = mt->map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TaoTermMappingCheckInsertMode(mt, mode) \
  do { \
    PetscCheck((mode) == INSERT_VALUES || (mode) == ADD_VALUES, PetscObjectComm((PetscObject)(mt)->term), PETSC_ERR_ARG_OUTOFRANGE, "insert mode must be INSERT_VALUES or ADD_VALUES"); \
  } while (0)

static PetscErrorCode TaoTermMappingMap(TaoTermMapping *mt, Vec x, Vec *Ax)
{
  PetscFunctionBegin;
  *Ax = x;
  if (mt->map) {
    if (!mt->_map_output) PetscCall(MatCreateVecs(mt->map, NULL, &mt->_map_output));
    PetscCall(MatMult(mt->map, x, mt->_map_output));
    *Ax = mt->_map_output;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjective(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, PetscReal *value)
{
  Vec       Ax;
  PetscReal v;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjective(mt->term, Ax, params, &v));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingGetGradients(TaoTermMapping *mt, InsertMode mode, Vec g, Vec *mapped_g, Vec *unmapped_g)
{
  PetscFunctionBegin;
  *mapped_g = g;
  if (mode == ADD_VALUES) {
    if (!mt->_mapped_gradient) PetscCall(VecDuplicate(g, &mt->_mapped_gradient));
    *mapped_g = mt->_mapped_gradient;
  }
  *unmapped_g = *mapped_g;
  if (mt->map) {
    if (!mt->_unmapped_gradient) PetscCall(TaoTermCreateSolutionVec(mt->term, &mt->_unmapped_gradient));
    *unmapped_g = mt->_unmapped_gradient;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingSetGradients(TaoTermMapping *mt, InsertMode mode, Vec g, Vec mapped_g, Vec unmapped_g)
{
  PetscFunctionBegin;
  if (mt->map) PetscCall(MatMultHermitianTranspose(mt->map, unmapped_g, mapped_g));
  else PetscAssert(mapped_g == unmapped_g, PETSC_COMM_SELF, PETSC_ERR_PLIB, "gradient not written to the right place");
  if (mode == ADD_VALUES) PetscCall(VecAXPY(g, mt->scale, mapped_g));
  else {
    PetscAssert(mapped_g == g, PETSC_COMM_SELF, PETSC_ERR_PLIB, "graident not written to the right place");
    if (mt->scale != 1.0) PetscCall(VecScale(g, mt->scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeGradient(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Vec g)
{
  Vec Ax, mapped_g, unmapped_g = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetGradients(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeGradient(mt->term, Ax, params, unmapped_g));
  PetscCall(TaoTermMappingSetGradients(mt, mode, g, mapped_g, unmapped_g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingComputeObjectiveAndGradient(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, PetscReal *value, Vec g)
{
  Vec       Ax, mapped_g, unmapped_g = NULL;
  PetscReal v;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask) && TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      *value = 0;
      PetscCall(VecZeroEntries(g));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0;
    PetscCall(TaoTermMappingComputeGradient(mt, x, params, mode, g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscCall(TaoTermMappingComputeObjective(mt, x, params, mode, value));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetGradients(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjectiveAndGradient(mt->term, Ax, params, &v, unmapped_g));
  PetscCall(TaoTermMappingSetGradients(mt, mode, g, mapped_g, unmapped_g));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingMatPtAP_Internal(Mat unmapped_H, Mat map, Mat mapped_H)
{
  Mat            A, P;
  MatProductType prod_type;

  PetscFunctionBegin;
  PetscCall(MatProductGetType(mapped_H, &prod_type));
  if (prod_type != MATPRODUCT_PtAP && prod_type != MATPRODUCT_UNSPECIFIED) PetscCall(MatProductClear(mapped_H));
  PetscCall(MatProductGetMats(mapped_H, &A, &P, NULL));
  if (A != unmapped_H || P != map) {
    PetscBool is_assembled;

    PetscCall(MatAssembled(unmapped_H, &is_assembled));
    if (!is_assembled) {
      PetscCall(MatAssemblyBegin(unmapped_H, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(unmapped_H, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatAssembled(map, &is_assembled));
    if (!is_assembled) {
      PetscCall(MatAssemblyBegin(map, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(map, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatAssembled(mapped_H, &is_assembled));
    if (!is_assembled) {
      PetscCall(MatAssemblyBegin(mapped_H, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(mapped_H, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatProductCreateWithMat(unmapped_H, map, NULL, mapped_H));
    PetscCall(MatProductSetType(mapped_H, MATPRODUCT_PtAP));
    PetscCall(MatProductSetFromOptions(mapped_H));
    PetscCall(MatProductSymbolic(mapped_H));
  }
  PetscCall(MatProductNumeric(mapped_H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingMatPtAP(Mat unmapped_H, Mat map, Mat mapped_H, Mat work)
{
  PetscBool      is_uH_diag, is_map_diag, is_uH_cdiag, is_map_cdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATDIAGONAL, &is_uH_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATCONSTANTDIAGONAL, &is_uH_cdiag));
  PetscCall(PetscObjectTypeCompare((PetscObject)map, MATDIAGONAL, &is_map_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)map, MATCONSTANTDIAGONAL, &is_map_cdiag));

  if (is_map_diag) {
    Vec m_diag;

    PetscCall(MatDiagonalGetDiagonal(map, &m_diag));
    if (is_uH_cdiag) {
      Vec         mapped_diag;
      PetscScalar cc;

      // mapped_H \gets cc map * map
      PetscCall(MatConstantDiagonalGetConstant(unmapped_H, &cc));
      PetscCall(MatDiagonalGetDiagonal(mapped_H, &mapped_diag));
      PetscCall(VecPointwiseMult(mapped_diag, m_diag, m_diag));
      PetscCall(VecScale(mapped_diag, cc));
      PetscCall(MatDiagonalRestoreDiagonal(mapped_H, &mapped_diag));
    } else if (is_uH_diag) {
      Vec mapped_diag, unmapped_diag;

      PetscCall(MatDiagonalGetDiagonal(mapped_H, &mapped_diag));
      PetscCall(MatDiagonalGetDiagonal(unmapped_H, &unmapped_diag));
      PetscCall(VecPointwiseMult(mapped_diag, m_diag, m_diag));
      PetscCall(VecPointwiseMult(mapped_diag, unmapped_diag, mapped_diag));
      PetscCall(MatDiagonalRestoreDiagonal(mapped_H, &mapped_diag));
      PetscCall(MatDiagonalRestoreDiagonal(unmapped_H, &unmapped_diag));
    } else {
      PetscCall(MatCopy(unmapped_H, mapped_H, SAME_NONZERO_PATTERN));
      PetscCall(MatDiagonalScale(mapped_H, m_diag, m_diag));
    }
    PetscCall(MatDiagonalRestoreDiagonal(map, &m_diag));
  } else if (is_map_cdiag) {
    PetscScalar cc;

    PetscCall(MatConstantDiagonalGetConstant(map, &cc));
    PetscCall(MatCopy(unmapped_H, mapped_H, SAME_NONZERO_PATTERN));
    PetscCall(MatScale(mapped_H, cc*cc));
  } else if (is_uH_diag) {
    Vec unmapped_diag;

    //TODO inefficient. Remove when diag PtAP gets implemented
    PetscCall(MatDiagonalGetDiagonal(unmapped_H, &unmapped_diag));
    PetscCall(MatCopy(map, work, SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalScale(work, unmapped_diag, NULL));
    PetscCall(MatTransposeMatMult(map, work, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped_H));
    PetscCall(MatDiagonalRestoreDiagonal(unmapped_H, &unmapped_diag));
  } else if (is_uH_cdiag) {
    // cc * A^T A
    PetscScalar cc;

    PetscCall(MatConstantDiagonalGetConstant(unmapped_H, &cc));
    PetscCall(MatTransposeMatMult(map, map, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped_H));
    PetscCall(MatScale(mapped_H, cc));
  } else {
    PetscCall(TaoTermMappingMatPtAP_Internal(unmapped_H, map, mapped_H));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingGetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat *mapped_H, Mat *mapped_Hpre, Mat *unmapped_H, Mat *unmapped_Hpre)
{
  PetscFunctionBegin;
  *mapped_H    = H;
  *mapped_Hpre = Hpre;
  if (mode == ADD_VALUES || mt->map) {
    // we will need _unmapped_H / _unmapped_Hpre
    if (!mt->_unmapped_H) {
      PetscBool is_defined = PETSC_FALSE;

      PetscCall(TaoTermIsCreateHessianMatricesDefined(mt->term, &is_defined));
      if (is_defined) {
        PetscCall(MatDestroy(&mt->_unmapped_Hpre));
        PetscCall(TaoTermCreateHessianMatrices(mt->term, &mt->_unmapped_H, &mt->_unmapped_Hpre));
      }
      if (!mt->map) {
        PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
        PetscCall(MatDestroy(&mt->_mapped_H));
        mt->_mapped_H = mt->_unmapped_H;

        PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
        PetscCall(MatDestroy(&mt->_mapped_Hpre));
        mt->_mapped_Hpre = mt->_unmapped_Hpre;
      }
    }
  }
  if (mode == ADD_VALUES) {
    if (H) {
      if (!mt->_mapped_H) PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &mt->_mapped_H));
      *mapped_H = mt->_mapped_H;
    }
    if (Hpre) {
      if (!mt->_mapped_Hpre) PetscCall(MatDuplicate(Hpre, MAT_DO_NOT_COPY_VALUES, &mt->_mapped_Hpre));
      *mapped_Hpre = mt->_mapped_Hpre;
    }
  }
  *unmapped_H    = *mapped_H;
  *unmapped_Hpre = *mapped_Hpre;
  if (mt->map) {
    if (H) *unmapped_H = mt->_unmapped_H;
    if (Hpre) *unmapped_Hpre = mt->_unmapped_Hpre;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// if (map) mapped_H \gets map^T @ unmapped_H @ map
// else (assumes that unmapped == mapped.
//
// if INSERT
//   H \gets mapped_H
// else if ADD
//   H \gets H + scale * mapped_H
static PetscErrorCode TaoTermMappingSetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat mapped_H, Mat mapped_Hpre, Mat unmapped_H, Mat unmapped_Hpre)
{
  PetscFunctionBegin;
  if (mt->map) {
    // currently only implements Gauss-Newton Hessian approximation
    if (mapped_H) PetscCall(TaoTermMappingMatPtAP(unmapped_H, mt->map, mapped_H, mt->_mapped_H_work));
    if (mapped_Hpre) PetscCall(TaoTermMappingMatPtAP(unmapped_Hpre, mt->map, mapped_Hpre, mt->_mapped_Hpre_work));
  }
  if (mode == ADD_VALUES) {
    if (H) PetscCall(MatAXPY(H, mt->scale, mapped_H, UNKNOWN_NONZERO_PATTERN));
    if (Hpre) PetscCall(MatAXPY(Hpre, mt->scale, mapped_Hpre, UNKNOWN_NONZERO_PATTERN));
  } else {
    if (H) PetscCall(MatCopy(H, mapped_H, DIFFERENT_NONZERO_PATTERN));
    if (Hpre && (H != Hpre)) PetscCall(MatCopy(Hpre, mapped_Hpre, DIFFERENT_NONZERO_PATTERN));
    if (mt->scale != 1.0) {
      if (H) PetscCall(MatScale(H, mt->scale));
      if (Hpre && Hpre != H) PetscCall(MatScale(Hpre, mt->scale));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//Either called by TaoComputeHessian (one term in Tao), or by TAOTERMSUM
//
// First case: (one term in Tao)
// TaoComputeHessian
//   -> TaoTermMappingComputeHessian
//      (unmapped_H == mapped_H)
//
// Second case: TAOTERMSUM, (more than one term in Tao)
// TaoComputeHessian
//   -> TaoTermMappingComputeHessian
//     -> (mt->_unmapped_H == mt->_mapped_H == tao->hessian) (SUM does not take mapping)
//     -> TaoTermComputeHessian
//       -> TaoTermComputeHessian_Sum
//         -> for(i:n_terms)
//         -> TaoTermMappingComputeHessian
//           -> (unmapped_H may not == mapped_H)
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessian(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Mat H, Mat Hpre)
{
  Vec Ax;
  Mat mapped_H, mapped_Hpre, unmapped_H = NULL, unmapped_Hpre = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      if (H) PetscCall(MatZeroEntries(H));
      if (Hpre && Hpre != H) PetscCall(MatZeroEntries(Hpre));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
#if 1
  PetscCall(TaoTermMappingGetHessians(mt, mode, H, Hpre, &mapped_H, &mapped_Hpre, &unmapped_H, &unmapped_Hpre));
  PetscCall(TaoTermComputeHessian(mt->term, Ax, params, unmapped_H, unmapped_Hpre));
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
#else
  //TODO if TAOTERMSUM, assuming that tao->hessian == H == mt->_unmapped_H == mt->_mapped_H. Is this true?
  //What about n_terms == 1? TODO
  PetscCall(TaoTermComputeHessian(mt->term, Ax, params, mt->_unmapped_H, mt->_unmapped_Hpre));
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mt->_mapped_H, mt->_mapped_Hpre, mt->_unmapped_H, mt->_unmapped_Hpre));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingSetUp(TaoTermMapping *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermSetUp(mt->term));
  if (mt->map) PetscCall(MatSetUp(mt->map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingCreateSolutionVec(TaoTermMapping *mt, Vec *solution)
{
  PetscFunctionBegin;
  if (mt->map) PetscCall(MatCreateVecs(mt->map, solution, NULL));
  else PetscCall(TaoTermCreateSolutionVec(mt->term, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermMappingCreateParametersVec(TaoTermMapping *mt, Vec *params)
{
  PetscFunctionBegin;
  PetscCall(TaoTermCreateParametersVec(mt->term, params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingCreateAPWorkMatrix(Mat map, Mat unmapped, Mat *mapped_work)
{
  PetscBool is_uH_diag, is_map_diag;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)map, &is_map_diag,  MATDIAGONAL, MATCONSTANTDIAGONAL, ""));
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped, MATDIAGONAL, &is_uH_diag));
  if (is_uH_diag && !is_map_diag) PetscCall(MatDuplicate(map, MAT_DO_NOT_COPY_VALUES, mapped_work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingCreatePtAP(Mat unmapped_H, Mat map, Mat *H)
{
  PetscBool is_uH_diag, is_map_diag, is_uH_cdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATDIAGONAL, &is_uH_diag));
  PetscCall(PetscObjectTypeCompare((PetscObject)unmapped_H, MATCONSTANTDIAGONAL, &is_uH_cdiag));
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)map, &is_map_diag,  MATDIAGONAL, MATCONSTANTDIAGONAL, ""));

  // TODO support for PtAP with diagonal would be ideal
  if (is_map_diag) {
    // map is diag, which means mapped's same size as unmapped
    // Technically, if A is dense, PtAP is available, but ignoring for now
    PetscCall(MatDuplicate(unmapped_H, MAT_DO_NOT_COPY_VALUES, H));
  } else if (is_uH_cdiag) {
    // mapped \gets \alpha P^T P
    PetscCall(MatTransposeMatMult(map, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, H));
  } else if (is_uH_diag) {
    PetscCall(MatTransposeMatMult(map, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, H));
  } else {
    //TODO MatProductSetFromOptions_Private has composed function querying to see if PtAP is availble
    //That should be turned into intern function to be used here. If unavailable, densify matrices
    PetscCall(MatProductCreate(unmapped_H, map, NULL, H));
    PetscCall(MatProductSetType(*H, MATPRODUCT_PtAP));
    PetscCall(MatProductSetFromOptions(*H));
    PetscCall(MatProductSymbolic(*H));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//TODO what does this function should really do??
// only if unmapped, it created _unmapped_H. shouldn't it do the same for unmapped case?
// the ternary is wrong? i dont see a case where H, which is (Mat *) be null?
//
// thsi functino should
// 1. If (map)
//      H \gets mt->_mapped_H
// 2. else
//     H \gets mt->_unmapped_H = mt->_mapped_H (should this just be null?)
//
//     BUT TaoTermCreateHessianMatrices does not change internal state. obviously, its calling on TaoTerm which does not store Hessians
//
// This is eiher called, if n_terms == 1, and not callback, or if SUM.
//
// first case: TODO can't set PtAP mat manually, as there is no TaoTermSumSetSubtermHessianMatrices...?
//    if (!map)
//        if (!mt->_unmapped_H)
//            create
//        else no-op
//        H \gets mt->_unmapped
//        TODO should mt->_mapped_H be mt->_unmappe_H? let's say no for now
//
//
//What are possible inputs? H and Hpre:
//
//H&&Hpre: okay
//!H && Hpre: ??
//  -> !map
//    -> just create Hpre?
//H && !Hpre: ??
//!H && !Hpre: ?? -> err
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianMatrices(TaoTermMapping *mt, Mat *H, Mat *Hpre)
{
  Mat uH, uHpre, mH, mHpre;

  PetscFunctionBegin;
  uH    = mt->_unmapped_H;
  uHpre = mt->_unmapped_Hpre;
  mH    = mt->_mapped_H;
  mHpre = mt->_mapped_Hpre;
  //TODO if sum, there should not be a map.
  PetscCheck(H, PetscObjectComm((PetscObject)mt), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input for H");
  PetscCheck(Hpre, PetscObjectComm((PetscObject)mt), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input Hpre");
  if (!mt->map) {
    // mt->_unmapped_{H,Hpre} == mt->_unmapped_{H,Hpre}
    if (uH && mH) PetscCheck(uH == mH, PetscObjectComm((PetscObject)mt), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian and unmapped Hessian needs to be same");
    if (uHpre && mHpre) PetscCheck(uHpre == mHpre, PetscObjectComm((PetscObject)mt), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian preconditioner and unmapped Hessian preconditioner needs to be same");

    //If mapped matrices are present, it should be set to unmapped matrices
    if (mt->_mapped_H && !mt->_unmapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
      mt->_unmapped_H = mt->_mapped_H;
    }
    if (mt->_mapped_Hpre && !mt->_unmapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      mt->_unmapped_Hpre = mt->_mapped_Hpre;
    }
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    //If mapped matrices are NULL, it should be set to mapped matrices
    if (mt->_unmapped_H && !mt->_mapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
      mt->_mapped_H = mt->_unmapped_H;
    }
    if (mt->_unmapped_Hpre && !mt->_mapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
      mt->_mapped_Hpre = mt->_unmapped_Hpre;
    }

    //always returns Hpre, even if same as H
    if (*H != mt->_unmapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
    if (*Hpre != mt->_unmapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
    *H    = mt->_unmapped_H;
    *Hpre = mt->_unmapped_Hpre;
  }
  else {
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    //Create PtAP only if mt->_mapped_H is empty
    //Warning: not all matrices combinations have PtAP available. If not, have to manually set it.
    //TODO just because mapped_H has been set ...doesnt work... Need to set manual routines... how?
    //Special case: if L2, the A=I, so A^T A
    if (mt->_unmapped_H && !mt->_mapped_H) PetscCall(TaoTermMappingCreatePtAP(mt->_unmapped_H, mt->map, &mt->_mapped_H));
    // Creating expensive work matrix to store AP
    // TODO if diag, then you need work matrix same size as map...
    if (!mt->_mapped_H_work) PetscCall(TaoTermMappingCreateAPWorkMatrix(mt->map, mt->_unmapped_H, &mt->_mapped_H_work));
    if (*H != mt->_mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
    *H = mt->_mapped_H;
    if (mt->_unmapped_Hpre) {
      // Hpre_is_H true
      if (mt->_unmapped_Hpre == mt->_unmapped_H) {
        PetscCall(PetscObjectReference((PetscObject)*H));
        *Hpre = *H;
      } else {
        if (!mt->_mapped_Hpre) PetscCall(TaoTermMappingCreatePtAP(mt->_unmapped_Hpre, mt->map, &mt->_mapped_Hpre));
        if (!mt->_mapped_Hpre_work) PetscCall(TaoTermMappingCreateAPWorkMatrix(mt->map, mt->_unmapped_H, &mt->_mapped_Hpre_work));
        if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
        *H = mt->_mapped_Hpre;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
