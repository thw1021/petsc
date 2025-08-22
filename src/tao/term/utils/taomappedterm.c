#include <petsc/private/taoimpl.h>

PETSC_INTERN PetscErrorCode TaoMappedTermSetData(TaoMappedTerm *mt, const char *prefix, PetscReal scale, TaoTerm term, Mat map)
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

PETSC_INTERN PetscErrorCode TaoMappedTermReset(TaoMappedTerm *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoMappedTermSetData(mt, NULL, 0.0, NULL, NULL));
  PetscCall(VecDestroy(&mt->_mapped_gradient));
  PetscCall(MatDestroy(&mt->_mapped_H));
  PetscCall(MatDestroy(&mt->_mapped_Hpre));
  mt->mask = TAOTERM_MASK_NONE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermGetData(TaoMappedTerm *mt, const char **prefix, PetscReal *scale, TaoTerm *term, Mat *map)
{
  PetscFunctionBegin;
  if (prefix) *prefix = mt->prefix;
  if (term) *term = mt->term;
  if (scale) *scale = mt->scale;
  if (map) *map = mt->map;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define TaoMapppedTermCheckInsertMode(mt, mode) \
  do { \
    PetscCheck((mode) == INSERT_VALUES || (mode) == ADD_VALUES, PetscObjectComm((PetscObject)(mt)->term), PETSC_ERR_ARG_OUTOFRANGE, "insert mode must be INSERT_VALUES or ADD_VALUES"); \
  } while (0)

static PetscErrorCode TaoMappedTermMap(TaoMappedTerm *mt, Vec x, Vec *Ax)
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

PETSC_INTERN PetscErrorCode TaoMappedTermComputeObjective(TaoMappedTerm *mt, Vec x, Vec params, InsertMode mode, PetscReal *value)
{
  Vec       Ax;
  PetscReal v;

  PetscFunctionBegin;
  TaoMapppedTermCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoMappedTermMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjective(mt->term, Ax, params, &v));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoMappedTermGetGradients(TaoMappedTerm *mt, InsertMode mode, Vec g, Vec *mapped_g, Vec *unmapped_g)
{
  PetscFunctionBegin;
  *mapped_g = g;
  if (mode == ADD_VALUES) {
    if (!mt->_mapped_gradient) PetscCall(VecDuplicate(g, &mt->_mapped_gradient));
    *mapped_g = mt->_mapped_gradient;
  }
  *unmapped_g = *mapped_g;
  if (mt->map) {
    if (!mt->_unmapped_gradient) PetscCall(TaoTermCreateVecs(mt->term, &mt->_unmapped_gradient, NULL));
    *unmapped_g = mt->_unmapped_gradient;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoMappedTermSetGradients(TaoMappedTerm *mt, InsertMode mode, Vec g, Vec mapped_g, Vec unmapped_g)
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

PETSC_INTERN PetscErrorCode TaoMappedTermComputeGradient(TaoMappedTerm *mt, Vec x, Vec params, InsertMode mode, Vec g)
{
  Vec Ax, mapped_g, unmapped_g = NULL;

  PetscFunctionBegin;
  TaoMapppedTermCheckInsertMode(mt, mode);
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoMappedTermGetGradients(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoMappedTermMap(mt, x, &Ax));
  PetscCall(TaoTermComputeGradient(mt->term, Ax, params, unmapped_g));
  PetscCall(TaoMappedTermSetGradients(mt, mode, g, mapped_g, unmapped_g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermComputeObjectiveAndGradient(TaoMappedTerm *mt, Vec x, Vec params, InsertMode mode, PetscReal *value, Vec g)
{
  Vec       Ax, mapped_g, unmapped_g = NULL;
  PetscReal v;

  PetscFunctionBegin;
  TaoMapppedTermCheckInsertMode(mt, mode);
  if (TaoTermObjectiveMasked(mt->mask) && TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      *value = 0;
      PetscCall(VecZeroEntries(g));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0;
    PetscCall(TaoMappedTermComputeGradient(mt, x, params, mode, g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscCall(TaoMappedTermComputeObjective(mt, x, params, mode, value));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoMappedTermGetGradients(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoMappedTermMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjectiveAndGradient(mt->term, Ax, params, &v, unmapped_g));
  PetscCall(TaoMappedTermSetGradients(mt, mode, g, mapped_g, unmapped_g));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoMappedTermMatPtAP(Mat unmapped_H, Mat map, Mat mapped_H)
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

static PetscErrorCode TaoMappedTermGetHessians(TaoMappedTerm *mt, InsertMode mode, Mat H, Mat Hpre, Mat *mapped_H, Mat *mapped_Hpre, Mat *unmapped_H, Mat *unmapped_Hpre)
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

static PetscErrorCode TaoMappedTermSetHessians(TaoMappedTerm *mt, InsertMode mode, Mat H, Mat Hpre, Mat mapped_H, Mat mapped_Hpre, Mat unmapped_H, Mat unmapped_Hpre)
{
  PetscFunctionBegin;
  if (mt->map) {
    // currently only implements Gauss-Newton Hessian approximation
    if (mapped_H) PetscCall(TaoMappedTermMatPtAP(unmapped_H, mt->map, mapped_H));
    if (mapped_Hpre) PetscCall(TaoMappedTermMatPtAP(unmapped_Hpre, mt->map, mapped_Hpre));
  }
  if (mode == ADD_VALUES) {
    if (H) PetscCall(MatAXPY(H, mt->scale, mapped_H, UNKNOWN_NONZERO_PATTERN));
    if (Hpre) PetscCall(MatAXPY(Hpre, mt->scale, mapped_Hpre, UNKNOWN_NONZERO_PATTERN));
  } else {
    if (mt->scale != 1.0) {
      if (H) PetscCall(MatScale(H, mt->scale));
      if (Hpre && Hpre != H) PetscCall(MatScale(Hpre, mt->scale));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermComputeHessian(TaoMappedTerm *mt, Vec x, Vec params, InsertMode mode, Mat H, Mat Hpre)
{
  Vec Ax;
  Mat mapped_H, mapped_Hpre, unmapped_H = NULL, unmapped_Hpre = NULL;

  PetscFunctionBegin;
  TaoMapppedTermCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      if (H) PetscCall(MatZeroEntries(H));
      if (Hpre && Hpre != H) PetscCall(MatZeroEntries(H));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(Hpre == NULL || Hpre != H, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_PLIB, "Hessian inputs not sanitized");
  PetscCall(TaoMappedTermMap(mt, x, &Ax));
  PetscCall(TaoMappedTermGetHessians(mt, mode, H, Hpre, &mapped_H, &mapped_Hpre, &unmapped_H, &unmapped_Hpre));
  PetscCall(TaoTermComputeHessian(mt->term, Ax, params, unmapped_H, unmapped_Hpre));
  PetscCall(TaoMappedTermSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermComputeHessianMult(TaoMappedTerm *mt, Vec Ax, Vec params, Mat unmapped_H, Vec v, InsertMode mode, Vec Hv)
{
  Vec Av;
  Vec mapped_Hv, unmapped_Hv = NULL;

  PetscFunctionBegin;
  TaoMapppedTermCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(Hv));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoMappedTermMap(mt, v, &Av));
  PetscCall(TaoMappedTermGetGradients(mt, mode, Hv, &mapped_Hv, &unmapped_Hv));
  if (unmapped_H) PetscCall(MatMult(unmapped_H, Av, unmapped_Hv));
  else PetscCall(TaoTermComputeHessianMult(mt->term, Ax, params, Av, unmapped_Hv));
  PetscCall(TaoMappedTermSetGradients(mt, mode, Hv, mapped_Hv, unmapped_Hv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermSetUp(TaoMappedTerm *mt)
{
  PetscFunctionBegin;
  PetscCall(TaoTermSetUp(mt->term));
  if (mt->map) PetscCall(MatSetUp(mt->map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermCreateVecs(TaoMappedTerm *mt, Vec *solution, Vec *params)
{
  PetscFunctionBegin;
  if (mt->map) {
    PetscCall(MatCreateVecs(mt->map, solution, NULL));
    PetscCall(TaoTermCreateVecs(mt->term, NULL, params));
  } else PetscCall(TaoTermCreateVecs(mt->term, solution, params));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoMappedTermCreatePtAP(Mat unmapped_H, Mat map, Mat *H)
{
  PetscFunctionBegin;
  PetscCall(MatProductCreate(unmapped_H, map, NULL, H));
  PetscCall(MatProductSetType(*H, MATPRODUCT_PtAP));
  PetscCall(MatProductSetFromOptions(*H));
  PetscCall(MatProductSymbolic(*H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoMappedTermCreateHessianMatrices(TaoMappedTerm *mt, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  if (!mt->map) PetscCall(TaoTermCreateHessianMatrices(mt->term, H, Hpre));
  else {
    PetscCall(TaoTermCreateHessianMatrices(mt->term, H ? &mt->_unmapped_H : NULL, Hpre ? &mt->_unmapped_Hpre : NULL));
    if (mt->_unmapped_H) PetscCall(TaoMappedTermCreatePtAP(mt->_unmapped_H, mt->map, H));
    if (mt->_unmapped_Hpre) {
      if (mt->_unmapped_Hpre == mt->_unmapped_H) {
        PetscCall(PetscObjectReference((PetscObject)*H));
        *Hpre = *H;
      } else PetscCall(TaoMappedTermCreatePtAP(mt->_unmapped_Hpre, mt->map, Hpre));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
