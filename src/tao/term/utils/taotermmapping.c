#include <petsc/private/taoimpl.h>
#include <petsc/private/matimpl.h>

/*
  TaoTermMatSnapshotGet - Record the ID , state, and nonzero-pattern state of a matrix

  Not Collective

  Input Parameter:
. mat - the matrix, or `NULL`

  Output Parameter:
. snapshot - the matrix snapshot

  Level: developer

  Note:
  A `NULL` matrix produces the snapshot `(0, 0, 0)`, which cannot match a live `PetscObject`.

.seealso: `TaoTermMatSnapshot`, `TaoTermMatSnapshotMatches()`
*/
PETSC_INTERN PetscErrorCode TaoTermMatSnapshotGet(Mat mat, TaoTermMatSnapshot *snapshot)
{
  PetscFunctionBegin;
  if (mat) {
    PetscCall(PetscObjectGetId((PetscObject)mat, &snapshot->id));
    PetscCall(MatGetState(mat, &snapshot->state));
    PetscCall(MatGetNonzeroState(mat, &snapshot->nonzero_state));
  } else {
    snapshot->id            = 0;
    snapshot->state         = 0;
    snapshot->nonzero_state = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMatSnapshotMatches - Determine whether a matrix matches a recorded snapshot

  Not Collective

  Input Parameters:
+ mat      - the matrix, or `NULL`
- snapshot - a snapshot produced by `TaoTermMatSnapshotGet()`

  Output Parameter:
. matches - `PETSC_TRUE` if the matrix ID, state, and nonzero-pattern state all match

  Level: developer

.seealso: `TaoTermMatSnapshot`, `TaoTermMatSnapshotGet()`
*/
PETSC_INTERN PetscErrorCode TaoTermMatSnapshotMatches(Mat mat, const TaoTermMatSnapshot *snapshot, PetscBool *matches)
{
  TaoTermMatSnapshot current;

  PetscFunctionBegin;
  PetscCall(TaoTermMatSnapshotGet(mat, &current));
  *matches = (PetscBool)(current.id == snapshot->id && current.state == snapshot->state && current.nonzero_state == snapshot->nonzero_state);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappedHessianStateGet(Mat raw, Mat map, Mat mapped, TaoTermMappedHessianState *state)
{
  PetscFunctionBegin;
  PetscCall(TaoTermMatSnapshotGet(raw, &state->raw));
  PetscCall(TaoTermMatSnapshotGet(map, &state->map));
  PetscCall(TaoTermMatSnapshotGet(mapped, &state->mapped));
  state->valid = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingCreateMappedHessianPlaceholder(TaoTermMapping *mt, Mat *mapped)
{
  PetscLayout layout;
  VecType     vec_type;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)mt->map), mapped));
  PetscCall(MatGetLayouts(mt->map, NULL, &layout));
  PetscCall(MatSetLayouts(*mapped, layout, layout));
  PetscCall(MatGetVecType(mt->map, &vec_type));
  PetscCall(MatSetType(*mapped, MATAIJ));
  PetscCall(MatSetVecType(*mapped, vec_type));
  PetscCall(MatSetUp(*mapped));
  PetscCall(MatAssemblyBegin(*mapped, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mapped, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermMappingUpdateMappedHessian(Mat raw, Mat map, Mat mapped, TaoTermMappedHessianState *cached, PetscBool *refreshed)
{
  TaoTermMappedHessianState current;
  Mat                       fresh             = NULL;
  PetscBool                 structure_changed = PETSC_FALSE, values_changed = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(TaoTermMappedHessianStateGet(raw, map, mapped, &current));
  if (!cached->valid) structure_changed = PETSC_TRUE;
  else {
    if (cached->raw.id != current.raw.id || cached->raw.nonzero_state != current.raw.nonzero_state) structure_changed = PETSC_TRUE;
    if (cached->map.id != current.map.id || cached->map.nonzero_state != current.map.nonzero_state) structure_changed = PETSC_TRUE;
    if (cached->mapped.id != current.mapped.id || cached->mapped.nonzero_state != current.mapped.nonzero_state) structure_changed = PETSC_TRUE;
    if (cached->raw.state != current.raw.state || cached->map.state != current.map.state || cached->mapped.state != current.mapped.state) values_changed = PETSC_TRUE;
  }
  *refreshed = PETSC_FALSE;
  if (structure_changed) {
    PetscCall(MatPtAP(raw, map, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &fresh));
    /* `mapped` may already be referenced by Tao or KSP. Preserve the outer object they hold while
       replacing its obsolete product implementation and symbolic data with those from `fresh`. */
    PetscCall(MatHeaderReplace(mapped, &fresh));
    *refreshed = PETSC_TRUE;
  } else if (values_changed) {
    PetscCall(MatPtAP(raw, map, MAT_REUSE_MATRIX, PETSC_DETERMINE, &mapped));
    *refreshed = PETSC_TRUE;
  }
  /* The caller records the final cache state after any term scaling has modified `mapped`. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
    PetscCall(VecDestroy(&mt->_unmapped_vec_work));
    PetscCall(MatDestroy(&mt->_unmapped_H));
    PetscCall(MatDestroy(&mt->_unmapped_Hpre));
    PetscCall(MatDestroy(&mt->_mapped_H));
    PetscCall(MatDestroy(&mt->_mapped_Hpre));
  }
  if (term != mt->term || map != mt->map) {
    mt->mapped_H_state.valid    = PETSC_FALSE;
    mt->mapped_Hpre_state.valid = PETSC_FALSE;
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
  PetscCall(VecDestroy(&mt->_mapped_vec_work));
  PetscCall(MatDestroy(&mt->_unmapped_H));
  PetscCall(MatDestroy(&mt->_unmapped_Hpre));
  PetscCall(MatDestroy(&mt->_mapped_H));
  PetscCall(MatDestroy(&mt->_mapped_Hpre));
  mt->mapped_H_state.valid    = PETSC_FALSE;
  mt->mapped_Hpre_state.valid = PETSC_FALSE;
  mt->mask                    = TAOTERM_MASK_NONE;
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
    PetscCheck((mt)->term, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "TaoTermMapping has no TaoTerm set"); \
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

/*
  TaoTermMappingGetWorkVecs - Get the row-space and column-space
  work vectors for a `TaoTerm`

  Collective

  Input Parameters:
+ mt   - the `TaoTermMapping`
. mode - `INSERT_VALUES` or `ADD_VALUES`
- g    - the destination vector, in the row space of `mt->map`

  Output Parameters:
+ mapped_g   - row-space buffer sized like `g`
- unmapped_g - column-space buffer the `TaoTerm` writes its raw result into

  Level: developer

  Notes:
  For `INSERT_VALUES`, `mapped_g` is `g` itself.
  For `ADD_VALUES`, `mapped_g` is a separate internal vector.

  `unmapped_g` is the same vector as `mapped_g`, unless `mt->map` is set,
  in which case it is a separate vector in the column space of `mt->map`.

  The internal vectors are allocated on first use.  Pair every call with
  `TaoTermMappingAccumulateWorkVecs()`.

.seealso: `TaoTermMapping`, `TaoTermMappingAccumulateWorkVecs()`
*/
static PetscErrorCode TaoTermMappingGetWorkVecs(TaoTermMapping *mt, InsertMode mode, Vec g, Vec *mapped_g, Vec *unmapped_g)
{
  PetscFunctionBegin;
  *mapped_g = g;
  if (mode == ADD_VALUES) {
    if (!mt->_mapped_vec_work) PetscCall(VecDuplicate(g, &mt->_mapped_vec_work));
    *mapped_g = mt->_mapped_vec_work;
  }
  *unmapped_g = *mapped_g;
  if (mt->map) {
    if (!mt->_unmapped_vec_work) PetscCall(TaoTermCreateSolutionVec(mt->term, &mt->_unmapped_vec_work));
    *unmapped_g = mt->_unmapped_vec_work;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingAccumulateWorkVecs - Combine the staged work vectors into `g`,
  applying `mt->map`, `mt->scale`, and the requested `InsertMode`

  Collective

  Input Parameters:
+ mt         - the `TaoTermMapping`
. mode       - `INSERT_VALUES` or `ADD_VALUES`
. mapped_g   - row-space work vector from `TaoTermMappingGetWorkVecs()`
- unmapped_g - column-space work vector from `TaoTermMappingGetWorkVecs()`

  Output Parameter:
. g - the destination vector, in the row space of `mt->map`

  Level: developer

  Notes:
  When `mt->map` is set, `mapped_g <- map^T * unmapped_g`; otherwise the two are the same vector.

  For `INSERT_VALUES`, `mapped_g` is `g` itself, and is scaled in place by `mt->scale`.
  For `ADD_VALUES`, `g <- g + mt->scale * mapped_g`.

  This is the counterpart to `TaoTermMappingGetWorkVecs()` and must be called with the vectors
  it returned.

.seealso: `TaoTermMapping`, `TaoTermMappingGetWorkVecs()`
*/
static PetscErrorCode TaoTermMappingAccumulateWorkVecs(TaoTermMapping *mt, InsertMode mode, Vec mapped_g, Vec unmapped_g, Vec g)
{
  PetscFunctionBegin;
  if (mt->map) PetscCall(MatMultTranspose(mt->map, unmapped_g, mapped_g));
  else PetscAssert(mapped_g == unmapped_g, PETSC_COMM_SELF, PETSC_ERR_PLIB, "without a map, mapped_g and unmapped_g must be the same vector returned by TaoTermMappingGetWorkVecs()");
  if (mode == ADD_VALUES) PetscCall(VecAXPY(g, mt->scale, mapped_g));
  else {
    PetscAssert(mapped_g == g, PETSC_COMM_SELF, PETSC_ERR_PLIB, "for INSERT_VALUES, mapped_g and g must be the same vector");
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
  PetscCall(TaoTermMappingGetWorkVecs(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeGradient(mt->term, Ax, params, unmapped_g));
  PetscCall(TaoTermMappingAccumulateWorkVecs(mt, mode, mapped_g, unmapped_g, g));
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
      *value = 0.0;
      PetscCall(VecZeroEntries(g));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermObjectiveMasked(mt->mask)) {
    if (mode == INSERT_VALUES) *value = 0.0;
    PetscCall(TaoTermMappingComputeGradient(mt, x, params, mode, g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermGradientMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(g));
    PetscCall(TaoTermMappingComputeObjective(mt, x, params, mode, value));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingGetWorkVecs(mt, mode, g, &mapped_g, &unmapped_g));
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermComputeObjectiveAndGradient(mt->term, Ax, params, &v, unmapped_g));
  PetscCall(TaoTermMappingAccumulateWorkVecs(mt, mode, mapped_g, unmapped_g, g));
  if (mode == ADD_VALUES) *value += mt->scale * v;
  else *value = mt->scale * v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingGetHessians - Get the column-space (unmapped) and row-space (mapped)
  Hessian matrices for a `TaoTerm`

  Collective

  Input Parameters:
+ mt   - the `TaoTermMapping`
. mode - `INSERT_VALUES` or `ADD_VALUES`
. H    - the destination Hessian, in the row space of `mt->map` (may be `NULL`)
- Hpre - the destination Hessian preconditioning matrix (may be `NULL`)

  Output Parameters:
+ mapped_H      - row-space matrix that will receive `map^T * unmapped_H * map`
. mapped_Hpre   - row-space matrix that will receive the mapped preconditioner
. unmapped_H    - column-space matrix the `TaoTerm` writes its raw Hessian into
- unmapped_Hpre - column-space matrix the `TaoTerm` writes its raw preconditioner into

  Level: developer

  Notes:
  This is the Hessian analogue of `TaoTermMappingGetWorkVecs()`: it selects the matrices that
  `TaoTermComputeHessian()` fills (`unmapped_H`, `unmapped_Hpre`) and the matrices that hold the
  mapped result destined for `H` and `Hpre` (`mapped_H`, `mapped_Hpre`).

  When `mt->map == NULL`, the `unmapped_H == mapped_H`.
  When `mt->map != NULL`, `unmapped_H`/`unmapped_Hpre` are separate matrices
  in the column space of `mt->map`, cached on `mt` and allocated on first use
  (with `TaoTermCreateHessianMatrices()` when the `TaoTerm` defines it).

  For `INSERT_VALUES` the `mapped_H == H`, unless a cached `mt->_mapped_H` (and `mt->_mapped_Hpre`)
  from `TaoTermMappingCreateHessianMatrices()` exists. The first genuine Hessian evaluation
  replaces this placeholder's implementation with the initial `MatPtAP()` result while preserving
  the outer object held by Tao and KSP; subsequent evaluations may reuse that product's symbolic state.

  For `ADD_VALUES` the mapped matrices are always separate internal matrices.

  Pair every call with `TaoTermMappingSetHessians()`.

.seealso: `TaoTermMapping`, `TaoTermMappingSetHessians()`, `TaoTermMappingGetWorkVecs()`, `TaoTermMappingCreateHessianMatrices()`
*/
static PetscErrorCode TaoTermMappingGetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat *mapped_H, Mat *mapped_Hpre, Mat *unmapped_H, Mat *unmapped_Hpre)
{
  PetscFunctionBegin;
  *mapped_H    = H;
  *mapped_Hpre = Hpre;
  /* When `mode == INSERT_VALUES`, and the per-summand cached _mapped_H exists, use it as the
     PtAP target. The first evaluation replaces its placeholder implementation with the initial
     product, after which it carries the symbolic state needed for MAT_REUSE_MATRIX. */
  if (mode == INSERT_VALUES) {
    if (H && mt->_mapped_H) *mapped_H = mt->_mapped_H;
    if (Hpre && mt->_mapped_Hpre) *mapped_Hpre = mt->_mapped_Hpre;
  }
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
  /* When the outer Hessian is matrix-free the caller passes H == NULL, but the term still has to
     run its Hessian evaluation to assemble the separate preconditioner Hpre.  A TAOTERMCALLBACKS
     term forwards H directly to a user callback that, per the classic TaoSetHessian() contract,
     assumes a valid matrix and dereferences it.  Reuse the summand's own Hessian matrix as
     throwaway scratch so the callback receives a valid H; TaoTermMappingSetHessians() will not
     propagate it to the (absent) outer H.  Only handled without a map, where mapped_H is the
     matrix the term writes into directly. */
  if (!*mapped_H && Hpre && !mt->map && mt->_mapped_H) *mapped_H = mt->_mapped_H;
  *unmapped_H    = *mapped_H;
  *unmapped_Hpre = *mapped_Hpre;
  if (mt->map) {
    if (H) *unmapped_H = mt->_unmapped_H;
    if (Hpre) *unmapped_Hpre = mt->_unmapped_Hpre;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingSetHessians - Assemble the destination Hessian matrices from the staged
  mapped/unmapped matrices, applying `mt->map`, `mt->scale`, and the requested `InsertMode`

  Collective

  Input Parameters:
+ mt            - the `TaoTermMapping`
. mode          - `INSERT_VALUES` or `ADD_VALUES`
. mapped_H      - row-space matrix from `TaoTermMappingGetHessians()`
. mapped_Hpre   - row-space preconditioner matrix from `TaoTermMappingGetHessians()`
. unmapped_H    - column-space matrix the `TaoTerm` filled in
- unmapped_Hpre - column-space preconditioner matrix the `TaoTerm` filled in

  Output Parameters:
+ H    - the destination Hessian, in the row space of `mt->map` (may be `NULL`)
- Hpre - the destination Hessian preconditioning matrix (may be `NULL`)

  Level: developer

  Notes:
  If `mt->map == NULL`, then `mapped_H == unmapped_H.
  Otherwise, `mapped_H <- map^T * unmapped_H * map`.

  For `INSERT_VALUES`, `H <- mapped_H` and is then scaled in place by `mt->scale`.
  For `ADD_VALUES`, `H <- H + mt->scale * mapped_H`.

  For `INSERT_VALUES` with a map, the `PtAP` is written into the cached `mt->_mapped_H` scratch
  and then copied back into the outer `H`.

  This is the counterpart to `TaoTermMappingGetHessians()` and must be called with the matrices it
  returned.

.seealso: `TaoTermMapping`, `TaoTermMappingGetHessians()`, `TaoTermMappingAccumulateWorkVecs()`, `TaoTermMappingCreateHessianMatrices()`
*/
static PetscErrorCode TaoTermMappingSetHessians(TaoTermMapping *mt, InsertMode mode, Mat H, Mat Hpre, Mat mapped_H, Mat mapped_Hpre, Mat unmapped_H, Mat unmapped_Hpre)
{
  PetscBool H_refreshed = PETSC_FALSE, Hpre_refreshed = PETSC_FALSE;

  PetscFunctionBegin;
  if (mt->map) {
    if (mapped_H) PetscCall(TaoTermMappingUpdateMappedHessian(unmapped_H, mt->map, mapped_H, &mt->mapped_H_state, &H_refreshed));
    if (mapped_Hpre && mapped_Hpre != mapped_H) PetscCall(TaoTermMappingUpdateMappedHessian(unmapped_Hpre, mt->map, mapped_Hpre, &mt->mapped_Hpre_state, &Hpre_refreshed));
  }
  if (mode == ADD_VALUES) {
    if (H) PetscCall(MatAXPY(H, mt->scale, mapped_H, UNKNOWN_NONZERO_PATTERN));
    if (Hpre) PetscCall(MatAXPY(Hpre, mt->scale, mapped_Hpre, UNKNOWN_NONZERO_PATTERN));
  } else {
    if (H && (mapped_H != H)) PetscCall(MatCopy(mapped_H, H, DIFFERENT_NONZERO_PATTERN));
    if (Hpre && (H != Hpre) && (mapped_Hpre != Hpre)) PetscCall(MatCopy(mapped_Hpre, Hpre, DIFFERENT_NONZERO_PATTERN));
    if (mt->scale != 1.0) {
      if (H && (!mt->map || mapped_H != H || H_refreshed)) PetscCall(MatScale(H, mt->scale));
      if (Hpre && Hpre != H && (!mt->map || mapped_Hpre != Hpre || Hpre_refreshed)) PetscCall(MatScale(Hpre, mt->scale));
    }
  }
  if (mt->map) {
    if (mapped_H) PetscCall(TaoTermMappedHessianStateGet(unmapped_H, mt->map, mapped_H, &mt->mapped_H_state));
    if (mapped_Hpre && mapped_Hpre != mapped_H) PetscCall(TaoTermMappedHessianStateGet(unmapped_Hpre, mt->map, mapped_Hpre, &mt->mapped_Hpre_state));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingEnsureRawHessians - Get or create the canonical raw Hessian storage for a `TaoTermMapping`

  Collective

  Input Parameter:
. mt - the `TaoTermMapping`

  Output Parameters:
+ raw_H    - the term-space, unscaled Hessian matrix, or `NULL` if none can be created
- raw_Hpre - the term-space, unscaled preconditioning matrix, or `NULL` if `raw_H` is `NULL`

  Level: developer

  Notes:
  For a mapped term, only `mt->_unmapped_H` and `mt->_unmapped_Hpre` are raw storage;
  `mt->_mapped_H` and `mt->_mapped_Hpre` are outer-space `MatPtAP()` targets.  Without a map,
  mapped and unmapped storage have the same layout and may alias.

  Missing raw storage is created with `TaoTermCreateHessianMatrices()` when that operation is
  defined.  If no separate raw preconditioning matrix exists, `raw_Hpre` is set to `raw_H`.

.seealso: `TaoTermMapping`, `TaoTermCreateHessianMatrices()`, `TaoTermMappingApplyHessian()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingEnsureRawHessians(TaoTermMapping *mt, Mat *raw_H, Mat *raw_Hpre)
{
  PetscFunctionBegin;
  if (!mt->_unmapped_H && (mt->map || !mt->_mapped_H)) {
    PetscBool is_defined = PETSC_FALSE;

    PetscCall(TaoTermIsCreateHessianMatricesDefined(mt->term, &is_defined));
    if (is_defined) {
      Mat H = NULL, Hpre = NULL;

      PetscCall(TaoTermCreateHessianMatrices(mt->term, &H, (mt->_unmapped_Hpre || (!mt->map && mt->_mapped_Hpre)) ? NULL : &Hpre));
      mt->_unmapped_H = H;
      if (Hpre) mt->_unmapped_Hpre = Hpre;
      if (!mt->map) {
        if (H && !mt->_mapped_H) {
          PetscCall(PetscObjectReference((PetscObject)H));
          mt->_mapped_H = H;
        }
        if (Hpre && !mt->_mapped_Hpre) {
          PetscCall(PetscObjectReference((PetscObject)Hpre));
          mt->_mapped_Hpre = Hpre;
        }
      }
    }
  }
  if (mt->map) {
    *raw_H    = mt->_unmapped_H;
    *raw_Hpre = mt->_unmapped_Hpre;
  } else {
    *raw_H    = mt->_unmapped_H ? mt->_unmapped_H : mt->_mapped_H;
    *raw_Hpre = mt->_unmapped_Hpre ? mt->_unmapped_Hpre : mt->_mapped_Hpre;
  }
  if (!*raw_Hpre) *raw_Hpre = *raw_H;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  TaoTermMappingApplyHessian - Map, scale, and accumulate raw Hessian matrices into destination matrices

  Collective

  Input Parameters:
+ mt            - the `TaoTermMapping`
. mode          - `INSERT_VALUES` or `ADD_VALUES`
. unmapped_H    - (optional) the raw, unscaled Hessian in the term solution space
- unmapped_Hpre - (optional) the raw, unscaled preconditioning matrix in the term solution space

  Output Parameters:
+ H    - (optional) the destination Hessian in the outer solution space
- Hpre - (optional) the destination preconditioning matrix in the outer solution space

  Level: developer

  Notes:
  If `mt->map` is set, this routine forms `map^T * unmapped_H * map` and the corresponding
  preconditioning matrix using the mapping's cached mapped storage.  Otherwise the raw matrices
  are used directly.

  `INSERT_VALUES` overwrites each destination and `ADD_VALUES` accumulates into it.  In both
  cases the contribution is scaled by `mt->scale`.

.seealso: `TaoTermMapping`, `TaoTermMappingEnsureRawHessians()`, `TaoTermMappingComputeHessian()`
*/
PETSC_INTERN PetscErrorCode TaoTermMappingApplyHessian(TaoTermMapping *mt, InsertMode mode, Mat unmapped_H, Mat unmapped_Hpre, Mat H, Mat Hpre)
{
  Mat mapped_H = unmapped_H, mapped_Hpre = unmapped_Hpre;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (mt->map) {
    if (H && unmapped_H) {
      if (!mt->_mapped_H) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->_mapped_H));
      mapped_H = mt->_mapped_H;
    } else mapped_H = NULL;
    if (Hpre && unmapped_Hpre) {
      if (unmapped_Hpre == unmapped_H) mapped_Hpre = mapped_H;
      else {
        if (!mt->_mapped_Hpre) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->_mapped_Hpre));
        mapped_Hpre = mt->_mapped_Hpre;
      }
    } else mapped_Hpre = NULL;
  }
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Either called by TaoComputeHessian (one term in Tao), or by TAOTERMSUM
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
//         -> TaoTermSumHessCacheGetHessian
//         -> TaoTermMappingApplyHessian
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessian(TaoTermMapping *mt, Vec x, Vec params, InsertMode mode, Mat H, Mat Hpre)
{
  Vec Ax;
  Mat mapped_H, mapped_Hpre, unmapped_H = NULL, unmapped_Hpre = NULL;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (mt->map) {
    /* A matrix-free (shell) outer Hessian applies map^H (grad^2 f)(map x) map lazily in MatMult();
       refresh its cached (x, params) here and skip assembly, mirroring TaoTermComputeHessian(). */
    PetscCall(TaoTermMappingPreprocessHessianShells(mt, x, params, &H, &Hpre));
    if (!H && !Hpre) PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) {
      if (H) {
        PetscCall(MatZeroEntries(H));
        PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
      }
      if (Hpre && Hpre != H) {
        PetscCall(MatZeroEntries(Hpre));
        PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
      }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, x, &Ax));
  PetscCall(TaoTermMappingGetHessians(mt, mode, H, Hpre, &mapped_H, &mapped_Hpre, &unmapped_H, &unmapped_Hpre));
  PetscCall(TaoTermComputeHessian(mt->term, Ax, params, unmapped_H, unmapped_Hpre));
  PetscCall(TaoTermMappingSetHessians(mt, mode, H, Hpre, mapped_H, mapped_Hpre, unmapped_H, unmapped_Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Hessian-vector product through the mapping.

   Input `Ax` may be `x` in outerspace, if `map == NULL`. */
PETSC_INTERN PetscErrorCode TaoTermMappingComputeHessianMult(TaoTermMapping *mt, Vec Ax, Vec params, Mat unmapped_H, Vec v, InsertMode mode, Vec Hv)
{
  Vec mapped_Hv, unmapped_Hv = NULL, Av;

  PetscFunctionBegin;
  TaoTermMappingCheckInsertMode(mt, mode);
  if (TaoTermHessianMasked(mt->mask)) {
    if (mode == INSERT_VALUES) PetscCall(VecZeroEntries(Hv));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(TaoTermMappingMap(mt, v, &Av));
  PetscCall(TaoTermMappingGetWorkVecs(mt, mode, Hv, &mapped_Hv, &unmapped_Hv));
  if (unmapped_H) PetscCall(MatMult(unmapped_H, Av, unmapped_Hv));
  else PetscCall(TaoTermComputeHessianMult(mt->term, Ax, params, Av, unmapped_Hv));
  PetscCall(TaoTermMappingAccumulateWorkVecs(mt, mode, mapped_Hv, unmapped_Hv, Hv));
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

/*
 * Internal function to create Hessian matrices for TaoTermMapping
 *
 * map: m x n
 *
 * This function will internally create unmapped, and mapped  H and Hpre,
 * and return H \gets mt->_mapped_H, and Hpre \gets mt->_mapped_Hpre.
 *
 * if (mt->map)
 *   mapped:   n x n
 *   unmapped: m x m
 *
 * else
 *   mapped:   n x n
 *   unmapped: n x n
 *
 */
PETSC_INTERN PetscErrorCode TaoTermMappingCreateHessianMatrices(TaoTermMapping *mt, Mat *H, Mat *Hpre)
{
  Mat       uH, uHpre, mH, mHpre;
  PetscBool is_sum;

  PetscFunctionBegin;
  uH    = mt->_unmapped_H;
  uHpre = mt->_unmapped_Hpre;
  mH    = mt->_mapped_H;
  mHpre = mt->_mapped_Hpre;
  PetscCall(PetscObjectTypeCompare((PetscObject)mt->term, TAOTERMSUM, &is_sum));
  if (is_sum && mt->map) PetscCall(PetscInfo(mt->term, "%s: TaoTermType is TAOTERMSUM, but Map is given. Ignoring it.\n", ((PetscObject)mt->term)->prefix));
  /* H may be NULL to request only the Hpre (preconditioner) matrices -- used when the outer
     Hessian is matrix-free (a shell) and the mapped Hessian matrix itself is not needed. */
  PetscCheck(Hpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "TaoTermMappingCreateHessianMatrices does not take NULL input Hpre");
  if (!mt->map) {
    // mt->_unmapped_{H,Hpre} == mt->_unmapped_{H,Hpre}
    if (uH && mH) PetscCheck(uH == mH, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian and unmapped Hessian must be same");
    if (uHpre && mHpre) PetscCheck(uHpre == mHpre, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_USER, "For unmapped TaoTerm, mapped Hessian preconditioner and unmapped Hessian preconditioner needs to be same");

    // If mapped matrices are present, it should be set to unmapped matrices
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
    // If mapped matrices are NULL, it should be set to mapped matrices
    if (mt->_unmapped_H && !mt->_mapped_H) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
      mt->_mapped_H = mt->_unmapped_H;
    }
    if (mt->_unmapped_Hpre && !mt->_mapped_Hpre) {
      PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
      mt->_mapped_Hpre = mt->_unmapped_Hpre;
    }

    // always returns Hpre, even if same as H
    if (H && *H != mt->_unmapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_H));
    if (*Hpre != mt->_unmapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_unmapped_Hpre));
    if (H) *H = mt->_unmapped_H;
    *Hpre = mt->_unmapped_Hpre;
  } else {
    PetscBool is_outer_shell = PETSC_FALSE;

    /* A matrix-free (shell) outer Hessian: build a shell that applies map^H (grad^2 f)(map x) map
       through the term's Hessian-vector product, instead of assembling map^H H map (PtAP). */
    if (mt->term->H_mattype) PetscCall(PetscStrcmp(mt->term->H_mattype, MATSHELL, &is_outer_shell));
    if (is_outer_shell) {
      PetscCheck(mt->term->Hpre_is_H, PetscObjectComm((PetscObject)mt->term), PETSC_ERR_SUP, "A separate preconditioner matrix with a matrix-free (shell) Hessian is not supported for a mapped term; use -tao_term_hessian_pre_is_hessian true");
      if (!mt->_mapped_H) PetscCall(TaoTermMappingCreateHessianShell(mt, &mt->_mapped_H));
      if (!mt->_mapped_Hpre) {
        PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
        mt->_mapped_Hpre = mt->_mapped_H;
      }
      if (H) {
        if (*H != mt->_mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
        *H = mt->_mapped_H;
      }
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      *Hpre = mt->_mapped_Hpre;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    // create _unmapped only if they are empty
    PetscCall(TaoTermCreateHessianMatrices(mt->term, (mt->_unmapped_H) ? NULL : &mt->_unmapped_H, (mt->_unmapped_Hpre) ? NULL : &mt->_unmapped_Hpre));
    /* Tao and KSP need stable outer matrix objects during setup, before a genuine raw Hessian
       exists. Create empty placeholders with the mapped layout; the first Hessian evaluation
       computes the initial PtAP and replaces each placeholder's implementation in place. */
    if (mt->_unmapped_H && !mt->_mapped_H) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->_mapped_H));
    if (H) {
      if (*H != mt->_mapped_H) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
      *H = mt->_mapped_H;
    }
    if (mt->_unmapped_Hpre == mt->_unmapped_H) {
      // Hpre_is_H true, so mapped_Hpre = mapped_H
      if (!mt->_mapped_Hpre) {
        PetscCall(PetscObjectReference((PetscObject)mt->_mapped_H));
        mt->_mapped_Hpre = mt->_mapped_H;
      }
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      *Hpre = mt->_mapped_Hpre;
    } else {
      if (!mt->_mapped_Hpre) PetscCall(TaoTermMappingCreateMappedHessianPlaceholder(mt, &mt->_mapped_Hpre));
      if (*Hpre != mt->_mapped_Hpre) PetscCall(PetscObjectReference((PetscObject)mt->_mapped_Hpre));
      *Hpre = mt->_mapped_Hpre;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
