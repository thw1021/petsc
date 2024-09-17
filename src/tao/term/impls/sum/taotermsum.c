#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Sum TaoTerm_Sum;

struct _n_TaoTerm_Sum {
  PetscInt       n_terms;
  TaoMappedTerm *terms;
};

/*@
  TaoTermSumConcatenateParameters - Concatenate the parameters for subterms into a parameter vector for a `TAOTERMSUM`

  Collective

  Input Parameters:
+ term      - a `TaoTerm` of type `TAOTERMSUM`
- subparams - an array of parameters `Vec`s, one for each subterm in the sum.  An entry can be NULL for a subterm that doesn't take parameters.

  Output Parameter:
. params - a `Vec` of type `VECNEST` that concatenates all of the parameters

  Level: intermediate

  Note:
  This is a wrapper around `VecCreateNest()`, but that function does not allow NULL for any of the `Vec`s in the array.  A 0-length
  vector will be created for each NULL `Vec` that wil be internally ignored by `TAOTERMSUM`.

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSUM`, `VecCreateNest()`
@*/
PetscErrorCode TaoTermSumConcatenateParameters(TaoTerm term, Vec subparams[], Vec *params)
{
  PetscInt n_terms;
  Vec     *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms));
  PetscCall(PetscMalloc1(n_terms, &p));
  for (PetscInt i = 0; i < n_terms; i++) {
    if (subparams[i]) {
      PetscValidHeaderSpecific(subparams[i], VEC_CLASSID, 2);
      p[i] = subparams[i];
    } else {
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)term), 1, 0, 0, NULL, &p[i]));
    }
  }
  PetscCall(VecCreateNest(PetscObjectComm((PetscObject)term), n_terms, NULL, p, params));
  PetscCall(PetscFree(p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermDestroy_Sum(TaoTerm term)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < sum->n_terms; i++) PetscCall(TaoMappedTermReset(&sum->terms[i]));
  PetscCall(PetscFree(sum->terms));
  PetscCall(PetscFree(sum));
  term->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetNumSubterms_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetNumSubterms_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetSubterm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubterm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumAddSubterm_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Sum(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;
  PetscBool    iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Sum of %" PetscInt_FMT " terms:\n", sum->n_terms));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    for (PetscInt i = 0; i < sum->n_terms; i++) {
      TaoMappedTerm *summand = &sum->terms[i];

      PetscCall(PetscViewerASCIIPrintf(viewer, "Summand %" PetscInt_FMT ":\n", i));
      PetscCall(PetscViewerASCIIPushTab(viewer));

      PetscCall(PetscViewerASCIIPrintf(viewer, "Scale: %g\n", (double)summand->scale));
      if (summand->map == NULL) PetscCall(PetscViewerASCIIPrintf(viewer, "Map: unmapped\n"));
      else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Map:\n"));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "Term:\n"));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(TaoTermView(summand->term, viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));

      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumSetNumSubterms - Set the number of terms in the sum

  Collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSUM`
- n_terms - the number of terms that will be in the sum

  Level: intermediate

  Note:
  If `n_terms` is smaller than the current number of terms, the trailing terms will be dropped.

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSUM`
@*/
PetscErrorCode TaoTermSumSetNumSubterms(TaoTerm term, PetscInt n_terms)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(term, n_terms, 2);
  PetscTryMethod(term, "TaoTermSumSetNumSubterms_C", (TaoTerm, PetscInt), (term, n_terms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumSetNumSubterms_Sum(TaoTerm term, PetscInt n_terms)
{
  TaoTerm_Sum   *sum         = (TaoTerm_Sum *)term->data;
  PetscInt       n_terms_old = sum->n_terms;
  TaoMappedTerm *new_summands;

  PetscFunctionBegin;
  if (n_terms == n_terms_old) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = n_terms; i < n_terms_old; i++) PetscCall(TaoMappedTermReset(&sum->terms[i]));
  PetscCall(PetscMalloc1(n_terms, &new_summands));
  PetscCall(PetscArraycpy(new_summands, sum->terms, PetscMin(n_terms, n_terms_old)));
  PetscCall(PetscArrayzero(&new_summands[n_terms_old], PetscMax(0, n_terms - n_terms_old)));
  for (PetscInt i = n_terms_old; i < n_terms; i++) {
    TaoMappedTerm *summand = &new_summands[i];

    summand->scale = 1.0;
  }
  PetscCall(PetscFree(sum->terms));
  sum->terms   = new_summands;
  sum->n_terms = n_terms;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumGetNumSubterms - Get the number of terms in the sum

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMSUM`

  Output Parameter:
. n_terms - the number of terms that will be in the sum

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSUM`
@*/
PetscErrorCode TaoTermSumGetNumSubterms(TaoTerm term, PetscInt *n_terms)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(n_terms, 2);
  PetscUseMethod(term, "TaoTermSumGetNumSubterms_C", (TaoTerm, PetscInt *), (term, n_terms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumGetNumSubterms_Sum(TaoTerm term, PetscInt *n_terms)
{
  TaoTerm_Sum *sum = (TaoTerm_Sum *)term->data;

  PetscFunctionBegin;
  *n_terms = sum->n_terms;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumGetSubterm - Get the data for a subterm in a sum of terms

  Not collective

  Input Parameter:
+ term  - a `TaoTerm` of type `TAOTERMSUM`
- index - a number $0 \leq i < n$, where $n$ is the number of terms in `TaoTermSumGetNumSubterms()`

  Output Parameters:
+ name    - the name assigned to the subterm
. scale   - the coefficient scaling the term in the sum
. subterm - the `TaoTerm` for the subterm
- map     - the map from the term solution space to the subterm solution space (NULL if they are the same space)

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSUM`
@*/
PetscErrorCode TaoTermSumGetSubterm(TaoTerm term, PetscInt index, const char **name, PetscReal *scale, TaoTerm *subterm, Mat *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (name) PetscAssertPointer(name, 3);
  if (subterm) PetscAssertPointer(subterm, 5);
  if (scale) PetscAssertPointer(scale, 4);
  if (map) PetscAssertPointer(map, 6);
  PetscUseMethod(term, "TaoTermSumGetSubterm_C", (TaoTerm, PetscInt, const char **, PetscReal *, TaoTerm *, Mat *), (term, index, name, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumGetSubterm_Sum(TaoTerm term, PetscInt index, const char **name, PetscReal *scale, TaoTerm *subterm, Mat *map)
{
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand = &sum->terms[index];
  PetscCall(TaoMappedTermGetData(summand, name, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumSetSubterm - Set a subterm in a sum of terms

  Collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSUM`
. index   - a number $0 \leq i < n$, where $n$ is the number of terms in `TaoTermSumGetNumSubterms()`
. name    - the name assigned to the subterm (optional, can be NULL)
. scale   - the coefficient scaling the term in the sum
. subterm - the `TaoTerm` for the subterm
- map     - the map from the term solution space to the subterm solution space (NULL if they are the same space)

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSUM`
@*/
PetscErrorCode TaoTermSumSetSubterm(TaoTerm term, PetscInt index, const char name[], PetscReal scale, TaoTerm subterm, Mat map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveInt(term, index, 2);
  if (name) PetscAssertPointer(name, 3);
  PetscValidHeaderSpecific(subterm, TAOTERM_CLASSID, 5);
  PetscValidLogicalCollectiveReal(term, scale, 4);
  if (map) PetscValidHeaderSpecific(map, MAT_CLASSID, 6);
  PetscTryMethod(term, "TaoTermSumSetSubterm_C", (TaoTerm, PetscInt, const char[], PetscReal, TaoTerm, Mat), (term, index, name, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumSetSubterm_Sum(TaoTerm term, PetscInt index, const char name[], PetscReal scale, TaoTerm subterm, Mat map)
{
  TaoTerm_Sum   *sum = (TaoTerm_Sum *)term->data;
  TaoMappedTerm *summand;

  PetscFunctionBegin;
  PetscCheck(index >= 0 && index < sum->n_terms, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in [0, %" PetscInt_FMT ")", index, sum->n_terms);
  summand = &sum->terms[index];
  PetscCall(TaoMappedTermSetData(summand, name, scale, subterm, map));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSumAddSubterm - Append a subterm to the terms being summed

  Collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSUM`
. name    - the name assigned to the subterm (optional, can be NULL)
. scale   - the coefficient scaling the term in the sum
. subterm - the `TaoTerm` for the subterm
- map     - the map from the term solution space to the subterm solution space (NULL if they are the same space)

  Output Parameter:
. index - the index of the newly added term (optional, pass NULL if not needed)

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERMSUM`
@*/
PetscErrorCode TaoTermSumAddSubterm(TaoTerm term, const char name[], PetscReal scale, TaoTerm subterm, Mat map, PetscInt *index)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (name) PetscAssertPointer(name, 2);
  PetscValidHeaderSpecific(subterm, TAOTERM_CLASSID, 4);
  PetscValidLogicalCollectiveReal(term, scale, 3);
  if (map) PetscValidHeaderSpecific(map, MAT_CLASSID, 5);
  if (index) PetscAssertPointer(index, 6);
  PetscTryMethod(term, "TaoTermSumAddSubterm_C", (TaoTerm, const char[], PetscReal, TaoTerm, Mat, PetscInt *), (term, name, scale, subterm, map, index));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSumAddSubterm_Sum(TaoTerm term, const char name[], PetscReal scale, TaoTerm subterm, Mat map, PetscInt *index)
{
  PetscInt n_terms_old;

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms_old));
  PetscCall(TaoTermSumSetNumSubterms(term, n_terms_old + 1));
  PetscCall(TaoTermSumSetSubterm(term, n_terms_old, name, scale, subterm, map));
  if (index) *index = n_terms_old;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_Sum(TaoTerm term, PetscOptionItems *PetscOptionsObject)
{
  PetscInt    n_terms;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(TaoTermSumGetNumSubterms(term, &n_terms));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)term, &prefix));
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm sum options");
  PetscCall(PetscOptionsBoundedInt("-taoterm_sum_num_subterms", "The number of terms in the sum", "TaoTermSumSetNumSubterms", n_terms, &n_terms, NULL, 0));
  PetscCall(TaoTermSumSetNumSubterms(term, n_terms));
  for (PetscInt i = 0; i < n_terms; i++) {
    const char *name;
    Mat         map;
    PetscReal   scale;
    TaoTerm     subterm;
    char        arg[256];
    char        newname[256];
    PetscBool   flg;

    PetscCall(TaoTermSumGetSubterm(term, i, &name, &scale, &subterm, &map));
    if (subterm == NULL) {
      char postfix[256];

      PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)term), &subterm));
      PetscCall(PetscSNPrintf(postfix, 256, "subterm_%" PetscInt_FMT "_", i));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)subterm, postfix));
    } else PetscCall(PetscObjectReference((PetscObject)subterm));
    PetscCall(TaoTermSetFromOptions(subterm));

    PetscCall(PetscSNPrintf(arg, 256, "-taoterm_sum_subterm_%" PetscInt_FMT "_scale", i));
    PetscCall(PetscOptionsReal(arg, "The scale of the subterm in the sum", "TaoTermSumSetSubterm", scale, &scale, NULL));

    PetscCall(PetscSNPrintf(arg, 256, "-taoterm_sum_subterm_%" PetscInt_FMT "_name", i));
    PetscCall(PetscOptionsString(arg, "The name of the subterm in the sum", "TaoTermSumSetSubterm", name, newname, 256, &flg));

    if (map) PetscCall(MatSetFromOptions(map));
    PetscCall(TaoTermSumSetSubterm(term, i, flg ? newname : name, scale, subterm, map));
    PetscCall(TaoTermDestroy(&subterm));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// convert zero-length vectors to NULL
static PetscErrorCode TaoTermSumVecArrayGetSubvec(Vec vecs[], PetscInt i, Vec *subvec)
{
  PetscInt N;

  PetscFunctionBegin;
  *subvec = NULL;
  if (vecs) {
    PetscCall(VecGetSize(vecs[i], &N));
    if (N > 0) { *subvec = vecs[i]; }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjective_Sum(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(VecNestGetSubVecs(params, NULL, &sub_params));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand = &sum->terms[i];
    Vec            sub_param;

    PetscCall(TaoTermSumVecArrayGetSubvec(sub_params, i, &sub_param));
    PetscCall(TaoMappedTermObjective(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, value));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermGradient_Sum(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(VecNestGetSubVecs(params, NULL, &sub_params));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand = &sum->terms[i];
    Vec            sub_param;

    PetscCall(TaoTermSumVecArrayGetSubvec(sub_params, i, &sub_param));
    PetscCall(TaoMappedTermGradient(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermObjectiveAndGradient_Sum(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(VecNestGetSubVecs(params, NULL, &sub_params));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand   = &sum->terms[i];
    Vec            sub_param = sub_params ? sub_params[i] : NULL;

    PetscCall(TaoMappedTermObjectiveAndGradient(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, value, g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermHessian_Sum(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Sum *sum        = (TaoTerm_Sum *)term->data;
  Vec         *sub_params = NULL;

  PetscFunctionBegin;
  if (params) PetscCall(VecNestGetSubVecs(params, NULL, &sub_params));
  for (PetscInt i = 0; i < sum->n_terms; i++) {
    TaoMappedTerm *summand = &sum->terms[i];
    Vec            sub_param;

    PetscCall(TaoTermSumVecArrayGetSubvec(sub_params, i, &sub_param));
    PetscCall(TaoMappedTermHessian(summand, x, sub_param, i == 0 ? INSERT_VALUES : ADD_VALUES, H, Hpre == H ? NULL : Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMSUM - A `TaoTerm` that is a sum of multiple other terms.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Sum(TaoTerm term)
{
  TaoTerm_Sum *sum;

  PetscFunctionBegin;
  PetscCall(PetscNew(&sum));
  term->data = (void *)sum;

  term->ops->destroy              = TaoTermDestroy_Sum;
  term->ops->view                 = TaoTermView_Sum;
  term->ops->setfromoptions       = TaoTermSetFromOptions_Sum;
  term->ops->objective            = TaoTermObjective_Sum;
  term->ops->gradient             = TaoTermGradient_Sum;
  term->ops->objectiveandgradient = TaoTermObjectiveAndGradient_Sum;
  term->ops->hessian              = TaoTermHessian_Sum;
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetNumSubterms_C", TaoTermSumGetNumSubterms_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetNumSubterms_C", TaoTermSumSetNumSubterms_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumGetSubterm_C", TaoTermSumGetSubterm_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumSetSubterm_C", TaoTermSumSetSubterm_Sum));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermSumAddSubterm_C", TaoTermSumAddSubterm_Sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}
