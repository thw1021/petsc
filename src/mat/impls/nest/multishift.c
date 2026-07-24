#include <../src/mat/impls/nest/matnestimpl.h> /*I "petscmat.h" I*/

static PetscErrorCode MatMultiShiftDestroy(PetscCtxRt ctx)
{
  Mat_MultiShift mctx = *(Mat_MultiShift *)ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&mctx->K));
  PetscCall(MatDestroy(&mctx->M));
  PetscCall(PetscFree(mctx->sigma));
  PetscCall(PetscFree(mctx->cmplx));
  PetscCall(PetscFree(mctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build G = K + sigma M, explicitly or not */
static PetscErrorCode BuildShiftedMatrix(Mat K, PetscScalar sigma, Mat M, MatStructure str, PetscBool explicitmat, Mat *G)
{
  PetscInt    Mk, Nk, mk, nk;
  PetscScalar scal[] = {1.0, 1.0};

  PetscFunctionBegin;
  if (explicitmat) {
    PetscCall(MatDuplicate(K, MAT_COPY_VALUES, G));
    if (M) PetscCall(MatAXPY(*G, sigma, M, str));
    else PetscCall(MatShift(*G, sigma));
  } else {
    PetscCall(MatGetSize(K, &Mk, &Nk));
    PetscCall(MatGetLocalSize(K, &mk, &nk));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)K), G));
    PetscCall(MatSetSizes(*G, mk, nk, Mk, Nk));
    PetscCall(MatSetType(*G, MATCOMPOSITE));
    PetscCall(MatCompositeAddMat(*G, K));
    if (M) PetscCall(MatCompositeAddMat(*G, M));
    PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));
    if (M) {
      scal[1] = sigma;
      PetscCall(MatCompositeSetScalings(*G, scal));
    } else PetscCall(MatShift(*G, sigma));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !PetscDefined(USE_COMPLEX)
/* Build G = sigma M, explicitly or not, where M may be I; K is used for dimensions only */
static PetscErrorCode BuildScaledMatrix(Mat K, PetscScalar sigma, Mat M, PetscBool explicitmat, Mat *G)
{
  PetscInt Mk, Nk, mk, nk;

  PetscFunctionBegin;
  if (explicitmat) {
    if (M) {
      PetscCall(MatDuplicate(M, MAT_COPY_VALUES, G));
      PetscCall(MatScale(*G, sigma));
    } else { // M=I
      PetscCall(MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, G));
      PetscCall(MatShift(*G, sigma));
    }
  } else {
    PetscCall(MatGetSize(K, &Mk, &Nk));
    PetscCall(MatGetLocalSize(K, &mk, &nk));
    if (M) {
      PetscCall(MatCreate(PetscObjectComm((PetscObject)K), G));
      PetscCall(MatSetSizes(*G, mk, nk, Mk, Nk));
      PetscCall(MatSetType(*G, MATCOMPOSITE));
      PetscCall(MatCompositeAddMat(*G, M));
      PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));
      PetscCall(MatCompositeSetScalings(*G, &sigma));
    } else PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)K), mk, nk, Mk, Nk, sigma, G));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
  MatCreateMultiShift - Creates a matrix that represents a family of shifted matrices
  $K + \sigma_i M$ for a number of shifts $\sigma_i$.

  Collective

  Input Parameters:
+ K           - the first `Mat` (stiffness) forming the shifted matrices
. nshift      - number of shifts
. sigma       - array of shifts $\sigma_i$
. M           - the second `Mat` (mass) forming the shifted matrices (if `NULL` the identity matrix is assumed)
. explicitmat - whether the shifted matrices should be built explicitly or not
- str         - `MatStructure` flag

  Output Parameter:
. A - the resulting matrix

  Level: intermediate

  Notes:
  This is intended for solving a family of shifted linear systems, $(K + \sigma_i M) x_i = b$,
  where in some applications $M = I$. This function returns a `MATNEST` `A` whose `nshift`
  diagonal blocks are the matrices $K + \sigma_i M$, either built explicitly or not, depending
  on the `explicitmat` argument. If not explicit, the diagonal blocks are created as `MATCOMPOSITE`.
  If `explicitmat` is true, and $M$ is not the identity, then $K + \sigma_i M$ is built with a
  call to `MatAXPY()`, where the flag `str` is used to indicate the relation between the sparsity
  patterns of $K$ and $M$; otherwise, the argument `str` is not used.

  To solve all the shifted linear systems simultaneously, one should pass this matrix to a
  `KSP` solver such as `KSPEKSM`, along with compatible solution and right-hand-side vectors. Since
  the right-hand-side vector $b$ is the same for all the shifted linear systems, one can call
  `MatMultiShiftCreateVec()` to easily create a nested `Vec` containing `nshift` copies of $b$.

  The shifts are allowed to be complex, even in real scalars. Note that in case of real scalars,
  shifts must come in complex conjugate pairs. Furthermore, in real scalars if any `sigma` is
  complex then the generated matrix `A` will no longer be block diagonal, but contain 2x2 blocks
  for each complex conjugate pair.

.seealso: [](ch_matrices), `MatMultiShiftCreateVec()`, `MatAXPY()`, `MATCOMPOSITE`, `KSP`, `KSPEKSM`
@*/
PetscErrorCode MatCreateMultiShift(Mat K, PetscInt nshift, PetscComplex sigma[], Mat M, PetscBool explicitmat, MatStructure str, Mat *A)
{
  PetscInt       i;
  Mat           *mats;
  Mat_MultiShift mctx;
#if !PetscDefined(USE_COMPLEX)
  PetscReal re1, im1, re2, im2;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(K, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(K, nshift, 2);
  PetscAssertPointer(sigma, 3);
  if (M) PetscValidHeaderSpecific(M, MAT_CLASSID, 4);
  PetscValidLogicalCollectiveBool(K, explicitmat, 5);
  PetscValidLogicalCollectiveEnum(K, str, 6);
  PetscAssertPointer(A, 7);

  PetscCheck(nshift > 0, PetscObjectComm((PetscObject)K), PETSC_ERR_ARG_OUTOFRANGE, "The value nshift must be > 0");

  /* build context and array of shifted matrices */
  PetscCall(PetscNew(&mctx));
  PetscCall(PetscObjectReference((PetscObject)K));
  mctx->K = K;
  PetscCall(PetscObjectReference((PetscObject)M));
  mctx->M      = M;
  mctx->nshift = nshift;
  mctx->str    = str;
  PetscCall(PetscMalloc1(nshift, &mctx->sigma));
  if (!PetscDefined(USE_COMPLEX)) PetscCall(PetscCalloc1(nshift, &mctx->cmplx));
  PetscCall(PetscCalloc1(nshift * nshift, &mats));

  for (i = 0; i < nshift; i++) {
#if PetscDefined(USE_COMPLEX)
    mctx->sigma[i] = sigma[i];
    PetscCall(BuildShiftedMatrix(K, sigma[i], M, str, explicitmat, mats + i + i * nshift));
#else
    mctx->sigma[i] = PetscRealPartComplex(sigma[i]);
    PetscCall(BuildShiftedMatrix(K, mctx->sigma[i], M, str, explicitmat, mats + i + i * nshift));
    im1 = PetscImaginaryPartComplex(sigma[i]);
    if (im1 != 0.0) {
      PetscCheck(i < nshift - 1, PetscObjectComm((PetscObject)K), PETSC_ERR_ARG_WRONG, "The last shift of the array is complex; shifts must come in complex conjugate pairs");
      re1 = PetscRealPartComplex(sigma[i]);
      re2 = PetscRealPartComplex(sigma[i + 1]);
      im2 = PetscImaginaryPartComplex(sigma[i + 1]);
      PetscCheck(re1 == re2 && im1 == -im2, PetscObjectComm((PetscObject)K), PETSC_ERR_ARG_WRONG, "Shift must come in complex conjugate pairs, which must be consecutive: %g%+gi vs %g%+gi", (double)re1, (double)im1, (double)re2, (double)im2);
      mctx->sigma[i + 1] = im1;
      mctx->cmplx[i]     = PETSC_TRUE;
      // 2x2 block in the nested matrix
      PetscCall(PetscObjectReference((PetscObject)mats[i + i * nshift]));
      mats[i + 1 + (i + 1) * nshift] = mats[i + i * nshift];
      PetscCall(BuildScaledMatrix(K, im1, M, explicitmat, mats + i + (i + 1) * nshift));
      PetscCall(BuildScaledMatrix(K, -im1, M, explicitmat, mats + i + 1 + i * nshift));
      i++; // skip next shift
    }
#endif
  }

  /* build MATNEST */
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)K), nshift, NULL, nshift, NULL, mats, A));
  for (i = 0; i < nshift * nshift; i++) PetscCall(MatDestroy(&mats[i]));
  PetscCall(PetscFree(mats));

  /* compose context */
  PetscCall(PetscObjectContainerCompose((PetscObject)*A, "MatMultiShift", mctx, MatMultiShiftDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMultiShiftCreateVec - Creates a nested vector that is compatible with a matrix
  created with `MatCreateMultiShift()`.

  Collective

  Input Parameters:
+ A - a `Mat` created with `MatCreateMultiShift()`
- v - an optional vector (set to `NULL` if not needed)

  Output Parameter:
. vout - the resulting vector

  Level: intermediate

  Notes:
  The result is a nested vector compatible with `A`, so that it can be e.g. multiplied against.
  If the input vector `v` is passed, `vout` will contain `nshift` references to `v`, where
  `nshift` is the number of subvectors of `vout`. Hence, it is intended as a read-only right-hand
  side, not a solution vector).

  In real scalars, in the case of a complex-conjugate pair only the first block of the pair is
  set while the second is left zero.

.seealso: [](ch_matrices), `MatCreateMultiShift()`
@*/
PetscErrorCode MatMultiShiftCreateVec(Mat A, Vec v, Vec *vout)
{
  PetscInt       i;
  VecType        vtype;
  PetscContainer container;
  Mat_MultiShift mctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (v) PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscAssertPointer(vout, 3);
  MatCheckMultiShift(A);

  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatNestSetVecType(A, VECNEST));
  PetscCall(MatCreateVecs(A, NULL, vout));
  PetscCall(MatNestSetVecType(A, vtype));
  if (v) {
    PetscCall(PetscObjectQuery((PetscObject)A, "MatMultiShift", (PetscObject *)&container));
    PetscCall(PetscContainerGetPointer(container, (void **)&mctx));
    for (i = 0; i < mctx->nshift; i++) {
      PetscCall(VecNestSetSubVec(*vout, i, v));
      if (!PetscDefined(USE_COMPLEX) && mctx->cmplx[i]) i++; // complex shift, leave a zero block since b is real
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
