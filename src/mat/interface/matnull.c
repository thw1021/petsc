/*
    Routines to project vectors out of null spaces.
*/

#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>

PetscClassId MAT_NULLSPACE_CLASSID;

/*@C
  MatNullSpaceSetFunction - set a function that removes a null space from a vector
  out of null spaces.

  Logically Collective

  Input Parameters:
+ sp  - the `MatNullSpace` null space object
. rem - the function that removes the null space
- ctx - context for the remove function

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceDestroy()`, `MatNullSpaceRemove()`, `MatSetNullSpace()`, `MatNullSpaceCreate()`
@*/
PetscErrorCode MatNullSpaceSetFunction(MatNullSpace sp, PetscErrorCode (*rem)(MatNullSpace, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  sp->remove = rem;
  sp->rmctx  = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatNullSpaceGetVecs - get the vectors defining the null space

  Not Collective

  Input Parameter:
. sp - null space object

  Output Parameters:
+ has_const - `PETSC_TRUE` if the null space contains the constant vector, otherwise `PETSC_FALSE`
. n         - number of vectors (excluding constant vector) in the null space
- vecs      - returns array of length `n` containing the orthonormal vectors that span the null space (excluding the constant vector), `NULL` if `n` is 0

  Level: developer

  Note:
  These vectors and the array are owned by the `MatNullSpace` and should not be destroyed or freeded by the caller

  Fortran Note:
  One must pass in an array `vecs` that is large enough to hold all of the requested vectors

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatGetNullSpace()`, `MatGetNearNullSpace()`
@*/
PetscErrorCode MatNullSpaceGetVecs(MatNullSpace sp, PetscBool *has_const, PetscInt *n, const Vec *vecs[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  if (has_const) *has_const = sp->has_cnst;
  if (n) *n = sp->n;
  if (vecs) *vecs = sp->vecs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceCreateRigidBody - create rigid body modes from coordinates

  Collective

  Input Parameter:
. coords - block of coordinates of each node, must have block size set

  Output Parameter:
. sp - the null space

  Level: advanced

  Notes:
  If you are solving an elasticity problem you should likely use this, in conjunction with `MatSetNearNullSpace()`, to provide information that
  the `PCGAMG` preconditioner can use to construct a much more efficient preconditioner.

  If you are solving an elasticity problem with pure Neumann boundary conditions you can use this in conjunction with `MatSetNullSpace()` to
  provide this information to the linear solver so it can handle the null space appropriately in the linear solution.

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatSetNearNullSpace()`, `MatSetNullSpace()`, `PCGAMG`
@*/
PetscErrorCode MatNullSpaceCreateRigidBody(Vec coords, MatNullSpace *sp)
{
  const PetscScalar *x;
  PetscScalar       *v[6], dots[5];
  Vec                vec[6];
  PetscInt           n, N, dim, nmodes, i, j;
  PetscReal          sN;

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(coords, &dim));
  PetscCall(VecGetLocalSize(coords, &n));
  PetscCall(VecGetSize(coords, &N));
  n /= dim;
  N /= dim;
  sN = 1. / PetscSqrtReal((PetscReal)N);
  switch (dim) {
  case 1:
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)coords), PETSC_TRUE, 0, NULL, sp));
    break;
  case 2:
  case 3:
    nmodes = (dim == 2) ? 3 : 6;
    PetscCall(VecCreate(PetscObjectComm((PetscObject)coords), &vec[0]));
    PetscCall(VecSetSizes(vec[0], dim * n, dim * N));
    PetscCall(VecSetBlockSize(vec[0], dim));
    PetscCall(VecSetUp(vec[0]));
    for (i = 1; i < nmodes; i++) PetscCall(VecDuplicate(vec[0], &vec[i]));
    for (i = 0; i < nmodes; i++) PetscCall(VecGetArray(vec[i], &v[i]));
    PetscCall(VecGetArrayRead(coords, &x));
    for (i = 0; i < n; i++) {
      if (dim == 2) {
        v[0][i * 2 + 0] = sN;
        v[0][i * 2 + 1] = 0.;
        v[1][i * 2 + 0] = 0.;
        v[1][i * 2 + 1] = sN;
        /* Rotations */
        v[2][i * 2 + 0] = -x[i * 2 + 1];
        v[2][i * 2 + 1] = x[i * 2 + 0];
      } else {
        v[0][i * 3 + 0] = sN;
        v[0][i * 3 + 1] = 0.;
        v[0][i * 3 + 2] = 0.;
        v[1][i * 3 + 0] = 0.;
        v[1][i * 3 + 1] = sN;
        v[1][i * 3 + 2] = 0.;
        v[2][i * 3 + 0] = 0.;
        v[2][i * 3 + 1] = 0.;
        v[2][i * 3 + 2] = sN;

        v[3][i * 3 + 0] = x[i * 3 + 1];
        v[3][i * 3 + 1] = -x[i * 3 + 0];
        v[3][i * 3 + 2] = 0.;
        v[4][i * 3 + 0] = 0.;
        v[4][i * 3 + 1] = -x[i * 3 + 2];
        v[4][i * 3 + 2] = x[i * 3 + 1];
        v[5][i * 3 + 0] = x[i * 3 + 2];
        v[5][i * 3 + 1] = 0.;
        v[5][i * 3 + 2] = -x[i * 3 + 0];
      }
    }
    for (i = 0; i < nmodes; i++) PetscCall(VecRestoreArray(vec[i], &v[i]));
    PetscCall(VecRestoreArrayRead(coords, &x));
    for (i = dim; i < nmodes; i++) {
      /* Orthonormalize vec[i] against vec[0:i-1] */
      PetscCall(VecMDot(vec[i], i, vec, dots));
      for (j = 0; j < i; j++) dots[j] *= -1.;
      PetscCall(VecMAXPY(vec[i], i, dots, vec));
      PetscCall(VecNormalize(vec[i], NULL));
    }
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)coords), PETSC_FALSE, nmodes, vec, sp));
    for (i = 0; i < nmodes; i++) PetscCall(VecDestroy(&vec[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceView - Visualizes a null space object.

  Collective; No Fortran Support

  Input Parameters:
+ sp     - the null space
- viewer - visualization context

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `PetscViewer`, `MatNullSpaceCreate()`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode MatNullSpaceView(MatNullSpace sp, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sp), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(sp, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscInt          i;
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sp, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Contains %" PetscInt_FMT " vector%s%s\n", sp->n, sp->n == 1 ? "" : "s", sp->has_cnst ? " and the constant" : ""));
    if (sp->remove) PetscCall(PetscViewerASCIIPrintf(viewer, "Has user-provided removal function\n"));
    if (!(format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL)) {
      for (i = 0; i < sp->n; i++) PetscCall(VecView(sp->vecs[i], viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceCreate - Creates a `MatNullSpace` data structure used to project vectors out of null spaces.

  Collective

  Input Parameters:
+ comm     - the MPI communicator associated with the object
. has_cnst - `PETSC_TRUE` if the null space contains the constant vector; otherwise `PETSC_FALSE`
. n        - number of vectors (excluding constant vector) in null space
- vecs     - the vectors that span the null space (excluding the constant vector);
             these vectors must be orthonormal. These vectors are NOT copied, so do not change them
             after this call. You should free the array that you pass in and destroy the vectors (this will reduce the reference count
             for them by one).

  Output Parameter:
. SP - the null space context

  Level: advanced

  Notes:
  See `MatNullSpaceSetFunction()` as an alternative way of providing the null space information instead of providing the vectors.

  See `MatNullSpaceCreateFromSpanningVecs()` for creating a nullspace from an arbitrary set of vectors.

  If has_cnst is `PETSC_TRUE` you do not need to pass a constant vector in as a fourth argument to this routine, nor do you
  need to pass in a function that eliminates the constant function into `MatNullSpaceSetFunction()`.

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceDestroy()`, `MatNullSpaceRemove()`, `MatSetNullSpace()`, `MatNullSpaceSetFunction()`, `MatNullSpaceCreateFromSpanningVecs()`
@*/
PetscErrorCode MatNullSpaceCreate(MPI_Comm comm, PetscBool has_cnst, PetscInt n, const Vec vecs[], MatNullSpace *SP)
{
  MatNullSpace sp;
  PetscInt     i;

  PetscFunctionBegin;
  PetscCheck(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vectors (given %" PetscInt_FMT ") cannot be negative", n);
  if (n) PetscAssertPointer(vecs, 4);
  for (i = 0; i < n; i++) PetscValidHeaderSpecific(vecs[i], VEC_CLASSID, 4);
  PetscAssertPointer(SP, 5);
  if (n) {
    for (i = 0; i < n; i++) {
      /* prevent the user from changes values in the vector */
      PetscCall(VecLockReadPush(vecs[i]));
    }
  }
  if (PetscUnlikelyDebug(n)) {
    PetscScalar *dots;
    for (i = 0; i < n; i++) {
      PetscReal norm;
      PetscCall(VecNorm(vecs[i], NORM_2, &norm));
      PetscCheck(PetscAbsReal(norm - 1) <= PETSC_SQRT_MACHINE_EPSILON, PetscObjectComm((PetscObject)vecs[i]), PETSC_ERR_ARG_WRONG, "Vector %" PetscInt_FMT " must have 2-norm of 1.0, it is %g", i, (double)norm);
    }
    if (has_cnst) {
      for (i = 0; i < n; i++) {
        PetscScalar sum;
        PetscCall(VecSum(vecs[i], &sum));
        PetscCheck(PetscAbsScalar(sum) <= PETSC_SQRT_MACHINE_EPSILON, PetscObjectComm((PetscObject)vecs[i]), PETSC_ERR_ARG_WRONG, "Vector %" PetscInt_FMT " must be orthogonal to constant vector, inner product is %g", i, (double)PetscAbsScalar(sum));
      }
    }
    PetscCall(PetscMalloc1(n - 1, &dots));
    for (i = 0; i < n - 1; i++) {
      PetscInt j;
      PetscCall(VecMDot(vecs[i], n - i - 1, vecs + i + 1, dots));
      for (j = 0; j < n - i - 1; j++) {
        PetscCheck(PetscAbsScalar(dots[j]) <= PETSC_SQRT_MACHINE_EPSILON, PetscObjectComm((PetscObject)vecs[i]), PETSC_ERR_ARG_WRONG, "Vector %" PetscInt_FMT " must be orthogonal to vector %" PetscInt_FMT ", inner product is %g", i, i + j + 1, (double)PetscAbsScalar(dots[j]));
      }
    }
    PetscCall(PetscFree(dots));
  }

  *SP = NULL;
  PetscCall(MatInitializePackage());

  PetscCall(PetscHeaderCreate(sp, MAT_NULLSPACE_CLASSID, "MatNullSpace", "Null space", "Mat", comm, MatNullSpaceDestroy, MatNullSpaceView));

  sp->has_cnst        = has_cnst;
  sp->n               = n;
  sp->n_spanning_vecs = 0;
  sp->vecs            = NULL;
  sp->spanning_vecs   = NULL;
  sp->alpha           = NULL;
  sp->remove          = NULL;
  sp->rmctx           = NULL;

  if (n) {
    PetscCall(PetscMalloc1(n, &sp->vecs));
    PetscCall(PetscMalloc1(n, &sp->alpha));
    for (i = 0; i < n; i++) {
      PetscCall(PetscObjectReference((PetscObject)vecs[i]));
      sp->vecs[i] = vecs[i];
    }
  }

  *SP = sp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceSetSpanningVecs - Attach a set of vectors that span the
  nullspace.  Unlike the orthonormal vectors used to create the space in
  `MatNullSpaceCreate()`, these vectors do not need to be orthonormal or even
  linearly independent.

  Collective

  Input Parameters:
+ sp              - the `MatNullSpace` context
. n_spanning_vecs - the number of vectors in `spanning_vecs`
- spanning_vecs   - a set of vectors that span the null space.

  Level: advanced

  Notes:
  This method is useful if the $n$-dimensional nullspace is defined by some vectors $\{v_0, \dots, v_{m-1}\}$.
  The vectors returned by `MatNullSpaceGetVecs()` describe an orthonormal basis of the nullspace, and so do not
  retain information about $\{v\}$.  The spanning vectors used internally by a
  `MatNullSpace` and do not affect `MatNullSpaceRemove()`.  They can accessed by `MatNullSpaceGetSpanningVecs()`.

  The array `spanning_vecs` is treated the same as the array of vectors passed to `MatNullSpaceCreate()`: the vectors
  themselves are referenced, but the array is not.

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatNullSpaceGetVecs()`, `MatNullSpaceGetSpanningVecs()`.
@*/
PetscErrorCode MatNullSpaceSetSpanningVecs(MatNullSpace sp, PetscInt n_spanning_vecs, const Vec spanning_vecs[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  for (PetscInt i = 0; i < n_spanning_vecs; i++) PetscValidHeaderSpecific(spanning_vecs[i], VEC_CLASSID, 3);
  for (PetscInt i = 0; i < n_spanning_vecs; i++) PetscCall(VecLockReadPush(spanning_vecs[i]));
  if (n_spanning_vecs) {
    PetscCall(PetscMalloc1(n_spanning_vecs, &sp->spanning_vecs));
    for (PetscInt i = 0; i < n_spanning_vecs; i++) {
      PetscCall(PetscObjectReference((PetscObject)spanning_vecs[i]));
      sp->spanning_vecs[i] = spanning_vecs[i];
    }
  }
  sp->n_spanning_vecs = n_spanning_vecs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatNullSpaceGetSpanningVecs - Get the spanning vectors of the nullspace set with `MatNullSpaceSetSpanningVecs()`.

  Not collective

  Input Parameter:
. sp - the `MatNullSpace` context

  Output Parameters:
+ n_spanning_vecs - the number of vectors in `spanning_vecs`
- spanning_vecs   - vectors that span the nullspace set in `MatNullSpaceSetSpanningVecs()`

  Level: advanced

  Note:
  Use `MatNullSpaceGetVecs()` to get an orthonormal basis of the nullspace.  There may be more spanning vectors
  than the dimension of the nullspace, they may not be normalized, and they may be linearly dependent.

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatNullSpaceGetVecs()`, `MatNullSpaceGetSpanningVecs()`.
@*/
PetscErrorCode MatNullSpaceGetSpanningVecs(MatNullSpace sp, PetscInt *n_spanning_vecs, const Vec **spanning_vecs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  if (n_spanning_vecs) *n_spanning_vecs = sp->n_spanning_vecs;
  if (spanning_vecs) *spanning_vecs = sp->spanning_vecs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceCreateFromSpanningVecs - Create a nullspace from an arbitrary set
  of spanning vectors (as opposed to an orthonormal basis, which is required by
  `MatNullSpaceCreate()`).

  Collective

  Input Parameters:
+ comm     - the MPI communicator associated with the object
. n        - number of vectors in null space
- vecs     - the vectors that span the null space. These vectors are NOT
             copied, so do not change them after this call. You should free the
             array that you pass in and destroy the vectors (this will reduce
             the reference count for them by one).

  Output Parameter:
. sp - the null space context

  Level: advanced

  Options Database Keys:
+ -mat_nullspace_spanning_vecs_atol - the absolute tolerance for singular values considered numerically zero when determining an orthonormal basis for the span
+ -mat_nullspace_spanning_vecs_rtol - the relative tolerance (relative to the largest singular value) for singular values considered numerically zero when determining an orthonormal basis for the span

  Notes:
  An orthonormal basis (possibly containing fewer vectors, if the spanning
  vectors are linearly dependent) is internally computed, which is accessible
  from `MatNullSpaceGetVecs()`.  The original spanning vecs are avaiable from
  `MatNullSpaceGetSpanningVecs()`.

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceDestroy()`,
          `MatNullSpaceRemove()`, `MatSetNullSpace()`, `MatNullSpaceSetFunction()`,
          `MatNullSpaceCreate()`, `MatNullSpaceSetSpanningVecs()`,
          `MatNullSpaceGetSpanningVecs()`
@*/
PetscErrorCode MatNullSpaceCreateFromSpanningVecs(MPI_Comm comm, PetscInt n, const Vec vecs[], MatNullSpace *sp)
{
  PetscInt    m, M, n_cols_self, r, r_orig;
  PetscMPIInt rank;
  VecType     vec_type;
  Mat         B, U;
  Vec         S;
  Vec        *ortho_vecs;

  PetscFunctionBegin;
  PetscCheck(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vectors (given %" PetscInt_FMT ") cannot be negative", n);
  if (n == 0) {
    PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, n, vecs, sp));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssertPointer(vecs, 3);
  for (PetscInt i = 0; i < n; i++) PetscValidHeaderSpecific(vecs[i], VEC_CLASSID, 3);
  PetscAssertPointer(sp, 4);
  PetscCall(VecGetSize(vecs[0], &M));
  PetscCall(VecGetLocalSize(vecs[0], &m));
  PetscCall(VecGetType(vecs[0], &vec_type));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  n_cols_self = (rank == 0) ? n : 0;
  PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, n_cols_self, M, n, PETSC_DEFAULT, NULL, &B));
  for (PetscInt i = 0; i < n; i++) {
    Vec b_i;
    PetscCall(MatDenseGetColumnVecWrite(B, i, &b_i));
    PetscCall(VecCopy(vecs[i], b_i));
    PetscCall(MatDenseRestoreColumnVecWrite(B, i, &b_i));
  }
  PetscCall(MatDenseTallSkinnySVD(B, MAT_INITIAL_MATRIX, &U, &S, NULL));
  PetscCall(MatDestroy(&B));
  PetscCall(MatGetSize(U, NULL, &r));
  r_orig = r;
  if (rank == 0 && r > 0) {
    PetscReal          a_tol = PetscSqrtReal((PetscReal)M) * PETSC_MACHINE_EPSILON;
    PetscReal          r_tol = PETSC_MACHINE_EPSILON;
    PetscReal          tol;
    PetscOptions       options;
    const PetscScalar *_S;

    PetscCall(PetscObjectGetOptions((PetscObject)vecs[0], &options));
    PetscCall(PetscOptionsGetReal(options, NULL, "-mat_nullspace_spanning_vecs_atol", &a_tol, NULL));
    PetscCall(PetscOptionsGetReal(options, NULL, "-mat_nullspace_spanning_vecs_rtol", &r_tol, NULL));
    PetscCall(VecGetArrayRead(S, &_S));
    tol = PetscMax(PetscRealPart(_S[0]) * r_tol, a_tol);
    for (PetscInt i = 0; i < r_orig; i++) {
      if (PetscRealPart(_S[i]) <= tol) r = PetscMin(r, i);
    }
    PetscCall(VecRestoreArrayRead(S, &_S));
  }
  PetscCallMPI(MPI_Bcast(&r, 1, MPIU_INT, 0, comm));
  PetscCall(PetscMalloc1(r, &ortho_vecs));
  for (PetscInt i = 0; i < r; i++) {
    Vec u_i;

    PetscCall(MatDenseGetColumnVecRead(U, i, &u_i));
    PetscCall(VecDuplicate(u_i, &ortho_vecs[i]));
    PetscCall(VecCopy(u_i, ortho_vecs[i]));
    PetscCall(MatDenseRestoreColumnVecRead(U, i, &u_i));
  }
  PetscCall(VecDestroy(&S));
  PetscCall(MatDestroy(&U));
  PetscCall(MatNullSpaceCreate(comm, PETSC_FALSE, r, ortho_vecs, sp));
  for (PetscInt i = 0; i < r; i++) PetscCall(VecDestroy(&ortho_vecs[i]));
  PetscCall(PetscFree(ortho_vecs));
  PetscCall(MatNullSpaceSetSpanningVecs(*sp, n, vecs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatNullSpaceSetComputeSpanningVecsAdjoint - Set a function that computes the adjoint of the spanning vectors (`MatNullSpaceSetSpanningVecs()`) with respect to a solution vector.

  Logically collective

  Input Parameters:
+ sp         - a `MatNullSpace` that has had spanning vectors set with `MatNullSpaceSetSpanningVecs()` or was created with `MatNullSpaceCreateFromSpanningVecs()`
. adjoint_fn - a pointer to a function that will compute the adjoint of the spanning vectors with respect to a solution variable, which will be called by `MatNullSpaceComputeSpanningVecsAdjoint()`
- ctx        - the user-defined context for `adjoint_fn`

  Calling sequence of `adjoint_fn`:
+ sp       - the `MatNullSpace`
. x        - the current solution vector
. g        - a variation vector that is the same shape as the spanning vectors in `MatNullSpaceGetSpanningVecs()`
. g_dot_dx - an array of of vectors. `g_dot_dx[i]` should contain the value $\partial_x (g^T s_i(x))$, where $s_i(x)$ is the $i$th spanning vector, treated as a vector-valued function of $x$.
- ctx      - the user-defined context

  Level: developer

  Note:
  This function facilitates automatic computation of projection spaces in `SNESComputeJacobianProjection()`

.seealso: `MatNullSpace`, `MatNullSpaceSetSpanningVecs()`, `MatNullSpaceGetSpanningVecs(), `MatNullSpaceCreateFromSpanningVecs()`, `MatNullSpaceGetComputeSpanningVecsAdjoint()`, `SNESComputeJacobianProjection()`, `MatNullSpaceComputeSpanningVecs
@*/
PetscErrorCode MatNullSpaceSetComputeSpanningVecsAdjoint(MatNullSpace sp, PetscErrorCode (*adjoint_fn)(MatNullSpace sp, Vec x, Vec g, const Vec g_dot_dx[], void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  if (adjoint_fn) PetscAssertPointer(adjoint_fn, 2);
  if (ctx) PetscAssertPointer(ctx, 3);
  sp->adjoint_fn  = adjoint_fn;
  sp->adjoint_ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatNullSpaceGetComputeSpanningVecsAdjoint - Get the function that computes the adjoint of the spanning vectors (`MatNullSpaceSetSpanningVecs()`) with respect to a solution vector.

  Logically collective

  Input Parameter:
. sp         - a `MatNullSpace`

  Output Parameters:
+ adjoint_fn - if not `NULL`, set to a pointer to the function that computes the adjoint of the spanning vectors with respect to a solution variable.  The output is `NULL` if `MatNullSpaceSetComputeSpanningVecsAdjoint()` was not set
- ctx        - if not `NULL`, set to the user-defined context for `adjoint_fn`

  Calling sequence of `adjoint_fn`:
+ sp       - the `MatNullSpace`
. x        - the current solution vector
. g        - a variation vector that is the same shape as the spanning vectors in `MatNullSpaceGetSpanningVecs()`
. g_dot_dx - an array of of vectors. `g_dot_dx[i]` should contain the value $\partial_x (g^T s_i(x))$, where $s_i(x)$ is the $i$th spanning vector, treated as a vector-valued function of $x$.
- ctx      - the user-defined context

  Level: developer

.seealso: `MatNullSpace`, `MatNullSpaceSetSpanningVecs()`, `MatNullSpaceGetSpanningVecs(), `MatNullSpaceGetSpanningVecs()`, `MatNullSpaceCreateFromSpanningVecs()`, `MatNullSpaceSetComputeSpanningVecsAdjoint()`, `MatNullSpaceComputeSpanningVecsAdjoint()`
@*/
PetscErrorCode MatNullSpaceGetComputeSpanningVecsAdjoint(MatNullSpace sp, PetscErrorCode (**adjoint_fn)(MatNullSpace sp, Vec x, Vec g, const Vec g_dot_dx[], void *ctx), void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  if (adjoint_fn) PetscAssertPointer(adjoint_fn, 2);
  if (ctx) PetscAssertPointer(ctx, 3);
  if (adjoint_fn) *adjoint_fn = sp->adjoint_fn;
  if (ctx) *ctx = sp->adjoint_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceComputeSpanningVecsAdjoint - Compute the adjoint of the spanning vectors of a nullspace (`MatNullSpaceSetSpanningVecs()`) with respect to a solution variable

  Collective

  Input Parameters:
+ sp - a `MatNullSpace` that has had spanning vectors defined (using `MatNullSpaceSetSpanningVecs()` or `MatNullSpaceCreateFromSpanningVecs()`) and the action of the adjoint defined with `MatNullSpaceSetComputeSpanningVecsAdjoint()`.
. x  - a solution vector
- g  - a variation vector that is the same shape as the spanning vectors in `MatNullSpaceGetSpanningVecs()`

  Output Parameter:
. g_dot_dx - an array of of vectors. `g_dot_dx[i]` should contain the value $\partial_x (g^T s_i(x))$, where $s_i(x)$ is the $i$th spanning vector, treated as a vector-valued function of $x$.

  Level: developer

.seealso: `MatNullSpace`, `MatNullSpaceSetSpanningVecs()`, `MatNullSpaceGetSpanningVecs(), `MatNullSpaceGetSpanningVecs()`, `MatNulllSpaceCreateFromSpanningVecs()`, `MatNullSpaceComputeSpanningVecsAdjoint()`
@*/
PetscErrorCode MatNullSpaceComputeSpanningVecsAdjoint(MatNullSpace sp, Vec x, Vec g, const Vec g_dot_dx[])
{
  PetscInt n_spanning_vecs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCall(MatNullSpaceGetSpanningVecs(sp, &n_spanning_vecs, NULL));
  PetscAssertPointer(g_dot_dx, 4);
  if (PetscDefined(USE_DEBUG)) {
    for (PetscInt i = 0; i < n_spanning_vecs; i++) { PetscValidHeaderSpecific(g_dot_dx[i], VEC_CLASSID, 4); }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceDestroy - Destroys a data structure used to project vectors out of null spaces.

  Collective

  Input Parameter:
. sp - the null space context to be destroyed

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatNullSpaceRemove()`, `MatNullSpaceSetFunction()`
@*/
PetscErrorCode MatNullSpaceDestroy(MatNullSpace *sp)
{
  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*sp, MAT_NULLSPACE_CLASSID, 1);
  if (--((PetscObject)*sp)->refct > 0) {
    *sp = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  for (PetscInt i = 0; i < (*sp)->n; i++) PetscCall(VecLockReadPop((*sp)->vecs[i]));
  for (PetscInt i = 0; i < (*sp)->n_spanning_vecs; i++) PetscCall(VecLockReadPop((*sp)->spanning_vecs[i]));

  PetscCall(VecDestroyVecs((*sp)->n, &(*sp)->vecs));
  PetscCall(VecDestroyVecs((*sp)->n_spanning_vecs, &(*sp)->spanning_vecs));
  PetscCall(PetscFree((*sp)->alpha));
  PetscCall(PetscHeaderDestroy(sp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceRemove - Removes all the components of a null space from a vector.

  Collective

  Input Parameters:
+ sp  - the null space context (if this is `NULL` then no null space is removed)
- vec - the vector from which the null space is to be removed

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatNullSpaceDestroy()`, `MatNullSpaceSetFunction()`
@*/
PetscErrorCode MatNullSpaceRemove(MatNullSpace sp, Vec vec)
{
  PetscScalar sum;
  PetscInt    i, N;

  PetscFunctionBegin;
  if (!sp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 2);

  if (sp->has_cnst) {
    PetscCall(VecGetSize(vec, &N));
    if (N > 0) {
      PetscCall(VecSum(vec, &sum));
      sum = sum / ((PetscScalar)(-1.0 * N));
      PetscCall(VecShift(vec, sum));
    }
  }

  if (sp->n) {
    PetscCall(VecMDot(vec, sp->n, sp->vecs, sp->alpha));
    for (i = 0; i < sp->n; i++) sp->alpha[i] = -sp->alpha[i];
    PetscCall(VecMAXPY(vec, sp->n, sp->alpha, sp->vecs));
  }

  if (sp->remove) PetscCall((*sp->remove)(sp, vec, sp->rmctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNullSpaceTest  - Tests if the claimed null space is really a null space of a matrix

  Collective

  Input Parameters:
+ sp  - the null space context
- mat - the matrix

  Output Parameter:
. isNull - `PETSC_TRUE` if the nullspace is valid for this matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatNullSpace`, `MatNullSpaceCreate()`, `MatNullSpaceDestroy()`, `MatNullSpaceSetFunction()`
@*/
PetscErrorCode MatNullSpaceTest(MatNullSpace sp, Mat mat, PetscBool *isNull)
{
  PetscScalar sum;
  PetscReal   nrm, tol = 10. * PETSC_SQRT_MACHINE_EPSILON;
  PetscInt    j, n, N;
  Vec         l, r;
  PetscBool   flg1 = PETSC_FALSE, flg2 = PETSC_FALSE, consistent = PETSC_TRUE;
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, MAT_NULLSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 2);
  n = sp->n;
  PetscCall(PetscOptionsGetBool(((PetscObject)sp)->options, ((PetscObject)mat)->prefix, "-mat_null_space_test_view", &flg1, NULL));
  PetscCall(PetscOptionsGetBool(((PetscObject)sp)->options, ((PetscObject)mat)->prefix, "-mat_null_space_test_view_draw", &flg2, NULL));

  if (n) {
    PetscCall(VecDuplicate(sp->vecs[0], &l));
  } else {
    PetscCall(MatCreateVecs(mat, &l, NULL));
  }

  PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sp), &viewer));
  if (sp->has_cnst) {
    PetscCall(VecDuplicate(l, &r));
    PetscCall(VecGetSize(l, &N));
    sum = 1.0 / PetscSqrtReal(N);
    PetscCall(VecSet(l, sum));
    PetscCall(MatMult(mat, l, r));
    PetscCall(VecNorm(r, NORM_2, &nrm));
    if (nrm >= tol) consistent = PETSC_FALSE;
    if (flg1) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp), "Constants are %s null vector ", consistent ? "likely" : "unlikely"));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp), "|| A * 1/sqrt(N) || = %g\n", (double)nrm));
    }
    if (!consistent && (flg1 || flg2)) PetscCall(VecView(r, viewer));
    PetscCall(VecDestroy(&r));
  }

  for (j = 0; j < n; j++) {
    PetscUseTypeMethod(mat, mult, sp->vecs[j], l);
    PetscCall(VecNorm(l, NORM_2, &nrm));
    if (nrm >= tol) consistent = PETSC_FALSE;
    if (flg1) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp), "Null vector %" PetscInt_FMT " is %s null vector ", j, consistent ? "likely" : "unlikely"));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sp), "|| A * v[%" PetscInt_FMT "] || = %g\n", j, (double)nrm));
    }
    if (!consistent && (flg1 || flg2)) PetscCall(VecView(l, viewer));
  }

  PetscCheck(!sp->remove, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Cannot test a null space provided as a function with MatNullSpaceSetFunction()");
  PetscCall(VecDestroy(&l));
  if (isNull) *isNull = consistent;
  PetscFunctionReturn(PETSC_SUCCESS);
}
