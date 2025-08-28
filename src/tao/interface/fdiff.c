#include <petsctao.h> /*I  "petsctao.h"  I*/
#include <petsc/private/taoimpl.h>
#include <petscsnes.h>
#include <petscdmshell.h>

/*@C
  TaoDefaultComputeGradient - computes the gradient using finite differences.

  Collective

  Input Parameters:
+ tao   - the Tao context
. Xin   - compute gradient at this point
- dummy - not used

  Output Parameter:
. G - Gradient Vector

  Options Database Key:
+ -tao_fd_gradient      - activates TaoDefaultComputeGradient()
- -tao_fd_delta <delta> - change in X used to calculate finite differences

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized
  to take advantage of sparsity in the problem.  Although
  not recommended for general use
  in large-scale applications, it can be useful in checking the
  correctness of a user-provided gradient using the command-line option `-tao_test_gradient`
  This finite difference gradient evaluation can be set using the routine `TaoSetGradient()` or by using the command line option -tao_fd_gradient

.seealso: `Tao`, `TaoSetGradient()`, `TaoTermComputeGradientFD()`
@*/
PetscErrorCode TaoDefaultComputeGradient(Tao tao, Vec Xin, Vec G, void *dummy)
{
  PetscBool flg;
  PetscReal h;

  PetscFunctionBegin;
  PetscCall(TaoTermGetFDDelta(tao->objective_term.term, &h));
  PetscCall(PetscOptionsGetReal(((PetscObject)tao)->options, ((PetscObject)tao)->prefix, "-tao_fd_delta", &h, &flg));
  if (flg) PetscCall(TaoTermSetFDDelta(tao->objective_term.term, h));
  PetscCall(TaoTermComputeGradientUseFDPush(tao->objective_term.term));
  PetscCall(TaoMappedTermComputeGradient(&tao->objective_term, Xin, tao->objective_parameters, INSERT_VALUES, G));
  PetscCall(TaoTermComputeGradientUseFDPop(tao->objective_term.term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoDefaultComputeHessian - Computes the Hessian using finite differences.

  Collective

  Input Parameters:
+ tao   - the Tao context
. V     - compute Hessian at this point
- dummy - not used

  Output Parameters:
+ H - Hessian matrix (not altered in this routine)
- B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

  Options Database Key:
. -tao_fd_hessian - activates TaoDefaultComputeHessian()

  Level: advanced

  Notes:
  This routine is slow and expensive, and is not optimized
  to take advantage of sparsity in the problem.  Although
  it is not recommended for general use
  in large-scale applications, It can be useful in checking the
  correctness of a user-provided Hessian.

.seealso: `Tao`, `TaoSetHessian()`, `TaoDefaultComputeHessianColor()`, `SNESComputeJacobianDefault()`, `TaoSetGradient()`, `TaoDefaultComputeGradient()`
@*/
PetscErrorCode TaoDefaultComputeHessian(Tao tao, Vec V, Mat H, Mat B, void *dummy)
{
  PetscFunctionBegin;
  PetscCall(TaoTermComputeHessianUseFDPush(tao->objective_term.term));
  PetscCall(TaoMappedTermComputeHessian(&tao->objective_term, V, tao->objective_parameters, INSERT_VALUES, NULL, B ? B : H));
  PetscCall(TaoTermComputeHessianUseFDPop(tao->objective_term.term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoDefaultComputeHessianColor - Computes the Hessian using colored finite differences.

  Collective

  Input Parameters:
+ tao - the Tao context
. V   - compute Hessian at this point
- ctx - the color object of type `MatFDColoring`

  Output Parameters:
+ H - Hessian matrix (not altered in this routine)
- B - newly computed Hessian matrix to use with preconditioner (generally the same as H)

  Level: advanced

.seealso: `Tao`, `MatColoring`, `TaoSetHessian()`, `TaoDefaultComputeHessian()`, `SNESComputeJacobianDefaultColor()`, `TaoSetGradient()`
@*/
PetscErrorCode TaoDefaultComputeHessianColor(Tao tao, Vec V, Mat H, Mat B, void *ctx)
{
  MatFDColoring coloring = (MatFDColoring)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coloring, MAT_FDCOLORING_CLASSID, 5);
  B = B ? B : H;
  PetscCall(PetscObjectCompose((PetscObject)B, "__TaoTermHessianMatFDColoring", (PetscObject)coloring));
  PetscCall(TaoTermComputeHessianUseFDPush(tao->objective_term.term));
  PetscCall(TaoMappedTermComputeHessian(&tao->objective_term, V, tao->objective_parameters, INSERT_VALUES, NULL, B));
  PetscCall(TaoTermComputeHessianUseFDPop(tao->objective_term.term));
  PetscCall(PetscObjectCompose((PetscObject)B, "__TaoTermHessianMatFDColoring", (PetscObject)NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoDefaultComputeHessianMFFD - Computes the Hessian using `MATMFFD`.

  Collective

  Input Parameters:
+ tao - the Tao context
. X   - compute Hessian at this point
- ctx - ignored

  Output Parameters:
+ H - Hessian matrix of type `MATMFFD`
- B - should be NULL or equal to H

  Level: advanced

  Note:
  This can be passed to `TaoSetHessian()` to use `MATMFFD` for approximate Hessian-vector products.  The matrix `H` can originate from
  `MatCreateMFFD()` or from `TaoTermCreateHessianMFFD()`

.seealso: `Tao`, `MATMFFD`, `MatCreateMFFD()`, `TaoTermCreateMFFD()`
@*/
PetscErrorCode TaoDefaultComputeHessianMFFD(Tao tao, Vec X, Mat H, Mat B, void *ctx)
{
  PetscFunctionBegin;
  PetscCheck(!B || B == H, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Preconditioning Hessian matrix");
  PetscCall(TaoTermUpdateHessianMFFD(tao->objective_term.term, H, X, tao->objective_parameters));
  PetscFunctionReturn(PETSC_SUCCESS);
}
