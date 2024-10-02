/*
Context for Bounded Regularized Gauss-Newton algorithm.
Extended with L1-regularizer with a linear transformation matrix D:
0.5*||Ax-b||^2 + lambda*||D*x||_1
When D is an identity matrix, we have the classic lasso, aka basis pursuit denoising in compressive sensing problem.
*/

#pragma once

#include <petsc/private/taoimpl.h>

typedef struct {
  Mat                       Hreg, D; /* Hessian, Hessian for regulization part, and Dictionary matrix have size N*N, and K*N respectively. (Jacobian M*N not used here) */
  Vec                       y;       /* x, r=J*x, and y=D*x have size N, M, and K respectively. */
  Vec                       damping; /* a copy of the Levenberg-Marquardt diagonal, needed only for legacy support of TaoBRGNGetDampingVector() */
  Tao                       subsolver, parent;
  PetscReal                 epsilon, fc_old;                              /* lambda is regularizer weight for both L2-norm Gaussian-Newton and L1-norm, ||x||_1 is approximated with sum(sqrt(x.^2+epsilon^2)-epsilon)*/
  PetscReal                 downhill_lambda_change, uphill_lambda_change; /* With the lm regularizer lambda diag(J^T J),
                                                                 lambda = downhill_lambda_change * lambda on steps that decrease the objective.
                                                                 lambda = uphill_lambda_change * lambda on steps that increase the objective. */
  TaoBRGNRegularizationType reg_type;
  PetscBool                 mat_explicit;
  TaoTerm                   orig_callbacks;
} TAO_BRGN;
