#pragma once

/* Context for an (nonmonotone) linesearch for proximal
 splitting algorithms.

 Currently, PS supports TAOFB and TAOCV.

 It should be noted that general Armijo-type linesearch is looking for
 search direction, which is x + step*d. Howvever, for proxiaml-type algorithms,
 the search direction is NOT a linear combination of two vectors, but rather
 an output of a proximal operator, which is often called a ``solution map".
 Therefore, TAOLINESEARCHPS requires a proximal operator inside the linesearch context.

 For TaoFB:
 Given a function f, the current iterate x_k, and a next iterate
 x_{k+1}, find the biggest stepsize tau_k such that

    f(x_{k+1}) < f(x_k) + <grad f(x_k),S_k> + (1/(2*tau_k))*|S_k|^2

 where S_k is x_{k+1} - x_k.
 The nonmonotone modification of this linesearch replaces the f(x_k) term
 with a reference value, R, and seeks to find the x_{k+1} that satisfies

    f(x_{k+1}) < R + <grad f(x_k),S_k> + (1/(2*tau_k))*|S_k|^2

 This modification does effect neither the convergence nor rate of
 convergence of an algorithm when R is chosen appropriately.
 R can be defined as max(f(x_{k-1}), f(x_{k-2}), ..., f(x_{k-min(M,k)}),
 where M is history size. The benefit of a nonmonotone
 linesearch is that local minimizers can be avoided (by allowing increase
 in function value), and typically, fewer iterations are performed in
 the main code.

 The reference value is chosen based upon some historical information
 consisting of function values for previous iterates.  The amount of
 historical information used is determined by the memory size where the
 memory is used to store the previous function values.  The memory is
 initialized 5.

 For TaoCV:

 See documentation, and citation contained in TAOCV documentation.

 References:
+ * - Armijo, "Minimization of Functions Having Lipschitz Continuous
    First-Partial Derivatives," Pacific Journal of Mathematics, volume 16,
    pages 1-3, 1966.
. * - Ferris and Lucidi, "Nonmonotone Stabilization Methods for Nonlinear
    Equations," Journal of Optimization Theory and Applications, volume 81,
    pages 53-71, 1994.
. * - Grippo, Lampariello, and Lucidi, "A Nonmonotone Line Search Technique
    for Newton's Method," SIAM Journal on Numerical Analysis, volume 23,
    pages 707-716, 1986.
. * - Grippo, Lampariello, and Lucidi, "A Class of Nonmonotone Stabilization
    Methods in Unconstrained Optimization," Numerische Mathematik, volume 59,
  pages 779-805, 1991.
- * - Latafat, Themelis, Stella, and Patrinos, "Adaptive proximal algorithms
    for convex optimization under local Lipschitz continuity of the gradient,"
    arXiv preprint arXiv:2301.04431. */
#include <petsc/private/taolinesearchimpl.h>
typedef struct {
  PetscReal *memory;

  PetscReal eta;           /* Stepsize decrease factor < 1 */
  PetscReal lastReference; /* Reference value of last iteration */

  PetscInt memorySize; /* Number of functions kept in memory */
  PetscInt current;    /* Current element for FIFO */

  PetscBool memorySetup;

  Vec x; /* Maintain reference to variable vector to check for changes */
  Vec work, work2;

  Vec dualvec_work, dualvec_test;
  /* cert = R + <gradf(x), xnew - x> + 1/2step * |xnew - x|_2^2 */
  PetscReal ref, cert, L, C, D, xi, test_step, step_new;
} TaoLineSearch_PS;
