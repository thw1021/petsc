#pragma once

#include <petscksptypes.h>
#include <petscdmdatypes.h>

/* SUBMANSEC = SNES */

/*S
  SNES - Abstract PETSc object that manages nonlinear solves

  Level: beginner

  Notes:
  The most commonly used `SNESType` is `SNESNEWTONLS` which uses Newton's method with a line search. For all the Newton based `SNES` nonlinear
  solvers, `KSP`, the PETSc abstract linear solver object, is used to (approximately) solve the required linear systems.

  See `SNESType` for a list of all the nonlinear solver algorithms provided by PETSc.

  Some of the `SNES` solvers support nonlinear preconditioners, which themselves are also `SNES` objects managed with `SNESGetNPC()`

.seealso: [](doc_nonlinsolve), [](ch_snes), `SNESCreate()`, `SNESSolve()`, `SNESSetType()`, `SNESType`, `TS`, `KSP`, `PC`, `SNESDestroy()`
S*/
typedef struct _p_SNES *SNES;

/*J
   SNESType - String with the name of a PETSc `SNES` method. These are all the nonlinear solvers that PETSc provides.

   Level: beginner

   Note:
   Use `SNESSetType()` or the options database key `-snes_type` to set the specific nonlinear solver algorithm to use with a given `SNES` object

.seealso: [](doc_nonlinsolve), [](ch_snes), `SNESSetType()`, `SNES`, `SNESCreate()`, `SNESDestroy()`, `SNESSetFromOptions()`
J*/
typedef const char *SNESType;
#define SNESNEWTONLS         "newtonls"
#define SNESNEWTONTR         "newtontr"
#define SNESNEWTONTRDC       "newtontrdc"
#define SNESPYTHON           "python"
#define SNESNRICHARDSON      "nrichardson"
#define SNESKSPONLY          "ksponly"
#define SNESKSPTRANSPOSEONLY "ksptransposeonly"
#define SNESVINEWTONRSLS     "vinewtonrsls"
#define SNESVINEWTONSSLS     "vinewtonssls"
#define SNESNGMRES           "ngmres"
#define SNESQN               "qn"
#define SNESSHELL            "shell"
#define SNESNGS              "ngs"
#define SNESNCG              "ncg"
#define SNESFAS              "fas"
#define SNESMS               "ms"
#define SNESNASM             "nasm"
#define SNESANDERSON         "anderson"
#define SNESASPIN            "aspin"
#define SNESCOMPOSITE        "composite"
#define SNESPATCH            "patch"
#define SNESNEWTONAL         "newtonal"

/*E
   SNESNewtonTRFallbackType - type of fallback in case the solution of the trust-region subproblem is outside of the radius

   Values:
+  `SNES_TR_FALLBACK_NEWTON` - use scaled Newton step
.  `SNES_TR_FALLBACK_CAUCHY` - use Cauchy direction
-  `SNES_TR_FALLBACK_DOGLEG` - use dogleg method

   Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONTR`, `SNESNEWTONTRDC`
E*/
typedef enum {
  SNES_TR_FALLBACK_NEWTON,
  SNES_TR_FALLBACK_CAUCHY,
  SNES_TR_FALLBACK_DOGLEG,
} SNESNewtonTRFallbackType;

PETSC_EXTERN const char *const SNESNewtonTRFallbackTypes[];

/*E
    SNESNewtonTRQNType - type of quasi-Newton model to use

   Values:
+  `SNES_TR_QN_NONE`      - do not use a quasi-Newton model
.  `SNES_TR_QN_SAME`      - use the same quasi-Newton model for matrix and the generation of the preconditioner
-  `SNES_TR_QN_DIFFERENT` - use different quasi-Newton models for matrix and the generation of the preconditioner

   Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONTR`
E*/
typedef enum {
  SNES_TR_QN_NONE,
  SNES_TR_QN_SAME,
  SNES_TR_QN_DIFFERENT,
} SNESNewtonTRQNType;

PETSC_EXTERN const char *const SNESNewtonTRQNTypes[];

#define SNES_CONVERGED_TR_DELTA_DEPRECATED SNES_CONVERGED_TR_DELTA PETSC_DEPRECATED_ENUM(3, 12, 0, "SNES_DIVERGED_TR_DELTA", )
#define SNES_DIVERGED_FNORM_NAN_DEPRECATED SNES_DIVERGED_FNORM_NAN PETSC_DEPRECATED_ENUM(3, 25, 0, "SNES_DIVERGED_FUNCTION_NANORINF", )
/*E
    SNESConvergedReason - reason a `SNESSolve()` was determined to have converged or diverged

   Values:
+  `SNES_CONVERGED_FNORM_ABS`         - $ ||F|| \le abstol $
.  `SNES_CONVERGED_FNORM_RELATIVE`    - $ ||F|| <= rtol*||F(x_0))|| $ where $x_0 $ is the initial guess
.  `SNES_CONVERGED_SNORM_RELATIVE`    - The 2-norm of the last step $ \le stol * ||x|| $ where $ x $ is the current solution
.  `SNES_CONVERGED_USER`              - The user has indicated convergence for an arbitrary reason
.  `SNES_DIVERGED_FUNCTION_COUNT`     - The user provided function has been called more times than the maximum set in `SNESSetTolerances()`
.  `SNES_DIVERGED_DTOL`               - The norm of the function has increased by a factor of divtol set with `SNESSetDivergenceTolerance()`
.  `SNES_DIVERGED_FUNCTION_NANORINF`  - the 2-norm of the current function evaluation is not-a-number (NaN) or infinity, (this
                                        is usually caused by a division of 0 by 0) and the solver could not recover from this (by, for example, cutting the step size)
.  `SNES_DIVERGED_OBJECTIVE_NANORINF` - the object function evaluation is not-a-number (NaN) or infinity, (this
                                        is usually caused by a division of 0 by 0) and the solver could not recover from this (by, for example, cutting the step size)
.  `SNES_DIVERGED_FUNCTION_DOMAIN`    - the function evaluation occurred outside the function's domain (function callback provided by
                                        `SNESSetFunction()` called `SNESSetObjectiveDomainError()`) and the solver could not recover from this (by, for example, cutting the step size)
.  `SNES_DIVERGED_OBJECTIVE_DOMAIN`   - the object function evaluation occurred outside the function's domain (function callback provided by
                                        `SNESSetObjective()` called `SNESSetObjectiveDomainError()`) and the solver could not recover from this (by, for example, cutting the step size)
.  `SNES_DIVERGED_JACOBIAN_DOMAIN`    - the Jacobian evaluation occurred outside the function's domain (function callback provided by
                                        `SNESSetJacobian()` called `SNESSetJacobianDomainError()`)
.  `SNES_DIVERGED_MAX_IT`             - `SNESSolve()` has reached the maximum number of iterations requested
.  `SNES_DIVERGED_LINE_SEARCH`        - The line search has failed. This only occurs for `SNES` solvers that use a line search
.  `SNES_DIVERGED_LOCAL_MIN`          - the algorithm seems to have stagnated at a local minimum that is not zero.
-  `SNES_CONVERGED_ITERATING          - this only occurs if `SNESGetConvergedReason()` is called during the `SNESSolve()`

   Level: beginner

    Notes:
   The two most common reasons for divergence are an incorrectly coded or computed Jacobian or failure or lack of convergence in the linear system
   (in this case we recommend
   testing with `-pc_type lu` to eliminate the linear solver as the cause of the problem).

   `SNES_DIVERGED_LOCAL_MIN` can only occur when using a `SNES` solver that uses a line search (`SNESLineSearch`).
   The line search wants to $ \min Q(\alpha) = 1/2 || F(x + \alpha s) ||^2_2 $  this occurs
   at $ Q'(\alpha) = s^T F'(x+\alpha s)^T F(x+\alpha s) = 0$. If $s$ is the Newton direction $ - F'(x)^(-1)F(x)$ then
   $ Q'(\alpha) = -F(x)^T F'(x)^(-1)^T F'(x+\alpha s)F(x+\alpha s)$; when $\alpha = 0$
   $Q'(0) = - ||F(x)||^2_2 $ which is always NEGATIVE if $F'(x)$ is invertible. This means the Newton
   direction is a descent direction and the line search should succeed if $\alpha $ is small enough.

   If $F'(x)$ is NOT invertible AND $F'(x)^T F(x) = 0 $ then $Q'(0) = 0 $ and the Newton direction
   is NOT a descent direction so the line search will fail. All one can do at this point
   is change the initial guess and try again.

   An alternative explanation: Newton's method can be regarded as replacing the function with
   its linear approximation and minimizing the 2-norm of that. That is $F(x+s) \approx F(x) + F'(x)s$
   so we minimize $ || F(x) + F'(x) s ||^2_2$ using Least Squares. If $F'(x)$ is invertible then
   $s = - F'(x)^(-1)F(x)$ otherwise $F'(x)^T F'(x) s = -F'(x)^T F(x)$. If $F'(x)^T F(x)$ is NOT zero then there
   exists a nontrivial (that is $F'(x)s \ne 0$) solution to the equation and this direction is
   $s = - [F'(x)^T F'(x)]^(-1) F'(x)^T F(x)$ so $Q'(0) = - F(x)^T F'(x) [F'(x)^T F'(x)]^(-T) F'(x)^T F(x)
   = - (F'(x)^T F(x)) [F'(x)^T F'(x)]^(-T) (F'(x)^T F(x))$. Since we are assuming $(F'(x)^T F(x)) \ne 0$
   and $F'(x)^T F'(x)$ has no negative eigenvalues $Q'(0) < 0$ so $s$ is a descent direction and the line
   search should succeed for small enough $\alpha$.

   Note that this RARELY happens in practice. Far more likely the linear system is not being solved
   (well enough?) or the Jacobian is wrong.

   `SNES_DIVERGED_MAX_IT` means that the solver reached the maximum number of iterations without satisfying any
   convergence criteria. `SNES_CONVERGED_ITS` means that `SNESConvergedSkip()` was chosen as the convergence test;
   thus the usual convergence criteria have not been checked and may or may not be satisfied.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPConvergedReason`, `SNESSetConvergenceTest()`, `SNESSetTolerances()`
E*/
typedef enum {                       /* converged */
  SNES_CONVERGED_FNORM_ABS      = 2, /* ||F|| < atol */
  SNES_CONVERGED_FNORM_RELATIVE = 3, /* ||F|| < rtol*||F_initial|| */
  SNES_CONVERGED_SNORM_RELATIVE = 4, /* Newton computed step size small; || delta x || < stol || x || */
  SNES_CONVERGED_ITS            = 5, /* maximum iterations reached */
  SNES_BREAKOUT_INNER_ITER      = 6, /* Flag to break out of inner loop after checking custom convergence, used in multi-phase flow when state changes */
  SNES_CONVERGED_USER           = 7, /* The user has indicated convergence for an arbitrary reason */
  /* diverged */
  SNES_DIVERGED_FUNCTION_DOMAIN      = -1, /* the new x location passed the function is not in the domain of F */
  SNES_DIVERGED_FUNCTION_COUNT       = -2,
  SNES_DIVERGED_LINEAR_SOLVE         = -3, /* the linear solve failed */
  SNES_DIVERGED_FUNCTION_NANORINF    = -4,
  SNES_DIVERGED_FNORM_NAN_DEPRECATED = -4,
  SNES_DIVERGED_MAX_IT               = -5,
  SNES_DIVERGED_LINE_SEARCH          = -6,  /* the line search failed */
  SNES_DIVERGED_INNER                = -7,  /* inner solve failed */
  SNES_DIVERGED_LOCAL_MIN            = -8,  /* || J^T b || is small, implies converged to local minimum of F() */
  SNES_DIVERGED_DTOL                 = -9,  /* || F || > divtol*||F_initial|| */
  SNES_DIVERGED_JACOBIAN_DOMAIN      = -10, /* Jacobian calculation does not make sense */
  SNES_DIVERGED_TR_DELTA             = -11,
  SNES_CONVERGED_TR_DELTA_DEPRECATED = -11,
  SNES_DIVERGED_USER                 = -12, /* The user has indicated divergence for an arbitrary reason */
  SNES_DIVERGED_OBJECTIVE_DOMAIN     = -13,
  SNES_DIVERGED_OBJECTIVE_NANORINF   = -14,

  SNES_CONVERGED_ITERATING = 0
} SNESConvergedReason;
PETSC_EXTERN const char *const *SNESConvergedReasons;

/*MC
   SNES_CONVERGED_FNORM_ABS - $||F|| \le abstol$

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_CONVERGED_FNORM_RELATIVE - $||F|| \le rtol*||F(x_0)||$ where $x_0$ is the initial guess

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
  SNES_CONVERGED_SNORM_RELATIVE - The 2-norm of the last step $\le stol * ||x||$ where `x` is the current
  solution and `stol` is the 4th argument to `SNESSetTolerances()`

  Options Database Key:
  -snes_stol stol - the step tolerance

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_COUNT - The user provided function has been called more times then the final
   argument to `SNESSetTolerances()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_DTOL - The norm of the function has increased by a factor of divtol set with `SNESSetDivergenceTolerance()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`, `SNESSetDivergenceTolerance()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_NANORINF - the 2-norm of the current function evaluation is not-a-number (NaN) or infinity, this
   is usually caused by a division of 0 by 0, or infinity.  See `SNESSetFunctionDomainError()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_DOMAIN - the function provided with `SNESSetFunction()` called `SNESSetFunctionDomainError()` and
   the solver could not recoverer.

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_OBJECTIVE_DOMAIN - the function provided with `SNESSetObjective()` called `SNESSetObjectiveDomainError()` and
   the solver could not recoverer.

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_JACOBIAN_DOMAIN - the function provided with `SNESSetJacobian()` called `SNESSetJacobianDomainError()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_MAX_IT - SNESSolve() has reached the maximum number of iterations requested

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_LINE_SEARCH - The line search has failed. This only occurs for a `SNES` solvers that use a line search

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`, `SNESLineSearch`
M*/

/*MC
   SNES_DIVERGED_LOCAL_MIN - the algorithm seems to have stagnated at a local minimum that is not zero.
   See the manual page for `SNESConvergedReason` for more details

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_CONERGED_ITERATING - this only occurs if `SNESGetConvergedReason()` is called during the `SNESSolve()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*E
   SNESNormSchedule - Frequency with which the norm is computed during a nonliner solve

   Values:
+   `SNES_NORM_DEFAULT`            - use the default behavior for the current `SNESType`
.   `SNES_NORM_NONE`               - avoid all norm computations
.   `SNES_NORM_ALWAYS`             - compute the norms whenever possible
.   `SNES_NORM_INITIAL_ONLY`       - compute the norm only when the algorithm starts
.   `SNES_NORM_FINAL_ONLY`         - compute the norm only when the algorithm finishes
-   `SNES_NORM_INITIAL_FINAL_ONLY` - compute the norm at the start and end of the algorithm

   Level: advanced

   Notes:
   Support for these is highly dependent on the solver.

   Some options limit the convergence tests that can be used.

   The `SNES_NORM_NONE` option is most commonly used when the nonlinear solver is being used as a smoother, for example for `SNESFAS`

   This is primarily used to turn off extra norm and function computation
   when the solvers are composed.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPSetNormType()`,
          `KSPSetConvergenceTest()`, `KSPSetPCSide()`
E*/
typedef enum {
  SNES_NORM_DEFAULT            = -1,
  SNES_NORM_NONE               = 0,
  SNES_NORM_ALWAYS             = 1,
  SNES_NORM_INITIAL_ONLY       = 2,
  SNES_NORM_FINAL_ONLY         = 3,
  SNES_NORM_INITIAL_FINAL_ONLY = 4
} SNESNormSchedule;
PETSC_EXTERN const char *const *const SNESNormSchedules;

/*MC
   SNES_NORM_NONE - Don't compute function and its L2 norm when possible

   Level: advanced

   Note:
   This is most useful for stationary solvers with a fixed number of iterations used as smoothers.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_DEFAULT`
M*/

/*MC
   SNES_NORM_ALWAYS - Compute the function and its L2 norm at each iteration.

   Level: advanced

   Note:
   Most solvers will use this no matter what norm type is passed to them.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_NONE`
M*/

/*MC
   SNES_NORM_INITIAL_ONLY - Compute the function and its L2 at iteration 0, but do not update it.

   Level: advanced

   Notes:
   This method is useful in composed methods, when a true solution might actually be found before `SNESSolve()` is called.
   This option enables the solve to abort on the zeroth iteration if this is the case.

   For solvers that require the computation of the L2 norm of the function as part of the method, this merely cancels
   the norm computation at the last iteration (if possible).

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_FINAL_ONLY`, `SNES_NORM_INITIAL_FINAL_ONLY`
M*/

/*MC
   SNES_NORM_FINAL_ONLY - Compute the function and its L2 norm on only the final iteration.

   Level: advanced

   Note:
   For solvers that require the computation of the L2 norm of the function as part of the method, behaves
   exactly as `SNES_NORM_DEFAULT`.  This method is useful when the function is gotten after `SNESSolve()` and
   used in subsequent computation for methods that do not need the norm computed during the rest of the
   solution procedure.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_INITIAL_ONLY`, `SNES_NORM_INITIAL_FINAL_ONLY`
M*/

/*MC
   SNES_NORM_INITIAL_FINAL_ONLY - Compute the function and its L2 norm on only the initial and final iterations.

   Level: advanced

   Note:
   This method combines the benefits of `SNES_NORM_INITIAL_ONLY` and `SNES_NORM_FINAL_ONLY`.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_SNES_NORM_INITIAL_ONLY`, `SNES_NORM_FINAL_ONLY`
M*/

/*E
   SNESFunctionType - Type of function computed

   Values:
+  `SNES_FUNCTION_DEFAULT`          - the default behavior for the current `SNESType`
.  `SNES_FUNCTION_UNPRECONDITIONED` - the original function provided
-  `SNES_FUNCTION_PRECONDITIONED`   - the modification of the function by the preconditioner

   Level: advanced

   Note:
   Support for these is dependent on the solver.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPSetNormType()`,
          `KSPSetConvergenceTest()`, `KSPSetPCSide()`
E*/
typedef enum {
  SNES_FUNCTION_DEFAULT          = -1,
  SNES_FUNCTION_UNPRECONDITIONED = 0,
  SNES_FUNCTION_PRECONDITIONED   = 1
} SNESFunctionType;
PETSC_EXTERN const char *const *const SNESFunctionTypes;

/*S
   SNESLineSearch - Abstract PETSc object that manages line-search operations for nonlinear solvers

   Level: beginner

   Note:
   See `SNESLineSearchSetFromOptions()` for all the line search options.

.seealso: [](ch_snes), `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNES`, `SNESLineSearchSetFromOptions()`
S*/
typedef struct _p_SNESLineSearch *SNESLineSearch;

/*J
   SNESLineSearchType - String with the name of a PETSc line search method `SNESLineSearch`. Provides all the linesearches for the nonlinear solvers, `SNES`,
                        in PETSc.

   Values:
+  `SNESLINESEARCHNONE`      - Simple damping line search, defaults to using the full Newton step
.  `SNESLINESEARCHBT`        - Backtracking line search over the L2 norm of the function or an objective function
.  `SNESLINESEARCHSECANT`    - Secant line search over the L2 norm of the function or an objective function
.  `SNESLINESEARCHCP`        - Critical point secant line search assuming $F(x) = \nabla G(x)$ for some unknown $G(x)$
.  `SNESLINESEARCHNLEQERR`   - Affine-covariant error-oriented linesearch
-  `SNESLINESEARCHBISECTION` - bisection line search for a root in the directional derivative
-  `SNESLINESEARCHSHELL`     - User provided `SNESLineSearch` implementation

   Level: beginner

   Note:
   Use `SNESLineSearchSetType()` or the options database key `-snes_linesearch_type` to set
   the specific line search algorithm to use with a given `SNES` object. Not all `SNESType` can utilize a line search.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchSetType()`, `SNES`
J*/
typedef const char *SNESLineSearchType;
#define SNESLINESEARCHBT        "bt"
#define SNESLINESEARCHNLEQERR   "nleqerr"
#define SNESLINESEARCHNONE      "none"
#define SNESLINESEARCHBASIC     PETSC_DEPRECATED_MACRO(3, 26, 0, "SNESLINESEARCHNONE", ) SNESLINESEARCHNONE
#define SNESLINESEARCHSECANT    "secant"
#define SNESLINESEARCHL2        PETSC_DEPRECATED_MACRO(3, 24, 0, "SNESLINESEARCHSECANT", ) SNESLINESEARCHSECANT
#define SNESLINESEARCHCP        "cp"
#define SNESLINESEARCHSHELL     "shell"
#define SNESLINESEARCHNCGLINEAR "ncglinear"
#define SNESLINESEARCHBISECTION "bisection"

/*E
    SNESLineSearchReason - indication if the line search has succeeded or failed and why

  Values:
+  `SNES_LINESEARCH_SUCCEEDED`              - the line search succeeded
.  `SNES_LINESEARCH_FAILED_NANORINF`        - a not a number of infinity appeared in the computions
.  `SNES_LINESEARCH_FAILED_FUNCTION_DOMAIN` - the function was evaluated outside of its domain, see `SNESSetFunctionDomainError()`
.  `SNES_LINESEARCH_FAILED_OBJECTIVE_DOMAIN`- the objective function was evaluated outside of its domain, see `SNESSetObjectiveDomainError()`
.  `SNES_LINESEARCH_FAILED_JACOBIAN_DOMAIN` - the Jacobian was evaluated outside of its domain, see `SNESSetJacobianDomainError()`
.  `SNES_LINESEARCH_FAILED_REDUCT`          - the linear search failed to get the requested decrease in its norm or objective
.  `SNES_LINESEARCH_FAILED_USER`            - used by `SNESLINESEARCHNLEQERR` to indicate the user changed the search direction inappropriately
-  `SNES_LINESEARCH_FAILED_FUNCTION`        - indicates the maximum number of function evaluations allowed has been surpassed, `SNESConvergedReason` is also
                                              set to `SNES_DIVERGED_FUNCTION_COUNT`

   Level: intermediate

   Developer Note:
   Some of these reasons overlap with values of `SNESConvergedReason`. It is possibly a better design to have `SNESConvergedReaon` alone used also for indicating line
   search failures.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPConvergedReason`, `SNESSetConvergenceTest()`,
          `SNESSetFunctionDomainError()`, `SNESSetJacobianDomainError()`
E*/
typedef enum {
  SNES_LINESEARCH_SUCCEEDED,
  SNES_LINESEARCH_FAILED_NANORINF,
  SNES_LINESEARCH_FAILED_FUNCTION_DOMAIN,
  SNES_LINESEARCH_FAILED_OBJECTIVE_DOMAIN,
  SNES_LINESEARCH_FAILED_JACOBIAN_DOMAIN,
  SNES_LINESEARCH_FAILED_REDUCT, /* INSUFFICIENT REDUCTION */
  SNES_LINESEARCH_FAILED_USER,
  SNES_LINESEARCH_FAILED_FUNCTION
} SNESLineSearchReason;

/*J
   SNESMSType - String with the name of a PETSc `SNESMS` method.

   Level: intermediate

.seealso: [](ch_snes), `SNESMS`, `SNESMSGetType()`, `SNESMSSetType()`, `SNES`
J*/
typedef const char *SNESMSType;
#define SNESMSM62       "m62"
#define SNESMSEULER     "euler"
#define SNESMSJAMESON83 "jameson83"
#define SNESMSVLTP11    "vltp11"
#define SNESMSVLTP21    "vltp21"
#define SNESMSVLTP31    "vltp31"
#define SNESMSVLTP41    "vltp41"
#define SNESMSVLTP51    "vltp51"
#define SNESMSVLTP61    "vltp61"

/*MC
   SNESNGMRESRestartType - the restart approach used by `SNESNGMRES`

  Values:
+   `SNES_NGMRES_RESTART_NONE`       - never restart
.   `SNES_NGMRES_RESTART_DIFFERENCE` - restart based upon difference criteria
-   `SNES_NGMRES_RESTART_PERIODIC`   - restart after a fixed number of iterations

  Options Database Keys:
+ -snes_ngmres_restart_type (difference|periodic|none) - set the restart type
- -snes_ngmres_restart 30                              - sets the number of iterations before restart for periodic

   Level: intermediate

.seealso: `SNES`, `SNESNGMRES`, `SNESNGMRESSetSelectType()`, `SNESNGMRESGetSelectType()`, `SNESNGMRESSetRestartType()`,
          `SNESNGMRESGetRestartType()`, `SNESNGMRESSelectType`
M*/
typedef enum {
  SNES_NGMRES_RESTART_NONE       = 0,
  SNES_NGMRES_RESTART_PERIODIC   = 1,
  SNES_NGMRES_RESTART_DIFFERENCE = 2
} SNESNGMRESRestartType;
PETSC_EXTERN const char *const SNESNGMRESRestartTypes[];

/*MC
   SNESNGMRESSelectType - the approach used by `SNESNGMRES` to determine how the candidate solution and
  combined solution are used to create the next iterate.

   Values:
+   `SNES_NGMRES_SELECT_NONE`       - choose the combined solution all the time
.   `SNES_NGMRES_SELECT_DIFFERENCE` - choose based upon the selection criteria
-   `SNES_NGMRES_SELECT_LINESEARCH` - choose based upon line search combination

  Options Database Key:
. -snes_ngmres_select_type (difference|none|linesearch) - select how the next iterate is created

   Level: intermediate

.seealso: `SNES`, `SNESNGMRES`, `SNESNGMRESSetSelectType()`, `SNESNGMRESGetSelectType()`, `SNESNGMRESSetRestartType()`,
          `SNESNGMRESGetRestartType()`, `SNESNGMRESRestartType`
M*/
typedef enum {
  SNES_NGMRES_SELECT_NONE       = 0,
  SNES_NGMRES_SELECT_DIFFERENCE = 1,
  SNES_NGMRES_SELECT_LINESEARCH = 2
} SNESNGMRESSelectType;
PETSC_EXTERN const char *const SNESNGMRESSelectTypes[];

/*MC
   SNESNCGType - the conjugate update approach for `SNESNCG`

   Values:
+   `SNES_NCG_FR`  - Fletcher-Reeves update
.   `SNES_NCG_PRP` - Polak-Ribiere-Polyak update, the default and the only one that tolerates generalized search directions
.   `SNES_NCG_HS`  - Hestenes-Steifel update
.   `SNES_NCG_DY`  - Dai-Yuan update
-   `SNES_NCG_CD`  - Conjugate Descent update

  Options Database Key:
. -snes_ncg_type (fr|prp|hs|dy|cd) - select the type

   Level: intermediate

.seealso: `SNES`, `SNESNCG`, `SNESNCGSetType()`
M*/
typedef enum {
  SNES_NCG_FR  = 0,
  SNES_NCG_PRP = 1,
  SNES_NCG_HS  = 2,
  SNES_NCG_DY  = 3,
  SNES_NCG_CD  = 4
} SNESNCGType;
PETSC_EXTERN const char *const SNESNCGTypes[];

/*MC
   SNESQNScaleType - the scaling type used by `SNESQN`

   Values:
+   `SNES_QN_SCALE_NONE`     - don't scale the problem
.   `SNES_QN_SCALE_SCALAR`   - use Shanno scaling
.   `SNES_QN_SCALE_DIAGONAL` - scale with a diagonalized BFGS formula (see Gilbert and Lemarechal 1989), available
-   `SNES_QN_SCALE_JACOBIAN` - scale by solving a linear system coming from the Jacobian you provided with `SNESSetJacobian()`
                               computed at the first iteration of `SNESQN` and at ever restart.

    Options Database Key:
. -snes_qn_scale_type (diagonal|none|scalar|jacobian) - Select the scaling type

   Level: intermediate

.seealso: `SNES`, `SNESQN`, `SNESQNSetScaleType()`, `SNESQNType`, `SNESQNSetType()`, `SNESQNSetRestartType()`, `SNESQNRestartType`
M*/
typedef enum {
  SNES_QN_SCALE_DEFAULT  = 0,
  SNES_QN_SCALE_NONE     = 1,
  SNES_QN_SCALE_SCALAR   = 2,
  SNES_QN_SCALE_DIAGONAL = 3,
  SNES_QN_SCALE_JACOBIAN = 4
} SNESQNScaleType;
PETSC_EXTERN const char *const SNESQNScaleTypes[];

/*MC
   SNESQNRestartType - the restart approached used by `SNESQN`

   Values:
+   `SNES_QN_RESTART_NONE`     - never restart
.   `SNES_QN_RESTART_POWELL`   - restart based upon descent criteria
-   `SNES_QN_RESTART_PERIODIC` - restart after a fixed number of iterations

  Options Database Keys:
+ -snes_qn_restart_type (powell|periodic|none) - set the restart type
- -snes_qn_m m                                 - sets the number of stored updates and the restart period for periodic

   Level: intermediate

.seealso: `SNES`, `SNESQN`, `SNESQNSetScaleType()`, `SNESQNType`, `SNESQNSetType()`, `SNESQNSetRestartType()`, `SNESQNScaleType`
M*/
typedef enum {
  SNES_QN_RESTART_DEFAULT  = 0,
  SNES_QN_RESTART_NONE     = 1,
  SNES_QN_RESTART_POWELL   = 2,
  SNES_QN_RESTART_PERIODIC = 3
} SNESQNRestartType;
PETSC_EXTERN const char *const SNESQNRestartTypes[];

/*MC
   SNESQNType - the type used by `SNESQN`

  Values:
+   `SNES_QN_LBFGS`      - LBFGS variant
.   `SNES_QN_BROYDEN`    - Broyden variant
-   `SNES_QN_BADBROYDEN` - Bad Broyden variant

  Options Database Key:
. -snes_qn_type (lbfgs|broyden|badbroyden) - quasi-Newton type

   Level: intermediate

.seealso: `SNES`, `SNESQN`, `SNESQNSetScaleType()`, `SNESQNSetType()`, `SNESQNScaleType`, `SNESQNRestartType`, `SNESQNSetRestartType()`
M*/
typedef enum {
  SNES_QN_LBFGS      = 0,
  SNES_QN_BROYDEN    = 1,
  SNES_QN_BADBROYDEN = 2
} SNESQNType;
PETSC_EXTERN const char *const SNESQNTypes[];

/*E
  SNESCompositeType - Determines how two or more preconditioners are composed with the `SNESType` of `SNESCOMPOSITE`

  Values:
+ `SNES_COMPOSITE_ADDITIVE`        - results from application of all preconditioners are added together
. `SNES_COMPOSITE_MULTIPLICATIVE`  - preconditioners are applied sequentially to the residual freshly
                                     computed after the previous preconditioner application
- `SNES_COMPOSITE_ADDITIVEOPTIMAL` - uses a linear combination of the solutions obtained with each preconditioner that approximately minimize the function
                                     value at the new iteration.

   Level: beginner

.seealso: [](sec_pc), `PCCOMPOSITE`, `PCFIELDSPLIT`, `PC`, `PCCompositeSetType()`, `PCCompositeType`
E*/
typedef enum {
  SNES_COMPOSITE_ADDITIVE,
  SNES_COMPOSITE_MULTIPLICATIVE,
  SNES_COMPOSITE_ADDITIVEOPTIMAL
} SNESCompositeType;
PETSC_EXTERN const char *const SNESCompositeTypes[];

/*E
    SNESFASType - Determines the type of nonlinear multigrid method that is run.

   Values:
+  `SNES_FAS_MULTIPLICATIVE` (default) - traditional V or W cycle as determined by `SNESFASSetCycles()`
.  `SNES_FAS_ADDITIVE`                 - additive FAS cycle
.  `SNES_FAS_FULL`                     - full FAS cycle
-  `SNES_FAS_KASKADE`                  - Kaskade FAS cycle

   Level: beginner

.seealso: [](ch_snes), `SNESFAS`, `PCMGSetType()`, `PCMGType`
E*/
typedef enum {
  SNES_FAS_MULTIPLICATIVE,
  SNES_FAS_ADDITIVE,
  SNES_FAS_FULL,
  SNES_FAS_KASKADE
} SNESFASType;
PETSC_EXTERN const char *const SNESFASTypes[];

/*MC
   SNESNewtonALCorrectionType - the approach used by `SNESNEWTONAL` to determine
   the correction to the current increment. While the exact correction satisfies
   the constraint surface at every iteration, it also requires solving a quadratic
   equation which may not have real roots. Conversely, the normal correction is more
   efficient and always yields a real correction and is the default.

   Values:
+   `SNES_NEWTONAL_CORRECTION_EXACT`  - choose the correction which exactly satisfies the constraint
-   `SNES_NEWTONAL_CORRECTION_NORMAL` - choose the correction in the updated normal hyper-surface to the constraint surface

   Options Database Key:
. -snes_newtonal_correction_type (exact|normal) - exactly satisfy the constraint or satisfy it on the normal hyper-surface

   Level: intermediate

.seealso: `SNES`, `SNESNEWTONAL`, `SNESNewtonALSetCorrectionType()`
M*/
typedef enum {
  SNES_NEWTONAL_CORRECTION_EXACT  = 0,
  SNES_NEWTONAL_CORRECTION_NORMAL = 1,
} SNESNewtonALCorrectionType;
PETSC_EXTERN const char *const SNESNewtonALCorrectionTypes[];
