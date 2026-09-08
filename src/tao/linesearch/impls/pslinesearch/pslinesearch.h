#pragma once
#include <petsc/private/taolinesearchimpl.h>

/* Context for the (nonmonotone) backtracking line search of proximal splitting algorithms.

   An Armijo-type line search moves along a direction, x + step * d. In proximal algorithms the trial
   point is instead the output of a proximal map, so TAOLINESEARCHPS holds the smooth term f and the
   proximal term g of the TAOFB objective and recomputes the trial point
   prox_{step g}(x_k - step grad f(x_k)) each time the step shrinks. The step is accepted when

     f(x_{k+1}) <= R + <grad f(x_k), x_{k+1} - x_k> + |x_{k+1} - x_k|^2 / (2 step),

   where R = f(x_k) for the monotone line search and R = max(f(x_k), ..., f(x_{k-M+1})) for the
   nonmonotone one with memory size M, as in FASTA (Goldstein, Studer, Baraniuk, arXiv:1501.04979). */
typedef struct {
  PetscReal *memory;     /* previous values of f at the base points */
  PetscReal  eta;        /* step size decrease factor < 1 */
  PetscReal  ref;        /* nonmonotone reference value R */
  PetscReal  cert;       /* f(x_{k+1}) - (R + <grad f(x_k), x_{k+1} - x_k> + |x_{k+1} - x_k|^2 / (2 step)) */
  PetscReal  f_scale;    /* scale of the smooth term */
  PetscReal  term_scale; /* scale of the proximal term */
  PetscInt   memorySize;
  PetscInt   current; /* FIFO position in memory */
  PetscBool  memorySetup;
  TaoTerm    f_term;
  TaoTerm    prox_term;
  Vec        f_param, term_param;
  Vec        x; /* reference to the base point vector, to detect layout changes */
  Vec        work, work2;
} TaoLineSearch_PS;

PETSC_INTERN PetscErrorCode TaoPSLineSearchSetTerms(TaoLineSearch, TaoTermMapping, Vec, TaoTermMapping, Vec);
