#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

/*MC
  TAOTERMKL - A `TaoTerm` that computes the KL diveregence $\Sum_i x_i log(x_i / p_i)$ for solution $x$ and parameters $p$.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoTerm`, `TAOTERML1`, `TAOTERMHALFL2SQUARED`
MC*/
PETSC_INTERN PetscErrorCode TaoTermCreate_KL(TaoTerm term)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}
