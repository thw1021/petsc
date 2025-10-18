#include <petsc/private/taoimpl.h>

PETSC_INTERN PetscErrorCode VecIfNotCongruentGetSameLayoutVec(Vec a, Vec *b)
{
  PetscFunctionBegin;
  if (!(*b)) {
    PetscCall(VecDuplicate(a, b));
  } else {
    PetscLayout layout_a, layout_b;
    PetscBool   is_same;

    PetscCall(VecGetLayout(a, &layout_a));
    PetscCall(VecGetLayout(*b, &layout_b));
    PetscCall(PetscLayoutCompare(layout_a, layout_b, &is_same));
    if (!is_same) {
      PetscCall(VecDestroy(b));
      PetscCall(VecDuplicate(a, b));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
