#include <petscmacros.h>

#if PetscDefined(HAVE_DEVICE)
  #include <petsc/private/vecimpl.h>
  #include <petsc/private/cpp/type_traits.hpp> // PetscObjectCast()

  #include <limits> // std::numeric_limits

PETSC_INTERN PetscErrorCode VecCUPMCheckMinimumPinnedMemory_Internal(Vec v, PetscBool *set)
{
  auto      mem = static_cast<PetscInt>(v->minimum_bytes_pinned_memory);
  PetscBool flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin(PetscObjectCast(v));
  PetscCall(PetscOptionsRangeInt("-vec_pinned_memory_min", "Minimum size (in bytes) for an allocation to use pinned memory on host", "VecSetPinnedMemoryMin", mem, &mem, &flg, 0, std::numeric_limits<decltype(mem)>::max()));
  if (flg) v->minimum_bytes_pinned_memory = mem;
  PetscOptionsEnd();
  if (set) *set = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
