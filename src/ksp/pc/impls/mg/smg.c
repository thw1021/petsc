/*
     Additive Multigrid V Cycle routine
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGACycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  KSP      smooth;
  PetscInt i, l = mglevels[0]->levels;

  PetscFunctionBegin;
  /* compute RHS on each level */
  for (i = l - 1; i > 0; i--) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
    } else {
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->interpolate, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->interpolate, mglevels[i]->b, mglevels[i - 1]->b));
    }
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  /* solve separately on each level */
  for (i = 0; i < l; i++) {
    if (matapp) {
      if (!mglevels[i]->X) PetscCall(MatDuplicate(mglevels[i]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[i]->X));
      else PetscCall(MatZeroEntries(mglevels[i]->X));
    } else PetscCall(VecZeroEntries(mglevels[i]->x));
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    /* The additive cycle applies a single smoothing step per level: there is no pre/post sandwich for
       PCMGSetSymmetric() to exploit, so (unlike PCMGMCycle_Private()) it is ignored here. The operator
       is symmetric only if the caller ensures smoothu (== smoothd unless distinct) is itself self-adjoint. */
    smooth = transpose ? mglevels[i]->smoothu : mglevels[i]->smoothd;
    PetscCall(PCMGKSPSmooth_Private(pc, smooth, mglevels[i]->b, mglevels[i]->x, mglevels[i]->B, mglevels[i]->X, transpose, matapp));
    if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
  }
  for (i = 1; i < l; i++) {
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    if (!transpose) {
      if (matapp) PetscCall(MatMatInterpolateAdd(mglevels[i]->interpolate, mglevels[i - 1]->X, mglevels[i]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolateAdd(mglevels[i]->interpolate, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    } else {
      if (matapp) PetscCall(MatMatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->X, mglevels[i]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
    }
    if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
