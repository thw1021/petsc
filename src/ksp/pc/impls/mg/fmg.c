/*
     Full multigrid using either additive or multiplicative V or W cycle
*/
#include <petsc/private/pcmgimpl.h>

PetscErrorCode PCMGFCycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PetscInt i, l = mglevels[0]->levels;

  PetscFunctionBegin;
  if (!transpose) {
    /* restrict the RHS through all levels to coarsest. */
    for (i = l - 1; i > 0; i--) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }

    /* work our way up through the levels */
    if (matapp) {
      if (!mglevels[0]->X) PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
      else PetscCall(MatZeroEntries(mglevels[0]->X));
    } else PetscCall(VecZeroEntries(mglevels[0]->x));
    for (i = 0; i < l - 1; i++) {
      PetscCall(PCMGMCycle_Private(pc, &mglevels[i], transpose, matapp, NULL));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->X, &mglevels[i + 1]->X));
      else PetscCall(MatInterpolate(mglevels[i + 1]->interpolate, mglevels[i]->x, mglevels[i + 1]->x));
      if (mglevels[i + 1]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i + 1]->eventinterprestrict, 0, 0, 0, 0));
    }
    PetscCall(PCMGMCycle_Private(pc, &mglevels[l - 1], transpose, matapp, NULL));
  } else {
    /* Adjoint of the above: each level's V-cycle is linear from a zero initial guess, so its own
       transpose is exactly PCMGMCycle_Private(transpose=true). What remains is the adjoint of
       feeding the coarser V-cycle's output forward as the next level's initial guess, which
       PCMGMCycle_Private() alone cannot express (see PCMGKCycle_Private() for the derivation);
       peel it off here the same way, using whole V-cycles in place of single smoothing sweeps. */
    for (i = l - 1; i > 0; i--) {
      if (matapp) {
        if (!mglevels[i]->X) PetscCall(MatDuplicate(mglevels[i]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[i]->X));
        else PetscCall(MatZeroEntries(mglevels[i]->X));
      } else PetscCall(VecZeroEntries(mglevels[i]->x));
      PetscCall(PCMGMCycle_Private(pc, &mglevels[i], transpose, matapp, NULL));

      if (mglevels[i]->eventresidual) PetscCall(PetscLogEventBegin(mglevels[i]->eventresidual, 0, 0, 0, 0));
      if (matapp) {
        if (!mglevels[i]->R) PetscCall(MatDuplicate(mglevels[i]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[i]->R));
        PetscCall((*mglevels[i]->matresidualtranspose)(mglevels[i]->A, mglevels[i]->B, mglevels[i]->X, mglevels[i]->R));
      } else PetscCall((*mglevels[i]->residualtranspose)(mglevels[i]->A, mglevels[i]->b, mglevels[i]->x, mglevels[i]->r));
      if (mglevels[i]->eventresidual) PetscCall(PetscLogEventEnd(mglevels[i]->eventresidual, 0, 0, 0, 0));

      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->interpolate, mglevels[i]->R, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->interpolate, mglevels[i]->r, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }
    if (matapp) {
      if (!mglevels[0]->X) PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
      else PetscCall(MatZeroEntries(mglevels[0]->X));
    } else PetscCall(VecZeroEntries(mglevels[0]->x));
    PetscCall(PCMGMCycle_Private(pc, &mglevels[0], transpose, matapp, NULL));

    for (i = 1; i < l; i++) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->X, mglevels[i]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCMGKCycle_Private(PC pc, PC_MG_Levels **mglevels, PetscBool transpose, PetscBool matapp)
{
  PC_MG    *mg = (PC_MG *)pc->data;
  PetscInt  i, l = mglevels[0]->levels;
  KSP       smooth;
  PetscBool smoothtranspose;

  PetscFunctionBegin;
  if (!transpose) {
    /* restrict the RHS through all levels to coarsest. */
    for (i = l - 1; i > 0; i--) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->restrct, mglevels[i]->B, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->restrct, mglevels[i]->b, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }

    /* coarse solve: there is no coarser level to interpolate an initial guess from, so this is
       always the (unconditional) down smoother from a zero initial guess. */
    if (matapp) {
      if (!mglevels[0]->X) PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
      else PetscCall(MatZeroEntries(mglevels[0]->X));
    } else PetscCall(VecZeroEntries(mglevels[0]->x));
    if (mglevels[0]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[0]->eventsmoothsolve, 0, 0, 0, 0));
    PetscCall(PCMGKSPSmooth_Private(pc, mglevels[0]->smoothd, mglevels[0]->b, mglevels[0]->x, mglevels[0]->B, mglevels[0]->X, PETSC_FALSE, matapp));
    if (mglevels[0]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[0]->eventsmoothsolve, 0, 0, 0, 0));

    /* work our way up through the levels, interpolating the coarser solution as the initial guess
       and smoothing with the up smoother (the transpose of the down smoother, when symmetric) */
    for (i = 1; i < l; i++) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolate(mglevels[i]->interpolate, mglevels[i - 1]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolate(mglevels[i]->interpolate, mglevels[i - 1]->x, mglevels[i]->x));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));

      smooth          = mg->symmetric ? mglevels[i]->smoothd : mglevels[i]->smoothu;
      smoothtranspose = mg->symmetric;
      if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
      PetscCall(PCMGKSPSmooth_Private(pc, smooth, mglevels[i]->b, mglevels[i]->x, mglevels[i]->B, mglevels[i]->X, smoothtranspose, matapp));
      if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
    }
  } else {
    /* Adjoint of the above. Writing the up-smoothing step at level i as x_i = x0_i + Z_i(b_i - A_i
       x0_i), with x0_i the interpolated coarser solution and Z_i the zero-initial-guess smoother,
       the joint linear map (x0_i, b_i) -> x_i has adjoint (y) -> (y - A_i^T Z_i^T(y), Z_i^T(y)):
       the second component feeds directly into b_i's adjoint, and the first is what must still be
       propagated (through the transpose of the interpolation) to the coarser level as its own
       output's adjoint. Peeling this off from the finest level down to the coarsest produces, at
       each level, the adjoint contribution to that level's b (stored into x_i, reusing it as
       scratch); a second pass from coarsest to finest then adds in the adjoint of the
       interpolation chain, level by level, to assemble the true adjoint of the whole cascade. */
    for (i = l - 1; i > 0; i--) {
      smooth          = mg->symmetric ? mglevels[i]->smoothd : mglevels[i]->smoothu;
      smoothtranspose = (PetscBool)!mg->symmetric;
      if (matapp) {
        if (!mglevels[i]->X) PetscCall(MatDuplicate(mglevels[i]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[i]->X));
        else PetscCall(MatZeroEntries(mglevels[i]->X));
      } else PetscCall(VecZeroEntries(mglevels[i]->x));
      if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));
      PetscCall(PCMGKSPSmooth_Private(pc, smooth, mglevels[i]->b, mglevels[i]->x, mglevels[i]->B, mglevels[i]->X, smoothtranspose, matapp));
      if (mglevels[i]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[i]->eventsmoothsolve, 0, 0, 0, 0));

      if (mglevels[i]->eventresidual) PetscCall(PetscLogEventBegin(mglevels[i]->eventresidual, 0, 0, 0, 0));
      if (matapp) {
        if (!mglevels[i]->R) PetscCall(MatDuplicate(mglevels[i]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[i]->R));
        PetscCall((*mglevels[i]->matresidualtranspose)(mglevels[i]->A, mglevels[i]->B, mglevels[i]->X, mglevels[i]->R));
      } else PetscCall((*mglevels[i]->residualtranspose)(mglevels[i]->A, mglevels[i]->b, mglevels[i]->x, mglevels[i]->r));
      if (mglevels[i]->eventresidual) PetscCall(PetscLogEventEnd(mglevels[i]->eventresidual, 0, 0, 0, 0));

      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatRestrict(mglevels[i]->interpolate, mglevels[i]->R, &mglevels[i - 1]->B));
      else PetscCall(MatRestrict(mglevels[i]->interpolate, mglevels[i]->r, mglevels[i - 1]->b));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }

    /* coarsest level: adjoint of the unconditional down smoother from a zero initial guess */
    if (matapp) {
      if (!mglevels[0]->X) PetscCall(MatDuplicate(mglevels[0]->B, MAT_DO_NOT_COPY_VALUES, &mglevels[0]->X));
      else PetscCall(MatZeroEntries(mglevels[0]->X));
    } else PetscCall(VecZeroEntries(mglevels[0]->x));
    if (mglevels[0]->eventsmoothsolve) PetscCall(PetscLogEventBegin(mglevels[0]->eventsmoothsolve, 0, 0, 0, 0));
    PetscCall(PCMGKSPSmooth_Private(pc, mglevels[0]->smoothd, mglevels[0]->b, mglevels[0]->x, mglevels[0]->B, mglevels[0]->X, PETSC_TRUE, matapp));
    if (mglevels[0]->eventsmoothsolve) PetscCall(PetscLogEventEnd(mglevels[0]->eventsmoothsolve, 0, 0, 0, 0));

    /* accumulate the adjoint of the interpolation chain, coarsest to finest */
    for (i = 1; i < l; i++) {
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventBegin(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
      if (matapp) PetscCall(MatMatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->X, mglevels[i]->X, &mglevels[i]->X));
      else PetscCall(MatInterpolateAdd(mglevels[i]->restrct, mglevels[i - 1]->x, mglevels[i]->x, mglevels[i]->x));
      if (mglevels[i]->eventinterprestrict) PetscCall(PetscLogEventEnd(mglevels[i]->eventinterprestrict, 0, 0, 0, 0));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
