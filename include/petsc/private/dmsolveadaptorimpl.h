#pragma once

#include <petscdmsolveadaptor.h>
#include <petsc/private/petscimpl.h>

typedef struct _DMSolveAdaptorOps *DMSolveAdaptorOps;
struct _DMSolveAdaptorOps {
  PetscErrorCode (*setfromoptions)(DMSolveAdaptor);
  PetscErrorCode (*setup)(DMSolveAdaptor);
  PetscErrorCode (*view)(DMSolveAdaptor, PetscViewer);
  PetscErrorCode (*destroy)(DMSolveAdaptor);
  PetscErrorCode (*adapt)(DMSolveAdaptor);
};

struct _p_DMSolveAdaptor {
  PETSCHEADER(struct _DMSolveAdaptorOps);
  void *data;

  DM        dm;       // DM for solver
  KSP       ksp;      // Solver to be adapted
  SNES      snes;     // Solver to be adapted
  TS        ts;       // Solver to be adapted
  Tao       tao;      // Solver to be adapted
  VecTagger wrongTag; // Criteria for adaptivity
  PetscInt  numSeq;   // Number of sequential adaptations
};
