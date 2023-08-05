#pragma once

#include <petscbm.h>

struct _PetscBMOps {
  PetscErrorCode (*setfromoptions)(PetscBM, PetscOptionItems *);
  PetscErrorCode (*setup)(PetscBM);
  PetscErrorCode (*run)(PetscBM);
  PetscErrorCode (*view)(PetscBM, PetscViewer);
  PetscErrorCode (*reset)(PetscBM);
  PetscErrorCode (*destroy)(PetscBM);
};

struct _p_PetscBM {
  PETSCHEADER(struct _PetscBMOps);
  PetscBool       setupcalled;
  PetscInt        size;
  PetscLogHandler lhdlr;
  void           *data;
};
