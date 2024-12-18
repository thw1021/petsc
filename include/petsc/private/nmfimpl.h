#pragma once

#include <petscnmf.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>

PETSC_EXTERN PetscBool      PetscNMFRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscNMFRegisterAll(void);

typedef struct _PetscNMFOps *PetscNMFOps;

struct _PetscNMFOps {
  PetscErrorCode (*fit)(PetscNMF);

  PetscErrorCode (*setup)(PetscNMF);
  PetscErrorCode (*view)(PetscNMF, PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscNMF, PetscOptionItems *);
  PetscErrorCode (*destroy)(PetscNMF);
};

typedef struct _n_NMFMappedTerm NMFMappedTerm;

struct _n_NMFMappedTerm {
  TaoMappedTerm             mterm;
  PetscNMFTermParameterType x_type;
  PetscNMFTermParameterType p_type;
};

struct _p_PetscNMF {
  PETSCHEADER(struct _PetscNMFOps);

  void *ctx; /* user provided context */

  void *data;
  Tao   tao;

  TaoMappedTerm objective_term;

  TaoTerm orig_sum;

  PetscNMFInitType init_type;

  PetscInt rank;
  PetscInt num_terms;

  Vec solution_vec;
  Mat solution_mat;

  PetscBool setupcalled;
};

PETSC_EXTERN PetscLogEvent PETSCNMF_Fit;
