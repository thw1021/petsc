#pragma once
#include <petsc/private/taoimpl.h>
#include <petsc/private/petscimpl.h>

typedef struct _TaoRegularizerOps *TaoRegularizerOps;
struct _TaoRegularizerOps {
  PetscErrorCode (*computeobjective)(TaoRegularizer, Vec, PetscReal *, void *);
  PetscErrorCode (*computegradient)(TaoRegularizer, Vec, Vec, void *);
  PetscErrorCode (*computeobjectiveandgradient)(TaoRegularizer, Vec, PetscReal *, Vec, void *);
  PetscErrorCode (*computehessian)(TaoRegularizer, Vec, Mat, Mat, void *);
  PetscErrorCode (*setup)(TaoRegularizer);
  PetscErrorCode (*apply)(TaoRegularizer, Vec, PetscReal *, Vec, Vec);
  PetscErrorCode (*view)(TaoRegularizer, PetscViewer);
  PetscErrorCode (*setfromoptions)(TaoRegularizer, PetscOptionItems *);
  PetscErrorCode (*reset)(TaoRegularizer);
  PetscErrorCode (*destroy)(TaoRegularizer);
  PetscErrorCode (*monitor)(TaoRegularizer);
};

struct _p_TaoRegularizer {
  PETSCHEADER(struct _TaoRegularizerOps);
  void *userctx_func;
  void *userctx_grad;
  void *userctx_funcgrad;

  PetscViewer viewer;

  PetscBool setupcalled;
  PetscBool usetaoroutines;
  PetscBool usemonitor;
  PetscBool hasobjective;
  PetscBool hasgradient;
  PetscBool hasobjectiveandgradient;
  PetscBool hashessian;
  void     *data;

  Vec y, workvec; //TODO need to create workvec

  PetscReal          scale;
  TaoRegularizerType type;

  Tao parent_tao;
  Tao reg_tao;
};

PETSC_EXTERN PetscLogEvent TAOREGULARIZER_Apply;
PETSC_EXTERN PetscLogEvent TAOREGULARIZER_Eval;
