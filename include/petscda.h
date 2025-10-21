#ifndef PETSC_DA_H
#define PETSC_DA_H

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

/*S
   DA - Abstract PETSc object that orchestrates ensemble-based data assimilation workflows

   Level: intermediate

   Notes:
   DA manages ensemble state vectors, observation-error descriptions, and user-provided
   model/observation operators so that analysis steps and forecast propagations can be
   expressed independently of the underlying parallel layout.

.seealso: `DACreate()`, `DASetType()`, `DASetFromOptions()`, `DASetSizes()`, `DAComputeMean()`, `DAComputeAnomalies()`, `DAAnalysis()`, `DAApplyModel()`, `DAType`, `DAGetType()`
S*/

/* MANSEC = TS */
/* SUBMANSEC = DA */

typedef struct _p_DA *DA;

typedef enum {
  DAETKF_SQRT_CHOLESKY = 0,
  DAETKF_SQRT_EIGEN    = 1
} DAETKFSqrtType;

typedef const char *DAType;
#define DAETKF "daetkf"

/* Logging support */
PETSC_EXTERN PetscClassId DA_CLASSID;

PETSC_EXTERN PetscErrorCode DAInitializePackage(void);
PETSC_EXTERN PetscErrorCode DAFinalizePackage(void);

PETSC_EXTERN PetscErrorCode DARegister(const char[], PetscErrorCode (*)(DA));
PETSC_EXTERN PetscErrorCode DARegisterAll(void);

PETSC_EXTERN PetscErrorCode DACreate(MPI_Comm, DA *);
PETSC_EXTERN PetscErrorCode DADestroy(DA *);
PETSC_EXTERN PetscErrorCode DASetType(DA, DAType);
PETSC_EXTERN PetscErrorCode DAGetType(DA, DAType *);
PETSC_EXTERN PetscErrorCode DAView(DA, PetscViewer);
PETSC_EXTERN PetscErrorCode DAViewFromOptions(DA, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode DASetFromOptions(DA);

PETSC_EXTERN PetscErrorCode DASetSizes(DA, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DAGetSizes(DA, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DASetUp(DA);

PETSC_EXTERN PetscErrorCode DASetObsErrorVariance(DA, Vec);
PETSC_EXTERN PetscErrorCode DAGetObsErrorVariance(DA, Vec *);

PETSC_EXTERN PetscErrorCode DASetOrthogonalTransform(DA, Mat);
PETSC_EXTERN PetscErrorCode DAGetOrthogonalTransform(DA, Mat *);

PETSC_EXTERN PetscErrorCode DAGetEnsembleMember(DA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode DARestoreEnsembleMember(DA, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode DASetEnsembleMember(DA, PetscInt, Vec);

PETSC_EXTERN PetscErrorCode DAComputeMean(DA, Vec);
PETSC_EXTERN PetscErrorCode DAComputeAnomalies(DA, Mat *);
PETSC_EXTERN PetscErrorCode DAAnalysis(DA, Vec, PetscErrorCode (*)(Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DAApplyModel(DA, PetscErrorCode (*)(Vec, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode VecSetRandomGaussian(Vec, PetscRandom, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode DAETKFRegister(void);
PETSC_EXTERN PetscErrorCode DAETKFInitializePackage(void);
PETSC_EXTERN PetscErrorCode DAETKFFinalizePackage(void);
PETSC_EXTERN PetscErrorCode DAETKFSetSqrtType(DA, DAETKFSqrtType);
PETSC_EXTERN PetscErrorCode DAETKFGetSqrtType(DA, DAETKFSqrtType *);

/* Convenience aliases mapping DA API to legacy ETKF naming */
typedef DA             ETKF;
typedef DAType         ETKFType;
typedef DAETKFSqrtType ETKFSqrtType;

#define ETKFSTANDARD DAETKF

static inline PetscErrorCode ETKFCreate(MPI_Comm comm, ETKF *etkf)
{
  return DACreate(comm, etkf);
}
static inline PetscErrorCode ETKFDestroy(ETKF *etkf)
{
  return DADestroy(etkf);
}
static inline PetscErrorCode ETKFSetType(ETKF etkf, ETKFType type)
{
  return DASetType(etkf, type);
}
static inline PetscErrorCode ETKFGetType(ETKF etkf, ETKFType *type)
{
  return DAGetType(etkf, type);
}
static inline PetscErrorCode ETKFSetFromOptions(ETKF etkf)
{
  return DASetFromOptions(etkf);
}
static inline PetscErrorCode ETKFSetUp(ETKF etkf)
{
  return DASetUp(etkf);
}
static inline PetscErrorCode ETKFSetSizes(ETKF etkf, PetscInt n, PetscInt p, PetscInt m)
{
  return DASetSizes(etkf, n, p, m);
}
static inline PetscErrorCode ETKFGetSizes(ETKF etkf, PetscInt *n, PetscInt *p, PetscInt *m)
{
  return DAGetSizes(etkf, n, p, m);
}
static inline PetscErrorCode ETKFSetObsErrorVariance(ETKF etkf, Vec vec)
{
  return DASetObsErrorVariance(etkf, vec);
}
static inline PetscErrorCode ETKFGetObsErrorVariance(ETKF etkf, Vec *vec)
{
  return DAGetObsErrorVariance(etkf, vec);
}
static inline PetscErrorCode ETKFSetOrthogonalTransform(ETKF etkf, Mat U)
{
  return DASetOrthogonalTransform(etkf, U);
}
static inline PetscErrorCode ETKFGetOrthogonalTransform(ETKF etkf, Mat *U)
{
  return DAGetOrthogonalTransform(etkf, U);
}
static inline PetscErrorCode ETKFGetEnsembleMember(ETKF etkf, PetscInt i, Vec *v)
{
  return DAGetEnsembleMember(etkf, i, v);
}
static inline PetscErrorCode ETKFRestoreEnsembleMember(ETKF etkf, PetscInt i, Vec *v)
{
  return DARestoreEnsembleMember(etkf, i, v);
}
static inline PetscErrorCode ETKFSetEnsembleMember(ETKF etkf, PetscInt i, Vec v)
{
  return DASetEnsembleMember(etkf, i, v);
}
static inline PetscErrorCode ETKFComputeMean(ETKF etkf, Vec mean)
{
  return DAComputeMean(etkf, mean);
}
static inline PetscErrorCode ETKFComputeAnomalies(ETKF etkf, Mat *anoms)
{
  return DAComputeAnomalies(etkf, anoms);
}
static inline PetscErrorCode ETKFAnalysis(ETKF etkf, Vec obs, PetscErrorCode (*H)(Vec, Vec, void *), void *ctx)
{
  return DAAnalysis(etkf, obs, H, ctx);
}
static inline PetscErrorCode ETKFApplyModel(ETKF etkf, PetscErrorCode (*M)(Vec, Vec, void *), void *ctx)
{
  return DAApplyModel(etkf, M, ctx);
}
static inline PetscErrorCode ETKFSetSqrtType(ETKF etkf, ETKFSqrtType type)
{
  return DAETKFSetSqrtType(etkf, type);
}
static inline PetscErrorCode ETKFGetSqrtType(ETKF etkf, ETKFSqrtType *type)
{
  return DAETKFGetSqrtType(etkf, type);
}
#endif /* PETSC_DA_H */
