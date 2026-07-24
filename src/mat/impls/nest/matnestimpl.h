#pragma once

#include <petsc/private/matimpl.h>

struct MatNestISPair {
  IS *row, *col;
};

typedef struct {
  PetscInt             nr, nc; /* nr x nc blocks */
  Mat                **m;
  struct MatNestISPair isglobal;
  struct MatNestISPair islocal;
  Vec                 *left, *right;
  PetscInt            *row_len, *col_len;
  PetscObjectState    *nnzstate;
  PetscBool            splitassembly;
} Mat_Nest;

/* context for multi-shift matrices created via MatCreateNestFromMultipleShifts() */
struct _n_Mat_MultiShift {
  Mat          K;
  Mat          M;
  PetscInt     nshift;
  PetscScalar *sigma;
  PetscBool   *cmplx;
  MatStructure str;
  MatState     state;
};
typedef struct _n_Mat_MultiShift *Mat_MultiShift;

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode MatMultiShiftBuildShiftedMatrix_Internal(Mat, PetscScalar, Mat, MatStructure, PetscBool, Mat *);

/*
  MatCheckMultiShift - Check that a given Mat was created via MatCreateNestFromMultipleShifts().
*/
#define MatCheckMultiShift(A) \
  do { \
    PetscContainer container; \
    Mat_MultiShift actx; \
    PetscBool      same; \
    PetscCall(PetscObjectQuery((PetscObject)(A), "MatMultiShift", (PetscObject *)&container)); \
    PetscCheck(container, PetscObjectComm((PetscObject)(A)), PETSC_ERR_ARG_WRONG, "The Mat is not a multi-shift matrix"); \
    PetscCall(PetscContainerGetPointer(container, (void **)&actx)); \
    PetscCall(MatStateCompareUpdate((A), &actx->state, &same)); \
    PetscCheck(same, PetscObjectComm((PetscObject)(A)), PETSC_ERR_ARG_WRONGSTATE, "The multi-shift Mat has been modified after creation"); \
  } while (0)
