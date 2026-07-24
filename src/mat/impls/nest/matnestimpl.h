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

/* context for multi-shift matrices created via MatCreateMultiShift() */
struct _n_Mat_MultiShift {
  Mat          K;
  Mat          M;
  PetscInt     nshift;
  PetscScalar *sigma;
  PetscBool   *cmplx;
  MatStructure str;
};
typedef struct _n_Mat_MultiShift *Mat_MultiShift;

/*
  MatCheckMultiShift - Check that a given Mat was created via MatCreateMultiShift().
*/
#define MatCheckMultiShift(A) \
  do { \
    PetscContainer container; \
    PetscCall(PetscObjectQuery((PetscObject)A, "MatMultiShift", (PetscObject *)&container)); \
    PetscCheck(container, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "The Mat is not a multi-shift matrix"); \
  } while (0)
