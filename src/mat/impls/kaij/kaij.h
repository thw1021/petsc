#pragma once

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define KAIJHEADER \
  PetscInt         p, q; \
  Mat              AIJ; \
  Mat              B; /* second AIJ operand for the (B \otimes S) term; NULL means B is the identity */ \
  PetscScalar     *S; \
  PetscScalar     *T; \
  PetscScalar     *ibdiag; \
  PetscObjectState ibdiagstate;  /* state of AIJ when ibdiag was last computed */ \
  PetscObjectState ibdiagbstate; /* state of B when ibdiag was last computed */ \
  PetscObjectState ibdiagkstate; /* state of the KAIJ matrix itself, which MatKAIJRestoreS() and MatKAIJRestoreT() increase, when ibdiag was last computed */ \
  PetscObjectState aijnnzstate;  /* nonzero state of the AIJ operand at the last assembly of the KAIJ matrix; MatAssemblyEnd_KAIJ() forwards operand nonzero-pattern changes to the KAIJ matrix */ \
  PetscObjectState bnnzstate;    /* nonzero state of the B operand at the last assembly of the KAIJ matrix */ \
  PetscBool        ibdiagvalid, getrowactive, isTI; \
  struct { \
    PetscBool    setup; \
    PetscScalar *w, *work, *t, *arr, *y; \
  } sor;

typedef struct {
  KAIJHEADER
} Mat_SeqKAIJ;

typedef struct {
  KAIJHEADER
  Mat              OAIJ;    /* sequential KAIJ matrix that corresponds to off-diagonal matrix entries (diagonal entries are stored in 'AIJ') */
  Mat              A;       /* AIJ matrix describing the blockwise action of the KAIJ matrix; compare with struct member 'AIJ' in sequential case */
  Mat              OB;      /* sequential KAIJ matrix for the (B \otimes S) action of the off-diagonal entries of B; B must share the nonzero structure of A so the same ghost scatter applies */
  VecScatter       ctx;     /* update ghost points for parallel case */
  Vec              w;       /* work space for ghost values for parallel case */
  PetscInt        *ghosts;  /* copy of the off-diagonal ghost columns of A that 'ctx' and 'w' were built from */
  PetscInt         nghosts; /* length of 'ghosts' */
  PetscObjectState state;   /* state of the matrix A when AIJ and OAIJ were last updated */
  PetscObjectState bstate;  /* state of the matrix B when the B-related submatrices were last updated */
  PetscObjectState kstate;  /* state of the KAIJ matrix itself, which MatKAIJRestoreS() and MatKAIJRestoreT() increase, when the submatrices were last updated; the submatrices hold copies of S and T */
} Mat_MPIKAIJ;
