
#include <petscksp.h>

typedef struct _n_PetscMatrix *PetscMatrix;

struct _n_PetscMatrix {
  Mat mat;
  KSP ksp;
};

/*@C
   PetscMatrixCreate - creates a PETSc matrix

   Input Parameters:
+  nz - the number of nonzeros in the matrix
.  i - the row indices of the matrix
.  j - the column indices of the matrix
.  a - the nonzero values of the matrix
-  options - options for the matrix and solver

   Returns the matrix

   Level: basic

   Developer Note:
   Support indices starting at zero?

.seealso: `PetscMatrixDestroy()`, `PetscMatrixUpdateValues()`, `PetscMatrixSolve()`
@*/
PetscMatrix PetscMatrixCreate(PetscInt nz, const PetscInt *i, const PetscInt *j, PetscScalar *a, const char *options)
{
  PetscMatrix matrix;
  PetscInt    m = 0, n = 0;

  PetscFunctionBegin;
  PetscCallNull(PetscNew(&matrix));
  PetscCallNull(MatCreate(PETSC_COMM_SELF, &matrix->mat));
  PetscCallNull(KSPCreate(PETSC_COMM_SELF, &matrix->ksp));
  for (PetscInt k = 0; k < nz; k++) {
    m = PetscMax(m, i[k]);
    n = PetscMax(n, j[k]);
  }
  PetscCallNull(MatSetSizes(matrix->mat, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscFunctionReturn(matrix);
}

/*@C
   PetscMatrixSolve - solves a linear system with the given matrix

   Input Parameters:
+  matrix - the matrix
-  rhs - the right hand side of the linear system

   Output Parameter:
.  x - the solution

   Level: basic

.seealso: `PetscMatrixCreate()`, `PetscMatrixUpdateValues()`, `PetscMatrixDestroy()`
@*/
PetscErrorCode PetscMatrixSolve(PetscMatrix matrix, PetscScalar *rhs, PetscScalar *x)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscMatrixUpdateValues - update the numerical values in a PETSc matrix

   Input Parameters:
+  matrix - the matrix
-  a - the new numerical values, row and column indices must match those previously provided

   Level: basic

.seealso: `PetscMatrixCreate()`, `PetscMatrixUpdateValues()`, `PetscMatrixDestroy()`
@*/
PetscErrorCode PetscMatrixUpdateValues(PetscMatrix matrix, PetscScalar *a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscMatrixDestroy - destroys a PETSc matrix

   Input Parameter:
.  matrix - the matrix

   Level: basic

.seealso: `PetscMatrixCreate()`, `PetscMatrixUpdateValues()`, `PetscMatrixSolve()`
@*/
PetscErrorCode PetscMatrixDestroy(PetscMatrix matrix)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&matrix->mat));
  PetscCall(KSPDestroy(&matrix->ksp));
  PetscCall(PetscFree(matrix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
