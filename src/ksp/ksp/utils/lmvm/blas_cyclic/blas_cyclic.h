#pragma once

#include <petscvec.h>
#include <petscmat.h>

PETSC_INTERN PetscLogEvent AXPBY_Cyc;
PETSC_INTERN PetscLogEvent DMV_Cyc;
PETSC_INTERN PetscLogEvent DSV_Cyc;
PETSC_INTERN PetscLogEvent TRSV_Cyc;
PETSC_INTERN PetscLogEvent GEMV_Cyc;
PETSC_INTERN PetscLogEvent HEMV_Cyc;

/* These methods are intended for small vectors / dense matrices that are either sequential or have all of their entries on the first rank */
// y <- alpha * x + beta * y
PETSC_INTERN PetscErrorCode VecAXPBYCyclic(PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Vec);
// y <- alpha * (A .* x) + beta * y
PETSC_INTERN PetscErrorCode VecDMVCyclic(PetscBool, PetscInt, PetscInt, PetscScalar, Vec, Vec, PetscScalar, Vec);
// y <- (x ./ A)
PETSC_INTERN PetscErrorCode VecDSVCyclic(PetscBool, PetscInt, PetscInt, Vec, Vec, Vec);
// y <- (triu(A) \ x)
PETSC_INTERN PetscErrorCode MatSeqDenseTRSVCyclic(PetscBool, PetscInt, PetscInt, Mat, Vec, Vec);
// y <- alpha * A * x + beta * y
PETSC_INTERN PetscErrorCode MatSeqDenseGEMVCyclic(PetscBool, PetscInt, PetscInt, PetscScalar, Mat, Vec, PetscScalar, Vec);
// y <- alpha * symm(A) * x + beta * y   [sym(A) = triu(A) + striu(A)^H]
PETSC_INTERN PetscErrorCode MatSeqDenseHEMVCyclic(PetscInt, PetscInt, PetscScalar, Mat, Vec, PetscScalar, Vec);
// A[i,:] <- alpha * x + beta * A[i,:]
PETSC_INTERN PetscErrorCode MatSeqDenseRowAXPBYCyclic(PetscInt, PetscInt, PetscScalar, Vec, PetscScalar, Mat, PetscInt);

/* These methods are intended for for tall matrices of column vectors where we would like to compute products with ranges of the
   vectors.  The column layout should place all columns on the first rank.  Because they may involve MPI communication,
   they rely on implementations from the dense matrix backends */
PETSC_INTERN PetscErrorCode MatMultColumnRange(Mat, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultAddColumnRange(Mat, Vec, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultHermitianTransposeColumnRange(Mat, Vec, Vec, PetscInt, PetscInt);
PETSC_INTERN PetscErrorCode MatMultHermitianTransposeAddColumnRange(Mat, Vec, Vec, Vec, PetscInt, PetscInt);
