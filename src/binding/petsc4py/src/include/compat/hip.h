#ifndef PETSC4PY_COMPAT_HIP_H
#define PETSC4PY_COMPAT_HIP_H

#if !defined(PETSC_HAVE_HIP)

#if 0
#define PetscHIPError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() requires HIP",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

/* currently do not exist */
PetscErrorCode MatDenseHIPGetArrayRead(Mat A, const PetscScalar **gpuarray) {PetscHIPError;}
PetscErrorCode MatDenseHIPRestoreArrayRead(Mat A, const PetscScalar **gpuarray) {PetscHIPError;}
PetscErrorCode MatDenseHIPGetArrayWrite(Mat A, PetscScalar **gpuarray) {PetscHIPError;}
PetscErrorCode MatDenseHIPRestoreArrayWrite(Mat A, PetscScalar **gpuarray) {PetscHIPError;}
PetscErrorCode MatDenseHIPGetArray(Mat A, PetscScalar **gpuarray) {PetscHIPError;}
PetscErrorCode MatDenseHIPRestoreArray(Mat A, PetscScalar **gpuarray) {PetscHIPError;}

#undef PetscHIPError
#endif /* 0 */

#endif

#endif/*PETSC4PY_COMPAT_HIP_H*/
