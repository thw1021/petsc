#if !defined(PETSCCUBLAS_H)
#define PETSCCUBLAS_H

#include <petscdevice.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CHKERRCUBLAS(stat) \
do { \
   if (PetscUnlikely(stat)) { \
      const char *name = PetscCUBLASGetErrorName(stat); \
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuBLAS error %d (%s)",(int)stat,name); \
   } \
} while (0)

PETSC_INTERN PetscErrorCode PetscCUBLASInitializeHandle(void);
PETSC_INTERN PetscErrorCode PetscCUSOLVERDnInitializeHandle(void);

/* cuBLAS does not have cublasGetErrorName(). We create one on our own. */
PETSC_EXTERN const char* PetscCUBLASGetErrorName(cublasStatus_t); /* PETSC_EXTERN since it is exposed by the CHKERRCUBLAS macro */
PETSC_EXTERN PetscErrorCode PetscCUBLASGetHandle(cublasHandle_t*);
PETSC_EXTERN PetscErrorCode PetscCUSOLVERDnGetHandle(cusolverDnHandle_t*);

#endif
