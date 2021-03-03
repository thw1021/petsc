#if !defined(__MPIHIPSPARSEMATIMPL)
#define __MPIHIPSPARSEMATIMPL

#include <hipsparse.h>
#include <petsc/private/hipvecimpl.h>

typedef struct {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatHIPSPARSEStorageFormat   diagGPUMatFormat;
  MatHIPSPARSEStorageFormat   offdiagGPUMatFormat;
  hipStream_t               stream;
  rocsparse_handle           handle;
  PetscSplitCSRDataStructure *deviceMat;
  PetscInt                   coo_nd,coo_no;
  THRUSTINTARRAY             *coo_p;
  THRUSTARRAY                *coo_pw;
} Mat_MPIAIJHIPSPARSE;

PETSC_INTERN PetscErrorCode MatHIPSPARSESetStream(Mat, const hipStream_t stream);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetHandle(Mat, const rocsparse_handle handle);
PETSC_INTERN PetscErrorCode MatHIPSPARSEClearHandle(Mat);

#endif
