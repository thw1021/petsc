#ifndef PETSC_MPICUSPARSEMATIMPL_H
#define PETSC_MPICUSPARSEMATIMPL_H

#include <cusparse_v2.h>
#include <petsc/private/veccupmimpl.h>

struct Mat_MPIAIJCUSPARSE {
  /* The following are used by GPU capabilities to store matrix storage formats on the device */
  MatCUSPARSEStorageFormat diagGPUMatFormat;
  MatCUSPARSEStorageFormat offdiagGPUMatFormat;

  /* COO stuff */
  PetscCount  *Ajmap1_d, *Aperm1_d;            /* Local entries to diag */
  PetscCount  *Bjmap1_d, *Bperm1_d;            /* Local entries to offdiag */
  PetscCount  *Aimap2_d, *Ajmap2_d, *Aperm2_d; /* Remote entries to diag */
  PetscCount  *Bimap2_d, *Bjmap2_d, *Bperm2_d; /* Remote entries to offdiag */
  PetscCount  *Cperm1_d;                       /* Permutation to fill send buffer. 'C' for communication */
  PetscScalar *sendbuf_d, *recvbuf_d;          /* Buffers for remote values in MatSetValuesCOO() */

  Mat_MPIAIJCUSPARSE()
  {
    diagGPUMatFormat    = MAT_CUSPARSE_CSR;
    offdiagGPUMatFormat = MAT_CUSPARSE_CSR;
  }

  // ATTENTION: In MatDuplicate_MPIAIJCUSPARSE() we use the default copy ctor to shallow copy all COO stuff.
  // If you add new members in this class,  you need to think over whether shallow copy is correct for the new members.
};
#endif // PETSC_MPICUSPARSEMATIMPL_H
