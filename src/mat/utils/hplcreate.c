#include <petsc/private/matimpl.h> /*I "petscmat.h"  I*/


/*@
  MatSetHPL - fills a `MATSEQDENSE` matrix using the HPL 2.3 random matrix generation routine

  Collective

  Input Parameters:
+ A     - the matrix
- iseed - the random number seed

.seealso: [](ch_matrices), `Mat`, `MatCreate()`
@*/
PetscErrorCode MatSetHPL(Mat A,int iseed)
{
  PetscBool isDense;
  PetscInt M,N,LDA;
  PetscBLASInt bM,bN,bLDA;
  PetscScalar *values;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isDense));
  PetscCheck(isDense,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only supports sequential dense matrices");
  PetscCall(MatGetSizes(A,NULL,NULL,&M,&N));
  PetscCall(PetscBLASIntCast(M,&bM));
  PetscCall(PetscBLASIntCast(N,&bN));
  PetscCall(MatDenseGetLDA(A,&LDA));
  PetscCall(PetscBLASIntCast(LDA,&bLDA));
  PetscCall(MatDenseGetArrayWrite(A,&values));
  PetscStackCallExternalVoid(HPL_dmatgen(bM,bN,values,bLDA,iseed));
  PetscCall(MatDenseRestoreArrayWrite(A,&values));
  PetscFunctionReturn(PETSC_SUCCESS);
}
