#define TAOREGULARIZER_DLL
#include <petsc/private/taoregularizerimpl.h>

PETSC_EXTERN PetscErrorCode TaoRegularizerCreate_L1(TaoRegularizer);
PETSC_EXTERN PetscErrorCode TaoRegularizerCreate_L2(TaoRegularizer);
PETSC_EXTERN PetscErrorCode TaoRegularizerCreate_KL(TaoRegularizer);
static PetscBool            TaoRegularizerPackageInitialized = PETSC_FALSE;

/*@C
  TaoRegularizerFinalizePackage - This function destroys everything in the `TaoRegularizer` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: `Tao`, `TaoRegularizer`
@*/
PetscErrorCode TaoRegularizerFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoRegularizerList));
  TaoRegularizerPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoRegularizerInitializePackage - This function registers the regularizer
  algorithms in `Tao`.  When using shared or static libraries, this function is called from the
  first entry to `TaoCreate()`; when using dynamic, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: `Tao`, `TaoRegularizer`, `TaoRegularizerCreate()`
@*/
PetscErrorCode TaoRegularizerInitializePackage(void)
{
  PetscFunctionBegin;
  if (TaoRegularizerPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  TaoRegularizerPackageInitialized = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscClassIdRegister("TaoRegularizer", &TAOREGULARIZER_CLASSID));
  PetscCall(TaoRegularizerRegister("l1", TaoRegularizerCreate_L1));
  PetscCall(TaoRegularizerRegister("l2", TaoRegularizerCreate_L2));
  PetscCall(TaoRegularizerRegister("kl", TaoRegularizerCreate_KL));
  PetscCall(PetscLogEventRegister("TaoRegEval", TAOREGULARIZER_CLASSID, &TAOREGULARIZER_Eval));
#endif
  PetscCall(PetscRegisterFinalize(TaoRegularizerFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
