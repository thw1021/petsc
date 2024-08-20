#include <petsc/private/taoimpl.h>

PETSC_EXTERN PetscErrorCode DMTaoCreate_L1_Private(DMTao);
PETSC_EXTERN PetscErrorCode DMTaoCreate_L2_Private(DMTao);
PETSC_EXTERN PetscErrorCode DMTaoCreate_Simplex_Private(DMTao);
PETSC_EXTERN PetscErrorCode DMTaoCreate_Box_Private(DMTao);
PETSC_EXTERN PetscErrorCode DMTaoCreate_Zero_Private(DMTao);
PETSC_EXTERN PetscErrorCode DMTaoCreate_Shell_Private(DMTao);
static PetscBool            DMTaoPackageInitialized = PETSC_FALSE;

/*@C
  DMTaoFinalizePackage - This function destroys everything in the `DMTao` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&DMTaoList));
  DMTaoPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoInitializePackage - This function registers the problem description
  routine in `Tao`.  When using shared or static libraries, this function is called from the
  first entry to `TaoCreate()`; when using dynamic, it is called
  from PetscDLLibraryRegister_tao()

  Level: developer

.seealso: `Tao`, `DMTao`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoInitializePackage(void)
{
  PetscFunctionBegin;
  if (DMTaoPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  DMTaoPackageInitialized = PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscClassIdRegister("DMTao", &DMTAO_CLASSID));
  PetscCall(DMTaoRegister(DMTAOL2, DMTaoCreate_L2_Private));
  PetscCall(DMTaoRegister(DMTAOL1, DMTaoCreate_L1_Private));
  PetscCall(DMTaoRegister(DMTAOSIMPLEX, DMTaoCreate_Simplex_Private));
  PetscCall(DMTaoRegister(DMTAOBOX, DMTaoCreate_Box_Private));
  PetscCall(DMTaoRegister(DMTAOZERO, DMTaoCreate_Zero_Private));
  PetscCall(DMTaoRegister(DMTAOSHELL, DMTaoCreate_Shell_Private));
  PetscCall(PetscLogEventRegister("DMTaoApplyPRox", DMTAO_CLASSID, &DMTAO_ApplyProx));
  PetscCall(PetscLogEventRegister("DMTaoObjectiveEval", DMTAO_CLASSID, &DMTAO_ObjectiveEval));
  PetscCall(PetscLogEventRegister("DMTaoGradientEval", DMTAO_CLASSID, &DMTAO_GradientEval));
  PetscCall(PetscLogEventRegister("DMTaoObjectiveGradientEval", DMTAO_CLASSID, &DMTAO_ObjGradEval));
#endif
  PetscCall(PetscRegisterFinalize(DMTaoFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}
