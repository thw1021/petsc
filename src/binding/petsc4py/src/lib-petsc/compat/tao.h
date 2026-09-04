#if !defined(PETSC4PY_COMPAT_TAO_H)
#define PETSC4PY_COMPAT_TAO_H
#if PetscDefined(USE_COMPLEX)

#define PetscTaoError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode TaoSetLMVMMatrix(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat mat) {PetscTaoError;}
PetscErrorCode TaoGetLMVMMatrix(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat *mat) {PetscTaoError;}

PetscErrorCode TaoLMVMSetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat mat) {PetscTaoError;}
PetscErrorCode TaoLMVMGetH0(PETSC_UNUSED Tao tao,PETSC_UNUSED Mat *mat) {PetscTaoError;}
PetscErrorCode TaoLMVMGetH0KSP(PETSC_UNUSED Tao tao,PETSC_UNUSED KSP *ksp) {PetscTaoError;}

PetscErrorCode TaoBRGNGetSubsolver(PETSC_UNUSED Tao tao,PETSC_UNUSED Tao *subsolver) {PetscTaoError;}
PetscErrorCode TaoBRGNAddRegularizerTerm(PETSC_UNUSED Tao tao,PETSC_UNUSED const char prefix[],PETSC_UNUSED PetscReal scale,PETSC_UNUSED TaoTerm term,PETSC_UNUSED Vec parameters,PETSC_UNUSED Mat mapping) {PetscTaoError;}
PetscErrorCode TaoBRGNGetRegularizerTerm(PETSC_UNUSED Tao tao,PETSC_UNUSED TaoTerm *term) {PetscTaoError;}
PetscErrorCode TaoBRGNSetUseLM(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscBool use_lm) {PetscTaoError;}
PetscErrorCode TaoBRGNGetUseLM(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscBool *use_lm) {PetscTaoError;}
PetscErrorCode TaoBRGNSetLMLambda(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscReal lambda) {PetscTaoError;}
PetscErrorCode TaoBRGNGetLMLambda(PETSC_UNUSED Tao tao,PETSC_UNUSED PetscReal *lambda) {PetscTaoError;}

PetscErrorCode TaoBNCGSetType(PETSC_UNUSED Tao tao, PETSC_UNUSED TaoBNCGType type) {PetscTaoError;}
PetscErrorCode TaoBNCGGetType(PETSC_UNUSED Tao tao, PETSC_UNUSED TaoBNCGType *type) {PetscTaoError;}

PetscErrorCode TaoALMMGetSubsolver(PETSC_UNUSED Tao tao, PETSC_UNUSED Tao *subsolver) {PetscTaoError;}
PetscErrorCode TaoALMMSetSubsolver(PETSC_UNUSED Tao tao, PETSC_UNUSED Tao subsolver) {PetscTaoError;}
PetscErrorCode TaoALMMGetType(PETSC_UNUSED Tao tao, PETSC_UNUSED TaoALMMType *type) {PetscTaoError;}
PetscErrorCode TaoALMMSetType(PETSC_UNUSED Tao tao, PETSC_UNUSED TaoALMMType type) {PetscTaoError;}

#undef PetscTaoError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_TAO_H*/
