#if !defined(PETSC4PY_COMPAT_REGRESSOR_H)
#define PETSC4PY_COMPAT_REGRESSOR_H
#if PetscDefined(USE_COMPLEX)

#define PetscRegressorError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"%s() not supported with complex scalars",PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP);} while (0)

PetscErrorCode PetscRegressorLinearSetFitIntercept(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscBool flag) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearSetUseKSP(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscBool flag) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetKSP(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED KSP *ksp) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetCoefficients(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Vec *vec) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetIntercept(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscScalar *intercept) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearSetType(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscRegressorLinearType type) {PetscRegressorError;}
PetscErrorCode PetscRegressorLinearGetType(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED PetscRegressorLinearType *type) {PetscRegressorError;}
PetscErrorCode PetscRegressorNLLSSetFunction(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Vec f,PETSC_UNUSED PetscRegressorNLLSFunctionFn *fn,PETSC_UNUSED void *ctx) {PetscRegressorError;}
PetscErrorCode PetscRegressorNLLSGetFunction(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Vec *f,PETSC_UNUSED PetscRegressorNLLSFunctionFn **fn,PETSC_UNUSED void **ctx) {PetscRegressorError;}
PetscErrorCode PetscRegressorNLLSSetJacobian(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Mat J,PETSC_UNUSED Mat Jpre,PETSC_UNUSED PetscRegressorNLLSJacobianFn *fn,PETSC_UNUSED void *ctx) {PetscRegressorError;}
PetscErrorCode PetscRegressorNLLSGetJacobian(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Mat *J,PETSC_UNUSED Mat *Jpre,PETSC_UNUSED PetscRegressorNLLSJacobianFn **fn,PETSC_UNUSED void **ctx) {PetscRegressorError;}
PetscErrorCode PetscRegressorNLLSSetInitialParameters(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Vec p0) {PetscRegressorError;}
PetscErrorCode PetscRegressorNLLSGetParameters(PETSC_UNUSED PetscRegressor regressor,PETSC_UNUSED Vec *p) {PetscRegressorError;}
#undef PetscRegressorError

#endif/*PETSC_USE_COMPLEX*/
#endif/*PETSC4PY_COMPAT_REGRESSOR_H*/
