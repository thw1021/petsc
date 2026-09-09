cdef extern from * nogil:

    ctypedef const char* PetscRegressorType "PetscRegressorType"
    PetscRegressorType PETSCREGRESSORLINEAR
    PetscRegressorType PETSCREGRESSORNLLS

    PetscErrorCode PetscRegressorCreate(MPI_Comm, PetscRegressor*)
    PetscErrorCode PetscRegressorReset(PetscRegressor)
    PetscErrorCode PetscRegressorDestroy(PetscRegressor*)
    PetscErrorCode PetscRegressorSetType(PetscRegressor, PetscRegressorType)
    PetscErrorCode PetscRegressorGetType(PetscRegressor, PetscRegressorType*)
    PetscErrorCode PetscRegressorSetRegularizerWeight(PetscRegressor, PetscReal)
    PetscErrorCode PetscRegressorSetUp(PetscRegressor)
    PetscErrorCode PetscRegressorSetFromOptions(PetscRegressor)
    PetscErrorCode PetscRegressorView(PetscRegressor, PetscViewer)
    PetscErrorCode PetscRegressorFit(PetscRegressor, PetscMat, PetscVec)
    PetscErrorCode PetscRegressorPredict(PetscRegressor, PetscMat, PetscVec)
    PetscErrorCode PetscRegressorGetTao(PetscRegressor, PetscTAO*)

    PetscErrorCode PetscRegressorLinearSetFitIntercept(PetscRegressor, PetscBool)
    PetscErrorCode PetscRegressorLinearSetUseKSP(PetscRegressor, PetscBool)
    PetscErrorCode PetscRegressorLinearGetKSP(PetscRegressor, PetscKSP*)
    PetscErrorCode PetscRegressorLinearGetCoefficients(PetscRegressor, PetscVec*)
    PetscErrorCode PetscRegressorLinearGetIntercept(PetscRegressor, PetscScalar*)
    PetscErrorCode PetscRegressorLinearSetType(PetscRegressor, PetscRegressorLinearType)
    PetscErrorCode PetscRegressorLinearGetType(PetscRegressor, PetscRegressorLinearType*)

    ctypedef enum PetscRegressorLinearType:
        REGRESSOR_LINEAR_OLS
        REGRESSOR_LINEAR_LASSO
        REGRESSOR_LINEAR_RIDGE

    ctypedef PetscErrorCode PetscRegressorNLLSFunctionFn(PetscRegressor, PetscMat, PetscVec, PetscVec, void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode PetscRegressorNLLSJacobianFn(PetscRegressor, PetscMat, PetscVec, PetscMat, PetscMat, void*) except PETSC_ERR_PYTHON

    PetscErrorCode PetscRegressorNLLSSetFunction(PetscRegressor, PetscVec, PetscRegressorNLLSFunctionFn*, void*)
    PetscErrorCode PetscRegressorNLLSGetFunction(PetscRegressor, PetscVec*, PetscRegressorNLLSFunctionFn**, void**)
    PetscErrorCode PetscRegressorNLLSSetJacobian(PetscRegressor, PetscMat, PetscMat, PetscRegressorNLLSJacobianFn*, void*)
    PetscErrorCode PetscRegressorNLLSGetJacobian(PetscRegressor, PetscMat*, PetscMat*, PetscRegressorNLLSJacobianFn**, void**)
    PetscErrorCode PetscRegressorNLLSSetInitialParameters(PetscRegressor, PetscVec)
    PetscErrorCode PetscRegressorNLLSGetParameters(PetscRegressor, PetscVec*)

# --------------------------------------------------------------------

cdef inline Regressor ref_Regressor(PetscRegressor regressor):
    cdef Regressor ob = <Regressor> Regressor()
    ob.regressor = regressor
    CHKERR(PetscINCREF(ob.obj))
    return ob

# --------------------------------------------------------------------

cdef PetscErrorCode Regressor_NLLSFunction(PetscRegressor _regressor,
                                           PetscMat _X,
                                           PetscVec _p,
                                           PetscVec _f,
                                           void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef Regressor regressor = ref_Regressor(_regressor)
    cdef Mat X = ref_Mat(_X)
    cdef Vec p = ref_Vec(_p)
    cdef Vec f = ref_Vec(_f)
    context = regressor.get_attr("__nlls_function__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(regressor, X, p, f, *args, **kargs)
    return PETSC_SUCCESS

cdef PetscErrorCode Regressor_NLLSJacobian(PetscRegressor _regressor,
                                           PetscMat _X,
                                           PetscVec _p,
                                           PetscMat _J,
                                           PetscMat _P,
                                           void* ctx) except PETSC_ERR_PYTHON with gil:
    cdef Regressor regressor = ref_Regressor(_regressor)
    cdef Mat X = ref_Mat(_X)
    cdef Vec p = ref_Vec(_p)
    cdef Mat J = ref_Mat(_J)
    cdef Mat P = ref_Mat(_P)
    context = regressor.get_attr("__nlls_jacobian__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(regressor, X, p, J, P, *args, **kargs)
    return PETSC_SUCCESS

# --------------------------------------------------------------------
