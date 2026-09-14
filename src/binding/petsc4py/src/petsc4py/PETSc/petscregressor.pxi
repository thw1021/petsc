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

    # NLLS callbacks. The C header declares the setters as
    # `(PetscRegressor, ..., PetscRegressorNLLSFunctionFn *, void *)` where the
    # `Fn` is a function *type* typedef, so the old-style K&R parameter `Fn *`
    # is a C function pointer. Cython models it the same way SNESSetJacobian
    # models `SNESJacobianFunction`: a by-value function pointer carrying
    # `except PETSC_ERR_PYTHON`, so the C `PetscErrorCode(PetScRegressor,
    # PetscMat, ...)` C function pointers match the `with gil` trampolines
    # below (which are `nogil` at the C ABI because they acquire the GIL).
    ctypedef PetscErrorCode (*PetscRegressorNLLSModelFunction)(PetscRegressor,
                                                            PetscMat,
                                                            PetscVec,
                                                            PetscVec,
                                                            void*) except PETSC_ERR_PYTHON
    ctypedef PetscErrorCode (*PetscRegressorNLLSJacobianFunction)(PetscRegressor,
                                                            PetscMat,
                                                            PetscVec,
                                                            PetscMat,
                                                            PetscMat,
                                                            void*) except PETSC_ERR_PYTHON

    PetscErrorCode PetscRegressorNLLSSetFunction(PetscRegressor, PetscVec,
                                                 PetscRegressorNLLSModelFunction, void*)
    PetscErrorCode PetscRegressorNLLSGetFunction(PetscRegressor, PetscVec*,
                                                 PetscRegressorNLLSModelFunction*, void**)
    PetscErrorCode PetscRegressorNLLSSetJacobian(PetscRegressor, PetscMat, PetscMat,
                                                 PetscRegressorNLLSJacobianFunction, void*)
    PetscErrorCode PetscRegressorNLLSGetJacobian(PetscRegressor, PetscMat*, PetscMat*,
                                                 PetscRegressorNLLSJacobianFunction*, void**)
    PetscErrorCode PetscRegressorNLLSSetInitialParameters(PetscRegressor, PetscVec)
    PetscErrorCode PetscRegressorNLLSGetParameters(PetscRegressor, PetscVec*)

    ctypedef enum PetscRegressorLinearType:
        REGRESSOR_LINEAR_OLS
        REGRESSOR_LINEAR_LASSO
        REGRESSOR_LINEAR_RIDGE

# -------------------------------------------------------------------

cdef inline Regressor ref_Regressor(PetscRegressor regressor):
    cdef Regressor ob = <Regressor> Regressor()
    ob.regressor = regressor
    CHKERR(PetscINCREF(ob.obj))
    return ob

# -------------------------------------------------------------------

cdef PetscErrorCode Regressor_NLLSFunction(PetscRegressor _regressor,
                                           PetscMat _X,
                                           PetscVec _p,
                                           PetscVec _f,
                                           void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef Regressor regressor = ref_Regressor(_regressor)
    cdef Mat Xmat = ref_Mat(_X)
    cdef Vec pvec = ref_Vec(_p)
    cdef Vec fvec = ref_Vec(_f)
    cdef object context = regressor.get_attr("__nlls_function__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(regressor, Xmat, pvec, fvec, *args, **kargs)
    return PETSC_SUCCESS

# -------------------------------------------------------------------

cdef PetscErrorCode Regressor_NLLSJacobian(PetscRegressor _regressor,
                                           PetscMat _X,
                                           PetscVec _p,
                                           PetscMat _J,
                                           PetscMat _Jpre,
                                           void *ctx) except PETSC_ERR_PYTHON with gil:
    cdef Regressor regressor = ref_Regressor(_regressor)
    cdef Mat Xmat = ref_Mat(_X)
    cdef Vec pvec = ref_Vec(_p)
    cdef Mat Jmat = ref_Mat(_J)
    cdef Mat Jpremat = ref_Mat(_Jpre)
    cdef object context = regressor.get_attr("__nlls_jacobian__")
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(regressor, Xmat, pvec, Jmat, Jpremat, *args, **kargs)
    return PETSC_SUCCESS
