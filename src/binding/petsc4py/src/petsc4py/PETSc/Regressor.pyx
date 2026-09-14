class RegressorType(object):
    """REGRESSOR solver type.

    See Also
    --------
    petsc.PetscRegressorType

    """
    LINEAR = S_(PETSCREGRESSORLINEAR)
    NLLS = S_(PETSCREGRESSORNLLS)


class RegressorLinearType(object):
    """Linear regressor type.

    See Also
    --------
    petsc.PetscRegressorLinearType
    """
    OLS   = REGRESSOR_LINEAR_OLS
    LASSO = REGRESSOR_LINEAR_LASSO
    RIDGE = REGRESSOR_LINEAR_RIDGE


cdef class Regressor(Object):
    """Regression solver.

    REGRESSOR  is described in the `PETSc manual <petsc:manual/regressor>`.

    See Also
    --------
    petsc.PetscRegressor

    """

    Type = RegressorType
    LinearType = RegressorLinearType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.regressor
        self.regressor = NULL

    def view(self, Viewer viewer=None) -> None:
        """View the solver.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.PetscRegressorView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR(PetscRegressorView(self.regressor, cviewer))

    def create(self, comm=None) -> Self:
        """Create a REGRESSOR solver.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscRegressorCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscRegressor newregressor = NULL
        CHKERR(PetscRegressorCreate(ccomm, &newregressor))
        PetscCLEAR(self.obj); self.regressor = newregressor
        return self

    def setRegularizerWeight(self, weight: float) -> None:
        """Set the weight to be used for the regularizer.

        Logically collective.

        See Also
        --------
        setType, petsc.PetscRegressorSetRegularizerWeight
        """
        CHKERR(PetscRegressorSetRegularizerWeight(self.regressor, weight))

    def setFromOptions(self) -> None:
        """Configure the solver from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscRegressorSetFromOptions
        """
        CHKERR(PetscRegressorSetFromOptions(self.regressor))

    def setUp(self) -> None:
        """Set up the internal data structures for using the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorSetUp

        """
        CHKERR(PetscRegressorSetUp(self.regressor))

    def fit(self, Mat X, Vec y) -> None:
        """Fit the regression problem.

        Collective.

        Parameters
        ----------
        X
            The matrix of training data
        y
            The vector of target values from the training dataset

        See Also
        --------
        petsc.PetscRegressorPredict

        """
        CHKERR(PetscRegressorFit(self.regressor, X.mat, y.vec))

    def predict(self, Mat X, Vec y) -> None:
        """Predict the regression problem.

        Collective.

        Parameters
        ----------
        X
            The matrix of unlabeled observations
        y
            The vector of predicted labels

        See Also
        --------
        petsc.PetscRegressorFit

        """
        CHKERR(PetscRegressorPredict(self.regressor, X.mat, y.vec))

    def getTAO(self) -> TAO:
        """Return the underlying `TAO` object .

        Not collective.

        See Also
        --------
        getLinearKSP, petsc.PetscRegressorGetTao
        """
        cdef TAO tao = TAO()
        CHKERR(PetscRegressorGetTao(self.regressor, &tao.tao))
        CHKERR(PetscINCREF(tao.obj))
        return tao

    def reset(self) -> None:
        """Destroy internal data structures of the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorDestroy

        """
        CHKERR(PetscRegressorReset(self.regressor))

    def destroy(self) -> Self:
        """Destroy the solver.

        Collective.

        See Also
        --------
        petsc.PetscRegressorDestroy

        """
        CHKERR(PetscRegressorDestroy(&self.regressor))
        return self

    def setType(self, regressor_type: Type | str) -> None:
        """Set the type of the solver.

        Logically collective.

        Parameters
        ----------
        regressor_type
            The type of the solver.

        See Also
        --------
        getType, petsc.PetscRegressorSetType

        """
        cdef PetscRegressorType cval = NULL
        regressor_type = str2bytes(regressor_type, &cval)
        CHKERR(PetscRegressorSetType(self.regressor, cval))

    def getType(self) -> str:
        """Return the type of the solver.

        Not collective.

        See Also
        --------
        setType, petsc.PetscRegressorGetType

        """
        cdef PetscRegressorType ctype = NULL
        CHKERR(PetscRegressorGetType(self.regressor, &ctype))
        return bytes2str(ctype)

    # --- Linear ---

    def setLinearFitIntercept(self, flag: bool) -> None:
        """Set a flag to indicate that the intercept should be calculated.

        Logically collective.

        See Also
        --------
        petsc.PetscRegressorLinearSetFitIntercept
        """
        cdef PetscBool fitintercept = flag
        CHKERR(PetscRegressorLinearSetFitIntercept(self.regressor, fitintercept))

    def setLinearUseKSP(self, flag: bool) -> None:
        """Set a flag to indicate that `KSP` instead of `TAO` solvers should be used.

        Logically collective.

        See Also
        --------
        petsc.PetscRegressorLinearSetUseKSP
        """
        cdef PetscBool useksp = flag
        CHKERR(PetscRegressorLinearSetUseKSP(self.regressor, useksp))

    def getLinearKSP(self) -> KSP:
        """Returns the `KSP` context used by the linear regressor.

        Not collective.

        See Also
        --------
        petsc.PetscRegressorLinearGetKSP
        """
        cdef KSP ksp = KSP()
        CHKERR(PetscRegressorLinearGetKSP(self.regressor, &ksp.ksp))
        CHKERR(PetscINCREF(ksp.obj))
        return ksp

    def getLinearCoefficients(self) -> Vec:
        """Get a vector of the fitted coefficients from a linear regression model.

        Not collective.

        See Also
        --------
        getLinearIntercept, petsc.PetscRegressorLinearGetCoefficients
        """
        cdef Vec coeffs = Vec()
        CHKERR(PetscRegressorLinearGetCoefficients(self.regressor, &coeffs.vec))
        CHKERR(PetscINCREF(coeffs.obj))
        return coeffs

    def getLinearIntercept(self) -> Scalar:
        """Get the intercept from a linear regression model.

        Not collective.

        See Also
        --------
        setLinearFitIntercept, petsc.PetscRegressorLinearGetIntercept
        """
        cdef PetscScalar intercept = 0.0
        CHKERR(PetscRegressorLinearGetIntercept(self.regressor, &intercept))
        return toScalar(intercept)

    def setLinearType(self, lineartype: RegressorLinearType) -> None:
        """Set the type of linear regression to be performed.

        Logically collective.

        See Also
        --------
        getLinearType, petsc.PetscRegressorLinearSetType
        """
        CHKERR(PetscRegressorLinearSetType(self.regressor, lineartype))

    def getLinearType(self) -> RegressorLinearType:
        """Return the type of the linear regressor.

        Not collective.

        See Also
        --------
        setLinearType, petsc.PetscRegressorLinearGetType
        """
        cdef PetscRegressorLinearType cval = REGRESSOR_LINEAR_OLS
        CHKERR(PetscRegressorLinearGetType(self.regressor, &cval))
        return cval


    # --- NLLS ---

    def setNLLSFunction(self, function: RegressorNLLSFunction,
                        Vec f=None,
                        args: tuple[Any, ...] | None = None,
                        kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the nonlinear least-squares model values.

        Logically collective.

        Parameters
        ----------
        function
            The callback evaluating the model $f(X, p)$.
        f
            An optional vector to store the model evaluations. If omitted, a
            matching vector is created from the target vector at setup time.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        Notes
        -----
        The callback receives the model values rather than the residual; the
        implementation forms the residual $f(X, p) - y$ internally. This
        matches the ``scipy.optimize.curve_fit`` convention.

        See Also
        --------
        getNLLSFunction, petsc.PetscRegressorNLLSSetFunction

        """
        cdef PetscVec fvec = NULL
        if f is not None: fvec = f.vec
        if function is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (function, args, kargs)
            self.set_attr("__nlls_function__", context)
            CHKERR(PetscRegressorNLLSSetFunction(self.regressor, fvec, Regressor_NLLSFunction, <void*>context))
        else:
            if f is None:
                self.set_attr("__nlls_function__", None)
            CHKERR(PetscRegressorNLLSSetFunction(self.regressor, fvec, NULL, NULL))

    def getNLLSFunction(self) -> tuple[Vec, RegressorNLLSFunction]:
        """Return the model function set with :meth:`setNLLSFunction`.

        Not collective.

        See Also
        --------
        setNLLSFunction, petsc.PetscRegressorNLLSGetFunction

        """
        cdef Vec f = Vec()
        CHKERR(PetscRegressorNLLSGetFunction(self.regressor, &f.vec, NULL, NULL))
        CHKERR(PetscINCREF(f.obj))
        cdef object function = self.get_attr("__nlls_function__")
        if function is not None:
            return (f, function)
        return (f, None)

    def setNLLSJacobian(self, jacobian: RegressorNLLSJacobianFunction,
                        Mat J=None,
                        Mat P=None,
                        args: tuple[Any, ...] | None = None,
                        kargs: dict[str, Any] | None = None) -> None:
        """Set the callback to compute the Jacobian of the model values.

        Logically collective.

        Parameters
        ----------
        jacobian
            The Jacobian callback.
        J
            The matrix to store the Jacobian.
        P
            The matrix to construct the preconditioner. Defaults to ``J``.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        Notes
        -----
        Omitting the Jacobian is supported; a forward-difference Jacobian is
        then computed by the underlying solver.

        See Also
        --------
        getNLLSJacobian, petsc.PetscRegressorNLLSSetJacobian

        """
        cdef PetscMat Jmat = NULL
        if J is not None: Jmat = J.mat
        cdef PetscMat Pmat = Jmat
        if P is not None: Pmat = P.mat
        if jacobian is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (jacobian, args, kargs)
            self.set_attr("__nlls_jacobian__", context)
            CHKERR(PetscRegressorNLLSSetJacobian(self.regressor, Jmat, Pmat, Regressor_NLLSJacobian, <void*>context))
        else:
            if J is None:
                self.set_attr("__nlls_jacobian__", None)
            CHKERR(PetscRegressorNLLSSetJacobian(self.regressor, Jmat, Pmat, NULL, NULL))

    def getNLLSJacobian(self) -> tuple[Mat, Mat, RegressorNLLSJacobianFunction]:
        """Return the matrices and callback used to compute the Jacobian.

        Not collective.

        See Also
        --------
        setNLLSJacobian, petsc.PetscRegressorNLLSGetJacobian

        """
        cdef Mat J = Mat()
        cdef Mat P = Mat()
        CHKERR(PetscRegressorNLLSGetJacobian(self.regressor, &J.mat, &P.mat, NULL, NULL))
        CHKERR(PetscINCREF(J.obj))
        CHKERR(PetscINCREF(P.obj))
        cdef object jacobian = self.get_attr("__nlls_jacobian__")
        return (J, P, jacobian)

    def setNLLSInitialParameters(self, Vec p0) -> None:
        """Set the initial guess for the parameters.

        Logically collective.

        Parameters
        ----------
        p0
            Vector containing the initial parameter values. Its length
            determines the number of model parameters. Must be called before
            :meth:`fit`.

        See Also
        --------
        getNLLSParameters, petsc.PetscRegressorNLLSSetInitialParameters

        """
        CHKERR(PetscRegressorNLLSSetInitialParameters(self.regressor, p0.vec))

    def getNLLSParameters(self) -> Vec:
        """Return the vector of fitted parameters.

        Not collective, but the vector is parallel if the regressor is.

        See Also
        --------
        setNLLSInitialParameters, petsc.PetscRegressorNLLSGetParameters

        """
        cdef Vec parameters = Vec()
        CHKERR(PetscRegressorNLLSGetParameters(self.regressor, &parameters.vec))
        CHKERR(PetscINCREF(parameters.obj))
        return parameters

del RegressorType
