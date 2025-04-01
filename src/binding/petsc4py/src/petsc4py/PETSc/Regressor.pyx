class RegressorType(object):
    """REGRESSOR solver type.

    See Also
    --------
    petsc.RegressorType

    """
    LINEAR = S_(PETSCREGRESSORLINEAR)


cdef class Regressor(Object):
    """Regression solver.

    REGRESSOR  is described in the `PETSc manual <petsc:manual/regressor>`.

    See Also
    --------
    petsc.Regressor

    """

    Type = RegressorType

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
        petsc.RegressorView

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
        Sys.getDefaultComm, petsc.RegressorCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscRegressor newregressor = NULL
        CHKERR(PetscRegressorCreate(ccomm, &newregressor))
        PetscCLEAR(self.obj); self.regressor = newregressor
        return self

    def setUp(self) -> None:
        """Set up the internal data structures for using the solver.

        Collective.

        See Also
        --------
        petsc.RegressorSetUp

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
        petsc.RegressorPredict

        """
        CHKERR(PetscRegressorFit(self.regressor, X.mat, y.vec))

    def predict(self, Mat X, Vec y) -> None:
        """Predict the regression problem.

        Collective.

        Parameters
        ----------
        X
            The matrix of unlabled observations
        y
            The vector of predicted labels

        See Also
        --------
        petsc.RegressorFit

        """
        CHKERR(PetscRegressorPredict(self.regressor, X.mat, y.vec))

    def reset(self) -> None:
        """Destroy internal data structures of REGRESSOR

        Collective.

        See Also
        --------
        petsc.RegressorDestroy

        """
        CHKERR(PetscRegressorReset(self.regressor))

    def destroy(self) -> Self:
        """Destroy the solver.

        Collective.

        See Also
        --------
        petsc.RegressorDestroy

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
        getType, petsc.RegressorSetType

        """
        cdef PetscRegressorType cval = NULL
        regressor_type = str2bytes(regressor_type, &cval)
        CHKERR(PetscRegressorSetType(self.regressor, cval))

    def getType(self) -> str:
        """Return the type of the solver.

        Not collective.

        See Also
        --------
        setType, petsc.RegressorGetType

        """
        cdef PetscRegressorType ctype = NULL
        CHKERR(PetscRegressorGetType(self.regressor, &ctype))
        return bytes2str(ctype)

del RegressorType
