cdef extern from * nogil:

    ctypedef const char* PetscRegressorType "PetscRegressorType"
    PetscRegressorType PETSCREGRESSORLINEAR

    int PetscRegressorCreate(MPI_Comm,PetscRegressor*)
    int PetscRegressorReset(PetscRegressor)
    int PetscRegressorDestroy(PetscRegressor*)
    int PetscRegressorSetType(PetscRegressor,PetscRegressorType)
    int PetscRegressorSetUp(PetscRegressor)
    int PetscRegressorSetFromOptions(PetscRegressor)
    int PetscRegressorView(PetscRegressor,PetscViewer)
    int PetscRegressorFit(PetscRegressor,PetscMat,PetscVec)
    int PetscRegressorPredict(PetscRegressor,PetscMat,PetscVec)
