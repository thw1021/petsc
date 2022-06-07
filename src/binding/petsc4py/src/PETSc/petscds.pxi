
ctypedef void(*PetscDSObjectiveFunction)(PetscInt, PetscInt, PetscInt,
                        const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                        const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                        PetscReal, const PetscReal*, PetscInt, const PetscScalar*, PetscScalar*)

ctypedef void(*PetscDSResidualFunction)(PetscInt, PetscInt, PetscInt,
                        const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                        const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                        PetscReal, const PetscReal*, PetscInt, const PetscScalar*, PetscScalar*)

ctypedef void (*PetscDSJacobianFunction)(PetscInt, PetscInt, PetscInt,
                        const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                        const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                        PetscReal, PetscReal, const PetscReal*, PetscInt, const PetscScalar*, PetscScalar*)
  

cdef extern from * nogil:

    ctypedef const char* PetscDSType
    PetscDSType PETSCDSBASIC

    int PetscDSCreate(MPI_Comm,PetscDS*)
    int PetscDSDestroy(PetscDS*)
    int PetscDSView(PetscDS,PetscViewer)
    int PetscDSSetType(PetscDS,PetscDSType)
    int PetscDSGetType(PetscDS,PetscDSType*)
    int PetscDSSetFromOptions(PetscDS)
    int PetscDSSetUp(PetscDS)

    int PetscDSGetHeightSubspace(PetscDS,PetscInt,PetscDS*)
    int PetscDSGetSpatialDimension(PetscDS,PetscInt*)
    int PetscDSGetCoordinateDimension(PetscDS,PetscInt*)
    int PetscDSSetCoordinateDimension(PetscDS,PetscInt)
    int PetscDSGetNumFields(PetscDS,PetscInt*)
    int PetscDSGetTotalDimension(PetscDS,PetscInt*)
    int PetscDSGetTotalComponents(PetscDS,PetscInt*)
    int PetscDSGetFieldIndex(PetscDS,PetscObject,PetscInt*)
    int PetscDSGetFieldSize(PetscDS,PetscInt,PetscInt*)
    int PetscDSGetFieldOffset(PetscDS,PetscInt,PetscInt*)
    int PetscDSGetDimensions(PetscDS,PetscInt*[])
    int PetscDSGetComponents(PetscDS,PetscInt*[])
    int PetscDSGetComponentOffset(PetscDS,PetscInt,PetscInt*)
    int PetscDSGetComponentOffsets(PetscDS,PetscInt*[])
    int PetscDSGetComponentDerivativeOffsets(PetscDS,PetscInt*[])

    int PetscDSGetDiscretization(PetscDS,PetscInt,PetscObject*)
    int PetscDSSetDiscretization(PetscDS,PetscInt,PetscObject)
    int PetscDSAddDiscretization(PetscDS,PetscObject)
    int PetscDSGetImplicit(PetscDS,PetscInt,PetscBool*)
    int PetscDSSetImplicit(PetscDS,PetscInt,PetscBool)

    int PetscDSSetObjective(PetscDS,PetscInt,PetscDSObjectiveFunction*)
    int PetscDSSetJacobian(PetscDS,PetscInt,PetscInt,PetscDSJacobianFunction,PetscDSJacobianFunction,PetscDSJacobianFunction,PetscDSJacobianFunction)
    int PetscDSSetJacobianPreconditioner(PetscDS,PetscInt,PetscInt,PetscDSJacobianFunction,PetscDSJacobianFunction,PetscDSJacobianFunction,PetscDSJacobianFunction)
    int PetscDSSetResidual(PetscDS,PetscInt,PetscDSResidualFunction,PetscDSResidualFunction)


cdef class DS_Objective:
    cdef PetscDSObjectiveFunction fn

cdef class DS_Residual:
    cdef PetscDSResidualFunction fn

cdef class DS_Jacobian:
    cdef PetscDSJacobianFunction fn