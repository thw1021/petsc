# --------------------------------------------------------------------

class MatPartitioningType(object):
    PARTITIONINGCURRENT  = S_(MATPARTITIONINGCURRENT)
    PARTITIONINGAVERAGE  = S_(MATPARTITIONINGAVERAGE)
    PARTITIONINGSQUARE   = S_(MATPARTITIONINGSQUARE)
    PARTITIONINGPARMETIS = S_(MATPARTITIONINGPARMETIS)
    PARTITIONINGCHACO    = S_(MATPARTITIONINGCHACO)
    PARTITIONINGPARTY    = S_(MATPARTITIONINGPARTY)
    PARTITIONINGPTSCOTCH = S_(MATPARTITIONINGPTSCOTCH)
    PARTITIONINGHIERARCH = S_(MATPARTITIONINGHIERARCH)

# --------------------------------------------------------------------

cdef class MatPartitioning(Object):

    Type = MatPartitioningType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.matpartitioning
        self.matpartitioning = NULL

    def __call__(self):
        return self.getValue()

    def view(self, Viewer viewer=None):
        assert self.obj != NULL
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatPartitioningView(self.matpartitioning, vwr) )

    def destroy(self):
        CHKERR( MatPartitioningDestroy(&self.matpartitioning) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( MatPartitioningCreate(ccomm, &self.matpartitioning) )
        return self

    def setType(self, matpartitioning_type):
        cdef PetscMatPartitioningType cval = NULL
        matpartitioning_type = str2bytes(matpartitioning_type, &cval)
        CHKERR( MatPartitioningSetType(self.matpartitioning, cval) )

    def getType(self):
        cdef PetscMatPartitioningType cval = NULL
        CHKERR( MatPartitioningGetType(self.matpartitioning, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( MatPartitioningSetFromOptions(self.matpartitioning) )

    def setAdjacency(self, Mat adj):
        CHKERR( MatPartitioningSetAdjacency(self.matpartitioning, adj.mat) )

    def apply(self, IS partitioning):
        CHKERR( MatPartitioningApply(self.matpartitioning, &partitioning.iset) )

    # def apply(self):
    #     cdef IS partitioning = IS()
    #     CHKERR( MatPartitioningApply(self.matpartitioning, &partitioning.iset) )
    #     return partitioning

# --------------------------------------------------------------------

del MatPartitioningType

# --------------------------------------------------------------------
