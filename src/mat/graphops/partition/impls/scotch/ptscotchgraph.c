/*
  PT-SCOTCH graph partitioning shared by the MATPARTITIONINGPTSCOTCH and PETSCPARTITIONERPTSCOTCH
  implementations.
*/

#include <petsc/private/matimpl.h>

EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END

/*
  PetscPTScotchPartitionGraph_Private - Partitions a graph distributed over comm with PT-SCOTCH

  Collective

  Input Parameters:
+ comm       - the communicator the graph is distributed over
. nloc       - number of vertices owned by this process, which may be zero
. xadj       - adjacency offsets of the local vertices, of length nloc + 1
. adjncy     - adjacency of the local vertices, in global vertex numbers
. vwgt       - weight of each local vertex, or NULL for unit weights
. adjwgt     - weight of each local edge, or NULL for unit weights
. nparts     - number of parts to partition into
. tpwgts     - target weight of each part, of length nparts, or NULL for equally weighted parts
. strategy   - PT-SCOTCH strategy flags
- imbalance  - allowed load imbalance ratio

  Output Parameter:
. assignment - part each local vertex is assigned to, of length nloc

  Notes:
  PT-SCOTCH requires every process of the communicator it is given to own at least one vertex, and
  SCOTCH_dgraphBuild() requires those processes to agree on whether each optional weight array is
  null, which a process owning nothing cannot do because its zero-size allocation returns null.
  Both requirements are met by partitioning on the processes that own vertices; a process owning
  none has nothing to assign and returns without calling PT-SCOTCH. The graph itself is unchanged,
  so the partition does not depend on how many processes own no vertices.

  The weight arrays are passed to PT-SCOTCH as SCOTCH_Num, which PETSc configures to match
  PetscInt.

.seealso: `PetscCommCreateNonempty()`, `MATPARTITIONINGPTSCOTCH`, `PETSCPARTITIONERPTSCOTCH`
*/
PetscErrorCode PetscPTScotchPartitionGraph_Private(MPI_Comm comm, PetscInt nloc, PetscInt xadj[], PetscInt adjncy[], PetscInt vwgt[], PetscInt adjwgt[], PetscInt nparts, PetscInt tpwgts[], PetscInt strategy, double imbalance, PetscInt assignment[])
{
  MPI_Comm     gcomm;
  SCOTCH_Arch  archdat;
  SCOTCH_Strat stradat;
  SCOTCH_Num   vertnbr = nloc, edgenbr, archnbr;
  SCOTCH_Num  *velotab = (SCOTCH_Num *)vwgt, *edlotab = (SCOTCH_Num *)adjwgt, *edlotabempty = NULL;
  double       kbalval = imbalance;
  PetscBool    hasweights[2];
  PetscMPIInt  size;

  PetscFunctionBegin;
  PetscCall(PetscCommCreateNonempty(comm, (PetscBool)(nloc == 0), &gcomm));
  if (gcomm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_size(gcomm, &size));
  edgenbr = xadj[nloc];

  /* SCOTCH_dgraphBuild() requires the processes to agree on whether each optional weight array is
     null. Every process here owns vertices, so a null vertex weight array means the caller supplied
     weights unevenly, which leaves the weight of these vertices undefined. A process can own
     vertices and no edges, though, and then its zero-size edge weight allocation returns null
     through no fault of the caller, so give it a stand-in array; PT-SCOTCH reads one entry per local
     edge and this process has none, so a single element it never looks at is enough. */
  hasweights[0] = vwgt ? PETSC_TRUE : PETSC_FALSE;
  hasweights[1] = adjwgt ? PETSC_TRUE : PETSC_FALSE;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, hasweights, 2, MPI_C_BOOL, MPI_LOR, gcomm));
  PetscCheck(!hasweights[0] || vwgt, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Vertex weights supplied on another process but not on this one; supply them on every process that owns vertices");
  PetscCheck(!hasweights[1] || adjwgt || edgenbr == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Edge weights supplied on another process but not on this one; supply them on every process that owns edges");
  if (hasweights[1] && !adjwgt) {
    PetscCall(PetscCalloc1(1, &edlotabempty));
    edlotab = edlotabempty;
  }

  PetscCallExternal(SCOTCH_stratInit, &stradat);
  PetscCallExternal(SCOTCH_archInit, &archdat);
  if (size == 1) {
    SCOTCH_Graph grafdat;

    /* The whole graph is on this process, so partition it sequentially. Asking for more parts than
       there are vertices leaves the surplus parts empty, and PT-SCOTCH divides by a per-part
       average load that truncates to zero once the graph is smaller than the number of parts. */
    archnbr = PetscMin(nparts, vertnbr);
    PetscCallExternal(SCOTCH_stratGraphMapBuild, &stradat, strategy, nparts, kbalval);
    PetscCallExternal(SCOTCH_graphInit, &grafdat);
    PetscCallExternal(SCOTCH_graphBuild, &grafdat, 0, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, adjncy, edlotab);
    if (PetscDefined(USE_DEBUG)) PetscCallExternal(SCOTCH_graphCheck, &grafdat);
    if (tpwgts) PetscCallExternal(SCOTCH_archCmpltw, &archdat, archnbr, (SCOTCH_Num *)tpwgts);
    else PetscCallExternal(SCOTCH_archCmplt, &archdat, archnbr);
    PetscCallExternal(SCOTCH_graphMap, &grafdat, &archdat, &stradat, assignment);
    SCOTCH_graphExit(&grafdat);
  } else {
    SCOTCH_Dgraph   grafdat;
    SCOTCH_Dmapping mappdat;

    archnbr = nparts;
    PetscCallExternal(SCOTCH_stratDgraphMapBuild, &stradat, strategy, size, nparts, kbalval);
    PetscCallExternal(SCOTCH_dgraphInit, &grafdat, gcomm);
    PetscCallExternal(SCOTCH_dgraphBuild, &grafdat, 0, vertnbr, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, edgenbr, adjncy, NULL, edlotab);
    if (PetscDefined(USE_DEBUG)) PetscCallExternal(SCOTCH_dgraphCheck, &grafdat);
    if (tpwgts) PetscCallExternal(SCOTCH_archCmpltw, &archdat, archnbr, (SCOTCH_Num *)tpwgts);
    else PetscCallExternal(SCOTCH_archCmplt, &archdat, archnbr);
    PetscCallExternal(SCOTCH_dgraphMapInit, &grafdat, &mappdat, &archdat, assignment);
    PetscCallExternal(SCOTCH_dgraphMapCompute, &grafdat, &mappdat, &stradat);
    SCOTCH_dgraphMapExit(&grafdat, &mappdat);
    SCOTCH_dgraphExit(&grafdat);
  }
  SCOTCH_archExit(&archdat);
  SCOTCH_stratExit(&stradat);
  PetscCall(PetscFree(edlotabempty));
  PetscCallMPI(MPI_Comm_free(&gcomm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
