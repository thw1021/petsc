
#include <../src/mat/impls/adj/mpi/mpiadj.h>    /*I "petscmat.h" I*/

#include <metis.h>

#define CHKERRQMETIS(n,func)                                             \
  if (n == METIS_ERROR_INPUT) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"METIS error due to wrong inputs and/or options for %s",func); \
  else if (n == METIS_ERROR_MEMORY) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"METIS error due to insufficient memory in %s",func); \
  else if (n == METIS_ERROR) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"METIS general error in %s",func); \

#define PetscStackCallMetis(func,args) do {PetscStackPush(#func);int status = func args;PetscStackPop;CHKERRQMETIS(status,#func);} while (0)


/*@
     MatMeshToCellGraph -   Uses the METIS package to convert a Mat that represents a mesh to a Mat the represents the graph of the coupling
                       between cells (the "dual" graph) and is suitable for partitioning with the MatPartitioning object. Use this to partition
                       cells of a mesh.

   Collective on Mat

   Input Parameters:
+     mesh - the graph that represents the mesh
-     ncommonnodes - mesh elements that share this number of common nodes are considered neighbors, use 2 for triangles and
                     quadrilaterials, 3 for tetrahedrals and 4 for hexahedrals

   Output Parameter:
.     dual - the dual graph

   Notes:
     Currently requires Metis to be installed and uses METIS_MeshToDual()

$     Each row of the mesh object represents a single cell in the mesh. For triangles it has 3 entries, quadrilaterials 4 entries,
$         tetrahedrals 4 entries and hexahedrals 8 entries. You can mix triangles and quadrilaterals in the same mesh, but cannot
$         mix  tetrahedrals and hexahedrals
$     The columns of each row of the Mat mesh are the global vertex numbers of the vertices of that row's cell.
$     The number of rows in mesh is number of cells, the number of columns is the number of vertices.

   Level: advanced

.seealso: MatMeshToVertexGraph(), MatCreateMPIAdj(), MatPartitioningCreate()

@*/
PetscErrorCode MatMeshToCellGraph(Mat mesh,PetscInt ncommonnodes,Mat *dual)
{
  PetscErrorCode ierr;
  PetscInt       *newxadj,*newadjncy;
  PetscInt       numflag=0;
  Mat_MPIAdj     *adj   = (Mat_MPIAdj*)mesh->data,*newadj;
  PetscBool      flg;
  MPI_Comm       comm;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)mesh, MATMPIADJ, &flg));
  PetscCheck(flg, comm, PETSC_ERR_SUP, "Must use MPIAdj matrix type");

  PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));
  PetscCall(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_SUP, "MatMeshToCellGraph() requires a sequential matrix (communicator size must be 1)");

  idx_t ne = mesh->rmap->N;
  idx_t nn = adj->nz;
  PetscStackCallMetis(METIS_MeshToDual, (&ne, &nn, (idx_t*)adj->i,(idx_t*)adj->j,(idx_t*)&ncommonnodes,(idx_t*)&numflag,(idx_t**)&newxadj,(idx_t**)&newadjncy));

  ierr   = MatCreateMPIAdj(PetscObjectComm((PetscObject)mesh),mesh->rmap->n,mesh->rmap->N,newxadj,newadjncy,NULL,dual);CHKERRQ(ierr);
  newadj = (Mat_MPIAdj*)(*dual)->data;

  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by METIS */
  PetscFunctionReturn(0);
}
