#include <petsc/private/matpartitioningimpl.h> /*I "petscmatpartitioning.h" I*/
#include <../src/mat/impls/adj/mpi/mpiadj.h>

#include <parmetis.h>

/*
      The first 5 elements of this structure are the input control array to Metis
*/
typedef struct {
  MPI_Comm  comm;         /* duplicated communicator to be sure that ParMETIS attribute caching does not interfere with PETSc */
  PetscInt  cuts;         /* number of cuts made (output) */
  PetscInt  foldfactor;
  PetscInt  indexing;     /* 0 indicates C indexing, 1 Fortran */
  PetscBool repartition;
  idx_t     options[24];
  /* weights */
  idx_t     ncon;                     /* number of weights that each vertex has */
  idx_t     *vwgt, *adjwgt, wgtflag;  /* vertex weights, edge weights, flag indicating use of vertex/edge weights */
  real_t    *tpwgts;                  /* partition weights */
  real_t    *ubvec;                   /* array of size ncon used to specify imbalance tolerance for each vertex weight */
} MatPartitioning_Parmetis;

#define CHKERRQPARMETIS(n,func)                                             \
  if (n == METIS_ERROR_INPUT) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to wrong inputs and/or options for %s",func); \
  else if (n == METIS_ERROR_MEMORY) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS error due to insufficient memory in %s",func); \
  else if (n == METIS_ERROR) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ParMETIS general error in %s",func); \

#define PetscStackCallParmetis(func,args) do {PetscStackPush(#func);int status = func args;PetscStackPop;CHKERRQPARMETIS(status,#func);} while (0)

/* called only on ranks with pmat != NULL */
static PetscErrorCode MatPartitioningSetUp_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *pm      = (MatPartitioning_Parmetis*)part->data;
  Mat                      pmat     = part->adj_work;
  Mat_MPIAdj               *adj     = (Mat_MPIAdj*)pmat->data;
  PetscInt                 *xadj    = adj->i;
  PetscInt                 *adjncy  = adj->j;
  PetscInt                 nparts   = part->n;
  PetscInt                 i,j;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (PetscDefined(PETSC_USE_DEBUG)) {
    /* check that matrix has no diagonal entries */
    PetscInt rstart;
    ierr = MatGetOwnershipRange(pmat,&rstart,NULL);CHKERRQ(ierr);
    for (i=0; i<pmat->rmap->n; i++) {
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        if (adjncy[j] == i+rstart) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Row %D has diagonal entry; Parmetis forbids diagonal entry",i+rstart);
      }
    }
  }

  /* Vertex/edge weights */
  pm->ncon    = 1;                               /* we currently support only one constraint per vertex */
  pm->vwgt    = part->use_vertex_weights ? (idx_t*)part->vertex_weights : NULL;
  pm->adjwgt  = part->use_edge_weights   ? (idx_t*)adj->values : NULL;
  pm->wgtflag                               = 0;  /* no edge/vertex weights */
  if  (pm->adjwgt && !pm->vwgt) pm->wgtflag = 1;  /* weights on edges only */
  if (!pm->adjwgt &&  pm->vwgt) pm->wgtflag = 2;  /* weights on vertices only */
  if  (pm->adjwgt &&  pm->vwgt) pm->wgtflag = 3;  /* weights on both edges and vertices */

  /* Partition weights */
  ierr = PetscMalloc1(nparts,&pm->tpwgts);CHKERRQ(ierr);
  for (i=0; i<pm->ncon; i++) {
    for (j=0; j<nparts; j++) {
      if (part->use_part_weights && part->part_weights) {
        pm->tpwgts[i*nparts+j] = part->part_weights[i*nparts+j];
      } else {
        pm->tpwgts[i*nparts+j] = 1./nparts;
      }
    }
  }

  /* Imbalance tolerance */
  //TODO hard-wired value should be settable from options
  ierr = PetscMalloc1(pm->ncon,&pm->ubvec);CHKERRQ(ierr);
  for (i=0; i<pm->ncon; i++) pm->ubvec[i] = 1.05;

  /* This sets the defaults */
  {
    PetscInt len = (PetscInt) (sizeof(pm->options)/sizeof(idx_t));

    pm->options[0] = 0;
    for (i=1; i<len; i++) pm->options[i] = -1;
  }

  /* Duplicate the communicator to be sure that ParMETIS attribute caching does not interfere with PETSc. */
  {
    MPI_Comm pcomm;

    ierr = PetscObjectGetComm((PetscObject)pmat,&pcomm);CHKERRQ(ierr);
    ierr = MPI_Comm_dup(pcomm,&pm->comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS parallel matrix partitioner to partition the matrix in parallel
*/
static PetscErrorCode MatPartitioningApply_Parmetis(MatPartitioning part, PetscInt locals[])
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  Mat                      pmat    = part->adj_work;

  PetscFunctionBegin;
  //TODO handle this on interface level?
  if (pmat) {
    Mat_MPIAdj *adj     = (Mat_MPIAdj*)pmat->data;
    idx_t      *vtxdist = (idx_t*) pmat->rmap->range;
    idx_t      *xadj    = (idx_t*) adj->i;
    idx_t      *adjncy  = (idx_t*) adj->j;
    idx_t      numflag=0, nparts=part->n;
    real_t     itr=0.1;

    if (pmetis->repartition) {
      //TODO should this be separate, like MatPartitioningAdaptiveRepart
      PetscStackCallParmetis(ParMETIS_V3_AdaptiveRepart,(vtxdist,xadj,adjncy,pmetis->vwgt,pmetis->vwgt,pmetis->adjwgt,&pmetis->wgtflag,&numflag,&pmetis->ncon,&nparts,pmetis->tpwgts,pmetis->ubvec,&itr,pmetis->options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&pmetis->comm));
    } else {
      PetscStackCallParmetis(ParMETIS_V3_PartKway,(vtxdist,xadj,adjncy,pmetis->vwgt,pmetis->adjwgt,&pmetis->wgtflag,&numflag,&pmetis->ncon,&nparts,pmetis->tpwgts,pmetis->ubvec,pmetis->options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&pmetis->comm));
    }
  }
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS parallel matrix partitioner to compute a nested dissection ordering of the matrix in parallel
*/
static PetscErrorCode MatPartitioningApplyND_Parmetis(MatPartitioning part, IS *partitioning)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;
  PetscInt                 *locals = NULL;
  Mat                      pmat    = part->adj_work;
  PetscInt                 bs      = part->bs;

  PetscFunctionBegin;
  //TODO handle this on interface level?
  if (pmat) {
    Mat_MPIAdj  *adj     = (Mat_MPIAdj*)pmat->data;
    idx_t       *vtxdist = (idx_t*) pmat->rmap->range;
    idx_t       *xadj    = (idx_t*) adj->i;
    idx_t       *adjncy  = (idx_t*) adj->j;
    PetscInt    *NDorder = NULL;
    idx_t       numflag=0;
    PetscInt    *sizes, *seps, log2size, subd, *level;
    PetscInt    i;
    PetscMPIInt size;
    idx_t       mtype = PARMETIS_MTYPE_GLOBAL, rtype = PARMETIS_SRTYPE_2PHASE, p_nseps = 1, s_nseps = 1;
    real_t      ubfrac = 1.05;

    ierr = PetscMalloc1(pmat->rmap->n,&locals);CHKERRQ(ierr);

    ierr = MPI_Comm_size(pmetis->comm,&size);CHKERRQ(ierr);
    ierr = PetscMalloc1(pmat->rmap->n,&NDorder);CHKERRQ(ierr);
    ierr = PetscMalloc3(2*size,&sizes,4*size,&seps,size,&level);CHKERRQ(ierr);
    PetscStackCallParmetis(ParMETIS_V32_NodeND,(vtxdist,xadj,adjncy,pmetis->vwgt,&numflag,&mtype,&rtype,&p_nseps,&s_nseps,&ubfrac,NULL/* seed */,NULL/* dbglvl */,(idx_t*)NDorder,(idx_t*)sizes,&pmetis->comm));
    log2size = PetscLog2Real(size);
    subd = PetscPowInt(2,log2size);
    ierr = MatPartitioningSizesToSep_Private(subd,sizes,seps,level);CHKERRQ(ierr);
    for (i=0;i<pmat->rmap->n;i++) {
      PetscInt loc;

      ierr = PetscFindInt(NDorder[i],2*subd,seps,&loc);CHKERRQ(ierr);
      if (loc < 0) {
        loc = -(loc+1);
        if (loc%2) { /* part of subdomain */
          locals[i] = loc/2;
        } else {
          ierr = PetscFindInt(NDorder[i],2*(subd-1),seps+2*subd,&loc);CHKERRQ(ierr);
          loc = loc < 0 ? -(loc+1)/2 : loc/2;
          locals[i] = level[loc];
        }
      } else locals[i] = loc/2;
    }
    ierr = PetscFree3(sizes,seps,level);CHKERRQ(ierr);

    if (bs > 1) {
      PetscInt i,j,*newlocals;
      ierr = PetscMalloc1(bs*pmat->rmap->n,&newlocals);CHKERRQ(ierr);
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=0; j<bs; j++) {
          newlocals[bs*i + j] = locals[i];
        }
      }
      ierr = PetscFree(locals);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),bs*pmat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    }
    {
      IS ndis;

      if (bs > 1) {
        ierr = ISCreateBlock(PetscObjectComm((PetscObject)part),bs,pmat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis);CHKERRQ(ierr);
      } else {
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,NDorder,PETSC_OWN_POINTER,&ndis);CHKERRQ(ierr);
      }
      ierr = ISSetPermutation(ndis);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis);CHKERRQ(ierr);
      ierr = ISDestroy(&ndis);CHKERRQ(ierr);
    }
  } else {
    IS ndis;

    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,partitioning);CHKERRQ(ierr);
    if (bs > 1) {
      ierr = ISCreateBlock(PetscObjectComm((PetscObject)part),bs,0,NULL,PETSC_COPY_VALUES,&ndis);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,&ndis);CHKERRQ(ierr);
    }
    ierr = ISSetPermutation(ndis);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)(*partitioning),"_petsc_matpartitioning_ndorder",(PetscObject)ndis);CHKERRQ(ierr);
    ierr = ISDestroy(&ndis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Uses the ParMETIS to improve the quality  of a partition
*/
static PetscErrorCode MatPartitioningImprove_Parmetis(MatPartitioning part, PetscInt locals[])
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  Mat                      pmat    = part->adj_work;

  PetscFunctionBegin;
  //TODO handle this on interface level?
  if (pmat) {
    Mat_MPIAdj *adj     = (Mat_MPIAdj*)pmat->data;
    idx_t      *vtxdist = (idx_t*) pmat->rmap->range;
    idx_t      *xadj    = (idx_t*) adj->i;
    idx_t      *adjncy  = (idx_t*) adj->j;
    idx_t      numflag=0, nparts=part->n;

    PetscStackCallParmetis(ParMETIS_V3_RefineKway,(vtxdist,xadj,adjncy,pmetis->vwgt,pmetis->adjwgt,&pmetis->wgtflag,&numflag,&pmetis->ncon,&nparts,pmetis->tpwgts,pmetis->ubvec,pmetis->options,(idx_t*)&pmetis->cuts,(idx_t*)locals,&pmetis->comm));
  }
  PetscFunctionReturn(0);
}

//TODO move to partition.c
static PetscErrorCode  MatPartitioningPostApply_Private(MatPartitioning part,const PetscInt locals[],IS *partitioning)
{
  Mat            pmat    = part->adj_work;
  PetscInt       bs      = part->bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pmat) {
    if (bs > 1) {
      PetscInt i,j,*newlocals;
      ierr = PetscMalloc1(bs*pmat->rmap->n,&newlocals);CHKERRQ(ierr);
      for (i=0; i<pmat->rmap->n; i++) {
        for (j=0; j<bs; j++) {
          newlocals[bs*i + j] = locals[i];
        }
      }
      ierr = PetscFree(locals);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),bs*pmat->rmap->n,newlocals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    } else {
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),pmat->rmap->n,locals,PETSC_OWN_POINTER,partitioning);CHKERRQ(ierr);
    }
  } else {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)part),0,NULL,PETSC_COPY_VALUES,partitioning);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

//TODO move to partition.c, immerse into MatPartitioningApply()
PetscErrorCode  MatPartitioningApply_New(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  Mat            pmat    = part->adj_work;
  PetscInt       *locals = NULL;

  PetscFunctionBegin;
  if (pmat) {
    ierr = PetscMalloc1(pmat->rmap->n,&locals);CHKERRQ(ierr);
  }
  //TODO replace with   ierr = (*part->ops->apply)(part,locals);CHKERRQ(ierr);
  ierr = MatPartitioningApply_Parmetis(part, locals);CHKERRQ(ierr);

  ierr = MatPartitioningPostApply_Private(part,locals,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//TODO move to partition.c, immerse into MatPartitioningImprove()
PetscErrorCode  MatPartitioningImprove_New(MatPartitioning part,IS *partitioning)
{
  PetscErrorCode ierr;
  Mat            pmat    = part->adj_work;
  PetscInt       bs      = part->bs;
  PetscInt       *locals = NULL;

  PetscFunctionBegin;
  if (pmat) {
    const PetscInt *part_indices;
    PetscInt       i;

    ierr = PetscMalloc1(pmat->rmap->n,&locals);CHKERRQ(ierr);
    ierr = ISGetIndices(*partitioning,&part_indices);CHKERRQ(ierr);
    for (i=0; i<pmat->rmap->n; i++) locals[i] = part_indices[i*bs];
    ierr = ISRestoreIndices(*partitioning,&part_indices);CHKERRQ(ierr);
    ierr = ISDestroy(partitioning);CHKERRQ(ierr);
  }
  //TODO replace with   ierr = (*part->ops->improve)(part,locals);CHKERRQ(ierr);
  ierr = MatPartitioningImprove_Parmetis(part, locals);CHKERRQ(ierr);

  ierr = MatPartitioningPostApply_Private(part,locals,partitioning);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningView_Parmetis(MatPartitioning part,PetscViewer viewer)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;
  PetscMPIInt              rank;
  PetscBool                iascii;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)part),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Using %D fold factor\n",pmetis->foldfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d]Number of cuts found %D\n",rank,pmetis->cuts);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
     MatPartitioningParmetisSetRepartition - Repartition
     current mesh to rebalance computation.

  Logically Collective on MatPartitioning

  Input Parameter:
.  part - the partitioning context

   Level: advanced

@*/
PetscErrorCode  MatPartitioningParmetisSetRepartition(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;

  PetscFunctionBegin;
  pmetis->repartition = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  MatPartitioningParmetisGetEdgeCut - Returns the number of edge cuts in the vertex partition.

  Input Parameter:
. part - the partitioning context

  Output Parameter:
. cut - the edge cut

   Level: advanced

@*/
PetscErrorCode  MatPartitioningParmetisGetEdgeCut(MatPartitioning part, PetscInt *cut)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*) part->data;

  PetscFunctionBegin;
  *cut = pmetis->cuts;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningSetFromOptions_Parmetis(PetscOptionItems *PetscOptionsObject,MatPartitioning part)
{
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Set ParMeTiS partitioning options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_partitioning_parmetis_repartition","","MatPartitioningParmetisSetRepartition",flag,&flag,NULL);CHKERRQ(ierr);
  if (flag){
    ierr =  MatPartitioningParmetisSetRepartition(part);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatPartitioningReset_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (pmetis->comm != MPI_COMM_NULL) {
    ierr = MPI_Comm_free(&pmetis->comm);CHKERRQ(ierr);
  }
  ierr = PetscFree(pmetis->tpwgts);CHKERRQ(ierr);
  ierr = PetscFree(pmetis->ubvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPartitioningDestroy_Parmetis(MatPartitioning part)
{
  MatPartitioning_Parmetis *pmetis = (MatPartitioning_Parmetis*)part->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pmetis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   MATPARTITIONINGPARMETIS - Creates a partitioning context via the external package PARMETIS.

   Collective

   Input Parameter:
.  part - the partitioning context

   Options Database Keys:
.  -mat_partitioning_parmetis_coarse_sequential - use sequential PARMETIS coarse partitioner

   Level: beginner

   Notes:
    See https://www-users.cs.umn.edu/~karypis/metis/

.seealso: MatPartitioningSetType(), MatPartitioningType

M*/

PETSC_EXTERN PetscErrorCode MatPartitioningCreate_Parmetis(MatPartitioning part)
{
  PetscErrorCode           ierr;
  MatPartitioning_Parmetis *pmetis;

  PetscFunctionBegin;
  ierr       = PetscNewLog(part,&pmetis);CHKERRQ(ierr);
  part->data = (void*)pmetis;

  pmetis->cuts       = 0;   /* output variable */
  pmetis->foldfactor = 150; /* folding factor */
  pmetis->indexing   = 0;   /* index numbering starts from 0 */
  pmetis->repartition= PETSC_FALSE;
  pmetis->comm       = MPI_COMM_NULL;

  part->parallel            = PETSC_TRUE;
  part->ops->apply          = MatPartitioningApply_New;
  part->ops->applynd        = MatPartitioningApplyND_Parmetis;
  part->ops->improve        = MatPartitioningImprove_New;
  part->ops->view           = MatPartitioningView_Parmetis;
  part->ops->destroy        = MatPartitioningDestroy_Parmetis;
  part->ops->reset          = MatPartitioningReset_Parmetis;
  part->ops->setfromoptions = MatPartitioningSetFromOptions_Parmetis;
  part->ops->setup          = MatPartitioningSetUp_Parmetis;
  PetscFunctionReturn(0);
}

/*@
     MatMeshToCellGraph -   Uses the ParMETIS package to convert a Mat that represents a mesh to a Mat the represents the graph of the coupling
                       between cells (the "dual" graph) and is suitable for partitioning with the MatPartitioning object. Use this to partition
                       cells of a mesh.

   Collective on Mat

   Input Parameter:
+     mesh - the graph that represents the mesh
-     ncommonnodes - mesh elements that share this number of common nodes are considered neighbors, use 2 for triangles and
                     quadrilaterials, 3 for tetrahedrals and 4 for hexahedrals

   Output Parameter:
.     dual - the dual graph

   Notes:
     Currently requires ParMetis to be installed and uses ParMETIS_V3_Mesh2Dual()

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

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)mesh,MATMPIADJ,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must use MPIAdj matrix type");

  ierr = PetscObjectGetComm((PetscObject)mesh,&comm);CHKERRQ(ierr);
  PetscStackCallParmetis(ParMETIS_V3_Mesh2Dual,((idx_t*)mesh->rmap->range,(idx_t*)adj->i,(idx_t*)adj->j,(idx_t*)&numflag,(idx_t*)&ncommonnodes,(idx_t**)&newxadj,(idx_t**)&newadjncy,&comm));
  ierr   = MatCreateMPIAdj(PetscObjectComm((PetscObject)mesh),mesh->rmap->n,mesh->rmap->N,newxadj,newadjncy,NULL,dual);CHKERRQ(ierr);
  newadj = (Mat_MPIAdj*)(*dual)->data;

  newadj->freeaijwithfree = PETSC_TRUE; /* signal the matrix should be freed with system free since space was allocated by ParMETIS */
  PetscFunctionReturn(0);
}
