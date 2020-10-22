static char help[] = "This example tests subnetwork coupling. \n\
              \n\n";

/* T
  Concepts: DMNetwork
*/
#include <petscdmnetwork.h>

struct _p_Comp0{
  PetscInt id;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Comp0 *Comp0;

struct _p_Comp1{
  PetscScalar val;
} PETSC_ATTRIBUTEALIGNED(sizeof(PetscScalar));
typedef struct _p_Comp1 *Comp1;

int main(int argc,char ** argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  DM             dmnetwork;
  PetscInt       i,j,net,nsubnet,ne,nv,nvar,v,ncomp,compkey0,compkey1,goffset,row;
  PetscInt       nsubnetCouple=0,numVertices[10],numEdges[10],numVtxCouple[1],*edgelist[10],*edgelist_couple=NULL;
  const PetscInt *vtx,*edges;
  PetscBool      iscouplev,ghost,distribute=PETSC_TRUE;
  Vec            X;
  Comp0          comp0;
  Comp1          comp1;
  PetscScalar    val;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Create a network of subnetworks */
  if (size == 1) nsubnet = 2;
  else nsubnet = (PetscInt)size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsubnet",&nsubnet,NULL);CHKERRQ(ierr);
  if (nsubnet > 10) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"nsubnet cannot >10 for this example");

  for (i=0; i<nsubnet; i++) {
    numVertices[i] = 0; numEdges[i] = 0;
  }
  numVtxCouple[0] = 0;

  /* when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numVertices[i] = 4; numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 2;
      edgelist[i][2] = 2; edgelist[i][3] = 1;
      edgelist[i][4] = 1; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numVertices[i] = 4; numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 3;
      edgelist[i][2] = 3; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 1;

    } else if (i>1 && (size == 1 || (rank == i && size >1))){
      numVertices[i] = 4; numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Set coupling vertices between subnetworks -- all processes hold this info */
  nsubnetCouple   = 1; /* global */
  numVtxCouple[0] = nsubnet - 1;

  ierr = PetscMalloc1(4*numVtxCouple[0],&edgelist_couple);CHKERRQ(ierr);
  for (j=0; j<numVtxCouple[0]; j++) {
    edgelist_couple[4*j+0] = 0;   edgelist_couple[4*j+1] = 0; /* CV_from: net[0] vertex[0] */
    edgelist_couple[4*j+2] = j+1; edgelist_couple[4*j+3] = 0; /* CV_to  : net[j+1] vertex[0] */
  }

  /* Create componnets to be added to the coupling vertex.
   -- Current implimentation requires that the component must take same values on all processors */
  ierr = PetscMalloc2(1,&comp0,1,&comp1);CHKERRQ(ierr);
  comp0[0].id  = rank;       /* intentionally take rank-dependent value for test */
  comp1[0].val = 10.0*rank;

  /* Create a dmnetwork and register components */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"comp0",sizeof(struct _p_Comp0),&compkey0);CHKERRQ(ierr);
  ierr = DMNetworkRegisterComponent(dmnetwork,"comp1",sizeof(struct _p_Comp1),&compkey1);CHKERRQ(ierr);

  /* Set number of vertices and edges -- nsubnetCouple is ignored and will be removed from API */
  ierr = DMNetworkSetSizes(dmnetwork,nsubnet,numVertices,numEdges,nsubnetCouple,numVtxCouple);CHKERRQ(ierr);

  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(dmnetwork,edgelist,&edgelist_couple);CHKERRQ(ierr);

  /* Setup the network layout */
  ierr = DMNetworkLayoutSetUp(dmnetwork);CHKERRQ(ierr);

  /* Get SubnetworkInfo(); Add nvar=1 to subnet[0] and nvar=2 to other subnets, excluding coupling vertex */
  for (net=0; net<nsubnet; net++) {
    ierr = DMNetworkGetSubnetworkInfo(dmnetwork,net,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    for (v=0; v<nv; v++) {
      ierr = DMNetworkIsCouplingVertex(dmnetwork,vtx[v],&iscouplev);CHKERRQ(ierr);
      if (iscouplev) continue;

      if (!net) {
        ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],1);CHKERRQ(ierr);
      } else {
        ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],2);CHKERRQ(ierr);
      }
    }
  }

  /* At the coupling vertex, add componenets 'comp0' 'comp1', and the associated num of variables */
  /* All processors must do it, thus component must have same values for different processors -- do not know why? */
  ierr = DMNetworkGetSubnetworkSharedVertices(dmnetwork,&nv,&vtx);CHKERRQ(ierr);
  for (i=0; i<nv; i++) {
    ierr = DMNetworkAddComponent(dmnetwork,vtx[i],compkey0,&comp0[0]);CHKERRQ(ierr);
    ierr = DMNetworkAddComponent(dmnetwork,vtx[i],compkey1,&comp1[0]);CHKERRQ(ierr);
    ierr = DMNetworkSetComponentNumVariables(dmnetwork,vtx[i],0,1);CHKERRQ(ierr);
    ierr = DMNetworkSetComponentNumVariables(dmnetwork,vtx[i],1,2);CHKERRQ(ierr);
  }

  /* Enable runtime option of graph partition type -- must be called before DMSetUp() */
  if (size > 1) {
    DM               plexdm;
    PetscPartitioner part;
    ierr = DMNetworkGetPlex(dmnetwork,&plexdm);CHKERRQ(ierr);
    ierr = DMPlexGetPartitioner(plexdm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part,PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL,"-dm_plex_csr_via_mat","true");CHKERRQ(ierr); /* for parmetis */
  }

  /* Setup dmnetwork */
  ierr = DMSetUp(dmnetwork);CHKERRQ(ierr);

  /* Redistribute the network layout; use '-distribute false' to skip */
  ierr = PetscOptionsGetBool(NULL,NULL,"-distribute",&distribute,NULL);CHKERRQ(ierr);
  if (distribute) {
    ierr = DMNetworkDistribute(&dmnetwork,0);CHKERRQ(ierr);
    ierr = DMView(dmnetwork,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Create a global vector */
  ierr = DMCreateGlobalVector(dmnetwork,&X);CHKERRQ(ierr);
  ierr = VecSet(X,0.0);CHKERRQ(ierr);

  /* Set X values at the coupling vertex */
  ierr = DMNetworkGetSubnetworkSharedVertices(dmnetwork,&nv,&vtx);CHKERRQ(ierr);
  for (v=0; v<nv; v++) {
    ierr = DMNetworkIsGhostVertex(dmnetwork,vtx[v],&ghost);CHKERRQ(ierr);
    if (ghost) continue;

    /* only one process holds a non-ghost vertex */
    ierr = DMNetworkGetNumComponents(dmnetwork,vtx[v],&ncomp);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] coupling nv %D, v %D; ncomp %D\n",rank,nv,vtx[v],ncomp);CHKERRQ(ierr);
    for (j=0; j<ncomp; j++) {
      ierr = DMNetworkGetComponentNumVariables(dmnetwork,vtx[v],j,&nvar);CHKERRQ(ierr);
      ierr = DMNetworkGetComponentVariableGlobalOffset(dmnetwork,vtx[v],j,&goffset);CHKERRQ(ierr);
      for (i=0; i<nvar; i++) {
        row = goffset + i;
        val = j + 1.0;
        ierr = VecSetValues(X,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free work space */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  for (i=0; i<nsubnet; i++) {
    if (size == 1 || rank == i) {ierr = PetscFree(edgelist[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(edgelist_couple);CHKERRQ(ierr);
  ierr = PetscFree2(comp0,comp1);CHKERRQ(ierr);

  ierr = DMDestroy(&dmnetwork);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !single double define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args:

   test:
      suffix: 2
      nsize: 2
      args: -options_left no

   test:
      suffix: 3
      nsize: 4
      args: -options_left no

TEST*/
