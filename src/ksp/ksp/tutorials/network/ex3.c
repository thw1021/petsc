static char help[] = "This example tests subnetwork coupling. \n\
              \n\n";

/* T
  Concepts: DMNetwork
*/
#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  DM             dmnetwork;
  PetscInt       i,j,net,nsubnet,nsubnetCouple=0,numVertices[10],numEdges[10],numVtxCouple[1],*edgelist[10],*edgelist_couple=NULL,ne,nv;
  const PetscInt *vtx,*edges;

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

  /* Create a dmnetwork */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);

  /* Set number of vertices and edges */
  ierr = DMNetworkSetSizes(dmnetwork,nsubnet,numVertices,numEdges,nsubnetCouple,numVtxCouple);CHKERRQ(ierr);

  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(dmnetwork,edgelist,&edgelist_couple);CHKERRQ(ierr);

  /* Setup the network layout */
  ierr = DMNetworkLayoutSetUp(dmnetwork);CHKERRQ(ierr);

  /* Get SubnetworkInfo() */
  for (net=0; net<nsubnet; net++) {
    ierr = DMNetworkGetSubnetworkInfo(dmnetwork,net,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    if (ne) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] subnet[%d]: ne %d, nv %d\n",rank,net,ne,nv);CHKERRQ(ierr);
    }

    if (nv) {
      PetscInt v;
      for (v=0; v<nv; v++) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] net %d, vtx %d\n",rank,net,vtx[v]);CHKERRQ(ierr);
        if (net == 0) {
          ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],1);CHKERRQ(ierr);
        } else { // net > 0
          if (v == 0) {
            ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],2);CHKERRQ(ierr); //must be same as its coupling vertex!!!
          } else {
            ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],2);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = DMSetUp(dmnetwork);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* Get SubnetworkCoupleInfo() */
  ierr = DMNetworkGetSubnetworkCoupleInfo(dmnetwork,0,&nv,&vtx);CHKERRQ(ierr);
  if (nv) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] coupling info: nv %d\n",rank,nv);CHKERRQ(ierr);
    for (i=0; i<nv; i++) printf(" cvtx[%d] = %d\n",i,vtx[i]);
  }

  Vec X;
  ierr = DMCreateGlobalVector(dmnetwork,&X);CHKERRQ(ierr);
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free work space */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  for (i=0; i<nsubnet; i++) {
    if (size == 1 || rank == i) {ierr = PetscFree(edgelist[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(edgelist_couple);CHKERRQ(ierr);

  ierr = DMDestroy(&dmnetwork);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !single double define(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
      args: -ksp_converged_reason

   test:
      suffix: 2
      nsize: 2
      args: -petscpartitioner_type simple -pc_type asm -sub_pc_type ilu -ksp_converged_reason

   test:
      suffix: 3
      nsize: 4
      args: -petscpartitioner_type simple -pc_type asm -sub_pc_type lu -sub_pc_factor_shift_type nonzero -ksp_converged_reason

   test:
      suffix: graphindex
      args: -n 20 -vertex_global_section_view -edge_global_section_view

   test:
      suffix: graphindex_2
      nsize: 2
      args: -petscpartitioner_type simple -n 20 -vertex_global_section_view -edge_global_section_view

TEST*/
