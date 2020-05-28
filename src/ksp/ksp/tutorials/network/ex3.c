static char help[] = "This example tests subnetwork coupling. \n\
              \n\n";

/* T
  Concepts: DMNetwork
*/

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size, rank;
  DM                dmnetwork;
  PetscInt          i,j,nsubnet,nsubnetCouple=0,numVertices[3],numEdges[3],numEdgesCouple[1],*edgelist[3],*edgelist_couple=NULL;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  //ierr = PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL);CHKERRQ(ierr);

  /* Create a network which consists of subnetworks */
  /* initialization */
  //nsubnet = (PetscInt)size;
  nsubnet = 3;
  for (i=0; i<nsubnet; i++) {
    numVertices[i] = 0;
    numEdges[i]    = 0;
  }
  numEdgesCouple[0] = 0;

  for (i=0; i<nsubnet; i++) {
    //if (rank == i) {
      numVertices[i] = 4;
      numEdges[i]    = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);

      if (i == 0) {
        edgelist[i][0] = 0; edgelist[i][1] = 2;
        edgelist[i][2] = 2; edgelist[i][3] = 1;
        edgelist[i][4] = 1; edgelist[i][5] = 3;
      } else if (i == 1) {
        edgelist[i][0] = 0; edgelist[i][1] = 3;
        edgelist[i][2] = 3; edgelist[i][3] = 2;
        edgelist[i][4] = 2; edgelist[i][5] = 1;
      } else {
        for (j=0; j< numEdges[i]; j++) {
          edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
        }
      }
      //}
  }

  /* Coupling edges between subnetworks */
  nsubnetCouple = 1; /* global */
  numEdgesCouple[0] = 0;
  if (rank == 0) {
    numEdgesCouple[0] = nsubnet - 1;

    ierr = PetscMalloc1(4*numEdgesCouple[0],&edgelist_couple);CHKERRQ(ierr);
    for (j=0; j<numEdgesCouple[0]; j++) {
      edgelist_couple[4*j+0] = 0; edgelist_couple[4*j+1] = 0; /* from node: net[0] vertex[0] */
      edgelist_couple[4*j+2] = j+1; edgelist_couple[4*j+3] = 0; /* to node: net[j] vertex[0] */
    }
  }

  /* Create a dmnetwork */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);

  /* Set number of vertices and edges */
  ierr = DMNetworkSetSizes(dmnetwork,nsubnet,numVertices,numEdges,nsubnetCouple,numEdgesCouple);CHKERRQ(ierr);

  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(dmnetwork,edgelist,&edgelist_couple);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(dmnetwork);CHKERRQ(ierr);
  ierr = DMSetUp(dmnetwork);CHKERRQ(ierr);

  PetscInt       ne,nv,e;
  const PetscInt *vtx,*edges,*vcone;
  for (i=0; i<nsubnet; i++) {
    ierr = DMNetworkGetSubnetworkInfo(dmnetwork,i,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    printf("\n[%d] subnet[%d]: ne %d, nv %d\n",rank,i,ne,nv);
    for (e=0; e<ne; e++) {
      ierr = DMNetworkGetConnectedVertices(dmnetwork,edges[e],&vcone);CHKERRQ(ierr);
      printf("edges[%d]= %d: %D --> %D\n",e,edges[e],vcone[0],vcone[1]);
    }
  }

  ierr = DMNetworkGetSubnetworkCoupleInfo(dmnetwork,0,&nv,&vtx);CHKERRQ(ierr);
  printf("\n[%d] coupling info: nv %d\n",rank,nv);CHKERRQ(ierr);
  if (nv) {
    for (i=0; i<nv; i++) {
      printf(" vtx[%d] = %d\n",i,vtx[i]);
    }
  }

  /* Free work space */
  for (i=0; i<nsubnet; i++) {
    //if (rank == i) {
      ierr = PetscFree(edgelist[i]);CHKERRQ(ierr);
      //}
  }
  if (rank == 0) {ierr = PetscFree(edgelist_couple);CHKERRQ(ierr);}

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
