/*
    Simple example demonstrating how to create a one sub-network DMNetwork in parallel.
*/

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  DM                network;
  PetscMPIInt       size,rank;
  MPI_Comm          comm;
  PetscInt          e,ne,nv,v,ecompkey,vcompkey,itest=0;
  PetscInt          *edgelist = NULL;
  const PetscInt    *nodes,*edges;
  DM                plex;
  PetscSection      section;
  PetscInt          Ne,Ni;
  PetscInt          nodeOffset,nedge;

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  PetscCall(PetscOptionsSetValue(NULL,"-petscpartitioner_use_vertex_weights","No"));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));

  PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&network));

  /* Register zero size components to get compkeys to be used by DMNetworkAddComponent() */
  PetscCall(DMNetworkRegisterComponent(network,"ecomp",0,&ecompkey));
  PetscCall(DMNetworkRegisterComponent(network,"vcomp",0,&vcompkey));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-test",&itest,NULL));

  /* nodeOffset for rank > 0 */
  Ne = 2;
  Ni = 1;
  nodeOffset = (Ne+Ni)*rank;   /* The global node index of the first node defined on this process */

  /* There are three nodes on each rank and two edges. The edges only connect nodes on the given rank */
  nedge = 2 * Ni;
  PetscCall(PetscCalloc1(2*nedge,&edgelist));

  if (rank == 0) {
    nedge = 1; nv = 4;
    switch (itest) {
    case 0:
      /* v0, v1 are isolated vertices */
      edgelist[0] = nodeOffset + 2;
      edgelist[1] = nodeOffset + 3;
      break;
    case 1:
      /* v0,v2 are isolated vertices */
      edgelist[0] = nodeOffset + 1;
      edgelist[1] = nodeOffset + 3;
      break;
    case 2:
      /* v0,v3 are isolated vertices */
      edgelist[0] = nodeOffset + 1;
      edgelist[1] = nodeOffset + 2;
      break;
    default: PetscCheck(itest < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Test case %" PetscInt_FMT " not supported.",itest);
    }
  } else { /* rank > 0 */
    nedge = 2; nv = 4;
    switch (itest) {
    case 0: /* no isolated vertex */
      edgelist[0] = nodeOffset + 0;
      edgelist[1] = nodeOffset + 2;
      edgelist[2] = nodeOffset + 1;
      edgelist[3] = nodeOffset + 3;
    break;
    case 1:
      /* v=nodeOffset+3 is an isolated vertex */
      edgelist[0] = nodeOffset + 0;
      edgelist[1] = nodeOffset + 1;
      edgelist[2] = nodeOffset + 1;
      edgelist[3] = nodeOffset + 2;
      break;
    case 2:
      /* v=nodeOffset+0 is an isolated vertex */
      edgelist[0] = nodeOffset + 1;
      edgelist[1] = nodeOffset + 2;
      edgelist[2] = nodeOffset + 2;
      edgelist[3] = nodeOffset + 3;
      break;
    default: PetscCheck(itest < 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Test case %" PetscInt_FMT " not supported.",itest);
    }
  }

  PetscCall(DMNetworkSetNumSubNetworks(network,PETSC_DECIDE,1));
  PetscCall(DMNetworkAddSubnetwork(network,"Subnetwork 1",nedge,edgelist,NULL));
  PetscCall(DMNetworkLayoutSetUp(network));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Network after DMNetworkLayoutSetUp:\n"));
  PetscCall(DMView(network,PETSC_VIEWER_STDOUT_WORLD));

  /* Add components and variables for the network */
  PetscCall(DMNetworkGetSubnetwork(network,0,&nv,&ne,&nodes,&edges));

  for (e = 0; e < ne; e++) {
    /* The edges have no degrees of freedom */
    PetscCall(DMNetworkAddComponent(network,edges[e],ecompkey,NULL,1));
  }

  for (v = 0; v < nv; v++) {
    PetscCall(DMNetworkAddComponent(network,nodes[v],vcompkey,NULL,2));
  }

  PetscCall(DMSetUp(network));
  PetscCall(DMNetworkGetPlex(network,&plex));
  /* PetscCall(DMView(plex,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(DMGetLocalSection(plex,&section));
  PetscCall(PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFree(edgelist));

  PetscCall(DMNetworkDistribute(&network,0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nNetwork after DMNetworkDistribute:\n"));
  PetscCall(DMView(network,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMNetworkGetPlex(network,&plex));
  /* PetscCall(DMView(plex,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(DMGetLocalSection(plex,&section));
  PetscCall(PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&network));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex double

   test:
      nsize: 2
      args: -petscpartitioner_type simple

TEST*/
