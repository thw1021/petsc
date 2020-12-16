static char help[] = "This example tests subnetwork coupling. \n\n";

#include <petscdmnetwork.h>

int main(int argc,char ** argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  DM             dmnetwork;
  PetscInt       i,j,net,Nsubnet,ne,nv,nvar,v,goffset,row;
  PetscInt       numVertices[10],numEdges[10],*edgelist[10],asvtx,bsvtx;
  const PetscInt *vtx,*edges;
  PetscBool      iscouplev,ghost,distribute=PETSC_TRUE;
  Vec            X;
  PetscScalar    val;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Create a network of subnetworks */
  if (size == 1) Nsubnet = 2;
  else Nsubnet = (PetscInt)size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-Nsubnet",&Nsubnet,NULL);CHKERRQ(ierr);
  if (Nsubnet > 10) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Nsubnet cannot >10 for this example");

  for (i=0; i<Nsubnet; i++) {numVertices[i] = 0; numEdges[i] = 0;}

  /* when size>1, process[i] creates subnetwork[i] */
  for (i=0; i<Nsubnet; i++) {
    if (i == 0 && (size == 1 || (rank == i && size >1))) {
      numVertices[i] = 4; numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i == 1 && (size == 1 || (rank == i && size >1))) {
      numVertices[i] = 4; numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      edgelist[i][0] = 0; edgelist[i][1] = 1;
      edgelist[i][2] = 1; edgelist[i][3] = 2;
      edgelist[i][4] = 2; edgelist[i][5] = 3;

    } else if (i>1 && (size == 1 || (rank == i && size >1))){
      numVertices[i] = 4; numEdges[i] = 3;
      ierr = PetscMalloc1(2*numEdges[i],&edgelist[i]);CHKERRQ(ierr);
      for (j=0; j< numEdges[i]; j++) {
        edgelist[i][2*j] = j; edgelist[i][2*j+1] = j+1;
      }
    }
  }

  /* Create a dmnetwork and register components */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);

  /* Set number of subnetworks, numbers of vertices and edges over each subnetwork */
  ierr = DMNetworkSetSizes(dmnetwork,PETSC_DECIDE,Nsubnet);CHKERRQ(ierr);

  for (i=0; i<Nsubnet; i++) {
    PetscInt netNum = -1;
    ierr = DMNetworkAddSubnetwork(dmnetwork,NULL,numVertices[i],numEdges[i],&netNum);CHKERRQ(ierr);
  }

  /* Add shared vertices -- all processes hold this info at current implementation */
  asvtx = bsvtx = 0;
  for (j=1; j<Nsubnet; j++) {
    /* vertex subnet[0].0 shares with vertex subnet[j].0 */
    ierr = DMNetworkAddSubnetworkSharedVertices(dmnetwork,0,j,1,&asvtx,&bsvtx);CHKERRQ(ierr);
  }

  /* Add edge connectivity */
  ierr = DMNetworkSetEdgeList(dmnetwork,edgelist);CHKERRQ(ierr);

  /* Setup the network layout */
  ierr = DMNetworkLayoutSetUp(dmnetwork);CHKERRQ(ierr);

  /* Get SubnetworkInfo(); Add nvar=1 to subnet[0] and nvar=2 to other subnets */
  for (net=0; net<Nsubnet; net++) {
    ierr = DMNetworkGetSubnetworkInfo(dmnetwork,net,&nv,&ne,&vtx,&edges);CHKERRQ(ierr);
    for (v=0; v<nv; v++) {
      ierr = DMNetworkIsCouplingVertex(dmnetwork,vtx[v],&iscouplev);CHKERRQ(ierr);

      if (!net) {
        /* Set nvar = 2 for subnet0 */
        ierr = DMNetworkAddComponentAndNumVariables(dmnetwork,vtx[v],-1,NULL,2);CHKERRQ(ierr);
      } else {
        /* Set nvar = 1 for other subnets */
        ierr = DMNetworkAddComponentAndNumVariables(dmnetwork,vtx[v],-1,NULL,1);CHKERRQ(ierr);
      }
    }
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

  /* Set X values at shared vertex */
  ierr = DMNetworkGetSubnetworkSharedVertices(dmnetwork,&nv,&vtx);CHKERRQ(ierr);
  for (v=0; v<nv; v++) {
    ierr = DMNetworkIsGhostVertex(dmnetwork,vtx[v],&ghost);CHKERRQ(ierr);
    if (ghost) continue;

    /* only one process holds a non-ghost vertex */
    ierr = DMNetworkGetComponentAndNumVariables(dmnetwork,vtx[v],PETSC_DECIDE,NULL,NULL,&nvar);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableGlobalOffset(dmnetwork,vtx[v],&goffset);CHKERRQ(ierr);
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = (PetscScalar)rank + 1.0;
      ierr = VecSetValues(X,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* Free work space */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  for (i=0; i<Nsubnet; i++) {
    if (size == 1 || rank == i) {ierr = PetscFree(edgelist[i]);CHKERRQ(ierr);}
  }

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
