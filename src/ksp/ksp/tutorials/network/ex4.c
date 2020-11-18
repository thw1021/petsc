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
  PetscInt       i,j,net,Nsubnet,ne,nv,nvar,v,ncomp,compkey0,compkey1,compkey,goffset,row;
  PetscInt       numVertices[10],numEdges[10],*edgelist[10],asvtx,bsvtx;
  const PetscInt *vtx,*edges;
  PetscBool      iscouplev,ghost,distribute=PETSC_FALSE;
  Vec            X;
  Comp0          comp0;
  Comp1          comp1;
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

  /* Create componnets */
  ierr = PetscMalloc2(1,&comp0,1,&comp1);CHKERRQ(ierr);
  comp0[0].id  = rank + 1;       /* intentionally take rank-dependent value for test */
  comp1[0].val = 10.0*rank + 1.0;

  /* Create a dmnetwork and register components */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dmnetwork);CHKERRQ(ierr);
  //ierr = DMNetworkRegisterComponent(dmnetwork,"comp0",sizeof(struct _p_Comp0),&compkey0);CHKERRQ(ierr);
  //ierr = DMNetworkRegisterComponent(dmnetwork,"comp1",sizeof(struct _p_Comp1),&compkey1);CHKERRQ(ierr);

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
      if (iscouplev) {
        if (size > 1) continue; //do not set variables at shared vertex; remove this line  --> crash when np>1!!!
      }

      if (!net) {
        /* Set nvar = 1 for subnet0 */
        ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],1);CHKERRQ(ierr);
      } else {
        /* Set nvar = 2 for other subnets */
        ierr = DMNetworkAddNumVariables(dmnetwork,vtx[v],2);CHKERRQ(ierr);
      }
    }
  }
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

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
  ierr = DMView(dmnetwork,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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
    ierr = DMNetworkGetNumVariables(dmnetwork,vtx[v],&nvar);CHKERRQ(ierr);
    ierr = DMNetworkGetVariableGlobalOffset(dmnetwork,vtx[v],&goffset);CHKERRQ(ierr);
    for (i=0; i<nvar; i++) {
      row = goffset + i;
      val = 1.0;
      ierr = VecSetValues(X,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
    }

    #if 0
    ierr = DMNetworkGetNumComponents(dmnetwork,vtx[v],&ncomp);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] coupling nv %D, v %D; ncomp %D\n",rank,nv,vtx[v],ncomp);CHKERRQ(ierr);
    for (j=0; j<ncomp; j++) {
      ierr = DMNetworkGetComponentNumVariables(dmnetwork,vtx[v],j,&nvar);CHKERRQ(ierr);
      ierr = DMNetworkGetComponentVariableGlobalOffset(dmnetwork,vtx[v],j,&goffset);CHKERRQ(ierr);
      ierr = DMNetworkGetComponentKeyOffset(dmnetwork,vtx[v],j,&compkey,NULL);CHKERRQ(ierr);

      for (i=0; i<nvar; i++) {
        row = goffset + i;
        val = compkey + 1.0;
        ierr = VecSetValues(X,1,&row,&val,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    #endif
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
