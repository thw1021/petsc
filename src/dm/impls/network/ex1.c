static char help[] = "Demonstrates  DMNETWORK when two edges connect the two same vertices and an edge that connects the same vertex.\n\n";

#include <petscdmnetwork.h>
#include <petscsf.h>

struct _Vertex
{
  PetscInt id;
};
typedef struct _Vertex Vertex;

int main(int argc,char **argv)
{
  PetscErrorCode      ierr;
  DM                  networkdm;
  PetscMPIInt         size,rank;
  PetscInt            nv,ne,edgelist[12],vertex_key,i;
  const PetscInt      *vtx,*etx;
  Vec                 X,Xlocal;
  DM                  plex;
  PetscViewer         sviewer;
  PetscSF             sf;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (size != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with two MPI ranks");

  /* Create an empty network object */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&networkdm);CHKERRQ(ierr);

  /* Register the components in the network */
  ierr = DMNetworkRegisterComponent(networkdm,"vertex",sizeof(struct _Vertex),&vertex_key);CHKERRQ(ierr);

  ierr = DMNetworkSetNumSubNetworks(networkdm,PETSC_DECIDE,1);CHKERRQ(ierr);
  if (rank == 0) {
    edgelist[0] = 0;
    edgelist[1] = 1;
    edgelist[2] = 1;
    edgelist[3] = 2;
    edgelist[4] = 0;
    edgelist[5] = 2;
    edgelist[6] = 0;
    edgelist[7] = 3;
    edgelist[8] = 3;
    edgelist[9] = 4;
    edgelist[10] = 0;
    edgelist[11] = 4;
    ne = 6;
  } else {
    ne = 0;
  }
  ierr = DMNetworkAddSubnetwork(networkdm,"subnetwork",ne,edgelist,NULL);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  ierr = DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,NULL);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    ierr = DMNetworkAddComponent(networkdm,vtx[i],vertex_key,NULL,1);CHKERRQ(ierr);
  }
  ierr = DMSetUp(networkdm);CHKERRQ(ierr);
  ierr = DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMNetworkGetPlex(networkdm,&plex);CHKERRQ(ierr);
  ierr = DMView(plex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMGetSectionSF(plex,&sf);CHKERRQ(ierr);
  ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Re-distribute networkdm to multiple processes for better job balance */
  ierr = DMNetworkDistribute(&networkdm,0);CHKERRQ(ierr);
  ierr = DMView(networkdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMNetworkGetPlex(networkdm,&plex);CHKERRQ(ierr);
  ierr = DMView(plex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(networkdm,&X);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(networkdm,&Xlocal);CHKERRQ(ierr);

  ierr = VecSet(X,-1.0);CHKERRQ(ierr);
  /* loop over all local (owned and ghosted) vertices and add values into the global vector for them */
  ierr = DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,&etx);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    PetscInt offset;
    ierr = DMNetworkGetGlobalVecOffset(networkdm,vtx[i],0,&offset);CHKERRQ(ierr);
    ierr = VecSetValue(X,offset,10*rank+i,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMGlobalToLocal(networkdm,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(Xlocal,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  for (i = 0; i < ne; i++) {
    const PetscInt *cvtx;
    ierr = DMNetworkGetConnectedVertices(networkdm,etx[i],&cvtx);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local vertices for local edge %D: %D %D\n",etx[i],cvtx[0],cvtx[1]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    const PetscInt *cetx;
    PetscInt       j,nedge;

    ierr = DMNetworkGetSupportingEdges(networkdm,vtx[i],&nedge,&cetx);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local edges supporing local vertex %D: %D",vtx[i],cetx[0]);CHKERRQ(ierr);
    for (j=1; j<nedge; j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD," %D",cetx[j]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* verify the ghost vertex (which is i = 0, vtx[0] = 3) comes last in the local vector */
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    PetscInt loffset,goffset;
    ierr = DMNetworkGetLocalVecOffset(networkdm,vtx[i],0,&loffset);CHKERRQ(ierr);
    ierr = DMNetworkGetGlobalVecOffset(networkdm,vtx[i],0,&goffset);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Local vertext %D %D local offset %D global offset %D\n",i,vtx[i],loffset,goffset);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Xlocal);CHKERRQ(ierr);
  ierr = DMDestroy(&networkdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !complex double defined(PETSC_HAVE_ATTRIBUTEALIGNED)

   test:
     nsize: 2

TEST*/
