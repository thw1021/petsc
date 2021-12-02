static char help[] = "Demonstrates failure of DMNETWORK when two edges connect the two same vertices.\n\n";

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
  PetscInt            nv,ne,edgelist[2],vertex_key,i;
  const PetscInt      *vtx;
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
  } else {
    edgelist[0] = 1;
    edgelist[1] = 0;
  }
  ierr = DMNetworkAddSubnetwork(networkdm,"subnetwork",1,edgelist,NULL);CHKERRQ(ierr);

  /* Set up the network layout */
  ierr = DMNetworkLayoutSetUp(networkdm);CHKERRQ(ierr);

  ierr = DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,NULL);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    ierr = DMNetworkAddComponent(networkdm,vtx[i],vertex_key,NULL,2);CHKERRQ(ierr);
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
  ierr = DMNetworkGetSubnetwork(networkdm,0,&nv,&ne,&vtx,NULL);CHKERRQ(ierr);
  for (i = 0; i < nv; i++) {
    ierr = VecSetValue(X,vtx[i],10*rank+i,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMGlobalToLocal(networkdm,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(Xlocal,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);

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
