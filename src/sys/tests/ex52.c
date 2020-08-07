static char help[] = "A benchmark for testing PetscSortInt() and PetscSortIntWithArrayPair()\n\
  The array is filled with random numbers, but one can control average duplicates for each unique integer with the -d option.\n\
  Usage:\n\
   mpirun -n 1 ./ex52 -n <length of the array to sort>, default=100 \n\
                      -r <repeat times for each sort>, default=10 \n\
                      -d <average duplicates for each unique integer>, default=1, i.e., no duplicates \n\n";

#include <petscsys.h>
#include <petsctime.h>
#include <petscviewer.h>
#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,l,n=100,r=10,d=1,qctr,tctr;
  PetscInt       *X,*X1,*XT,*Y,*Z;
  PetscReal      val,norm1;
  PetscRandom    rdm;
  PetscLogDouble time, time1;
  PetscMPIInt    size;
  PetscViewer    vwr;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This is a uniprocessor example only!");

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-r",&r,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-array_view",&vwr,NULL,NULL);CHKERRQ(ierr);
  if (n<1 || r<1 || d<1 || d>n) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Wrong input n=%D,r=%D,d=%d. They must be >=1 and n>=d\n",n,r,d);

  ierr = PetscCalloc4(n,&X,n,&X1,n,&Y,n,&Z);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  ierr = PetscCalloc1(n, &XT);CHKERRQ(ierr);

  for (i=0; i<n; ++i) {
    ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
    XT[i] = val*PETSC_MAX_INT;
    if (d > 1) XT[i] = XT[i] % (n/d);
  }

  time = 0.0;
  time1 = 0.0;
  if (vwr) {ierr = PetscIntView(n, XT, vwr);CHKERRQ(ierr);}
  /*
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,1000000000);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
   */
  for (l=0; l<r; l++) { /* r loops */
    //ierr = PetscArraycpy(X,XT,n);CHKERRQ(ierr);
    //ierr = PetscArraycpy(X1,XT,n);CHKERRQ(ierr);
    for (i=0; i<n; i++) { // Init X[]
      ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
      X[i] = val*PETSC_MAX_INT;
      if (d > 1) X[i] = X[i] % (n/d);
      X1[i] = X[i];
    }

    //ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = PetscTimeSubtract(&time);CHKERRQ(ierr);
    ierr = PetscSortInt(n,X);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time);CHKERRQ(ierr);

    //ierr = VecNorm(x,NORM_1,&norm1);CHKERRQ(ierr);
    ierr = PetscTimeSubtract(&time1);CHKERRQ(ierr);
    ierr = PetscTimSortInt(n,X1);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time1);CHKERRQ(ierr);

    for (i=0; i<n-1; i++) {if (X[i] > X[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
    for (i=0; i<n; i++) {if (X[i] != X1[i]) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscTimSortInt() rep %D X1[%D]:%D does not match PetscSortInt() X[%D]:%D!",l,i,X1[i],i,X[i]);}
    for (i=0; i<n-1; i++) {if (X1[i] > X1[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscTimSortInt() produced wrong results!");}
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortInt()              with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscTimSortInt()           with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time1/r);CHKERRQ(ierr);
  ierr = PetscFree(XT);CHKERRQ(ierr);

  time = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    for (i=0; i<n; i++) { /* Init X[] */
      ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
      X[i] = val*PETSC_MAX_INT;
      if (d > 1) X[i] = X[i] % (n/d);
    }

    ierr = PetscTimeSubtract(&time);CHKERRQ(ierr);
    ierr = PetscSortIntWithArrayPair(n,X,Y,Z);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time);CHKERRQ(ierr);

    for (i=0; i<n-1; i++) {if (X[i] > X[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArrayPair() with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"SUCCEEDED\n");CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vwr);CHKERRQ(ierr);
  ierr = PetscFree4(X,X1,Y,Z);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -n 1000 -r 10 -d 1
      # Do not need to output timing results for test
      filter: grep -v "per unique value took"
TEST*/
