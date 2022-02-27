static const char help[] = "Tests PetscDeviceContextFork/Join.\n\n";

#include "petscdevicetestcommon.h"

static PetscErrorCode TestNestedPetscDeviceContextForkJoin(PetscDeviceContext parCtx, PetscDeviceContext *sub)
{
  const PetscInt      nsub = 4;
  PetscDeviceContext *subsub;

  PetscFunctionBegin;
  PetscValidDeviceContext(parCtx,1);
  PetscValidPointer(sub,2);
  PetscCall(AssertPetscDeviceContextsValidAndEqual(parCtx,sub[0],"Current global context does not match expected global context"));
  /* create some children from an active child */
  PetscCall(PetscDeviceContextFork(sub[1],nsub,&subsub));
  /* join on a sibling to the parent */
  PetscCall(PetscDeviceContextJoin(sub[2],nsub-2,PETSC_DEVICE_CONTEXT_JOIN_SYNC,&subsub));
  /* join on the grandparent */
  PetscCall(PetscDeviceContextJoin(parCtx,nsub-2,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&subsub));
  PetscCall(PetscDeviceContextJoin(sub[1],nsub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub));
  PetscFunctionReturn(0);
}

/* test fork-join */
static PetscErrorCode TestPetscDeviceContextForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* mostly for valgrind to catch errors */
  PetscCall(PetscDeviceContextFork(dctx,n,&sub));
  PetscCall(PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub));
  /* do it twice */
  PetscCall(PetscDeviceContextFork(dctx,n,&sub));
  PetscCall(PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub));

  /* create some children */
  PetscCall(PetscDeviceContextFork(dctx,n+1,&sub));
  /* test forking within nested function */
  PetscCall(TestNestedPetscDeviceContextForkJoin(sub[0],sub));
  /* join a subset */
  PetscCall(PetscDeviceContextJoin(dctx,n-1,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&sub));
  /* back to the ether from whence they came */
  PetscCall(PetscDeviceContextJoin(dctx,n+1,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(PetscDeviceContextCreate(&dctx));
  PetscCall(PetscDeviceContextSetFromOptions(PETSC_COMM_WORLD,"local_",dctx));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(TestPetscDeviceContextForkJoin(dctx));
  PetscCall(PetscDeviceContextDestroy(&dctx));

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(TestPetscDeviceContextForkJoin(dctx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 3}}
   args: -device_enable {{lazy eager}}
   args: -local_device_context_stream_type {{global_blocking default_blocking global_nonblocking}}
   test:
     requires: !device
     suffix: host_no_device
   test:
     requires: device
     args: -root_device_context_device_type host
     suffix: host_with_device
   test:
     requires: cuda
     args: -root_device_context_device_type cuda
     suffix: cuda
   test:
     requires: hip
     args: -root_device_context_device_type hip
     suffix: hip

TEST*/
