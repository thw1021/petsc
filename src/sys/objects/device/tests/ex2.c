static const char help[] = "Tests creation and destruction of PetscDeviceContext.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDeviceContext dctx = NULL,ddup = NULL;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /* basic creation and destruction */
  ierr = PetscDeviceContextCreate(&dctx);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRDEVICE(ierr);
  ierr = PetscDeviceContextDestroy(&dctx);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextDoesNotExist(dctx);CHKERRDEVICE(ierr);
  /* double free is no-op */
  ierr = PetscDeviceContextDestroy(&dctx);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextDoesNotExist(dctx);CHKERRDEVICE(ierr);

  /* test global context returns a valid context */
  dctx = NULL;
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRDEVICE(ierr);
  /* test locally setting to null doesn't clobber the global */
  dctx = NULL;
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRDEVICE(ierr);

  /* test duplicate */
  ierr = PetscDeviceContextDuplicate(dctx,&ddup);CHKERRDEVICE(ierr);
  /* both device contexts should exist */
  ierr = AssertDeviceContextExists(dctx);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextExists(ddup);CHKERRDEVICE(ierr);

  /* destroying the dup should leave the original untouched */
  ierr = PetscDeviceContextDestroy(&ddup);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextDoesNotExist(ddup);CHKERRDEVICE(ierr);
  ierr = AssertDeviceContextExists(dctx);CHKERRDEVICE(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n");CHKERRDEVICE(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX)

 test:
   requires: !device
   suffix: no_device
   filter: grep -E -o -e ".*No support for this operation for this object type" -e ".*PETSc is not configured with device support.*" -e "^\[0\]PETSC ERROR:.*[0-9]{1} [A-z]+\(\)"

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 2 4}}
   test:
     requires: cuda
     suffix: cuda
   test:
     requires: hip
     suffix: hip

TEST*/
