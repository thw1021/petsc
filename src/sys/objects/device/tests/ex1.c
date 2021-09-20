static const char help[] = "Tests creation and destruction of PetscDevice.\n\n";

#include <petsc/private/deviceimpl.h>
#include "petscdevicetestcommon.h"

int main(int argc, char *argv[])
{
  PetscDevice    device = NULL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /* normal create and destroy */
  ierr = PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device);CHKERRQ(ierr);
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  ierr = AssertDeviceDoesNotExist(device);CHKERRQ(ierr);
  /* should not destroy twice */
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  ierr = AssertDeviceDoesNotExist(device);CHKERRQ(ierr);

  /* test reference counting */
  device = NULL;
  ierr = PetscDeviceCreate(PETSC_DEVICE_DEFAULT,PETSC_DECIDE,&device);CHKERRQ(ierr);
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  for (int i = 0; i < 10; ++i) device = PetscDeviceReference_Internal(device);
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  for (int i = 0; i < 10; ++i) {
    ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
    ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  }
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  ierr = AssertDeviceDoesNotExist(device);CHKERRQ(ierr);

  /* test the default devices exist */
  device = NULL;
  {
    PetscDeviceContext dctx;
    /* global context will have the default device */
    ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextGetDevice(dctx,&device);CHKERRQ(ierr);
  }
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  /* test reference counting for default device */
  for (int i = 0; i < 10; ++i) device = PetscDeviceReference_Internal(device);
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  for (int i = 0; i < 10; ++i) {
    ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
    ierr = AssertDeviceExists(device);CHKERRQ(ierr);
  }
  ierr = PetscDeviceDestroy(&device);CHKERRQ(ierr);
  /* default device should still exist */
  ierr = AssertDeviceExists(device);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 build:
   requires: defined(PETSC_HAVE_CXX_DIALECT_CXX11)

 testset:
   requires: !device
   filter: Error: grep -A 1 "No support for this operation for this object type"
   suffix: no_device
   test:
     requires: debug
     suffix:   debug
   test:
     requires: !debug
     suffix:   opt

 testset:
   output_file: ./output/ExitSuccess.out
   nsize: {{1 2 5}}
   test:
     requires: cuda
     suffix: test_harness_doesnt_respect_bare_requires_unless_you_add_a_suffix
   test:
     requires: hip
     suffix: test_harness_doesnt_respect_bare_requires_unless_you_add_a_suffix_2

TEST*/
