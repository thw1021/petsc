static const char help[] = "Tests creation and descruction of PetscDeviceContext.\n\n";

#include <petscdevice.h>

/* test duplication creates the same object type */
static PetscErrorCode testDuplicate(PetscDeviceContext dctx)
{
  PetscStreamType        stype,dupSType;
  PetscDeviceContext     dtmp,ddup;
  PetscDeviceContextType type,dupType;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextGetType(dctx,&type);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetStreamType(dctx,&stype);CHKERRQ(ierr);

  /* create manually first */
  ierr = PetscDeviceContextCreate(&dtmp);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetType(dtmp,type);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetStreamType(dtmp,stype);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(dtmp);CHKERRQ(ierr);

  /* duplicate */
  ierr = PetscDeviceContextDuplicate(dctx,&ddup);CHKERRQ(ierr);

  ierr = PetscDeviceContextGetType(ddup,&dupType);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetType(dtmp,&type);CHKERRQ(ierr);
  if (dupType != type) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscDeviceContextTypes %s and %s do not match",dupType,type);

  ierr = PetscDeviceContextGetStreamType(ddup,&dupSType);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetStreamType(dtmp,&stype);CHKERRQ(ierr);
  if (dupSType != stype) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamTypes %d and %d do not match",dupSType,stype);

  ierr = PetscDeviceContextDestroy(&dtmp);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&ddup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode testForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* mostly for valgrind to catch errors */
  ierr = PetscDeviceContextFork(dctx,n,&sub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);

  ierr = PetscDeviceContextFork(dctx,n+1,&sub);CHKERRQ(ierr);
  /* test nesting */
  {
    const PetscInt      nsub = 4;
    PetscDeviceContext *subsub;

    ierr = PetscDeviceContextFork(sub[0],nsub,&subsub);CHKERRQ(ierr);
    /* join on a different sub */
    ierr = PetscDeviceContextJoin(sub[1],nsub-2,PETSC_DEVICE_CONTEXT_JOIN_SYNC,&subsub);CHKERRQ(ierr);
    ierr = PetscDeviceContextJoin(sub[0],nsub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub);CHKERRQ(ierr);
  }
  /* join a subset */
  ierr = PetscDeviceContextJoin(dctx,n-1,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&subsub);CHKERRQ(ierr);
  /* back to the ether from whence they came */
  ierr = PetscDeviceContextJoin(dctx,n+1,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define CHKERRABORT2(ierr)                                              \
  do {                                                                  \
    PetscErrorCode ierr__ = (ierr);                                     \
    if (PetscUnlikely(ierr__)) {                                        \
      int init,finalized;                                               \
      PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr__,PETSC_ERROR_REPEAT," "); \
      MPI_Initialized(&init);                                           \
      MPI_Finalized(&finalized);                                        \
      if (init && !finalized) MPI_Abort(MPI_COMM_WORLD,ierr);           \
      else exit(ierr);                                                  \
    }                                                                   \
  } while (0)

int main(int argc, char *argv[])
{
  PetscDeviceContextType defaultType,type,stype;
  PetscStreamType        defaultSType;
  PetscDeviceContext     dctx;
  PetscErrorCode         ierr;

  /* test setting different root context settings */
  ierr = PetscDeviceContextGetDefaultRootContextSettings(&defaultType,&defaultSType);CHKERRABORT2(ierr);
  /* purposefully set to hip */
  ierr = PetscDeviceContextSetDefaultRootContextSettings(PETSCDEVICECONTEXTHIP,PETSC_STREAM_GLOBAL_BLOCKING);CHKERRABORT2(ierr);
  ierr = PetscDeviceContextGetDefaultContextSettings(&type,&stype);CHKERRABORT2(ierr);
  if (type != PETSCDEVICECONTEXTHIP) CHKERRABORT2(PETSC_ERR_PLIB);
  if (stype != PETSC_STREAM_GLOBAL_BLOCKING) CHKERRABORT2(PETSC_ERR_PLIB);
  /* reset everything back to normal */
  ierr = PetscDeviceContextSetDefaultRootContextSettings(type,stype);CHKERRABORT2(ierr);

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscDeviceContextGetDefaultContextSettings(&type,&stype);CHKERRQ(ierr);
  if (type != defaultType) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Default root context type does not match default type");
  if (stype != defaultSType) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Default root context stream type does not match default stream type");

  /* check getting and setting */
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  /* check one last time that these wen back to normal */
  ierr = PetscDeviceContextGetType(dctx,&type);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetStreamType(dctx,&stype);CHKERRQ(ierr);
  if (type != defaultType) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Default root context type does not match default type");
  if (stype != defaultSType) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Default root context stream type does not match default stream type");
  ierr = PetscDeviceContextSetCurrentContext(dctx);CHKERRQ(ierr);

  /* we keep this around from now on */
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);

  /* tests */
  ierr = testDuplicate(dctx);CHKERRQ(ierr);
  ierr = testForkJoin(dctx);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"SUCCESS\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
