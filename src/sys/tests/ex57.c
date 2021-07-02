static const char help[] = "Tests creation and descruction of PetscDeviceContext.\n\n";

#include <petscdevice.h>

/* test duplication creates the same object type */
static PetscErrorCode testDuplicate(PetscDeviceContext dctx)
{
  PetscBool              same;
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
  ierr = PetscStrcmp(type,dupType,&same);CHKERRQ(ierr);
  if (!same) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"PetscDeviceContextTypes %s and %s do not match",dupType,type);

  ierr = PetscDeviceContextGetStreamType(ddup,&dupSType);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetStreamType(dtmp,&stype);CHKERRQ(ierr);
  if (dupSType != stype) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamTypes %d and %d do not match",dupSType,stype);

  ierr = PetscDeviceContextDestroy(&dtmp);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&ddup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode testNestedForkJoin(PetscDeviceContext *sub)
{
  const PetscInt      nsub = 4;
  PetscDeviceContext *subsub;
  PetscDeviceContext  parCtx;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextGetCurrentContext(&parCtx);CHKERRQ(ierr);
  if (parCtx != sub[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Current global context does not match expected global context");
  ierr = PetscDeviceContextFork(parCtx,nsub,&subsub);CHKERRQ(ierr);
  /* join on a different sub */
  ierr = PetscDeviceContextJoin(sub[1],nsub-2,PETSC_DEVICE_CONTEXT_JOIN_SYNC,&subsub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(parCtx,nsub,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&subsub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* test fork-join */
static PetscErrorCode testForkJoin(PetscDeviceContext dctx)
{
  PetscDeviceContext *sub;
  const PetscInt      n = 10;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* mostly for valgrind to catch errors */
  ierr = PetscDeviceContextFork(dctx,n,&sub);CHKERRQ(ierr);
  ierr = PetscDeviceContextJoin(dctx,n,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);

  /* create some children */
  ierr = PetscDeviceContextFork(dctx,n+1,&sub);CHKERRQ(ierr);

  /* make the first child the new current context, and test forking within nested function */
  ierr = PetscDeviceContextSetCurrentContext(sub[0]);CHKERRQ(ierr);
  ierr = testNestedForkJoin(sub);CHKERRQ(ierr);
  /* should always reset global context when finished */
  ierr = PetscDeviceContextSetCurrentContext(dctx);CHKERRQ(ierr);

  /* join a subset */
  ierr = PetscDeviceContextJoin(dctx,n-1,PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC,&sub);CHKERRQ(ierr);
  /* back to the ether from whence they came */
  ierr = PetscDeviceContextJoin(dctx,n+1,PETSC_DEVICE_CONTEXT_JOIN_DESTROY,&sub);CHKERRQ(ierr);
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
#if PetscDefined(HAVE_CUDA)
  /* we have cuda backend only */
  PetscDeviceContextType wrongType = PETSCDEVICECONTEXTHIP;
#else
  /* we may or may not have hip */
  PetscDeviceContextType wrongType= PETSCDEVICECONTEXTCUDA;
#endif
  PetscDeviceContextType defaultType,type;
  PetscStreamType        defaultSType,stype;
  PetscDeviceContext     dctx;
  PetscBool              same;
  PetscErrorCode         ierr;

  /* test setting different root context settings */
  ierr = PetscDeviceContextGetDefaultRootContextSettings(&defaultType,&defaultSType);CHKERRABORT2(ierr);
  /* purposefully set to wrong backend type */
  ierr = PetscDeviceContextSetDefaultRootContextSettings(wrongType,PETSC_STREAM_GLOBAL_BLOCKING);CHKERRABORT2(ierr);
  ierr = PetscDeviceContextGetDefaultRootContextSettings(&type,&stype);CHKERRABORT2(ierr);
  ierr = PetscStrcmp(type,wrongType,&same);CHKERRABORT2(ierr);
  if (!same) CHKERRABORT2(PETSC_ERR_ARG_NOTSAMETYPE);
  if (stype != PETSC_STREAM_GLOBAL_BLOCKING) CHKERRABORT2(PETSC_ERR_ARG_NOTSAMETYPE);
  /* reset everything back to normal */
  ierr = PetscDeviceContextSetDefaultRootContextSettings(defaultType,defaultSType);CHKERRABORT2(ierr);

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = PetscDeviceContextGetDefaultRootContextSettings(&type,&stype);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,defaultType,&same);CHKERRQ(ierr);
  if (!same) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"Default root context type %s does not match default type %s",type,defaultType);
  if (stype != defaultSType) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"Default root context stream type does not match default stream type");

  /* check getting and setting */
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);
  /* check one last time that these wen back to normal */
  ierr = PetscDeviceContextGetType(dctx,&type);CHKERRQ(ierr);
  ierr = PetscDeviceContextGetStreamType(dctx,&stype);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,defaultType,&same);CHKERRQ(ierr);
  if (!same) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"Default root context type %s does not match default type %s",type,defaultType);
  if (stype != defaultSType) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NOTSAMETYPE,"Default root context stream type does not match default stream type");
  ierr = PetscDeviceContextSetCurrentContext(dctx);CHKERRQ(ierr);

  /* we keep this around from now on */
  ierr = PetscDeviceContextGetCurrentContext(&dctx);CHKERRQ(ierr);

  /* tests */
  ierr = testDuplicate(dctx);CHKERRQ(ierr);
  ierr = testForkJoin(dctx);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"EXIT_SUCCESS\n");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: define(PETSC_HAVE_CXX_DIALECT_CXX11)

  test:
    requires: cuda
    suffix: cuda

  test:
    requires: hip
    suffix: hip
TEST*/
