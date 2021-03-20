static const char help[] = "Tests PetscStreamScalar set/get operations\n";

#include <petscdevice.h>

static PetscErrorCode CompareHost(const PetscScalar *ref, const PetscScalar *ret, PetscScalar *valHost, PetscBool *eq)
{
  const PetscScalar l = ref ? *ref : (PetscScalar)0.0, r = *ret;

  PetscFunctionBegin;
  *valHost = l;
  if (l == r) *eq = PETSC_TRUE;
  else if (PetscIsNanScalar(l)) *eq = PetscIsNanScalar(r) ? PETSC_TRUE : PETSC_FALSE;
  else if (PetscIsInfScalar(l)) *eq = PetscIsInfScalar(r) ? PETSC_TRUE : PETSC_FALSE;
  else *eq = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
static PetscErrorCode CompareDevice(const PetscScalar *dref, const PetscScalar *ret, PetscScalar *valHost, PetscBool *eq)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  PetscScalar    ref[1];

  PetscFunctionBegin;
  cerr = cudaMemcpy(ref,dref,sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  ierr = CompareHost(ref,ret,valHost,eq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDeviceValues(const PetscScalar *hostarr, PetscScalar **devarr, PetscInt n)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaMalloc((void **)devarr,n*sizeof(PetscScalar));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(*devarr,hostarr,n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyDeviceValues(PetscScalar **dev)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaFree(*dev);CHKERRCUDA(cerr);
  *dev = NULL;
  PetscFunctionReturn(0);
}
#elif PetscDefined(HAVE_HIP)
static PetscErrorCode CompareDevice(const PetscScalar *dref, const PetscScalar *ret, PetscScalar *valHost, PetscBool *eq)
{
  PetscErrorCode ierr;
  hipError_t     herr;
  PetscScalar    ref[1];

  PetscFunctionBegin;
  herr = hipMemcpy(ref,dref,sizeof(PetscScalar),hipMemcpyDeviceToHost);CHKERRHIP(herr);
  ierr = CompareHost(ref,ret,valHost,eq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDeviceValues(const PetscScalar *hostarr, PetscScalar **devarr, PetscInt n)
{
  hipError_t herr;

  PetscFunctionBegin;
  herr = hipMalloc((void **)devarr,n*sizeof(PetscScalar));CHKERRHIP(herr);
  herr = hipMemcpy(*devarr,hostarr,n*sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyDeviceValues(PetscScalar **dev)
{
  hipError_t herr;

  PetscFunctionBegin;
  herr = hipFree(*dev);CHKERRHIP(herr);
  *dev = NULL;
  PetscFunctionReturn(0);
}
#else
static PetscErrorCode CompareDevice(const PetscScalar *dref, const PetscScalar *ret, PetscScalar *valHost, PetscBool *eq)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires a device");
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateDeviceValues(const PetscScalar *hostarr, PetscScalar **devarr, PetscInt n)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires a device");
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyDeviceValues(PetscScalar **dev)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires a device");
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode TestSetValWithMemtype(PetscStreamScalar ptest, const PetscScalar *valSet, PSSCacheType trueType, PetscMemType mtype, PetscStream pstream)
{
  const PSSCacheType types[] = {PSS_ZERO,PSS_ZERO,PSS_ONE,PSS_INF,PSS_NAN};
  PetscScalar        valRet = 0, valHost = 0;
  PetscBool          equal = PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscStreamScalarSetValue(ptest,valSet,mtype,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(ptest,&valRet,pstream);CHKERRQ(ierr);
  switch (mtype) {
  case PETSC_MEMTYPE_HOST:
    ierr = CompareHost(valSet,&valRet,&valHost,&equal);CHKERRQ(ierr);
    break;
  case PETSC_MEMTYPE_DEVICE:
    if (valSet) {ierr = CompareDevice(valSet,&valRet,&valHost,&equal);CHKERRQ(ierr);}
    /* is valSet is NULL, then it is zero, easier to just default to host compare where this is handled rather than
     introduce messy logic to device compare */
    else {ierr = CompareHost(valSet,&valRet,&valHost,&equal);CHKERRQ(ierr);}
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Memtype %s has no valid comparison function",PetscMemTypeHost(mtype)?"host":"device");
    break;
  }
  if (!equal) {
    SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values do not match for memtype %s. Reference: %.10g+%gi != returned: %.10g+%gi",PetscMemTypeHost(mtype)?"host":"device",(double)PetscRealPart(valHost),(double)PetscImaginaryPart(valHost),(double)PetscRealPart(valRet),(double)PetscImaginaryPart(valRet));
  }
  for (PetscInt i = 0; i < PSS_CACHE_MAX; ++i) {
    PetscBool res;

    ierr = PetscStreamScalarGetInfo(ptest,types[i],PETSC_FALSE,&res,pstream);CHKERRQ(ierr);
    if ((types[i] == trueType) && !res) {
      SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamScalar cache corrupted for memtype %s, %s%s should be PETSC_TRUE for %s%s",PetscMemTypeHost(mtype)?"host":"device",PSSCacheTypes[PSS_CACHE_MAX+1],PSSCacheTypes[types[i]],PSSCacheTypes[PSS_CACHE_MAX+1],PSSCacheTypes[trueType]);
    } else if ((types[i] != trueType) && res) {
      SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamScalar cache corrupted for memtype %s, %s%s should be PETSC_FALSE for %s%s",PetscMemTypeHost(mtype)?"host":"device",PSSCacheTypes[PSS_CACHE_MAX+1],PSSCacheTypes[types[i]],PSSCacheTypes[PSS_CACHE_MAX+1],PSSCacheTypes[trueType]);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
#if PetscDefined(HAVE_CUDA)
  PetscStreamType   itype = PETSCSTREAMCUDA;
#elif PetscDefined(HAVE_HIP)
  PetscStreamType   itype = PETSCSTREAMHIP;
#else
  PetscStreamType   itype = "invalidType";
#endif
  PetscStreamType   type;
  /* Obfuscate the fact that we are dividing by zero to some overzealous compilers */
  const PetscReal   zero = PetscRealConstant(0.0), one = PetscRealConstant(1.0);
  const PetscScalar hArr[5] = {zero,one,-one/zero,one/zero,zero/zero};
  PetscScalar       *dArr = NULL;
  PetscStream       pstream;
  PetscStreamScalar pscal;
  PetscErrorCode    ierr;
  MPI_Comm          comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = PetscStreamCreate(&pstream);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(pstream,PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
  ierr = PetscStreamSetType(pstream,itype);CHKERRQ(ierr);
  ierr = PetscStreamSetFromOptions(comm,"",pstream);CHKERRQ(ierr);

  ierr = PetscStreamGetType(pstream,&type);CHKERRQ(ierr);
  ierr = PetscStreamScalarCreate(&pscal);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetType(pscal,type);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetUp(pscal);CHKERRQ(ierr);

  ierr = TestSetValWithMemtype(pscal,NULL,PSS_ZERO,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,hArr,PSS_ZERO,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,hArr+1,PSS_ONE,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,hArr+2,PSS_INF,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,hArr+3,PSS_INF,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,hArr+4,PSS_NAN,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);

  ierr = CreateDeviceValues(hArr,&dArr,5);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,NULL,PSS_ZERO,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,dArr,PSS_ZERO,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,dArr+1,PSS_ONE,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,dArr+2,PSS_INF,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,dArr+3,PSS_INF,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscal,dArr+4,PSS_NAN,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = DestroyDeviceValues(&dArr);CHKERRQ(ierr);

  ierr = PetscPrintf(comm,"All operations completed successfully\n");CHKERRQ(ierr);
  ierr = PetscStreamScalarDestroy(&pscal);CHKERRQ(ierr);
  ierr = PetscStreamDestroy(&pstream);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 test:
   requires: cuda
   suffix: cuda
   args: -stream_type cuda

 test:
   requires: hip
   suffix: hip
   args: -stream_type hip

TEST*/
