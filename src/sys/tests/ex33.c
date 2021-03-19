static const char help[] = "Tests PetscStreamScalar set/get operations\n";

#include <petscdevice.h>

PETSC_STATIC_INLINE PetscErrorCode CompareHost(const PetscScalar *ref, const PetscScalar *ret, PetscScalar *valHost, PetscBool *eq)
{
  const PetscScalar l = *ref, r = *ret;

  PetscFunctionBegin;
  *valHost = l;
  *eq = l == r ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if PetscDefined(HAVE_CUDA)
PETSC_STATIC_INLINE PetscErrorCode CompareDevice(const PetscScalar *dref, const PetscScalar *ret, PetscScalar *valhost, PetscBool *eq)
{
  cudaError_t cerr;
  PetscScalar ref[1];

  PetscFunctionBegin;
  cerr = cudaMemcpy(ref,dref,sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  *valHost = ref[0];
  *eq = ref[0] == *ret ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode CreateDeviceValues(const PetscScalar *hostarr, PetscScalar **devarr, PetscInt n)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaMalloc((void **)devarr,n*sizeof(PetscScalar));CHKERRCUDA(cerr);
  cerr = cudaMemcpy(devarr,hostarr,n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DestroyDeviceValues(PetscScalar **dev)
{
  cudaError_t cerr;

  PetscFunctionBegin;
  cerr = cudaFree(*dev);CHKERRCUDA(cerr);
  *dev = NULL;
  PetscFunctionReturn(0);
}
#elif PetscDefined(HAVE_HIP)
PETSC_STATIC_INLINE PetscErrorCode CompareDevice(const PetscScalar *dref, const PetscScalar ret*, PetscScalar *valhost, PetscBool *eq)
{
  hipError_t  herr;
  PetscScalar ref[1];

  PetscFunctionBegin;
  herr = hipMemcpy(ref,dref,sizeof(PetscScalar),hipMemcpyDeviceToHost);CHKERRHIP(herr);
  *valHost = ref[0];
  *eq = ref[0] == *ret ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode CreateDeviceValues(const PetscScalar *hostarr, PetscScalar **devarr, PetscInt n)
{
  hipError_t herr;

  PetscFunctionBegin;
  herr = hipMalloc((void **)devarr,n*sizeof(PetscScalar));CHKERRHIP(herr);
  herr = hipMemcpy(devarr,hostarr,n*sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(herr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DestroyDeviceValues(PetscScalar **dev)
{
  hipError_t herr;

  PetscFunctionBegin;
  herr = hipFree(*dev);CHKERRHIP(herr);
  *dev = NULL;
  PetscFunctionReturn(0);
}
#else
PETSC_STATIC_INLINE PetscErrorCode CompareDevice(const PetscScalar *dref, const PetscScalar *ret, PetscScalar *valhost, PetscBool *eq)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires a device");
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode CreateDeviceValues(const PetscScalar *hostarr, PetscScalar **devarr, PetscInt n)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires a device");
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DestroyDeviceValues(PetscScalar **dev)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This test requires a device");
  PetscFunctionReturn(0);
}
#endif

PETSC_STATIC_INLINE PetscErrorCode TestSetValWithMemtype(PetscStreamScalar ptest, const PetscScalar *valSet, PSSCacheType trueType, PetscMemType mtype, PetscStream pstream)
{
  const PSSCacheType types[] = {PSS_ZERO,PSS_ONE,PSS_INF,PSS_NAN};
  PetscScalar        valRet,valHost;
  PetscBool          equal;
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  ierr = PetscStreamScalarSetValue(ptest,valSet,mtype,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(ptest,&valRet,pstream);CHKERRQ(ierr);
  switch (mtype) {
  case PETSC_MEMTYPE_HOST:
    ierr = CompareHost(valSet,&valRet,&valHost,&equal);CHKERRQ(ierr);
    break;
  case PETSC_MEMTYPE_DEVICE:
    ierr = CompareDevice(valSet,&valRet,&valHost,&equal);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Memtype selected has no valid comparison function");
    break;
  }
  if (!equal) {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values do not match for host memtype. Reference: %.10g+%gi != returned: %.10g+%gi",(double)PetscRealPart(valHost),(double)PetscImaginaryPart(valHost),(double)PetscRealPart(valRet),(double)PetscImaginaryPart(valRet));
  }
  for (PetscInt i = 0; i < PSS_CACHE_MAX; ++i) {
    PetscBool res;

    ierr = PetscStreamScalarGetInfo(ptest,types[i],PETSC_FALSE,&res,pstream);CHKERRQ(ierr);
    if ((types[i] == trueType) && !res) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamScalar cache corrupted, %s%s should be PETSC_TRUE",PSSCacheTypes[PSS_CACHE_MAX+1],PSSCacheTypes[types[i]]);
    } else if (res) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscStreamScalar cache corrupted, %s%s should be PETSC_FALSE",PSSCacheTypes[PSS_CACHE_MAX+1],PSSCacheTypes[types[i]]);
    }
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscStreamType   type;
  const PetscScalar hArr[5] = {0.0,1.0,-1.0/0.0,1.0/0.0,0.0/0.0};
  PetscScalar       *dArr;
  PetscStream       pstream;
  PetscStreamScalar pscalx;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscStreamCreate(&pstream);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(pstream,PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
  ierr = PetscStreamSetType(pstream,PETSCSTREAMCUDA);CHKERRQ(ierr);
  ierr = PetscStreamSetFromOptions(PETSC_COMM_WORLD,"",pstream);CHKERRQ(ierr);

  ierr = PetscStreamGetType(pstream,&type);CHKERRQ(ierr);
  ierr = PetscStreamScalarCreate(&pscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetType(pscalx,type);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetUp(pscalx);CHKERRQ(ierr);

  ierr = TestSetValWithMemtype(pscalx,hArr,PSS_ZERO,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,hArr+1,PSS_ONE,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,hArr+2,PSS_INF,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,hArr+3,PSS_INF,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,hArr+4,PSS_NAN,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);

  ierr = CreateDeviceValues(hArr,&dArr,5);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,dArr,PSS_ZERO,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,dArr+1,PSS_ONE,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,dArr+2,PSS_INF,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,dArr+3,PSS_INF,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = TestSetValWithMemtype(pscalx,dArr+4,PSS_NAN,PETSC_MEMTYPE_DEVICE,pstream);CHKERRQ(ierr);
  ierr = DestroyDeviceValues(&dArr);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"All operations completed successfully\n");CHKERRQ(ierr);
  ierr = PetscStreamScalarDestroy(&pscalx);CHKERRQ(ierr);
  ierr = PetscStreamDestroy(&pstream);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 testset:
   requires: {{cuda hip}}
   suffix: {{cuda hip}}
   args: -stream_type {{cuda hip}}
TEST*/
