static const char help[] = "Tests PetscStreamScalar arithmetic operations\n";

#include <petscdevice.h>

static PetscErrorCode PetscCheckCloseScalar(PetscScalar dret, PetscScalar host)
{
  const PetscReal dr = PetscRealPart(dret),di = PetscImaginaryPart(dret);
  const PetscReal hr = PetscRealPart(host),hi = PetscImaginaryPart(host);
  const PetscBool closeR = PetscIsCloseAtTol(dr,hr,1e-5,1e-7);
  const PetscBool closeI = PetscIsCloseAtTol(di,hi,1e-5,1e-7);

  PetscFunctionBegin;
  if (!(closeR && closeI)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Returned value: %.10g+%.10gi != Host value: %.10g+%.10gi",(double)dr,(double)di,(double)hr,(double)hi);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestAYDX(PetscRandom rand, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  const PetscScalar one=1.0;
  PetscScalar       alpha,hscalx,hscaly,pscalret;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);

  /* alpha = 0.0, y = whatever, should be 0.0 */
  alpha = (PetscScalar)0.0;
  ierr = PetscStreamScalarAYDX(alpha,pscaly,pscalx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,alpha);CHKERRQ(ierr);

  /* alpha = random, y = x, should be alpha */
  ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAYDX(alpha,pscalx,pscalx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,alpha);CHKERRQ(ierr);

  /* alpha = random, y = 1.0, should be alpha/x */
  ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscaly,&one,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  hscalx = alpha/hscalx;
  ierr = PetscStreamScalarAYDX(alpha,pscaly,pscalx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);

  /* Fully random, should match host */
  ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&hscaly);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscaly,&hscaly,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  hscalx = alpha*hscaly/hscalx;
  ierr = PetscStreamScalarAYDX(alpha,pscaly,pscalx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestAXTY(PetscRandom rand, PetscStreamScalar pscalx, PetscStreamScalar pscaly, PetscStream pstream)
{
  const PetscScalar one=1.0;
  PetscScalar       alpha,hscalx,hscaly,pscalret;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscaly,&one,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);

  /* alpha = 1.0, y = 1.0, should be NO-OP */
  alpha = (PetscScalar)1.0;
  ierr = PetscStreamScalarAXTY(alpha,pscalx,pscaly,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);

  /* alpha = 1.0, y = NULL (i.e. 1.0), should be NO-OP */
  ierr = PetscStreamScalarAXTY(alpha,pscalx,NULL,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);

  /* alpha = 0.0, y = whatever, should be 0.0 */
  alpha = (PetscScalar)0.0;
  ierr = PetscStreamScalarAXTY(alpha,pscalx,pscaly,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,alpha);CHKERRQ(ierr);

  /* alpha = random, y = x, should be alpha*x */
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  hscalx = alpha*hscalx;
  ierr = PetscStreamScalarAXTY(alpha,pscalx,NULL,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);

  /* alpha = random, y = x, should be alpha*x**2 */
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  hscalx = alpha*hscalx*hscalx;
  ierr = PetscStreamScalarAXTY(alpha,pscalx,pscalx,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);

  /* Fully random and fully general should match host */
  ierr = PetscRandomGetValue(rand,&hscalx);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&hscaly);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&alpha);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscalx,&hscalx,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetValue(pscaly,&hscaly,PETSC_MEMTYPE_HOST,pstream);CHKERRQ(ierr);
  hscalx = alpha*hscalx*hscaly;
  ierr = PetscStreamScalarAXTY(alpha,pscalx,pscaly,pstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarAwait(pscalx,&pscalret,pstream);CHKERRQ(ierr);
  ierr = PetscCheckCloseScalar(pscalret,hscalx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscStreamType   itype = PETSCSTREAMCUDA,type;
  PetscErrorCode    ierr;
  PetscStreamScalar pscalx,pscaly;
  PetscStream       pstream;
  PetscRandom       rand;
  MPI_Comm          comm;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = PetscRandomViewFromOptions(rand,NULL,"-rand_view");CHKERRQ(ierr);

  ierr = PetscStreamCreate(&pstream);CHKERRQ(ierr);
  ierr = PetscStreamSetMode(pstream,PETSC_STREAM_DEFAULT_BLOCKING);CHKERRQ(ierr);
  ierr = PetscStreamSetType(pstream,itype);CHKERRQ(ierr);
  ierr = PetscStreamSetFromOptions(comm,"",pstream);CHKERRQ(ierr);

  ierr = PetscStreamGetType(pstream,&type);CHKERRQ(ierr);
  ierr = PetscStreamScalarCreate(&pscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetType(pscalx,type);CHKERRQ(ierr);
  ierr = PetscStreamScalarSetUp(pscalx);CHKERRQ(ierr);
  ierr = PetscStreamScalarDuplicate(pscalx,&pscaly);CHKERRQ(ierr);

  ierr = TestAXTY(rand,pscalx,pscaly,pstream);CHKERRQ(ierr);
  ierr = TestAYDX(rand,pscalx,pscaly,pstream);CHKERRQ(ierr);

  ierr = PetscPrintf(comm,"All operations completed successfully\n");CHKERRQ(ierr);
  ierr = PetscStreamScalarDestroy(&pscaly);CHKERRQ(ierr);
  ierr = PetscStreamScalarDestroy(&pscalx);CHKERRQ(ierr);
  ierr = PetscStreamDestroy(&pstream);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 test:
   requires: cuda
   suffix: cuda
   args: -rand_view -stream_type cuda

TEST*/
