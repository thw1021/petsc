
/*
   Defines some vector operation functions that are shared by
   sequential and parallel vectors.
*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/kernels/petscaxpy.h>
#include <petscdevice.h>

#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#include <../src/vec/vec/impls/seq/ftn-kernels/fmdot.h>
PetscErrorCode VecMDot_Seq(Vec xin, PetscManagedInt nv, const Vec yin[], PetscManagedScalar zt, PetscDeviceContext dctx)
{
  PetscInt          *nvptr;
  PetscInt          i,nv_rem,n = xin->map->n;
  PetscScalar       sum0,sum1,sum2,sum3;
  PetscScalar       *zptr;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x;
  Vec               *yy;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx,zt,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&zptr));
  sum0  = 0.0;
  sum1  = 0.0;
  sum2  = 0.0;

  i      = *nvptr;
  nv_rem = i&0x3;
  yy     = (Vec*)yin;
  PetscCall(VecGetArrayRead(xin,&x));

  switch (nv_rem) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    zptr[0] = sum0;
    zptr[1] = sum1;
    zptr[2] = sum2;
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    zptr[0] = sum0;
    zptr[1] = sum1;
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    fortranmdot1_(x,yy0,&n,&sum0);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    zptr[0] = sum0;
    break;
  case 0:
    break;
  }
  zptr  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    PetscCall(VecGetArrayRead(yy[3],&yy3));
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    PetscCall(VecRestoreArrayRead(yy[3],&yy3));
    yy  += 4;
    zptr[0] = sum0;
    zptr[1] = sum1;
    zptr[2] = sum2;
    zptr[3] = sum3;
    zptr   += 4;
    i   -= 4;
  }
  PetscCall(VecRestoreArrayRead(xin,&x));
  PetscCall(PetscLogFlops(PetscMax((*nvptr)*(2.0*xin->map->n-1),0.0)));
  PetscFunctionReturn(0);
}

#else
PetscErrorCode VecMDot_Seq(Vec xin, PetscManagedInt nv, const Vec yin[], PetscManagedScalar zt, PetscDeviceContext dctx)
{
  const PetscInt     n    = xin->map->n;
  PetscInt           j    = n,nv_rem,j_rem,i;
  PetscScalar        sum0 = 0.,sum1 = 0.,sum2 = 0.,sum3,x0,x1,x2,x3;
  Vec               *yy   = (Vec*)yin;
  PetscScalar       *z;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  PetscInt          *nvptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx,zt,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,PETSC_TRUE,&z));
  PetscCall(VecGetArrayRead(xin,&xbase));

  x = xbase;
  i = *nvptr;

  switch (nv_rem = i&0x3) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
      sum2 += x2*PetscConj(yy2[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
      sum2 += x1*PetscConj(yy2[1]);
    case 1:
      x0    = x[0];
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
      sum2 += x0*PetscConj(yy2[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
    case 1:
      x0    = x[0];
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    switch (j_rem=j&0x3) {
    case 3:
      x2 = x[2]; sum0 += x2*PetscConj(yy0[2]);
    case 2:
      x1 = x[1]; sum0 += x1*PetscConj(yy0[1]);
    case 1:
      x0 = x[0]; sum0 += x0*PetscConj(yy0[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*PetscConj(yy0[0]) + x[1]*PetscConj(yy0[1])
            + x[2]*PetscConj(yy0[2]) + x[3]*PetscConj(yy0[3]);
      yy0  +=4;
      j    -= 4; x+=4;
    }
    z[0] = sum0;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    PetscCall(VecGetArrayRead(yy[3],&yy3));

    j = n;
    x = xbase;
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
      sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
      sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
    case 1:
      x0    = x[0];
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
      sum2 += x0*PetscConj(yy2[0]); sum3 += x0*PetscConj(yy3[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      sum3 += x0*PetscConj(yy3[0]) + x1*PetscConj(yy3[1]) + x2*PetscConj(yy3[2]) + x3*PetscConj(yy3[3]); yy3+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    PetscCall(VecRestoreArrayRead(yy[3],&yy3));
    yy  += 4;
  }
  PetscCall(VecRestoreArrayRead(xin,&xbase));
  PetscCall(PetscLogFlops(PetscMax((*nvptr)*(2.0*xin->map->n-1),0.0)));
  PetscFunctionReturn(0);
}
#endif

/* ----------------------------------------------------------------------------*/
PetscErrorCode VecMTDot_Seq(Vec xin, PetscManagedInt nv, const Vec yin[], PetscManagedScalar zt, PetscDeviceContext dctx)
{
  PetscInt          *nvptr;
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  PetscScalar       *z;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  Vec               *yy;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx,zt,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_WRITE,PETSC_TRUE,&z));
  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = *nvptr;
  nv_rem = i&0x3;
  yy     = (Vec*)yin;
  j      = n;
  PetscCall(VecGetArrayRead(xin,&xbase));
  x      = xbase;

  switch (nv_rem) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
      sum2 += x2*yy2[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
      sum2 += x1*yy2[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
      sum2 += x0*yy2[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    switch (j_rem=j&0x3) {
    case 3:
      x2 = x[2]; sum0 += x2*yy0[2];
    case 2:
      x1 = x[1]; sum0 += x1*yy0[1];
    case 1:
      x0 = x[0]; sum0 += x0*yy0[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*yy0[0] + x[1]*yy0[1] + x[2]*yy0[2] + x[3]*yy0[3]; yy0+=4;
      j    -= 4; x+=4;
    }
    z[0] = sum0;

    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    PetscCall(VecGetArrayRead(yy[0],&yy0));
    PetscCall(VecGetArrayRead(yy[1],&yy1));
    PetscCall(VecGetArrayRead(yy[2],&yy2));
    PetscCall(VecGetArrayRead(yy[3],&yy3));
    x    = xbase;

    j = n;
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
      sum2 += x2*yy2[2]; sum3 += x2*yy3[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
      sum2 += x1*yy2[1]; sum3 += x1*yy3[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
      sum2 += x0*yy2[0]; sum3 += x0*yy3[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      sum3 += x0*yy3[0] + x1*yy3[1] + x2*yy3[2] + x3*yy3[3]; yy3+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    PetscCall(VecRestoreArrayRead(yy[0],&yy0));
    PetscCall(VecRestoreArrayRead(yy[1],&yy1));
    PetscCall(VecRestoreArrayRead(yy[2],&yy2));
    PetscCall(VecRestoreArrayRead(yy[3],&yy3));
    yy  += 4;
  }
  PetscCall(VecRestoreArrayRead(xin,&xbase));
  PetscCall(PetscLogFlops(PetscMax((*nvptr)*(2.0*xin->map->n-1),0.0)));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMinMax_Seq(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx, PetscReal minmax, PetscBool(*const cmp)(PetscReal,PetscReal))
{
  const PetscInt n = xin->map->n;
  PetscInt       j = -1;

  PetscFunctionBegin;
  if (n) {
    const PetscScalar *xx;

    PetscCall(VecGetArrayRead(xin,&xx));
    minmax = PetscRealPart(xx[(j = 0)]);
    for (PetscInt i = 1; i < n; ++i) {
      const PetscReal tmp = PetscRealPart(xx[i]);
      if (cmp(tmp,minmax)) { j = i; minmax = tmp; }
    }
    PetscCall(VecRestoreArrayRead(xin,&xx));
  }
  PetscCall(PetscManagedRealSetValues(dctx,z,PETSC_MEMTYPE_HOST,&minmax,1));
  if (idx) PetscCall(PetscManagedIntSetValues(dctx,idx,PETSC_MEMTYPE_HOST,&j,1));
  PetscFunctionReturn(0);
}

static PetscBool VecMax_Seq_GT(PetscReal l, PetscReal r) { return (PetscBool)(l > r); }

PetscErrorCode VecMax_Seq(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(VecMinMax_Seq(xin,idx,z,dctx,PETSC_MIN_REAL,VecMax_Seq_GT));
  PetscFunctionReturn(0);
}

static PetscBool VecMin_Seq_LT(PetscReal l, PetscReal r) { return (PetscBool)(l < r); }

PetscErrorCode VecMin_Seq(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(VecMinMax_Seq(xin,idx,z,dctx,PETSC_MAX_REAL,VecMin_Seq_LT));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSet_Seq(Vec xin, PetscManagedScalar alpha, PetscDeviceContext dctx)
{
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&aptr));
  {
    const PetscScalar  aval = *aptr;
    const PetscInt     n    = xin->map->n;
    PetscScalar       *xx;

    PetscCall(VecGetArrayWrite(xin,&xx));
    if (aval == (PetscScalar)0.0) {
      PetscCall(PetscArrayzero(xx,n));
    } else {
      for (PetscInt i = 0; i < n; ++i) xx[i] = aval;
    }
    PetscCall(VecRestoreArrayWrite(xin,&xx));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_Seq(Vec xin, PetscManagedInt nvt, PetscManagedScalar alpha, Vec *y, PetscDeviceContext dctx)
{
  PetscInt    *nvptr;
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedIntGetValues(dctx,nvt,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&aptr));
  {
    const PetscInt     nv = *nvptr,j_rem = nv&0x3,n = xin->map->n;
    const PetscScalar *yptr[4];
    PetscScalar       *xx;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,**yptr,*aptr)
#endif
    PetscCall(PetscLogFlops(nv*2.0*n));
    PetscCall(VecGetArray(xin,&xx));
    for (PetscInt i = 0; i < j_rem; ++i) PetscCall(VecGetArrayRead(y[i],yptr+i));
    switch (j_rem) {
    case 3:
      PetscKernelAXPY3(xx,aptr[0],aptr[1],aptr[2],yptr[0],yptr[1],yptr[2],n);
      break;
    case 2:
      PetscKernelAXPY2(xx,aptr[0],aptr[1],yptr[0],yptr[1],n);
      break;
    case 1:
      PetscKernelAXPY(xx,aptr[0],yptr[0],n);
    default:
      break;
    }
    for (PetscInt i = 0; i < j_rem; ++i) PetscCall(VecRestoreArrayRead(y[i],yptr+i));
    aptr += j_rem; y += j_rem;
    for (PetscInt j = j_rem,inc = 4; j < nv; j += inc, aptr += inc, y += inc) {
      for (PetscInt i = 0; i < inc; ++i) PetscCall(VecGetArrayRead(y[i],yptr+i));
      PetscKernelAXPY4(xx,aptr[0],aptr[1],aptr[2],aptr[3],yptr[0],yptr[1],yptr[2],yptr[3],n);
      for (PetscInt i = 0; i < inc; ++i) PetscCall(VecRestoreArrayRead(y[i],yptr+i));
    }
    PetscCall(VecRestoreArray(xin,&xx));
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/faypx.h>

PetscErrorCode VecAYPX_Seq(Vec yin, PetscManagedScalar alpha, Vec xin, PetscDeviceContext dctx)
{
  PetscScalar *aptr;

  PetscFunctionBegin;
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&aptr));
  {
    const PetscScalar aval = *aptr;

    if (aval == (PetscScalar)0.0) {
      PetscCall(VecCopy(xin,yin));
    } else if (aval == (PetscScalar)1.0) {
      PetscCall(VecAXPY_Seq(yin,alpha,xin,dctx));
    } else {
      const PetscInt     n = yin->map->n;
      const PetscScalar *xx;
      PetscScalar       *yy;

      PetscCall(VecGetArrayRead(xin,&xx));
      PetscCall(VecGetArray(yin,&yy));
      if (aval == (PetscScalar)-1.0) {
        for (PetscInt i = 0; i < n; ++i) yy[i] = xx[i] - yy[i];
        PetscCall(PetscLogFlops(n));
      } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
        fortranaypx_(&n,aptr,xx,yy);
#else
        for (PetscInt i = 0; i < n; ++i) yy[i] = xx[i] + aval*yy[i];
#endif
        PetscCall(PetscLogFlops(2*n));
      }
      PetscCall(VecRestoreArrayRead(xin,&xx));
      PetscCall(VecRestoreArray(yin,&yy));
    }
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fwaxpy.h>
/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
   to be slower than a regular C loop.  Hence,we do not include it.
   void ?zaxpy(int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*);
*/

PetscErrorCode VecWAXPY_Seq(Vec win, PetscManagedScalar alpha, Vec xin, Vec yin, PetscDeviceContext dctx)
{
  const PetscInt     n = win->map->n;
  PetscScalar       *ww,*aptr;
  const PetscScalar *yy,*xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  PetscCall(VecGetArrayRead(yin,&yy));
  PetscCall(VecGetArray(win,&ww));
  PetscCall(PetscManagedScalarGetValues(dctx,alpha,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&aptr));
  if (*aptr == (PetscScalar)1.0) {
    PetscCall(PetscLogFlops(n));
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (PetscInt i = 0; i < n; ++i) ww[i] = yy[i] + xx[i];
  } else if (*aptr == (PetscScalar)-1.0) {
    PetscCall(PetscLogFlops(n));
    for (PetscInt i = 0; i < n; ++i) ww[i] = yy[i] - xx[i];
  } else if (*aptr == (PetscScalar)0.0) {
    PetscCall(PetscArraycpy(ww,yy,n));
  } else {
    PetscScalar oalpha = *aptr;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,&oalpha,xx,yy,ww);
#else
    for (PetscInt i = 0; i < n; ++i) ww[i] = yy[i] + oalpha * xx[i];
#endif
    PetscCall(PetscLogFlops(2.0*n));
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscCall(VecRestoreArrayRead(yin,&yy));
  PetscCall(VecRestoreArray(win,&ww));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin, Vec yin, PetscManagedReal max, PetscDeviceContext dctx)
{
  const PetscInt     n = xin->map->n;
  const PetscScalar *xx,*yy;
  PetscReal          m = 0.0;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  PetscCall(VecGetArrayRead(yin,&yy));
  for (PetscInt i = 0; i < n; ++i) {
    if (yy[i] == (PetscScalar)0.0) {
      m = PetscMax(PetscAbsScalar(xx[i]),m);
    } else {
      m = PetscMax(PetscAbsScalar(xx[i]/yy[i]),m);
    }
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscCall(VecRestoreArrayRead(yin,&yy));
  // REVIEW ME: why on EARTH is there an allreduce in a seq function?????????????
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&m,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  PetscCall(PetscManagedRealSetValues(dctx,max,PETSC_MEMTYPE_HOST,&m,1));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_Seq(Vec vin, const PetscScalar *a, PetscDeviceContext PETSC_UNUSED dctx)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  PetscCheck(!v->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array         = (PetscScalar*)a;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_Seq(Vec vin, const PetscScalar *a, PetscDeviceContext PETSC_UNUSED dctx)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(v->array_allocated));
  v->array_allocated = v->array = (PetscScalar*)a;
  PetscFunctionReturn(0);
}
