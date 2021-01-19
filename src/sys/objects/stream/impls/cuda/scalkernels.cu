#include "streamcuda.h"

struct addFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return (*arg1)+arg2;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return arg1+(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return (*arg1)+(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return arg1+arg2;}
};

struct subFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return left2right ? (*arg1)-arg2 : arg2-(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return left2right ? arg1-(*arg2) : (*arg2)-arg1;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return left2right ? (*arg1)-(*arg2) : (*arg2)-(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return left2right ? arg1-arg2 : arg2-arg1;}
};

struct divFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return left2right ? (*arg1)/arg2 : arg2/(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return left2right ? arg1/(*arg2) : (*arg2)/arg1;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return left2right ? (*arg1)/(*arg2) : (*arg2)/(*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return left2right ? arg1/arg2 : arg2/arg1;}
};

struct multFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return (*arg1)*arg2;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return arg1*(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return (*arg1)*(*arg2);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return arg1*arg2;}
};

struct equiFunctor {
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type arg2) {return left2right ? arg2 : (*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type *arg2) {return left2right ? (*arg2) : arg1;}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type *arg1, Type *arg2) {return left2right ? (*arg2) : (*arg1);}
  template <typename Type, PetscBool left2right=PETSC_TRUE> __device__ __forceinline__
  Type operator()(Type arg1, Type arg2) {return left2right ? arg2 : arg1;}
};

#if PetscDefined(HAVE_CXX_DIALECT_CXX11) /* C++11 */
/* We use the final iteration of recursion to enforce the type */
template <typename AccumType> __device__ __forceinline__
PetscScalar kernelAccumFunctorDevice(AccumType accFunctor, PetscScalar *top)
{
  return accFunctor(*top, static_cast<PetscScalar>(0));
}

template <typename AccumType, typename Type, typename... Args> __device__ __forceinline__
Type kernelAccumFunctorDevice(AccumType accFunctor, Type *top, Args... argList)
{
  return accFunctor(*top, kernelAccumFunctorDevice(accFunctor, argList...));
}

template <typename EpilogueType, typename AccumType, typename Type, typename... Args> __global__ __launch_bounds__(1)
void kernelAccumFunctor(EpilogueType epiFunctor, AccumType accFunctor, Type *ret, Args... argList)
{
  Type accum = kernelAccumFunctorDevice(accFunctor, argList...);
  *ret = epiFunctor(*ret, accum);
  return;
}
#else
template <typename EpilogueType, typename AccumType, typename Type> __global__ __launch_bounds__(1)
void kernelAccumFunctor(EpilogueType epiFunctor, AccumType accFunctor, Type *ret, Type *in1, Type *in2=nullptr, Type *in3=nullptr, Type *in4=nullptr, Type *in5=nullptr, Type *in6=nullptr, Type *in7=nullptr)
{
  Type acc = accFunctor(acc, *in1);

  if (in2) acc = accFunctor(acc, *in2);
  if (in3) acc = accFunctor(acc, *in3);
  if (in4) acc = accFunctor(acc, *in4);
  if (in5) acc = accFunctor(acc, *in5);
  if (in6) acc = accFunctor(acc, *in6);
  if (in7) acc = accFunctor(acc, *in7);
  *ret = epiFunctor(*ret, acc);
  return;
}
#endif

template <typename EpilogueType, typename AccumType>
PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAccumOp_Internal(EpilogueType epiFunctor, AccumType accFunctor, PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStream pstream)
{
  PetscErrorCode ierr;
  PetscScalar    *dev[8];
  cudaStream_t   cstream;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < n; ++i) {
    ierr = PetscStreamScalarGetDeviceRead(pscal[i], (const PetscScalar**)dev+1+i, pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamScalarGetDeviceWrite(pscalret, dev, pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  /* We must do it this way since we cannot pass just dev, it is allocated on host, but
     contents of dev are allocated on device */
  switch (n) {
  case 1:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1]);
    break;
  case 2:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1],dev[2]);
    break;
  case 3:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1],dev[2],dev[3]);
    break;
  case 4:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1],dev[2],dev[3],dev[4]);
    break;
  case 5:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1],dev[2],dev[3],dev[4],dev[5]);
    break;
  case 6:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6]);
    break;
  case 7:
    kernelAccumFunctor<<<1,1,0,cstream>>>(epiFunctor,accFunctor,dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6],dev[7]);
  default:
    break;
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarRestoreDeviceWrite(pscalret, dev, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Have this odd 2-step process of switch-casing the ops from the enum, because templates
   are fun :) */
template <typename AccumType>
PETSC_STATIC_INLINE PetscErrorCode PetscStreamScalarAccumOpDispatch2_Internal(AccumType accFunctor, PetscStreamComputeOp epiop, PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (epiop) {
  case STREAM_OP_SUM:
    ierr = PetscStreamScalarAccumOp_Internal(addFunctor(),accFunctor,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_SUB:
    ierr = PetscStreamScalarAccumOp_Internal(subFunctor(),accFunctor,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_DIV:
    ierr = PetscStreamScalarAccumOp_Internal(divFunctor(),accFunctor,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_MULT:
    ierr = PetscStreamScalarAccumOp_Internal(multFunctor(),accFunctor,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_EQUAL:
    ierr = PetscStreamScalarAccumOp_Internal(equiFunctor(),accFunctor,pscalret,n,pscal,pstream);CHKERRQ(ierr);
  default:
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscStreamScalarAccumOpDispatch_Internal(PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStreamComputeOp epiop, PetscStreamComputeOp accop, PetscStream pstream)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (accop) {
  case STREAM_OP_SUM:
    ierr = PetscStreamScalarAccumOpDispatch2_Internal(addFunctor(),epiop,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_SUB:
    ierr = PetscStreamScalarAccumOpDispatch2_Internal(subFunctor(),epiop,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_DIV:
    ierr = PetscStreamScalarAccumOpDispatch2_Internal(divFunctor(),epiop,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_MULT:
    ierr = PetscStreamScalarAccumOpDispatch2_Internal(multFunctor(),epiop,pscalret,n,pscal,pstream);CHKERRQ(ierr);
    break;
  case STREAM_OP_EQUAL:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot have equal as accumulator\n");
  default:
    break;
  }
  PetscFunctionReturn(0);
}
