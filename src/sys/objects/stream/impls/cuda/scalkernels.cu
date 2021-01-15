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

#if PetscDefined(HAVE_CXX_DIALECT_CXX11) /* C++11 */
/* We use the final iteration of recursion to enforce the type */
template <typename FuncType> __device__ __forceinline__
PetscScalar kernelFunctorDevice(FuncType functor, PetscScalar *top)
{
  return functor(*top, static_cast<PetscScalar>(0));
}

template <typename FuncType, typename Type, typename... Args> __device__ __forceinline__
Type kernelFunctorDevice(FuncType functor, Type *top, Args... argList)
{
  return functor(*top, kernelFunctorDevice(functor, argList...));
}

template <typename FuncType, typename Type, typename... Args> __global__ __launch_bounds__(1)
void kernelFunctor(FuncType functor, Type *ret, Args... argList)
{
  Type rettemp = kernelFunctorDevice(functor, argList...);
  *ret = functor(ret, rettemp);
  return;
}
#else
template <typename FuncType, typename Type> __global__ __launch_bounds__(1)
void kernelFunctor(FuncType functor, Type *ret, Type *in1, Type *in2=nullptr, Type *in3=nullptr, Type *in4=nullptr, Type *in5=nullptr, Type *in6=nullptr, Type *in7=nullptr)
{
  Type acc = *in1;

  if (in2) acc = functor(acc, *in2);
  if (in3) acc = functor(acc, *in3);
  if (in4) acc = functor(acc, *in4);
  if (in5) acc = functor(acc, *in5);
  if (in6) acc = functor(acc, *in6);
  if (in7) acc = functor(acc, *in7);
  *ret = functor(acc, *ret);
}
#endif
template <typename FuncType, typename Type> __global__ __launch_bounds__(1)
void kernelFunctor(FuncType functor, Type *ret)
{
  Type rettemp = *ret;
  *ret = functor(rettemp, rettemp);
  return;
}

PetscErrorCode PetscStreamScalarOperator_CUDA(PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStream pstream)
{
  PetscErrorCode ierr;
  PetscScalar    *dev[8];
  cudaStream_t   cstream;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalret,1,pstream,4);
  /* n may only be 1 <= n <= 7, for a total of 8 sums */
  if (PetscUnlikelyDebug(n > 7)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only sum up to 8 scalars at a time\n");
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckValidSameStreamType(pscal[i],3,pstream,4);
    ierr = PetscStreamScalarGetDeviceRead(pscal[i], (const PetscScalar**)dev+1+i, pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamScalarGetDeviceWrite(pscalret, dev, pstream);CHKERRQ(ierr);
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  /* We must do it this way since we cannot pass just dev, it is allocated on host, but
     contents of dev are allocated on device */
  switch (n) {
  case 0:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0]);
    break;
  case 1:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1]);
    break;
  case 2:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1],dev[2]);
    break;
  case 3:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1],dev[2],dev[3]);
    break;
  case 4:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1],dev[2],dev[3],dev[4]);
    break;
  case 5:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1],dev[2],dev[3],dev[4],dev[5]);
    break;
  case 6:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6]);
    break;
  case 7:
    kernelFunctor<<<1,1,0,cstream>>>(addFunctor(),dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6],dev[7]);
  default:
    break;
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarRestoreDeviceWrite(pscalret, dev, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
