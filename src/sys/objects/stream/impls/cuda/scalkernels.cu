#include "streamcuda.h"

#if PetscDefined(HAVE_CXX_DIALECT_CXX11) /* C++11 */
/* We use the final iteration of recursion to enforce the type */
__device__ __forceinline__
PetscScalar addKernelDevice(PetscScalar *top)
{
  return (*top);
}

template <typename Type, typename... Args> __device__ __forceinline__
Type addKernelDevice(Type *top, Args... argList)
{
  static_assert(std::is_arithmetic<Type>::value, "Add kernel only supports arithmetic types");
  return (*top)+addKernelDevice(argList...);
}

template <typename Type, typename... Args> __global__ __launch_bounds__(1)
void addKernel(Type *ret, Args... argList)
{
  static_assert(std::is_arithmetic<Type>::value, "Add kernel only supports arithmetic types");
  Type rettemp = addKernelDevice(argList...);
  *ret += rettemp;
  return;
}
#else
template <typename Type>
__global__ __launch_bounds__(1)
void addKernel(Type *ret, Type *in1, Type *in2=nullptr, Type *in3=nullptr, Type *in4=nullptr, Type *in5=nullptr, Type *in6=nullptr, Type *in7=nullptr)
{
  Type acc = *in1;

  if (in2) acc += (*in2);
  if (in3) acc += (*in3);
  if (in4) acc += (*in4);
  if (in5) acc += (*in5);
  if (in6) acc += (*in6);
  if (in7) acc += (*in7);
  *ret += acc;
}
#endif

__global__ __launch_bounds__(1)
void addKernel(PetscScalar *ret)
{
  PetscScalar rettemp = *ret;
  *ret = rettemp+rettemp;
  return;
}

PetscErrorCode PetscStreamScalarAccumulate_CUDA(PetscStreamScalar pscalret, PetscInt n, PetscStreamScalar pscal[], PetscStream pstream)
{
  PetscErrorCode ierr;
  PetscScalar    *dev[8];
  cudaStream_t   cstream;

  PetscFunctionBegin;
  PetscCheckValidSameStreamType(pscalret,1,pstream,4);
  /* n may only be 1 <= n <= 7, for a total of 8 sums */
  if (PetscUnlikelyDebug(n > 7)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Can only sum up to 8 scalars at a time\n");
  ierr = PetscStreamScalarGetDeviceWrite(pscalret, dev, pstream);CHKERRQ(ierr);
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckValidSameStreamType(pscal[i],3,pstream,4);
    ierr = PetscStreamScalarGetDeviceRead(pscal[i], (const PetscScalar**)dev+1+i, pstream);CHKERRQ(ierr);
  }
  ierr = PetscStreamGetStream(pstream, &cstream);CHKERRQ(ierr);
  /* We must do it this way since we cannot pass just dev, it is allocated on host, but
     contents of dev are allocated on device */
  switch (n) {
  case 0:
    addKernel<<<1,1,0,cstream>>>(dev[0]);
    break;
  case 1:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1]);
    break;
  case 2:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1],dev[2]);
    break;
  case 3:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1],dev[2],dev[3]);
    break;
  case 4:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1],dev[2],dev[3],dev[4]);
    break;
  case 5:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1],dev[2],dev[3],dev[4],dev[5]);
    break;
  case 6:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6]);
    break;
  case 7:
    addKernel<<<1,1,0,cstream>>>(dev[0],dev[1],dev[2],dev[3],dev[4],dev[5],dev[6],dev[7]);
  default:
    break;
  }
  ierr = PetscStreamRestoreStream(pstream, &cstream);CHKERRQ(ierr);
  ierr = PetscStreamScalarRestoreDeviceWrite(pscalret, dev, pstream);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
