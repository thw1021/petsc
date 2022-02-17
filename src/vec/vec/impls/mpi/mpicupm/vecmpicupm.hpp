#ifndef PETSCVECMPICUPM_HPP
#define PETSCVECMPICUPM_HPP

#include <petsc/private/veccupmimpl.h>  /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/seq/seqcupm/vecseqcupm.hpp>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petsc/private/sfimpl.h> // for _p_VecScatter

namespace Petsc
{

namespace Vector
{

namespace CUPM
{

namespace Impl
{

template <Device::CUPM::DeviceType T>
struct VecMPI_CUPM : Vec_CUPMBase<T,VecMPI_CUPM<T>>
{
  PETSC_VEC_CUPM_BASE_CLASS_HEADER(base_type,T,VecMPI_CUPM<T>);
  using VecSeq_T = VecSeq_CUPM<T>;

private:
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast_(Vec v)) PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_MPI*>(v->data));
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 auto VECTYPE_()) PETSC_DECLTYPE_AUTO_RETURNS(VECMPICUPM());

  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupm_async_(Vec,PetscBool/*allocate_missing*/=PETSC_TRUE,PetscInt/*nghost*/=0,PetscScalar*/*host_array*/=nullptr,PetscScalar*/*device_array*/=nullptr));
  template <typename SeqFunction>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(Vec,PetscInt*,PetscReal*,SeqFunction,MPI_Op,MPI_Op));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupm_async(MPI_Comm,PetscInt,PetscInt,PetscInt,Vec*,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupmwitharrays_async(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Vec*));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec,PetscBool));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec,NormType,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec,Vec,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec,PetscInt,const Vec[],PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec,Vec,PetscScalar*,PetscScalar*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscInt*,PetscReal*));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscInt*,PetscReal*));
};

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async_(Vec v, PetscBool allocate_missing, PetscInt nghost, PetscScalar *host_array, PetscScalar *device_array))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // REVIEW ME: remove me
  PetscCheck(!VecIMPLCast(v),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Creating VecMPI for the second time!");
  ierr = VecCreate_MPI_Private(v,PETSC_FALSE,nghost,nullptr);CHKERRQ(ierr);
  ierr = Initialize_CUPMBase_(v,allocate_missing,host_array,device_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                                                                    //
//                                  public methods                                    //
//                                                                                    //
// ================================================================================== //

// ================================================================================== //
//                             constructors/destructors                               //

// v->ops->create
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::create_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = creatempicupm_async_(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VecCreateMPICUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, Vec *v, PetscBool call_set_type))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Create_CUPMBase_(comm,bs,n,N,v,call_set_type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VecCreateMPICUPMWithArray[s]()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupmwitharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // creatempicupm_async_() is called!
  ierr = creatempicupm_async(comm,bs,n,N,v,PETSC_FALSE);CHKERRQ(ierr);
  ierr = creatempicupm_async_(*v,PETSC_FALSE,0,PetscRemoveConstCast(host_array),PetscRemoveConstCast(device_array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::duplicate_async(Vec v, Vec *y))
{
  const auto     vimpl   = VecIMPLCast(v);
  const auto     nghost  = vimpl->nghost;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // does not call VecSetType(), we set up the data structures ourselves
  ierr = Duplicate_CUPMBase_(v,y,[=](Vec z){return creatempicupm_async_(z,PETSC_FALSE,nghost);});CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (const auto locrep = vimpl->localrep) {
    const auto  ops   = locrep->ops;
    const auto  yimpl = VecIMPLCast(*y);
    PetscScalar *array;

    ierr = VecGetArray(*y,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,v->map->n+nghost,array,&yimpl->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(yimpl->localrep->ops,ops,sizeof(*ops));CHKERRQ(ierr);
    ierr = VecRestoreArray(*y,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(PetscObjectCast(*y),PetscObjectCast(yimpl->localrep));CHKERRQ(ierr);
    yimpl->localupdate = vimpl->localupdate;
    if (yimpl->localupdate) {
      ierr = PetscObjectReference(PetscObjectCast(yimpl->localupdate));CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::destroy_async(Vec v))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Destroy_CUPMBase_(v);CHKERRQ(ierr);
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BindToCPU_CUPMBase_(v,usehost);CHKERRQ(ierr);

  VecSetOp_CUPM(dot,VecDot_MPI,dot_async);
  VecSetOp_CUPM(mdot,VecMDot_MPI,mdot_async);
  VecSetOp_CUPM(norm,VecNorm_MPI,norm_async);
  VecSetOp_CUPM(tdot,VecTDot_MPI,tdot_async);
  VecSetOp_CUPM(max,VecMax_MPI,max_async);
  VecSetOp_CUPM(min,VecMin_MPI,min_async);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::norm_async(Vec v, NormType type, PetscReal *z))
{
  PetscReal      work[2]  = {0};
  const auto     bothNorm = type == NORM_1_AND_2;
  const auto     norm2    = bothNorm || type == NORM_2 || type == NORM_FROBENIUS;
  const auto     count    = bothNorm ? 2 : 1;
  const auto     op       = type == NORM_INFINITY ? MPIU_MAX : MPIU_SUM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_T::norm_async(v,type,work);CHKERRQ(ierr);
  if (norm2) work[bothNorm] *= work[bothNorm];
  ierr = MPIU_Allreduce(work,z,count,MPIU_REAL,op,PetscObjectComm(PetscObjectCast(v)));CHKERRMPI(ierr);
  if (norm2) z[bothNorm] = PetscSqrtReal(z[bothNorm]);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dot_async(Vec x, Vec y, PetscScalar *z))
{
  PetscScalar    work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_T::dot_async(x,y,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::tdot_async(Vec x, Vec y, PetscScalar *z))
{
  PetscScalar    work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_T::tdot_async(x,y,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::mdot_async(Vec x, PetscInt nv, const Vec y[], PetscScalar *z))
{
  auto           stackwork = std::array<PetscScalar,128>{};
  // needed to silence warning: comparison between signed and unsigned integer expressions
  const auto     allocate  = stackwork.size() < static_cast<decltype(stackwork.size())>(nv);
  auto           work      = stackwork.data();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (allocate) {ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);}
  ierr = VecSeq_T::mdot_async(x,nv,y,work);CHKERRQ(ierr);
  {
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
  }
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  if (allocate) {ierr = PetscFree(work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dotnorm2_async(Vec x, Vec y, PetscScalar *dp, PetscScalar *nm))
{
  PetscScalar    work[2],sum[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSeq_T::dotnorm2_async(x,y,work,work+1);CHKERRQ(ierr);
  {
    PetscDeviceContext dctx;

    ierr = GetHandles_(&dctx);CHKERRQ(ierr);
    ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
  }
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x)));CHKERRMPI(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename SeqFunction>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::minmax_async_(Vec x, PetscInt *idx, PetscReal *z, SeqFunction seqfn, MPI_Op idxOp, MPI_Op noIdxOp))
{
  PetscReal          work;
  PetscDeviceContext dctx;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = seqfn(x,idx,&work);CHKERRQ(ierr);
  ierr = GetHandles_(&dctx);CHKERRQ(ierr);
  ierr = PetscDeviceContextSynchronize(dctx);CHKERRQ(ierr);
  if (PetscDefined(HAVE_MPIUNI)) {
    *z = work;
  } else {
    const auto comm = PetscObjectComm(PetscObjectCast(x));

    if (idx) {
      struct { PetscReal v; PetscInt i; } in,out;

      in.v  = work;
      in.i  = *idx + x->map->rstart;
      ierr  = MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,idxOp,comm);CHKERRMPI(ierr);
      *z    = out.v;
      *idx  = out.i;
    } else {ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,noIdxOp,comm);CHKERRMPI(ierr);}
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::max_async(Vec x, PetscInt *idx, PetscReal *z))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(x,idx,z,VecSeq_T::max_async,MPIU_MAXLOC,MPIU_MAX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::min_async(Vec x, PetscInt *idx, PetscReal *z))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = minmax_async_(x,idx,z,VecSeq_T::min_async,MPIU_MINLOC,MPIU_MIN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// declare the extern templates, each is explicitly instantiated in the respective
// implementation directories
#if PetscDefined(HAVE_CUDA)
extern template struct VecMPI_CUPM<Device::CUPM::DeviceType::CUDA>;
#endif

#if PetscDefined(HAVE_HIP)
extern template struct VecMPI_CUPM<Device::CUPM::DeviceType::HIP>;
#endif

} // namespace Impl

} // namespace CUPM

} // namespace Vector

} // namespace Petsc

#endif // PETSCVECMPICUPM_HPP
