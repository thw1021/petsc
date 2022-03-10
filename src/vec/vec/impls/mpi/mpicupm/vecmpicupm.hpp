#ifndef PETSCVECMPICUPM_HPP
#define PETSCVECMPICUPM_HPP

#include <petsc/private/veccupmimpl.h>  /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/seq/seqcupm/vecseqcupm.hpp>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petsc/private/sfimpl.h> // for vec->localupdate (_p_VecScatter) in duplicate()

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
  PETSC_CXX_COMPAT_DECL(constexpr auto VecIMPLCast_(Vec v))
  PETSC_DECLTYPE_AUTO_RETURNS(static_cast<Vec_MPI*>(v->data));
  PETSC_CXX_COMPAT_DECL(PETSC_CONSTEXPR_14 auto VECTYPE_()) PETSC_DECLTYPE_AUTO_RETURNS(VECMPICUPM());

  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupm_async_(Vec,PetscDeviceContext,PetscBool/*allocate_missing*/=PETSC_TRUE,PetscInt/*nghost*/=0,PetscScalar*/*host_array*/=nullptr,PetscScalar*/*device_array*/=nullptr));
  template <typename SeqFunction>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode minmax_async_(Vec,PetscManagedInt,PetscManagedReal,SeqFunction,MPI_Op,MPI_Op,PetscDeviceContext));

public:
  // callable directly via a bespoke function
  PETSC_CXX_COMPAT_DECL(PetscErrorCode create_async(Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupm_async(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscDeviceContext,Vec*,PetscBool));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode creatempicupmwitharrays_async(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],PetscDeviceContext,Vec*));

  PETSC_CXX_COMPAT_DECL(PetscErrorCode destroy_async(Vec,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode duplicate_async(Vec,Vec*,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode bindtocpu_async(Vec,PetscBool,PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec,PetscDeviceContext));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode norm_async(Vec,NormType,PetscManagedReal,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dot_async(Vec,Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode tdot_async(Vec,Vec,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode mdot_async(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode dotnorm2_async(Vec,Vec,PetscManagedScalar,PetscManagedScalar,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode max_async(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext));
  PETSC_CXX_COMPAT_DECL(PetscErrorCode min_async(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext));
};

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async_(Vec v, PetscDeviceContext dctx, PetscBool allocate_missing, PetscInt nghost, PetscScalar *host_array, PetscScalar *device_array))
{
  PetscFunctionBegin;
  // REVIEW ME: remove me
  PetscCheck(!VecIMPLCast(v),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Creating VecMPI for the second time!");
  PetscCall(VecCreate_MPI_Private(v,PETSC_FALSE,nghost,nullptr,dctx));
  PetscCall(Initialize_CUPMBase(v,allocate_missing,host_array,device_array,dctx));
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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::create_async(Vec v, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(creatempicupm_async_(v,dctx));
  PetscFunctionReturn(0);
}

// VecCreateMPICUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, PetscDeviceContext dctx, Vec *v, PetscBool call_set_type))
{
  PetscFunctionBegin;
  PetscCall(Create_CUPMBase(comm,bs,n,N,dctx,v,call_set_type));
  PetscFunctionReturn(0);
}

// VecCreateMPICUPMWithArray[s]()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupmwitharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar host_array[], const PetscScalar device_array[], PetscDeviceContext dctx, Vec *v))
{
  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // creatempicupm_async_() is called!
  PetscCall(creatempicupm_async(comm,bs,n,N,dctx,v,PETSC_FALSE));
  PetscCall(creatempicupm_async_(*v,dctx,PETSC_FALSE,0,PetscRemoveConstCast(host_array),PetscRemoveConstCast(device_array)));
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::duplicate_async(Vec v, Vec *y, PetscDeviceContext dctx))
{
  const auto vimpl  = VecIMPLCast(v);
  const auto nghost = vimpl->nghost;

  PetscFunctionBegin;
  // does not call VecSetType(), we set up the data structures ourselves
  PetscCall(Duplicate_CUPMBase(v,y,dctx,[=](Vec z, PetscDeviceContext dctx){return creatempicupm_async_(z,dctx,PETSC_FALSE,nghost);}));

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (const auto& locrep = vimpl->localrep) {
    const auto   ops   = locrep->ops;
    const auto   yimpl = VecIMPLCast(*y);
    PetscScalar *array;

    PetscCall(VecGetArrayAsync(*y,&array,dctx));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,v->map->n+nghost,array,&yimpl->localrep));
    PetscCall(PetscMemcpy(yimpl->localrep->ops,ops,sizeof(*ops)));
    PetscCall(VecRestoreArrayAsync(*y,&array,dctx));
    PetscCall(PetscLogObjectParent(PetscObjectCast(*y),PetscObjectCast(yimpl->localrep)));
    yimpl->localupdate = vimpl->localupdate;
    if (auto &scatter = yimpl->localupdate) PetscCall(PetscObjectReference(PetscObjectCast(scatter)));
  }
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::destroy_async(Vec v, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(Destroy_CUPMBase(v,dctx,VecDestroy_MPI));
  PetscFunctionReturn(0);
}

// v->ops->bintocpu
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(BindToCPU_CUPMBase(v,usehost,dctx));

  VecSetOp_CUPM(dot,VecDot_MPI,dot_async);
  VecSetOp_CUPM(mdot,VecMDot_MPI,mdot_async);
  VecSetOp_CUPM(norm,VecNorm_MPI,norm_async);
  VecSetOp_CUPM(tdot,VecTDot_MPI,tdot_async);
  VecSetOp_CUPM(resetarray,VecResetArray_MPI,resetarray_async<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(placearray,VecPlaceArray_MPI,placearray_async<PETSC_MEMTYPE_HOST>);
  VecSetOp_CUPM(max,VecMax_MPI,max_async);
  VecSetOp_CUPM(min,VecMin_MPI,min_async);
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                    mutatators                                      //

// v->ops->resetarray or VecCUPMResetArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::resetarray_async(Vec v, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(base_type::template ResetArray_CUPMBase<mtype>(v,VecResetArray_MPI,dctx));
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::placearray_async(Vec v, const PetscScalar *a, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(base_type::template PlaceArray_CUPMBase<mtype>(v,a,VecPlaceArray_MPI,dctx));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::norm_async(Vec v, NormType type, PetscManagedReal z, PetscDeviceContext dctx))
{
  const auto  bothNorm = type == NORM_1_AND_2;
  const auto  norm2    = bothNorm || (type == NORM_2) || (type == NORM_FROBENIUS);
  const auto  count    = bothNorm ? 2 : 1;
  const auto  op       = type == NORM_INFINITY ? MPIU_MAX : MPIU_SUM;
  PetscReal  *zptr;

  PetscFunctionBegin;
  PetscCall(VecSeq_T::norm_async(v,type,z,dctx));
  // implicit sync
  PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&zptr));
  if (norm2) zptr[bothNorm] *= zptr[bothNorm];
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,zptr,count,MPIU_REAL,op,PetscObjectComm(PetscObjectCast(v))));
  if (norm2) zptr[bothNorm] = PetscSqrtReal(zptr[bothNorm]);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dot_async(Vec x, Vec y, PetscManagedScalar z, PetscDeviceContext dctx))
{
  constexpr auto one = PetscInt{1};

  PetscFunctionBegin;
  PetscCall(VecSeq_T::dot_async(x,y,z,dctx));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx,z,&one,MPIU_SUM,PetscObjectCast(x)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::tdot_async(Vec x, Vec y, PetscManagedScalar z, PetscDeviceContext dctx))
{
  constexpr auto one = PetscInt{1};

  PetscFunctionBegin;
  PetscCall(VecSeq_T::tdot_async(x,y,z,dctx));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx,z,&one,MPIU_SUM,PetscObjectCast(x)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::mdot_async(Vec x, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx))
{
  PetscInt *nvptr;

  PetscFunctionBegin;
  PetscCall(VecSeq_T::mdot_async(x,nv,y,z,dctx));
  // implicit sync
  PetscCall(PetscManagedIntGetValues(dctx,nv,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ,PETSC_TRUE,&nvptr));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx,z,nvptr,MPIU_SUM,PetscObjectCast(x)));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dotnorm2_async(Vec x, Vec y, PetscManagedScalar dp, PetscManagedScalar nm, PetscDeviceContext dctx))
{
  PetscScalar *dpptr,*nmptr;

  PetscFunctionBegin;
  PetscCall(VecSeq_T::dotnorm2_async(x,y,dp,nm,dctx));
  // implicit sync
  PetscCall(PetscManagedScalarGetValues(dctx,dp,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&dpptr));
  PetscCall(PetscManagedScalarGetValues(dctx,nm,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&nmptr));
  {
    PetscScalar sum[2] = {*dpptr,*nmptr};
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x))));
    *dpptr = sum[0];
    *nmptr = sum[1];
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename SeqFunction>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::minmax_async_(Vec x, PetscManagedInt idx, PetscManagedReal z, SeqFunction seqfn, MPI_Op idxOp, MPI_Op noIdxOp, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(seqfn(x,idx,z,dctx));
  if (!PetscDefined(HAVE_MPIUNI)) {
    const auto obj = PetscObjectCast(x);

    if (idx) {
      PetscReal *zptr;
      PetscInt  *idxptr;

      // implicit sync
      PetscCall(PetscManagedRealGetValues(dctx,z,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&zptr));
      PetscCall(PetscManagedIntGetValues(dctx,idx,PETSC_MEMTYPE_HOST,PETSC_MEMORY_ACCESS_READ_WRITE,PETSC_TRUE,&idxptr));
      {
        struct { PetscReal v; PetscInt i; } tmp = {*zptr,*idxptr+x->map->rstart};
        PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&tmp,1,MPIU_REAL_INT,idxOp,PetscObjectComm(obj)));
        *zptr   = tmp.v;
        *idxptr = tmp.i;
      }
    } else {
      constexpr auto one = PetscInt{1};

      PetscCall(PetscDeviceContextAllReduceManagedReal_Internal(dctx,z,&one,noIdxOp,obj));
    }
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::max_async(Vec x, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(minmax_async_(x,idx,z,VecSeq_T::max_async,MPIU_MAXLOC,MPIU_MAX,dctx));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::min_async(Vec x, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx))
{
  PetscFunctionBegin;
  PetscCall(minmax_async_(x,idx,z,VecSeq_T::min_async,MPIU_MINLOC,MPIU_MIN,dctx));
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
