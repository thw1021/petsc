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
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode resetarray_async(Vec));
  template <PetscMemType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode placearray_async(Vec,const PetscScalar*));
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
  PetscFunctionBegin;
  // REVIEW ME: remove me
  PetscCheck(!VecIMPLCast(v),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Creating VecMPI for the second time!");
  PetscCall(VecCreate_MPI_Private(v,PETSC_FALSE,nghost,nullptr));
  PetscCall(Initialize_CUPMBase(v,allocate_missing,host_array,device_array));
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
  PetscFunctionBegin;
  PetscCall(creatempicupm_async_(v));
  PetscFunctionReturn(0);
}

// VecCreateMPICUPM()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupm_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, Vec *v, PetscBool call_set_type))
{
  PetscFunctionBegin;
  PetscCall(Create_CUPMBase(comm,bs,n,N,v,call_set_type));
  PetscFunctionReturn(0);
}

// VecCreateMPICUPMWithArray[s]()
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::creatempicupmwitharrays_async(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, const PetscScalar host_array[], const PetscScalar device_array[], Vec *v))
{
  PetscFunctionBegin;
  // do NOT call VecSetType(), otherwise ops->create() -> create_async() ->
  // creatempicupm_async_() is called!
  PetscCall(creatempicupm_async(comm,bs,n,N,v,PETSC_FALSE));
  PetscCall(creatempicupm_async_(*v,PETSC_FALSE,0,PetscRemoveConstCast(host_array),PetscRemoveConstCast(device_array)));
  PetscFunctionReturn(0);
}

// v->ops->duplicate
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::duplicate_async(Vec v, Vec *y))
{
  const auto vimpl  = VecIMPLCast(v);
  const auto nghost = vimpl->nghost;

  PetscFunctionBegin;
  // does not call VecSetType(), we set up the data structures ourselves
  PetscCall(Duplicate_CUPMBase(v,y,[=](Vec z){return creatempicupm_async_(z,PETSC_FALSE,nghost);}));

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (const auto locrep = vimpl->localrep) {
    const auto   ops   = locrep->ops;
    const auto   yimpl = VecIMPLCast(*y);
    PetscScalar *array;

    PetscCall(VecGetArray(*y,&array));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,v->map->n+nghost,array,&yimpl->localrep));
    PetscCall(PetscMemcpy(yimpl->localrep->ops,ops,sizeof(*ops)));
    PetscCall(VecRestoreArray(*y,&array));
    PetscCall(PetscLogObjectParent(PetscObjectCast(*y),PetscObjectCast(yimpl->localrep)));
    yimpl->localupdate = vimpl->localupdate;
    if (yimpl->localupdate) PetscCall(PetscObjectReference(PetscObjectCast(yimpl->localupdate)));
  }
  PetscFunctionReturn(0);
}

// v->ops->destroy
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::destroy_async(Vec v))
{
  PetscFunctionBegin;
  PetscCall(Destroy_CUPMBase(v));
  PetscCall(VecDestroy_MPI(v));
  PetscFunctionReturn(0);
}

// v->ops->bintocpu
template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::bindtocpu_async(Vec v, PetscBool usehost))
{
  PetscFunctionBegin;
  PetscCall(BindToCPU_CUPMBase(v,usehost));

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
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::resetarray_async(Vec v))
{
  PetscFunctionBegin;
  PetscCall(base_type::template ResetArray_CUPMBase<mtype>(v,VecResetArray_MPI));
  PetscFunctionReturn(0);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <Device::CUPM::DeviceType T>
template <PetscMemType mtype>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::placearray_async(Vec v, const PetscScalar *a))
{
  PetscFunctionBegin;
  PetscCall(base_type::template PlaceArray_CUPMBase<mtype>(v,a,VecPlaceArray_MPI));
  PetscFunctionReturn(0);
}

// ================================================================================== //
//                                   compute methods                                  //

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::norm_async(Vec v, NormType type, PetscReal *z))
{
  PetscReal  work[2]  = {0};
  const auto bothNorm = type == NORM_1_AND_2;
  const auto norm2    = bothNorm || type == NORM_2 || type == NORM_FROBENIUS;
  const auto count    = bothNorm ? 2 : 1;
  const auto op       = type == NORM_INFINITY ? MPIU_MAX : MPIU_SUM;

  PetscFunctionBegin;
  PetscCall(VecSeq_T::norm_async(v,type,work));
  if (norm2) work[bothNorm] *= work[bothNorm];
  PetscCallMPI(MPIU_Allreduce(work,z,count,MPIU_REAL,op,PetscObjectComm(PetscObjectCast(v))));
  if (norm2) z[bothNorm] = PetscSqrtReal(z[bothNorm]);
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dot_async(Vec x, Vec y, PetscScalar *z))
{
  PetscScalar work;

  PetscFunctionBegin;
  PetscCall(VecSeq_T::dot_async(x,y,&work));
  PetscCallMPI(MPIU_Allreduce(&work,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x))));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::tdot_async(Vec x, Vec y, PetscScalar *z))
{
  PetscScalar work;

  PetscFunctionBegin;
  PetscCall(VecSeq_T::tdot_async(x,y,&work));
  PetscCallMPI(MPIU_Allreduce(&work,z,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x))));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::mdot_async(Vec x, PetscInt nv, const Vec y[], PetscScalar *z))
{
  auto       stackwork = std::array<PetscScalar,128>{};
  // needed to silence warning: comparison between signed and unsigned integer expressions
  const auto allocate  = stackwork.size() < static_cast<decltype(stackwork.size())>(nv);
  auto       work      = stackwork.data();

  PetscFunctionBegin;
  if (allocate) PetscCall(PetscMalloc1(nv,&work));
  PetscCall(VecSeq_T::mdot_async(x,nv,y,work));
  {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscCallMPI(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x))));
  if (allocate) PetscCall(PetscFree(work));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::dotnorm2_async(Vec x, Vec y, PetscScalar *dp, PetscScalar *nm))
{
  PetscScalar work[2],sum[2];

  PetscFunctionBegin;
  PetscCall(VecSeq_T::dotnorm2_async(x,y,work,work+1));
  {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
  }
  PetscCallMPI(MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm(PetscObjectCast(x))));
  *dp = sum[0];
  *nm = sum[1];
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
template <typename SeqFunction>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::minmax_async_(Vec x, PetscInt *idx, PetscReal *z, SeqFunction seqfn, MPI_Op idxOp, MPI_Op noIdxOp))
{
  PetscReal          work;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(seqfn(x,idx,&work));
  PetscCall(GetHandles_(&dctx));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  if (PetscDefined(HAVE_MPIUNI)) {
    *z = work;
  } else {
    const auto comm = PetscObjectComm(PetscObjectCast(x));

    if (idx) {
      struct { PetscReal v; PetscInt i; } in,out;

      in.v  = work;
      in.i  = *idx + x->map->rstart;
      PetscCallMPI(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,idxOp,comm));
      *z    = out.v;
      *idx  = out.i;
    } else PetscCallMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,noIdxOp,comm));
  }
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::max_async(Vec x, PetscInt *idx, PetscReal *z))
{
  PetscFunctionBegin;
  PetscCall(minmax_async_(x,idx,z,VecSeq_T::max_async,MPIU_MAXLOC,MPIU_MAX));
  PetscFunctionReturn(0);
}

template <Device::CUPM::DeviceType T>
PETSC_CXX_COMPAT_DEFN(PetscErrorCode VecMPI_CUPM<T>::min_async(Vec x, PetscInt *idx, PetscReal *z))
{
  PetscFunctionBegin;
  PetscCall(minmax_async_(x,idx,z,VecSeq_T::min_async,MPIU_MINLOC,MPIU_MIN));
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
