#if !defined(__VECKOKKOSIMPL_HPP)
#define __VECKOKKOSIMPL_HPP

#include <petsc/private/vecimpl_kokkos.hpp>

/* Stuff related to Vec_Kokkos */

struct Vec_Kokkos {
  PetscScalarKokkosDualView      v_dual;

  /* Construct Vec_Kokkos with the given array(s). n is the length of the array.
    If n != 0, host array (array_h) must not be NULL.
    If device array (array_d) is NULL, then a proper device mirror will be allocated.
    Otherwise, the mirror will be created using the given array_d.
  */
  Vec_Kokkos(PetscInt n,PetscScalar *array_h,PetscScalar *array_d = NULL) {
    PetscScalarKokkosViewHost    v_h(array_h,n);
    PetscScalarKokkosView        v_d;

    if (array_d) {
      v_d = PetscScalarKokkosView(array_d,n); /* Use the given device array */
    } else {
      v_d = Kokkos::create_mirror_view(DefaultMemorySpace(),v_h); /* Create a mirror in DefaultMemorySpace but do not copy values */
    }
    v_dual = PetscScalarKokkosDualView(v_d,v_h);
    if (!array_d) v_dual.modify_host();
  }

  /* SFINAE: Update the object with an array in the given memory space,
     assuming the given array contains the latest value for this vector.
   */
  template<typename MemorySpace,
           std::enable_if_t<std::is_same<MemorySpace,Kokkos::HostSpace>::value, bool> = true,
           std::enable_if_t<std::is_same<MemorySpace,DefaultMemorySpace>::value,bool> = true>
  void UpdateArray(PetscScalar *array) {
    PetscScalarKokkosViewHost v_h(array,v_dual.extent(0));
    /* Kokkos said they would add error-checking so that users won't accidentally pass two different Views in this case */
    v_dual = PetscScalarKokkosDualView(v_h,v_h);
  }

  template<typename MemorySpace,
           std::enable_if_t<std::is_same<MemorySpace,Kokkos::HostSpace>::value,  bool> = true,
           std::enable_if_t<!std::is_same<MemorySpace,DefaultMemorySpace>::value,bool> = true>
  void UpdateArray(PetscScalar *array) {
    PetscScalarKokkosViewHost v_h(array,v_dual.extent(0));
    v_dual = PetscScalarKokkosDualView(v_dual.view<DefaultMemorySpace>(),v_h);
    v_dual.modify_host();
  }

  template<typename MemorySpace,
           std::enable_if_t<!std::is_same<MemorySpace,Kokkos::HostSpace>::value, bool> = true,
           std::enable_if_t<std::is_same<MemorySpace,DefaultMemorySpace>::value, bool> = true>
  void UpdateArray(PetscScalar *array) {
    PetscScalarKokkosView v_d(array,v_dual.extent(0));
    v_dual = PetscScalarKokkosDualView(v_d,v_dual.view<Kokkos::HostSpace>());
    v_dual.modify_device();
  }
};

PETSC_INTERN PetscErrorCode VecAbs_SeqKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqKokkos(Vec,Vec,PetscManagedScalar,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqKokkos(Vec,PetscManagedScalar,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMDot_SeqKokkos(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMTDot_SeqKokkos(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecSet_SeqKokkos(Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqKokkos(Vec,PetscManagedInt,PetscManagedScalar,Vec*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec,PetscManagedScalar,PetscManagedScalar,PetscManagedScalar,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqKokkos(Vec,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqKokkos(Vec,const PetscScalar*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqKokkos(Vec,const PetscScalar*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDot_SeqKokkos(Vec,Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecTDot_SeqKokkos(Vec,Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecScale_SeqKokkos(Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCopy_SeqKokkos(Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSwap_SeqKokkos(Vec,Vec,PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecAXPY_SeqKokkos(Vec,PetscManagedScalar,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqKokkos(Vec,PetscManagedScalar,PetscManagedScalar,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqKokkos(Vec,Vec*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecConjugate_SeqKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecNorm_SeqKokkos(Vec,NormType,PetscManagedReal,PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_SeqKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos_Private(Vec,PetscBool,PetscInt,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecCreate_Kokkos(Vec);
PETSC_INTERN PetscErrorCode VecDestroy_SeqKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDestroy_MPIKokkos(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecAYPX_SeqKokkos(Vec,PetscManagedScalar,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos(Vec,PetscRandom,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqKokkos(Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec,PetscScalar**,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMin_SeqKokkos(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMax_SeqKokkos(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSum_SeqKokkos(Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecShift_SeqKokkos(Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecGetArray_SeqKokkos(Vec,PetscScalar**,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqKokkos(Vec,PetscScalar**,PetscDeviceContext);

PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqKokkos(Vec,PetscScalar**,PetscMemType*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqKokkos(Vec,PetscScalar**,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecGetArrayWriteAndMemType_SeqKokkos(Vec,PetscScalar**,PetscMemType*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecGetSubVector_Kokkos_Private(Vec,PetscBool,IS,Vec*);
PETSC_INTERN PetscErrorCode VecRestoreSubVector_SeqKokkos(Vec,IS,Vec*);
#endif
