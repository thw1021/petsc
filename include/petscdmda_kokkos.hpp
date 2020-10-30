#if !defined(__PETSCDMDA_KOKKOS_HPP)
#define __PETSCDMDA_KOKKOS_HPP

#include <petscdmda.h>
#include <petscvec_kokkos.hpp>

#if defined(PETSC_HAVE_KOKKOS)
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,     PetscScalarKokkosView1DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,     PetscScalarKokkosView1DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,ConstPetscScalarKokkosView1DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,ConstPetscScalarKokkosView1DType<MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,     PetscScalarKokkosView2DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,     PetscScalarKokkosView2DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,ConstPetscScalarKokkosView2DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,ConstPetscScalarKokkosView2DType<MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,     PetscScalarKokkosView3DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,     PetscScalarKokkosView3DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,ConstPetscScalarKokkosView3DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,ConstPetscScalarKokkosView3DType<MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,     PetscScalarKokkosView4DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,     PetscScalarKokkosView4DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosView    (DM,Vec,ConstPetscScalarKokkosView4DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosView(DM,Vec,ConstPetscScalarKokkosView4DType<MemorySpace>*);


template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,     PetscScalarKokkosOffsetView1DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,     PetscScalarKokkosOffsetView1DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,     PetscScalarKokkosOffsetView2DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,     PetscScalarKokkosOffsetView2DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,     PetscScalarKokkosOffsetView3DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,     PetscScalarKokkosOffsetView3DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,ConstPetscScalarKokkosOffsetView3DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView3DType<MemorySpace>*);

template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,     PetscScalarKokkosOffsetView4DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,     PetscScalarKokkosOffsetView4DType<MemorySpace>*,PetscBool=PETSC_FALSE);
template<class MemorySpace> PetscErrorCode DMDAVecGetKokkosOffsetView    (DM,Vec,ConstPetscScalarKokkosOffsetView4DType<MemorySpace>*);
template<class MemorySpace> PetscErrorCode DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView4DType<MemorySpace>*);
#endif

#endif
