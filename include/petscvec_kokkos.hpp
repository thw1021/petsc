#if !defined(__PETSCVEC_KOKKOS_HPP)
#define __PETSCVEC_KOKKOS_HPP

#include <petscvec.h>

#if defined(PETSC_HAVE_KOKKOS)
  #include <Kokkos_Core.hpp>
  #include <Kokkos_DualView.hpp>
  #include <Kokkos_OffsetView.hpp>

  using DefaultExecutionSpace   = Kokkos::DefaultExecutionSpace;
  using DefaultMemorySpace      = Kokkos::DefaultExecutionSpace::memory_space;
  using HostMemorySpace         = Kokkos::HostSpace;

  template<class MemorySpace> using PetscScalarKokkosViewType                  = Kokkos::View<PetscScalar*,MemorySpace>;
  template<class MemorySpace> using ConstPetscScalarKokkosViewType             = Kokkos::View<const PetscScalar*,MemorySpace>;

  template<class MemorySpace> using PetscScalarKokkosOffsetViewType            = Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>;
  template<class MemorySpace> using PetscScalarKokkosOffsetView1DType          = Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>;
  template<class MemorySpace> using PetscScalarKokkosOffsetView2DType          = Kokkos::Experimental::OffsetView<PetscScalar**,MemorySpace>;
  template<class MemorySpace> using PetscScalarKokkosOffsetView3DType          = Kokkos::Experimental::OffsetView<PetscScalar***,MemorySpace>;
  template<class MemorySpace> using PetscScalarKokkosOffsetView4DType          = Kokkos::Experimental::OffsetView<PetscScalar****,MemorySpace>;

  template<class MemorySpace> using ConstPetscScalarKokkosOffsetViewType       = Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>;
  template<class MemorySpace> using ConstPetscScalarKokkosOffsetView1DType     = Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>;
  template<class MemorySpace> using ConstPetscScalarKokkosOffsetView2DType     = Kokkos::Experimental::OffsetView<const PetscScalar**,MemorySpace>;
  template<class MemorySpace> using ConstPetscScalarKokkosOffsetView3DType     = Kokkos::Experimental::OffsetView<const PetscScalar***,MemorySpace>;
  template<class MemorySpace> using ConstPetscScalarKokkosOffsetView4DType     = Kokkos::Experimental::OffsetView<const PetscScalar****,MemorySpace>;

  /* Some shortcut types */
  using PetscScalarKokkosDualView                  = Kokkos::DualView<PetscScalar*>;
  using PetscScalarKokkosView                      = PetscScalarKokkosViewType<DefaultMemorySpace>;
  using PetscScalarKokkosViewHost                  = PetscScalarKokkosViewType<Kokkos::HostSpace>;
  using ConstPetscScalarKokkosView                 = ConstPetscScalarKokkosViewType<DefaultMemorySpace>;
  using ConstPetscScalarKokkosViewHost             = ConstPetscScalarKokkosViewType<Kokkos::HostSpace>;

  using PetscScalarKokkosOffsetView                = PetscScalarKokkosOffsetViewType<DefaultMemorySpace>;
  using PetscScalarKokkosOffsetView1D              = PetscScalarKokkosOffsetView1DType<DefaultMemorySpace>;
  using PetscScalarKokkosOffsetView2D              = PetscScalarKokkosOffsetView2DType<DefaultMemorySpace>;
  using PetscScalarKokkosOffsetView3D              = PetscScalarKokkosOffsetView3DType<DefaultMemorySpace>;
  using PetscScalarKokkosOffsetView4D              = PetscScalarKokkosOffsetView4DType<DefaultMemorySpace>;

  using PetscScalarKokkosOffsetViewHost            = PetscScalarKokkosOffsetViewType<Kokkos::HostSpace>;
  using PetscScalarKokkosOffsetView1DHost          = PetscScalarKokkosOffsetView1DType<Kokkos::HostSpace>;
  using PetscScalarKokkosOffsetView2DHost          = PetscScalarKokkosOffsetView2DType<Kokkos::HostSpace>;
  using PetscScalarKokkosOffsetView3DHost          = PetscScalarKokkosOffsetView3DType<Kokkos::HostSpace>;
  using PetscScalarKokkosOffsetView4DHost          = PetscScalarKokkosOffsetView4DType<Kokkos::HostSpace>;


  using ConstPetscScalarKokkosOffsetView           = ConstPetscScalarKokkosOffsetViewType<DefaultMemorySpace>;
  using ConstPetscScalarKokkosOffsetView1D         = ConstPetscScalarKokkosOffsetView1DType<DefaultMemorySpace>;
  using ConstPetscScalarKokkosOffsetView2D         = ConstPetscScalarKokkosOffsetView2DType<DefaultMemorySpace>;
  using ConstPetscScalarKokkosOffsetView3D         = ConstPetscScalarKokkosOffsetView3DType<DefaultMemorySpace>;
  using ConstPetscScalarKokkosOffsetView4D         = ConstPetscScalarKokkosOffsetView4DType<DefaultMemorySpace>;

  using ConstPetscScalarKokkosOffsetViewHost       = ConstPetscScalarKokkosOffsetViewType<Kokkos::HostSpace>;
  using ConstPetscScalarKokkosOffsetView1DHost     = ConstPetscScalarKokkosOffsetView1DType<Kokkos::HostSpace>;
  using ConstPetscScalarKokkosOffsetView2DHost     = ConstPetscScalarKokkosOffsetView2DType<Kokkos::HostSpace>;
  using ConstPetscScalarKokkosOffsetView3DHost     = ConstPetscScalarKokkosOffsetView3DType<Kokkos::HostSpace>;
  using ConstPetscScalarKokkosOffsetView4DHost     = ConstPetscScalarKokkosOffsetView4DType<Kokkos::HostSpace>;

  /* Routines to get/restore Kokkos Views from PETSc vectors */

  /* Like VecGetArrayRead() */
  template<class MemorySpace> PetscErrorCode VecGetKokkosView(Vec,ConstPetscScalarKokkosViewType<MemorySpace>*);
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosView(Vec,ConstPetscScalarKokkosViewType<MemorySpace>*){return 0;}

  /* Like VecGetArray() and VecGetArrayWrite(), using the overwrite argument to select VecGetArrayWrite().
     In other words, when overwrite=TRUE, caller indicates it will not use the vector data and it will simply overwrite it.
  */
  template<class MemorySpace> PetscErrorCode VecGetKokkosView(Vec,PetscScalarKokkosViewType<MemorySpace>*,PetscBool=PETSC_FALSE);
  template<class MemorySpace> PetscErrorCode VecRestoreKokkosView(Vec,PetscScalarKokkosViewType<MemorySpace>*,PetscBool=PETSC_FALSE);
#endif

#endif
