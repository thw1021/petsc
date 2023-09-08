#pragma once

#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petsc/private/vecimpl_kokkos.hpp>
#include <petscsf.h>

#if defined(PETSC_USE_DEBUG)
  #define VecErrorIfNotKokkos(v) \
    do { \
      PetscBool isKokkos = PETSC_FALSE; \
      PetscCall(PetscObjectTypeCompareAny((PetscObject)(v), &isKokkos, VECSEQKOKKOS, VECMPIKOKKOS, VECKOKKOS, "")); \
      PetscCheck(isKokkos, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Calling VECKOKKOS methods on a non-VECKOKKOS object"); \
    } while (0)
#else
  #define VecErrorIfNotKokkos(v) \
    do { \
      (void)(v); \
    } while (0)
#endif

/* Stuff related to Vec_Kokkos */

struct VecCOOStruct_SeqKokkos {
  PetscInt             m;
  PetscCount           coo_n;
  PetscCount           tot1;
  PetscCountKokkosView jmap1_d;
  PetscCountKokkosView perm1_d;
  VecCOOStruct_SeqKokkos(VecCOOStruct_Seq const *coo_host)
  {
    m     = coo_host->m;
    coo_n = coo_host->coo_n;
    tot1  = coo_host->tot1;
    PetscCallCXXAbort(PETSC_COMM_SELF, jmap1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->jmap1, coo_host->m + 1)));
    PetscCallCXXAbort(PETSC_COMM_SELF, perm1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->perm1, coo_host->tot1)));
  }
};

struct VecCOOStruct_MPIKokkos {
  PetscInt             m;
  PetscCount           coo_n;
  PetscCount           tot1;
  PetscCount           nnz2;
  PetscCount           sendlen, recvlen;
  PetscCountKokkosView jmap1_d;
  PetscCountKokkosView perm1_d;
  PetscCountKokkosView imap2_d;
  PetscCountKokkosView jmap2_d;
  PetscCountKokkosView perm2_d;
  PetscCountKokkosView Cperm_d;
  PetscSF              coo_sf;
  VecCOOStruct_MPIKokkos(VecCOOStruct_MPI const *coo_host)
  {
    m       = coo_host->m;
    coo_n   = coo_host->coo_n;
    tot1    = coo_host->tot1;
    nnz2    = coo_host->nnz2;
    sendlen = coo_host->sendlen;
    recvlen = coo_host->recvlen;
    PetscCallCXXAbort(PETSC_COMM_SELF, jmap1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->jmap1, m + 1)));
    PetscCallCXXAbort(PETSC_COMM_SELF, perm1_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->perm1, coo_host->tot1)));
    PetscCallCXXAbort(PETSC_COMM_SELF, imap2_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->imap2, coo_host->nnz2)));
    PetscCallCXXAbort(PETSC_COMM_SELF, jmap2_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->jmap2, coo_host->nnz2 + 1)));
    PetscCallCXXAbort(PETSC_COMM_SELF, perm2_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->perm2, coo_host->recvlen)));
    PetscCallCXXAbort(PETSC_COMM_SELF, Cperm_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_host->Cperm, coo_host->sendlen)));
    PetscCallAbort(PETSC_COMM_SELF, PetscObjectReference((PetscObject)(coo_host->coo_sf)));
    coo_sf = coo_host->coo_sf;
  }
  ~VecCOOStruct_MPIKokkos() { PetscCallAbort(PETSC_COMM_SELF, PetscSFDestroy(&coo_sf)); }
};

struct Vec_Kokkos {
  PetscScalarKokkosDualView v_dual;

  PetscScalarKokkosView sendbuf_d, recvbuf_d; /* Buffers for remote values in VecSetValuesCOO() */

  /* Construct Vec_Kokkos with the given array(s). n is the length of the array.
    If n != 0, host array (array_h) must not be NULL.
    If device array (array_d) is NULL, then a proper device mirror will be allocated.
    Otherwise, the mirror will be created using the given array_d.
  */
  Vec_Kokkos(PetscInt n, PetscScalar *array_h, PetscScalar *array_d = NULL)
  {
    PetscScalarKokkosViewHost v_h(array_h, n);
    PetscScalarKokkosView     v_d;

    if (array_d) {
      v_d = PetscScalarKokkosView(array_d, n); /* Use the given device array */
    } else {
      v_d = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, DefaultMemorySpace(), v_h); /* Create a mirror in DefaultMemorySpace but do not copy values */
    }
    v_dual = PetscScalarKokkosDualView(v_d, v_h);
    if (!array_d) v_dual.modify_host();
  }

  /* SFINAE: Update the object with an array in the given memory space,
     assuming the given array contains the latest value for this vector.
   */
  template <typename MemorySpace, std::enable_if_t<std::is_same<MemorySpace, Kokkos::HostSpace>::value, bool> = true, std::enable_if_t<std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  PetscErrorCode UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosViewHost v_h(array, v_dual.extent(0));
    PetscFunctionBegin;
    /* Kokkos said they would add error-checking so that users won't accidentally pass two different Views in this case */
    PetscCallCXX(v_dual = PetscScalarKokkosDualView(v_h, v_h));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename MemorySpace, std::enable_if_t<std::is_same<MemorySpace, Kokkos::HostSpace>::value, bool> = true, std::enable_if_t<!std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  PetscErrorCode UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosViewHost v_h(array, v_dual.extent(0));
    PetscFunctionBegin;
    PetscCallCXX(v_dual = PetscScalarKokkosDualView(v_dual.view<DefaultMemorySpace>(), v_h));
    PetscCallCXX(v_dual.modify_host());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <typename MemorySpace, std::enable_if_t<!std::is_same<MemorySpace, Kokkos::HostSpace>::value, bool> = true, std::enable_if_t<std::is_same<MemorySpace, DefaultMemorySpace>::value, bool> = true>
  PetscErrorCode UpdateArray(PetscScalar *array)
  {
    PetscScalarKokkosView v_d(array, v_dual.extent(0));
    PetscFunctionBegin;
    PetscCallCXX(v_dual = PetscScalarKokkosDualView(v_d, v_dual.view<Kokkos::HostSpace>()));
    PetscCallCXX(v_dual.modify_device());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static inline PetscErrorCode VecCOOStructDestroy_SeqKokkos(void *data)
  {
    VecCOOStruct_SeqKokkos *coo_struct = (VecCOOStruct_SeqKokkos *)data;

    PetscFunctionBegin;
    delete coo_struct;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static inline PetscErrorCode VecCOOStructDestroy_MPIKokkos(void *data)
  {
    VecCOOStruct_MPIKokkos *coo_struct = (VecCOOStruct_MPIKokkos *)data;

    PetscFunctionBegin;
    delete coo_struct;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode SetUpCOO(Vec vec, const Vec_Seq *vecseq)
  {
    VecCOOStruct_SeqKokkos *coo_struct;
    VecCOOStruct_Seq       *coo_host;
    PetscCountKokkosView    jmap1_d, perm1_d;
    PetscContainer          container, kokkos_container;

    PetscFunctionBegin;
    PetscCall(PetscObjectQuery((PetscObject)vec, "__PETSc_VecCOOStruct_Host", (PetscObject *)&container));
    PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not found VecCOOStruct on this vectors");
    PetscCall(PetscContainerGetPointer(container, (void **)&coo_host));
    PetscCallCXX(coo_struct = new VecCOOStruct_SeqKokkos(coo_host));
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &kokkos_container));
    PetscCall(PetscContainerSetPointer(kokkos_container, (void *)coo_struct));
    PetscCall(PetscContainerSetUserDestroy(kokkos_container, VecCOOStructDestroy_SeqKokkos));
    PetscCall(PetscObjectCompose((PetscObject)vec, "__PETSc_VecCOOStruct_Device", (PetscObject)kokkos_container));
    PetscCall(PetscObjectDereference((PetscObject)kokkos_container));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode SetUpCOO(Vec vec, const Vec_MPI *vecmpi)
  {
    VecCOOStruct_MPIKokkos *coo_struct;
    VecCOOStruct_MPI       *coo_host;
    PetscCountKokkosView    jmap1_d, perm1_d;
    PetscContainer          container, kokkos_container;

    PetscFunctionBegin;
    PetscCall(PetscObjectQuery((PetscObject)vec, "__PETSc_VecCOOStruct_Host", (PetscObject *)&container));
    PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not found VecCOOStruct on this vectors");
    PetscCall(PetscContainerGetPointer(container, (void **)&coo_host));
    PetscCallCXX(coo_struct = new VecCOOStruct_MPIKokkos(coo_host));
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &kokkos_container));
    PetscCall(PetscContainerSetPointer(kokkos_container, (void *)coo_struct));
    PetscCall(PetscContainerSetUserDestroy(kokkos_container, VecCOOStructDestroy_MPIKokkos));
    PetscCall(PetscObjectCompose((PetscObject)vec, "__PETSc_VecCOOStruct_Device", (PetscObject)kokkos_container));
    PetscCall(PetscObjectDereference((PetscObject)kokkos_container));
    PetscCallCXX(sendbuf_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(vecmpi->sendbuf, coo_host->sendlen)));
    PetscCallCXX(recvbuf_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscScalarKokkosViewHost(vecmpi->recvbuf, coo_host->recvlen)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

PETSC_INTERN PetscErrorCode VecAbs_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReciprocal_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecDotNorm2_SeqKokkos(Vec, Vec, PetscScalar *, PetscScalar *);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqKokkos(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqKokkos(Vec, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqKokkos(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecMTDot_SeqKokkos(Vec, PetscInt, const Vec[], PetscScalar *);
PETSC_INTERN PetscErrorCode VecSet_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqKokkos(Vec, PetscInt, const PetscScalar *, Vec *);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqKokkos(Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqKokkos(Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqKokkos(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqKokkos(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecDot_SeqKokkos(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecTDot_SeqKokkos(Vec, Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecScale_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqKokkos(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqKokkos(Vec, PetscScalar, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecConjugate_SeqKokkos(Vec xin);
PETSC_INTERN PetscErrorCode VecNorm_SeqKokkos(Vec, NormType, PetscReal *);
PETSC_INTERN PetscErrorCode VecErrorWeightedNorms_SeqKokkos(Vec, Vec, Vec, NormType, PetscReal, Vec, PetscReal, Vec, PetscReal, PetscReal *, PetscInt *, PetscReal *, PetscInt *, PetscReal *, PetscInt *);
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqKokkos_Private(Vec, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPIKokkos_Private(Vec, PetscBool, PetscInt, const PetscScalar *);
PETSC_INTERN PetscErrorCode VecCreate_Kokkos(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqKokkos(Vec, PetscScalar, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos(Vec, PetscRandom);
PETSC_INTERN PetscErrorCode VecGetLocalVector_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecRestoreLocalVector_SeqKokkos(Vec, Vec);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecCopy_SeqKokkos_Private(Vec, Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqKokkos_Private(Vec, PetscRandom);
PETSC_INTERN PetscErrorCode VecResetArray_SeqKokkos_Private(Vec);
PETSC_INTERN PetscErrorCode VecMin_SeqKokkos(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecMax_SeqKokkos(Vec, PetscInt *, PetscReal *);
PETSC_INTERN PetscErrorCode VecSum_SeqKokkos(Vec, PetscScalar *);
PETSC_INTERN PetscErrorCode VecShift_SeqKokkos(Vec, PetscScalar);
PETSC_INTERN PetscErrorCode VecGetArray_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqKokkos(Vec, PetscScalar **);

PETSC_INTERN PetscErrorCode VecGetArrayAndMemType_SeqKokkos(Vec, PetscScalar **, PetscMemType *);
PETSC_INTERN PetscErrorCode VecRestoreArrayAndMemType_SeqKokkos(Vec, PetscScalar **);
PETSC_INTERN PetscErrorCode VecGetArrayWriteAndMemType_SeqKokkos(Vec, PetscScalar **, PetscMemType *);
PETSC_INTERN PetscErrorCode VecGetSubVector_Kokkos_Private(Vec, PetscBool, IS, Vec *);
PETSC_INTERN PetscErrorCode VecRestoreSubVector_SeqKokkos(Vec, IS, Vec *);
