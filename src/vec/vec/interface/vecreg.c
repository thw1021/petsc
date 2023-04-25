
#include <petsc/private/vecimpl.h> /*I "petscvec.h"  I*/

PetscFunctionList VecList              = NULL;
PetscBool         VecRegisterAllCalled = PETSC_FALSE;

/*@C
  VecSetType - Builds a vector, for a particular vector implementation.

  Collective

  Input Parameters:
+ vec     - The vector object
- newType - The name of the vector type

  Options Database Key:
. -vec_type <type> - Sets the vector type; use -help for a list
                     of available types

  Level: intermediate

  Notes:
  See `VecType` for available vector types (for instance, `VECSEQ` or `VECMPI`)

  Use `VecDuplicate()` or `VecDuplicateVecs()` to form additional vectors of the same type as an existing vector.

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecGetType()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`
@*/
PetscErrorCode VecSetType(Vec vec, VecType newType)
{
  PetscErrorCode (*r)(Vec);
  VecType     curType;
  PetscBool   match, upcast;
  PetscMPIInt size;
  PetscBool   srcMPI = PETSC_FALSE, srcSeq = PETSC_FALSE; // type info of the current type
  PetscBool   dstMPI = PETSC_FALSE, dstSeq = PETSC_FALSE; // type info of the new type
  MPI_Comm    comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);

  /* return if same type */
  PetscCall(PetscObjectTypeCompare((PetscObject)vec, newType, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscObjectGetComm((PetscObject)vec, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCall(VecGetType(vec, &curType));
  if (!curType) goto newvec; // vec's type is not set yet

  PetscCall(PetscStrbeginswith(newType, VECSEQ, &dstSeq));
  PetscCheck(!(size > 1 && dstSeq), comm, PETSC_ERR_ARG_WRONG, "Cannot convert MPI vectors to sequential ones");

  /* upcasting (e.g., from VecSeqCUDA to VecSeq) is not supported. VecBindToCPU(v,PETSC_TRUE) could
     achieve that. It looks like a wrong design in user's code, thus no support.
   */
  PetscCall(PetscStrbeginswith(curType, newType, &upcast)); /* e.g., error with curType="seqcuda", newType="seq" */
  PetscCheck(!upcast, comm, PETSC_ERR_ARG_WRONG, "Converting a Vec of type %s to %s is not supported", curType, newType);

  /* don't count on comm size to determine vec type. A vector could be VECMPI with comm size = 1 */
  PetscCall(PetscObjectTypeCompare((PetscObject)vec, VECMPI, &srcMPI));
  if (!srcMPI) PetscCall(PetscObjectTypeCompare((PetscObject)vec, VECSEQ, &srcSeq));

  /* return if the conversion is seq=>standard/seq or mpi=>standard/mpi */
  PetscCall(PetscStrcmp(newType, VECSTANDARD, &match));
  if (!match) PetscCall(PetscStrcmp(newType, VECMPI, &dstMPI));
  if (!match && !dstMPI) PetscCall(PetscStrcmp(newType, VECSEQ, &dstSeq));
  if (match || ((srcSeq || srcMPI) && (dstSeq || dstMPI))) PetscFunctionReturn(PETSC_SUCCESS);

    /* downcasting (e.g., from VecSeq to VecSeqCUDA) */
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscStrcmp(newType, VECCUDA, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPICUDA : VECSEQCUDA, &match));
    if (match) PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscStrcmp(newType, VECHIP, &match));
  if (match) {
    PetscCall(PetscObjectTypeCompare((PetscObject)vec, size > 1 ? VECMPIHIP : VECSEQHIP, &match));
    if (match) PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(PetscStrcmp(newType, VECVIENNACL, &match));
  if (!match) PetscCall(PetscStrcmp(newType, VECMPIVIENNACL, &dstMPI));
  if (!match && !dstMPI) PetscCall(PetscStrcmp(newType, VECSEQVIENNACL, &dstSeq));
  if (match || (srcSeq && (dstSeq || dstMPI))) {
    PetscCall(VecConvert_Seq_SeqViennaCL_inplace(vec));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (match || (srcMPI && (dstMPI || dstSeq))) {
    PetscCall(VecConvert_MPI_MPIViennaCL_inplace(vec));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscStrcmp(newType, VECKOKKOS, &match));
  if (!match) PetscCall(PetscStrcmp(newType, VECMPIKOKKOS, &dstMPI));
  if (!match && !dstMPI) PetscCall(PetscStrcmp(newType, VECSEQKOKKOS, &dstSeq));
  if (match || (srcSeq && (dstSeq || dstMPI))) { /*  allow Seq => MPIKokkos with comm size = 1 */
    PetscCall(VecConvert_Seq_SeqKokkos_inplace(vec));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (match || (srcMPI && (dstMPI || dstSeq))) { /* allow MPI => SeqKokkos with comm size = 1*/
    PetscCall(VecConvert_MPI_MPIKokkos_inplace(vec));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif

newvec:
  /* Other conversion scenarios; just destroy the old vector and build the new type from scratch */
  PetscCall(PetscFunctionListFind(VecList, newType, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown vector type: %s", newType);
  if (curType) { /* no need to destroy a vec without type */
    PetscTryTypeMethod(vec, destroy);
    PetscCall(PetscMemzero(vec->ops, sizeof(struct _VecOps)));
  } else {
    PetscCall(PetscFree(vec->defaultrandtype));
    PetscCall(PetscStrallocpy(PETSCRANDER48, &vec->defaultrandtype));
  }
  if (vec->map->n < 0 && vec->map->N < 0) {
    vec->ops->create = r;
    vec->ops->load   = VecLoad_Default;
  } else {
    PetscCall((*r)(vec));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecGetType - Gets the vector type name (as a string) from a `Vec`.

  Not Collective

  Input Parameter:
. vec  - The vector

  Output Parameter:
. type - The `VecType` of the vector

  Level: intermediate

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecGetType()`, `VecCreate()`, `VecDuplicate()`, `VecDuplicateVecs()`
@*/
PetscErrorCode VecGetType(Vec vec, VecType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidPointer(type, 2);
  PetscCall(VecRegisterAll());
  *type = ((PetscObject)vec)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecGetRootType_Private(Vec vec, VecType *vtype)
{
  PetscBool iscuda, iship, iskokkos, isvcl;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidPointer(vtype, 2);
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &iscuda, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &iship, VECHIP, VECMPIHIP, VECSEQHIP, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &iskokkos, VECKOKKOS, VECMPIKOKKOS, VECSEQKOKKOS, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)vec, &isvcl, VECVIENNACL, VECMPIVIENNACL, VECSEQVIENNACL, ""));
  if (iscuda) {
    *vtype = VECCUDA;
  } else if (iship) {
    *vtype = VECHIP;
  } else if (iskokkos) {
    *vtype = VECKOKKOS;
  } else if (isvcl) {
    *vtype = VECVIENNACL;
  } else {
    *vtype = VECSTANDARD;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*--------------------------------------------------------------------------------------------------------------------*/

/*@C
  VecRegister -  Adds a new vector component implementation

  Not Collective

  Input Parameters:
+ sname        - The name of a new user-defined creation routine
- function - The creation routine

  Notes:
  `VecRegister()` may be called multiple times to add several user-defined vectors

  Sample usage:
.vb
    VecRegister("my_vec",MyVectorCreate);
.ve

  Then, your vector type can be chosen with the procedural interface via
.vb
    VecCreate(MPI_Comm, Vec *);
    VecSetType(Vec,"my_vector_name");
.ve
   or at runtime via the option
.vb
    -vec_type my_vector_name
.ve

  Level: advanced

.seealso: `VecRegisterAll()`, `VecRegisterDestroy()`
@*/
PetscErrorCode VecRegister(const char sname[], PetscErrorCode (*function)(Vec))
{
  PetscFunctionBegin;
  PetscCall(VecInitializePackage());
  PetscCall(PetscFunctionListAdd(&VecList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
