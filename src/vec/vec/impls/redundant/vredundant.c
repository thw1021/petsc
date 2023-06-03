
#include <../src/vec/vec/impls/dvecimpl.h> /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>

typedef struct {
  VECSEQHEADER
  PetscInt do_not_check_redundancy;
} Vec_Redundant;

PETSC_INTERN PetscErrorCode VecRedundantPushDoNotCheckRedundancy(Vec v)
{
  Vec_Redundant *vr = (Vec_Redundant *) v->data;

  PetscFunctionBegin;
  vr->do_not_check_redundancy++;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode VecRedundantPopDoNotCheckRedundancy(Vec v)
{
  Vec_Redundant *vr = (Vec_Redundant *) v->data;

  PetscFunctionBegin;
  vr->do_not_check_redundancy = PetscMax(0, vr->do_not_check_redundancy - 1);
  if (PetscDefined(USE_DEBUG) && vr->do_not_check_redundancy == 0) {
    const PetscScalar *array;

    PetscCall(VecGetArrayRead(v, &array));
    unsigned int hash = PetscArrayHash(array, v->map->N, NULL);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)v));
    PetscCall(VecRestoreArrayRead(v, &array));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetValues_Redundant(Vec xin, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode m)
{
  Vec_Redundant *vr = (Vec_Redundant *) xin->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(xin, ni, 2);
  PetscValidLogicalCollectiveEnum(xin, m, 5);
  if (PetscDefined(USE_DEBUG) && !vr->do_not_check_redundancy) {
    unsigned int hash = PetscArrayHash(ix, ni, NULL);
    hash = PetscArrayHash(y, ni, &hash);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)xin));
  }
  PetscCall(VecSetValues_Seq(xin, ni, ix, y, m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetValuesBlocked_Redundant(Vec xin, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode m)
{
  Vec_Redundant *vr = (Vec_Redundant *) xin->data;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(xin, ni, 2);
  PetscValidLogicalCollectiveEnum(xin, m, 5);
  if (PetscDefined(USE_DEBUG) && !vr->do_not_check_redundancy) {
    unsigned int hash = PetscArrayHash(ix, ni, NULL);
    hash = PetscArrayHash(y, ni, &hash);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)xin));
  }
  PetscCall(VecSetValuesBlocked_Seq(xin, ni, ix, y, m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetLocalSize_Redundant(Vec v, PetscInt *size)
{
  PetscFunctionBegin;
  *size = v->map->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetRandom_Redundant(Vec xin, PetscRandom r)
{
  MPI_Comm    comm = PetscObjectComm((PetscObject)xin);
  PetscScalar *xx;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecGetArrayWrite(xin, &xx));
  PetscCall(PetscRandomGetValues(r, xin->map->n, xx));
  PetscCallMPI(MPI_Bcast(xx, xin->map->N, MPIU_SCALAR, size - 1, comm));
  PetscCall(VecRestoreArrayWrite(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecPlaceArray_Redundant(Vec vin, const PetscScalar *a)
{
  Vec_Redundant *vr = (Vec_Redundant *) vin->data;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG) && !vr->do_not_check_redundancy) {
    unsigned int hash = PetscArrayHash(a, vin->map->N, NULL);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)vin));
  }
  PetscCall(VecPlaceArray_Seq(vin, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecReplaceArray_Redundant(Vec vin, const PetscScalar *a)
{
  Vec_Redundant *vr = (Vec_Redundant *) vin->data;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG) && !vr->do_not_check_redundancy) {
    unsigned int hash = PetscArrayHash(a, vin->map->N, NULL);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)vin));
  }
  PetscCall(VecReplaceArray_Seq(vin, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecLoad_Redundant(Vec vec, PetscViewer viewer)
{
  MPI_Comm     comm = PetscObjectComm((PetscObject)vec);
  PetscScalar *a;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecLoad_Default(vec, viewer));
  PetscCall(VecGetArray(vec, &a));
  PetscCallMPI(MPI_Bcast(a, vec->map->N, MPIU_SCALAR, size - 1, comm));
  PetscCall(VecRestoreArray(vec, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetPreallocationCOO_Redundant(Vec x, PetscCount coo_n, const PetscInt coo_i[])
{
  Vec_Redundant *vr = (Vec_Redundant *) x->data;
  PetscBool have_coo_i = coo_i ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(x, coo_n, 2);
  PetscValidLogicalCollectiveBool(x, have_coo_i, 3);
  if (PetscDefined(USE_DEBUG) && !vr->do_not_check_redundancy && coo_i) {
    unsigned int hash = PetscArrayHash(coo_i, x->map->N, NULL);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetValuesCOO_Redundant(Vec x, const PetscScalar coo_v[], InsertMode imode)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveEnum(x, imode, 2);
  if (PetscDefined(USE_DEBUG)) {
    Vec_Redundant    *vs    = (Vec_Redundant *)x->data;
    unsigned int      hash = PetscArrayHash(coo_v, vs->coo_n, NULL);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)x));
  }
  PetscCall(VecSetValuesCOO_Seq(x, coo_v, imode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRestoreArray_Redundant(Vec x, PetscScalar **a)
{
  Vec_Redundant *vr = (Vec_Redundant *) x->data;

  PetscFunctionBegin;
  if (!vr->do_not_check_redundancy && PetscDefined(USE_DEBUG)) {
    unsigned int hash = PetscArrayHash((*a), x->map->N, NULL);
    PetscCheckLogicalCollectiveHash(hash, PetscObjectComm((PetscObject)x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _VecOps RvOps = {PetscDesignatedInitializer(duplicate, VecDuplicate_Seq), /* 1 */
                               PetscDesignatedInitializer(duplicatevecs, VecDuplicateVecs_Default),
                               PetscDesignatedInitializer(destroyvecs, VecDestroyVecs_Default),
                               PetscDesignatedInitializer(dot, VecDot_Seq),
                               PetscDesignatedInitializer(mdot, VecMDot_Seq),
                               PetscDesignatedInitializer(norm, VecNorm_Seq),
                               PetscDesignatedInitializer(tdot, VecTDot_Seq),
                               PetscDesignatedInitializer(mtdot, VecMTDot_Seq),
                               PetscDesignatedInitializer(scale, VecScale_Seq),
                               PetscDesignatedInitializer(copy, VecCopy_Seq), /* 10 */
                               PetscDesignatedInitializer(set, VecSet_Seq),
                               PetscDesignatedInitializer(swap, VecSwap_Seq),
                               PetscDesignatedInitializer(axpy, VecAXPY_Seq),
                               PetscDesignatedInitializer(axpby, VecAXPBY_Seq),
                               PetscDesignatedInitializer(maxpy, VecMAXPY_Seq),
                               PetscDesignatedInitializer(aypx, VecAYPX_Seq),
                               PetscDesignatedInitializer(waxpy, VecWAXPY_Seq),
                               PetscDesignatedInitializer(axpbypcz, VecAXPBYPCZ_Seq),
                               PetscDesignatedInitializer(pointwisemult, VecPointwiseMult_Seq),
                               PetscDesignatedInitializer(pointwisedivide, VecPointwiseDivide_Seq),
                               PetscDesignatedInitializer(setvalues, VecSetValues_Redundant), /* 20 */
                               PetscDesignatedInitializer(assemblybegin, NULL),
                               PetscDesignatedInitializer(assemblyend, NULL),
                               PetscDesignatedInitializer(getarray, NULL),
                               PetscDesignatedInitializer(getsize, VecGetSize_MPI),
                               PetscDesignatedInitializer(getlocalsize, VecGetLocalSize_Redundant),
                               PetscDesignatedInitializer(restorearray, VecRestoreArray_Redundant),
                               PetscDesignatedInitializer(max, VecMax_Seq),
                               PetscDesignatedInitializer(min, VecMin_Seq),
                               PetscDesignatedInitializer(setrandom, VecSetRandom_Redundant),
                               PetscDesignatedInitializer(setoption, VecSetOption_Seq),
                               PetscDesignatedInitializer(setvaluesblocked, VecSetValuesBlocked_Redundant),
                               PetscDesignatedInitializer(destroy, VecDestroy_Seq),
                               PetscDesignatedInitializer(view, VecView_MPI),
                               PetscDesignatedInitializer(placearray, VecPlaceArray_Redundant),
                               PetscDesignatedInitializer(replacearray, VecReplaceArray_Redundant),
                               PetscDesignatedInitializer(dot_local, VecDot_Seq),
                               PetscDesignatedInitializer(tdot_local, VecTDot_Seq),
                               PetscDesignatedInitializer(norm_local, VecNorm_Seq),
                               PetscDesignatedInitializer(mdot_local, VecMDot_Seq),
                               PetscDesignatedInitializer(mtdot_local, VecMTDot_Seq),
                               PetscDesignatedInitializer(load, VecLoad_Redundant),
                               PetscDesignatedInitializer(reciprocal, VecReciprocal_Default),
                               PetscDesignatedInitializer(conjugate, VecConjugate_Seq),
                               PetscDesignatedInitializer(setlocaltoglobalmapping, NULL),
                               PetscDesignatedInitializer(setvalueslocal, NULL),
                               PetscDesignatedInitializer(resetarray, VecResetArray_Seq),
                               PetscDesignatedInitializer(setfromoptions, NULL),
                               PetscDesignatedInitializer(maxpointwisedivide, VecMaxPointwiseDivide_Seq),
                               PetscDesignatedInitializer(pointwisemax, VecPointwiseMax_Seq),
                               PetscDesignatedInitializer(pointwisemaxabs, VecPointwiseMaxAbs_Seq),
                               PetscDesignatedInitializer(pointwisemin, VecPointwiseMin_Seq),
                               PetscDesignatedInitializer(getvalues, VecGetValues_Seq),
                               PetscDesignatedInitializer(sqrt, NULL),
                               PetscDesignatedInitializer(abs, NULL),
                               PetscDesignatedInitializer(exp, NULL),
                               PetscDesignatedInitializer(log, NULL),
                               PetscDesignatedInitializer(shift, NULL),
                               PetscDesignatedInitializer(create, NULL),
                               PetscDesignatedInitializer(stridegather, VecStrideGather_Default),
                               PetscDesignatedInitializer(stridescatter, VecStrideScatter_Default),
                               PetscDesignatedInitializer(dotnorm2, NULL),
                               PetscDesignatedInitializer(getsubvector, NULL),
                               PetscDesignatedInitializer(restoresubvector, NULL),
                               PetscDesignatedInitializer(getarrayread, NULL),
                               PetscDesignatedInitializer(restorearrayread, NULL),
                               PetscDesignatedInitializer(stridesubsetgather, VecStrideSubSetGather_Default),
                               PetscDesignatedInitializer(stridesubsetscatter, VecStrideSubSetScatter_Default),
                               PetscDesignatedInitializer(viewnative, VecView_MPI),
                               PetscDesignatedInitializer(loadnative, NULL),
                               PetscDesignatedInitializer(createlocalvector, NULL),
                               PetscDesignatedInitializer(getlocalvector, NULL),
                               PetscDesignatedInitializer(restorelocalvector, NULL),
                               PetscDesignatedInitializer(getlocalvectorread, NULL),
                               PetscDesignatedInitializer(restorelocalvectorread, NULL),
                               PetscDesignatedInitializer(bindtocpu, NULL),
                               PetscDesignatedInitializer(getarraywrite, NULL),
                               PetscDesignatedInitializer(restorearraywrite, VecRestoreArray_Redundant),
                               PetscDesignatedInitializer(getarrayandmemtype, NULL),
                               PetscDesignatedInitializer(restorearrayandmemtype, NULL),
                               PetscDesignatedInitializer(getarrayreadandmemtype, NULL),
                               PetscDesignatedInitializer(restorearrayreadandmemtype, NULL),
                               PetscDesignatedInitializer(getarraywriteandmemtype, NULL),
                               PetscDesignatedInitializer(restorearraywriteandmemtype, NULL),
                               PetscDesignatedInitializer(concatenate, NULL),
                               PetscDesignatedInitializer(sum, NULL),
                               PetscDesignatedInitializer(setpreallocationcoo, VecSetPreallocationCOO_Redundant),
                               PetscDesignatedInitializer(setvaluescoo, VecSetValuesCOO_Redundant),
                               PetscDesignatedInitializer(errorwnorm, NULL)};

/*
  VecCreate_Redundant

*/
static PetscErrorCode VecCreate_Redundant_Private(Vec v, PetscBool alloc, const PetscScalar array[])
{
  PetscInt       size, rank;
  MPI_Comm       comm;
  Vec_Redundant *s;

  PetscFunctionBegin;
  PetscCall(PetscNew(&s));
  v->data = (void *)s;
  PetscCall(PetscMemcpy(v->ops, &RvOps, sizeof(RvOps)));
  v->petscnative = PETSC_TRUE;

  comm = PetscObjectComm((PetscObject) v);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (array) v->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCheck(v->map->N >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must specify global size N >= 0 for a redundant vector, cannot be PETSC_DECIDE or PETSC_DETERMINE");
  PetscCheck(rank == size - 1 || v->map->n == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "By convention the local sizes of all MPI processes but the last are 0 for a redundant vector");
  PetscCall(PetscLayoutSetUp(v->map));
  v->map->redundant = PETSC_TRUE;

  s->array           = (PetscScalar *)array;
  s->array_allocated = NULL;
  if (alloc && !array) {
    PetscInt N = v->map->N;
    PetscCall(PetscCalloc1(N, &s->array));
    s->array_allocated = s->array;
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)v, NormIds[NORM_2], 0));
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)v, NormIds[NORM_1], 0));
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)v, NormIds[NORM_INFINITY], 0));
  }
  if (array && PetscDefined(USE_DEBUG)) {
    unsigned int hash = PetscArrayHash(array, v->map->N, NULL);
    PetscCheckLogicalCollectiveHash(hash, comm);
  }
  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECREDUNDANT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_DEBUG)

  #define PetscValidLogicalCollectiveIntComm(a, b, c) \
    do { \
      PetscInt b1[2], b2[2]; \
      b1[0] = -b; \
      b1[1] = b; \
      PetscCall(MPIU_Allreduce(b1, b2, 2, MPIU_INT, MPI_MAX, a)); \
      PetscCheck(-b2[0] == b2[1], a, PETSC_ERR_ARG_WRONG, "Int value must be same on all processes, argument # %d", c); \
    } while (0)
#else
  #define PetscValidLogicalCollectiveIntComm(a, b, c) \
    do { \
      (void)(a); \
      (void)(b); \
    } while (0)
#endif

/*@
   VecCreateRedundant - Create a `VECREDUNDANT`

   Collective

   Input Parameters:
+  comm - the communicator, should be `PETSC_COMM_SELF`
-  N - the vector length

   Output Parameter:
.  v - the vector

   Level: developer

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VECREDUNDANT`, `VecCreateSeq()`, `VecCreateMPI()`, `VecCreate()`
@*/
PetscErrorCode VecCreateRedundant(MPI_Comm comm, PetscInt N, Vec *v)
{
  PetscInt       size, rank;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, N, 2);
  PetscCheck(N >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must specify global size N >= 0 for a redundant vector, cannot be PETSC_DECIDE or PETSC_DETERMINE");
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecCreate(comm, v));
  PetscCall(VecSetSizes(*v, rank == size - 1 ? N : 0, N));
  PetscCall(VecSetType(*v, VECREDUNDANT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecCreateRedundantWithArray - Create a `VECREDUNDANT` around an array

   Collective

   Input Parameters:
+  comm  - the MPI communicator to use
.  bs    - the block size, same meaning as `VecSetBlockSize()`
.  N     - the vector length
-  array - the user provided array to store the vector values: must be identical on each MPI process

   Output Parameter:
.  v - the vector

   Level: developer

   Notes:

   If the user-provided array is` NULL`, then `VecPlaceArray()` can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via `VecDestroy()`.
   The user should not free the array until the vector is destroyed.

   See `VECREDUNDANT` about this usage of this type: users almost always want `VECMPI`
   for their probolems.

.seealso: `VECREDUNDANT`, `VecCreate()`, `VecCreateRedundant()`, `VecCreateSeq()`, `VecCreateMPI()`

@*/
PetscErrorCode VecCreateRedundantWithArray(MPI_Comm comm, PetscInt bs, PetscInt N, const PetscScalar array[], Vec *v)
{
  PetscFunctionBegin;
  PetscInt       size, rank;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, N, 2);
  PetscCheck(N >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must specify global size N >= 0 for a redundant vector, cannot be PETSC_DECIDE or PETSC_DETERMINE");
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecCreate(comm, v));
  PetscCall(VecSetSizes(*v, rank == size - 1 ? N : 0, N));
  PetscCall(VecCreate_Redundant_Private(*v, PETSC_FALSE, array));
  PetscCall(VecSetType(*v, VECREDUNDANT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   VECREDUNDANT - VECREDUNDANT = "redundant" - A vector with its entries identically copied and available on each MPI process

   Level: developer

   Notes:

   All of the entries in the vector are copied on all processes.

   It is the user's responsibility to ensure the copies on all of the processes
   stay the same.  If PETSc is compiled `--with-debugging` functions that
   change the entries of the vector (`VecSetValues()`, `VecSetValuesBlocked()`,
   `VecSetValuesCOO()`, as well as changes by direct array access like
   `VecPlaceArray()`, `VecReplaceArray()`, `VecRestoreArray()` and
   `VecRestoreArrayWrite()`) will check a simple hash of the values to
   see if they are the same on all processes.

   This vector type is not for representing a solution vector of a problem
   that is being solved in parallel.  It is meant for control-flow data
   that must be accessible to all processes of a data-distributed algorithm.

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateRedundant()`, `VecCreateRedundantWithArray()`, `VECSEQ`, `VECMPI`
M*/

PetscErrorCode VecCreate_Redundant(Vec vv)
{
  PetscFunctionBegin;
  PetscCall(VecCreate_Redundant_Private(vv, PETSC_TRUE, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
