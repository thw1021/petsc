#pragma once

/* SUBMANSEC = Vec */

/*S
   Vec - Abstract PETSc vector object. Used for holding solutions and right-hand sides for linear systems, nonlinear systems, and time integrators

   Level: beginner

   Note:
   Internally the actual vector representation is generally a simple array but most PETSc code can work on other representations through this abstraction

.seealso: [](doc_vector), [](ch_vectors), `VecCreate()`, `VecType`, `VecSetType()`
S*/
typedef struct _p_Vec *Vec;

/*E
  ScatterMode - Determines the direction of a scatter in `VecScatterBegin()` and `VecScatterEnd()`

  Values:
+  `SCATTER_FORWARD`       - Scatters the values as dictated by the `VecScatterCreate()` call
.  `SCATTER_REVERSE`       - Moves the values in the opposite direction than the directions indicated in the `VecScatterCreate()` call
.  `SCATTER_FORWARD_LOCAL` - Scatters the values as dictated by the `VecScatterCreate()` call except NO MPI communication is done
-  `SCATTER_REVERSE_LOCAL` - Moves the values in the opposite direction than the directions indicated in the `VecScatterCreate()` call
                             except NO MPI communication is done

  Level: beginner

.seealso: [](ch_vectors), `VecScatter`, `VecScatterBegin()`, `VecScatterEnd()`, `SCATTER_FORWARD`, `SCATTER_REVERSE`, `SCATTER_FORWARD_LOCAL`, `SCATTER_REVERSE_LOCAL`
E*/
typedef enum {
  SCATTER_FORWARD       = 0,
  SCATTER_REVERSE       = 1,
  SCATTER_FORWARD_LOCAL = 2,
  SCATTER_REVERSE_LOCAL = 3
} ScatterMode;

/*J
   VecType - String with the name of a PETSc vector, `Vec`, type

   Level: beginner

.seealso: [](doc_vector), [](ch_vectors), `VecSetType()`, `Vec`, `VecCreate()`, `VecDestroy()`
J*/
typedef const char *VecType;
#define VECSEQ         "seq"
#define VECMPI         "mpi"
#define VECSTANDARD    "standard" /* seq on one process and mpi on multiple */
#define VECSHARED      "shared"
#define VECSEQVIENNACL "seqviennacl"
#define VECMPIVIENNACL "mpiviennacl"
#define VECVIENNACL    "viennacl" /* seqviennacl on one process and mpiviennacl on multiple */
#define VECSEQCUDA     "seqcuda"
#define VECMPICUDA     "mpicuda"
#define VECCUDA        "cuda" /* seqcuda on one process and mpicuda on multiple */
#define VECSEQHIP      "seqhip"
#define VECMPIHIP      "mpihip"
#define VECHIP         "hip" /* seqhip on one process and mpihip on multiple */
#define VECNEST        "nest"
#define VECSEQKOKKOS   "seqkokkos"
#define VECMPIKOKKOS   "mpikokkos"
#define VECKOKKOS      "kokkos" /* seqkokkos on one process and mpikokkos on multiple */

/*E
    NormType - determines what type of norm to compute with `VecNorm()`, `VecNormBegin()`/`VecNormEnd()` and `MatNorm()`.

    Values:
+    `NORM_1`         - the one norm, $||v|| = \sum_i | v_i |$. $||A|| = \max_j || A_{*j} ||$, maximum column sum
.    `NORM_2`         - the two norm, $||v|| = sqrt(\sum_i |v_i|^2)$ (vectors only)
.    `NORM_FROBENIUS` - $||A|| = sqrt(\sum_{ij} |A_{ij}|^2)$, same as `NORM_2` for vectors
.    `NORM_INFINITY`  - $||v|| = \max_i |v_i|$. $||A|| = \max_i || A_{i*} ||_1$, maximum row sum
-    `NORM_1_AND_2`   - computes both the 1 and 2 norm of a vector. The values are stored in two adjacent `PetscReal` memory locations

    Level: beginner

    Note:
    The `v` above represents a `Vec` while the `A` represents a `Mat`

.seealso: [](ch_vectors), `Vec`, `Mat`, `VecNorm()`, `VecNormBegin()`, `VecNormEnd()`, `MatNorm()`, `NORM_1`,
          `NORM_2`, `NORM_FROBENIUS`, `NORM_INFINITY`, `NORM_1_AND_2`, `ReductionType`
E*/
typedef enum {
  NORM_1         = 0,
  NORM_2         = 1,
  NORM_FROBENIUS = 2,
  NORM_INFINITY  = 3,
  NORM_1_AND_2   = 4
} NormType;
PETSC_EXTERN const char *const NormTypes[];

/*E
    ReductionType - determines what type of column reduction (one that is not a type of norm defined in `NormType`) to obtain with `MatGetColumnReductions()`

    Values:
+  `REDUCTION_SUM_REALPART`       - sum of real part of each matrix column
.  `REDUCTION_SUM_IMAGINARYPART`  - sum of imaginary part of each matrix column
.  `REDUCTION_MEAN_REALPART`      - arithmetic mean of real part of each matrix column
-  `REDUCTION_MEAN_IMAGINARYPART` - arithmetic mean of imaginary part of each matrix column

    Level: beginner

    Developer Note:
    The constants defined in `ReductionType` MUST BE DISTINCT from those defined in `NormType`.
    This is because `MatGetColumnReductions()` is used to compute both norms and other types of reductions,
    and the constants defined in both `NormType` and `ReductionType` are used to designate the desired operation.

.seealso: [](ch_vectors), `MatGetColumnReductions()`, `MatGetColumnNorms()`, `NormType`, `REDUCTION_SUM_REALPART`,
          `REDUCTION_SUM_IMAGINARYPART`, `REDUCTION_MEAN_REALPART`, `REDUCTION_MEAN_IMAGINARYPART`
E*/
typedef enum {
  REDUCTION_SUM_REALPART       = 10,
  REDUCTION_MEAN_REALPART      = 11,
  REDUCTION_SUM_IMAGINARYPART  = 12,
  REDUCTION_MEAN_IMAGINARYPART = 13
} ReductionType;

/*E
  VecSignMode - How `VecPointwiseSign()` should handle zero value

  Values:
+ `VEC_SIGN_ZERO_TO_ZERO`        - `-0.0` and `0.0` map to `0.0`
. `VEC_SIGN_ZERO_TO_SIGNED_ZERO` - `-0.0` maps to `-0.0` and `0.0` maps to `0.0`
- `VEC_SIGN_ZERO_TO_SIGNED_UNIT` - `-0.0` maps to `-1.0` and `0.0` maps to `1.0`

  Level: advanced

.seealso: [](ch_vectors), `VecPointwiseSign()`
E*/
typedef enum {
  VEC_SIGN_ZERO_TO_ZERO,
  VEC_SIGN_ZERO_TO_SIGNED_ZERO,
  VEC_SIGN_ZERO_TO_SIGNED_UNIT,
} VecSignMode;

/*E
   VecOption - Options that may be set for a vector regarding entries passed to `VecSetValues()` and related routines

   Values:
+   `VEC_IGNORE_OFF_PROC_ENTRIES` - causes `VecSetValues()` to ignore entries destined to be stored on a separate processor.
                                    This can be used to eliminate the global reduction in `VecAssemblyBegin()` if you know that
                                    you have only used `VecSetValues()` to set local elements
.   `VEC_IGNORE_NEGATIVE_INDICES` - means you can pass negative indices in `ix` in calls to `VecSetValues()` or `VecGetValues()`.
                                    These rows are simply ignored.
-   `VEC_SUBSET_OFF_PROC_ENTRIES` - causes `VecAssemblyBegin()` to assume that the off-process entries will always be a subset
                                    (possibly equal) of the off-process entries set on the first assembly which had a true
                                    `VEC_SUBSET_OFF_PROC_ENTRIES` and the vector has not changed this flag afterwards. If this
                                    assembly is not such first assembly, then this assembly can reuse the communication pattern
                                    setup in that first assembly, thus avoiding a global reduction. Subsequent assemblies setting
                                    off-process values should use the same `InsertMode` as the first assembly.

   Level: beginner

.seealso: [](ch_matrices), `Vec`, `MatSetOption()`, `VecSetOption()`, `VecSetValues()`, `VecAssemblyBegin()`
E*/
typedef enum {
  VEC_IGNORE_OFF_PROC_ENTRIES,
  VEC_IGNORE_NEGATIVE_INDICES,
  VEC_SUBSET_OFF_PROC_ENTRIES
} VecOption;

/*E
  VecOperation - Enumeration of overide-able methods in the `Vec` implementation function-table.

  Values:
+ `VECOP_DUPLICATE`  - `VecDuplicate()`
. `VECOP_SET`        - `VecSet()`
. `VECOP_VIEW`       - `VecView()`
. `VECOP_LOAD`       - `VecLoad()`
. `VECOP_VIEWNATIVE` - `VecViewNative()`
- `VECOP_LOADNATIVE` - `VecLoadNative()`

  Level: advanced

  Notes:
  Some operations may serve as the implementation for other routines not listed above. For
  example `VECOP_SET` can be used to simultaneously overriding the implementation used in
  `VecSet()`, `VecSetInf()`, and `VecZeroEntries()`.

  Entries to `VecOperation` are added as needed so if you do not see the operation listed which
  you'd like to replace, please send mail to `petsc-maint@mcs.anl.gov`!

.seealso: [](ch_vectors), `Vec`, `VecSetOperation()`
E*/
typedef enum {
  VECOP_DUPLICATE  = 0,
  VECOP_SET        = 10,
  VECOP_VIEW       = 33,
  VECOP_LOAD       = 41,
  VECOP_VIEWNATIVE = 68,
  VECOP_LOADNATIVE = 69
} VecOperation;

/*S
  Vecs - Collection of `Vec`s where the storage for the vectors is held in a single contiguous block of memory

  Level: advanced

  Notes:
  Temporary construct for handling multiple right-hand side solves.

  This is faked by storing a single `Vec` whose array is sized to hold `n` vectors back to back.

.seealso: `Vec`, `VecsCreateSeq()`, `VecsCreateSeqWithArray()`, `VecsDuplicate()`, `VecsDestroy()`
S*/
typedef struct _n_Vecs *Vecs;

#if PetscDefined(HAVE_VIENNACL)
/*S
  PetscViennaCLIndices - Opaque handle to an index buffer used by PETSc's ViennaCL `VECVIENNACL` vector backend to perform partial scatters between CPU and GPU memory

  Level: developer

.seealso: `Vec`, `VECVIENNACL`, `VecCreateSeqViennaCL()`
S*/
typedef struct _p_PetscViennaCLIndices *PetscViennaCLIndices;
#endif

/*S
  VecTagger - Object used to manage the tagging of a subset of indices based on the values of a vector.  The
              motivating application is the selection of cells for refinement or coarsening based on vector containing
              the values in an error indicator metric.

  Values:
+  `VECTAGGERABSOLUTE` - "absolute" values are in a interval (box for complex values) of explicitly defined values
.  `VECTAGGERRELATIVE` - "relative" values are in a interval (box for complex values) of values relative to the set of all values in the vector
.  `VECTAGGERCDF`      - "cdf" values are in a relative range of the *cumulative distribution* of values in the vector
.  `VECTAGGEROR`       - "or" values are in the union of other tags
-  `VECTAGGERAND`      - "and" values are in the intersection of other tags

  Level: advanced

  Developer Note:
  Why not use a `DMLabel` or similar object

.seealso: [](ch_vectors), `Vec`, `VecTaggerType`, `VecTaggerCreate()`
S*/
typedef struct _p_VecTagger *VecTagger;

/*J
  VecTaggerType - String with the name of a `VecTagger` type

  Level: advanced

.seealso: [](ch_vectors), `Vec`, `VecTagger`, `VecTaggerCreate()`
J*/
typedef const char *VecTaggerType;
#define VECTAGGERABSOLUTE "absolute"
#define VECTAGGERRELATIVE "relative"
#define VECTAGGERCDF      "cdf"
#define VECTAGGEROR       "or"
#define VECTAGGERAND      "and"

/*S
   VecTaggerBox - A interval (box for complex numbers) range used to tag values.  For real scalars, this is just a closed interval; for complex scalars,
   the box is the closed region in the complex plane such that real(min) <= real(z) <= real(max) and imag(min) <= imag(z) <= imag(max).  `INF` is an acceptable endpoint.

   Level: beginner

.seealso: [](ch_vectors), `Vec`, `VecTagger`, `VecTaggerType`, `VecTaggerCreate()`, `VecTaggerComputeIntervals()`
S*/
typedef struct {
  PetscScalar min;
  PetscScalar max;
} VecTaggerBox;

/*E
  VecTaggerCDFMethod - Determines what method is used to compute absolute values from cumulative distribution values (e.g., what value is the preimage of .95 in the cdf).

   Values:
+  `VECTAGGER_CDF_GATHER`    - gather the data to MPI rank 0, perform the computation and broadcast the result
-  `VECTAGGER_CDF_ITERATIVE` - compute the results on all ranks iteratively using `MPI_Allreduce()`

  Level: advanced

  Note:
  Relevant only in parallel: in serial it is directly computed.

  Developer Note:
  In PETSc enums of this type are referred to with the term type, not method.

.seealso: [](ch_vectors), `Vec`, `VecTagger`, `VecTaggerType`, `VecTaggerCreate()`, `VecTaggerCDFSetMethod()`
E*/
typedef enum {
  VECTAGGER_CDF_GATHER,
  VECTAGGER_CDF_ITERATIVE,
  VECTAGGER_CDF_NUM_METHODS
} VecTaggerCDFMethod;
PETSC_EXTERN const char *const VecTaggerCDFMethods[];
