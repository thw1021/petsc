#pragma once

#include <petscvectypes.h>

/* SUBMANSEC = Mat */

/*S
   Mat - Abstract PETSc matrix object used to manage all linear operators in PETSc, even those without
         an explicit sparse representation (such as matrix-free operators). Also used to hold representations of graphs
         for graph operations such as coloring, `MatColoringCreate()`

   Level: beginner

   Note:
   See [](doc_matrix), [](ch_matrices) and `MatType` for available matrix types

.seealso: [](doc_matrix), [](ch_matrices), `MatCreate()`, `MatType`, `MatSetType()`, `MatDestroy()`
S*/
typedef struct _p_Mat *Mat;

/*J
   MatType - String with the name of a PETSc matrix type. These are all the matrix formats that PETSc provides.

   Level: beginner

   Notes:
   [](doc_matrix) for a table of available matrix types

   Use `MatSetType()` or the options database keys `-mat_type` or `-dm_mat_type` to set the matrix format to use for a given `Mat`

.seealso: [](doc_matrix), [](ch_matrices), `MatSetType()`, `Mat`, `MatSolverType`, `MatRegister()`
J*/
typedef const char *MatType;
#define MATSAME                      "same"
#define MATMAIJ                      "maij"
#define MATSEQMAIJ                   "seqmaij"
#define MATMPIMAIJ                   "mpimaij"
#define MATKAIJ                      "kaij"
#define MATSEQKAIJ                   "seqkaij"
#define MATMPIKAIJ                   "mpikaij"
#define MATIS                        "is"
#define MATAIJ                       "aij"
#define MATSEQAIJ                    "seqaij"
#define MATMPIAIJ                    "mpiaij"
#define MATAIJCRL                    "aijcrl"
#define MATSEQAIJCRL                 "seqaijcrl"
#define MATMPIAIJCRL                 "mpiaijcrl"
#define MATAIJCUSPARSE               "aijcusparse"
#define MATSEQAIJCUSPARSE            "seqaijcusparse"
#define MATMPIAIJCUSPARSE            "mpiaijcusparse"
#define MATAIJHIPSPARSE              "aijhipsparse"
#define MATSEQAIJHIPSPARSE           "seqaijhipsparse"
#define MATMPIAIJHIPSPARSE           "mpiaijhipsparse"
#define MATAIJKOKKOS                 "aijkokkos"
#define MATSEQAIJKOKKOS              "seqaijkokkos"
#define MATMPIAIJKOKKOS              "mpiaijkokkos"
#define MATAIJVIENNACL               "aijviennacl"
#define MATSEQAIJVIENNACL            "seqaijviennacl"
#define MATMPIAIJVIENNACL            "mpiaijviennacl"
#define MATAIJPERM                   "aijperm"
#define MATSEQAIJPERM                "seqaijperm"
#define MATMPIAIJPERM                "mpiaijperm"
#define MATAIJSELL                   "aijsell"
#define MATSEQAIJSELL                "seqaijsell"
#define MATMPIAIJSELL                "mpiaijsell"
#define MATAIJMKL                    "aijmkl"
#define MATSEQAIJMKL                 "seqaijmkl"
#define MATMPIAIJMKL                 "mpiaijmkl"
#define MATBAIJMKL                   "baijmkl"
#define MATSEQBAIJMKL                "seqbaijmkl"
#define MATMPIBAIJMKL                "mpibaijmkl"
#define MATSHELL                     "shell"
#define MATCENTERING                 "centering"
#define MATDENSE                     "dense"
#define MATDENSECUDA                 "densecuda"
#define MATDENSEHIP                  "densehip"
#define MATSEQDENSE                  "seqdense"
#define MATSEQDENSECUDA              "seqdensecuda"
#define MATSEQDENSEHIP               "seqdensehip"
#define MATMPIDENSE                  "mpidense"
#define MATMPIDENSECUDA              "mpidensecuda"
#define MATMPIDENSEHIP               "mpidensehip"
#define MATELEMENTAL                 "elemental"
#define MATSCALAPACK                 "scalapack"
#define MATBAIJ                      "baij"
#define MATSEQBAIJ                   "seqbaij"
#define MATMPIBAIJ                   "mpibaij"
#define MATBAIJLIBXSMM               "baijlibxsmm"
#define MATSEQBAIJLIBXSMM            "seqbaijlibxsmm"
#define MATMPIBAIJLIBXSMM            "mpibaijlibxsmm"
#define MATMPIADJ                    "mpiadj"
#define MATSBAIJ                     "sbaij"
#define MATSEQSBAIJ                  "seqsbaij"
#define MATMPISBAIJ                  "mpisbaij"
#define MATMFFD                      "mffd"
#define MATNORMAL                    "normal"
#define MATNORMALHERMITIAN           "normalh"
#define MATLRC                       "lrc"
#define MATSCATTER                   "scatter"
#define MATBLOCKMAT                  "blockmat"
#define MATCOMPOSITE                 "composite"
#define MATFFT                       "fft"
#define MATFFTW                      "fftw"
#define MATSEQCUFFT                  "seqcufft"
#define MATSEQHIPFFT                 "seqhipfft"
#define MATTRANSPOSEMAT              PETSC_DEPRECATED_MACRO(3, 18, 0, "MATTRANSPOSEVIRTUAL", ) "transpose"
#define MATTRANSPOSEVIRTUAL          "transpose"
#define MATHERMITIANTRANSPOSEVIRTUAL "hermitiantranspose"
#define MATSCHURCOMPLEMENT           "schurcomplement"
#define MATPYTHON                    "python"
#define MATHYPRE                     "hypre"
#define MATHYPRESTRUCT               "hyprestruct"
#define MATHYPRESSTRUCT              "hypresstruct"
#define MATSUBMATRIX                 "submatrix"
#define MATLOCALREF                  "localref"
#define MATNEST                      "nest"
#define MATPREALLOCATOR              "preallocator"
#define MATSELL                      "sell"
#define MATSEQSELL                   "seqsell"
#define MATMPISELL                   "mpisell"
#define MATSELLCUDA                  "sellcuda"
#define MATSEQSELLCUDA               "seqsellcuda"
#define MATMPISELLCUDA               "mpisellcuda"
#define MATSELLHIP                   "sellhip"
#define MATSEQSELLHIP                "seqsellhip"
#define MATMPISELLHIP                "mpisellhip"
#define MATDUMMY                     "dummy"
#define MATLMVM                      "lmvm"
#define MATLMVMDFP                   "lmvmdfp"
#define MATLMVMDDFP                  "lmvmddfp"
#define MATLMVMBFGS                  "lmvmbfgs"
#define MATLMVMDBFGS                 "lmvmdbfgs"
#define MATLMVMDQN                   "lmvmdqn"
#define MATLMVMSR1                   "lmvmsr1"
#define MATLMVMBROYDEN               "lmvmbroyden"
#define MATLMVMBADBROYDEN            "lmvmbadbroyden"
#define MATLMVMSYMBROYDEN            "lmvmsymbroyden"
#define MATLMVMSYMBADBROYDEN         "lmvmsymbadbroyden"
#define MATLMVMDIAGBROYDEN           "lmvmdiagbroyden"
#define MATCONSTANTDIAGONAL          "constantdiagonal"
#define MATDIAGONAL                  "diagonal"
#define MATHTOOL                     "htool"
#define MATH2OPUS                    "h2opus"

/*J
   MatSolverType - String with the name of a PETSc factorization-based matrix solver type.

   For example: "petsc" indicates what PETSc provides, "superlu_dist" the parallel SuperLU_DIST package etc

   Level: beginner

   Note:
   `MATSOLVERUMFPACK`, `MATSOLVERCHOLMOD`, `MATSOLVERKLU`, `MATSOLVERSPQR` form the SuiteSparse package; you can use `--download-suitesparse` as
   a ./configure option to install them

.seealso: [](sec_matfactor), [](ch_matrices), `MatGetFactor()`, `PCFactorSetMatSolverType()`, `PCFactorGetMatSolverType()`
J*/
typedef const char *MatSolverType;
#define MATSOLVERSUPERLU      "superlu"
#define MATSOLVERSUPERLU_DIST "superlu_dist"
#define MATSOLVERSTRUMPACK    "strumpack"
#define MATSOLVERUMFPACK      "umfpack"
#define MATSOLVERCHOLMOD      "cholmod"
#define MATSOLVERKLU          "klu"
#define MATSOLVERELEMENTAL    "elemental"
#define MATSOLVERSCALAPACK    "scalapack"
#define MATSOLVERESSL         "essl"
#define MATSOLVERLUSOL        "lusol"
#define MATSOLVERMUMPS        "mumps"
#define MATSOLVERMKL_PARDISO  "mkl_pardiso"
#define MATSOLVERMKL_CPARDISO "mkl_cpardiso"
#define MATSOLVERPASTIX       "pastix"
#define MATSOLVERMATLAB       "matlab"
#define MATSOLVERPETSC        "petsc"
#define MATSOLVERBAS          "bas"
#define MATSOLVERCUSPARSE     "cusparse"
#define MATSOLVERCUDA         "cuda"
#define MATSOLVERHIPSPARSE    "hipsparse"
#define MATSOLVERHIP          "hip"
#define MATSOLVERKOKKOS       "kokkos"
#define MATSOLVERSPQR         "spqr"
#define MATSOLVERHTOOL        "htool"

/*E
    MatFactorType - indicates what type of factorization is requested

    Values:
+  `MAT_FACTOR_LU`       - LU factorization
.  `MAT_FACTOR_CHOLESKY` - Cholesky factorization
.  `MAT_FACTOR_ILU`      - ILU factorization
.  `MAT_FACTOR_ICC`      - incomplete Cholesky factorization
.  `MAT_FACTOR_ILUDT`    - ILU factorization with drop tolerance
-  `MAT_FACTOR_QR`       - QR factorization

    Level: beginner

.seealso: [](ch_matrices), `MatSolverType`, `MatGetFactor()`, `MatGetFactorAvailable()`, `MatSolverTypeRegister()`
E*/
typedef enum {
  MAT_FACTOR_NONE,
  MAT_FACTOR_LU,
  MAT_FACTOR_CHOLESKY,
  MAT_FACTOR_ILU,
  MAT_FACTOR_ICC,
  MAT_FACTOR_ILUDT,
  MAT_FACTOR_QR,
  MAT_FACTOR_NUM_TYPES
} MatFactorType;
PETSC_EXTERN const char *const MatFactorTypes[];

/*S
  MatSolverFn - Function type for the factor-creation callback registered with `MatSolverTypeRegister()`, used by `MatGetFactor()` to allocate a factored matrix of a particular `MatSolverType` and `MatFactorType`

  Synopsis:
  #include <petscmat.h>
  PetscErrorCode MatSolverFn(Mat A, MatFactorType ftype, Mat *F)

  Calling Sequence:
+ A     - the matrix to be factored
. ftype - the kind of factorization requested (e.g. `MAT_FACTOR_LU`, `MAT_FACTOR_CHOLESKY`)
- F     - on output, the newly created factor `Mat` of the appropriate `MatType` for the solver

  Level: developer

.seealso: `Mat`, `MatGetFactor()`, `MatSolverType`, `MatFactorType`, `MatSolverTypeRegister()`, `MatSolverTypeGet()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatSolverFn(Mat, MatFactorType, Mat *);
PETSC_EXTERN_TYPEDEF typedef MatSolverFn   *MatSolverFunction;

/*E
    MatProductType - indicates what type of matrix product to compute

    Values:
+  `MATPRODUCT_AB`   - product of two matrices
.  `MATPRODUCT_AtB`  - product of the transpose of a given matrix with a matrix
.  `MATPRODUCT_ABt`  - product of a matrix with the transpose of another given matrix
.  `MATPRODUCT_PtAP` - the triple product of the transpose of a matrix with another matrix and itself
.  `MATPRODUCT_RARt` - the triple product of a matrix, another matrix and the transpose of the first matrix
-  `MATPRODUCT_ABC`  - the product of three matrices

    Level: beginner

.seealso: [](sec_matmatproduct), [](ch_matrices), `MatProductSetType()`
E*/
typedef enum {
  MATPRODUCT_UNSPECIFIED,
  MATPRODUCT_AB,
  MATPRODUCT_AtB,
  MATPRODUCT_ABt,
  MATPRODUCT_PtAP,
  MATPRODUCT_RARt,
  MATPRODUCT_ABC
} MatProductType;
PETSC_EXTERN const char *const MatProductTypes[];

/*J
    MatProductAlgorithm - String with the name of an algorithm for a PETSc matrix product implementation

   Level: beginner

.seealso: [](sec_matmatproduct), [](ch_matrices), `MatSetType()`, `Mat`, `MatProductSetAlgorithm()`, `MatProductType`
J*/
typedef const char *MatProductAlgorithm;
#define MATPRODUCTALGORITHMDEFAULT         "default"
#define MATPRODUCTALGORITHMSORTED          "sorted"
#define MATPRODUCTALGORITHMSCALABLE        "scalable"
#define MATPRODUCTALGORITHMSCALABLEFAST    "scalable_fast"
#define MATPRODUCTALGORITHMHEAP            "heap"
#define MATPRODUCTALGORITHMBHEAP           "btheap"
#define MATPRODUCTALGORITHMLLCONDENSED     "llcondensed"
#define MATPRODUCTALGORITHMROWMERGE        "rowmerge"
#define MATPRODUCTALGORITHMOUTERPRODUCT    "outerproduct"
#define MATPRODUCTALGORITHMATB             "at*b"
#define MATPRODUCTALGORITHMRAP             "rap"
#define MATPRODUCTALGORITHMNONSCALABLE     "nonscalable"
#define MATPRODUCTALGORITHMSEQMPI          "seqmpi"
#define MATPRODUCTALGORITHMBACKEND         "backend"
#define MATPRODUCTALGORITHMOVERLAPPING     "overlapping"
#define MATPRODUCTALGORITHMMERGED          "merged"
#define MATPRODUCTALGORITHMALLATONCE       "allatonce"
#define MATPRODUCTALGORITHMALLATONCEMERGED "allatonce_merged"
#define MATPRODUCTALGORITHMALLGATHERV      "allgatherv"
#define MATPRODUCTALGORITHMCYCLIC          "cyclic"
#define MATPRODUCTALGORITHMHYPRE           "hypre"

/*E
   MatReuse - Indicates if matrices obtained from a previous call to `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatConvert()` or several other functions
   are to be reused to store the new matrix values.

   Values:
+  `MAT_INITIAL_MATRIX` - create a new matrix
.  `MAT_REUSE_MATRIX`   - reuse the matrix created with a previous call that used `MAT_INITIAL_MATRIX`
.  `MAT_INPLACE_MATRIX` - replace the first input matrix with the new matrix (not applicable to all functions)
-  `MAT_IGNORE_MATRIX`  - do not create a new matrix or reuse a given matrix, just ignore that matrix argument (not applicable to all functions)

    Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatCreateSubMatrices()`, `MatCreateSubMatrix()`, `MatDestroyMatrices()`, `MatConvert()`
E*/
typedef enum {
  MAT_INITIAL_MATRIX,
  MAT_REUSE_MATRIX,
  MAT_IGNORE_MATRIX,
  MAT_INPLACE_MATRIX
} MatReuse;

/*E
    MatCreateSubMatrixOption - Indicates if matrices obtained from a call to `MatCreateSubMatrices()`
    include the matrix values. Currently it is only used by `MatGetSeqNonzeroStructure()`.

    Values:
+  `MAT_DO_NOT_GET_VALUES` - do not copy the matrix values
-  `MAT_GET_VALUES`        - copy the matrix values

    Level: developer

    Developer Note:
    Why is not just a boolean used for this information?

.seealso: [](ch_matrices), `Mat`, `MatDuplicateOption`, `PetscCopyMode`, `MatGetSeqNonzeroStructure()`
E*/
typedef enum {
  MAT_DO_NOT_GET_VALUES,
  MAT_GET_VALUES
} MatCreateSubMatrixOption;

/*E
   MatStructure - Indicates if two matrices have the same nonzero structure

   Values:
+  `SAME_NONZERO_PATTERN`      - the two matrices have identical nonzero patterns
.  `DIFFERENT_NONZERO_PATTERN` - the two matrices may have different nonzero patterns
.  `SUBSET_NONZERO_PATTERN`    - the nonzero pattern of the second matrix is a subset of the nonzero pattern of the first matrix
-  `UNKNOWN_NONZERO_PATTERN`   - there is no known relationship between the nonzero patterns. In this case the implementations
                                 may try to detect a relationship to optimize the operation

   Level: beginner

   Note:
   Certain matrix operations (such as `MatAXPY()`) can run much faster if the sparsity pattern of the matrices are the same. But actually determining if
   the patterns are the same may be costly. This provides a way for users who know something about the sparsity patterns to provide this information
   to certain PETSc routines.

.seealso: [](ch_matrices), `Mat`, `MatCopy()`, `MatAXPY()`, `MatAYPX()`
E*/
typedef enum {
  DIFFERENT_NONZERO_PATTERN,
  SUBSET_NONZERO_PATTERN,
  SAME_NONZERO_PATTERN,
  UNKNOWN_NONZERO_PATTERN
} MatStructure;
PETSC_EXTERN const char *const MatStructures[];

/*E
   MatCompositeMergeType - Selects the order in which the matrices held by a `MATCOMPOSITE` are combined when `MatCompositeMerge()` is called

   Values:
+   `MAT_COMPOSITE_MERGE_RIGHT` - merge into a single matrix starting from the rightmost matrix (multiplicative compositions use this to preserve the natural left-to-right application order)
-   `MAT_COMPOSITE_MERGE_LEFT`  - merge into a single matrix starting from the leftmost matrix

   Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATCOMPOSITE`, `MatCompositeMerge()`, `MatCompositeSetMergeType()`, `MatCompositeType`
E*/
typedef enum {
  MAT_COMPOSITE_MERGE_RIGHT,
  MAT_COMPOSITE_MERGE_LEFT
} MatCompositeMergeType;

/*E
    MatCompositeType - indicates what type of `MATCOMPOSITE` is used

    Values:
+  `MAT_COMPOSITE_ADDITIVE`       - sum of matrices (default)
-  `MAT_COMPOSITE_MULTIPLICATIVE` - product of matrices

    Level: beginner

.seealso: [](ch_matrices), `MATCOMPOSITE`, `MatCompositeSetType()`, `MatCompositeGetType()`
E*/
typedef enum {
  MAT_COMPOSITE_ADDITIVE,
  MAT_COMPOSITE_MULTIPLICATIVE
} MatCompositeType;

/*S
   MatStencil - Data structure (C struct) for storing information about rows and
   columns of a matrix as indexed on an associated grid. These are arguments to `MatSetStencil()` and `MatSetBlockStencil()`

   Level: beginner

   Notes:
   The i,j, and k represent the logical coordinates over the entire grid (for 2 and 1 dimensional problems the k and j entries are ignored).
   The c represents the degrees of freedom at each grid point (the dof argument to `DMDASetDOF()`). If dof is 1 then this entry is ignored.

   For stencil access to vectors see `DMDAVecGetArray()`

   For staggered grids, see `DMStagStencil`

.seealso: [](ch_matrices), `Mat`, `MatSetValuesStencil()`, `MatSetStencil()`, `MatSetValuesBlockedStencil()`, `DMDAVecGetArray()`,
          `DMStagStencil`
S*/
typedef struct {
  PetscInt k, j, i, c;
} MatStencil;

/*E
    MatAssemblyType - Indicates if the process of setting values into the matrix is complete, and the matrix is ready for use

    Values:
+   `MAT_FLUSH_ASSEMBLY` - you will continue to put values into the matrix
-   `MAT_FINAL_ASSEMBLY` - you wish to use the matrix with the values currently inserted

    Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatSetValues`, `MatAssemblyBegin()`, `MatAssemblyEnd()`
E*/
typedef enum {
  MAT_FLUSH_ASSEMBLY = 1,
  MAT_FINAL_ASSEMBLY = 0
} MatAssemblyType;

/*E
   MatOption - Options that may be set for a matrix that indicate properties of the matrix or affect its behavior or storage

   Level: beginner

   Note:
   See `MatSetOption()` for the use of the options

   Developer Note:
   Entries that are negative need not be called collectively by all processes.

.seealso: [](ch_matrices), `Mat`, `MatSetOption()`, `VecOption`
E*/
typedef enum {
  MAT_OPTION_MIN                  = -3,
  MAT_UNUSED_NONZERO_LOCATION_ERR = -2,
  MAT_ROW_ORIENTED                = -1,
  MAT_SYMMETRIC                   = 1,
  MAT_STRUCTURALLY_SYMMETRIC      = 2,
  MAT_FORCE_DIAGONAL_ENTRIES      = 3,
  MAT_IGNORE_OFF_PROC_ENTRIES     = 4,
  MAT_USE_HASH_TABLE              = 5,
  MAT_KEEP_NONZERO_PATTERN        = 6,
  MAT_IGNORE_ZERO_ENTRIES         = 7,
  MAT_USE_INODES                  = 8,
  MAT_HERMITIAN                   = 9,
  MAT_SYMMETRY_ETERNAL            = 10,
  MAT_NEW_NONZERO_LOCATION_ERR    = 11,
  MAT_IGNORE_LOWER_TRIANGULAR     = 12,
  MAT_ERROR_LOWER_TRIANGULAR      = 13,
  MAT_GETROW_UPPERTRIANGULAR      = 14,
  MAT_SPD                         = 15,
  MAT_NO_OFF_PROC_ZERO_ROWS       = 16,
  MAT_NO_OFF_PROC_ENTRIES         = 17,
  MAT_NEW_NONZERO_LOCATIONS       = 18,
  MAT_NEW_NONZERO_ALLOCATION_ERR  = 19,
  MAT_SUBSET_OFF_PROC_ENTRIES     = 20,
  MAT_SUBMAT_SINGLEIS             = 21,
  MAT_STRUCTURE_ONLY              = 22,
  MAT_SORTED_FULL                 = 23,
  MAT_FORM_EXPLICIT_TRANSPOSE     = 24,
  MAT_STRUCTURAL_SYMMETRY_ETERNAL = 25,
  MAT_SPD_ETERNAL                 = 26,
  MAT_OPTION_MAX                  = 27
} MatOption;
PETSC_EXTERN const char *const *MatOptions;

/*E
  MatDuplicateOption - Indicates if a duplicated sparse matrix should have
  its numerical values copied over or just its nonzero structure.

  Values:
+ `MAT_DO_NOT_COPY_VALUES`    - Create a matrix using the same nonzero pattern as the original matrix,
                                with zeros for the numerical values
. `MAT_COPY_VALUES`           - Create a matrix with the same nonzero pattern as the original matrix
                                and with the same numerical values.
- `MAT_SHARE_NONZERO_PATTERN` - Create a matrix that shares the nonzero structure with the previous matrix
                                and does not copy it, using zeros for the numerical values. The parent and
                                child matrices will share their index (i and j) arrays, and you cannot
                                insert new nonzero entries into either matrix

  Level: beginner

  Note:
  Many matrix types (including `MATSEQAIJ`) do not support the `MAT_SHARE_NONZERO_PATTERN` optimization; in
  this case the behavior is as if `MAT_DO_NOT_COPY_VALUES` has been specified.

.seealso: [](ch_matrices), `Mat`, `MatDuplicate()`
E*/
typedef enum {
  MAT_DO_NOT_COPY_VALUES,
  MAT_COPY_VALUES,
  MAT_SHARE_NONZERO_PATTERN
} MatDuplicateOption;

/*S
   MatInfo - Context of matrix information, used with `MatGetInfo()`

   Level: intermediate

   Fortran Note:
   `MatInfo` is a derived type, use e.g. `matinfo%nz_allocated` to access its components.

.seealso: [](ch_matrices), `Mat`, `MatGetInfo()`, `MatInfoType`
S*/
typedef struct {
  PetscLogDouble block_size;                          /* block size */
  PetscLogDouble nz_allocated, nz_used, nz_unneeded;  /* number of nonzeros */
  PetscLogDouble memory;                              /* memory allocated */
  PetscLogDouble assemblies;                          /* number of matrix assemblies called */
  PetscLogDouble mallocs;                             /* number of mallocs during MatSetValues() */
  PetscLogDouble fill_ratio_given, fill_ratio_needed; /* fill ratio for LU/ILU */
  PetscLogDouble factor_mallocs;                      /* number of mallocs during factorization */
} MatInfo;

/*E
    MatInfoType - Indicates if you want information about the local part of the matrix,
    the entire parallel matrix or the maximum over all the local parts.

    Values:
+   `MAT_LOCAL`      - values for each MPI process part of the matrix
.   `MAT_GLOBAL_MAX` - maximum of each value over all MPI processes
-   `MAT_GLOBAL_SUM` - sum of each value over all MPI processes

    Level: beginner

.seealso: [](ch_matrices), `MatGetInfo()`, `MatInfo`
E*/
typedef enum {
  MAT_LOCAL      = 1,
  MAT_GLOBAL_MAX = 2,
  MAT_GLOBAL_SUM = 3
} MatInfoType;

/*J
   MatOrderingType - String with the name of a PETSc matrix ordering. These orderings are most commonly used
   to reduce fill in sparse factorizations.

   Level: beginner

   Notes:
   If `MATORDERINGEXTERNAL` is used then PETSc does not compute an ordering and instead the external factorization solver package called utilizes one
   of its own.

   There is no `MatOrdering` object, the ordering is obtained directly from the matrix with `MatGetOrdering()`

   Developer Note:
   This API should be converted to an API similar to those for `MatColoring` and `MatPartitioning`

.seealso: [](ch_matrices), [](sec_graph), `MatGetFactor()`, `MatGetOrdering()`, `MatColoringType`, `MatPartitioningType`, `MatCoarsenType`, `PCFactorSetOrderingType()`
J*/
typedef const char *MatOrderingType;
#define MATORDERINGNATURAL       "natural"
#define MATORDERINGND            "nd"
#define MATORDERING1WD           "1wd"
#define MATORDERINGRCM           "rcm"
#define MATORDERINGQMD           "qmd"
#define MATORDERINGROWLENGTH     "rowlength"
#define MATORDERINGWBM           "wbm"
#define MATORDERINGSPECTRAL      "spectral"
#define MATORDERINGAMD           "amd"           /* only works if UMFPACK is installed with PETSc */
#define MATORDERINGMETISND       "metisnd"       /* only works if METIS is installed with PETSc */
#define MATORDERINGNATURAL_OR_ND "natural_or_nd" /* special coase used for Cholesky and ICC, allows ND when AIJ matrix is used but Natural when SBAIJ is used */
#define MATORDERINGEXTERNAL      "external"      /* uses an ordering type internal to the factorization package */

/*S
   MatFactorShiftType - Type of numeric shift used for factorizations

   Values:
+  `MAT_SHIFT_NONE`              - do not shift the matrix diagonal entries
.  `MAT_SHIFT_NONZERO`           - shift the entries to be non-zero
.  `MAT_SHIFT_POSITIVE_DEFINITE` - shift the entries to force the factorization to be positive definite
-  `MAT_SHIFT_INBLOCKS`          - only shift the factors inside the small dense diagonal blocks of the matrix, for example with `MATBAIJ`

   Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `PCFactorSetShiftType()`
S*/
typedef enum {
  MAT_SHIFT_NONE,
  MAT_SHIFT_NONZERO,
  MAT_SHIFT_POSITIVE_DEFINITE,
  MAT_SHIFT_INBLOCKS
} MatFactorShiftType;
PETSC_EXTERN const char *const MatFactorShiftTypes[];
PETSC_EXTERN const char *const MatFactorShiftTypesDetail[];

/*S
    MatFactorError - indicates what type of error was generated in a matrix factorization

    Values:
+   `MAT_FACTOR_NOERROR`           - there was no error during the factorization
.   `MAT_FACTOR_STRUCT_ZEROPIVOT`  - there was a missing entry in a diagonal location of the matrix
.   `MAT_FACTOR_NUMERIC_ZEROPIVOT` - there was a (near) zero pivot during the factorization
.   `MAT_FACTOR_OUTMEMORY`         - the factorization has run out of memory
-   `MAT_FACTOR_OTHER`             - some other error has occurred.

    Level: intermediate

    Note:
    When a factorization is done in a preconditioner `PC` the error may be propagated up to a `PCFailedReason` or a `KSPConvergedReason`

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatFactorGetError()`, `MatFactorGetErrorZeroPivot()`, `MatFactorClearError()`,
          `PCFailedReason`, `PCGetFailedReason()`, `KSPConvergedReason`
S*/
typedef enum {
  MAT_FACTOR_NOERROR,
  MAT_FACTOR_STRUCT_ZEROPIVOT,
  MAT_FACTOR_NUMERIC_ZEROPIVOT,
  MAT_FACTOR_OUTMEMORY,
  MAT_FACTOR_OTHER
} MatFactorError;

/*S
   MatFactorInfo - Data passed into the matrix factorization routines, and information about the resulting factorization

   Level: developer

   Note:
   You can use `MatFactorInfoInitialize()` to set default values.

   Fortran Note:
   `MatFactorInfo` is a derived type, use e.g. `matfactorinfo%dt` to access its components.

.seealso: [](ch_matrices), `Mat`, `MatInfo`, `MatGetFactor()`, `MatLUFactorSymbolic()`, `MatILUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`,
          `MatICCFactorSymbolic()`, `MatICCFactor()`, `MatFactorInfoInitialize()`
S*/
typedef struct {
  PetscReal diagonal_fill; /* force diagonal to fill in if initially not filled */
  PetscReal usedt;
  PetscReal dt;            /* drop tolerance */
  PetscReal dtcol;         /* tolerance for pivoting */
  PetscReal dtcount;       /* maximum nonzeros to be allowed per row */
  PetscReal fill;          /* expected fill, nonzeros in factored matrix/nonzeros in original matrix */
  PetscReal levels;        /* ICC/ILU(levels) */
  PetscReal pivotinblocks; /* BAIJ and SBAIJ matrices pivot in factorization on blocks, default 1.0 factorization may be faster if do not pivot */
  PetscReal zeropivot;     /* pivot is called zero if less than this */
  PetscReal shifttype;     /* type of shift added to matrix factor to prevent zero pivots */
  PetscReal shiftamount;   /* how large the shift is */
  PetscBool factoronhost;  /* do factorization on host instead of device (for device matrix types) */
  PetscBool solveonhost;   /* do mat solve on host with the factor (for device matrix types) */
} MatFactorInfo;

/*E
   MatFactorSchurStatus - Records the current state of the dense Schur complement that is maintained by a factored matrix when a user has requested its formation with `MatFactorSetSchurIS()`

   Values:
+   `MAT_FACTOR_SCHUR_UNFACTORED` - the Schur complement has been assembled but has not yet been factored or inverted
.   `MAT_FACTOR_SCHUR_FACTORED`   - the Schur complement has been factored (for example via dense LU) and can be used to solve via `MatFactorSolveSchurComplement()`
-   `MAT_FACTOR_SCHUR_INVERTED`   - the Schur complement has been explicitly inverted in place; subsequent solves multiply by the dense inverse

   Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatFactorSetSchurIS()`, `MatFactorGetSchurComplement()`, `MatFactorRestoreSchurComplement()`,
          `MatFactorInvertSchurComplement()`, `MatFactorCreateSchurComplement()`
E*/
typedef enum {
  MAT_FACTOR_SCHUR_UNFACTORED,
  MAT_FACTOR_SCHUR_FACTORED,
  MAT_FACTOR_SCHUR_INVERTED
} MatFactorSchurStatus;

/*E
   MatSORType - What type of (S)SOR to perform

   Values:
+  `SOR_FORWARD_SWEEP`         - do a sweep from the first row of the matrix to the last
.  `SOR_BACKWARD_SWEEP`        - do a sweep from the last row to the first
.  `SOR_SYMMETRIC_SWEEP`       - do a sweep from the first row to the last and then back to the first
.  `SOR_LOCAL_FORWARD_SWEEP`   - each MPI process does its own forward sweep with no communication
.  `SOR_LOCAL_BACKWARD_SWEEP`  - each MPI process does its own backward sweep with no communication
.  `SOR_LOCAL_SYMMETRIC_SWEEP` - each MPI process does its own symmetric sweep with no communication
.  `SOR_ZERO_INITIAL_GUESS`    - indicates the initial solution is zero so the sweep can avoid unneeded computation
.  `SOR_EISENSTAT`             - apply the Eisentat application of SOR, see `PCEISENSTAT`
.  `SOR_APPLY_UPPER`           - multiply by the upper triangular portion of the matrix
-  `SOR_APPLY_LOWER`           - multiply by the lower triangular portion of the matrix

   Level: beginner

   Note:
   These may be bitwise ORd together

   Developer Note:
   Since `MatSORType` may be bitwise ORd together, so do not change the numerical values below

.seealso: [](ch_matrices), `MatSOR()`
E*/
typedef enum {
  SOR_FORWARD_SWEEP         = 1,
  SOR_BACKWARD_SWEEP        = 2,
  SOR_SYMMETRIC_SWEEP       = 3,
  SOR_LOCAL_FORWARD_SWEEP   = 4,
  SOR_LOCAL_BACKWARD_SWEEP  = 8,
  SOR_LOCAL_SYMMETRIC_SWEEP = 12,
  SOR_ZERO_INITIAL_GUESS    = 16,
  SOR_EISENSTAT             = 32,
  SOR_APPLY_UPPER           = 64,
  SOR_APPLY_LOWER           = 128
} MatSORType;

/*S
   MatColoring - Object for managing the coloring of matrices.

   Level: beginner

   Notes:
   Coloring of matrices can be computed directly from the sparse matrix nonzero structure via the `MatColoring` object or from the mesh from which the
   matrix comes from via `DMCreateColoring()`. In general using the mesh produces a more optimal coloring (fewer colors).

   Once a coloring is available `MatFDColoringCreate()` creates an object that can be used to efficiently compute Jacobians using that coloring. This
   same object can also be used to efficiently convert data created by Automatic Differentiation tools to PETSc sparse matrices.

.seealso: [](ch_matrices), [](sec_graph), `MatFDColoringCreate()`, `MatColoringWeightType`, `ISColoring`, `MatFDColoring`, `DMCreateColoring()`, `MatColoringCreate()`,
          `MatPartitioning`, `MatColoringType`, `MatPartitioningType`, `MatOrderingType`, `MatColoringSetWeightType()`,
          `MatColoringSetWeights()`, `MatCoarsenType`, `MatCoarsen`
S*/
typedef struct _p_MatColoring *MatColoring;

/*J
   MatColoringType - String with the name of a PETSc matrix coloring

   Level: beginner

.seealso: [](ch_matrices), [](sec_graph), `Mat`, `MatFDColoringCreate()`, `MatColoringSetType()`, `MatColoring`
J*/
typedef const char *MatColoringType;
#define MATCOLORINGJP      "jp"
#define MATCOLORINGPOWER   "power"
#define MATCOLORINGNATURAL "natural"
#define MATCOLORINGSL      "sl"
#define MATCOLORINGLF      "lf"
#define MATCOLORINGID      "id"
#define MATCOLORINGGREEDY  "greedy"

/*E
   MatColoringWeightType - Type of weight scheme used for the coloring algorithm

   Values:
+  `MAT_COLORING_RANDOM`  - Random weights
.  `MAT_COLORING_LEXICAL` - Lexical weighting based upon global numbering.
-  `MAT_COLORING_LF`      - Last-first weighting.

    Level: intermediate

.seealso: [](ch_matrices), `MatColoring`, `MatColoringCreate()`, `MatColoringSetWeightType()`, `MatColoringSetWeights()`
E*/
typedef enum {
  MAT_COLORING_WEIGHT_RANDOM,
  MAT_COLORING_WEIGHT_LEXICAL,
  MAT_COLORING_WEIGHT_LF,
  MAT_COLORING_WEIGHT_SL
} MatColoringWeightType;

/*S
   MatFDColoring - Object for computing a sparse Jacobian via finite differences with coloring

   Level: beginner

   Options Database Key:
.  -snes_fd_coloring - cause the Jacobian needed by `SNES` to be computed via a use of this object

   Note:
   This object is created utilizing a coloring provided by the `MatColoring` object or `DMCreateColoring()`

.seealso: [](ch_matrices), `Mat`, `MatFDColoringCreate()`, `MatFDColoringSetFunction()`, `MatColoring`, `DMCreateColoring()`
S*/
typedef struct _p_MatFDColoring *MatFDColoring;

/*S
  MatFDColoringFn - Function provided to `MatFDColoringSetFunction()` that computes the function being differenced

  Level: advanced

  Calling Sequence:
+ snes - either a `SNES` object if used within `SNES` otherwise an unused parameter
. in   - the location where the Jacobian is to be computed
. out  - the location to put the computed function value
- fctx - the function context passed into `MatFDColoringSetFunction()`

.seealso: [](ch_matrices), `Mat`, `MatCreateMFFD()`, `MatMFFDSetFunction()`, `MatMFFDiFn`, `MatMFFDiBaseFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatFDColoringFn(void *snes, Vec x, Vec y, void *fctx);

/*S
   MatTransposeColoring - Object for computing a sparse matrix product $C = A*B^T$ via coloring

   Level: developer

.seealso: [](ch_matrices), `Mat`, `MatProductType`, `MatTransposeColoringCreate()`
S*/
typedef struct _p_MatTransposeColoring *MatTransposeColoring;

/*S
   MatPartitioning - Object for managing the partitioning of a matrix or graph

   Level: beginner

   Note:
   There is also a `PetscPartitioner` object that provides the same functionality. It can utilize the `MatPartitioning` operations
   via `PetscPartitionerSetType`(p,`PETSCPARTITIONERMATPARTITIONING`)

   Developer Note:
   It is an extra maintenance and documentation cost to have two objects with the same functionality. `PetscPartitioner` should be removed

.seealso: [](ch_matrices), [](sec_graph), `Mat`, `MatPartitioningCreate()`, `MatPartitioningType`, `MatColoring`, `MatGetOrdering()`, `MatOrderingType`,
          `MatCoarsenType`
S*/
typedef struct _p_MatPartitioning *MatPartitioning;

/*J
    MatPartitioningType - String with the name of a PETSc matrix partitioning

   Level: beginner
dm
.seealso: [](ch_matrices), [](sec_graph), `Mat`, `MatPartitioningCreate()`, `MatPartitioning`, `MatPartitioningSetType()`, `MatColoringType`, `MatOrderingType`,
          `MatCoarsenType`
J*/
typedef const char *MatPartitioningType;
#define MATPARTITIONINGCURRENT  "current"
#define MATPARTITIONINGAVERAGE  "average"
#define MATPARTITIONINGSQUARE   "square"
#define MATPARTITIONINGPARMETIS "parmetis"
#define MATPARTITIONINGCHACO    "chaco"
#define MATPARTITIONINGPARTY    "party"
#define MATPARTITIONINGPTSCOTCH "ptscotch"
#define MATPARTITIONINGHIERARCH "hierarch"

/*E
   MPChacoGlobalType - Global partitioning method used by `MATPARTITIONINGCHACO` when delegating to the Chaco library

   Values:
+   `MP_CHACO_MULTILEVEL` - multilevel-Kernighan--Lin style partitioning
.   `MP_CHACO_SPECTRAL`   - spectral partitioning using Lanczos or RQI eigensolvers
.   `MP_CHACO_LINEAR`     - linear (rank-order) initial partition
.   `MP_CHACO_RANDOM`     - random initial partition
-   `MP_CHACO_SCATTERED`  - scattered (round-robin) initial partition

   Level: intermediate

.seealso: [](ch_matrices), `MatPartitioning`, `MATPARTITIONINGCHACO`, `MatPartitioningChacoSetGlobal()`, `MatPartitioningChacoGetGlobal()`,
          `MPChacoLocalType`, `MPChacoEigenType`
E*/
typedef enum {
  MP_CHACO_MULTILEVEL = 1,
  MP_CHACO_SPECTRAL   = 2,
  MP_CHACO_LINEAR     = 4,
  MP_CHACO_RANDOM     = 5,
  MP_CHACO_SCATTERED  = 6
} MPChacoGlobalType;
PETSC_EXTERN const char *const MPChacoGlobalTypes[];
/*E
   MPChacoLocalType - Local refinement method used by `MATPARTITIONINGCHACO` after the global partition has been chosen

   Values:
+   `MP_CHACO_KERNIGHAN` - apply Kernighan--Lin local refinement to improve the partition
-   `MP_CHACO_NONE`      - no local refinement

   Level: intermediate

.seealso: [](ch_matrices), `MatPartitioning`, `MATPARTITIONINGCHACO`, `MatPartitioningChacoSetLocal()`, `MatPartitioningChacoGetLocal()`,
          `MPChacoGlobalType`
E*/
typedef enum {
  MP_CHACO_KERNIGHAN = 1,
  MP_CHACO_NONE      = 2
} MPChacoLocalType;
PETSC_EXTERN const char *const MPChacoLocalTypes[];
/*E
   MPChacoEigenType - Eigensolver used by `MATPARTITIONINGCHACO` when the global partitioning method is `MP_CHACO_SPECTRAL`

   Values:
+   `MP_CHACO_LANCZOS` - Lanczos algorithm
-   `MP_CHACO_RQI`     - Rayleigh quotient iteration

   Level: intermediate

.seealso: [](ch_matrices), `MatPartitioning`, `MATPARTITIONINGCHACO`, `MatPartitioningChacoSetEigenSolver()`, `MatPartitioningChacoGetEigenSolver()`,
          `MPChacoGlobalType`
E*/
typedef enum {
  MP_CHACO_LANCZOS = 0,
  MP_CHACO_RQI     = 1
} MPChacoEigenType;
PETSC_EXTERN const char *const MPChacoEigenTypes[];

/*E
   MPPTScotchStrategyType - Pre-defined strategy used by `MATPARTITIONINGPTSCOTCH` when delegating to the PT-Scotch library

   Values:
+   `MP_PTSCOTCH_DEFAULT`     - the default PT-Scotch strategy
.   `MP_PTSCOTCH_QUALITY`     - focus on partition quality at the expense of time
.   `MP_PTSCOTCH_SPEED`       - focus on time to compute the partition
.   `MP_PTSCOTCH_BALANCE`     - favor load balance
.   `MP_PTSCOTCH_SAFETY`      - use the most reliable internal heuristics
-   `MP_PTSCOTCH_SCALABILITY` - favor scalability at large process counts

   Level: intermediate

.seealso: [](ch_matrices), `MatPartitioning`, `MATPARTITIONINGPTSCOTCH`, `MatPartitioningPTScotchSetStrategy()`, `MatPartitioningPTScotchGetStrategy()`
E*/
typedef enum {
  MP_PTSCOTCH_DEFAULT,
  MP_PTSCOTCH_QUALITY,
  MP_PTSCOTCH_SPEED,
  MP_PTSCOTCH_BALANCE,
  MP_PTSCOTCH_SAFETY,
  MP_PTSCOTCH_SCALABILITY
} MPPTScotchStrategyType;
PETSC_EXTERN const char *const MPPTScotchStrategyTypes[];

/*E
   MatOperation - Identifies one of the operations stored in a `Mat`'s function table, for example `MATOP_MULT` or `MATOP_LUFACTOR`

   Level: developer

   Notes:
   Use `MatSetOperation()` to install a custom implementation of a given operation on a `Mat` (typically a `MATSHELL`), `MatGetOperation()` to retrieve the currently installed implementation, and `MatHasOperation()` to test whether one is installed.

   The numeric values are part of the binary interface used by the petsc4py bindings; if any of these enum values are changed, the `dMatOps` dictionary in `src/binding/petsc4py/src/petsc4py/PETSc/libpetsc4py.pyx` must be updated to match.

.seealso: [](ch_matrices), `Mat`, `MATSHELL`, `MatSetOperation()`, `MatGetOperation()`, `MatHasOperation()`,
          `MatShellSetOperation()`, `MatShellGetOperation()`
E*/
typedef enum {
  MATOP_SET_VALUES                = 0,
  MATOP_GET_ROW                   = 1,
  MATOP_RESTORE_ROW               = 2,
  MATOP_MULT                      = 3,
  MATOP_MULT_ADD                  = 4,
  MATOP_MULT_TRANSPOSE            = 5,
  MATOP_MULT_TRANSPOSE_ADD        = 6,
  MATOP_SOLVE                     = 7,
  MATOP_SOLVE_ADD                 = 8,
  MATOP_SOLVE_TRANSPOSE           = 9,
  MATOP_SOLVE_TRANSPOSE_ADD       = 10,
  MATOP_LUFACTOR                  = 11,
  MATOP_CHOLESKYFACTOR            = 12,
  MATOP_SOR                       = 13,
  MATOP_TRANSPOSE                 = 14,
  MATOP_GETINFO                   = 15,
  MATOP_EQUAL                     = 16,
  MATOP_GET_DIAGONAL              = 17,
  MATOP_DIAGONAL_SCALE            = 18,
  MATOP_NORM                      = 19,
  MATOP_ASSEMBLY_BEGIN            = 20,
  MATOP_ASSEMBLY_END              = 21,
  MATOP_SET_OPTION                = 22,
  MATOP_ZERO_ENTRIES              = 23,
  MATOP_ZERO_ROWS                 = 24,
  MATOP_LUFACTOR_SYMBOLIC         = 25,
  MATOP_LUFACTOR_NUMERIC          = 26,
  MATOP_CHOLESKY_FACTOR_SYMBOLIC  = 27,
  MATOP_CHOLESKY_FACTOR_NUMERIC   = 28,
  MATOP_SETUP                     = 29,
  MATOP_ILUFACTOR_SYMBOLIC        = 30,
  MATOP_ICCFACTOR_SYMBOLIC        = 31,
  MATOP_GET_DIAGONAL_BLOCK        = 32,
  MATOP_SET_INF                   = 33,
  MATOP_DUPLICATE                 = 34,
  MATOP_FORWARD_SOLVE             = 35,
  MATOP_BACKWARD_SOLVE            = 36,
  MATOP_ILUFACTOR                 = 37,
  MATOP_ICCFACTOR                 = 38,
  MATOP_AXPY                      = 39,
  MATOP_CREATE_SUBMATRICES        = 40,
  MATOP_INCREASE_OVERLAP          = 41,
  MATOP_GET_VALUES                = 42,
  MATOP_COPY                      = 43,
  MATOP_GET_ROW_MAX               = 44,
  MATOP_SCALE                     = 45,
  MATOP_SHIFT                     = 46,
  MATOP_DIAGONAL_SET              = 47,
  MATOP_ZERO_ROWS_COLUMNS         = 48,
  MATOP_SET_RANDOM                = 49,
  MATOP_GET_ROW_IJ                = 50,
  MATOP_RESTORE_ROW_IJ            = 51,
  MATOP_GET_COLUMN_IJ             = 52,
  MATOP_RESTORE_COLUMN_IJ         = 53,
  MATOP_FDCOLORING_CREATE         = 54,
  MATOP_COLORING_PATCH            = 55,
  MATOP_SET_UNFACTORED            = 56,
  MATOP_PERMUTE                   = 57,
  MATOP_SET_VALUES_BLOCKED        = 58,
  MATOP_CREATE_SUBMATRIX          = 59,
  MATOP_DESTROY                   = 60,
  MATOP_VIEW                      = 61,
  MATOP_CONVERT_FROM              = 62,
  MATOP_MATMAT_MULT_SYMBOLIC      = 63,
  MATOP_MATMAT_MULT_NUMERIC       = 64,
  MATOP_SET_LOCAL_TO_GLOBAL_MAP   = 65,
  MATOP_SET_VALUES_LOCAL          = 66,
  MATOP_ZERO_ROWS_LOCAL           = 67,
  MATOP_GET_ROW_MAX_ABS           = 68,
  MATOP_GET_ROW_MIN_ABS           = 69,
  MATOP_CONVERT                   = 70,
  MATOP_HAS_OPERATION             = 71,
  MATOP_FD_COLORING_APPLY         = 72,
  MATOP_SET_FROM_OPTIONS          = 73,
  MATOP_FIND_ZERO_DIAGONALS       = 74,
  MATOP_MULT_MULTIPLE             = 75,
  MATOP_SOLVE_MULTIPLE            = 76,
  MATOP_GET_INERTIA               = 77,
  MATOP_LOAD                      = 78,
  MATOP_IS_SYMMETRIC              = 79,
  MATOP_IS_HERMITIAN              = 80,
  MATOP_IS_STRUCTURALLY_SYMMETRIC = 81,
  MATOP_SET_VALUES_BLOCKEDLOCAL   = 82,
  MATOP_CREATE_VECS               = 83,
  MATOP_MAT_MULT_SYMBOLIC         = 84,
  MATOP_MAT_MULT_NUMERIC          = 85,
  MATOP_PTAP_NUMERIC              = 86,
  MATOP_MAT_TRANSPOSE_MULT_SYMBO  = 87,
  MATOP_MAT_TRANSPOSE_MULT_NUMER  = 88,
  MATOP_BIND_TO_CPU               = 89,
  MATOP_PRODUCTSETFROMOPTIONS     = 90,
  MATOP_PRODUCTSYMBOLIC           = 91,
  MATOP_PRODUCTNUMERIC            = 92,
  MATOP_CONJUGATE                 = 93,
  MATOP_VIEW_NATIVE               = 94,
  MATOP_SET_VALUES_ROW            = 95,
  MATOP_REAL_PART                 = 96,
  MATOP_IMAGINARY_PART            = 97,
  MATOP_GET_ROW_UPPER_TRIANGULAR  = 98,
  MATOP_RESTORE_ROW_UPPER_TRIANG  = 99,
  MATOP_MAT_SOLVE                 = 100,
  MATOP_MAT_SOLVE_TRANSPOSE       = 101,
  MATOP_GET_ROW_MIN               = 102,
  MATOP_GET_COLUMN_VECTOR         = 103,
  MATOP_GET_SEQ_NONZERO_STRUCTUR  = 104,
  MATOP_CREATE                    = 105,
  MATOP_GET_GHOSTS                = 106,
  MATOP_GET_LOCAL_SUB_MATRIX      = 107,
  MATOP_RESTORE_LOCALSUB_MATRIX   = 108,
  MATOP_MULT_DIAGONAL_BLOCK       = 109,
  MATOP_HERMITIAN_TRANSPOSE       = 110,
  MATOP_MULT_HERMITIAN_TRANSPOSE  = 111,
  MATOP_MULT_HERMITIAN_TRANS_ADD  = 112,
  MATOP_GET_MULTI_PROC_BLOCK      = 113,
  MATOP_FIND_NONZERO_ROWS         = 114,
  MATOP_GET_COLUMN_NORMS          = 115,
  MATOP_INVERT_BLOCK_DIAGONAL     = 116,
  MATOP_INVERT_VBLOCK_DIAGONAL    = 117,
  MATOP_CREATE_SUB_MATRICES_MPI   = 118,
  MATOP_TRANSPOSE_MAT_MULT_SYMBO  = 119,
  MATOP_TRANSPOSE_MAT_MULT_NUMER  = 120,
  MATOP_TRANSPOSE_COLORING_CREAT  = 121,
  MATOP_TRANS_COLORING_APPLY_SPT  = 122,
  MATOP_TRANS_COLORING_APPLY_DEN  = 123,
  MATOP_RART_NUMERIC              = 124,
  MATOP_SET_BLOCK_SIZES           = 125,
  MATOP_RESIDUAL                  = 126,
  MATOP_FDCOLORING_SETUP          = 127,
  MATOP_FIND_OFFBLOCK_ENTRIES     = 128,
  MATOP_MPICONCATENATESEQ         = 129,
  MATOP_DESTROYSUBMATRICES        = 130,
  MATOP_MAT_TRANSPOSE_SOLVE       = 131,
  MATOP_GET_VALUES_LOCAL          = 132,
  MATOP_CREATE_GRAPH              = 133,
  MATOP_TRANSPOSE_SYMBOLIC        = 134,
  MATOP_ELIMINATE_ZEROS           = 135,
  MATOP_GET_ROW_SUM_ABS           = 136,
  MATOP_GET_FACTOR                = 137,
  MATOP_GET_BLOCK_DIAGONAL        = 138, /* NOTE: caller of the two op functions owns the returned matrix */
  MATOP_GET_VBLOCK_DIAGONAL       = 139, /* and need to destroy it after use. */
  MATOP_COPY_HASH_TO_XAIJ         = 140,
  MATOP_GET_CURRENT_MEM_TYPE      = 141,
  MATOP_ZERO_ROWS_COLUMNS_LOCAL   = 142,
  MATOP_ADOT                      = 143,
  MATOP_ANORM                     = 144,
  MATOP_ADOT_LOCAL                = 145,
  MATOP_ANORM_LOCAL               = 146,
  MATOP_GET_ORDERING              = 147
} MatOperation;

/*S
   MatNullSpace - Object that removes a null space from a vector, i.e.
                  orthogonalizes the vector to a subspace

   Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatNullSpaceCreate()`, `MatNullSpaceSetFunction()`, `MatGetNullSpace()`, `MatSetNullSpace()`
S*/
typedef struct _p_MatNullSpace *MatNullSpace;

/*S
  MatNullSpaceRemoveFn - Function provided to `MatNullSpaceSetFunction()` that removes the null space from a vector

  Level: advanced

  Calling Sequence:
+ nsp - the `MatNullSpace` object
. x   - the vector from which to remove the null space
- ctx - [optional] user-defined function context provided with `MatNullSpaceSetFunction()`

.seealso: [](ch_matrices), `Mat`, `MatNullSpaceCreate()`, `MatNullSpaceSetFunction()`, `MatGetNullSpace()`, `MatSetNullSpace()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatNullSpaceRemoveFn(MatNullSpace nsp, Vec x, PetscCtx ctx);

/*S
  MatMFFDFn - Function provided to `MatMFFDSetFunction()` that computes the function being differenced

  Level: advanced

  Calling Sequence:
+ ctx - [optional] user-defined function context provided with `MatMFFDSetFunction()`
. x   - input vector
- y   - output vector

.seealso: [](ch_matrices), `Mat`, `MatCreateMFFD()`, `MatMFFDSetFunction()`, `MatMFFDiFn`, `MatMFFDiBaseFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatMFFDFn(PetscCtx ctx, Vec x, Vec y);

/*S
  MatMFFDiFn - Function provided to `MatMFFDSetFunctioni()` that computes the function being differenced at a single point

  Level: advanced

  Calling Sequence:
+ ctx    - [optional] user-defined function context provided with `MatMFFDSetFunction()`
. i      - the component of the vector to compute
. x      - input vector
- result - the value of the function at that component (output)

.seealso: [](ch_matrices), `Mat`, `MatCreateMFFD()`, `MatMFFDSetFunction()`, `MatMFFDFn`, `MatMFFDiBaseFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatMFFDiFn(PetscCtx ctx, PetscInt i, Vec x, PetscScalar *result);

/*S
  MatMFFDiBaseFn - Function provided to `MatMFFDSetFunctioniBase()` that computes the base of the function evaluations
  that will be used for differencing

  Level: advanced

  Calling Sequence:
+ ctx - [optional] user-defined function context provided with `MatMFFDSetFunction()`
- x   - input base vector

.seealso: [](ch_matrices), `Mat`, `MatCreateMFFD()`, `MatMFFDSetFunction()`, `MatMFFDSetFunctioniBase()`, `MatMFFDFn`, `MatMFFDiFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatMFFDiBaseFn(PetscCtx ctx, Vec x);

/*S
  MatMFFDCheckhFn - Function provided to `MatMFFDSetCheckh()` that checks and possibly adjusts the value of `h` to ensure some property.
  that will be used for differencing

  Level: advanced

  Calling Sequence:
+ ctx - [optional] user-defined function context provided with `MatMFFDSetCheckh()`
. x   - input base vector
. y   - input step vector that the product is computed with
- h   - input tentative step, output possibly adjusted step

  Note:
  `MatMFFDCheckPositivity()` is one such function

.seealso: [](ch_matrices), `Mat`, `MatCreateMFFD()`, `MatMFFDSetCheckh()`, `MatMFFDCheckPositivity()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode MatMFFDCheckhFn(PetscCtx ctx, Vec x, Vec y, PetscScalar *h);

/*S
    MatMFFD - A data structured used to manage the computation of the h differencing parameter for matrix-free
              Jacobian vector products

    Level: developer

    Notes:
    `MATMFFD` is a specific `MatType` which uses the `MatMFFD` data structure

     MatMFFD*() methods actually take the `Mat` as their first argument. Not a `MatMFFD` data structure

    This functionality is often obtained using `MatCreateSNESMF()` or with `SNES` solvers using `-snes_mf` or `-snes_mf_operator`

.seealso: [](ch_matrices), `MatMFFDType`, `MATMFFD`, `MatCreateMFFD()`, `MatMFFDSetFuction()`, `MatMFFDSetType()`, `MatMFFDRegister()`,
          `MatCreateSNESMF()`, `SNES`, `-snes_mf`, `-snes_mf_operator`
S*/
typedef struct _p_MatMFFD *MatMFFD;

/*J
   MatMFFDType - algorithm used to compute the `h` used in computing matrix-vector products via differencing of a function

   Values:
+   `MATMFFD_DS` - an algorithm described by Dennis and Schnabel {cite}`dennis:83`
-   `MATMFFD_WP` - the Walker-Pernice {cite}`pw98` strategy.

   Level: beginner

.seealso: [](ch_matrices), `MatMFFDSetType()`, `MatMFFDRegister()`, `MatMFFDSetFunction()`, `MatCreateMFFD()`
J*/
typedef const char *MatMFFDType;
#define MATMFFD_DS "ds"
#define MATMFFD_WP "wp"

#if PetscDefined(HAVE_H2OPUS)
PETSC_EXTERN_TYPEDEF typedef PetscScalar(MatH2OpusKernelFn)(PetscInt, PetscReal[], PetscReal[], void *);
PETSC_EXTERN_TYPEDEF typedef MatH2OpusKernelFn *MatH2OpusKernel;
#endif

/*S
  MatHtoolKernelFn - Function type for the user-supplied kernel callback used by `MATHTOOL` (`MatCreateHtoolFromKernel()`, `MatHtoolSetKernel()`) to evaluate the dense matrix entries on demand

  Synopsis:
  #include <petscmat.h>
  PetscErrorCode MatHtoolKernelFn(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *J, const PetscInt *K, PetscScalar *ptr, void *ctx)

  Calling Sequence:
+ sdim - the spatial dimension of the source/target geometries
. M    - the number of target points
. N    - the number of source points
. J    - array of `M` target point indices into the user's target coordinate array
. K    - array of `N` source point indices into the user's source coordinate array
. ptr  - column-major output buffer of length `M*N` to fill with kernel values
- ctx  - the optional application context passed at registration

  Level: intermediate

.seealso: `Mat`, `MATHTOOL`, `MatCreateHtoolFromKernel()`, `MatHtoolSetKernel()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode    MatHtoolKernelFn(PetscInt, PetscInt, PetscInt, const PetscInt *, const PetscInt *, PetscScalar *, void *);
PETSC_EXTERN_TYPEDEF typedef MatHtoolKernelFn *MatHtoolKernel;

/*E
   MatHtoolCompressorType - Indicates the type of compressor used by a `MATHTOOL`

   Values:
+  `MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA` (default) - symmetric partial adaptive cross approximation
.  `MAT_HTOOL_COMPRESSOR_FULL_ACA`                 - full adaptive cross approximation
-  `MAT_HTOOL_COMPRESSOR_SVD`                      - singular value decomposition

   Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatCreateHtoolFromKernel()`, `MATHTOOL`, `MatHtoolClusteringType`, `MatHtoolGetCompressorType()`, `MatHtoolSetCompressorType()`
E*/
typedef enum {
  MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA,
  MAT_HTOOL_COMPRESSOR_FULL_ACA,
  MAT_HTOOL_COMPRESSOR_SVD
} MatHtoolCompressorType;

/*E
   MatHtoolClusteringType - Indicates the type of clustering used by a `MATHTOOL`

   Values:
+  `MAT_HTOOL_CLUSTERING_PCA_REGULAR` (default)    - axis computed via principle component analysis, split uniformly
.  `MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC`            - axis computed via principle component analysis, split barycentrically
.  `MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR`   - axis along the largest extent of the bounding box, split uniformly
-  `MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC` - axis along the largest extent of the bounding box, split barycentrically

   Level: intermediate

   Note:
   Higher-dimensional clustering is not yet supported in Htool, but once it is, one should add BOUNDING_BOX_{2,3} types

.seealso: [](ch_matrices), `Mat`, `MatCreateHtoolFromKernel()`, `MATHTOOL`, `MatHtoolCompressorType`, `MatHtoolGetClusteringType()`, `MatHtoolSetClusteringType()`
E*/
typedef enum {
  MAT_HTOOL_CLUSTERING_PCA_REGULAR,
  MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC,
  MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR,
  MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC
} MatHtoolClusteringType;

/*E
    MatSTRUMPACKReordering - sparsity reducing ordering to be used in `MATSOLVERSTRUMPACK`

    Values:
+  `MAT_STRUMPACK_NATURAL`   - use the current ordering
.  `MAT_STRUMPACK_METIS`     - use MeTis to compute an ordering
.  `MAT_STRUMPACK_PARMETIS`  - use ParMeTis to compute an ordering
.  `MAT_STRUMPACK_SCOTCH`    - use Scotch to compute an ordering
.  `MAT_STRUMPACK_PTSCOTCH`  - use parallel Scotch to compute an ordering
.  `MAT_STRUMPACK_RCM`       - use an RCM ordering
.  `MAT_STRUMPACK_GEOMETRIC` - use a geometric ordering (needs mesh info from user)
.  `MAT_STRUMPACK_AMD`       - approximate minimum degree
.  `MAT_STRUMPACK_MMD`       - multiple minimum degree
.  `MAT_STRUMPACK_AND`       - approximate nested dissection
.  `MAT_STRUMPACK_MLF`       - minimum local fill
-  `MAT_STRUMPACK_SPECTRAL`  - spectral nested dissection

    Level: intermediate

    Developer Note:
    Should be called `MatSTRUMPACKReorderingType`

.seealso: `Mat`, `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`
E*/
typedef enum {
  MAT_STRUMPACK_NATURAL,
  MAT_STRUMPACK_METIS,
  MAT_STRUMPACK_PARMETIS,
  MAT_STRUMPACK_SCOTCH,
  MAT_STRUMPACK_PTSCOTCH,
  MAT_STRUMPACK_RCM,
  MAT_STRUMPACK_GEOMETRIC,
  MAT_STRUMPACK_AMD,
  MAT_STRUMPACK_MMD,
  MAT_STRUMPACK_AND,
  MAT_STRUMPACK_MLF,
  MAT_STRUMPACK_SPECTRAL
} MatSTRUMPACKReordering;
PETSC_EXTERN const char *const MatSTRUMPACKReorderingTypes[];

/*E
    MatSTRUMPACKCompressionType - Compression used in the approximate sparse factorization solver `MATSOLVERSTRUMPACK`

    Values:
+  `MAT_STRUMPACK_COMPRESSION_TYPE_NONE`          - no compression, direct solver
.  `MAT_STRUMPACK_COMPRESSION_TYPE_HSS`           - hierarchically semi-separable
.  `MAT_STRUMPACK_COMPRESSION_TYPE_BLR`           - block low rank
.  `MAT_STRUMPACK_COMPRESSION_TYPE_HODLR`         - hierarchically off-diagonal low rank (requires ButterflyPACK support, configure with `--download-butterflypack`)
.  `MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR`     - hybrid of BLR and HODLR (requires ButterflyPACK support, configure with `--download-butterflypack`)
.  `MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR` - hybrid of lossy (ZFP), BLR and HODLR (requires ButterflyPACK and ZFP support, configure with `--download-butterflypack --download-zfp`)
.  `MAT_STRUMPACK_COMPRESSION_TYPE_LOSSLESS`      - lossless compression (requires ZFP support, configure with `--download-zfp`)
-  `MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY`         - lossy compression (requires ZFP support, configure with `--download-zfp`)

    Level: intermediate

.seealso: `Mat`, `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetCompression()`
E*/
typedef enum {
  MAT_STRUMPACK_COMPRESSION_TYPE_NONE,
  MAT_STRUMPACK_COMPRESSION_TYPE_HSS,
  MAT_STRUMPACK_COMPRESSION_TYPE_BLR,
  MAT_STRUMPACK_COMPRESSION_TYPE_HODLR,
  MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR,
  MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR,
  MAT_STRUMPACK_COMPRESSION_TYPE_LOSSLESS,
  MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY
} MatSTRUMPACKCompressionType;
PETSC_EXTERN const char *const MatSTRUMPACKCompressionTypes[];

#if PetscDefined(HAVE_CUDA)
/*E
    MatCUSPARSEStorageFormat - indicates the storage format for `MATAIJCUSPARSE` (GPU)
    matrices.

    Values:
+   `MAT_CUSPARSE_CSR` - Compressed Sparse Row
.   `MAT_CUSPARSE_ELL` - Ellpack (requires CUDA 4.2 or later).
-   `MAT_CUSPARSE_HYB` - Hybrid, a combination of Ellpack and Coordinate format (requires CUDA 4.2 or later).

    Level: intermediate

.seealso: [](ch_matrices), `MATAIJCUSPARSE`, `MatCUSPARSESetFormat()`, `MatCUSPARSEFormatOperation`
E*/

typedef enum {
  MAT_CUSPARSE_CSR,
  MAT_CUSPARSE_ELL,
  MAT_CUSPARSE_HYB
} MatCUSPARSEStorageFormat;

/* these will be strings associated with enumerated type defined above */
PETSC_EXTERN const char *const MatCUSPARSEStorageFormats[];

/*E
    MatCUSPARSEFormatOperation - indicates the operation of `MATAIJCUSPARSE` (GPU)
    matrices whose operation should use a particular storage format.

    Values:
+   `MAT_CUSPARSE_MULT_DIAG`    - sets the storage format for the diagonal matrix in the parallel `MatMult()`
.   `MAT_CUSPARSE_MULT_OFFDIAG` - sets the storage format for the off-diagonal matrix in the parallel `MatMult()`
.   `MAT_CUSPARSE_MULT`         - sets the storage format for the entire matrix in the serial (single GPU) `MatMult()`
-   `MAT_CUSPARSE_ALL`          - sets the storage format for all `MATAIJCUSPARSE` (GPU) matrices

    Level: intermediate

.seealso: [](ch_matrices), `MatCUSPARSESetFormat()`, `MatCUSPARSEStorageFormat`
E*/
typedef enum {
  MAT_CUSPARSE_MULT_DIAG,
  MAT_CUSPARSE_MULT_OFFDIAG,
  MAT_CUSPARSE_MULT,
  MAT_CUSPARSE_ALL
} MatCUSPARSEFormatOperation;
#endif

#if PetscDefined(HAVE_HIP)
/*E
    MatHIPSPARSEStorageFormat - indicates the storage format for `MATAIJHIPSPARSE` (GPU)
    matrices.

    Values:
+   `MAT_HIPSPARSE_CSR` - Compressed Sparse Row
.   `MAT_HIPSPARSE_ELL` - Ellpack
-   `MAT_HIPSPARSE_HYB` - Hybrid, a combination of Ellpack and Coordinate format

    Level: intermediate

.seealso: [](ch_matrices), `MatHIPSPARSESetFormat()`, `MatHIPSPARSEFormatOperation`
E*/

typedef enum {
  MAT_HIPSPARSE_CSR,
  MAT_HIPSPARSE_ELL,
  MAT_HIPSPARSE_HYB
} MatHIPSPARSEStorageFormat;

/* these will be strings associated with enumerated type defined above */
PETSC_EXTERN const char *const MatHIPSPARSEStorageFormats[];

/*E
    MatHIPSPARSEFormatOperation - indicates the operation of `MATAIJHIPSPARSE` (GPU)
    matrices whose operation should use a particular storage format.

    Values:
+   `MAT_HIPSPARSE_MULT_DIAG`    - sets the storage format for the diagonal matrix in the parallel `MatMult()`
.   `MAT_HIPSPARSE_MULT_OFFDIAG` - sets the storage format for the off-diagonal matrix in the parallel `MatMult()`
.   `MAT_HIPSPARSE_MULT`         - sets the storage format for the entire matrix in the serial (single GPU) `MatMult()`
-   `MAT_HIPSPARSE_ALL`          - sets the storage format for all HIPSPARSE (GPU) matrices

    Level: intermediate

.seealso: [](ch_matrices), `MatHIPSPARSESetFormat()`, `MatHIPSPARSEStorageFormat`
E*/
typedef enum {
  MAT_HIPSPARSE_MULT_DIAG,
  MAT_HIPSPARSE_MULT_OFFDIAG,
  MAT_HIPSPARSE_MULT,
  MAT_HIPSPARSE_ALL
} MatHIPSPARSEFormatOperation;
#endif
