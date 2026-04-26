/*
     This file implements a subclass of the SeqAIJ matrix class that uses
     the NVIDIA cuDSS GPU-accelerated sparse direct solver library.
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <petscdevice_cuda.h>
#include <cudss.h>

/*
    Internal data structure for the cuDSS factored matrix
*/
typedef struct {
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t   data;
  cudssMatrix_t cudss_A; /* sparse matrix descriptor */
  cudssMatrix_t cudss_b; /* dense RHS descriptor */
  cudssMatrix_t cudss_x; /* dense solution descriptor */

  /* Device CSR arrays (owned by us when input is MATSEQAIJ) */
  PetscInt    *d_row_offsets; /* size m+1 */
  PetscInt    *d_col_indices; /* size nnz */
  PetscScalar *d_values;      /* size nnz */

  /* Device scratch buffers for RHS/solution descriptors (size n) */
  PetscScalar *d_b; /* initial valid pointer for cudss_b descriptor */
  PetscScalar *d_x; /* initial valid pointer for cudss_x descriptor */

  PetscInt n; /* matrix dimension */
  PetscInt nnz;

  /* Whether we own the device CSR memory (MATSEQAIJ input) */
  PetscBool ownDeviceCSR;

  /* The cuDSS index type matching the actual device CSR storage:
     CUDA_R_32I for MATSEQAIJCUSPARSE (always 32-bit int),
     PETSCINT_CUDSS_INDEX_TYPE for owned CSR (matches PetscInt) */
  cudaDataType indexType;

  /* Solver options - use cuDSS enum types directly for cudssConfigSet compatibility */
  cudssAlgType_t   reorderAlg;     /* CUDSS_CONFIG_REORDERING_ALG */
  cudssPivotType_t pivotType;      /* CUDSS_CONFIG_PIVOT_TYPE */
  double           pivotThreshold; /* CUDSS_CONFIG_PIVOT_THRESHOLD (always double) */
  double           pivotEpsilon;   /* CUDSS_CONFIG_PIVOT_EPSILON (always double) */
  int              useMatching;    /* CUDSS_CONFIG_USE_MATCHING (plain int: 0=false, 1=true) */
  int              irNSteps;       /* CUDSS_CONFIG_IR_N_STEPS (int) */

} Mat_cuDSS;

/* Map PetscScalar to the cuDSS data type */
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define CUDSS_SCALAR_TYPE CUDA_C_32F
  #else
    #define CUDSS_SCALAR_TYPE CUDA_C_64F
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define CUDSS_SCALAR_TYPE CUDA_R_32F
  #else
    #define CUDSS_SCALAR_TYPE CUDA_R_64F
  #endif
#endif

/* Map PetscInt to the cuDSS index type (for owned CSR buffers sized as PetscInt) */
#if defined(PETSC_USE_64BIT_INDICES)
  #define PETSCINT_CUDSS_INDEX_TYPE CUDA_R_64I
#else
  #define PETSCINT_CUDSS_INDEX_TYPE CUDA_R_32I
#endif

#define PetscCallCUDSS(func, ...) \
  do { \
    cudssStatus_t cudss_status_ = func(__VA_ARGS__); \
    PetscCheck(cudss_status_ == CUDSS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error %d in %s", (int)cudss_status_, PetscStringize(func)); \
  } while (0)

/* String names for cudssAlgType_t: CUDSS_ALG_DEFAULT=0, CUDSS_ALG_1..5 */
static const char *const MatcuDSSReorderAlgs[] = {"default", "alg1", "alg2", "alg3", "alg4", "alg5"};

/* String names for cudssPivotType_t: CUDSS_PIVOT_COL=0, CUDSS_PIVOT_ROW=1, CUDSS_PIVOT_NONE=2 */
static const char *const MatcuDSSPivotTypes[] = {"col", "row", "none"};

/* Forward declarations */
static PetscErrorCode MatLUFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatSetFromOptions_cuDSS(Mat);
static PetscErrorCode MatFactorSymbolic_cuDSS(Mat, Mat, cudssMatrixType_t, cudssMatrixViewType_t);
/* cuSPARSE CsrMatrix always stores row_offsets/column_indices as 32-bit int */
#define CUSPARSE_CSR_INDEX_TYPE CUDA_R_32I

static PetscErrorCode MatView_Info_cuDSS(Mat A, PetscViewer viewer)
{
  Mat_cuDSS  *lu          = (Mat_cuDSS *)A->data;
  const char *reorderName = ((int)lu->reorderAlg >= 0 && (size_t)lu->reorderAlg < PETSC_STATIC_ARRAY_LENGTH(MatcuDSSReorderAlgs)) ? MatcuDSSReorderAlgs[lu->reorderAlg] : "unknown";
  const char *pivotName   = ((int)lu->pivotType >= 0 && (size_t)lu->pivotType < PETSC_STATIC_ARRAY_LENGTH(MatcuDSSPivotTypes)) ? MatcuDSSPivotTypes[lu->pivotType] : "unknown";

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "cuDSS run parameters:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Reorder algorithm: %s\n", reorderName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot type: %s\n", pivotName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot threshold: %g\n", lu->pivotThreshold));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot epsilon: %g\n", lu->pivotEpsilon));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Use matching: %s\n", lu->useMatching ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  IR steps: %d\n", lu->irNSteps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_cuDSS(Mat A, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_cuDSS(A, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_cuDSS(Mat A)
{
  Mat_cuDSS *lu = (Mat_cuDSS *)A->data;

  PetscFunctionBegin;
  if (lu->cudss_x) PetscCallCUDSS(cudssMatrixDestroy, lu->cudss_x);
  if (lu->cudss_b) PetscCallCUDSS(cudssMatrixDestroy, lu->cudss_b);
  if (lu->cudss_A) PetscCallCUDSS(cudssMatrixDestroy, lu->cudss_A);
  if (lu->data) PetscCallCUDSS(cudssDataDestroy, lu->handle, lu->data);
  if (lu->config) PetscCallCUDSS(cudssConfigDestroy, lu->config);
  if (lu->handle) PetscCallCUDSS(cudssDestroy, lu->handle);
  if (lu->ownDeviceCSR) {
    PetscCallCUDA(cudaFree(lu->d_row_offsets));
    PetscCallCUDA(cudaFree(lu->d_col_indices));
    PetscCallCUDA(cudaFree(lu->d_values));
  }
  PetscCallCUDA(cudaFree(lu->d_b));
  PetscCallCUDA(cudaFree(lu->d_x));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_cuDSS(Mat F)
{
  Mat_cuDSS *lu         = (Mat_cuDSS *)F->data;
  PetscInt   reorderAlg = (PetscInt)lu->reorderAlg, pivotType = (PetscInt)lu->pivotType, irNSteps = (PetscInt)lu->irNSteps;
  PetscReal  pivotThreshold = (PetscReal)lu->pivotThreshold, pivotEpsilon = (PetscReal)lu->pivotEpsilon;
  PetscBool  useMatchingBool = (PetscBool)lu->useMatching;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "cuDSS Options", "Mat");
  PetscCall(PetscOptionsEList("-mat_cudss_reorder_alg", "Reordering algorithm", "None", MatcuDSSReorderAlgs, PETSC_STATIC_ARRAY_LENGTH(MatcuDSSReorderAlgs), MatcuDSSReorderAlgs[reorderAlg], &reorderAlg, NULL));
  lu->reorderAlg = (cudssAlgType_t)reorderAlg;
  PetscCall(PetscOptionsEList("-mat_cudss_pivot_type", "Pivot type", "None", MatcuDSSPivotTypes, PETSC_STATIC_ARRAY_LENGTH(MatcuDSSPivotTypes), MatcuDSSPivotTypes[pivotType], &pivotType, NULL));
  lu->pivotType = (cudssPivotType_t)pivotType;
  PetscCall(PetscOptionsReal("-mat_cudss_pivot_threshold", "Pivot threshold", "None", pivotThreshold, &pivotThreshold, NULL));
  lu->pivotThreshold = (double)pivotThreshold;
  PetscCall(PetscOptionsReal("-mat_cudss_pivot_epsilon", "Pivot epsilon", "None", pivotEpsilon, &pivotEpsilon, NULL));
  lu->pivotEpsilon = (double)pivotEpsilon;
  PetscCall(PetscOptionsBool("-mat_cudss_use_matching", "Enable matching (CUDSS_CONFIG_USE_MATCHING)", "None", useMatchingBool, &useMatchingBool, NULL));
  lu->useMatching = (int)useMatchingBool;
  PetscCall(PetscOptionsInt("-mat_cudss_ir_n_steps", "Number of iterative refinement steps", "None", irNSteps, &irNSteps, NULL));
  lu->irNSteps = (int)irNSteps;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Helper: apply cuDSS config options from the Mat_cuDSS struct to the cudssConfig_t object.
*/
static PetscErrorCode MatApplyConfig_cuDSS(Mat_cuDSS *lu)
{
  PetscFunctionBegin;
  PetscCallCUDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_REORDERING_ALG, &lu->reorderAlg, sizeof(lu->reorderAlg));
  PetscCallCUDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_PIVOT_TYPE, &lu->pivotType, sizeof(lu->pivotType));
  PetscCallCUDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_PIVOT_THRESHOLD, &lu->pivotThreshold, sizeof(lu->pivotThreshold));
  PetscCallCUDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_PIVOT_EPSILON, &lu->pivotEpsilon, sizeof(lu->pivotEpsilon));
  PetscCallCUDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_USE_MATCHING, &lu->useMatching, sizeof(lu->useMatching));
  PetscCallCUDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_IR_N_STEPS, &lu->irNSteps, sizeof(lu->irNSteps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Helper: ensure CSR data is on the device.
   For MATSEQAIJCUSPARSE: use MatSeqAIJCUSPARSECopyToGPU and get device pointers.
   For MATSEQAIJ: copy host arrays to lu->d_row_offsets/d_col_indices/d_values.
   Returns device pointers in d_row, d_col, d_val (not owned by caller).
*/
static PetscErrorCode MatEnsureOnDevice_cuDSS(Mat A, Mat_cuDSS *lu, void **d_row, void **d_col, PetscScalar **d_val)
{
  PetscBool    isCUSPARSE;
  cudaStream_t stream;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  if (isCUSPARSE) {
    /* Ensure data is current on device.
       Note: CsrMatrix row_offsets/column_indices are THRUSTINTARRAY32 (always 32-bit int),
       so we return them as void* and use CUSPARSE_CSR_INDEX_TYPE (CUDA_R_32I) in the descriptor. */
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
    {
      Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE *)A->spptr;
      Mat_SeqAIJCUSPARSEMultStruct *matstruct      = (Mat_SeqAIJCUSPARSEMultStruct *)cusparsestruct->mat;
      CsrMatrix                    *csr            = (CsrMatrix *)matstruct->mat;
      *d_row                                       = (void *)thrust::raw_pointer_cast(csr->row_offsets->data());
      *d_col                                       = (void *)thrust::raw_pointer_cast(csr->column_indices->data());
      *d_val                                       = (PetscScalar *)thrust::raw_pointer_cast(csr->values->data());
    }
  } else {
    /* MATSEQAIJ: async copy host CSR to device on the cuDSS stream (PetscInt-sized indices) */
    Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
    PetscCall(PetscGetCurrentCUDAStream(&stream));
    PetscCallCUDA(cudaMemcpyAsync(lu->d_row_offsets, a->i, (lu->n + 1) * sizeof(PetscInt), cudaMemcpyHostToDevice, stream));
    PetscCallCUDA(cudaMemcpyAsync(lu->d_col_indices, a->j, lu->nnz * sizeof(PetscInt), cudaMemcpyHostToDevice, stream));
    PetscCallCUDA(cudaMemcpyAsync(lu->d_values, a->a, lu->nnz * sizeof(PetscScalar), cudaMemcpyHostToDevice, stream));
    *d_row = (void *)lu->d_row_offsets;
    *d_col = (void *)lu->d_col_indices;
    *d_val = lu->d_values;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_cuDSS(Mat F, Mat A, cudssMatrixType_t mtype, cudssMatrixViewType_t mview)
{
  Mat_cuDSS   *lu = (Mat_cuDSS *)F->data;
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)A->data;
  PetscInt     m, n, nnz;
  void        *d_row, *d_col;
  PetscScalar *d_val;
  PetscBool    isCUSPARSE;
  cudaStream_t stream;

  PetscFunctionBegin;
  /* Destroy any previously created cuDSS objects (re-symbolization case) */
  if (lu->cudss_x) {
    PetscCallCUDSS(cudssMatrixDestroy, lu->cudss_x);
    lu->cudss_x = NULL;
  }
  if (lu->cudss_b) {
    PetscCallCUDSS(cudssMatrixDestroy, lu->cudss_b);
    lu->cudss_b = NULL;
  }
  if (lu->cudss_A) {
    PetscCallCUDSS(cudssMatrixDestroy, lu->cudss_A);
    lu->cudss_A = NULL;
  }
  if (lu->data) {
    PetscCallCUDSS(cudssDataDestroy, lu->handle, lu->data);
    lu->data = NULL;
  }
  if (lu->config) {
    PetscCallCUDSS(cudssConfigDestroy, lu->config);
    lu->config = NULL;
  }
  if (lu->handle) {
    PetscCallCUDSS(cudssDestroy, lu->handle);
    lu->handle = NULL;
  }
  if (lu->ownDeviceCSR) {
    PetscCallCUDA(cudaFree(lu->d_row_offsets));
    lu->d_row_offsets = NULL;
    PetscCallCUDA(cudaFree(lu->d_col_indices));
    lu->d_col_indices = NULL;
    PetscCallCUDA(cudaFree(lu->d_values));
    lu->d_values = NULL;
  }
  if (lu->d_b) {
    PetscCallCUDA(cudaFree(lu->d_b));
    lu->d_b = NULL;
  }
  if (lu->d_x) {
    PetscCallCUDA(cudaFree(lu->d_x));
    lu->d_x = NULL;
  }

  m   = A->rmap->n;
  n   = A->cmap->n;
  nnz = a->nz;

  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "cuDSS requires a square matrix, got %" PetscInt_FMT "x%" PetscInt_FMT, m, n);

  lu->n   = n;
  lu->nnz = nnz;

  PetscCall(MatSetFromOptions_cuDSS(F));

  /* Initialize cuDSS handle, config, data */
  PetscCallCUDSS(cudssCreate, &lu->handle);
  PetscCallCUDSS(cudssConfigCreate, &lu->config);
  PetscCallCUDSS(cudssDataCreate, lu->handle, &lu->data);
  PetscCall(MatApplyConfig_cuDSS(lu));

  /* Determine if input is CUSPARSE; allocate device CSR if needed.
     CUSPARSE CsrMatrix always uses 32-bit int for indices, so we must
     tell cuDSS the actual index type (CUDA_R_32I), not PetscInt's size. */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  lu->ownDeviceCSR = (PetscBool)(!isCUSPARSE);
  if (isCUSPARSE) {
    lu->indexType = CUSPARSE_CSR_INDEX_TYPE; /* always CUDA_R_32I */
  } else {
    lu->indexType = PETSCINT_CUDSS_INDEX_TYPE; /* matches sizeof(PetscInt) */
    PetscCallCUDA(cudaMalloc((void **)&lu->d_row_offsets, (m + 1) * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_col_indices, nnz * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_values, nnz * sizeof(PetscScalar)));
  }

  /* Set PETSc CUDA stream on the cuDSS handle */
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);

  /* Ensure CSR data is on device and create sparse matrix descriptor */
  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, &d_row, &d_col, &d_val));
  PetscCallCUDSS(cudssMatrixCreateCsr, &lu->cudss_A, m, n, nnz, d_row, NULL, d_col, d_val, lu->indexType, CUDSS_SCALAR_TYPE, mtype, mview, CUDSS_BASE_ZERO);

  /* Allocate scratch device buffers and create dense RHS/solution descriptors */
  PetscCallCUDA(cudaMalloc((void **)&lu->d_b, n * sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&lu->d_x, n * sizeof(PetscScalar)));
  PetscCallCUDSS(cudssMatrixCreateDn, &lu->cudss_b, n, 1, n, lu->d_b, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallCUDSS(cudssMatrixCreateDn, &lu->cudss_x, n, 1, n, lu->d_x, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);

  /* Run analysis (symbolic factorization / reordering) */
  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_ANALYSIS, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_cuDSS(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* cuDSS performs its own internal reordering; any PETSc-supplied row/column permutations are ignored */
  if (r || c) PetscCall(PetscInfo(F, "cuDSS ignores user-supplied row/column permutations; cuDSS will apply its own reordering\n"));
  PetscCall(MatFactorSymbolic_cuDSS(F, A, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL));
  F->ops->lufactornumeric = MatLUFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_cuDSS(Mat F, Mat A, IS perm, const MatFactorInfo *info)
{
  cudssMatrixType_t mtype;

  PetscFunctionBegin;
  /* cuDSS performs its own internal reordering; any PETSc-supplied permutation is ignored */
  if (perm) PetscCall(PetscInfo(F, "cuDSS ignores user-supplied permutation; cuDSS will apply its own reordering\n"));
#if defined(PETSC_USE_COMPLEX)
  if (A->spd == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_HPD;
  else if (A->hermitian == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_HERMITIAN;
  else mtype = CUDSS_MTYPE_SYMMETRIC;
#else
  if (A->spd == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_SPD;
  else mtype = CUDSS_MTYPE_SYMMETRIC;
#endif
  PetscCall(MatFactorSymbolic_cuDSS(F, A, mtype, CUDSS_MVIEW_UPPER));
  F->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_cuDSS(Mat F, Vec b, Vec x)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  cudaStream_t       stream;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArrayRead(b, &barray));
  PetscCall(VecCUDAGetArrayWrite(x, &xarray));
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_b, (void *)barray);
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_x, xarray);
  /* Ensure cuDSS uses the current PETSc stream for solve */
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);
  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);
  PetscCall(VecCUDARestoreArrayRead(b, &barray));
  PetscCall(VecCUDARestoreArrayWrite(x, &xarray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_cuDSS(Mat F, Mat B, Mat X)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  PetscInt           n, nrhs, nX, nrhsX, ldb, ldx;
  cudssMatrix_t      cudss_B = NULL, cudss_X = NULL;
  Mat                Bcuda = NULL, Xcuda = NULL;
  PetscBool          BisCUDA = PETSC_FALSE, XisCUDA = PETSC_FALSE;
  cudaStream_t       stream;

  PetscFunctionBegin;
  /* cuDSS is a sequential solver; only accept sequential dense CUDA types */
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSECUDA, &BisCUDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSECUDA, &XisCUDA));
  /* Convert B to a temporary CUDA matrix to avoid mutating the caller's matrix */
  if (!BisCUDA) PetscCall(MatConvert(B, MATDENSECUDA, MAT_INITIAL_MATRIX, &Bcuda));
  else Bcuda = B;
  /* Convert X to a temporary CUDA matrix to avoid mutating the caller's matrix type */
  if (!XisCUDA) PetscCall(MatConvert(X, MATDENSECUDA, MAT_INITIAL_MATRIX, &Xcuda));
  else Xcuda = X;
  PetscCall(MatGetSize(Bcuda, &n, &nrhs));
  PetscCall(MatGetSize(Xcuda, &nX, &nrhsX));
  PetscCheck(n == nX && nrhs == nrhsX, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible matrix dimensions: B is %" PetscInt_FMT " x %" PetscInt_FMT ", X is %" PetscInt_FMT " x %" PetscInt_FMT, n, nrhs, nX, nrhsX);
  PetscCall(MatDenseGetLDA(Bcuda, &ldb));
  PetscCall(MatDenseGetLDA(Xcuda, &ldx));
  PetscCall(MatDenseCUDAGetArrayRead(Bcuda, &barray));
  PetscCall(MatDenseCUDAGetArrayWrite(Xcuda, &xarray));
  PetscCallCUDSS(cudssMatrixCreateDn, &cudss_B, n, nrhs, ldb, (void *)barray, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallCUDSS(cudssMatrixCreateDn, &cudss_X, n, nrhs, ldx, xarray, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  /* Ensure cuDSS uses the current PETSc stream for solve */
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);
  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, cudss_X, cudss_B);
  PetscCallCUDSS(cudssMatrixDestroy, cudss_B);
  PetscCallCUDSS(cudssMatrixDestroy, cudss_X);
  PetscCall(MatDenseCUDARestoreArrayRead(Bcuda, &barray));
  PetscCall(MatDenseCUDARestoreArrayWrite(Xcuda, &xarray));
  if (!BisCUDA) PetscCall(MatDestroy(&Bcuda));
  if (!XisCUDA) {
    PetscCall(MatCopy(Xcuda, X, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Xcuda));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu    = (Mat_cuDSS *)F->data;
  void        *d_row = NULL, *d_col = NULL;
  PetscScalar *d_val = NULL;
  cudaStream_t stream;

  PetscFunctionBegin;
  /* Refresh device CSR values (structure unchanged, values may have changed) */
  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, &d_row, &d_col, &d_val));

  /* Update the values pointer in the cuDSS matrix descriptor */
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_A, d_val);

  /* Ensure cuDSS uses the current PETSc stream for factorization */
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);

  /* Numeric factorization */
  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_FACTORIZATION, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  F->ops->solve    = MatSolve_cuDSS;
  F->ops->matsolve = MatMatSolve_cuDSS;
  F->assembled     = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu    = (Mat_cuDSS *)F->data;
  void        *d_row = NULL, *d_col = NULL;
  PetscScalar *d_val = NULL;
  cudaStream_t stream;

  PetscFunctionBegin;
  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, &d_row, &d_col, &d_val));
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_A, d_val);
  /* Ensure cuDSS uses the current PETSc stream for factorization */
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);
  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_FACTORIZATION, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  F->ops->solve    = MatSolve_cuDSS;
  F->ops->matsolve = MatMatSolve_cuDSS;
  F->assembled     = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_cudss(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCUDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERCUDSS = "cudss" - A solver package providing LU and Cholesky factorization for
  sequential sparse matrices via the NVIDIA cuDSS GPU-accelerated sparse direct solver library.

  Use `./configure --download-cudss` or `./configure --with-cudss-dir=<dir>` to have PETSc use cuDSS.

  Use `-pc_type lu -pc_factor_mat_solver_type cudss` to use this direct solver.
  Use `-pc_type cholesky -pc_factor_mat_solver_type cudss` for SPD matrices.

  Options Database Keys:
+ -mat_cudss_reorder_alg default    - (choose one of) `default`, `alg1`, `alg2`, `alg3`, `alg4`, `alg5`
. -mat_cudss_pivot_type col         - (choose one of) `col`, `row`, `none`
. -mat_cudss_pivot_threshold 1.0    - Pivot threshold
. -mat_cudss_pivot_epsilon 0.0      - Pivot epsilon
. -mat_cudss_use_matching false     - Enable matching
- -mat_cudss_ir_n_steps 0           - Number of iterative refinement steps

  Level: beginner

  Registered for both `MATSEQAIJ` (host) and `MATSEQAIJCUSPARSE` (device) matrix types.
  When the input matrix is `MATSEQAIJ`, the CSR data is transparently copied to the GPU.

  Note:
    `MatSolveTranspose()` is not supported.

    cuDSS performs its own internal reordering during the symbolic phase. Any PETSc-supplied
    row, column, or Cholesky permutation (`IS r`, `IS c`, `IS perm`) passed to
    `MatLUFactorSymbolic()` or `MatCholeskyFactorSymbolic()` is silently ignored; cuDSS
    selects the reordering algorithm via `-mat_cudss_reorder_alg`.

    `MatSolve()` requires CUDA-aware vectors (`VECCUDA` / `VECSEQCUDA`). Using plain host
    `VECSEQ` vectors with this solver will result in an error. When the input matrix is
    `MATSEQAIJ`, ensure that the right-hand-side and solution vectors are of type
    `VECCUDA` (e.g., created with `VecSetType(v, VECCUDA)`).

.seealso: [](ch_matrices), `Mat`, `PCLU`, `PCCHOLESKY`, `PCFactorSetMatSolverType()`, `MatSolverType`
M*/

static PetscErrorCode MatGetFactor_seqaij_cudss(Mat A, MatFactorType ftype, Mat *F)
{
  Mat        B;
  Mat_cuDSS *lu;
  PetscInt   m = A->rmap->n, n = A->cmap->n;

  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported by cuDSS");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(PetscStrallocpy("cudss", &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));
  B->trivialsymbolic = PETSC_FALSE;
  B->factortype      = ftype;
  B->assembled       = PETSC_TRUE;
  B->preallocated    = PETSC_TRUE;

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_cuDSS;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_cuDSS;
  }
  B->ops->destroy = MatDestroy_cuDSS;
  B->ops->view    = MatView_cuDSS;
  B->ops->getinfo = MatGetInfo_External;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDSS, &B->solvertype));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_cudss));

  PetscCall(PetscNew(&lu));
  /* Set default options */
  lu->reorderAlg     = CUDSS_ALG_DEFAULT;
  lu->pivotType      = CUDSS_PIVOT_COL;
  lu->pivotThreshold = 1.0;
  lu->pivotEpsilon   = 0.0;
  lu->useMatching    = 0; /* CUDSS_CONFIG_USE_MATCHING disabled by default */
  lu->irNSteps       = 0;
  B->data            = lu;

  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_cuDSS(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_seqaij_cudss));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_cudss));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJCUSPARSE, MAT_FACTOR_LU, MatGetFactor_seqaij_cudss));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDSS, MATSEQAIJCUSPARSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_cudss));
  PetscFunctionReturn(PETSC_SUCCESS);
}
