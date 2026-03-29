/*
     This file implements a subclass of the SeqAIJ matrix class that uses
     the NVIDIA cuDSS GPU-accelerated sparse direct solver library.
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <petsc/private/vecimpl.h>
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

  /* Whether the analysis phase has been run */
  PetscBool analysisCompleted;

  /* Solver options - use cuDSS enum types directly for cudssConfigSet compatibility */
  cudssAlgType_t   reorderAlg;     /* CUDSS_CONFIG_REORDERING_ALG */
  cudssPivotType_t pivotType;      /* CUDSS_CONFIG_PIVOT_TYPE */
  double           pivotThreshold; /* CUDSS_CONFIG_PIVOT_THRESHOLD (always double) */
  double           pivotEpsilon;   /* CUDSS_CONFIG_PIVOT_EPSILON (always double) */
  int              useMatching;    /* CUDSS_CONFIG_USE_MATCHING (plain int: 0=false, 1=true) */
  int              irNSteps;       /* CUDSS_CONFIG_IR_N_STEPS (int) */

  /* Flag to clean up cuDSS objects during Destroy */
  PetscBool CleanUp;
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

/* Map PetscInt to the cuDSS index type */
#if defined(PETSC_USE_64BIT_INDICES)
  #define CUDSS_INDEX_TYPE CUDA_R_64I
  #define CUDSS_INDEX_BASE CUDSS_BASE_ZERO
#else
  #define CUDSS_INDEX_TYPE CUDA_R_32I
  #define CUDSS_INDEX_BASE CUDSS_BASE_ZERO
#endif

#define PetscCallcuDSS(func, ...) \
  do { \
    cudssStatus_t cudss_status_ = func(__VA_ARGS__); \
    PetscCheck(cudss_status_ == CUDSS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error %d in %s", (int)cudss_status_, PetscStringize(func)); \
  } while (0)

/* Forward declarations */
static PetscErrorCode MatLUFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);

static PetscErrorCode MatView_Info_cuDSS(Mat A, PetscViewer viewer)
{
  Mat_cuDSS *lu = (Mat_cuDSS *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "cuDSS run parameters:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Reorder algorithm: %d\n", (int)lu->reorderAlg));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot type: %d\n", (int)lu->pivotType));
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
  if (lu->CleanUp) {
    if (lu->cudss_x) PetscCallcuDSS(cudssMatrixDestroy, lu->cudss_x);
    if (lu->cudss_b) PetscCallcuDSS(cudssMatrixDestroy, lu->cudss_b);
    if (lu->cudss_A) PetscCallcuDSS(cudssMatrixDestroy, lu->cudss_A);
    if (lu->data) PetscCallcuDSS(cudssDataDestroy, lu->handle, lu->data);
    if (lu->config) PetscCallcuDSS(cudssConfigDestroy, lu->config);
    if (lu->handle) PetscCallcuDSS(cudssDestroy, lu->handle);
    if (lu->ownDeviceCSR) {
      PetscCallCUDA(cudaFree(lu->d_row_offsets));
      PetscCallCUDA(cudaFree(lu->d_col_indices));
      PetscCallCUDA(cudaFree(lu->d_values));
    }
    if (lu->d_b) PetscCallCUDA(cudaFree(lu->d_b));
    if (lu->d_x) PetscCallCUDA(cudaFree(lu->d_x));
  }
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Helper: apply cuDSS config options from the Mat_cuDSS struct to the cudssConfig_t object.
*/
static PetscErrorCode MatcuDSSApplyConfig(Mat_cuDSS *lu)
{
  PetscFunctionBegin;
  PetscCallcuDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_REORDERING_ALG, &lu->reorderAlg, sizeof(lu->reorderAlg));
  PetscCallcuDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_PIVOT_TYPE, &lu->pivotType, sizeof(lu->pivotType));
  PetscCallcuDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_PIVOT_THRESHOLD, &lu->pivotThreshold, sizeof(lu->pivotThreshold));
  PetscCallcuDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_PIVOT_EPSILON, &lu->pivotEpsilon, sizeof(lu->pivotEpsilon));
  PetscCallcuDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_USE_MATCHING, &lu->useMatching, sizeof(lu->useMatching));
  PetscCallcuDSS(cudssConfigSet, lu->config, CUDSS_CONFIG_IR_N_STEPS, &lu->irNSteps, sizeof(lu->irNSteps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Helper: ensure CSR data is on the device.
   For MATSEQAIJCUSPARSE: use MatSeqAIJCUSPARSECopyToGPU and get device pointers.
   For MATSEQAIJ: copy host arrays to lu->d_row_offsets/d_col_indices/d_values.
   Returns device pointers in d_row, d_col, d_val (not owned by caller).
*/
static PetscErrorCode MatcuDSSEnsureOnDevice(Mat A, Mat_cuDSS *lu, PetscInt **d_row, PetscInt **d_col, PetscScalar **d_val)
{
  PetscBool isCUSPARSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  if (isCUSPARSE) {
    /* Ensure data is current on device */
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
    {
      Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE *)A->spptr;
      Mat_SeqAIJCUSPARSEMultStruct *matstruct      = (Mat_SeqAIJCUSPARSEMultStruct *)cusparsestruct->mat;
      CsrMatrix                    *csr            = (CsrMatrix *)matstruct->mat;
      *d_row                                       = (PetscInt *)thrust::raw_pointer_cast(csr->row_offsets->data());
      *d_col                                       = (PetscInt *)thrust::raw_pointer_cast(csr->column_indices->data());
      *d_val                                       = (PetscScalar *)thrust::raw_pointer_cast(csr->values->data());
    }
  } else {
    /* MATSEQAIJ: copy host CSR to device */
    Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
    PetscCallCUDA(cudaMemcpy(lu->d_row_offsets, a->i, (lu->n + 1) * sizeof(PetscInt), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(lu->d_col_indices, a->j, lu->nnz * sizeof(PetscInt), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(lu->d_values, a->a, lu->nnz * sizeof(PetscScalar), cudaMemcpyHostToDevice));
    *d_row = lu->d_row_offsets;
    *d_col = lu->d_col_indices;
    *d_val = lu->d_values;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_cuDSS(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu = (Mat_cuDSS *)F->data;
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)A->data;
  PetscInt     m, n, nnz;
  PetscInt    *d_row, *d_col;
  PetscScalar *d_val;
  PetscBool    isCUSPARSE;

  PetscFunctionBegin;
  m   = A->rmap->n;
  n   = A->cmap->n;
  nnz = a->nz;

  lu->n   = n;
  lu->nnz = nnz;

  /* Process options - use PetscInt temporaries then cast to cuDSS enum types */
  {
    PetscInt  reorderAlg     = (PetscInt)lu->reorderAlg;
    PetscInt  pivotType      = (PetscInt)lu->pivotType;
    PetscReal pivotThreshold = (PetscReal)lu->pivotThreshold;
    PetscReal pivotEpsilon   = (PetscReal)lu->pivotEpsilon;
    PetscInt  irNSteps       = (PetscInt)lu->irNSteps;
    PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "cuDSS Options", "Mat");
    PetscCall(PetscOptionsInt("-mat_cudss_reorder_alg", "Reordering algorithm (0=default,1-5)", "None", reorderAlg, &reorderAlg, NULL));
    PetscCall(PetscOptionsInt("-mat_cudss_pivot_type", "Pivot type (0=col,1=row,2=none)", "None", pivotType, &pivotType, NULL));
    PetscCall(PetscOptionsReal("-mat_cudss_pivot_threshold", "Pivot threshold", "None", pivotThreshold, &pivotThreshold, NULL));
    PetscCall(PetscOptionsReal("-mat_cudss_pivot_epsilon", "Pivot epsilon", "None", pivotEpsilon, &pivotEpsilon, NULL));
    {
      PetscBool useMatchingBool = (PetscBool)lu->useMatching;
      PetscCall(PetscOptionsBool("-mat_cudss_use_matching", "Enable matching (CUDSS_CONFIG_USE_MATCHING)", "None", useMatchingBool, &useMatchingBool, NULL));
      lu->useMatching = (int)useMatchingBool;
    }
    PetscCall(PetscOptionsInt("-mat_cudss_ir_n_steps", "Number of iterative refinement steps", "None", irNSteps, &irNSteps, NULL));
    PetscOptionsEnd();
    lu->reorderAlg     = (cudssAlgType_t)reorderAlg;
    lu->pivotType      = (cudssPivotType_t)pivotType;
    lu->pivotThreshold = (double)pivotThreshold;
    lu->pivotEpsilon   = (double)pivotEpsilon;
    lu->irNSteps       = (int)irNSteps;
  }

  /* Initialize cuDSS handle, config, data */
  PetscCallcuDSS(cudssCreate, &lu->handle);
  PetscCallcuDSS(cudssConfigCreate, &lu->config);
  PetscCallcuDSS(cudssDataCreate, lu->handle, &lu->data);

  /* Apply config options */
  PetscCall(MatcuDSSApplyConfig(lu));

  /* Determine if input is CUSPARSE */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  lu->ownDeviceCSR = (PetscBool)(!isCUSPARSE);

  if (!isCUSPARSE) {
    /* Allocate device CSR arrays */
    PetscCallCUDA(cudaMalloc((void **)&lu->d_row_offsets, (m + 1) * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_col_indices, nnz * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_values, nnz * sizeof(PetscScalar)));
  }

  /* Set PETSc CUDA stream on the cuDSS handle */
  {
    cudaStream_t stream;
    PetscCall(PetscGetCurrentCUDAStream(&stream));
    PetscCallcuDSS(cudssSetStream, lu->handle, stream);
  }

  /* Ensure CSR data is on device */
  PetscCall(MatcuDSSEnsureOnDevice(A, lu, &d_row, &d_col, &d_val));

  /* Create cuDSS sparse matrix descriptor (general for LU) */
  PetscCallcuDSS(cudssMatrixCreateCsr, &lu->cudss_A, m, n, nnz, d_row, NULL, d_col, d_val, CUDSS_INDEX_TYPE, CUDSS_SCALAR_TYPE, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);

  /* Allocate scratch device buffers so descriptors have a valid initial pointer */
  PetscCallCUDA(cudaMalloc((void **)&lu->d_b, n * sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&lu->d_x, n * sizeof(PetscScalar)));

  /* Create dense RHS and solution descriptors; pointer updated per-solve via cudssMatrixSetValues */
  PetscCallcuDSS(cudssMatrixCreateDn, &lu->cudss_b, n, 1, n, lu->d_b, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallcuDSS(cudssMatrixCreateDn, &lu->cudss_x, n, 1, n, lu->d_x, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);

  /* Run analysis (symbolic factorization / reordering) */
  PetscCallcuDSS(cudssExecute, lu->handle, CUDSS_PHASE_ANALYSIS, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  lu->analysisCompleted   = PETSC_TRUE;
  lu->CleanUp             = PETSC_TRUE;
  F->ops->lufactornumeric = MatLUFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_cuDSS(Mat F, Mat A, IS perm, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu = (Mat_cuDSS *)F->data;
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)A->data;
  PetscInt     m, n, nnz;
  PetscInt    *d_row, *d_col;
  PetscScalar *d_val;
  PetscBool    isCUSPARSE;

  PetscFunctionBegin;
  m   = A->rmap->n;
  n   = A->cmap->n;
  nnz = a->nz;

  lu->n   = n;
  lu->nnz = nnz;

  /* Process options - use PetscInt temporaries then cast to cuDSS enum types */
  {
    PetscInt  reorderAlg     = (PetscInt)lu->reorderAlg;
    PetscInt  pivotType      = (PetscInt)lu->pivotType;
    PetscReal pivotThreshold = (PetscReal)lu->pivotThreshold;
    PetscReal pivotEpsilon   = (PetscReal)lu->pivotEpsilon;
    PetscInt  irNSteps       = (PetscInt)lu->irNSteps;
    PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "cuDSS Options", "Mat");
    PetscCall(PetscOptionsInt("-mat_cudss_reorder_alg", "Reordering algorithm (0=default,1-5)", "None", reorderAlg, &reorderAlg, NULL));
    PetscCall(PetscOptionsInt("-mat_cudss_pivot_type", "Pivot type (0=col,1=row,2=none)", "None", pivotType, &pivotType, NULL));
    PetscCall(PetscOptionsReal("-mat_cudss_pivot_threshold", "Pivot threshold", "None", pivotThreshold, &pivotThreshold, NULL));
    PetscCall(PetscOptionsReal("-mat_cudss_pivot_epsilon", "Pivot epsilon", "None", pivotEpsilon, &pivotEpsilon, NULL));
    {
      PetscBool useMatchingBool = (PetscBool)lu->useMatching;
      PetscCall(PetscOptionsBool("-mat_cudss_use_matching", "Enable matching (CUDSS_CONFIG_USE_MATCHING)", "None", useMatchingBool, &useMatchingBool, NULL));
      lu->useMatching = (int)useMatchingBool;
    }
    PetscCall(PetscOptionsInt("-mat_cudss_ir_n_steps", "Number of iterative refinement steps", "None", irNSteps, &irNSteps, NULL));
    PetscOptionsEnd();
    lu->reorderAlg     = (cudssAlgType_t)reorderAlg;
    lu->pivotType      = (cudssPivotType_t)pivotType;
    lu->pivotThreshold = (double)pivotThreshold;
    lu->pivotEpsilon   = (double)pivotEpsilon;
    lu->irNSteps       = (int)irNSteps;
  }

  /* Initialize cuDSS handle, config, data */
  PetscCallcuDSS(cudssCreate, &lu->handle);
  PetscCallcuDSS(cudssConfigCreate, &lu->config);
  PetscCallcuDSS(cudssDataCreate, lu->handle, &lu->data);

  /* Apply config options */
  PetscCall(MatcuDSSApplyConfig(lu));

  /* Determine if input is CUSPARSE */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  lu->ownDeviceCSR = (PetscBool)(!isCUSPARSE);

  if (!isCUSPARSE) {
    PetscCallCUDA(cudaMalloc((void **)&lu->d_row_offsets, (m + 1) * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_col_indices, nnz * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_values, nnz * sizeof(PetscScalar)));
  }

  /* Set PETSc CUDA stream on the cuDSS handle */
  {
    cudaStream_t stream;
    PetscCall(PetscGetCurrentCUDAStream(&stream));
    PetscCallcuDSS(cudssSetStream, lu->handle, stream);
  }

  PetscCall(MatcuDSSEnsureOnDevice(A, lu, &d_row, &d_col, &d_val));

  /* Create cuDSS sparse matrix descriptor (SPD for Cholesky) */
  PetscCallcuDSS(cudssMatrixCreateCsr, &lu->cudss_A, m, n, nnz, d_row, NULL, d_col, d_val, CUDSS_INDEX_TYPE, CUDSS_SCALAR_TYPE, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO);

  /* Allocate scratch device buffers so descriptors have a valid initial pointer */
  PetscCallCUDA(cudaMalloc((void **)&lu->d_b, n * sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&lu->d_x, n * sizeof(PetscScalar)));

  /* Create dense RHS and solution descriptors; pointer updated per-solve via cudssMatrixSetValues */
  PetscCallcuDSS(cudssMatrixCreateDn, &lu->cudss_b, n, 1, n, lu->d_b, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallcuDSS(cudssMatrixCreateDn, &lu->cudss_x, n, 1, n, lu->d_x, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);

  /* Run analysis */
  PetscCallcuDSS(cudssExecute, lu->handle, CUDSS_PHASE_ANALYSIS, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  lu->analysisCompleted         = PETSC_TRUE;
  lu->CleanUp                   = PETSC_TRUE;
  F->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_cuDSS(Mat F, Vec b, Vec x)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArrayRead(b, &barray));
  PetscCall(VecCUDAGetArrayWrite(x, &xarray));
  PetscCallcuDSS(cudssMatrixSetValues, lu->cudss_b, (void *)barray);
  PetscCallcuDSS(cudssMatrixSetValues, lu->cudss_x, xarray);
  PetscCallcuDSS(cudssExecute, lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);
  PetscCall(VecCUDARestoreArrayRead(b, &barray));
  PetscCall(VecCUDARestoreArrayWrite(x, &xarray));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_cuDSS(Mat F, Mat B, Mat X)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  PetscInt           n, nrhs, ldb, ldx;
  cudssMatrix_t      cudss_B = NULL, cudss_X = NULL;
  PetscBool          Bneedconv = PETSC_FALSE, Xneedconv = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &Bneedconv, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &Xneedconv, MATSEQDENSE, MATMPIDENSE, ""));
  if (Bneedconv) PetscCall(MatConvert(B, MATDENSECUDA, MAT_INPLACE_MATRIX, &B));
  if (Xneedconv) PetscCall(MatConvert(X, MATDENSECUDA, MAT_INPLACE_MATRIX, &X));
  PetscCall(MatGetSize(B, &n, &nrhs));
  PetscCall(MatDenseGetLDA(B, &ldb));
  PetscCall(MatDenseGetLDA(X, &ldx));
  PetscCall(MatDenseCUDAGetArrayRead(B, &barray));
  PetscCall(MatDenseCUDAGetArrayWrite(X, &xarray));
  PetscCallcuDSS(cudssMatrixCreateDn, &cudss_B, n, nrhs, ldb, (void *)barray, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallcuDSS(cudssMatrixCreateDn, &cudss_X, n, nrhs, ldx, xarray, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallcuDSS(cudssExecute, lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, cudss_X, cudss_B);
  PetscCallcuDSS(cudssMatrixDestroy, cudss_B);
  PetscCallcuDSS(cudssMatrixDestroy, cudss_X);
  PetscCall(MatDenseCUDARestoreArrayRead(B, &barray));
  PetscCall(MatDenseCUDARestoreArrayWrite(X, &xarray));
  if (Bneedconv) PetscCall(MatConvert(B, MATDENSE, MAT_INPLACE_MATRIX, &B));
  if (Xneedconv) PetscCall(MatConvert(X, MATDENSE, MAT_INPLACE_MATRIX, &X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu    = (Mat_cuDSS *)F->data;
  PetscInt    *d_row = NULL, *d_col = NULL;
  PetscScalar *d_val = NULL;

  PetscFunctionBegin;
  /* Refresh device CSR values (structure unchanged, values may have changed) */
  PetscCall(MatcuDSSEnsureOnDevice(A, lu, &d_row, &d_col, &d_val));

  /* Update the values pointer in the cuDSS matrix descriptor */
  PetscCallcuDSS(cudssMatrixSetValues, lu->cudss_A, d_val);

  /* Numeric factorization */
  PetscCallcuDSS(cudssExecute, lu->handle, CUDSS_PHASE_FACTORIZATION, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  F->ops->solve          = MatSolve_cuDSS;
  F->ops->solvetranspose = MatSolve_cuDSS; /* cuDSS 0.7.x does not yet support CUDSS_CONFIG_SOLVE_MODE */
  F->ops->matsolve       = MatMatSolve_cuDSS;
  F->assembled           = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu    = (Mat_cuDSS *)F->data;
  PetscInt    *d_row = NULL, *d_col = NULL;
  PetscScalar *d_val = NULL;

  PetscFunctionBegin;
  PetscCall(MatcuDSSEnsureOnDevice(A, lu, &d_row, &d_col, &d_val));
  PetscCallcuDSS(cudssMatrixSetValues, lu->cudss_A, d_val);
  PetscCallcuDSS(cudssExecute, lu->handle, CUDSS_PHASE_FACTORIZATION, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  F->ops->solve          = MatSolve_cuDSS;
  F->ops->solvetranspose = MatSolve_cuDSS;
  F->ops->matsolve       = MatMatSolve_cuDSS;
  F->assembled           = PETSC_TRUE;
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
+ -mat_cudss_reorder_alg <int>      - Reordering algorithm (cudssAlgType_t)
. -mat_cudss_pivot_type <int>       - Pivot type (0=classic, 1=nopivot)
. -mat_cudss_pivot_threshold <real> - Pivot threshold
. -mat_cudss_pivot_epsilon <real>   - Pivot epsilon
. -mat_cudss_use_matching <bool>    - Enable matching (CUDSS_CONFIG_USE_MATCHING, default: false)
- -mat_cudss_ir_n_steps <int>       - Number of iterative refinement steps

  Level: beginner

  Notes:
  cuDSS is a single-GPU solver. For MPI-parallel applications, use `PCREDUNDANT` or
  gather the matrix to a single process before factorization.

  Registered for both `MATSEQAIJ` (host) and `MATSEQAIJCUSPARSE` (device) matrix types.
  When the input matrix is `MATSEQAIJ`, the CSR data is transparently copied to the GPU.

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
  lu->CleanUp        = PETSC_FALSE;
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
