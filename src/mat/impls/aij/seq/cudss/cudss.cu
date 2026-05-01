/*
     This file implements a subclass of the SeqAIJ matrix class that uses
     the NVIDIA cuDSS GPU-accelerated sparse direct solver library.
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <petscdevice_cuda.h>
#include <cudss.h>

/* Internal data structure for the cuDSS factored matrix */
typedef struct {
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t   data;
  cudssMatrix_t cudss_A; /* sparse matrix descriptor */
  cudssMatrix_t cudss_b; /* dense RHS descriptor */
  cudssMatrix_t cudss_x; /* dense solution descriptor */

  /* Device CSR arrays (owned by us when input is MATSEQAIJ) */
  PetscInt    *d_row_offsets;
  PetscInt    *d_col_indices;
  PetscScalar *d_values;

  /* Device scratch buffers for RHS/solution (size n) */
  PetscScalar *d_b;
  PetscScalar *d_x;

  PetscInt  n; /* matrix dimension */
  PetscInt  nnz;
  PetscBool ownDeviceCSR;
  int64_t   lu_nnz; /* cached LU factor nnz for flop logging */

  /* cuDSS index type: CUDA_R_32I for MATSEQAIJCUSPARSE, PETSCINT_CUDSS_INDEX_TYPE for MATSEQAIJ */
  cudaDataType indexType;

  cudssAlgType_t   reorderAlg;
  cudssPivotType_t pivotType;
  double           pivotThreshold; /* cuDSS expects double for CUDSS_CONFIG_PIVOT_THRESHOLD */
  double           pivotEpsilon;   /* cuDSS expects double for CUDSS_CONFIG_PIVOT_EPSILON */
  int              useMatching;    /* cuDSS expects int for CUDSS_CONFIG_USE_MATCHING */
  int              irNSteps;       /* cuDSS expects int for CUDSS_CONFIG_IR_N_STEPS */

  MatStructure matstruc; /* tracks whether symbolic factorization can be reused */
  PetscBool    factored; /* true after first successful numeric factorization */
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

/* cuSPARSE CsrMatrix always stores row_offsets/column_indices as 32-bit int */
#define CUSPARSE_CSR_INDEX_TYPE CUDA_R_32I

#define PetscCallCUDSS(func, ...) \
  do { \
    cudssStatus_t cudss_status_ = func(__VA_ARGS__); \
    PetscCheck(cudss_status_ == CUDSS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error %d in %s", (int)cudss_status_, PetscStringize(func)); \
  } while (0)

static const char *const MatcuDSSReorderAlgs[] = {"default", "alg1", "alg2", "alg3", "alg4", "alg5"};
static const char *const MatcuDSSPivotTypes[]  = {"col", "row", "none"};

/* Forward declarations */
static PetscErrorCode MatLUFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatCholeskyFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatSetFromOptions_cuDSS(Mat);
static PetscErrorCode MatFactorSymbolic_cuDSS(Mat, Mat, cudssMatrixType_t, cudssMatrixViewType_t);

static PetscErrorCode MatView_Info_cuDSS(Mat A, PetscViewer viewer)
{
  Mat_cuDSS  *lu          = (Mat_cuDSS *)A->data;
  const char *reorderName = ((int)lu->reorderAlg >= 0 && (size_t)lu->reorderAlg < PETSC_STATIC_ARRAY_LENGTH(MatcuDSSReorderAlgs)) ? MatcuDSSReorderAlgs[lu->reorderAlg] : "unknown";
  const char *pivotName   = ((int)lu->pivotType >= 0 && (size_t)lu->pivotType < PETSC_STATIC_ARRAY_LENGTH(MatcuDSSPivotTypes)) ? MatcuDSSPivotTypes[lu->pivotType] : "unknown";

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "cuDSS run parameters:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Reorder algorithm: %s\n", reorderName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot type: %s\n", pivotName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot threshold: %g\n", (double)lu->pivotThreshold));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Pivot epsilon: %g\n", (double)lu->pivotEpsilon));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Use matching: %s\n", lu->useMatching ? "true" : "false"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  IR steps: %d\n", lu->irNSteps));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Symbolic reuse: %s\n", lu->matstruc == SAME_NONZERO_PATTERN ? "true (analysis reused)" : "false (fresh analysis)"));
  /* Query factorization statistics if available */
  if (lu->handle && lu->data) {
    int64_t lu_nnz  = 0;
    int     npivots = 0;
    size_t  written = 0;
    if (cudssDataGet(lu->handle, lu->data, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), &written) == CUDSS_STATUS_SUCCESS && written > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "  LU factor nnz: %" PetscInt64_FMT "\n", (PetscInt64)lu_nnz));
    if (cudssDataGet(lu->handle, lu->data, CUDSS_DATA_NPIVOTS, &npivots, sizeof(npivots), &written) == CUDSS_STATUS_SUCCESS && written > 0) PetscCall(PetscViewerASCIIPrintf(viewer, "  Number of pivots: %d\n", npivots));
  }
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
  PetscBool  useMatching = (PetscBool)lu->useMatching;

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
  PetscCall(PetscOptionsBool("-mat_cudss_use_matching", "Enable matching", "None", useMatching, &useMatching, NULL));
  lu->useMatching = (int)useMatching;
  PetscCall(PetscOptionsInt("-mat_cudss_ir_n_steps", "Number of iterative refinement steps", "None", irNSteps, &irNSteps, NULL));
  lu->irNSteps = (int)irNSteps;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

static PetscErrorCode MatEnsureOnDevice_cuDSS(Mat A, Mat_cuDSS *lu, void **d_row, void **d_col, PetscScalar **d_val)
{
  PetscBool    isCUSPARSE;
  cudaStream_t stream;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  if (isCUSPARSE) {
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
  /* If the sparsity pattern has not changed, skip the symbolic phase entirely.
     The subsequent numeric factorization will use CUDSS_PHASE_REFACTORIZATION
     to reuse the existing symbolic analysis. This mirrors the MUMPS approach. */
  if (lu->matstruc == SAME_NONZERO_PATTERN) {
    PetscCall(PetscInfo(F, "cuDSS: reusing previous symbolic factorization (same nonzero pattern)\n"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

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

  PetscCallCUDSS(cudssCreate, &lu->handle);
  PetscCallCUDSS(cudssConfigCreate, &lu->config);
  PetscCallCUDSS(cudssDataCreate, lu->handle, &lu->data);
  PetscCall(MatApplyConfig_cuDSS(lu));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  lu->ownDeviceCSR = (PetscBool)(!isCUSPARSE);
  if (isCUSPARSE) {
    lu->indexType = CUSPARSE_CSR_INDEX_TYPE;
  } else {
    lu->indexType = PETSCINT_CUDSS_INDEX_TYPE;
    PetscCallCUDA(cudaMalloc((void **)&lu->d_row_offsets, (m + 1) * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_col_indices, nnz * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_values, nnz * sizeof(PetscScalar)));
  }

  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);

  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, &d_row, &d_col, &d_val));
  PetscCallCUDSS(cudssMatrixCreateCsr, &lu->cudss_A, m, n, nnz, d_row, NULL, d_col, d_val, lu->indexType, CUDSS_SCALAR_TYPE, mtype, mview, CUDSS_BASE_ZERO);

  PetscCallCUDA(cudaMalloc((void **)&lu->d_b, n * sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&lu->d_x, n * sizeof(PetscScalar)));
  PetscCallCUDSS(cudssMatrixCreateDn, &lu->cudss_b, n, 1, n, lu->d_b, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);
  PetscCallCUDSS(cudssMatrixCreateDn, &lu->cudss_x, n, 1, n, lu->d_x, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR);

  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_ANALYSIS, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);

  lu->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_cuDSS(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (r || c) PetscCall(PetscInfo(F, "cuDSS ignores user-supplied row/column permutations; cuDSS will apply its own reordering\n"));
  PetscCall(MatFactorSymbolic_cuDSS(F, A, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL));
  F->ops->lufactornumeric = MatLUFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(PETSC_USE_COMPLEX)
static PetscErrorCode MatGetInertia_cuDSS(Mat F, PetscInt *nneg, PetscInt *nzero, PetscInt *npos)
{
  Mat_cuDSS *lu         = (Mat_cuDSS *)F->data;
  int        inertia[3] = {0, 0, 0}; /* positive, negative, zero */
  size_t     written    = 0;

  PetscFunctionBegin;
  PetscCheck(lu->handle && lu->data, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "cuDSS factorization has not been performed");
  PetscCallCUDSS(cudssDataGet, lu->handle, lu->data, CUDSS_DATA_INERTIA, inertia, sizeof(inertia), &written);
  if (npos) *npos = inertia[0];
  if (nneg) *nneg = inertia[1];
  if (nzero) *nzero = inertia[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatCholeskyFactorSymbolic_cuDSS(Mat F, Mat A, IS perm, const MatFactorInfo *info)
{
  cudssMatrixType_t mtype;

  PetscFunctionBegin;
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
#if !defined(PETSC_USE_COMPLEX)
  F->ops->getinertia = MatGetInertia_cuDSS;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_cuDSS(Mat F, Vec b, Vec x)
{
  Mat_cuDSS         *lu = (Mat_cuDSS *)F->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  cudaStream_t       stream;
  PetscBool          iscuda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)b, &iscuda, VECSEQCUDA, VECMPICUDA, VECCUDA, ""));
  PetscCheck(iscuda, PetscObjectComm((PetscObject)b), PETSC_ERR_ARG_WRONG, "cuDSS requires CUDA vectors (VECCUDA). Use -vec_type cuda or call MatCreateVecs() to create vectors from the matrix");
  PetscCall(VecCUDAGetArrayRead(b, &barray));
  PetscCall(VecCUDAGetArrayWrite(x, &xarray));
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_b, (void *)barray);
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_x, xarray);
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);
  PetscCallCUDSS(cudssExecute, lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);
  PetscCall(VecCUDARestoreArrayRead(b, &barray));
  PetscCall(VecCUDARestoreArrayWrite(x, &xarray));
  PetscCall(PetscLogGpuFlops(2.0 * lu->lu_nnz));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSECUDA, &BisCUDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSECUDA, &XisCUDA));
  if (!BisCUDA) PetscCall(MatConvert(B, MATDENSECUDA, MAT_INITIAL_MATRIX, &Bcuda));
  else Bcuda = B;
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
  PetscCall(PetscLogGpuFlops(2.0 * nrhs * lu->lu_nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorNumeric_cuDSS(Mat F, Mat A)
{
  Mat_cuDSS    *lu    = (Mat_cuDSS *)F->data;
  void         *d_row = NULL, *d_col = NULL;
  PetscScalar  *d_val = NULL;
  cudaStream_t  stream;
  cudssStatus_t status;
  cudssPhase_t  phase;

  PetscFunctionBegin;
  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, &d_row, &d_col, &d_val));
  PetscCallCUDSS(cudssMatrixSetValues, lu->cudss_A, d_val);
  PetscCall(MatApplyConfig_cuDSS(lu));
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream, lu->handle, stream);

  /* Use CUDSS_PHASE_REFACTORIZATION when reusing a previous symbolic analysis,
     otherwise use CUDSS_PHASE_FACTORIZATION for the first numeric factorization */
  if (lu->factored) phase = CUDSS_PHASE_REFACTORIZATION;
  else phase = CUDSS_PHASE_FACTORIZATION;
  PetscCall(PetscInfo(F, "cuDSS: using %s phase\n", phase == CUDSS_PHASE_REFACTORIZATION ? "REFACTORIZATION" : "FACTORIZATION"));

  status = cudssExecute(lu->handle, phase, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b);
  if (status != CUDSS_STATUS_SUCCESS) {
    PetscCall(PetscInfo(F, "cuDSS numerical factorization failed with status %d\n", (int)status));
    if (status == CUDSS_STATUS_ALLOC_FAILED) F->factorerrortype = MAT_FACTOR_OUTMEMORY;
    else F->factorerrortype = MAT_FACTOR_OTHER;
    PetscCheck(!A->erroriffailure, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error in numerical factorization: status %d", (int)status);
  } else {
    size_t written     = 0;
    F->factorerrortype = MAT_FACTOR_NOERROR;
    F->ops->solve      = MatSolve_cuDSS;
    F->ops->matsolve   = MatMatSolve_cuDSS;
    F->assembled       = PETSC_TRUE;
    lu->factored       = PETSC_TRUE;
    /* Cache LU factor nnz for flop logging */
    if (cudssDataGet(lu->handle, lu->data, CUDSS_DATA_LU_NNZ, &lu->lu_nnz, sizeof(lu->lu_nnz), &written) != CUDSS_STATUS_SUCCESS || written == 0) lu->lu_nnz = 0;
    PetscCall(PetscLogGpuFlops(lu->lu_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatFactorNumeric_cuDSS(F, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatFactorNumeric_cuDSS(F, A));
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

  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_cuDSS;
  else B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_cuDSS;
  B->ops->destroy = MatDestroy_cuDSS;
  B->ops->view    = MatView_cuDSS;
  B->ops->getinfo = MatGetInfo_External;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDSS, &B->solvertype));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_cudss));

  PetscCall(PetscNew(&lu));
  lu->reorderAlg     = CUDSS_ALG_DEFAULT;
  lu->pivotType      = CUDSS_PIVOT_COL;
  lu->pivotThreshold = 1.0;
  lu->pivotEpsilon   = 0.0;
  lu->useMatching    = PETSC_FALSE;
  lu->irNSteps       = 0;
  lu->matstruc       = DIFFERENT_NONZERO_PATTERN;
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
