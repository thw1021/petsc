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

  /* User-supplied permutation (host array of 32-bit ints, owned by us) */
  PetscInt *h_user_perm;
  PetscBool userPermSet;

  PetscInt  n; /* matrix dimension */
  PetscInt  nnz;
  PetscBool ownDeviceCSR;

  cudssReorderingAlg_t reorderAlg;
  cudssPivotType_t     pivotType;
  double               pivotThreshold; /* cuDSS expects double for CUDSS_CONFIG_PIVOT_THRESHOLD */
  double               pivotEpsilon;   /* cuDSS expects double for CUDSS_CONFIG_PIVOT_EPSILON */
  int                  useMatching;    /* cuDSS expects int, translated to CUDSS_CONFIG_MATCHING_ALG */
  int                  irNSteps;       /* cuDSS expects int for CUDSS_CONFIG_IR_N_STEPS */
} Mat_cuDSS;

/* Map PetscScalar to the cuDSS data type */
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define CUDSS_SCALAR_TYPE CUDSS_C_32F
  #else
    #define CUDSS_SCALAR_TYPE CUDSS_C_64F
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define CUDSS_SCALAR_TYPE CUDSS_R_32F
  #else
    #define CUDSS_SCALAR_TYPE CUDSS_R_64F
  #endif
#endif

/* Custom macro: cuDSS returns cudssStatus_t, not a PetscErrorCode, so
   PetscCallExternal() cannot be used here.  Follows the same convention
   as PetscCallCUBLAS / PetscCallCUSPARSE. */
#define PetscCallCUDSS(...) \
  do { \
    cudssStatus_t cudss_status_ = __VA_ARGS__; \
    PetscCheck(cudss_status_ == CUDSS_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuDSS error %d in %s", (int)cudss_status_, PetscStringize(__VA_ARGS__)); \
  } while (0)

/* These arrays are indexed by the cuDSS enum values directly.
   Mapping (cuDSS 0.8.0):
     cudssReorderingAlg_t: CUDSS_REORDERING_ALG_DEFAULT==0, ..._BTF_COLAMD==1, ..._COLAMD==2,
                           ..._AMD==3, ..._NESTED_DISSECTION==4, ..._NONE==5
     cudssPivotType_t: CUDSS_PIVOT_AUTO==0, ..._NONE==1, ..._GLOBAL_COL==2, ..._GLOBAL_ROW==3,
                       ..._DIAGONAL==4, ..._LOCAL_BLOCK==5, ..._BUNCH_KAUFMAN==6
   If NVIDIA renumbers these enumerators in a future release, the option parsing
   and view output will silently mismap; update both arrays and this comment. */
static const char *const MatCUDSSReorderAlgs[] = {"default", "btf_colamd", "colamd", "amd", "nested_dissection", "none"};
static const char *const MatCUDSSPivotTypes[]  = {"auto", "none", "col", "row", "diagonal", "local_block", "bunch_kaufman"};

/* Forward declarations */
static PetscErrorCode MatFactorNumeric_cuDSS(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatSetFromOptions_cuDSS(Mat);
static PetscErrorCode MatFactorSymbolic_cuDSS(Mat, Mat, cudssMatrixType_t, cudssMatrixViewType_t);
static PetscErrorCode MatCUDSSSetUserPermutation_cuDSS(Mat, IS);

static PetscErrorCode MatView_Info_cuDSS(Mat A, PetscViewer viewer)
{
  Mat_cuDSS  *lu          = (Mat_cuDSS *)A->data;
  const char *reorderName = ((int)lu->reorderAlg >= 0 && (size_t)lu->reorderAlg < PETSC_STATIC_ARRAY_LENGTH(MatCUDSSReorderAlgs)) ? MatCUDSSReorderAlgs[lu->reorderAlg] : "unknown";
  const char *pivotName   = ((int)lu->pivotType >= 0 && (size_t)lu->pivotType < PETSC_STATIC_ARRAY_LENGTH(MatCUDSSPivotTypes)) ? MatCUDSSPivotTypes[lu->pivotType] : "unknown";

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "cuDSS run parameters:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Reorder algorithm: %s\n", reorderName));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  User permutation: %s\n", lu->userPermSet ? "yes" : "no"));
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
  if (iascii == PETSC_TRUE) {
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
  if (lu->cudss_x != NULL) PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_x));
  if (lu->cudss_b != NULL) PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_b));
  if (lu->cudss_A != NULL) PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_A));
  if (lu->data != NULL) PetscCallCUDSS(cudssDataDestroy(lu->handle, lu->data));
  if (lu->config != NULL) PetscCallCUDSS(cudssConfigDestroy(lu->config));
  if (lu->handle != NULL) PetscCallCUDSS(cudssDestroy(lu->handle));
  if (lu->ownDeviceCSR == PETSC_TRUE) {
    PetscCallCUDA(cudaFree(lu->d_row_offsets));
    PetscCallCUDA(cudaFree(lu->d_col_indices));
    PetscCallCUDA(cudaFree(lu->d_values));
  }
  PetscCall(PetscFree(lu->h_user_perm));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCUDSSSetUserPermutation_C", NULL));
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
  PetscCall(PetscOptionsEList("-mat_cudss_reorder_alg", "Reordering algorithm", "None", MatCUDSSReorderAlgs, PETSC_STATIC_ARRAY_LENGTH(MatCUDSSReorderAlgs), MatCUDSSReorderAlgs[reorderAlg], &reorderAlg, NULL));
  lu->reorderAlg = (cudssReorderingAlg_t)reorderAlg;
  PetscCall(PetscOptionsEList("-mat_cudss_pivot_type", "Pivot type", "None", MatCUDSSPivotTypes, PETSC_STATIC_ARRAY_LENGTH(MatCUDSSPivotTypes), MatCUDSSPivotTypes[pivotType], &pivotType, NULL));
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
  cudssMatchingAlg_t matchingAlg = lu->useMatching ? CUDSS_MATCHING_ALG_AUTO : CUDSS_MATCHING_ALG_NONE;

  PetscFunctionBegin;
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_REORDERING_ALG, &lu->reorderAlg, sizeof(lu->reorderAlg)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_PIVOT_TYPE, &lu->pivotType, sizeof(lu->pivotType)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_PIVOT_THRESHOLD, &lu->pivotThreshold, sizeof(lu->pivotThreshold)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_PIVOT_EPSILON, &lu->pivotEpsilon, sizeof(lu->pivotEpsilon)));
  /* Matching is controlled via CUDSS_CONFIG_MATCHING_ALG (CUDSS_MATCHING_ALG_NONE disables it). */
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_MATCHING_ALG, &matchingAlg, sizeof(matchingAlg)));
  PetscCallCUDSS(cudssConfigSet(lu->config, CUDSS_CONFIG_IR_N_STEPS, &lu->irNSteps, sizeof(lu->irNSteps)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Ensure the CSR data for matrix A is resident on the device.
   For MATSEQAIJCUSPARSE, the existing device arrays (always 32-bit integers) are
   returned via d_row_32/d_col_32.  For plain MATSEQAIJ, the PetscInt device arrays
   owned by lu are populated via cudaMemcpyAsync and returned via d_row_pi/d_col_pi.
   Exactly one of {d_row_32, d_col_32} and {d_row_pi, d_col_pi} will be set on return;
   the other pair is left unchanged (callers initialise them to NULL before the call).
   d_val is always set.
   When valuesOnly is PETSC_TRUE (numeric-refactor path), the row-offset and column-index
   arrays are already on the device from the symbolic phase and are not re-uploaded;
   only the value array is copied.  The d_row and d_col output pointers are still set
   to the existing device buffers so callers can use them uniformly. */
static PetscErrorCode MatEnsureOnDevice_cuDSS(Mat A, Mat_cuDSS *lu, PetscBool valuesOnly, int **d_row_32, int **d_col_32, PetscInt **d_row_pi, PetscInt **d_col_pi, PetscScalar **d_val)
{
  PetscBool    isCUSPARSE;
  cudaStream_t stream;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  if (isCUSPARSE == PETSC_TRUE) {
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
    {
      Mat_SeqAIJCUSPARSE           *cusparsestruct = (Mat_SeqAIJCUSPARSE *)A->spptr;
      Mat_SeqAIJCUSPARSEMultStruct *matstruct      = (Mat_SeqAIJCUSPARSEMultStruct *)cusparsestruct->mat;
      CsrMatrix                    *csr            = (CsrMatrix *)matstruct->mat;
      *d_row_32                                    = thrust::raw_pointer_cast(csr->row_offsets->data());
      *d_col_32                                    = thrust::raw_pointer_cast(csr->column_indices->data());
      *d_val                                       = (PetscScalar *)thrust::raw_pointer_cast(csr->values->data());
    }
  } else {
    Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
    PetscCall(PetscGetCurrentCUDAStream(&stream));
    if (valuesOnly == PETSC_FALSE) {
      /* Symbolic phase: upload structure (row offsets + column indices) and values */
      PetscCallCUDA(cudaMemcpyAsync(lu->d_row_offsets, a->i, (lu->n + 1) * sizeof(PetscInt), cudaMemcpyHostToDevice, stream));
      PetscCallCUDA(cudaMemcpyAsync(lu->d_col_indices, a->j, lu->nnz * sizeof(PetscInt), cudaMemcpyHostToDevice, stream));
    }
    /* Numeric phase (valuesOnly==PETSC_TRUE): row offsets and column indices are
       unchanged since the symbolic phase; only upload the new values. */
    PetscCallCUDA(cudaMemcpyAsync(lu->d_values, a->a, lu->nnz * sizeof(PetscScalar), cudaMemcpyHostToDevice, stream));
    *d_row_pi = lu->d_row_offsets;
    *d_col_pi = lu->d_col_indices;
    *d_val    = lu->d_values;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_cuDSS(Mat F, Mat A, cudssMatrixType_t mtype, cudssMatrixViewType_t mview)
{
  Mat_cuDSS   *lu = (Mat_cuDSS *)F->data;
  Mat_SeqAIJ  *a  = (Mat_SeqAIJ *)A->data;
  PetscInt     m, n, nnz;
  int         *d_row_32 = NULL, *d_col_32 = NULL; /* MATSEQAIJCUSPARSE path: always 32-bit */
  PetscInt    *d_row_pi = NULL, *d_col_pi = NULL; /* MATSEQAIJ path: PetscInt width */
  PetscScalar *d_val = NULL;
  PetscBool    isCUSPARSE;
  cudaStream_t stream;

  PetscFunctionBegin;
  if (lu->cudss_x != NULL) {
    PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_x));
    lu->cudss_x = NULL;
  }
  if (lu->cudss_b != NULL) {
    PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_b));
    lu->cudss_b = NULL;
  }
  if (lu->cudss_A != NULL) {
    PetscCallCUDSS(cudssMatrixDestroy(lu->cudss_A));
    lu->cudss_A = NULL;
  }
  if (lu->data != NULL) {
    PetscCallCUDSS(cudssDataDestroy(lu->handle, lu->data));
    lu->data = NULL;
  }
  if (lu->config != NULL) {
    PetscCallCUDSS(cudssConfigDestroy(lu->config));
    lu->config = NULL;
  }
  if (lu->handle != NULL) {
    PetscCallCUDSS(cudssDestroy(lu->handle));
    lu->handle = NULL;
  }
  if (lu->ownDeviceCSR == PETSC_TRUE) {
    PetscCallCUDA(cudaFree(lu->d_row_offsets));
    lu->d_row_offsets = NULL;
    PetscCallCUDA(cudaFree(lu->d_col_indices));
    lu->d_col_indices = NULL;
    PetscCallCUDA(cudaFree(lu->d_values));
    lu->d_values = NULL;
  }
  m   = A->rmap->n;
  n   = A->cmap->n;
  nnz = a->nz;

  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "cuDSS requires a square matrix, got %" PetscInt_FMT "x%" PetscInt_FMT, m, n);

  lu->n   = n;
  lu->nnz = nnz;

  PetscCallCUDSS(cudssCreate(&lu->handle));
  PetscCallCUDSS(cudssConfigCreate(&lu->config));
  PetscCallCUDSS(cudssDataCreate(lu->handle, &lu->data));
  PetscCall(MatSetFromOptions_cuDSS(F));
  PetscCall(MatApplyConfig_cuDSS(lu));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJCUSPARSE, &isCUSPARSE));
  lu->ownDeviceCSR = (PetscBool)(isCUSPARSE == PETSC_FALSE);
  if (isCUSPARSE == PETSC_FALSE) {
    PetscCallCUDA(cudaMalloc((void **)&lu->d_row_offsets, (m + 1) * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_col_indices, nnz * sizeof(PetscInt)));
    PetscCallCUDA(cudaMalloc((void **)&lu->d_values, nnz * sizeof(PetscScalar)));
  }

  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));

  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, PETSC_FALSE, &d_row_32, &d_col_32, &d_row_pi, &d_col_pi, &d_val));
  {
    void           *d_row    = isCUSPARSE ? (void *)d_row_32 : (void *)d_row_pi;
    void           *d_col    = isCUSPARSE ? (void *)d_col_32 : (void *)d_col_pi;
    cudssDataType_t idx_type = CUDSS_R_32I; /* cuDSS requires 32-bit indices (cuDSS.py: requires32bitint=1), so PetscInt is 32-bit on both the MATSEQAIJ and MATSEQAIJCUSPARSE paths */
    PetscCallCUDSS(cudssMatrixCreateCsr(&lu->cudss_A, m, n, nnz, d_row, NULL, d_col, d_val, idx_type, idx_type, CUDSS_SCALAR_TYPE, mtype, mview, CUDSS_BASE_ZERO));

    PetscCallCUDSS(cudssMatrixCreateDn(&lu->cudss_b, n, 1, n, NULL, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
    PetscCallCUDSS(cudssMatrixCreateDn(&lu->cudss_x, n, 1, n, NULL, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
  }
  /* If the user supplied a permutation, pass it to cuDSS before analysis.
     cudssDataSet() takes (handle, data, CUDSS_DATA_USER_PERM, host_int_ptr, sizeInBytes). */
  if (lu->userPermSet == PETSC_TRUE) {
    PetscCallCUDSS(cudssDataSet(lu->handle, lu->data, CUDSS_DATA_USER_PERM, lu->h_user_perm, (size_t)n * sizeof(PetscInt)));
    PetscCall(PetscInfo(F, "cuDSS: using user-supplied permutation of size %" PetscInt_FMT "\n", n));
  }
  /* Analysis phase: reordering and symbolic factorization */
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_ANALYSIS, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_cuDSS(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (r != NULL) {
    if (c != NULL) PetscCall(PetscInfo(F, "cuDSS accepts only a single permutation; using row ordering IS, column ordering IS is ignored\n"));
    PetscCall(MatCUDSSSetUserPermutation_cuDSS(F, r));
  } else if (c != NULL) PetscCall(MatCUDSSSetUserPermutation_cuDSS(F, c));
  PetscCall(MatFactorSymbolic_cuDSS(F, A, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL));
  F->ops->lufactornumeric = MatFactorNumeric_cuDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_cuDSS(Mat F, Mat A, IS perm, const MatFactorInfo *info)
{
  cudssMatrixType_t mtype;

  PetscFunctionBegin;
  if (perm != NULL) PetscCall(MatCUDSSSetUserPermutation_cuDSS(F, perm));
#if defined(PETSC_USE_COMPLEX)
  if (A->spd == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_HPD;
  else if (A->hermitian == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_HERMITIAN;
  else mtype = CUDSS_MTYPE_SYMMETRIC;
#else
  if (A->spd == PETSC_BOOL3_TRUE) mtype = CUDSS_MTYPE_SPD;
  else mtype = CUDSS_MTYPE_SYMMETRIC;
#endif
  PetscCall(MatFactorSymbolic_cuDSS(F, A, mtype, CUDSS_MVIEW_UPPER));
  F->ops->choleskyfactornumeric = MatFactorNumeric_cuDSS;
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
  PetscCallCUDSS(cudssMatrixSetValues(lu->cudss_b, (void *)barray));
  PetscCallCUDSS(cudssMatrixSetValues(lu->cudss_x, xarray));
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSECUDA, &BisCUDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)X, MATSEQDENSECUDA, &XisCUDA));
  if (BisCUDA == PETSC_FALSE) {
    PetscCall(PetscInfo(F, "Converting B from host to MATDENSECUDA; consider using MATSEQDENSECUDA for better performance\n"));
    PetscCall(MatConvert(B, MATDENSECUDA, MAT_INITIAL_MATRIX, &Bcuda));
  } else Bcuda = B;
  if (XisCUDA == PETSC_FALSE) {
    PetscInt m_x, N_x;

    PetscCall(MatGetSize(X, &m_x, &N_x));
    PetscCall(MatCreateDenseCUDA(PetscObjectComm((PetscObject)X), m_x, N_x, m_x, N_x, NULL, &Xcuda));
  } else Xcuda = X;
  PetscCall(MatGetSize(Bcuda, &n, &nrhs));
  PetscCall(MatGetSize(Xcuda, &nX, &nrhsX));
  PetscCheck(n == nX && nrhs == nrhsX, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible matrix dimensions: B is %" PetscInt_FMT " x %" PetscInt_FMT ", X is %" PetscInt_FMT " x %" PetscInt_FMT, n, nrhs, nX, nrhsX);
  PetscCheck(n == lu->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "B row count %" PetscInt_FMT " does not match factored matrix size %" PetscInt_FMT, n, lu->n);
  PetscCall(MatDenseGetLDA(Bcuda, &ldb));
  PetscCall(MatDenseGetLDA(Xcuda, &ldx));
  PetscCall(MatDenseCUDAGetArrayRead(Bcuda, &barray));
  PetscCall(MatDenseCUDAGetArrayWrite(Xcuda, &xarray));
  PetscCallCUDSS(cudssMatrixCreateDn(&cudss_B, n, nrhs, ldb, (void *)barray, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
  PetscCallCUDSS(cudssMatrixCreateDn(&cudss_X, n, nrhs, ldx, xarray, CUDSS_SCALAR_TYPE, CUDSS_LAYOUT_COL_MAJOR));
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_SOLVE, lu->config, lu->data, lu->cudss_A, cudss_X, cudss_B));
  PetscCallCUDSS(cudssMatrixDestroy(cudss_B));
  PetscCallCUDSS(cudssMatrixDestroy(cudss_X));
  PetscCall(MatDenseCUDARestoreArrayRead(Bcuda, &barray));
  PetscCall(MatDenseCUDARestoreArrayWrite(Xcuda, &xarray));
  if (BisCUDA == PETSC_FALSE) PetscCall(MatDestroy(&Bcuda));
  if (XisCUDA == PETSC_FALSE) {
    PetscCall(MatCopy(Xcuda, X, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&Xcuda));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorNumeric_cuDSS(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_cuDSS   *lu       = (Mat_cuDSS *)F->data;
  int         *d_row_32 = NULL, *d_col_32 = NULL;
  PetscInt    *d_row_pi = NULL, *d_col_pi = NULL;
  PetscScalar *d_val = NULL;
  cudaStream_t stream;

  PetscFunctionBegin;
  /* Row offsets and column indices are unchanged since the symbolic phase;
     only upload the new numeric values to the device. */
  PetscCall(MatEnsureOnDevice_cuDSS(A, lu, PETSC_TRUE, &d_row_32, &d_col_32, &d_row_pi, &d_col_pi, &d_val));
  PetscCallCUDSS(cudssMatrixSetValues(lu->cudss_A, d_val));
  PetscCall(PetscGetCurrentCUDAStream(&stream));
  PetscCallCUDSS(cudssSetStream(lu->handle, stream));
  PetscCallCUDSS(cudssExecute(lu->handle, CUDSS_PHASE_FACTORIZATION, lu->config, lu->data, lu->cudss_A, lu->cudss_x, lu->cudss_b));
  F->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_cudss(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCUDSS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatCUDSSSetUserPermutation_cuDSS - Store a user-supplied permutation for use during the
  next symbolic factorization phase.

  Input Parameters:
+ F    - the factor matrix (obtained from MatGetFactor())
- perm - an IS of length n containing a 0-based permutation of {0,...,n-1}

  The permutation is copied to a host buffer of 32-bit ints and passed to cudssDataSet()
  with CUDSS_DATA_USER_PERM before CUDSS_PHASE_ANALYSIS.
*/
static PetscErrorCode MatCUDSSSetUserPermutation_cuDSS(Mat F, IS perm)
{
  Mat_cuDSS      *lu = (Mat_cuDSS *)F->data;
  PetscInt        n, i;
  const PetscInt *idx;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(perm, &n));
  PetscCall(ISGetIndices(perm, &idx));
  /* (Re-)allocate host buffer (cudssDataSet takes a host pointer) */
  PetscCall(PetscFree(lu->h_user_perm));
  PetscCall(PetscMalloc1(n, &lu->h_user_perm));
  for (i = 0; i < n; i++) lu->h_user_perm[i] = idx[i];
  PetscCall(ISRestoreIndices(perm, &idx));
  lu->userPermSet = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCUDSSSetUserPermutation - Supply a user-defined reordering permutation to the cuDSS
  sparse direct solver.

  Not Collective

  Input Parameters:
+ F    - the factor matrix obtained from `MatGetFactor()` with solver type `MATSOLVERCUDSS`
- perm - an `IS` of length n containing a 0-based permutation of {0,...,n-1}

  Level: advanced

  Notes:
  The permutation is applied during the next call to `MatLUFactorSymbolic()` or
  `MatCholeskyFactorSymbolic()`.  It overrides cuDSS's internal reordering for that
  symbolic phase.  The `IS` may be destroyed after this call returns.

  Alternatively, pass the `IS` directly as the row-permutation argument to
  `MatLUFactorSymbolic()` or as the permutation argument to
  `MatCholeskyFactorSymbolic()`; both routes call this function internally.

.seealso: [](ch_matrices), `Mat`, `MATSOLVERCUDSS`, `MatLUFactorSymbolic()`, `MatCholeskyFactorSymbolic()`
@*/
PetscErrorCode MatCUDSSSetUserPermutation(Mat F, IS perm)
{
  PetscErrorCode (*f)(Mat, IS);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(perm, IS_CLASSID, 2);
  PetscCall(PetscObjectQueryFunction((PetscObject)F, "MatCUDSSSetUserPermutation_C", &f));
  PetscCheck(f, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONG, "Mat is not a cuDSS factor matrix");
  PetscCall((*f)(F, perm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERCUDSS = "cudss" - A solver package providing LU and Cholesky factorization for
  sequential sparse matrices via the NVIDIA cuDSS GPU-accelerated sparse direct solver library.

  Options Database Keys:
+ -mat_cudss_reorder_alg (default|btf_colamd|colamd|amd|nested_dissection|none) - reordering algorithm
. -mat_cudss_pivot_type (auto|none|col|row|diagonal|local_block|bunch_kaufman) - pivoting type
. -mat_cudss_pivot_threshold threshold                      - Pivot threshold, default is 1.0
. -mat_cudss_pivot_epsilon epsilon                          - Pivot epsilon, default is 0.0
. -mat_cudss_use_matching flag                              - Enable matching, default is false
- -mat_cudss_ir_n_steps nsteps                              - Number of iterative refinement steps, default is 0

  Level: intermediate

  Notes:
    Registered for both `MATSEQAIJ` (host) and `MATSEQAIJCUSPARSE` (device) matrix types.
    When the input matrix is `MATSEQAIJ`, the CSR data is transparently copied to the GPU.
    `MatSolveTranspose()` is not supported. The `bunch_kaufman` value of `-mat_cudss_pivot_type` is
    reserved in cuDSS and not yet supported; selecting it causes cuDSS to fail at factorization time.

    By default cuDSS performs its own internal reordering during the symbolic phase.
    A user-supplied permutation can be provided via `MatCUDSSSetUserPermutation()`, or by
    passing a non-NULL `IS` to `MatLUFactorSymbolic()` or `MatCholeskyFactorSymbolic()`.
    Select the automatic reordering algorithm via `-mat_cudss_reorder_alg`.

    `MatSolve()` requires CUDA-aware vectors (`VECCUDA` / `VECSEQCUDA`). Using plain host
    `VECSEQ` vectors with this solver will result in an error. When the input matrix is
    `MATSEQAIJ`, ensure that the right-hand-side and solution vectors are of type
    `VECCUDA` (e.g., created with `VecSetType(v, VECCUDA)`).

.seealso: [](ch_matrices), `Mat`, `PCLU`, `PCCHOLESKY`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatCUDSSSetUserPermutation()`
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
  PetscCall(PetscStrallocpy(MATSOLVERCUDSS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));
  B->trivialsymbolic = PETSC_FALSE;
  B->factortype      = ftype;
  B->canuseordering  = PETSC_FALSE;
  B->assembled       = PETSC_TRUE;
  B->preallocated    = PETSC_TRUE;

  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_cuDSS;
  else B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_cuDSS;
  B->ops->solve    = MatSolve_cuDSS;
  B->ops->matsolve = MatMatSolve_cuDSS;
  B->ops->destroy  = MatDestroy_cuDSS;
  B->ops->view     = MatView_cuDSS;
  B->ops->getinfo  = MatGetInfo_External;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCUDSS, &B->solvertype));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_cudss));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatCUDSSSetUserPermutation_C", MatCUDSSSetUserPermutation_cuDSS));

  PetscCall(PetscNew(&lu));
  lu->reorderAlg     = CUDSS_REORDERING_ALG_DEFAULT;
  lu->pivotType      = CUDSS_PIVOT_AUTO;
  lu->pivotThreshold = 1.0;
  lu->pivotEpsilon   = 0.0;
  lu->useMatching    = 0;
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
