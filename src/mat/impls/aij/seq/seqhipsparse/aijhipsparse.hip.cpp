/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format using the ROCSPARSE library,
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_ROCMWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/vecimpl.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>

const char *const MatHIPSPARSEStorageFormats[]    = {"CSR","ELL","HYB","MatHIPSPARSEStorageFormat","MAT_HIPSPARSE_",0};

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,const MatFactorInfo*);
static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSE(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat,Mat,const MatFactorInfo*);

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering(Mat,Vec,Vec);
static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(PetscOptionItems *PetscOptionsObject,Mat);
static PetscErrorCode MatAXPY_SeqAIJHIPSPARSE(Mat,PetscScalar,Mat,MatStructure);
static PetscErrorCode MatScale_SeqAIJHIPSPARSE(Mat,PetscScalar);
static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultHermitianTranspose_SeqAIJHIPSPARSE(Mat,Vec,Vec);
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE(Mat,Vec,Vec,Vec);
static PetscErrorCode MatMultAddKernel_SeqAIJHIPSPARSE(Mat,Vec,Vec,Vec,PetscBool,PetscBool);

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix**);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSETriFactorStruct**);
static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct**,MatHIPSPARSEStorageFormat);
static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Reset(Mat_SeqAIJHIPSPARSETriFactors**);
static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors**);
static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat_SeqAIJHIPSPARSE**);

static PetscErrorCode MatSeqAIJHIPSPARSECopyToGPU(Mat);
static PetscErrorCode MatSeqAIJHIPSPARSECopyFromGPU(Mat);
static PetscErrorCode MatSeqAIJHIPSPARSEInvalidateTranspose(Mat,PetscBool);

PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat,PetscInt,const PetscInt[],const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat,const PetscScalar[],InsertMode);

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJHIPSPARSE(Mat,PetscInt,const PetscInt[],PetscScalar[]);

PetscErrorCode MatHIPSPARSESetStream(Mat A,const hipStream_t stream)
{
  rocsparseStatus_t   stat;
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (!hipsparsestruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  hipsparsestruct->stream = stream;
  stat = hipsparseSetStream(hipsparsestruct->handle,hipsparsestruct->stream);CHKERRHIPSPARSE(stat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHIPSPARSESetHandle(Mat A,const rocsparse_handle handle)
{
  rocsparseStatus_t   stat;
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  if (!hipsparsestruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (hipsparsestruct->handle != handle) {
    if (hipsparsestruct->handle) {
      stat = hipsparseDestroy(hipsparsestruct->handle);CHKERRHIPSPARSE(stat);
    }
    hipsparsestruct->handle = handle;
  }
  stat = hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE);CHKERRHIPSPARSE(stat);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHIPSPARSEClearHandle(Mat A)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg || !hipsparsestruct) PetscFunctionReturn(0);
  if (hipsparsestruct->handle) hipsparsestruct->handle = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_hipsparse(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERHIPSPARSE;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERHIPSPARSE = "hipsparse" - A matrix type providing triangular solvers for seq matrices
  on a single GPU of type, seqaijhipsparse, aijhipsparse. Currently supported
  algorithms are ILU(k) and ICC(k). Typically, deeper factorizations (larger k) results in poorer
  performance in the triangular solves. Full LU, and Cholesky decompositions can be solved through the
  HIPSPARSE triangular solve algorithm. However, the performance can be quite poor and thus these
  algorithms are not recommended. This class does NOT support direct solver operations.

  Level: beginner

.seealso: PCFactorSetMatSolverType(), MatSolverType, MatCreateSeqAIJHIPSPARSE(), MATAIJHIPSPARSE, MatCreateAIJHIPSPARSE(), MatHIPSPARSESetFormat(), MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  (*B)->useordering = PETSC_TRUE;
  ierr = MatSetType(*B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJHIPSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJHIPSPARSE;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJHIPSPARSE;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for HIPSPARSE Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_hipsparse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatHIPSPARSESetFormat_SeqAIJHIPSPARSE(Mat A,MatHIPSPARSEFormatOperation op,MatHIPSPARSEStorageFormat format)
{
  Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_HIPSPARSE_MULT:
    hipsparsestruct->format = format;
    break;
  case MAT_HIPSPARSE_ALL:
    hipsparsestruct->format = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatHIPSPARSEFormatOperation. MAT_HIPSPARSE_MULT and MAT_HIPSPARSE_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

/*@
   MatHIPSPARSESetFormat - Sets the storage format of HIPSPARSE matrices for a particular
   operation. Only the MatMult operation can use different GPU storage formats
   for MPIAIJHIPSPARSE matrices.
   Not Collective

   Input Parameters:
+  A - Matrix of type SEQAIJHIPSPARSE
.  op - MatHIPSPARSEFormatOperation. SEQAIJHIPSPARSE matrices support MAT_HIPSPARSE_MULT and MAT_HIPSPARSE_ALL. MPIAIJHIPSPARSE matrices support MAT_HIPSPARSE_MULT_DIAG, MAT_HIPSPARSE_MULT_OFFDIAG, and MAT_HIPSPARSE_ALL.
-  format - MatHIPSPARSEStorageFormat (one of MAT_HIPSPARSE_CSR, MAT_HIPSPARSE_ELL, MAT_HIPSPARSE_HYB.)

   Output Parameter:

   Level: intermediate

.seealso: MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
@*/
PetscErrorCode MatHIPSPARSESetFormat(Mat A,MatHIPSPARSEFormatOperation op,MatHIPSPARSEStorageFormat format)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID,1);
  ierr = PetscTryMethod(A,"MatHIPSPARSESetFormat_C",(Mat,MatHIPSPARSEFormatOperation,MatHIPSPARSEStorageFormat),(A,op,format));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatSeqAIJHIPSPARSESetGenerateTranspose - Sets the flag to explicitly generate the transpose matrix before calling MatMultTranspose

   Collective on mat

   Input Parameters:
+  A - Matrix of type SEQAIJHIPSPARSE
-  transgen - the boolean flag

   Level: intermediate

.seealso: MATSEQAIJHIPSPARSE, MatAIJHIPSPARSESetGenerateTranspose()
@*/
PetscErrorCode MatSeqAIJHIPSPARSESetGenerateTranspose(Mat A,PetscBool transgen)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare(((PetscObject)A),MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (flg) {
    Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;

    if (A->factortype) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
    hipsp->transgen = transgen;
    if (!transgen) { /* need to destroy the transpose matrix if present to prevent from logic errors if transgen is set to true later */
      ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqAIJHIPSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode           ierr;
  MatHIPSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_SeqAIJHIPSPARSE       *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqAIJHIPSPARSE options");CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscBool transgen = hipsparsestruct->transgen;

    ierr = PetscOptionsBool("-mat_hipsparse_transgen","Generate explicit transpose for MatMultTranspose","MatSeqAIJHIPSPARSESetGenerateTranspose",transgen,&transgen,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatSeqAIJHIPSPARSESetGenerateTranspose(A,transgen);CHKERRQ(ierr);}

    ierr = PetscOptionsEnum("-mat_hipsparse_mult_storage_format","sets storage format of (seq)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_MULT,format);CHKERRQ(ierr);}

    ierr = PetscOptionsEnum("-mat_hipsparse_storage_format","sets storage format of (seq)aijhipsparse gpu matrices for SpMV and TriSolve",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparsestruct->format,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_ALL,format);CHKERRQ(ierr);}
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors);CHKERRQ(ierr);
  ierr = MatILUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors);CHKERRQ(ierr);
  ierr = MatICCFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJHIPSPARSE(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)B->spptr;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic_SeqAIJ(B,A,perm,info);CHKERRQ(ierr);
  B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildILULowerTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  rocsparseStatus_t                  stat;
  const PetscInt                    *ai = a->i,*aj = a->j,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiLo, *AjLo;
  PetscInt                          i,nz, nzLower, offset, rowOffset;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* first figure out the number of nonzeros in the lower triangular matrix including 1's on the diagonal. */
      nzLower=n+ai[n]-ai[1];
      if (!loTriFactor) {
        PetscScalar                       *AALo;

        cerr = hipHostMalloc((void**) &AALo, nzLower*sizeof(PetscScalar));CHKERRHIP(cerr);

        /* Allocate Space for the lower triangular matrix */
        cerr = hipHostMalloc((void**) &AiLo, (n+1)*sizeof(PetscInt));CHKERRHIP(cerr);
        cerr = hipHostMalloc((void**) &AjLo, nzLower*sizeof(PetscInt));CHKERRHIP(cerr);

        /* Fill the lower triangular matrix */
        AiLo[0]  = (PetscInt) 0;
        AiLo[n]  = nzLower;
        AjLo[0]  = (PetscInt) 0;
        AALo[0]  = (MatScalar) 1.0;
        v        = aa;
        vi       = aj;
        offset   = 1;
        rowOffset= 1;
        for (i=1; i<n; i++) {
          nz = ai[i+1] - ai[i];
          /* additional 1 for the term on the diagonal */
          AiLo[i]    = rowOffset;
          rowOffset += nz+1;

          ierr = PetscArraycpy(&(AjLo[offset]), vi, nz);CHKERRQ(ierr);
          ierr = PetscArraycpy(&(AALo[offset]), v, nz);CHKERRQ(ierr);

          offset      += nz;
          AjLo[offset] = (PetscInt) i;
          AALo[offset] = (MatScalar) 1.0;
          offset      += 1;

          v  += nz;
          vi += nz;
        }

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&loTriFactor);CHKERRQ(ierr);
        loTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
        /* Create the matrix description */
        stat = rocsparse_create_mat_descr(&loTriFactor->descr);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_index_base(loTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_type(loTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_fill_mode(loTriFactor->descr, HIPSPARSE_FILL_MODE_LOWER);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_diag_type(loTriFactor->descr, HIPSPARSE_DIAG_TYPE_UNIT);CHKERRHIPSPARSE(stat);

        /* set the operation */
        loTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        /* set the matrix */
        loTriFactor->csrMat = new CsrMatrix;
        loTriFactor->csrMat->num_rows = n;
        loTriFactor->csrMat->num_cols = n;
        loTriFactor->csrMat->num_entries = nzLower;

        loTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(n+1);
        loTriFactor->csrMat->row_offsets->assign(AiLo, AiLo+n+1);

        loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzLower);
        loTriFactor->csrMat->column_indices->assign(AjLo, AjLo+nzLower);

        loTriFactor->csrMat->values = new THRUSTARRAY(nzLower);
        loTriFactor->csrMat->values->assign(AALo, AALo+nzLower);

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = hipsparse_create_analysis_info(&loTriFactor->solveInfo);CHKERRHIPSPARSE(stat);
        stat = hipsparse_get_svbuffsize(hipsparseTriFactors->handle, loTriFactor->solveOp,
                                       loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                       loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                       loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo,
                                       &loTriFactor->solveBufferSize);CHKERRHIPSPARSE(stat);
        cerr = hipMalloc(&loTriFactor->solveBuffer,loTriFactor->solveBufferSize);CHKERRHIP(cerr);

        /* perform the solve analysis */
        stat = hipsparse_analysis(hipsparseTriFactors->handle, loTriFactor->solveOp,
                                 loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                 loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                 loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo
                                 ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);
        cerr = WaitForHIP();CHKERRHIP(cerr);
        ierr = PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;
        loTriFactor->AA_h = AALo;
        cerr = hipHostFree(AiLo);CHKERRHIP(cerr);
        cerr = hipHostFree(AjLo);CHKERRHIP(cerr);
        ierr = PetscLogCpuToGpu((n+1+nzLower)*sizeof(int)+nzLower*sizeof(PetscScalar));CHKERRQ(ierr);
      } else { /* update values only */
        if (!loTriFactor->AA_h) {
          cerr = hipHostMalloc((void**) &loTriFactor->AA_h, nzLower*sizeof(PetscScalar));CHKERRHIP(cerr);
        }
        /* Fill the lower triangular matrix */
        loTriFactor->AA_h[0]  = 1.0;
        v        = aa;
        vi       = aj;
        offset   = 1;
        for (i=1; i<n; i++) {
          nz = ai[i+1] - ai[i];
          ierr = PetscArraycpy(&(loTriFactor->AA_h[offset]), v, nz);CHKERRQ(ierr);
          offset      += nz;
          loTriFactor->AA_h[offset] = 1.0;
          offset      += 1;
          v  += nz;
        }
        loTriFactor->csrMat->values->assign(loTriFactor->AA_h, loTriFactor->AA_h+nzLower);
        ierr = PetscLogCpuToGpu(nzLower*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildILUUpperTriMatrix(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  PetscInt                          n = A->rmap->n;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  rocsparseStatus_t                  stat;
  const PetscInt                    *aj = a->j,*adiag = a->diag,*vi;
  const MatScalar                   *aa = a->a,*v;
  PetscInt                          *AiUp, *AjUp;
  PetscInt                          i,nz, nzUpper, offset;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      /* next, figure out the number of nonzeros in the upper triangular matrix. */
      nzUpper = adiag[0]-adiag[n];
      if (!upTriFactor) {
        PetscScalar *AAUp;

        cerr = hipHostMalloc((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRHIP(cerr);

        /* Allocate Space for the upper triangular matrix */
        cerr = hipHostMalloc((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRHIP(cerr);
        cerr = hipHostMalloc((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRHIP(cerr);

        /* Fill the upper triangular matrix */
        AiUp[0]=(PetscInt) 0;
        AiUp[n]=nzUpper;
        offset = nzUpper;
        for (i=n-1; i>=0; i--) {
          v  = aa + adiag[i+1] + 1;
          vi = aj + adiag[i+1] + 1;

          /* number of elements NOT on the diagonal */
          nz = adiag[i] - adiag[i+1]-1;

          /* decrement the offset */
          offset -= (nz+1);

          /* first, set the diagonal elements */
          AjUp[offset] = (PetscInt) i;
          AAUp[offset] = (MatScalar)1./v[nz];
          AiUp[i]      = AiUp[i+1] - (nz+1);

          ierr = PetscArraycpy(&(AjUp[offset+1]), vi, nz);CHKERRQ(ierr);
          ierr = PetscArraycpy(&(AAUp[offset+1]), v, nz);CHKERRQ(ierr);
        }

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&upTriFactor);CHKERRQ(ierr);
        upTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        stat = rocsparse_create_mat_descr(&upTriFactor->descr);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_index_base(upTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_type(upTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_fill_mode(upTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_diag_type(upTriFactor->descr, HIPSPARSE_DIAG_TYPE_NON_UNIT);CHKERRHIPSPARSE(stat);

        /* set the operation */
        upTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        /* set the matrix */
        upTriFactor->csrMat = new CsrMatrix;
        upTriFactor->csrMat->num_rows = n;
        upTriFactor->csrMat->num_cols = n;
        upTriFactor->csrMat->num_entries = nzUpper;

        upTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(n+1);
        upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+n+1);

        upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(nzUpper);
        upTriFactor->csrMat->column_indices->assign(AjUp, AjUp+nzUpper);

        upTriFactor->csrMat->values = new THRUSTARRAY(nzUpper);
        upTriFactor->csrMat->values->assign(AAUp, AAUp+nzUpper);

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = hipsparse_create_analysis_info(&upTriFactor->solveInfo);CHKERRHIPSPARSE(stat);
        stat = hipsparse_get_svbuffsize(hipsparseTriFactors->handle, upTriFactor->solveOp,
                                     upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                     upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                     upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo,
                                     &upTriFactor->solveBufferSize);CHKERRHIPSPARSE(stat);
        cerr = hipMalloc(&upTriFactor->solveBuffer,upTriFactor->solveBufferSize);CHKERRHIP(cerr);

        /* perform the solve analysis */
        stat = hipsparse_analysis(hipsparseTriFactors->handle, upTriFactor->solveOp,
                                 upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                 upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                 upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo
                                 ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);
        cerr = WaitForHIP();CHKERRHIP(cerr);
        ierr = PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;
        upTriFactor->AA_h = AAUp;
        cerr = hipHostFree(AiUp);CHKERRHIP(cerr);
        cerr = hipHostFree(AjUp);CHKERRHIP(cerr);
        ierr = PetscLogCpuToGpu((n+1+nzUpper)*sizeof(int)+nzUpper*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        if (!upTriFactor->AA_h) {
          cerr = hipHostMalloc((void**) &upTriFactor->AA_h, nzUpper*sizeof(PetscScalar));CHKERRHIP(cerr);
        }
        /* Fill the upper triangular matrix */
        offset = nzUpper;
        for (i=n-1; i>=0; i--) {
          v  = aa + adiag[i+1] + 1;

          /* number of elements NOT on the diagonal */
          nz = adiag[i] - adiag[i+1]-1;

          /* decrement the offset */
          offset -= (nz+1);

          /* first, set the diagonal elements */
          upTriFactor->AA_h[offset] = 1./v[nz];
          ierr = PetscArraycpy(&(upTriFactor->AA_h[offset+1]), v, nz);CHKERRQ(ierr);
        }
        upTriFactor->csrMat->values->assign(upTriFactor->AA_h, upTriFactor->AA_h+nzUpper);
        ierr = PetscLogCpuToGpu(nzUpper*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  IS                           isrow = a->row,iscol = a->icol;
  PetscBool                    row_identity,col_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  if (!hipsparseTriFactors) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing hipsparseTriFactors");
  ierr = MatSeqAIJHIPSPARSEBuildILULowerTriMatrix(A);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSEBuildILUUpperTriMatrix(A);CHKERRQ(ierr);

  if (!hipsparseTriFactors->workVector) { hipsparseTriFactors->workVector = new THRUSTARRAY(n); }
  hipsparseTriFactors->nnz=a->nz;

  A->offloadmask = PETSC_OFFLOAD_BOTH;
  /* lower triangular indices */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  if (!row_identity && !hipsparseTriFactors->rpermIndices) {
    const PetscInt *r;

    ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(r, r+n);
    ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }

  /* upper triangular indices */
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!col_identity && !hipsparseTriFactors->cpermIndices) {
    const PetscInt *c;

    ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr);
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(c, c+n);
    ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEBuildICCTriMatrices(Mat A)
{
  Mat_SeqAIJ                        *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  rocsparseStatus_t                  stat;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;
  PetscInt                          *AiUp, *AjUp;
  PetscScalar                       *AAUp;
  PetscScalar                       *AALo;
  PetscInt                          nzUpper = a->nz,n = A->rmap->n,i,offset,nz,j;
  Mat_SeqSBAIJ                      *b = (Mat_SeqSBAIJ*)A->data;
  const PetscInt                    *ai = b->i,*aj = b->j,*vj;
  const MatScalar                   *aa = b->a,*v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    try {
      cerr = hipHostMalloc((void**) &AAUp, nzUpper*sizeof(PetscScalar));CHKERRHIP(cerr);
      cerr = hipHostMalloc((void**) &AALo, nzUpper*sizeof(PetscScalar));CHKERRHIP(cerr);
      if (!upTriFactor && !loTriFactor) {
        /* Allocate Space for the upper triangular matrix */
        cerr = hipHostMalloc((void**) &AiUp, (n+1)*sizeof(PetscInt));CHKERRHIP(cerr);
        cerr = hipHostMalloc((void**) &AjUp, nzUpper*sizeof(PetscInt));CHKERRHIP(cerr);

        /* Fill the upper triangular matrix */
        AiUp[0]=(PetscInt) 0;
        AiUp[n]=nzUpper;
        offset = 0;
        for (i=0; i<n; i++) {
          /* set the pointers */
          v  = aa + ai[i];
          vj = aj + ai[i];
          nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */

          /* first, set the diagonal elements */
          AjUp[offset] = (PetscInt) i;
          AAUp[offset] = (MatScalar)1.0/v[nz];
          AiUp[i]      = offset;
          AALo[offset] = (MatScalar)1.0/v[nz];

          offset+=1;
          if (nz>0) {
            ierr = PetscArraycpy(&(AjUp[offset]), vj, nz);CHKERRQ(ierr);
            ierr = PetscArraycpy(&(AAUp[offset]), v, nz);CHKERRQ(ierr);
            for (j=offset; j<offset+nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j]/v[nz];
            }
            offset+=nz;
          }
        }

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&upTriFactor);CHKERRQ(ierr);
        upTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        stat = rocsparse_create_mat_descr(&upTriFactor->descr);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_index_base(upTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_type(upTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_fill_mode(upTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_diag_type(upTriFactor->descr, HIPSPARSE_DIAG_TYPE_UNIT);CHKERRHIPSPARSE(stat);

        /* set the matrix */
        upTriFactor->csrMat = new CsrMatrix;
        upTriFactor->csrMat->num_rows = A->rmap->n;
        upTriFactor->csrMat->num_cols = A->cmap->n;
        upTriFactor->csrMat->num_entries = a->nz;

        upTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
        upTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+A->rmap->n+1);

        upTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
        upTriFactor->csrMat->column_indices->assign(AjUp, AjUp+a->nz);

        upTriFactor->csrMat->values = new THRUSTARRAY(a->nz);
        upTriFactor->csrMat->values->assign(AAUp, AAUp+a->nz);

        /* set the operation */
        upTriFactor->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = hipsparse_create_analysis_info(&upTriFactor->solveInfo);CHKERRHIPSPARSE(stat);
        stat = hipsparse_get_svbuffsize(hipsparseTriFactors->handle, upTriFactor->solveOp,
                                       upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                       upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                       upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo,
                                       &upTriFactor->solveBufferSize);CHKERRHIPSPARSE(stat);
        cerr = hipMalloc(&upTriFactor->solveBuffer,upTriFactor->solveBufferSize);CHKERRHIP(cerr);

        /* perform the solve analysis */
        stat = hipsparse_analysis(hipsparseTriFactors->handle, upTriFactor->solveOp,
                                 upTriFactor->csrMat->num_rows, upTriFactor->csrMat->num_entries, upTriFactor->descr,
                                 upTriFactor->csrMat->values->data().get(), upTriFactor->csrMat->row_offsets->data().get(),
                                 upTriFactor->csrMat->column_indices->data().get(), upTriFactor->solveInfo
                                 ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);
        cerr = WaitForHIP();CHKERRHIP(cerr);
        ierr = PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->upTriFactorPtr = upTriFactor;

        /* allocate space for the triangular factor information */
        ierr = PetscNew(&loTriFactor);CHKERRQ(ierr);
        loTriFactor->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

        /* Create the matrix description */
        stat = rocsparse_create_mat_descr(&loTriFactor->descr);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_index_base(loTriFactor->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_type(loTriFactor->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_fill_mode(loTriFactor->descr, HIPSPARSE_FILL_MODE_UPPER);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_diag_type(loTriFactor->descr, HIPSPARSE_DIAG_TYPE_NON_UNIT);CHKERRHIPSPARSE(stat);

        /* set the operation */
        loTriFactor->solveOp = HIPSPARSE_OPERATION_TRANSPOSE;

        /* set the matrix */
        loTriFactor->csrMat = new CsrMatrix;
        loTriFactor->csrMat->num_rows = A->rmap->n;
        loTriFactor->csrMat->num_cols = A->cmap->n;
        loTriFactor->csrMat->num_entries = a->nz;

        loTriFactor->csrMat->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
        loTriFactor->csrMat->row_offsets->assign(AiUp, AiUp+A->rmap->n+1);

        loTriFactor->csrMat->column_indices = new THRUSTINTARRAY32(a->nz);
        loTriFactor->csrMat->column_indices->assign(AjUp, AjUp+a->nz);

        loTriFactor->csrMat->values = new THRUSTARRAY(a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo+a->nz);

        /* Create the solve analysis information */
        ierr = PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
        stat = hipsparse_create_analysis_info(&loTriFactor->solveInfo);CHKERRHIPSPARSE(stat);
        stat = hipsparse_get_svbuffsize(hipsparseTriFactors->handle, loTriFactor->solveOp,
                                       loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                       loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                       loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo,
                                       &loTriFactor->solveBufferSize);CHKERRHIPSPARSE(stat);
        cerr = hipMalloc(&loTriFactor->solveBuffer,loTriFactor->solveBufferSize);CHKERRHIP(cerr);

        /* perform the solve analysis */
        stat = hipsparse_analysis(hipsparseTriFactors->handle, loTriFactor->solveOp,
                                 loTriFactor->csrMat->num_rows, loTriFactor->csrMat->num_entries, loTriFactor->descr,
                                 loTriFactor->csrMat->values->data().get(), loTriFactor->csrMat->row_offsets->data().get(),
                                 loTriFactor->csrMat->column_indices->data().get(), loTriFactor->solveInfo
                                 ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);
        cerr = WaitForHIP();CHKERRHIP(cerr);
        ierr = PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

        /* assign the pointer */
        ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->loTriFactorPtr = loTriFactor;

        ierr = PetscLogCpuToGpu(2*(((A->rmap->n+1)+(a->nz))*sizeof(int)+(a->nz)*sizeof(PetscScalar)));CHKERRQ(ierr);
        cerr = hipHostFree(AiUp);CHKERRHIP(cerr);
        cerr = hipHostFree(AjUp);CHKERRHIP(cerr);
      } else {
        /* Fill the upper triangular matrix */
        offset = 0;
        for (i=0; i<n; i++) {
          /* set the pointers */
          v  = aa + ai[i];
          nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */

          /* first, set the diagonal elements */
          AAUp[offset] = 1.0/v[nz];
          AALo[offset] = 1.0/v[nz];

          offset+=1;
          if (nz>0) {
            ierr = PetscArraycpy(&(AAUp[offset]), v, nz);CHKERRQ(ierr);
            for (j=offset; j<offset+nz; j++) {
              AAUp[j] = -AAUp[j];
              AALo[j] = AAUp[j]/v[nz];
            }
            offset+=nz;
          }
        }
        if (!upTriFactor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing hipsparseTriFactors");
        if (!loTriFactor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing hipsparseTriFactors");
        upTriFactor->csrMat->values->assign(AAUp, AAUp+a->nz);
        loTriFactor->csrMat->values->assign(AALo, AALo+a->nz);
        ierr = PetscLogCpuToGpu(2*(a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
      }
      cerr = hipHostFree(AAUp);CHKERRHIP(cerr);
      cerr = hipHostFree(AALo);CHKERRHIP(cerr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(Mat A)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a                  = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  IS                           ip = a->row;
  PetscBool                    perm_identity;
  PetscInt                     n = A->rmap->n;

  PetscFunctionBegin;
  if (!hipsparseTriFactors) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing hipsparseTriFactors");
  ierr = MatSeqAIJHIPSPARSEBuildICCTriMatrices(A);CHKERRQ(ierr);
  if (!hipsparseTriFactors->workVector) { hipsparseTriFactors->workVector = new THRUSTARRAY(n); }
  hipsparseTriFactors->nnz=(a->nz-n)*2 + n;

  A->offloadmask = PETSC_OFFLOAD_BOTH;

  /* lower triangular indices */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) {
    IS             iip;
    const PetscInt *irip,*rip;

    ierr = ISInvertPermutation(ip,PETSC_DECIDE,&iip);CHKERRQ(ierr);
    ierr = ISGetIndices(iip,&irip);CHKERRQ(ierr);
    ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(rip, rip+n);
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(irip, irip+n);
    ierr = ISRestoreIndices(iip,&irip);CHKERRQ(ierr);
    ierr = ISDestroy(&iip);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             isrow = b->row,iscol = b->col;
  PetscBool      row_identity,col_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSECopyFromGPU(A);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJHIPSPARSEILUAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJHIPSPARSE(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *b = (Mat_SeqAIJ*)B->data;
  IS             ip = b->row;
  PetscBool      perm_identity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSECopyFromGPU(A);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric_SeqAIJ(B,A,info);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  /* determine which version of MatSolve needs to be used. */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity) {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  } else {
    B->ops->solve = MatSolve_SeqAIJHIPSPARSE;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJHIPSPARSE;
    B->ops->matsolve = NULL;
    B->ops->matsolvetranspose = NULL;
  }

  /* get the triangular factors */
  ierr = MatSeqAIJHIPSPARSEICCAnalysisAndCopyToGPU(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(Mat A)
{
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorT;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorT;
  rocsparseStatus_t                  stat;
  hipsparseIndexBase_t               indexBase;
  hipsparseMatrixType_t              matrixType;
  hipsparseFillMode_t                fillMode;
  hipsparseDiagType_t                diagType;
  hipError_t                       cerr;
  PetscErrorCode                    ierr;

  PetscFunctionBegin;
  /* allocate space for the transpose of the lower triangular factor */
  ierr = PetscNew(&loTriFactorT);CHKERRQ(ierr);
  loTriFactorT->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the lower triangular factor */
  matrixType = hipsparseGetMatType(loTriFactor->descr);
  indexBase = hipsparseGetMatIndexBase(loTriFactor->descr);
  fillMode = hipsparseGetMatFillMode(loTriFactor->descr)==HIPSPARSE_FILL_MODE_UPPER ?
    HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;
  diagType = hipsparseGetMatDiagType(loTriFactor->descr);

  /* Create the matrix description */
  stat = rocsparse_create_mat_descr(&loTriFactorT->descr);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_index_base(loTriFactorT->descr, indexBase);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_type(loTriFactorT->descr, matrixType);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_fill_mode(loTriFactorT->descr, fillMode);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_diag_type(loTriFactorT->descr, diagType);CHKERRHIPSPARSE(stat);

  /* set the operation */
  loTriFactorT->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the lower triangular factor*/
  loTriFactorT->csrMat = new CsrMatrix;
  loTriFactorT->csrMat->num_rows       = loTriFactor->csrMat->num_cols;
  loTriFactorT->csrMat->num_cols       = loTriFactor->csrMat->num_rows;
  loTriFactorT->csrMat->num_entries    = loTriFactor->csrMat->num_entries;
  loTriFactorT->csrMat->row_offsets    = new THRUSTINTARRAY32(loTriFactorT->csrMat->num_rows+1);
  loTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(loTriFactorT->csrMat->num_entries);
  loTriFactorT->csrMat->values         = new THRUSTARRAY(loTriFactorT->csrMat->num_entries);

  /* compute the transpose of the lower triangular factor, i.e. the CSC */

  ierr = PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  stat = hipsparse_csr2csc(hipsparseTriFactors->handle, loTriFactor->csrMat->num_rows,
                          loTriFactor->csrMat->num_cols, loTriFactor->csrMat->num_entries,
                          loTriFactor->csrMat->values->data().get(),
                          loTriFactor->csrMat->row_offsets->data().get(),
                          loTriFactor->csrMat->column_indices->data().get(),
                          loTriFactorT->csrMat->values->data().get(),
                          loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                          HIPSPARSE_ACTION_NUMERIC, indexBase
);CHKERRHIPSPARSE(stat);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);

  /* Create the solve analysis information */
  ierr = PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
  stat = hipsparse_create_analysis_info(&loTriFactorT->solveInfo);CHKERRHIPSPARSE(stat);
  stat = hipsparse_get_svbuffsize(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                                loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr,
                                loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                                loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo,
                                &loTriFactorT->solveBufferSize);CHKERRHIPSPARSE(stat);
  cerr = hipMalloc(&loTriFactorT->solveBuffer,loTriFactorT->solveBufferSize);CHKERRHIP(cerr);

  /* perform the solve analysis */
  stat = hipsparse_analysis(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                           loTriFactorT->csrMat->num_rows, loTriFactorT->csrMat->num_entries, loTriFactorT->descr,
                           loTriFactorT->csrMat->values->data().get(), loTriFactorT->csrMat->row_offsets->data().get(),
                           loTriFactorT->csrMat->column_indices->data().get(), loTriFactorT->solveInfo
                           ,loTriFactorT->solvePolicy, loTriFactorT->solveBuffer
);CHKERRHIPSPARSE(stat);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->loTriFactorPtrTranspose = loTriFactorT;

  /*********************************************/
  /* Now the Transpose of the Upper Tri Factor */
  /*********************************************/

  /* allocate space for the transpose of the upper triangular factor */
  ierr = PetscNew(&upTriFactorT);CHKERRQ(ierr);
  upTriFactorT->solvePolicy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

  /* set the matrix descriptors of the upper triangular factor */
  matrixType = hipsparseGetMatType(upTriFactor->descr);
  indexBase = hipsparseGetMatIndexBase(upTriFactor->descr);
  fillMode = hipsparseGetMatFillMode(upTriFactor->descr)==HIPSPARSE_FILL_MODE_UPPER ?
    HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;
  diagType = hipsparseGetMatDiagType(upTriFactor->descr);

  /* Create the matrix description */
  stat = rocsparse_create_mat_descr(&upTriFactorT->descr);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_index_base(upTriFactorT->descr, indexBase);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_type(upTriFactorT->descr, matrixType);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_fill_mode(upTriFactorT->descr, fillMode);CHKERRHIPSPARSE(stat);
  stat = rocsparse_set_mat_diag_type(upTriFactorT->descr, diagType);CHKERRHIPSPARSE(stat);

  /* set the operation */
  upTriFactorT->solveOp = HIPSPARSE_OPERATION_NON_TRANSPOSE;

  /* allocate GPU space for the CSC of the upper triangular factor*/
  upTriFactorT->csrMat = new CsrMatrix;
  upTriFactorT->csrMat->num_rows       = upTriFactor->csrMat->num_cols;
  upTriFactorT->csrMat->num_cols       = upTriFactor->csrMat->num_rows;
  upTriFactorT->csrMat->num_entries    = upTriFactor->csrMat->num_entries;
  upTriFactorT->csrMat->row_offsets    = new THRUSTINTARRAY32(upTriFactorT->csrMat->num_rows+1);
  upTriFactorT->csrMat->column_indices = new THRUSTINTARRAY32(upTriFactorT->csrMat->num_entries);
  upTriFactorT->csrMat->values         = new THRUSTARRAY(upTriFactorT->csrMat->num_entries);

  /* compute the transpose of the upper triangular factor, i.e. the CSC */
  ierr = PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  stat = hipsparse_csr2csc(hipsparseTriFactors->handle, upTriFactor->csrMat->num_rows,
                          upTriFactor->csrMat->num_cols, upTriFactor->csrMat->num_entries,
                          upTriFactor->csrMat->values->data().get(),
                          upTriFactor->csrMat->row_offsets->data().get(),
                          upTriFactor->csrMat->column_indices->data().get(),
                          upTriFactorT->csrMat->values->data().get(),
                          upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                          HIPSPARSE_ACTION_NUMERIC, indexBase
);CHKERRHIPSPARSE(stat);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);

  /* Create the solve analysis information */
  ierr = PetscLogEventBegin(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);
  stat = hipsparse_create_analysis_info(&upTriFactorT->solveInfo);CHKERRHIPSPARSE(stat);
  stat = hipsparse_get_svbuffsize(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                                 upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr,
                                 upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                                 upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo,
                                 &upTriFactorT->solveBufferSize);CHKERRHIPSPARSE(stat);
  cerr = hipMalloc(&upTriFactorT->solveBuffer,upTriFactorT->solveBufferSize);CHKERRHIP(cerr);

  /* perform the solve analysis */
  stat = hipsparse_analysis(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                           upTriFactorT->csrMat->num_rows, upTriFactorT->csrMat->num_entries, upTriFactorT->descr,
                           upTriFactorT->csrMat->values->data().get(), upTriFactorT->csrMat->row_offsets->data().get(),
                           upTriFactorT->csrMat->column_indices->data().get(), upTriFactorT->solveInfo
                           ,upTriFactorT->solvePolicy, upTriFactorT->solveBuffer
);CHKERRHIPSPARSE(stat);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogEventEnd(MAT_HIPSPARSESolveAnalysis,A,0,0,0);CHKERRQ(ierr);

  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSETriFactors*)A->spptr)->upTriFactorPtrTranspose = upTriFactorT;
  PetscFunctionReturn(0);
}

struct PetscScalarToPetscInt
{
  __host__ __device__
  PetscInt operator()(PetscScalar s)
  {
    return (PetscInt)PetscRealPart(s);
  }
};

static PetscErrorCode MatSeqAIJHIPSPARSEGenerateTransposeForMult(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct, *matstructT;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  rocsparseStatus_t             stat;
  hipsparseIndexBase_t          indexBase;
  hipError_t                  err;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  if (!hipsparsestruct->transgen || !A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
  if (!matstruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing mat struct");
  matstructT = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
  if (hipsparsestruct->transupdated && !matstructT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing matTranspose struct");
  if (hipsparsestruct->transupdated) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(MAT_HIPSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  if (hipsparsestruct->format != MAT_HIPSPARSE_CSR) {
    ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!hipsparsestruct->matTranspose) { /* create hipsparse matrix */
    matstructT = new Mat_SeqAIJHIPSPARSEMultStruct;
    stat = rocsparse_create_mat_descr(&matstructT->descr);CHKERRHIPSPARSE(stat);
    indexBase = hipsparseGetMatIndexBase(matstruct->descr);
    stat = hipsparseSetMatIndexBase(matstructT->descr, indexBase);CHKERRHIPSPARSE(stat);
    stat = hipsparseSetMatType(matstructT->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);

    /* set alpha and beta */
    err = hipMalloc((void **)&(matstructT->alpha_one),sizeof(PetscScalar));CHKERRHIP(err);
    err = hipMalloc((void **)&(matstructT->beta_zero),sizeof(PetscScalar));CHKERRHIP(err);
    err = hipMalloc((void **)&(matstructT->beta_one), sizeof(PetscScalar));CHKERRHIP(err);
    err = hipMemcpy(matstructT->alpha_one,&PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(err);
    err = hipMemcpy(matstructT->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(err);
    err = hipMemcpy(matstructT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(err);

    if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
      CsrMatrix *matrixT = new CsrMatrix;
      matstructT->mat = matrixT;
      matrixT->num_rows = A->cmap->n;
      matrixT->num_cols = A->rmap->n;
      matrixT->num_entries = a->nz;
      matrixT->row_offsets = new THRUSTINTARRAY32(matrixT->num_rows+1);
      matrixT->column_indices = new THRUSTINTARRAY32(a->nz);
      matrixT->values = new THRUSTARRAY(a->nz);

      if (!hipsparsestruct->rowoffsets_gpu) { hipsparsestruct->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n+1); }
      hipsparsestruct->rowoffsets_gpu->assign(a->i,a->i+A->rmap->n+1);

    } else if (hipsparsestruct->format == MAT_HIPSPARSE_ELL || hipsparsestruct->format == MAT_HIPSPARSE_HYB) {
      CsrMatrix *temp  = new CsrMatrix;
      CsrMatrix *tempT = new CsrMatrix;
      /* First convert HYB to CSR */
      temp->num_rows = A->rmap->n;
      temp->num_cols = A->cmap->n;
      temp->num_entries = a->nz;
      temp->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
      temp->column_indices = new THRUSTINTARRAY32(a->nz);
      temp->values = new THRUSTARRAY(a->nz);

      stat = hipsparse_hyb2csr(hipsparsestruct->handle,
                              matstruct->descr, (hipsparseHybMat_t)matstruct->mat,
                              temp->values->data().get(),
                              temp->row_offsets->data().get(),
                              temp->column_indices->data().get());CHKERRHIPSPARSE(stat);

      /* Next, convert CSR to CSC (i.e. the matrix transpose) */
      tempT->num_rows = A->rmap->n;
      tempT->num_cols = A->cmap->n;
      tempT->num_entries = a->nz;
      tempT->row_offsets = new THRUSTINTARRAY32(A->rmap->n+1);
      tempT->column_indices = new THRUSTINTARRAY32(a->nz);
      tempT->values = new THRUSTARRAY(a->nz);

      stat = hipsparse_csr2csc(hipsparsestruct->handle, temp->num_rows,
                              temp->num_cols, temp->num_entries,
                              temp->values->data().get(),
                              temp->row_offsets->data().get(),
                              temp->column_indices->data().get(),
                              tempT->values->data().get(),
                              tempT->column_indices->data().get(),
                              tempT->row_offsets->data().get(),
                              HIPSPARSE_ACTION_NUMERIC, indexBase);CHKERRHIPSPARSE(stat);

      /* Last, convert CSC to HYB */
      hipsparseHybMat_t hybMat;
      stat = hipsparseCreateHybMat(&hybMat);CHKERRHIPSPARSE(stat);
      hipsparseHybPartition_t partition = hipsparsestruct->format==MAT_HIPSPARSE_ELL ?
        HIPSPARSE_HYB_PARTITION_MAX : HIPSPARSE_HYB_PARTITION_AUTO;
      stat = hipsparse_csr2hyb(hipsparsestruct->handle, A->rmap->n, A->cmap->n,
                              matstructT->descr, tempT->values->data().get(),
                              tempT->row_offsets->data().get(),
                              tempT->column_indices->data().get(),
                              hybMat, 0, partition);CHKERRHIPSPARSE(stat);

      /* assign the pointer */
      matstructT->mat = hybMat;
      hipsparsestruct->transupdated = PETSC_TRUE;
      /* delete temporaries */
      if (tempT) {
        if (tempT->values) delete (THRUSTARRAY*) tempT->values;
        if (tempT->column_indices) delete (THRUSTINTARRAY32*) tempT->column_indices;
        if (tempT->row_offsets) delete (THRUSTINTARRAY32*) tempT->row_offsets;
        delete (CsrMatrix*) tempT;
      }
      if (temp) {
        if (temp->values) delete (THRUSTARRAY*) temp->values;
        if (temp->column_indices) delete (THRUSTINTARRAY32*) temp->column_indices;
        if (temp->row_offsets) delete (THRUSTINTARRAY32*) temp->row_offsets;
        delete (CsrMatrix*) temp;
      }
    }
  }
  if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) { /* transpose mat struct may be already present, update data */
    CsrMatrix *matrix  = (CsrMatrix*)matstruct->mat;
    CsrMatrix *matrixT = (CsrMatrix*)matstructT->mat;
    if (!matrix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrix");
    if (!matrix->row_offsets) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrix rows");
    if (!matrix->column_indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrix cols");
    if (!matrix->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrix values");
    if (!matrixT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrixT");
    if (!matrixT->row_offsets) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrixT rows");
    if (!matrixT->column_indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrixT cols");
    if (!matrixT->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CsrMatrixT values");
    if (!hipsparsestruct->rowoffsets_gpu) { /* this may be absent when we did not construct the transpose with csr2csc */
      hipsparsestruct->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
      hipsparsestruct->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
      ierr = PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!hipsparsestruct->csr2csc_i) {
      THRUSTARRAY csr2csc_a(matrix->num_entries);
      PetscStackCallThrust(thrust::sequence(thrust::device, csr2csc_a.begin(), csr2csc_a.end(), 0.0));

      indexBase = hipsparseGetMatIndexBase(matstruct->descr);

      stat = hipsparse_csr2csc(hipsparsestruct->handle,
                              A->rmap->n,A->cmap->n,matrix->num_entries,
                              csr2csc_a.data().get(),hipsparsestruct->rowoffsets_gpu->data().get(),matrix->column_indices->data().get(),
                              matrixT->values->data().get(),
                              matrixT->column_indices->data().get(), matrixT->row_offsets->data().get(),
                              HIPSPARSE_ACTION_NUMERIC, indexBase
);CHKERRHIPSPARSE(stat);
      hipsparsestruct->csr2csc_i = new THRUSTINTARRAY(matrix->num_entries);
      PetscStackCallThrust(thrust::transform(thrust::device,matrixT->values->begin(),matrixT->values->end(),hipsparsestruct->csr2csc_i->begin(),PetscScalarToPetscInt()));
    }
    PetscStackCallThrust(thrust::copy(thrust::device,thrust::make_permutation_iterator(matrix->values->begin(), hipsparsestruct->csr2csc_i->begin()),
                                                     thrust::make_permutation_iterator(matrix->values->begin(), hipsparsestruct->csr2csc_i->end()),
                                                     matrixT->values->begin()));
  }
  ierr = PetscLogEventEnd(MAT_HIPSPARSEGenerateTranspose,A,0,0,0);CHKERRQ(ierr);
  /* the compressed row indices is not used for matTranspose */
  matstructT->cprowIndices = NULL;
  /* assign the pointer */
  ((Mat_SeqAIJHIPSPARSE*)A->spptr)->matTranspose = matstructT;
  hipsparsestruct->transupdated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Why do we need to analyze the transposed matrix again? Can't we just use op(A) = HIPSPARSE_OPERATION_TRANSPOSE in MatSolve_SeqAIJHIPSPARSE? */
static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE(Mat A,Vec bb,Vec xx)
{
  PetscInt                              n = xx->map->n;
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  rocsparseStatus_t                      stat;
  Mat_SeqAIJHIPSPARSETriFactors          *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                        ierr;
  hipError_t                           cerr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    ierr = MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecHIPGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU+n, hipsparseTriFactors->rpermIndices->end()),
               xGPU);

  /* First, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                        upTriFactorT->csrMat->num_rows,
                        upTriFactorT->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, upTriFactorT->descr,
                        upTriFactorT->csrMat->values->data().get(),
                        upTriFactorT->csrMat->row_offsets->data().get(),
                        upTriFactorT->csrMat->column_indices->data().get(),
                        upTriFactorT->solveInfo,
                        xarray, tempGPU->data().get()
                        ,upTriFactorT->solvePolicy, upTriFactorT->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* Then, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                        loTriFactorT->csrMat->num_rows,
                        loTriFactorT->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, loTriFactorT->descr,
                        loTriFactorT->csrMat->values->data().get(),
                        loTriFactorT->csrMat->row_offsets->data().get(),
                        loTriFactorT->csrMat->column_indices->data().get(),
                        loTriFactorT->solveInfo,
                        tempGPU->data().get(), xarray
                        ,loTriFactorT->solvePolicy, loTriFactorT->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* Last, copy the solution, xGPU, into a temporary with the column permutation ... can't be done in place. */
  thrust::copy(thrust::make_permutation_iterator(xGPU, hipsparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(xGPU+n, hipsparseTriFactors->cpermIndices->end()),
               tempGPU->begin());

  /* Copy the temporary to the full solution. */
  thrust::copy(tempGPU->begin(), tempGPU->end(), xGPU);

  /* restore */
  ierr = VecHIPRestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecHIPRestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqAIJHIPSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                 *barray;
  PetscScalar                       *xarray;
  rocsparseStatus_t                  stat;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorT = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  THRUSTARRAY                       *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  /* Analyze the matrix and create the transpose ... on the fly */
  if (!loTriFactorT && !upTriFactorT) {
    ierr = MatSeqAIJHIPSPARSEAnalyzeTransposeForSolve(A);CHKERRQ(ierr);
    loTriFactorT       = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtrTranspose;
    upTriFactorT       = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtrTranspose;
  }

  /* Get the GPU pointers */
  ierr = VecHIPGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPGetArrayRead(bb,&barray);CHKERRQ(ierr);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactorT->solveOp,
                        upTriFactorT->csrMat->num_rows,
                        upTriFactorT->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, upTriFactorT->descr,
                        upTriFactorT->csrMat->values->data().get(),
                        upTriFactorT->csrMat->row_offsets->data().get(),
                        upTriFactorT->csrMat->column_indices->data().get(),
                        upTriFactorT->solveInfo,
                        barray, tempGPU->data().get()
                        ,upTriFactorT->solvePolicy, upTriFactorT->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* Then, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactorT->solveOp,
                        loTriFactorT->csrMat->num_rows,
                        loTriFactorT->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, loTriFactorT->descr,
                        loTriFactorT->csrMat->values->data().get(),
                        loTriFactorT->csrMat->row_offsets->data().get(),
                        loTriFactorT->csrMat->column_indices->data().get(),
                        loTriFactorT->solveInfo,
                        tempGPU->data().get(), xarray
                        ,loTriFactorT->solvePolicy, loTriFactorT->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* restore */
  ierr = VecHIPRestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecHIPRestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  rocsparseStatus_t                      stat;
  Mat_SeqAIJHIPSPARSETriFactors          *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct     *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                        ierr;
  hipError_t                           cerr;

  PetscFunctionBegin;

  /* Get the GPU pointers */
  ierr = VecHIPGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->end()),
               tempGPU->begin());

  /* Next, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactor->solveOp,
                        loTriFactor->csrMat->num_rows,
                        loTriFactor->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, loTriFactor->descr,
                        loTriFactor->csrMat->values->data().get(),
                        loTriFactor->csrMat->row_offsets->data().get(),
                        loTriFactor->csrMat->column_indices->data().get(),
                        loTriFactor->solveInfo,
                        tempGPU->data().get(), xarray
                        ,loTriFactor->solvePolicy, loTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* Then, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactor->solveOp,
                        upTriFactor->csrMat->num_rows,
                        upTriFactor->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, upTriFactor->descr,
                        upTriFactor->csrMat->values->data().get(),
                        upTriFactor->csrMat->row_offsets->data().get(),
                        upTriFactor->csrMat->column_indices->data().get(),
                        upTriFactor->solveInfo,
                        xarray, tempGPU->data().get()
                        ,upTriFactor->solvePolicy, upTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->end()),
               xGPU);

  ierr = VecHIPRestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecHIPRestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSE_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                 *barray;
  PetscScalar                       *xarray;
  rocsparseStatus_t                  stat;
  Mat_SeqAIJHIPSPARSETriFactors      *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors*)A->spptr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->loTriFactorPtr;
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactor = (Mat_SeqAIJHIPSPARSETriFactorStruct*)hipsparseTriFactors->upTriFactorPtr;
  THRUSTARRAY                       *tempGPU = (THRUSTARRAY*)hipsparseTriFactors->workVector;
  PetscErrorCode                    ierr;
  hipError_t                       cerr;

  PetscFunctionBegin;
  /* Get the GPU pointers */
  ierr = VecHIPGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecHIPGetArrayRead(bb,&barray);CHKERRQ(ierr);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, solve L */
  stat = hipsparse_solve(hipsparseTriFactors->handle, loTriFactor->solveOp,
                        loTriFactor->csrMat->num_rows,
                        loTriFactor->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, loTriFactor->descr,
                        loTriFactor->csrMat->values->data().get(),
                        loTriFactor->csrMat->row_offsets->data().get(),
                        loTriFactor->csrMat->column_indices->data().get(),
                        loTriFactor->solveInfo,
                        barray, tempGPU->data().get(),
                        loTriFactor->solvePolicy, loTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);

  /* Next, solve U */
  stat = hipsparse_solve(hipsparseTriFactors->handle, upTriFactor->solveOp,
                        upTriFactor->csrMat->num_rows,
                        upTriFactor->csrMat->num_entries,
                        &PETSC_HIPSPARSE_ONE, upTriFactor->descr,
                        upTriFactor->csrMat->values->data().get(),
                        upTriFactor->csrMat->row_offsets->data().get(),
                        upTriFactor->csrMat->column_indices->data().get(),
                        upTriFactor->solveInfo,
                        tempGPU->data().get(), xarray,
                        upTriFactor->solvePolicy, upTriFactor->solveBuffer
);CHKERRHIPSPARSE(stat);

  ierr = VecHIPRestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecHIPRestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*hipsparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSECopyFromGPU(Mat A)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  hipError_t        cerr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    CsrMatrix *matrix = (CsrMatrix*)hipsp->mat->mat;

    ierr = PetscLogEventBegin(MAT_HIPSPARSECopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    cerr = hipMemcpy(a->a, matrix->values->data().get(), a->nz*sizeof(PetscScalar), hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    cerr = WaitForHIP();CHKERRHIP(cerr);
    ierr = PetscLogGpuToCpu(a->nz*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_HIPSPARSECopyFromGPU,A,0,0,0);CHKERRQ(ierr);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJHIPSPARSE(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSECopyFromGPU(A);CHKERRQ(ierr);
  *array = a->a;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSECopyToGPU(Mat A)
{
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct = hipsparsestruct->mat;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  PetscInt                     m = A->rmap->n,*ii,*ridx,tmp;
  PetscErrorCode               ierr;
  rocsparseStatus_t             stat;
  PetscBool                    both = PETSC_TRUE;
  hipError_t                  err;

  PetscFunctionBegin;
  if (A->boundtocpu) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot copy to GPU");
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (A->nonzerostate == hipsparsestruct->nonzerostate && hipsparsestruct->format == MAT_HIPSPARSE_CSR) { /* Copy values only */
      CsrMatrix *matrix;
      matrix = (CsrMatrix*)hipsparsestruct->mat->mat;

      if (a->nz && !a->a) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CSR values");
      ierr = PetscLogEventBegin(MAT_HIPSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      matrix->values->assign(a->a, a->a+a->nz);
      err  = WaitForHIP();CHKERRHIP(err);
      ierr = PetscLogCpuToGpu((a->nz)*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_HIPSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      PetscInt nnz;
      ierr = PetscLogEventBegin(MAT_HIPSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&hipsparsestruct->mat,hipsparsestruct->format);CHKERRQ(ierr);
      ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);
      delete hipsparsestruct->workVector;
      delete hipsparsestruct->rowoffsets_gpu;
      hipsparsestruct->workVector = NULL;
      hipsparsestruct->rowoffsets_gpu = NULL;
      try {
        if (a->compressedrow.use) {
          m    = a->compressedrow.nrows;
          ii   = a->compressedrow.i;
          ridx = a->compressedrow.rindex;
        } else {
          m    = A->rmap->n;
          ii   = a->i;
          ridx = NULL;
        }
        if (!ii) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CSR row data");
        if (m && !a->j) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing CSR column data");
        if (!a->a) { nnz = ii[m]; both = PETSC_FALSE; }
        else nnz = a->nz;

        /* create hipsparse matrix */
        hipsparsestruct->nrows = m;
        matstruct = new Mat_SeqAIJHIPSPARSEMultStruct;
        stat = hipsparseCreateMatDescr(&matstruct->descr);CHKERRHIPSPARSE(stat);
        stat = hipsparseSetMatIndexBase(matstruct->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = hipsparseSetMatType(matstruct->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);

        err = hipMalloc((void **)&(matstruct->alpha_one),sizeof(PetscScalar));CHKERRHIP(err);
        err = hipMalloc((void **)&(matstruct->beta_zero),sizeof(PetscScalar));CHKERRHIP(err);
        err = hipMalloc((void **)&(matstruct->beta_one), sizeof(PetscScalar));CHKERRHIP(err);
        err = hipMemcpy(matstruct->alpha_one,&PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(err);
        err = hipMemcpy(matstruct->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(err);
        err = hipMemcpy(matstruct->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(err);
        stat = hipsparseSetPointerMode(hipsparsestruct->handle, HIPSPARSE_POINTER_MODE_DEVICE);CHKERRHIPSPARSE(stat);

        /* Build a hybrid/ellpack matrix if this option is chosen for the storage */
        if (hipsparsestruct->format==MAT_HIPSPARSE_CSR) {
          /* set the matrix */
          CsrMatrix *mat= new CsrMatrix;
          mat->num_rows = m;
          mat->num_cols = A->cmap->n;
          mat->num_entries = nnz;
          mat->row_offsets = new THRUSTINTARRAY32(m+1);
          mat->row_offsets->assign(ii, ii + m+1);

          mat->column_indices = new THRUSTINTARRAY32(nnz);
          mat->column_indices->assign(a->j, a->j+nnz);

          mat->values = new THRUSTARRAY(nnz);
          if (a->a) mat->values->assign(a->a, a->a+nnz);

          /* assign the pointer */
          matstruct->mat = mat;
        } else if (hipsparsestruct->format==MAT_HIPSPARSE_ELL || hipsparsestruct->format==MAT_HIPSPARSE_HYB) {
          CsrMatrix *mat= new CsrMatrix;
          mat->num_rows = m;
          mat->num_cols = A->cmap->n;
          mat->num_entries = nnz;
          mat->row_offsets = new THRUSTINTARRAY32(m+1);
          mat->row_offsets->assign(ii, ii + m+1);

          mat->column_indices = new THRUSTINTARRAY32(nnz);
          mat->column_indices->assign(a->j, a->j+nnz);

          mat->values = new THRUSTARRAY(nnz);
          if (a->a) mat->values->assign(a->a, a->a+nnz);

          hipsparseHybMat_t hybMat;
          stat = hipsparseCreateHybMat(&hybMat);CHKERRHIPSPARSE(stat);
          hipsparseHybPartition_t partition = hipsparsestruct->format==MAT_HIPSPARSE_ELL ?
            HIPSPARSE_HYB_PARTITION_MAX : HIPSPARSE_HYB_PARTITION_AUTO;
          stat = hipsparse_csr2hyb(hipsparsestruct->handle, mat->num_rows, mat->num_cols,
              matstruct->descr, mat->values->data().get(),
              mat->row_offsets->data().get(),
              mat->column_indices->data().get(),
              hybMat, 0, partition);CHKERRHIPSPARSE(stat);
          /* assign the pointer */
          matstruct->mat = hybMat;

          if (mat) {
            if (mat->values) delete (THRUSTARRAY*)mat->values;
            if (mat->column_indices) delete (THRUSTINTARRAY32*)mat->column_indices;
            if (mat->row_offsets) delete (THRUSTINTARRAY32*)mat->row_offsets;
            delete (CsrMatrix*)mat;
          }
        }

        /* assign the compressed row indices */
        if (a->compressedrow.use) {
          hipsparsestruct->workVector = new THRUSTARRAY(m);
          matstruct->cprowIndices    = new THRUSTINTARRAY(m);
          matstruct->cprowIndices->assign(ridx,ridx+m);
          tmp = m;
        } else {
          hipsparsestruct->workVector = NULL;
          matstruct->cprowIndices    = NULL;
          tmp = 0;
        }
        ierr = PetscLogCpuToGpu(((m+1)+(a->nz))*sizeof(int)+tmp*sizeof(PetscInt)+(3+(a->nz))*sizeof(PetscScalar));CHKERRQ(ierr);

        /* assign the pointer */
        hipsparsestruct->mat = matstruct;
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
      }
      err  = WaitForHIP();CHKERRHIP(err);
      ierr = PetscLogEventEnd(MAT_HIPSPARSECopyToGPU,A,0,0,0);CHKERRQ(ierr);
      hipsparsestruct->nonzerostate = A->nonzerostate;
    }
    if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

struct VecHIPPlusEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<1>(t) + thrust::get<0>(t);
  }
};

struct VecHIPEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

struct VecHIPEqualsReverse
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t);
  }
};

struct MatMathipsparse {
  PetscBool             cisdense;
  PetscScalar           *Bt;
  Mat                   X;
  PetscBool             reusesym; /* hipsparse does not have split symbolic and numeric phases for sparse matmat operations */
  PetscLogDouble        flops;
  CsrMatrix             *Bcsr;
};

static PetscErrorCode MatDestroy_MatMathipsparse(void *data)
{
  PetscErrorCode   ierr;
  MatMathipsparse   *mmdata = (MatMathipsparse *)data;
  hipError_t      cerr;

  PetscFunctionBegin;
  cerr = hipFree(mmdata->Bt);CHKERRHIP(cerr);
  delete mmdata->Bcsr;
  ierr = MatDestroy(&mmdata->X);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(Mat,Mat,Mat,PetscBool,PetscBool);

static PetscErrorCode MatProductNumeric_SeqAIJHIPSPARSE_SeqDENSEHIP(Mat C)
{
  Mat_Product                  *product = C->product;
  Mat                          A,B;
  PetscInt                     m,n,blda,clda;
  PetscBool                    flg,biship;
  Mat_SeqAIJHIPSPARSE           *hipsp;
  rocsparseStatus_t             stat;
  hipsparseOperation_t          opA;
  const PetscScalar            *barray;
  PetscScalar                  *carray;
  PetscErrorCode               ierr;
  MatMathipsparse               *mmdata;
  Mat_SeqAIJHIPSPARSEMultStruct *mat;
  CsrMatrix                    *csrmat;
  hipError_t                  cerr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  mmdata = (MatMathipsparse*)product->data;
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);
  /* currently CopyToGpu does not copy if the matrix is bound to CPU
     Instead of silently accepting the wrong answer, I prefer to raise the error */
  if (A->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a HIPSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  switch (product->type) {
  case MATPRODUCT_AB:
  case MATPRODUCT_PtAP:
    mat = hipsp->mat;
    opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    if (!hipsp->transgen) {
      mat = hipsp->mat;
      opA = HIPSPARSE_OPERATION_TRANSPOSE;
    } else {
      ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
      mat  = hipsp->matTranspose;
      opA  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    }
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
  case MATPRODUCT_RARt:
    mat = hipsp->mat;
    opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    m   = A->rmap->n;
    n   = B->rmap->n;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  if (!mat) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csrmat = (CsrMatrix*)mat->mat;
  /* if the user passed a CPU matrix, copy the data to the GPU */
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSEHIP,&biship);CHKERRQ(ierr);
  if (!bisHIPTSEQDENSEHIP,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);}
  ierr = MatDenseHIPGetArrayRead(B,&barray);CHKERRQ(ierr);

  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    ierr = MatDenseHIPGetArrayWrite(mmdata->X,&carray);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(mmdata->X,&clda);CHKERRQ(ierr);
  } else {
    ierr = MatDenseHIPGetArrayWrite(C,&carray);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  }

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  PetscInt k;
  /* hipsparseXcsrmm does not support transpose on B */
  if (product->type == MATPRODUCT_ABt || product->type == MATPRODUCT_RARt) {
    hipblasHandle_t hipblasv2handle;
    hipblasStatus_t cerr;

    ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
    cerr = hipblasXgeam(hipblasv2handle,HIPBLAS_OP_T,HIPBLAS_OP_T,
                       B->cmap->n,B->rmap->n,
                       &PETSC_HIPSPARSE_ONE ,barray,blda,
                       &PETSC_HIPSPARSE_ZERO,barray,blda,
                       mmdata->Bt,B->cmap->n);CHKERRHIPBLAS(cerr);
    blda = B->cmap->n;
    k    = B->cmap->n;
  } else {
    k    = B->rmap->n;
  }

  /* perform the MatMat operation, op(A) is m x k, op(B) is k x n */
  stat = hipsparse_csr_spmm(hipsp->handle,opA,m,n,k,
                           csrmat->num_entries,mat->alpha_one,mat->descr,
                           csrmat->values->data().get(),
                           csrmat->row_offsets->data().get(),
                           csrmat->column_indices->data().get(),
                           mmdata->Bt ? mmdata->Bt : barray,blda,mat->beta_zero,
                           carray,clda);CHKERRHIPSPARSE(stat);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n*2.0*csrmat->num_entries);CHKERRQ(ierr);
  ierr = MatDenseHIPRestoreArrayRead(B,&barray);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_RARt) {
    ierr = MatDenseHIPRestoreArrayWrite(mmdata->X,&carray);CHKERRQ(ierr);
    ierr = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(B,mmdata->X,C,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  } else if (product->type == MATPRODUCT_PtAP) {
    ierr = MatDenseHIPRestoreArrayWrite(mmdata->X,&carray);CHKERRQ(ierr);
    ierr = MatMatMultNumeric_SeqDenseHIP_SeqDenseHIP_Private(B,mmdata->X,C,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = MatDenseHIPRestoreArrayWrite(C,&carray);CHKERRQ(ierr);
  }
  if (mmdata->cisdense) {
    ierr = MatConvert(C,MATSEQDENSE,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
  }
  if (!biship) {
    ierr = MatConvert(B,MATSEQDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJHIPSPARSE_SeqDENSEHIP(Mat C)
{
  Mat_Product        *product = C->product;
  Mat                A,B;
  PetscInt           m,n;
  PetscBool          cisdense,flg;
  PetscErrorCode     ierr;
  MatMathipsparse     *mmdata;
  Mat_SeqAIJHIPSPARSE *hipsp;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);
  hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  if (hipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");
  switch (product->type) {
  case MATPRODUCT_AB:
    m = A->rmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
    m = A->rmap->n;
    n = B->rmap->n;
    break;
  case MATPRODUCT_PtAP:
    m = B->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_RARt:
    m = B->rmap->n;
    n = B->rmap->n;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  /* if C is of type MATSEQDENSE (CPU), perform the operation on the GPU and then copy on the CPU */
  ierr = PetscObjectTypeCompare((PetscObject)C,MATSEQDENSE,&cisdense);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQDENSEHIP);CHKERRQ(ierr);

  /* product data */
  ierr = PetscNew(&mmdata);CHKERRQ(ierr);
  mmdata->cisdense = cisdense;
  /* for these products we need intermediate storage */
  if (product->type == MATPRODUCT_RARt || product->type == MATPRODUCT_PtAP) {
    ierr = MatCreate(PetscObjectComm((PetscObject)C),&mmdata->X);CHKERRQ(ierr);
    ierr = MatSetType(mmdata->X,MATSEQDENSEHIP);CHKERRQ(ierr);
    if (product->type == MATPRODUCT_RARt) { /* do not preallocate, since the first call to MatDenseHIPGetArray will preallocate on the GPU for us */
      ierr = MatSetSizes(mmdata->X,A->rmap->n,B->rmap->n,A->rmap->n,B->rmap->n);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(mmdata->X,A->rmap->n,B->cmap->n,A->rmap->n,B->cmap->n);CHKERRQ(ierr);
    }
  }
  C->product->data    = mmdata;
  C->product->destroy = MatDestroy_MatMathipsparse;

  C->ops->productnumeric = MatProductNumeric_SeqAIJHIPSPARSE_SeqDENSEHIP;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE(Mat C)
{
  Mat_Product                  *product = C->product;
  Mat                          A,B;
  Mat_SeqAIJHIPSPARSE           *Ahipsp,*Bhipsp,*Chipsp;
  Mat_SeqAIJ                   *c = (Mat_SeqAIJ*)C->data;
  Mat_SeqAIJHIPSPARSEMultStruct *Amat,*Bmat,*Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscBool                    flg;
  PetscErrorCode               ierr;
  rocsparseStatus_t             stat;
  hipError_t                  cerr;
  MatProductType               ptype;
  MatMathipsparse               *mmdata;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  ierr = PetscObjectTypeCompare((PetscObject)C,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for C of type %s",((PetscObject)C)->type_name);
  mmdata = (MatMathipsparse*)C->product->data;
  A = product->A;
  B = product->B;
  if (mmdata->reusesym) { /* this happens when api_user is true, meaning that the matrix values have been already computed in the MatProductSymbolic phase */
    mmdata->reusesym = PETSC_FALSE;
    Chipsp = (Mat_SeqAIJHIPSPARSE*)C->spptr;
    if (Chipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");
    Cmat = Chipsp->mat;
    if (!Cmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing C mult struct for product type %s",MatProductTypes[C->product->type]);
    Ccsr = (CsrMatrix*)Cmat->mat;
    if (!Ccsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing C CSR struct");
    goto finalize;
  }
  if (!c->nz) goto finalize;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for B of type %s",((PetscObject)B)->type_name);
  if (A->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a HIPSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  if (B->boundtocpu) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONG,"Cannot bind to CPU a HIPSPARSE matrix between MatProductSymbolic and MatProductNumeric phases");
  Ahipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Bhipsp = (Mat_SeqAIJHIPSPARSE*)B->spptr;
  Chipsp = (Mat_SeqAIJHIPSPARSE*)C->spptr;
  if (Ahipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");
  if (Bhipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");
  if (Chipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSECopyToGPU(B);CHKERRQ(ierr);

  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    Amat = Ahipsp->mat;
    Bmat = Bhipsp->mat;
    break;
  case MATPRODUCT_AtB:
    Amat = Ahipsp->matTranspose;
    Bmat = Bhipsp->mat;
    break;
  case MATPRODUCT_ABt:
    Amat = Ahipsp->mat;
    Bmat = Bhipsp->matTranspose;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  Cmat = Chipsp->mat;
  if (!Amat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing A mult struct for product type %s",MatProductTypes[ptype]);
  if (!Bmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing B mult struct for product type %s",MatProductTypes[ptype]);
  if (!Cmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing C mult struct for product type %s",MatProductTypes[ptype]);
  Acsr = (CsrMatrix*)Amat->mat;
  Bcsr = mmdata->Bcsr ? mmdata->Bcsr : (CsrMatrix*)Bmat->mat; /* B may be in compressed row storage */
  Ccsr = (CsrMatrix*)Cmat->mat;
  if (!Acsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing A CSR struct");
  if (!Bcsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing B CSR struct");
  if (!Ccsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing C CSR struct");
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  stat = hipsparse_csr_spgemm(Chipsp->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             Acsr->num_rows, Bcsr->num_cols, Acsr->num_cols,
                             Amat->descr, Acsr->num_entries, Acsr->values->data().get(), Acsr->row_offsets->data().get(), Acsr->column_indices->data().get(),
                             Bmat->descr, Bcsr->num_entries, Bcsr->values->data().get(), Bcsr->row_offsets->data().get(), Bcsr->column_indices->data().get(),
                             Cmat->descr, Ccsr->values->data().get(), Ccsr->row_offsets->data().get(), Ccsr->column_indices->data().get());CHKERRHIPSPARSE(stat);
  ierr = PetscLogGpuFlops(mmdata->flops);CHKERRQ(ierr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  C->offloadmask = PETSC_OFFLOAD_GPU;
finalize:
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  ierr = PetscInfo3(C,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",C->rmap->n,C->cmap->n,c->nz);CHKERRQ(ierr);
  ierr = PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(C,"Maximum nonzeros in any row is %D\n",c->rmax);CHKERRQ(ierr);
  c->reallocs         = 0;
  C->info.mallocs    += 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE(Mat C)
{
  Mat_Product                  *product = C->product;
  Mat                          A,B;
  Mat_SeqAIJHIPSPARSE           *Ahipsp,*Bhipsp,*Chipsp;
  Mat_SeqAIJ                   *a,*b,*c;
  Mat_SeqAIJHIPSPARSEMultStruct *Amat,*Bmat,*Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscInt                     i,j,m,n,k;
  PetscBool                    flg;
  PetscErrorCode               ierr;
  rocsparseStatus_t             stat;
  hipError_t                  cerr;
  MatProductType               ptype;
  MatMathipsparse               *mmdata;
  PetscLogDouble               flops;
  PetscBool                    biscompressed,ciscompressed;
  int                          cnz;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A    = product->A;
  B    = product->B;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQAIJHIPSPARSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for B of type %s",((PetscObject)B)->type_name);
  a = (Mat_SeqAIJ*)A->data;
  b = (Mat_SeqAIJ*)B->data;
  Ahipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Bhipsp = (Mat_SeqAIJHIPSPARSE*)B->spptr;
  if (Ahipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");
  if (Bhipsp->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Only for MAT_HIPSPARSE_CSR format");

  /* product data */
  ierr = PetscNew(&mmdata);CHKERRQ(ierr);
  C->product->data    = mmdata;
  C->product->destroy = MatDestroy_MatMathipsparse;

  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSECopyToGPU(B);CHKERRQ(ierr);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  biscompressed = PETSC_FALSE;
  ciscompressed = PETSC_FALSE;
  switch (ptype) {
  case MATPRODUCT_AB:
    m = A->rmap->n;
    n = B->cmap->n;
    k = A->cmap->n;
    Amat = Ahipsp->mat;
    Bmat = Bhipsp->mat;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_AtB:
    m = A->cmap->n;
    n = B->cmap->n;
    k = A->rmap->n;
    ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
    Amat = Ahipsp->matTranspose;
    Bmat = Bhipsp->mat;
    if (b->compressedrow.use) biscompressed = PETSC_TRUE;
    break;
  case MATPRODUCT_ABt:
    m = A->rmap->n;
    n = B->rmap->n;
    k = A->cmap->n;
    ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(B);CHKERRQ(ierr);
    Amat = Ahipsp->mat;
    Bmat = Bhipsp->matTranspose;
    if (a->compressedrow.use) ciscompressed = PETSC_TRUE;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  /* create hipsparse matrix */
  ierr  = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr  = MatSetType(C,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  c     = (Mat_SeqAIJ*)C->data;
  Chipsp = (Mat_SeqAIJHIPSPARSE*)C->spptr;
  Cmat  = new Mat_SeqAIJHIPSPARSEMultStruct;
  Ccsr  = new CsrMatrix;

  c->compressedrow.use = ciscompressed;
  if (c->compressedrow.use) { /* if a is in compressed row, than c will be in compressed row format */
    c->compressedrow.nrows = a->compressedrow.nrows;
    ierr = PetscMalloc2(c->compressedrow.nrows+1,&c->compressedrow.i,c->compressedrow.nrows,&c->compressedrow.rindex);CHKERRQ(ierr);
    ierr = PetscArraycpy(c->compressedrow.rindex,a->compressedrow.rindex,c->compressedrow.nrows);CHKERRQ(ierr);
    Chipsp->workVector  = new THRUSTARRAY(c->compressedrow.nrows);
    Cmat->cprowIndices = new THRUSTINTARRAY(c->compressedrow.nrows);
    Cmat->cprowIndices->assign(c->compressedrow.rindex,c->compressedrow.rindex + c->compressedrow.nrows);
  } else {
    c->compressedrow.nrows  = 0;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
    Chipsp->workVector      = NULL;
    Cmat->cprowIndices      = NULL;
  }
  Chipsp->nrows    = ciscompressed ? c->compressedrow.nrows : m;
  Chipsp->mat      = Cmat;
  Chipsp->mat->mat = Ccsr;
  Ccsr->num_rows    = Chipsp->nrows;
  Ccsr->num_cols    = n;
  Ccsr->row_offsets = new THRUSTINTARRAY32(Chipsp->nrows+1);
  stat = hipsparseCreateMatDescr(&Cmat->descr);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatIndexBase(Cmat->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
  stat = hipsparseSetMatType(Cmat->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
  cerr = hipMalloc((void **)&(Cmat->alpha_one),sizeof(PetscScalar));CHKERRHIP(cerr);
  cerr = hipMalloc((void **)&(Cmat->beta_zero),sizeof(PetscScalar));CHKERRHIP(cerr);
  cerr = hipMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar));CHKERRHIP(cerr);
  cerr = hipMemcpy(Cmat->alpha_one,&PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
  cerr = hipMemcpy(Cmat->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
  cerr = hipMemcpy(Cmat->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
  if (!Ccsr->num_rows || !Ccsr->num_cols || !a->nz || !b->nz) { /* hipsparse raise errors in different calls when matrices have zero rows/columns! */
    thrust::fill(thrust::device,Ccsr->row_offsets->begin(),Ccsr->row_offsets->end(),0);
    c->nz = 0;
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    Ccsr->values = new THRUSTARRAY(c->nz);
    goto finalizesym;
  }

  if (!Amat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing A mult struct for product type %s",MatProductTypes[ptype]);
  if (!Bmat) SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing B mult struct for product type %s",MatProductTypes[ptype]);
  Acsr = (CsrMatrix*)Amat->mat;
  if (!biscompressed) {
    Bcsr = (CsrMatrix*)Bmat->mat;
  } else { /* we need to use row offsets for the full matrix */
    CsrMatrix *cBcsr = (CsrMatrix*)Bmat->mat;
    Bcsr = new CsrMatrix;
    Bcsr->num_rows       = B->rmap->n;
    Bcsr->num_cols       = cBcsr->num_cols;
    Bcsr->num_entries    = cBcsr->num_entries;
    Bcsr->column_indices = cBcsr->column_indices;
    Bcsr->values         = cBcsr->values;
    if (!Bhipsp->rowoffsets_gpu) {
      Bhipsp->rowoffsets_gpu  = new THRUSTINTARRAY32(B->rmap->n + 1);
      Bhipsp->rowoffsets_gpu->assign(b->i,b->i + B->rmap->n + 1);
      ierr = PetscLogCpuToGpu((B->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
    }
    Bcsr->row_offsets = Bhipsp->rowoffsets_gpu;
    mmdata->Bcsr = Bcsr;
  }
  if (!Acsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing A CSR struct");
  if (!Bcsr) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing B CSR struct");
  /* precompute flops count */
  if (ptype == MATPRODUCT_AB) {
    for (i=0, flops = 0; i<A->rmap->n; i++) {
      const PetscInt st = a->i[i];
      const PetscInt en = a->i[i+1];
      for (j=st; j<en; j++) {
        const PetscInt brow = a->j[j];
        flops += 2.*(b->i[brow+1] - b->i[brow]);
      }
    }
  } else if (ptype == MATPRODUCT_AtB) {
    for (i=0, flops = 0; i<A->rmap->n; i++) {
      const PetscInt anzi = a->i[i+1] - a->i[i];
      const PetscInt bnzi = b->i[i+1] - b->i[i];
      flops += (2.*anzi)*bnzi;
    }
  } else { /* TODO */
    flops = 0.;
  }

  mmdata->flops = flops;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuFlops(mmdata->flops);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
finalizesym:
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
  if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
    PetscInt *d_i = c->i;
    THRUSTINTARRAY ii(Ccsr->row_offsets->size());
    THRUSTINTARRAY jj(Ccsr->column_indices->size());
    ii   = *Ccsr->row_offsets;
    jj   = *Ccsr->column_indices;
    if (ciscompressed) d_i = c->compressedrow.i;
    cerr = hipMemcpy(d_i,ii.data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    cerr = hipMemcpy(c->j,jj.data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
  } else {
    PetscInt *d_i = c->i;
    if (ciscompressed) d_i = c->compressedrow.i;
    cerr = hipMemcpy(d_i,Ccsr->row_offsets->data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    cerr = hipMemcpy(c->j,Ccsr->column_indices->data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
  }
  if (ciscompressed) { /* need to expand host row offsets */
    PetscInt r = 0;
    c->i[0] = 0;
    for (k = 0; k < c->compressedrow.nrows; k++) {
      const PetscInt next = c->compressedrow.rindex[k];
      const PetscInt old = c->compressedrow.i[k];
      for (; r < next; r++) c->i[r+1] = old;
    }
    for (; r < m; r++) c->i[r+1] = c->compressedrow.i[c->compressedrow.nrows];
  }
  ierr = PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size())*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c->ilen);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c->imax);CHKERRQ(ierr);
  c->maxnz = c->nz;
  c->nonzerorowcnt = 0;
  c->rmax = 0;
  for (k = 0; k < m; k++) {
    const PetscInt nn = c->i[k+1] - c->i[k];
    c->ilen[k] = c->imax[k] = nn;
    c->nonzerorowcnt += (PetscInt)!!nn;
    c->rmax = PetscMax(c->rmax,nn);
  }
  ierr = MatMarkDiagonal_SeqAIJ(C);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
  Ccsr->num_entries = c->nz;

  C->nonzerostate++;
  ierr = PetscLayoutSetUp(C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->cmap);CHKERRQ(ierr);
  Chipsp->nonzerostate = C->nonzerostate;
  C->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  C->preallocated  = PETSC_TRUE;
  C->assembled     = PETSC_FALSE;
  C->was_assembled = PETSC_FALSE;
  if (product->api_user && A->offloadmask == PETSC_OFFLOAD_BOTH && B->offloadmask == PETSC_OFFLOAD_BOTH) { /* flag the matrix C values as computed, so that the numeric phase will only call MatAssembly */
    mmdata->reusesym = PETSC_TRUE;
    C->offloadmask   = PETSC_OFFLOAD_GPU;
  }
  C->ops->productnumeric = MatProductNumeric_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat);

/* handles sparse or dense B */
static PetscErrorCode MatProductSetFromOptions_SeqAIJHIPSPARSE(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscErrorCode ierr;
  PetscBool      isdense = PETSC_FALSE,Bishipsp = PETSC_FALSE,Cishipsp = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)product->B,MATSEQDENSE,&isdense);CHKERRQ(ierr);
  if (!product->A->boundtocpu && !product->B->boundtocpu) {
    ierr = PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJHIPSPARSE,&Bishipsp);CHKERRQ(ierr);
  }
  if (product->type == MATPRODUCT_ABC) {
    Cishipsp = PETSC_FALSE;
    if (!product->C->boundtocpu) {
      ierr = PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJHIPSPARSE,&Cishipsp);CHKERRQ(ierr);
    }
  }
  if (isdense) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
     if (product->A->boundtocpu) {
        ierr = MatProductSetFromOptions_SeqAIJ_SeqDense(mat);CHKERRQ(ierr);
      } else {
        mat->ops->productsymbolic = MatProductSymbolic_SeqAIJHIPSPARSE_SeqDENSEHIP;
      }
      break;
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else if (Bishipsp && Cishipsp) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJHIPSPARSE_SeqAIJHIPSPARSE;
      break;
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else { /* fallback for AIJ */
    ierr = MatProductSetFromOptions_SeqAIJ(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJHIPSPARSE(A,xx,NULL,yy,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJHIPSPARSE(A,xx,yy,zz,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultHermitianTranspose_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJHIPSPARSE(A,xx,NULL,yy,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJHIPSPARSE(A,xx,yy,zz,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJHIPSPARSE(A,xx,NULL,yy,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = op(A) x + y. If trans & !herm, op = ^T; if trans & herm, op = ^H; if !trans, op = no-op */
static PetscErrorCode MatMultAddKernel_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz,PetscBool trans,PetscBool herm)
{
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJHIPSPARSE           *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstruct;
  PetscScalar                  *xarray,*zarray,*dptr,*beta,*xptr;
  PetscErrorCode               ierr;
  hipError_t                  cerr;
  rocsparseStatus_t             stat;
  hipsparseOperation_t          opA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  PetscBool                    compressed;

  PetscFunctionBegin;
  if (herm && !trans) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Hermitian and not transpose not supported");
  if (!a->nonzerorowcnt) {
    if (!yy) {ierr = VecSet_SeqHIP(zz,0);CHKERRQ(ierr);}
    else {ierr = VecCopy_SeqHIP(yy,zz);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  /* The line below is necessary due to the operations that modify the matrix on the CPU (axpy, scale, etc) */
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!trans) {
    matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
    if (!matstruct) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"SeqAIJHIPSPARSE does not have a 'mat' (need to fix)");
  } else {
    if (herm || !hipsparsestruct->transgen) {
      opA = herm ? HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE : HIPSPARSE_OPERATION_TRANSPOSE;
      matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
    } else {
      if (!hipsparsestruct->matTranspose) {ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);}
      matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->matTranspose;
    }
  }
  /* Does the matrix use compressed rows (i.e., drop zero rows)? */
  compressed = matstruct->cprowIndices ? PETSC_TRUE : PETSC_FALSE;

  try {
    ierr = VecHIPGetArrayRead(xx,(const PetscScalar**)&xarray);CHKERRQ(ierr);
    if (yy == zz) {ierr = VecHIPGetArray(zz,&zarray);CHKERRQ(ierr);} /* read & write zz, so need to get uptodate zarray on GPU */
    else {ierr = VecHIPGetArrayWrite(zz,&zarray);CHKERRQ(ierr);} /* write zz, so no need to init zarray on GPU */

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (opA == HIPSPARSE_OPERATION_NON_TRANSPOSE) {
      /* z = A x + beta y.
         If A is compressed (with less rows), then Ax is shorter than the full z, so we need a work vector to store Ax.
         When A is non-compressed, and z = y, we can set beta=1 to compute y = Ax + y in one call.
      */
      xptr = xarray;
      dptr = compressed ? hipsparsestruct->workVector->data().get() : zarray;
      beta = (yy == zz && !compressed) ? matstruct->beta_one : matstruct->beta_zero;
    } else {
      /* z = A^T x + beta y
         If A is compressed, then we need a work vector as the shorter version of x to compute A^T x.
         Note A^Tx is of full length, so we set beta to 1.0 if y exists.
       */
      xptr = compressed ? hipsparsestruct->workVector->data().get() : xarray;
      dptr = zarray;
      beta = yy ? matstruct->beta_one : matstruct->beta_zero;
      if (compressed) { /* Scatter x to work vector */
        thrust::device_ptr<PetscScalar> xarr = thrust::device_pointer_cast(xarray);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(xarr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(),
                         VecHIPEqualsReverse());
      }
    }

    /* csr_spmv does y = alpha op(A) x + beta y */
    if (hipsparsestruct->format == MAT_HIPSPARSE_CSR) {
      CsrMatrix *mat = (CsrMatrix*)matstruct->mat;
      stat = hipsparse_csr_spmv(hipsparsestruct->handle, opA,
                               mat->num_rows, mat->num_cols,
                               mat->num_entries, matstruct->alpha_one, matstruct->descr,
                               mat->values->data().get(), mat->row_offsets->data().get(),
                               mat->column_indices->data().get(), xptr, beta,
                               dptr);CHKERRHIPSPARSE(stat);
    } else {
      if (hipsparsestruct->nrows) {
        stat = hipsparse_hyb_spmv(hipsparsestruct->handle, opA,
                                 matstruct->alpha_one, matstruct->descr, hybMat,
                                 xptr, beta,
                                 dptr);CHKERRHIPSPARSE(stat);
      }
    }
    cerr = WaitForHIP();CHKERRHIP(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    if (opA == HIPSPARSE_OPERATION_NON_TRANSPOSE) {
      if (yy) { /* MatMultAdd: zz = A*xx + yy */
        if (compressed) { /* A is compressed. We first copy yy to zz, then ScatterAdd the work vector to zz */
          ierr = VecCopy_SeqHIP(yy,zz);CHKERRQ(ierr); /* zz = yy */
        } else if (zz != yy) { /* A is not compressed. zz already contains A*xx, and we just need to add yy */
          ierr = VecAXPY_SeqHIP(zz,1.0,yy);CHKERRQ(ierr); /* zz += yy */
        }
      } else if (compressed) { /* MatMult: zz = A*xx. A is compressed, so we zero zz first, then ScatterAdd the work vector to zz */
        ierr = VecSet_SeqHIP(zz,0);CHKERRQ(ierr);
      }

      /* ScatterAdd the result from work vector into the full vector when A is compressed */
      if (compressed) {
        thrust::device_ptr<PetscScalar> zptr = thrust::device_pointer_cast(zarray);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(hipsparsestruct->workVector->begin(), thrust::make_permutation_iterator(zptr, matstruct->cprowIndices->begin()))) + matstruct->cprowIndices->size(),
                         VecHIPPlusEquals());
        cerr = WaitForHIP();CHKERRHIP(cerr);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      }
    } else {
      if (yy && yy != zz) {
        ierr = VecAXPY_SeqHIP(zz,1.0,yy);CHKERRQ(ierr); /* zz += yy */
      }
    }
    ierr = VecHIPRestoreArrayRead(xx,(const PetscScalar**)&xarray);CHKERRQ(ierr);
    if (yy == zz) {ierr = VecHIPRestoreArray(zz,&zarray);CHKERRQ(ierr);}
    else {ierr = VecHIPRestoreArrayWrite(zz,&zarray);CHKERRQ(ierr);}
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"HIPSPARSE error: %s", ex);
  }
  if (yy) {
    ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  } else {
    ierr = PetscLogGpuFlops(2.0*a->nz-a->nonzerorowcnt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAddKernel_SeqAIJHIPSPARSE(A,xx,yy,zz,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqAIJHIPSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode              ierr;
  PetscSplitCSRDataStructure  *d_mat = NULL;
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    d_mat = ((Mat_SeqAIJHIPSPARSE*)A->spptr)->deviceMat;
  }
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr); // this does very little if assembled on GPU - call it?
  if (mode == MAT_FLUSH_ASSEMBLY || A->boundtocpu) PetscFunctionReturn(0);
  if (d_mat) {
    A->offloadmask = PETSC_OFFLOAD_GPU;
  }

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@
   MatCreateSeqAIJHIPSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately pushed down
   to NVidia GPUs and use the HIPSPARSE library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradgm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATSEQAIJHIPSPARSE, MATAIJHIPSPARSE
@*/
PetscErrorCode  MatCreateSeqAIJHIPSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJHIPSPARSE(Mat A)
{
  PetscErrorCode              ierr;
  PetscSplitCSRDataStructure  *d_mat = NULL;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    d_mat = ((Mat_SeqAIJHIPSPARSE*)A->spptr)->deviceMat;
    ((Mat_SeqAIJHIPSPARSE*)A->spptr)->deviceMat = NULL;
    ierr = MatSeqAIJHIPSPARSE_Destroy((Mat_SeqAIJHIPSPARSE**)&A->spptr);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJHIPSPARSETriFactors_Destroy((Mat_SeqAIJHIPSPARSETriFactors**)&A->spptr);CHKERRQ(ierr);
  }
  if (d_mat) {
    Mat_SeqAIJ                 *a = (Mat_SeqAIJ*)A->data;
    hipError_t                err;
    PetscSplitCSRDataStructure h_mat;
    ierr = PetscInfo(A,"Have device matrix\n");CHKERRQ(ierr);
    err = hipMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), hipMemcpyDeviceToHost);CHKERRHIP(err);
    if (a->compressedrow.use) {
      err = hipFree(h_mat.diag.i);CHKERRHIP(err);
    }
    err = hipFree(d_mat);CHKERRHIP(err);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHIPSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat,MatType,MatReuse,Mat*);
static PetscErrorCode MatBindToCPU_SeqAIJHIPSPARSE(Mat,PetscBool);
static PetscErrorCode MatDuplicate_SeqAIJHIPSPARSE(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJHIPSPARSE(*B,MATSEQAIJHIPSPARSE,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqAIJHIPSPARSE(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;
  Mat_SeqAIJHIPSPARSE *cy;
  Mat_SeqAIJHIPSPARSE *cx;
  PetscScalar        *ay;
  const PetscScalar  *ax;
  CsrMatrix          *csry,*csrx;
  hipError_t        cerr;

  PetscFunctionBegin;
  cy = (Mat_SeqAIJHIPSPARSE*)Y->spptr;
  cx = (Mat_SeqAIJHIPSPARSE*)X->spptr;
  if (X->ops->axpy != Y->ops->axpy) {
    ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(Y,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* if we are here, it means both matrices are bound to GPU */
  ierr = MatSeqAIJHIPSPARSECopyToGPU(Y);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSECopyToGPU(X);CHKERRQ(ierr);
  if (cy->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_PLIB,"only MAT_HIPSPARSE_CSR supported");
  if (cx->format != MAT_HIPSPARSE_CSR) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_PLIB,"only MAT_HIPSPARSE_CSR supported");
  csry = (CsrMatrix*)cy->mat->mat;
  csrx = (CsrMatrix*)cx->mat->mat;
  /* see if we can turn this into a hipblas axpy */
  if (str != SAME_NONZERO_PATTERN && x->nz == y->nz && !x->compressedrow.use && !y->compressedrow.use) {
    bool eq = thrust::equal(thrust::device,csry->row_offsets->begin(),csry->row_offsets->end(),csrx->row_offsets->begin());
    if (eq) {
      eq = thrust::equal(thrust::device,csry->column_indices->begin(),csry->column_indices->end(),csrx->column_indices->begin());
    }
    if (eq) str = SAME_NONZERO_PATTERN;
  }
  /* spgeam is buggy with one column */
  if (Y->cmap->n == 1 && str != SAME_NONZERO_PATTERN) str = DIFFERENT_NONZERO_PATTERN;

  if (str == SUBSET_NONZERO_PATTERN) {
    rocsparseStatus_t stat;
    PetscScalar      b = 1.0;

    ierr = MatSeqAIJHIPSPARSEGetArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEGetArray(Y,&ay);CHKERRQ(ierr);
    stat = hipsparseSetPointerMode(cy->handle, HIPSPARSE_POINTER_MODE_HOST);CHKERRHIPSPARSE(stat);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    stat = hipsparse_csr_spgeam(cy->handle,Y->rmap->n,Y->cmap->n,
                               &a,cx->mat->descr,x->nz,ax,csrx->row_offsets->data().get(),csrx->column_indices->data().get(),
                               &b,cy->mat->descr,y->nz,ay,csry->row_offsets->data().get(),csry->column_indices->data().get(),
                                  cy->mat->descr,      ay,csry->row_offsets->data().get(),csry->column_indices->data().get());CHKERRHIPSPARSE(stat);
    cerr = WaitForHIP();CHKERRHIP(cerr);
    ierr = PetscLogGpuFlops(x->nz + y->nz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    stat = hipsparseSetPointerMode(cy->handle, HIPSPARSE_POINTER_MODE_DEVICE);CHKERRHIPSPARSE(stat);
    ierr = MatSeqAIJHIPSPARSERestoreArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSERestoreArray(Y,&ay);CHKERRQ(ierr);
    ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
  } else if (str == SAME_NONZERO_PATTERN) {
    hipblasHandle_t hipblasv2handle;
    hipblasStatus_t berr;
    PetscBLASInt   one = 1, bnz = 1;

    ierr = MatSeqAIJHIPSPARSEGetArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEGetArray(Y,&ay);CHKERRQ(ierr);
    ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(x->nz,&bnz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    berr = hipblasXaxpy(hipblasv2handle,bnz,&a,ax,one,ay,one);CHKERRHIPBLAS(berr);
    cerr = WaitForHIP();CHKERRHIP(cerr);
    ierr = PetscLogGpuFlops(2.0*bnz);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSERestoreArrayRead(X,&ax);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSERestoreArray(Y,&ay);CHKERRQ(ierr);
    ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(Y,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqAIJHIPSPARSE(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *y = (Mat_SeqAIJ*)Y->data;
  PetscScalar    *ay;
  hipError_t    cerr;
  hipblasHandle_t hipblasv2handle;
  hipblasStatus_t berr;
  PetscBLASInt   one = 1, bnz = 1;

  PetscFunctionBegin;
  ierr = MatSeqAIJHIPSPARSEGetArray(Y,&ay);CHKERRQ(ierr);
  ierr = PetscHIPBLASGetHandle(&hipblasv2handle);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(y->nz,&bnz);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  berr = hipblasXscal(hipblasv2handle,bnz,&a,ay,one);CHKERRHIPBLAS(berr);
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuFlops(bnz);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSERestoreArray(Y,&ay);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJHIPSPARSE(Mat A)
{
  PetscErrorCode             ierr;
  PetscBool                  both = PETSC_FALSE;
  Mat_SeqAIJ                 *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJHIPSPARSE *spptr = (Mat_SeqAIJHIPSPARSE*)A->spptr;
    if (spptr->mat) {
      CsrMatrix* matrix = (CsrMatrix*)spptr->mat->mat;
      if (matrix->values) {
        both = PETSC_TRUE;
        thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
      }
    }
    if (spptr->matTranspose) {
      CsrMatrix* matrix = (CsrMatrix*)spptr->matTranspose->mat;
      if (matrix->values) {
        thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
      }
    }
  }
  //ierr = MatZeroEntries_SeqAIJ(A);CHKERRQ(ierr);
  ierr = PetscArrayzero(a->a,a->i[A->rmap->n]);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(A);CHKERRQ(ierr);
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;

  PetscFunctionReturn(0);
}

static PetscErrorCode MatBindToCPU_SeqAIJHIPSPARSE(Mat A,PetscBool flg)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype != MAT_FACTOR_NONE) PetscFunctionReturn(0);
  if (flg) {
    ierr = MatSeqAIJHIPSPARSECopyFromGPU(A);CHKERRQ(ierr);

    A->ops->scale                     = MatScale_SeqAIJ;
    A->ops->axpy                      = MatAXPY_SeqAIJ;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJ;
    A->ops->mult                      = MatMult_SeqAIJ;
    A->ops->multadd                   = MatMultAdd_SeqAIJ;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJ;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJ;
    A->ops->multhermitiantranspose    = NULL;
    A->ops->multhermitiantransposeadd = NULL;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJ;
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqdensehip_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqdense_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJ);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C",NULL);CHKERRQ(ierr);
  } else {
    A->ops->scale                     = MatScale_SeqAIJHIPSPARSE;
    A->ops->axpy                      = MatAXPY_SeqAIJHIPSPARSE;
    A->ops->zeroentries               = MatZeroEntries_SeqAIJHIPSPARSE;
    A->ops->mult                      = MatMult_SeqAIJHIPSPARSE;
    A->ops->multadd                   = MatMultAdd_SeqAIJHIPSPARSE;
    A->ops->multtranspose             = MatMultTranspose_SeqAIJHIPSPARSE;
    A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJHIPSPARSE;
    A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJHIPSPARSE;
    A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJHIPSPARSE;
    A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJHIPSPARSE;
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJCopySubArray_C",MatSeqAIJCopySubArray_SeqAIJHIPSPARSE);CHKERRQ(ierr);
    /*TODO:  SOmething is wrong here */
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqdenseHIPIJHIPSPARSE");CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqdense_C",MatProductSetFromOptions_SeqAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_SeqAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_SeqAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_seqaijhipsparse_seqaijhipsparse_C",MatProductSetFromOptions_SeqAIJHIPSPARSE);CHKERRQ(ierr);
  }
  A->boundtocpu = flg;
  a->inode.use = flg;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJHIPSPARSE(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode   ierr;
  rocsparseStatus_t stat;
  Mat              B;

  PetscFunctionBegin;
  ierr = PetscHIPInitializeCheck();CHKERRQ(ierr); /* first use of HIPSPARSE may be via MatConvert */
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  B = *newmat;

  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECHIP,&B->defaultvectype);CHKERRQ(ierr);

  if (reuse != MAT_REUSE_MATRIX && !B->spptr) {
    if (B->factortype == MAT_FACTOR_NONE) {
      Mat_SeqAIJHIPSPARSE *spptr;

      ierr = PetscNew(&spptr);CHKERRQ(ierr);
      spptr->format = MAT_HIPSPARSE_CSR;
      stat = rocsparse_create(&spptr->handle);CHKERRHIPSPARSE(stat);
      B->spptr = spptr;
      spptr->deviceMat = NULL;
    } else {
      Mat_SeqAIJHIPSPARSETriFactors *spptr;

      ierr = PetscNew(&spptr);CHKERRQ(ierr);
      stat = rocsparse_create(&spptr->handle);CHKERRHIPSPARSE(stat);
      B->spptr = spptr;
    }
    B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJHIPSPARSE;
  B->ops->destroy        = MatDestroy_SeqAIJHIPSPARSE;
  B->ops->setfromoptions = MatSetFromOptions_SeqAIJHIPSPARSE;
  B->ops->bindtocpu      = MatBindToCPU_SeqAIJHIPSPARSE;
  B->ops->duplicate      = MatDuplicate_SeqAIJHIPSPARSE;

  ierr = MatBindToCPU_SeqAIJHIPSPARSE(B,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatHIPSPARSESetFormat_C",MatHIPSPARSESetFormat_SeqAIJHIPSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJHIPSPARSE(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJHIPSPARSE(B,MATSEQAIJHIPSPARSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)B);CHKERRQ(ierr);
  ierr = MatSetFromOptions_SeqAIJHIPSPARSE(PetscOptionsObject,B);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQAIJHIPSPARSE - MATAIJHIPSPARSE = "(seq)aijhipsparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on AMD GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require HIP 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the HIPSPARSE library.

   Options Database Keys:
+  -mat_type aijhipsparse - sets the matrix type to "seqaijhipsparse" during a call to MatSetFromOptions()
.  -mat_hipsparse_storage_format csr - sets the storage format of matrices (for MatMult and factors in MatSolve) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_mult_storage_format csr - sets the storage format of matrices (for MatMult) during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

.seealso: MatCreateSeqAIJHIPSPARSE(), MATAIJHIPSPARSE, MatCreateAIJHIPSPARSE(), MatHIPSPARSESetFormat(), MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse(Mat,MatFactorType,Mat*);


PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_HIPSPARSE(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_LU,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_CHOLESKY,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_ILU,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERHIPSPARSE,MATSEQAIJHIPSPARSE,MAT_FACTOR_ICC,MatGetFactor_seqaijhipsparse_hipsparse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSE_Destroy(Mat_SeqAIJHIPSPARSE **hipsparsestruct)
{
  PetscErrorCode   ierr;
  rocsparseStatus_t stat;

  PetscFunctionBegin;
  if (*hipsparsestruct) {
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*hipsparsestruct)->mat,(*hipsparsestruct)->format);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*hipsparsestruct)->matTranspose,(*hipsparsestruct)->format);CHKERRQ(ierr);
    delete (*hipsparsestruct)->workVector;
    delete (*hipsparsestruct)->rowoffsets_gpu;
    delete (*hipsparsestruct)->cooPerm;
    delete (*hipsparsestruct)->cooPerm_a;
    delete (*hipsparsestruct)->csr2csc_i;
    if ((*hipsparsestruct)->handle) {stat = hipsparseDestroy((*hipsparsestruct)->handle);CHKERRHIPSPARSE(stat);}
    ierr = PetscFree(*hipsparsestruct);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CsrMatrix_Destroy(CsrMatrix **mat)
{
  PetscFunctionBegin;
  if (*mat) {
    delete (*mat)->values;
    delete (*mat)->column_indices;
    delete (*mat)->row_offsets;
    delete *mat;
    *mat = 0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSETriFactorStruct **trifactor)
{
  rocsparseStatus_t stat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (*trifactor) {
    if ((*trifactor)->descr) { stat = hipsparseDestroyMatDescr((*trifactor)->descr);CHKERRHIPSPARSE(stat); }
    if ((*trifactor)->solveInfo) { stat = hipsparse_destroy_analysis_info((*trifactor)->solveInfo);CHKERRHIPSPARSE(stat); }
    ierr = CsrMatrix_Destroy(&(*trifactor)->csrMat);CHKERRQ(ierr);
    if ((*trifactor)->solveBuffer)   {hipError_t cerr = hipFree((*trifactor)->solveBuffer);CHKERRHIP(cerr);}
    if ((*trifactor)->AA_h)   {hipError_t cerr = hipHostFree((*trifactor)->AA_h);CHKERRHIP(cerr);}
    ierr = PetscFree(*trifactor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSEMultStruct_Destroy(Mat_SeqAIJHIPSPARSEMultStruct **matstruct,MatHIPSPARSEStorageFormat format)
{
  CsrMatrix        *mat;
  rocsparseStatus_t stat;
  hipError_t      err;

  PetscFunctionBegin;
  if (*matstruct) {
    if ((*matstruct)->mat) {
      if (format==MAT_HIPSPARSE_ELL || format==MAT_HIPSPARSE_HYB) {
        hipsparseHybMat_t hybMat = (hipsparseHybMat_t)(*matstruct)->mat;
        stat = hipsparseDestroyHybMat(hybMat);CHKERRHIPSPARSE(stat);
      } else {
        mat = (CsrMatrix*)(*matstruct)->mat;
        CsrMatrix_Destroy(&mat);
      }
    }
    if ((*matstruct)->descr) { stat = hipsparseDestroyMatDescr((*matstruct)->descr);CHKERRHIPSPARSE(stat); }
    delete (*matstruct)->cprowIndices;
    if ((*matstruct)->alpha_one) { err=hipFree((*matstruct)->alpha_one);CHKERRHIP(err); }
    if ((*matstruct)->beta_zero) { err=hipFree((*matstruct)->beta_zero);CHKERRHIP(err); }
    if ((*matstruct)->beta_one)  { err=hipFree((*matstruct)->beta_one);CHKERRHIP(err); }

    delete *matstruct;
    *matstruct = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Reset(Mat_SeqAIJHIPSPARSETriFactors** trifactors)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*trifactors) {
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtr);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtr);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->loTriFactorPtrTranspose);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&(*trifactors)->upTriFactorPtrTranspose);CHKERRQ(ierr);
    delete (*trifactors)->rpermIndices;
    delete (*trifactors)->cpermIndices;
    delete (*trifactors)->workVector;
    (*trifactors)->rpermIndices = NULL;
    (*trifactors)->cpermIndices = NULL;
    (*trifactors)->workVector = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJHIPSPARSETriFactors_Destroy(Mat_SeqAIJHIPSPARSETriFactors** trifactors)
{
  PetscErrorCode   ierr;
  rocsparse_handle handle;
  rocsparseStatus_t stat;

  PetscFunctionBegin;
  if (*trifactors) {
    ierr = MatSeqAIJHIPSPARSETriFactors_Reset(trifactors);CHKERRQ(ierr);
    if (handle = (*trifactors)->handle) {
      stat = hipsparseDestroy(handle);CHKERRHIPSPARSE(stat);
    }
    ierr = PetscFree(*trifactors);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct IJCompare
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct IJEqual
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<PetscInt, PetscInt> &t1, const thrust::tuple<PetscInt, PetscInt> &t2)
  {
    if (t1.get<0>() != t2.get<0>() || t1.get<1>() != t2.get<1>()) return false;
    return true;
  }
};

struct IJDiff
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t1 == t2 ? 0 : 1;
  }
};

struct IJSum
{
  __host__ __device__
  inline PetscInt operator() (const PetscInt &t1, const PetscInt &t2)
  {
    return t1||t2;
  }
};

#include <thrust/iterator/discard_iterator.h>
PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_SeqAIJHIPSPARSE                    *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJ                            *a = (Mat_SeqAIJ*)A->data;
  THRUSTARRAY                           *cooPerm_v = NULL;
  thrust::device_ptr<const PetscScalar> d_v;
  CsrMatrix                             *matrix;
  PetscErrorCode                        ierr;
  hipError_t                           cerr;
  PetscInt                              n;

  PetscFunctionBegin;
  if (!hipsp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing HIPSPARSE struct");
  if (!hipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing HIPSPARSE CsrMatrix");
  if (!hipsp->cooPerm) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  matrix = (CsrMatrix*)hipsp->mat->mat;
  if (!matrix->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing HIP memory");
  if (!v) {
    if (imode == INSERT_VALUES) thrust::fill(thrust::device,matrix->values->begin(),matrix->values->end(),0.);
    goto finalize;
  }
  n = hipsp->cooPerm->size();
  if (ishipMem(v)) {
    d_v = thrust::device_pointer_cast(v);
  } else {
    cooPerm_v = new THRUSTARRAY(n);
    cooPerm_v->assign(v,v+n);
    d_v = cooPerm_v->data();
    ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (imode == ADD_VALUES) { /* ADD VALUES means add to existing ones */
    if (hipsp->cooPerm_a) {
      THRUSTARRAY *cooPerm_w = new THRUSTARRAY(matrix->values->size());
      auto vbit = thrust::make_permutation_iterator(d_v,hipsp->cooPerm->begin());
      thrust::reduce_by_key(hipsp->cooPerm_a->begin(),hipsp->cooPerm_a->end(),vbit,thrust::make_discard_iterator(),cooPerm_w->begin(),thrust::equal_to<PetscInt>(),thrust::plus<PetscScalar>());
      thrust::transform(cooPerm_w->begin(),cooPerm_w->end(),matrix->values->begin(),matrix->values->begin(),thrust::plus<PetscScalar>());
      delete cooPerm_w;
    } else {
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,hipsp->cooPerm->begin()),
                                                                matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,hipsp->cooPerm->end()),
                                                                matrix->values->end()));
      thrust::for_each(zibit,zieit,VecHIPPlusEquals());
    }
  } else {
    if (hipsp->cooPerm_a) { /* repeated entries in COO, with INSERT_VALUES -> reduce */
      auto vbit = thrust::make_permutation_iterator(d_v,hipsp->cooPerm->begin());
      thrust::reduce_by_key(hipsp->cooPerm_a->begin(),hipsp->cooPerm_a->end(),vbit,thrust::make_discard_iterator(),matrix->values->begin(),thrust::equal_to<PetscInt>(),thrust::plus<PetscScalar>());
    } else {
      auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,hipsp->cooPerm->begin()),
                                                                matrix->values->begin()));
      auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,hipsp->cooPerm->end()),
                                                                matrix->values->end()));
      thrust::for_each(zibit,zieit,VecHIPEquals());
    }
  }
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
finalize:
  delete cooPerm_v;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  ierr = PetscInfo3(A,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",A->rmap->n,A->cmap->n,a->nz);CHKERRQ(ierr);
  ierr = PetscInfo(A,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Maximum nonzeros in any row is %D\n",a->rmax);CHKERRQ(ierr);
  a->reallocs         = 0;
  A->info.mallocs    += 0;
  A->info.nz_unneeded = 0;
  A->assembled = A->was_assembled = PETSC_TRUE;
  A->num_ass++;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSEInvalidateTranspose(Mat A, PetscBool destroy)
{
  Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  if (!hipsp) PetscFunctionReturn(0);
  if (destroy) {
    ierr = MatSeqAIJHIPSPARSEMultStruct_Destroy(&hipsp->matTranspose,hipsp->format);CHKERRQ(ierr);
    delete hipsp->csr2csc_i;
    hipsp->csr2csc_i = NULL;
  }
  hipsp->transupdated = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#include <thrust/binary_search.h>
PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat A, PetscInt n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  PetscErrorCode     ierr;
  Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscInt           cooPerm_n, nzr = 0;
  hipError_t        cerr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  cooPerm_n = hipsp->cooPerm ? hipsp->cooPerm->size() : 0;
  if (n != cooPerm_n) {
    delete hipsp->cooPerm;
    delete hipsp->cooPerm_a;
    hipsp->cooPerm = NULL;
    hipsp->cooPerm_a = NULL;
  }
  if (n) {
    THRUSTINTARRAY d_i(n);
    THRUSTINTARRAY d_j(n);
    THRUSTINTARRAY ii(A->rmap->n);

    if (!hipsp->cooPerm)   { hipsp->cooPerm   = new THRUSTINTARRAY(n); }
    if (!hipsp->cooPerm_a) { hipsp->cooPerm_a = new THRUSTINTARRAY(n); }

    ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
    d_i.assign(coo_i,coo_i+n);
    d_j.assign(coo_j,coo_j+n);
    auto fkey = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin()));
    auto ekey = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end()));

    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    thrust::sequence(thrust::device, hipsp->cooPerm->begin(), hipsp->cooPerm->end(), 0);
    thrust::sort_by_key(fkey, ekey, hipsp->cooPerm->begin(), IJCompare());
    *hipsp->cooPerm_a = d_i;
    THRUSTINTARRAY w = d_j;

    auto nekey = thrust::unique(fkey, ekey, IJEqual());
    if (nekey == ekey) { /* all entries are unique */
      delete hipsp->cooPerm_a;
      hipsp->cooPerm_a = NULL;
    } else { /* I couldn't come up with a more elegant algorithm */
      adjacent_difference(hipsp->cooPerm_a->begin(),hipsp->cooPerm_a->end(),hipsp->cooPerm_a->begin(),IJDiff());
      adjacent_difference(w.begin(),w.end(),w.begin(),IJDiff());
      (*hipsp->cooPerm_a)[0] = 0;
      w[0] = 0;
      thrust::transform(hipsp->cooPerm_a->begin(),hipsp->cooPerm_a->end(),w.begin(),hipsp->cooPerm_a->begin(),IJSum());
      thrust::inclusive_scan(hipsp->cooPerm_a->begin(),hipsp->cooPerm_a->end(),hipsp->cooPerm_a->begin(),thrust::plus<PetscInt>());
    }
    thrust::counting_iterator<PetscInt> search_begin(0);
    thrust::upper_bound(d_i.begin(), nekey.get_iterator_tuple().get<0>(),
                        search_begin, search_begin + A->rmap->n,
                        ii.begin());
    cerr = WaitForHIP();CHKERRHIP(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

    ierr = MatSeqXAIJFreeAIJ(A,&a->a,&a->j,&a->i);CHKERRQ(ierr);
    a->singlemalloc = PETSC_FALSE;
    a->free_a       = PETSC_TRUE;
    a->free_ij      = PETSC_TRUE;
    ierr = PetscMalloc1(A->rmap->n+1,&a->i);CHKERRQ(ierr);
    a->i[0] = 0;
    cerr = hipMemcpy(a->i+1,ii.data().get(),A->rmap->n*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    a->nz = a->maxnz = a->i[A->rmap->n];
    a->rmax = 0;
    ierr = PetscMalloc1(a->nz,&a->a);CHKERRQ(ierr);
    ierr = PetscMalloc1(a->nz,&a->j);CHKERRQ(ierr);
    cerr = hipMemcpy(a->j,d_j.data().get(),a->nz*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    if (!a->ilen) { ierr = PetscMalloc1(A->rmap->n,&a->ilen);CHKERRQ(ierr); }
    if (!a->imax) { ierr = PetscMalloc1(A->rmap->n,&a->imax);CHKERRQ(ierr); }
    for (PetscInt i = 0; i < A->rmap->n; i++) {
      const PetscInt nnzr = a->i[i+1] - a->i[i];
      nzr += (PetscInt)!!(nnzr);
      a->ilen[i] = a->imax[i] = nnzr;
      a->rmax = PetscMax(a->rmax,nnzr);
    }
    a->nonzerorowcnt = nzr;
    A->preallocated = PETSC_TRUE;
    ierr = PetscLogGpuToCpu((A->rmap->n+a->nz)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJSetPreallocation(A,0,NULL);CHKERRQ(ierr);
  }
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  /* We want to allocate the HIPSPARSE struct for matvec now.
     The code is so convoluted now that I prefer to copy zeros */
  ierr = PetscArrayzero(a->a,a->nz);CHKERRQ(ierr);
  ierr = MatCheckCompressedRow(A,nzr,&a->compressedrow,a->i,A->rmap->n,0.6);CHKERRQ(ierr);
  A->offloadmask = PETSC_OFFLOAD_CPU;
  A->nonzerostate++;
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_TRUE);CHKERRQ(ierr);

  A->assembled = PETSC_FALSE;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSEGetArrayRead(Mat A, const PetscScalar** a)
{
  Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  if (hipsp->format == MAT_HIPSPARSE_ELL || hipsp->format == MAT_HIPSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!hipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix*)hipsp->mat->mat;
  if (!csr->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing HIP memory");
  *a = csr->values->data().get();
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayRead(Mat A, const PetscScalar** a)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  *a = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSEGetArray(Mat A, PetscScalar** a)
{
  Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  if (hipsp->format == MAT_HIPSPARSE_ELL || hipsp->format == MAT_HIPSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
  if (!hipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix*)hipsp->mat->mat;
  if (!csr->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing HIP memory");
  *a = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSERestoreArray(Mat A, PetscScalar** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  *a = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSEGetArrayWrite(Mat A, PetscScalar** a)
{
  Mat_SeqAIJHIPSPARSE *hipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr;
  CsrMatrix          *csr;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  if (hipsp->format == MAT_HIPSPARSE_ELL || hipsp->format == MAT_HIPSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (!hipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
  csr = (CsrMatrix*)hipsp->mat->mat;
  if (!csr->values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing HIP memory");
  *a = csr->values->data().get();
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(A,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayWrite(Mat A, PetscScalar** a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(a,2);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  *a = NULL;
  PetscFunctionReturn(0);
}

struct IJCompare4
{
  __host__ __device__
  inline bool operator() (const thrust::tuple<int, int, PetscScalar, int> &t1, const thrust::tuple<int, int, PetscScalar, int> &t2)
  {
    if (t1.get<0>() < t2.get<0>()) return true;
    if (t1.get<0>() == t2.get<0>()) return t1.get<1>() < t2.get<1>();
    return false;
  }
};

struct Shift
{
  int _shift;

  Shift(int shift) : _shift(shift) {}
  __host__ __device__
  inline int operator() (const int &c)
  {
    return c + _shift;
  }
};

/* merges to SeqAIJHIPSPARSE matrices, [A';B']' operation in matlab notation */
PetscErrorCode MatSeqAIJHIPSPARSEMergeMats(Mat A,Mat B,MatReuse reuse,Mat* C)
{
  PetscErrorCode               ierr;
  Mat_SeqAIJ                   *a = (Mat_SeqAIJ*)A->data, *b = (Mat_SeqAIJ*)B->data, *c;
  Mat_SeqAIJHIPSPARSE           *Ahipsp = (Mat_SeqAIJHIPSPARSE*)A->spptr, *Bhipsp = (Mat_SeqAIJHIPSPARSE*)B->spptr, *Chipsp;
  Mat_SeqAIJHIPSPARSEMultStruct *Cmat;
  CsrMatrix                    *Acsr,*Bcsr,*Ccsr;
  PetscInt                     Annz,Bnnz;
  rocsparseStatus_t             stat;
  PetscInt                     i,m,n,zero = 0;
  hipError_t                  cerr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidPointer(C,4);
  PetscCheckTypeName(A,MATSEQAIJHIPSPARSE);
  PetscCheckTypeName(B,MATSEQAIJHIPSPARSE);
  if (A->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %D != %D",A->rmap->n,B->rmap->n);
  if (reuse == MAT_INPLACE_MATRIX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_INPLACE_MATRIX not supported");
  if (Ahipsp->format == MAT_HIPSPARSE_ELL || Ahipsp->format == MAT_HIPSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (Bhipsp->format == MAT_HIPSPARSE_ELL || Bhipsp->format == MAT_HIPSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
  if (reuse == MAT_INITIAL_MATRIX) {
    m     = A->rmap->n;
    n     = A->cmap->n + B->cmap->n;
    ierr  = MatCreate(PETSC_COMM_SELF,C);CHKERRQ(ierr);
    ierr  = MatSetSizes(*C,m,n,m,n);CHKERRQ(ierr);
    ierr  = MatSetType(*C,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
    c     = (Mat_SeqAIJ*)(*C)->data;
    Chipsp = (Mat_SeqAIJHIPSPARSE*)(*C)->spptr;
    Cmat  = new Mat_SeqAIJHIPSPARSEMultStruct;
    Ccsr  = new CsrMatrix;
    Cmat->cprowIndices      = NULL;
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.nrows  = 0;
    c->compressedrow.i      = NULL;
    c->compressedrow.rindex = NULL;
    Chipsp->workVector       = NULL;
    Chipsp->nrows    = m;
    Chipsp->mat      = Cmat;
    Chipsp->mat->mat = Ccsr;
    Ccsr->num_rows  = m;
    Ccsr->num_cols  = n;
    stat = rocsparse_create_mat_descr(&Cmat->descr);CHKERRHIPSPARSE(stat);
    stat = rocspase_set_mat_index_base(Cmat->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
    stat = rocsparse_set_mat_type(Cmat->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
    cerr = hipMalloc((void **)&(Cmat->alpha_one),sizeof(PetscScalar));CHKERRHIP(cerr);
    cerr = hipMalloc((void **)&(Cmat->beta_zero),sizeof(PetscScalar));CHKERRHIP(cerr);
    cerr = hipMalloc((void **)&(Cmat->beta_one), sizeof(PetscScalar));CHKERRHIP(cerr);
    cerr = hipMemcpy(Cmat->alpha_one,&PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
    cerr = hipMemcpy(Cmat->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
    cerr = hipMemcpy(Cmat->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
    ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSECopyToGPU(B);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(A);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSEGenerateTransposeForMult(B);CHKERRQ(ierr);
    if (!Ahipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
    if (!Bhipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");

    Acsr = (CsrMatrix*)Ahipsp->mat->mat;
    Bcsr = (CsrMatrix*)Bhipsp->mat->mat;
    Annz = (PetscInt)Acsr->column_indices->size();
    Bnnz = (PetscInt)Bcsr->column_indices->size();
    c->nz = Annz + Bnnz;
    Ccsr->row_offsets = new THRUSTINTARRAY32(m+1);
    Ccsr->column_indices = new THRUSTINTARRAY32(c->nz);
    Ccsr->values = new THRUSTARRAY(c->nz);
    Ccsr->num_entries = c->nz;
    Chipsp->cooPerm = new THRUSTINTARRAY(c->nz);
    if (c->nz) {
      auto Acoo = new THRUSTINTARRAY32(Annz);
      auto Bcoo = new THRUSTINTARRAY32(Bnnz);
      auto Ccoo = new THRUSTINTARRAY32(c->nz);
      THRUSTINTARRAY32 *Aroff,*Broff;

      if (a->compressedrow.use) { /* need full row offset */
        if (!Ahipsp->rowoffsets_gpu) {
          Ahipsp->rowoffsets_gpu  = new THRUSTINTARRAY32(A->rmap->n + 1);
          Ahipsp->rowoffsets_gpu->assign(a->i,a->i + A->rmap->n + 1);
          ierr = PetscLogCpuToGpu((A->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
        }
        Aroff = Ahipsp->rowoffsets_gpu;
      } else Aroff = Acsr->row_offsets;
      if (b->compressedrow.use) { /* need full row offset */
        if (!Bhipsp->rowoffsets_gpu) {
          Bhipsp->rowoffsets_gpu  = new THRUSTINTARRAY32(B->rmap->n + 1);
          Bhipsp->rowoffsets_gpu->assign(b->i,b->i + B->rmap->n + 1);
          ierr = PetscLogCpuToGpu((B->rmap->n + 1)*sizeof(PetscInt));CHKERRQ(ierr);
        }
        Broff = Bhipsp->rowoffsets_gpu;
      } else Broff = Bcsr->row_offsets;
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      stat = rocsparseXcsr2coo(Ahipsp->handle,
                              Aroff->data().get(),
                              Annz,
                              m,
                              Acoo->data().get(),
                              HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      stat = rocsparseXcsr2coo(Bhipsp->handle,
                              Broff->data().get(),
                              Bnnz,
                              m,
                              Bcoo->data().get(),
                              HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      /* Issues when using bool with large matrices on SUMMIT 10.2.89 */
      auto Aperm = thrust::make_constant_iterator(1);
      auto Bperm = thrust::make_constant_iterator(0);
      /* there are issues instantiating the merge operation using a transform iterator for the columns of B */
      auto Bcib = Bcsr->column_indices->begin();
      auto Bcie = Bcsr->column_indices->end();
      thrust::transform(Bcib,Bcie,Bcib,Shift(A->cmap->n));
      auto wPerm = new THRUSTINTARRAY32(Annz+Bnnz);
      auto Azb = thrust::make_zip_iterator(thrust::make_tuple(Acoo->begin(),Acsr->column_indices->begin(),Acsr->values->begin(),Aperm));
      auto Aze = thrust::make_zip_iterator(thrust::make_tuple(Acoo->end(),Acsr->column_indices->end(),Acsr->values->end(),Aperm));
      auto Bzb = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->begin(),Bcib,Bcsr->values->begin(),Bperm));
      auto Bze = thrust::make_zip_iterator(thrust::make_tuple(Bcoo->end(),Bcie,Bcsr->values->end(),Bperm));
      auto Czb = thrust::make_zip_iterator(thrust::make_tuple(Ccoo->begin(),Ccsr->column_indices->begin(),Ccsr->values->begin(),wPerm->begin()));
      auto p1 = Chipsp->cooPerm->begin();
      auto p2 = Chipsp->cooPerm->begin();
      thrust::advance(p2,Annz);
      PetscStackCallThrust(thrust::merge(thrust::device,Azb,Aze,Bzb,Bze,Czb,IJCompare4()));
      auto cci = thrust::make_counting_iterator(zero);
      auto cce = thrust::make_counting_iterator(c->nz);
      auto pred = thrust::identity<int>();
      PetscStackCallThrust(thrust::copy_if(thrust::device,cci,cce,wPerm->begin(),p1,pred));
      PetscStackCallThrust(thrust::remove_copy_if(thrust::device,cci,cce,wPerm->begin(),p2,pred));
      stat = rocsparseXcoo2csr(Chipsp->handle,
                              Ccoo->data().get(),
                              c->nz,
                              m,
                              Ccsr->row_offsets->data().get(),
                              HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
      cerr = WaitForHIP();CHKERRHIP(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      delete wPerm;
      delete Acoo;
      delete Bcoo;
      delete Ccoo;
      if (Ahipsp->transgen && Bhipsp->transgen) { /* if A and B have the transpose, generate C transpose too */
        PetscBool AT = Ahipsp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bhipsp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        Mat_SeqAIJHIPSPARSEMultStruct *CmatT = new Mat_SeqAIJHIPSPARSEMultStruct;
        CsrMatrix *CcsrT = new CsrMatrix;
        CsrMatrix *AcsrT = AT ? (CsrMatrix*)Ahipsp->matTranspose->mat : NULL;
        CsrMatrix *BcsrT = BT ? (CsrMatrix*)Bhipsp->matTranspose->mat : NULL;

        Chipsp->transgen = PETSC_TRUE;
        Chipsp->transupdated = PETSC_TRUE;
        Chipsp->rowoffsets_gpu = NULL;
        CmatT->cprowIndices = NULL;
        CmatT->mat = CcsrT;
        CcsrT->num_rows = n;
        CcsrT->num_cols = m;
        CcsrT->num_entries = c->nz;

        CcsrT->row_offsets = new THRUSTINTARRAY32(n+1);
        CcsrT->column_indices = new THRUSTINTARRAY32(c->nz);
        CcsrT->values = new THRUSTARRAY(c->nz);

        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        auto rT = CcsrT->row_offsets->begin();
        if (AT) {
          rT = thrust::copy(AcsrT->row_offsets->begin(),AcsrT->row_offsets->end(),rT);
          thrust::advance(rT,-1);
        }
        if (BT) {
          auto titb = thrust::make_transform_iterator(BcsrT->row_offsets->begin(),Shift(a->nz));
          auto tite = thrust::make_transform_iterator(BcsrT->row_offsets->end(),Shift(a->nz));
          thrust::copy(titb,tite,rT);
        }
        auto cT = CcsrT->column_indices->begin();
        if (AT) cT = thrust::copy(AcsrT->column_indices->begin(),AcsrT->column_indices->end(),cT);
        if (BT) thrust::copy(BcsrT->column_indices->begin(),BcsrT->column_indices->end(),cT);
        auto vT = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(),AcsrT->values->end(),vT);
        if (BT) thrust::copy(BcsrT->values->begin(),BcsrT->values->end(),vT);
        cerr = WaitForHIP();CHKERRHIP(cerr);
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

        stat = rocsparse_create_mat_descr(&CmatT->descr);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_index_base(CmatT->descr, HIPSPARSE_INDEX_BASE_ZERO);CHKERRHIPSPARSE(stat);
        stat = rocsparse_set_mat_type(CmatT->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);CHKERRHIPSPARSE(stat);
        cerr = hipMalloc((void **)&(CmatT->alpha_one),sizeof(PetscScalar));CHKERRHIP(cerr);
        cerr = hipMalloc((void **)&(CmatT->beta_zero),sizeof(PetscScalar));CHKERRHIP(cerr);
        cerr = hipMalloc((void **)&(CmatT->beta_one), sizeof(PetscScalar));CHKERRHIP(cerr);
        cerr = hipMemcpy(CmatT->alpha_one,&PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
        cerr = hipMemcpy(CmatT->beta_zero,&PETSC_HIPSPARSE_ZERO,sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
        cerr = hipMemcpy(CmatT->beta_one, &PETSC_HIPSPARSE_ONE, sizeof(PetscScalar),hipMemcpyHostToDevice);CHKERRHIP(cerr);
        Chipsp->matTranspose = CmatT;
      }
    }

    c->singlemalloc = PETSC_FALSE;
    c->free_a       = PETSC_TRUE;
    c->free_ij      = PETSC_TRUE;
    ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
    ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
    if (PetscDefined(USE_64BIT_INDICES)) { /* 32 to 64 bit conversion on the GPU and then copy to host (lazy) */
      THRUSTINTARRAY ii(Ccsr->row_offsets->size());
      THRUSTINTARRAY jj(Ccsr->column_indices->size());
      ii   = *Ccsr->row_offsets;
      jj   = *Ccsr->column_indices;
      cerr = hipMemcpy(c->i,ii.data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
      cerr = hipMemcpy(c->j,jj.data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    } else {
      cerr = hipMemcpy(c->i,Ccsr->row_offsets->data().get(),Ccsr->row_offsets->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
      cerr = hipMemcpy(c->j,Ccsr->column_indices->data().get(),Ccsr->column_indices->size()*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    }
    ierr = PetscLogGpuToCpu((Ccsr->column_indices->size() + Ccsr->row_offsets->size())*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc1(m,&c->ilen);CHKERRQ(ierr);
    ierr = PetscMalloc1(m,&c->imax);CHKERRQ(ierr);
    c->maxnz = c->nz;
    c->nonzerorowcnt = 0;
    c->rmax = 0;
    for (i = 0; i < m; i++) {
      const PetscInt nn = c->i[i+1] - c->i[i];
      c->ilen[i] = c->imax[i] = nn;
      c->nonzerorowcnt += (PetscInt)!!nn;
      c->rmax = PetscMax(c->rmax,nn);
    }
    ierr = MatMarkDiagonal_SeqAIJ(*C);CHKERRQ(ierr);
    ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
    (*C)->nonzerostate++;
    ierr = PetscLayoutSetUp((*C)->rmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp((*C)->cmap);CHKERRQ(ierr);
    Chipsp->nonzerostate = (*C)->nonzerostate;
    (*C)->preallocated  = PETSC_TRUE;
  } else {
    if ((*C)->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Invalid number or rows %D != %D",(*C)->rmap->n,B->rmap->n);
    c = (Mat_SeqAIJ*)(*C)->data;
    if (c->nz) {
      Chipsp = (Mat_SeqAIJHIPSPARSE*)(*C)->spptr;
      if (!Chipsp->cooPerm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cooPerm");
      if (Chipsp->format == MAT_HIPSPARSE_ELL || Chipsp->format == MAT_HIPSPARSE_HYB) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
      if (Chipsp->nonzerostate != (*C)->nonzerostate) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Wrong nonzerostate");
      ierr = MatSeqAIJHIPSPARSECopyToGPU(A);CHKERRQ(ierr);
      ierr = MatSeqAIJHIPSPARSECopyToGPU(B);CHKERRQ(ierr);
      if (!Ahipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
      if (!Bhipsp->mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Mat_SeqAIJHIPSPARSEMultStruct");
      Acsr = (CsrMatrix*)Ahipsp->mat->mat;
      Bcsr = (CsrMatrix*)Bhipsp->mat->mat;
      Ccsr = (CsrMatrix*)Chipsp->mat->mat;
      if (Acsr->num_entries != (PetscInt)Acsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"A nnz %D != %D",Acsr->num_entries,(PetscInt)Acsr->values->size());
      if (Bcsr->num_entries != (PetscInt)Bcsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"B nnz %D != %D",Bcsr->num_entries,(PetscInt)Bcsr->values->size());
      if (Ccsr->num_entries != (PetscInt)Ccsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"C nnz %D != %D",Ccsr->num_entries,(PetscInt)Ccsr->values->size());
      if (Ccsr->num_entries != Acsr->num_entries + Bcsr->num_entries) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_COR,"C nnz %D != %D + %D",Ccsr->num_entries,Acsr->num_entries,Bcsr->num_entries);
      if (Chipsp->cooPerm->size() != Ccsr->values->size()) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"permSize %D != %D",(PetscInt)Chipsp->cooPerm->size(),(PetscInt)Ccsr->values->size());
      auto pmid = Chipsp->cooPerm->begin();
      thrust::advance(pmid,Acsr->num_entries);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      auto zibait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->begin(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),Chipsp->cooPerm->begin())));
      auto zieait = thrust::make_zip_iterator(thrust::make_tuple(Acsr->values->end(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),pmid)));
      thrust::for_each(zibait,zieait,VecHIPEquals());
      auto zibbit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->begin(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),pmid)));
      auto ziebit = thrust::make_zip_iterator(thrust::make_tuple(Bcsr->values->end(),
                                                                 thrust::make_permutation_iterator(Ccsr->values->begin(),Chipsp->cooPerm->end())));
      thrust::for_each(zibbit,ziebit,VecHIPEquals());
      ierr = MatSeqAIJHIPSPARSEInvalidateTranspose(*C,PETSC_FALSE);CHKERRQ(ierr);
      if (Ahipsp->transgen && Bhipsp->transgen && Chipsp->transgen) {
        if (!Chipsp->matTranspose) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing transpose Mat_SeqAIJHIPSPARSEMultStruct");
        PetscBool AT = Ahipsp->matTranspose ? PETSC_TRUE : PETSC_FALSE, BT = Bhipsp->matTranspose ? PETSC_TRUE : PETSC_FALSE;
        CsrMatrix *AcsrT = AT ? (CsrMatrix*)Ahipsp->matTranspose->mat : NULL;
        CsrMatrix *BcsrT = BT ? (CsrMatrix*)Bhipsp->matTranspose->mat : NULL;
        CsrMatrix *CcsrT = (CsrMatrix*)Chipsp->matTranspose->mat;
        auto vT = CcsrT->values->begin();
        if (AT) vT = thrust::copy(AcsrT->values->begin(),AcsrT->values->end(),vT);
        if (BT) thrust::copy(BcsrT->values->begin(),BcsrT->values->end(),vT);
        Chipsp->transupdated = PETSC_TRUE;
      }
      cerr = WaitForHIP();CHKERRHIP(cerr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectStateIncrease((PetscObject)*C);CHKERRQ(ierr);
  (*C)->assembled     = PETSC_TRUE;
  (*C)->was_assembled = PETSC_FALSE;
  (*C)->offloadmask   = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJCopySubArray_SeqAIJHIPSPARSE(Mat A, PetscInt n, const PetscInt idx[], PetscScalar v[])
{
  PetscErrorCode    ierr;
  bool              dmem;
  const PetscScalar *av;
  hipError_t       cerr;

  PetscFunctionBegin;
  dmem = ishipMem(v);
  ierr = MatSeqAIJHIPSPARSEGetArrayRead(A,&av);CHKERRQ(ierr);
  if (n && idx) {
    THRUSTINTARRAY widx(n);
    widx.assign(idx,idx+n);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);

    THRUSTARRAY *w = NULL;
    thrust::device_ptr<PetscScalar> dv;
    if (dmem) {
      dv = thrust::device_pointer_cast(v);
    } else {
      w = new THRUSTARRAY(n);
      dv = w->data();
    }
    thrust::device_ptr<const PetscScalar> dav = thrust::device_pointer_cast(av);

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav,widx.begin()),dv));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(dav,widx.end()),dv+n));
    thrust::for_each(zibit,zieit,VecHIPEquals());
    if (w) {
      cerr = hipMemcpy(v,w->data().get(),n*sizeof(PetscScalar),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
    }
    delete w;
  } else {
    cerr = hipMemcpy(v,av,n*sizeof(PetscScalar),dmem ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost);CHKERRHIP(cerr);
  }
  if (!dmem) { ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr); }
  ierr = MatSeqAIJHIPSPARSERestoreArrayRead(A,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
