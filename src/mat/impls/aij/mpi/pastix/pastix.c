/*
 Provides an interface to the PaStiX sparse solver
 */
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

#if defined(PETSC_USE_COMPLEX)
#define _H_COMPLEX
#endif

#include <pastix.h>

#if defined(PETSC_USE_COMPLEX)

#if defined(PETSC_USE_REAL_SINGLE)
#define SPM_FLTTYPE SpmComplex32
#else
#define SPM_FLTTYPE SpmComplex64
#endif

#else /* PETSC_USE_COMPLEX */

#if defined(PETSC_USE_REAL_SINGLE)
#define SPM_FLTTYPE SpmFloat
#else
#define SPM_FLTTYPE SpmDouble
#endif

#endif /* PETSC_USE_COMPLEX */

typedef PetscScalar PastixScalar;

typedef struct Mat_Pastix_ {
  pastix_data_t *pastix_data;        /* Pastix data storage structure                             */
  MPI_Comm       comm;               /* MPI Communicator used to initialize pastix                */
  spmatrix_t    *spm;                /* SPM matrix structure                                      */
  MatStructure   matstruc;           /* DIFFERENT_NONZERO_PATTERN if uninitilized, SAME otherwise */
  PetscScalar   *rhs;                /* Right-hand-side member                                    */
  PetscInt       rhsnbr;             /* Right-hand-side number                                    */
  PetscInt       iparm[IPARM_SIZE];  /* Integer parameters                                        */
  double         dparm[DPARM_SIZE];  /* Floating point parameters                                 */
  PetscBool      CleanUpPastix;      /* Boolean indicating if we call PaStiX clean step           */
  VecScatter     scat_rhs;           /* Indicates how to gather the RHS */
  VecScatter     scat_sol;           /* Indicates how to scatter the RHS */
  Vec            b_seq;
} Mat_Pastix;

extern PetscErrorCode MatDuplicate_Pastix(Mat,MatDuplicateOption,Mat*);

/*
  Gather right-hand-side.
  Call for Solve step.
  Scatter solution.
 */
PetscErrorCode MatSolve_PaStiX(Mat A,Vec b,Vec x)
{
  Mat_Pastix     *pastix=(Mat_Pastix*)A->data;
  PetscScalar    *array;
  Vec             x_seq;
  Vec             b_cpy;
  PetscErrorCode  ierr;
  PetscInt        ldrhs;

  PetscFunctionBegin;
  pastix->rhsnbr = 1;
  x_seq          = pastix->b_seq;
  ldrhs          = pastix->spm->n;
  if (pastix->spm->clustnbr > 1) {
    /* PaStiX only supports centralized rhs. Scatter b into a seqential rhs vector */
    ierr = VecScatterBegin(pastix->scat_rhs, b, x_seq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pastix->scat_rhs, b, x_seq, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  } else {  /* size == 1 */
    ierr = VecDuplicate(b, &x_seq);CHKERRQ(ierr);
    ierr = VecCopy(b,  x_seq);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(x_seq, &b_cpy);CHKERRQ(ierr);
  ierr = VecCopy(x_seq,  b_cpy);CHKERRQ(ierr);

  ierr = VecGetArray(x_seq, &(pastix->rhs));CHKERRQ(ierr);
  ierr = VecGetArray(b_cpy, &array);CHKERRQ(ierr);

  /* solve phase */
  /*-------------*/
  pastix_task_solve(pastix->pastix_data, pastix->rhsnbr, pastix->rhs, ldrhs);

  pastix_task_refine(pastix->pastix_data, ldrhs, pastix->rhsnbr,
                      array,               ldrhs,
                      pastix->rhs,         ldrhs);

  ierr = VecDestroy(&b_cpy);CHKERRQ(ierr);

  /* convert PaStiX centralized solution to petsc mpi x */
  if (pastix->spm->clustnbr > 1) {
    ierr = VecRestoreArray(x_seq, &(pastix->rhs));CHKERRQ(ierr);
    ierr = VecScatterBegin(pastix->scat_sol, x_seq, x, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(  pastix->scat_sol, x_seq, x, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(x, &array);CHKERRQ(ierr);
    ierr = VecRestoreArray(x, &(pastix->rhs));CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*
  Numeric factorisation using PaStiX solver.

  input:
    F       - PETSc matrix that contains PaStiX interface.
    A       - PETSC matrix in aij, bail or sbaij format
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple
              MAT_REUSE_MATRIX:   only the values in v array are updated
    valOnly - FALSE: spaces are allocated and values are set for the CSC
              TRUE:  Only fill values
  output:
    spm     - The SPM built from A
 */
PetscErrorCode PaStiX_MatFactorNumeric(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_Pastix    *pastix =(Mat_Pastix*)(F)->data;
  PetscErrorCode ierr = 0;
  IS             is_iden;
  Vec            b;
  PetscBool      isSeqAIJ,isSeqSBAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ,   &isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &isSeqSBAIJ);CHKERRQ(ierr);

  /* If it's the first time we set Mat_Pastix ->  Initialize everything */
  if (pastix->matstruc == DIFFERENT_NONZERO_PATTERN) {
    (F)->ops->solve = MatSolve_PaStiX;
  }

  /* PaStiX only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
  if (!(isSeqAIJ || isSeqSBAIJ) && !(pastix->b_seq)) {
    ierr = VecCreateSeq(PETSC_COMM_SELF, A->cmap->N, &(pastix->b_seq));CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF, A->cmap->N, 0, 1, &is_iden);CHKERRQ(ierr);
    ierr = MatCreateVecs(A, NULL, &b);CHKERRQ(ierr);

    /* Create Scatter (scat_rhs) and Gather (scat_sol) VecScatter */
    ierr = VecScatterCreate(b, is_iden, pastix->b_seq, is_iden, &(pastix->scat_rhs));CHKERRQ(ierr);
    ierr = VecScatterCreate(pastix->b_seq, is_iden, b, is_iden, &(pastix->scat_sol));CHKERRQ(ierr);

    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }

  /* Perform Numerical Factorization */
  assert(pastix->CleanUpPastix);
  pastix_task_numfact(pastix->pastix_data, pastix->spm);

  (F)->assembled        = PETSC_TRUE;
  pastix->matstruc      = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

PetscErrorCode PaStiX_MatLUFactorNumeric(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PaStiX_MatFactorNumeric(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PaStiX_MatCholeskyFactorNumeric(Mat F, Mat A, const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PaStiX_MatFactorNumeric(F, A, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   convert PETSc matrix to SPM structure

  input:
    A       - matrix in aij, bail or sbaij format
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple
              MAT_REUSE_MATRIX:   only the values in v array are updated
    valOnly - FALSE: spaces are allocated and values are set for the CSC
              TRUE:  Only fill values
  output:
    spm     - The SPM built from A
 */
PetscErrorCode MatConvertToSPM(Mat mpiA, MatReuse reuse, Mat_Pastix *pastix)
{
  Mat            *seqA;
  Mat_SeqAIJ     *aa;
  PetscInt       *row;
  PetscInt       *col;
  PetscScalar    *val;
  PetscErrorCode  ierr;
  PetscBool       isSym, isHer;
  spmatrix_t     *spm = NULL;
  spmatrix_t      spm2;

  PetscFunctionBegin;

  /* convert mpi A to seq mat A */
  {
    IS isrow;
    ierr = ISCreateStride(PETSC_COMM_SELF, mpiA->rmap->N, 0, 1, &isrow);CHKERRQ(ierr);
    ierr = MatCreateSubMatrices(mpiA, 1, &isrow, &isrow, MAT_INITIAL_MATRIX, &seqA);CHKERRQ(ierr);
    ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  }

  aa  = (Mat_SeqAIJ*)(*seqA)->data;
  row = aa->i;
  col = aa->j;
  val = aa->a;

  spm = malloc(sizeof(spmatrix_t));
  spmInitDist(spm, pastix->comm);

  spm->baseval = 0;
  spm->fmttype = SpmCSR;
  spm->flttype = SPM_FLTTYPE;

  ierr = MatIsSymmetric((*seqA), 0.0, &isSym);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = MatIsHermitian((*seqA), 0.0, &isHer);CHKERRQ(ierr);
#else
  isHer = PETSC_FALSE;
#endif
  if (isHer) {
    spm->mtxtype = SpmHermitian;
  }
  else if (isSym) {
    spm->mtxtype = SpmSymmetric;
  }
  else {
    spm->mtxtype = SpmGeneral;
  }

  spm->n   = (*seqA)->cmap->n;
  spm->nnz = aa->nz;
  spm->dof = 1;

  spmUpdateComputedFields(spm);
  spmAlloc(spm);

  /* Copy  arrays */
  ierr = PetscArraycpy(spm->colptr, col, spm->nnz);CHKERRQ(ierr);
  ierr = PetscArraycpy(spm->rowptr, row, spm->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy((PetscScalar*)spm->values, val, spm->nnzexp);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1, &seqA);CHKERRQ(ierr);

  /* Update matrix to be in PaStiX format */
  ierr = spmCheckAndCorrect(spm, &spm2);
  if (ierr != 0) {
    spmExit(spm);
    *(spm) = spm2;
    ierr = 0;
  }

  if (pastix->iparm[IPARM_VERBOSE] > 0)
    spmPrintInfo(spm, stdout);

  pastix->spm = spm;
  PetscFunctionReturn(0);
}

/*
  Perform Ordering step and Symbolic Factorization step

  Note the Petsc r and c permutations are ignored
  input:
    F       - PETSc matrix that contains PaStiX interface.
    A       - matrix in aij, bail or sbaij format
    r       - permutation ?
    c       - TODO
    info    - Informations about the factorization to perform.
  output:
    pastix_data - This instance will be updated with the SOlverMatrix allocated.
 */
static PetscErrorCode PaStiX_MatFactorSymbolic(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_Pastix    *pastix = (Mat_Pastix*)(F->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pastix->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Initialise SPM structure */
  ierr = MatConvertToSPM(A, MAT_INITIAL_MATRIX, pastix);CHKERRQ(ierr);

  /* Ordering - Symbolic factorization - Build SolverMatrix  */
  pastix_task_analyze(pastix->pastix_data, pastix->spm);

  pastix->CleanUpPastix = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PaStiX_MatLUFactorSymbolic(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_Pastix *pastix = (Mat_Pastix*)(F->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;

 CHKERRQ((pastix->iparm[IPARM_FACTORIZATION] != PastixFactGETRF));
  ierr = PaStiX_MatFactorSymbolic(F, A, r, c, info);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Note the Petsc r permutation is ignored */
PetscErrorCode PaStiX_MatCholeskyFactorSymbolic(Mat F, Mat A, IS r, const MatFactorInfo *info)
{
  Mat_Pastix    *pastix = (Mat_Pastix*)(F->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
 CHKERRQ((pastix->iparm[IPARM_FACTORIZATION] != PastixFactSYTRF));
  ierr = PaStiX_MatFactorSymbolic(F, A, r, NULL, info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Call clean step of PaStiX if lu->CleanUpPastix == true.
  Free the CSC matrix.
 */
PetscErrorCode PaStiX_MatDestroy(Mat A)
{
  Mat_Pastix     *lu=(Mat_Pastix*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lu->CleanUpPastix) {
    /* Terminate instance, deallocate memories */
    ierr = VecScatterDestroy(&lu->scat_rhs);CHKERRQ(ierr);
    ierr = VecDestroy(&lu->b_seq);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&lu->scat_sol);CHKERRQ(ierr);

    spmExit(lu->spm);
    free(lu->spm);
    pastixFinalize(&(lu->pastix_data));
  }
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PaStiX_MatView(Mat A, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      Mat_Pastix *pastix = (Mat_Pastix*)A->data;
      spmatrix_t *spm    = pastix->spm;
      assert(spm != NULL);

      ierr = PetscViewerASCIIPrintf(viewer,"PaStiX run parameters:\n");CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"  Matrix type :                      %s \n",
                                     ((spm->mtxtype == SpmSymmetric) ? "Symmetric" : "Unsymmetric"));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Level of printing (0,1,2):         %d \n",pastix->iparm[IPARM_VERBOSE]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Number of refinements iterations : %d \n",pastix->iparm[IPARM_NBITER]);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,  "  Error :                            %g \n",pastix->dparm[DPARM_RELATIVE_ERROR]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
     MATSOLVERPASTIX  - A solver package providing direct solvers (LU) for distributed
  and sequential matrices via the external package PaStiX.

  Use ./configure --download-pastix --download-ptscotch  to have PETSc installed with PasTiX

  Use -pc_type lu -pc_factor_mat_solver_type pastix to use this direct solver

  Options Database Keys:
+ -mat_pastix_verbose   <0,1,2>   - print level
- -mat_pastix_threadnbr <integer> - Set the thread number by MPI task.

  Notes:
    This only works for matrices with symmetric nonzero structure, if you pass it a matrix with
   nonsymmetric structure PasTiX and hence PETSc return with an error.

  Level: beginner

.seealso: PCFactorSetMatSolverType(), MatSolverType

M*/

PetscErrorCode PaStiX_MatGetInfo(Mat A, MatInfoType flag, MatInfo *info)
{
  Mat_Pastix *pastix =(Mat_Pastix*)A->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = pastix->iparm[IPARM_NNZEROS];
  info->nz_used           = pastix->iparm[IPARM_NNZEROS];
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PaStiX_MatFactorGetSolverType(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERPASTIX;
  PetscFunctionReturn(0);
}

/*
    Options Database Keys:
      + -mat_pastix_verbose   <0,1,2>   - print level
      - -mat_pastix_threadnbr <integer> - Set the thread number by MPI task.
 */
static PetscErrorCode PaStiX_MatSetOptions(Mat A)
{
  Mat_Pastix    *pastix = (Mat_Pastix *)A->data;
  PetscErrorCode ierr;
  PetscInt       icntl;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix, "PaStiX Options", "Mat");
 CHKERRQ(ierr);
  icntl = -1;

  /* By Default : No output from PaStiX */
  pastix->iparm[IPARM_VERBOSE] = PastixVerboseNot;
  ierr = PetscOptionsInt("-mat_pastix_verbose", "iparm[IPARM_VERBOSE] : level of printing (0 to 2)", "None",
                          pastix->iparm[IPARM_VERBOSE], &icntl, &flg);
 CHKERRQ(ierr);
  if ((flg && (icntl >= 0)) || PetscLogPrintInfo) {
    pastix->iparm[IPARM_VERBOSE] = icntl;
  }

  icntl = -1;
  ierr  = PetscOptionsInt("-mat_pastix_threadnbr", "iparm[IPARM_THREAD_NBR] : Number of thread by MPI node", "None",
                           pastix->iparm[IPARM_THREAD_NBR], &icntl, &flg);
 CHKERRQ(ierr);
  if (flg && (icntl > 0)) {
    pastix->iparm[IPARM_THREAD_NBR] = icntl;
  }
  PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode PaStiX_MatGetFactor(Mat A, MatFactorType ftype, Mat *F,
                                           const char *mattype)
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Pastix    *pastix;

  PetscFunctionBegin;

  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A), &B);CHKERRQ(ierr);
  ierr = MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPASTIX, &((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  if ((ftype != MAT_FACTOR_LU) && (ftype != MAT_FACTOR_CHOLESKY)) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported by PaStiX");
  }

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPASTIX,&B->solvertype);CHKERRQ(ierr);

  B->ops->lufactorsymbolic       = PaStiX_MatLUFactorSymbolic;
  B->ops->lufactornumeric        = PaStiX_MatLUFactorNumeric;
  B->ops->choleskyfactorsymbolic = PaStiX_MatCholeskyFactorSymbolic;
  B->ops->choleskyfactornumeric  = PaStiX_MatCholeskyFactorNumeric;
  B->ops->view                   = PaStiX_MatView;
  B->ops->getinfo                = PaStiX_MatGetInfo;
  B->ops->destroy                = PaStiX_MatDestroy;

  ierr = PetscObjectComposeFunction((PetscObject)B,
                                     "MatFactorGetSolverType_C",
                                     PaStiX_MatFactorGetSolverType);
 CHKERRQ(ierr);

  B->factortype = ftype;

  /* Create the pastix structure */
  ierr = PetscNewLog(B, &pastix);CHKERRQ(ierr);
  B->data = (void*)pastix;

  pastix->CleanUpPastix = PETSC_FALSE;
  pastix->scat_rhs      = NULL;
  pastix->scat_sol      = NULL;

  /* Call to set default pastix options */
  pastixInitParam(pastix->iparm, pastix->dparm);
  PaStiX_MatSetOptions(B);

  /* Get PETSc Communicator */
  ierr = PetscObjectGetComm((PetscObject)A, &(pastix->comm));CHKERRQ(ierr);

  /* Initialise PaStiX structure */
  pastixInit(&(pastix->pastix_data), pastix->comm,
              pastix->iparm, pastix->dparm);

  if (ftype == MAT_FACTOR_CHOLESKY) {
    pastix->iparm[IPARM_FACTORIZATION] = PastixFactSYTRF;
  }
  else {
    pastix->iparm[IPARM_FACTORIZATION] = PastixFactGETRF;
  }

  *F = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode PaStiX_MatGetFactor_mpiaij(Mat A, MatFactorType ftype, Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_LU) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");
  }
  ierr = PaStiX_MatGetFactor(A, ftype, F, MATMPIAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PaStiX_MatGetFactor_seqaij(Mat A, MatFactorType ftype, Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_LU) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");
  }
  ierr = PaStiX_MatGetFactor(A, ftype, F, MATSEQAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PaStiX_MatGetFactor_mpisbaij(Mat A, MatFactorType ftype, Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  }
  ierr = PaStiX_MatGetFactor(A, ftype, F, MATMPISBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PaStiX_MatGetFactor_seqsbaij(Mat A, MatFactorType ftype, Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  }
  ierr = PaStiX_MatGetFactor(A, ftype, F, MATSEQSBAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Pastix(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERPASTIX, MATMPIAIJ,   MAT_FACTOR_LU,       PaStiX_MatGetFactor_mpiaij);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPASTIX, MATSEQAIJ,   MAT_FACTOR_LU,       PaStiX_MatGetFactor_seqaij);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPASTIX, MATMPISBAIJ, MAT_FACTOR_CHOLESKY, PaStiX_MatGetFactor_mpisbaij);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPASTIX, MATSEQSBAIJ, MAT_FACTOR_CHOLESKY, PaStiX_MatGetFactor_seqsbaij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
