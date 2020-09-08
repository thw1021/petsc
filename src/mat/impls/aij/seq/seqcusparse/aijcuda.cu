/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format in Cuda kernels,
*/
// #define PETSC_SKIP_SPINLOCK
// #define PETSC_SKIP_CXX_COMPLEX_FIX
// #define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

// #include <petscconf.h>
// #include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

static PetscErrorCode MatAssemblyEnd_SeqAIJCUDA(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJCUDA_GPUData *d_mat, h_mat;
  Mat_SeqAIJ             *a = (Mat_SeqAIJ*)A->data;
  PetscInt                n = A->rmap->n, nnz = a->i[n]; 
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    d_mat = spptr->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    d_mat = spptr->cudaMat;
  }
  // copy back to CPU (move someplace else later)
  CUDA_SAFE_CALL(cudaMemcpy( &h_mat, d_mat, sizeof(Mat_SeqAIJCUDA_GPUData), cudaMemcpyDeviceToHost));
  a->nz            = h_mat.nz;
  A->nonzerostate  = h_mat.nonzerostate;
  a->nonzerorowcnt = h_mat.nonzerorowcnt;
  a->rmax          = h_mat.rmax;  
  CUDA_SAFE_CALL(cudaMemcpy( a->i,    h_mat.i,    (n+1)*sizeof(PetscInt), cudaMemcpyDeviceToHost));
  nnz = a->i[n];
  CUDA_SAFE_CALL(cudaMemcpy( a->ilen, h_mat.ilen, (n)*sizeof(PetscInt),     cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy( a->imax, h_mat.imax, (n)*sizeof(PetscInt),     cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy( a->j,    h_mat.j,    (nnz)*sizeof(PetscInt),   cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy( a->a,    h_mat.a,    (nnz)*sizeof(PetscScalar),cudaMemcpyDeviceToHost));

  A->offloadmask = PETSC_OFFLOAD_CPU; // MatCUSPARSE can now copy to its GPU data structure

  ierr = MatAssemblyEnd_SeqAIJCUSPARSE( A, mode);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJCUDA(Mat A)
{
  Mat_SeqAIJ             *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUDA_GPUData *mat;
  PetscInt                N = A->rmap->n, nnz = a->i[N];

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    mat = spptr->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    mat = spptr->cudaMat;
  }
  CUDA_SAFE_CALL(cudaMemset( mat->a, 0, (nnz)*sizeof(PetscScalar)));
  PetscFunctionReturn(0);
}

// get GPU pointer to stripped down Mat
PetscErrorCode MatCUSPARSEGetCudaData(Mat A, void **B)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    *B = (void*)spptr->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    *B = (void*)spptr->cudaMat;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJCUDA(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJCUDA_GPUData *d_mat, h_mat;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    d_mat = spptr->cudaMat;
    spptr->cudaMat = NULL;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    d_mat = spptr->cudaMat;
    spptr->cudaMat = NULL;
  }
  if (d_mat) {
    CUDA_SAFE_CALL(cudaMemcpy( &h_mat, d_mat, sizeof(Mat_SeqAIJCUDA_GPUData), cudaMemcpyDeviceToHost));
    if (h_mat.i)    CUDA_SAFE_CALL(cudaFree(h_mat.i));
    if (h_mat.ilen) CUDA_SAFE_CALL(cudaFree(h_mat.ilen));
    if (h_mat.j)    CUDA_SAFE_CALL(cudaFree(h_mat.j));
    if (h_mat.a)    CUDA_SAFE_CALL(cudaFree(h_mat.a));
    if (h_mat.imax) CUDA_SAFE_CALL(cudaFree(h_mat.imax));
    CUDA_SAFE_CALL(                cudaFree(d_mat));
  }
  ierr = MatDestroy_SeqAIJCUSPARSE(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqAIJCUDA(Mat A)
{
  PetscErrorCode          ierr;
  Mat_SeqAIJCUDA_GPUData  **p_d_mat;

  PetscFunctionBegin;
  ierr = MatSetUp_SeqAIJ(A);CHKERRQ(ierr);
  // create GPU Mat
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    p_d_mat = &spptr->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    p_d_mat = &spptr->cudaMat;
  }
  if (!*p_d_mat) {
    // create and copy
    Mat_SeqAIJCUDA_GPUData  h_mat, *d_mat;
    Mat_SeqAIJ              *jaca = (Mat_SeqAIJ*)A->data;
    PetscInt                N =  A->rmap->n, nnz = jaca->i[N];
    h_mat.n = N;
    h_mat.nz = 0;
    h_mat.ignorezeroentries = jaca->ignorezeroentries;
    h_mat.nonew =jaca->nonew;
    h_mat.nonzerostate =A->nonzerostate;
    h_mat.nonzerorowcnt =jaca->nonzerorowcnt;
    h_mat.rmax = jaca->rmax;
    // copy data
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_mat.i,               (N+1)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_mat.i,    jaca->i,   (N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_mat.ilen,            (N)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_mat.ilen, jaca->ilen,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_mat.imax,            (N)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_mat.imax, jaca->imax,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_mat.j,               (nnz)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_mat.j,    jaca->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_mat.a,               (nnz)*sizeof(PetscScalar))); // kernel output
    CUDA_SAFE_CALL(cudaMemcpy(          h_mat.a,    jaca->a,   (nnz)*sizeof(PetscScalar), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_mat,  sizeof(Mat_SeqAIJCUDA_GPUData)));
    CUDA_SAFE_CALL(cudaMemcpy(          d_mat, &h_mat, sizeof(Mat_SeqAIJCUDA_GPUData), cudaMemcpyHostToDevice));

    ierr = PetscInfo7(A,"MatSetUp_SeqAIJCUDA: n=%D rmax=%D  nonzerorowcnt=%D nonew=%D nonzerostate=%D nonzerorowcnt=%D nnz=%D\n",h_mat.n, h_mat.rmax, h_mat.nonzerorowcnt, h_mat.nonew, h_mat.nonzerostate,jaca->nonzerorowcnt,nnz);CHKERRQ(ierr);

    *p_d_mat = d_mat;
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*@
   MatCreateSeqAIJCUDA - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).

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


   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ()
@*/
PetscErrorCode  MatCreateSeqAIJCUDA(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[], Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJCUDA);CHKERRQ(ierr);
  ierr = PetscInfo2(*A,"MatCreateSeqAIJCUDA: %D X %D CUDA matrix created\n",m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatDuplicate_SeqAIJCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B);
// constructor
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUDA(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode          ierr;
  Mat                     B;
  PetscBool               flgcusparse, flgaij, flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(newmat,4);

  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate_SeqAIJ(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_SeqAIJCUSPARSE(*newmat,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr); // AIJ
  } else newmat = &A;
  B = *newmat;

  ierr = PetscObjectTypeCompare(((PetscObject)B),MATSEQAIJCUDA,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscObjectTypeCompare(((PetscObject)B),MATSEQAIJCUSPARSE,&flgcusparse);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare(((PetscObject)B),MATSEQAIJ,&flgaij);CHKERRQ(ierr);
    if (!flgcusparse && !flgaij) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)B)->type_name);
    if (flgaij) {
      ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
      ierr = MatConvert_SeqAIJ_SeqAIJCUSPARSE(B,MATSEQAIJCUSPARSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    }
    *newmat = B;
    
    B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUDA;
    B->ops->destroy        = MatDestroy_SeqAIJCUDA;
    B->ops->duplicate      = MatDuplicate_SeqAIJCUDA;
    B->ops->setvalues      = NULL; // we don't want to mix
    B->ops->zeroentries    = MatZeroEntries_SeqAIJCUDA;
    B->ops->setup          = MatSetUp_SeqAIJCUDA;
    
    ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUDA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatDuplicate_SeqAIJCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJCUDA(*B,MATSEQAIJCUDA,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  ierr = PetscInfo(A,"MatDuplicate_SeqAIJCUDA done\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqAIJCUSPARSE(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJCUDA(B,MATSEQAIJCUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUDA(Mat B)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MatCreate_MPIAIJCUDA not supported");
  //PetscFunctionReturn(0);
}

