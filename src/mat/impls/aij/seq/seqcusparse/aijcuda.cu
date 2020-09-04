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

static __device__ void PetscMemmove_cuda(void *a, void *b, size_t n)
{
  if (n > 0 && !a) printf("Trying to copy to null pointer");
  if (n > 0 && !b) printf("Trying to copy from a null pointer");
  if (a < b) {
    if ((char*)a <= ((char*)b - n)) memcpy(a,b,n);
    else {
      memcpy(a, b, (int)((char*)b - (char*)a));
      PetscMemmove_cuda(b,(char*)b + (int)((char*)b - (char*)a),n - (int)((char*)b - (char*)a));
    }
  } else {
    if (b <= ((char*)a - n)) memcpy(a,b,n);
    else {
      memcpy((char*)b + n,(char*)b + (n - (int)((char*)a - (char*)b)),(int)((char*)a - (char*)b));
      PetscMemmove_cuda(a,b,n - (int)((char*)a - (char*)b));
    }
  }
}

// a serial assemble now
__device__ void MatSetValues_SeqAIJCUDA_device(Mat_SeqAIJCUDA_GPUData *a, PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  PetscInt       *rp,k,low,high,t,row,nrow,i,col,l,rmax,N;
  PetscInt       *ai = a->i,*ailen = a->ilen;
  PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
  MatScalar      *ap=NULL,value=0.0,*aa = a->a;
  PetscBool      ignorezeroentries = (PetscBool)a->ignorezeroentries;
  PetscBool      inserted          = PETSC_FALSE;

  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    rp   = aj + ai[row];
    ap = aa + ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      while (l<n && (value = v[l + k*n]) == 0.0) l++;
      if (l==n) break;
      col = in[l];
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
	  if (is == ADD_VALUES) ap[i] += value;
	  else ap[i] = value;
	  low = i + 1;
          goto noinsert;
        }
      }
      if (value == 0.0 && ignorezeroentries && row != col) goto noinsert;
      if (nonew == 1) goto noinsert;
      if (nonew == -1) printf("Inserting a new nonzero at (%d,%d) in the matrix",row,col);
      if (nrow >= rmax) printf("ERROR, ran out of preallocated space in row %d\n",(int)row);
      N = nrow++ - 1; a->nnz++; high++;
      /* shift up all the later entries in this row */
      PetscMemmove_cuda(rp+i+1,rp+i,(N-i+1)*sizeof(PetscInt));
      rp[i] = col;
      PetscMemmove_cuda(ap+i+1,ap+i,(N-i+1)*sizeof(MatScalar));
      ap[i] = value;
      low = i + 1;
      a->nonzerostate++;
      inserted = PETSC_TRUE;
    noinsert:;
    }
    ailen[row] = nrow;
  }
  if (a->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) a->offloadmask = (PetscInt)PETSC_OFFLOAD_GPU; // wrong logic here!!!!!!
}

// serial now
static __global__
void MatAssemblyEnd_SeqAIJCUDA_device(Mat_SeqAIJCUDA_GPUData *a)
{
  PetscInt       fshift = 0,i,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m = a->n,*ip,N,*ailen = a->ilen,rmax = 0;
  MatScalar      *aa    = a->a,*ap;

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i=1; i<m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax    = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i];
      ap = aa + ai[i];
      N  = ailen[i];
      //ierr = PetscArraymove(ip-fshift,ip,N);CHKERRQ(ierr);
      PetscMemmove_cuda(ip-fshift,ip,N*sizeof(PetscInt));
      //ierr = PetscArraymove(ap-fshift,ap,N);CHKERRQ(ierr);
      PetscMemmove_cuda(ap-fshift,ap,N*sizeof(MatScalar));
    }
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (m) {
    fshift += imax[m-1] - ailen[m-1];
    ai[m]   = ai[m-1] + ailen[m-1];
  }
  /* reset ilen and imax for each row */
  a->nonzerorowcnt = 0;
  for (i=0; i<m; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
    a->nonzerorowcnt += ((ai[i+1] - ai[i]) > 0);
  }
  a->nnz = ai[m];
  a->rmax             = rmax;
}
static PetscErrorCode MatAssemblyEnd_SeqAIJCUDA(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJCUDA_GPUData *a;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0); // ???
  if (A->was_assembled && A->ass_nonzerostate == A->nonzerostate) PetscFunctionReturn(0); // ???

  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    a = spptr->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    a = spptr->cudaMat;
  }

  MatAssemblyEnd_SeqAIJCUDA_device<<<1,1>>>(a); // serial

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
  Mat_SeqAIJCUDA_GPUData *mat;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    mat = spptr->cudaMat;
    spptr->cudaMat = NULL;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    mat = spptr->cudaMat;
    spptr->cudaMat = NULL;
  }

  if (mat->i) CUDA_SAFE_CALL(cudaFree(mat->i));
  if (mat->ilen) CUDA_SAFE_CALL(cudaFree(mat->ilen));
  if (mat->j) CUDA_SAFE_CALL(cudaFree(mat->j));
  if (mat->a) CUDA_SAFE_CALL(cudaFree(mat->a));
  if (mat->imax) CUDA_SAFE_CALL(cudaFree(mat->imax));

  CUDA_SAFE_CALL(cudaFree(mat));

  ierr = MatDestroy_MatMatCusparse(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_SeqAIJCUDA_private(Mat A)
{
  PetscErrorCode          ierr;
  Mat_SeqAIJ              *jaca = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJCUDA_GPUData  h_mat, *d_mat;
  PetscInt                N =  A->rmap->n, nnz = jaca->i[N];

  PetscFunctionBegin;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_mat,  sizeof(Mat_SeqAIJCUDA_GPUData)));
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
    spptr->cudaMat = d_mat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *spptr = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    spptr->cudaMat = d_mat;
  }
  h_mat.n = N;
  h_mat.nnz = 0;
  h_mat.ignorezeroentries = jaca->ignorezeroentries;
  h_mat.nonew =jaca->nonew;
  h_mat.nonzerostate =A->nonzerostate;
  h_mat.offloadmask = A->offloadmask;
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
  CUDA_SAFE_CALL(cudaMemcpy(          d_mat, &h_mat, sizeof(Mat_SeqAIJCUDA_GPUData), cudaMemcpyHostToDevice));
  ierr = PetscInfo7(A,"PetscErrorCodeMatConvert_SeqAIJCUDA_private: n=%D rmax=%D  ignorezeroentries=%D nonew=%D nonzerostate=%D nonzerorowcnt=%D nnz=%D\n",h_mat.n, h_mat.rmax, h_mat.ignorezeroentries, h_mat.nonew, h_mat.nonzerostate,jaca->nonzerorowcnt,nnz);CHKERRQ(ierr);

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
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJCUDA_private(*A);CHKERRQ(ierr);
  ierr = PetscInfo2(*A,"MatCreateSeqAIJCUDA: %D X %D CUDA matrix created\n",m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatDuplicate_SeqAIJCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B);
__device__ void (*MatSetValues_SeqAIJCUDA_static)(Mat_SeqAIJCUDA_GPUData*, PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode) = MatSetValues_SeqAIJCUDA_device;
// constructor
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJCUDA(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode          ierr;

  Mat                     B;
  PetscBool               flgcusparse, flgaij;
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

  A->offloadmask = PETSC_OFFLOAD_GPU; //

  ierr = PetscInfo(A,"MatConvert_SeqAIJ_SeqAIJCUDA: Converted to CUDA matrix (ready to go)\n");CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUDA);CHKERRQ(ierr);
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
  ierr = PetscInfo(B,"MatCreate_SeqAIJCUDA called\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUDA(Mat B)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MatCreate_MPIAIJCUDA not supported");
  //PetscFunctionReturn(0);
}

