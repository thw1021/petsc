/*
  Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format in Cuda kernels,
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

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

PetscErrorCode MatDuplicate_SeqAIJCUDA(Mat A, MatDuplicateOption cpvalues, Mat *B);
__device__ void MatSetValues_SeqAIJCUDA(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  PetscErrorCode ierr;
  Mat            cudamat, *pCudaMat;
  //Mat_SeqAIJ     *amat = (Mat_SeqAIJ*)A->data;
  PetscInt       *aj,lastcol = -1, n;
  MatScalar      *ap=NULL,value=0.0,*aa;
  PetscInt       *ai,*ailen;
  
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *b = (Mat_SeqAIJCUSPARSE*)A->spptr;
    pCudaMat = &b->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *b = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    pCudaMat = &b->cudaMat;
  }
  if (*pCudaMat) cudamat = *pCudaMat;
  else {
    ierr = MatDuplicate_SeqAIJCUDA(A, MAT_COPY_VALUES, &cudamat);CHKERRQ(ierr);
    *pCudaMat = cudamat;
  }
  ai = cudamat->i;
  ailen = cudamat->ilen;
  aj = cudamat->j;
  aa = cudamat->a;
  n = cudamat->rmap->n;

  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    rp   = aj + ai[row];
    ap = aa + ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
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
        // if (rp[i] > col) break;
        if (rp[i] == col) {
	  ap[i] += value;
	  low = i + 1;
          goto noinsert;
        }
      }
      printf("\t\t\t ERROR in assemble_kernel\n");
    noinsert:;
    }
  }
}
/*
   MemmoveLoc - Copies n bytes, beginning at location b, to the space
   beginning at location a. Copying  between regions that overlap will
   take place correctly. Use PetscMemcpy() if the locations do not overlap

   Not Collective

   Input Parameters:
+  b - pointer to initial memory space
.  a - pointer to copy space
-  n - length (in bytes) of space to copy

   Level: intermediate

   Note:
   PetscArraymove() is preferred
   This routine is analogous to memmove().

   Developers Note: This is inlined for performance

.seealso: PetscMemcpy(), PetscMemcmp(), PetscArrayzero(), PetscMemzero(), PetscArraycmp(), PetscArraycpy(), PetscStrallocpy(),
          PetscArraymove()
*/
static __device__ voind MemmoveLoc(void *a,const void *b, size_t n, size_t sz_of)
{
  if (a < b) {
    if (a <= b - n) CUDA_SAFE_CALL(cudaMemcpy( a, b, n*sz_of, cudaMemcpyDeviceToDevice)); // memcpy(a,b,n);
    else {
      //memcpy(a,b,(int)(b - a));
      CUDA_SAFE_CALL(cudaMemcpy( a, b, (int)((char*)b - (char*)a)*sz_of, cudaMemcpyDeviceToDevice));
      MemmoveLoc(b,b + (int)(b - a),n - (int)(b - a), sz_of);
    }
  } else {
    if (b <= a - n) CUDA_SAFE_CALL(cudaMemcpy( a, b, n*sz_of, cudaMemcpyDeviceToDevice)); // memcpy(a,b,n);
    else {
      //memcpy(b + n,b + (n - (int)(a - b)),(int)(a - b));
      CUDA_SAFE_CALL(cudaMemcpy( a, b, (int)((char*)b - (char*)a)*sz_of, cudaMemcpyDeviceToDevice));
      MemmoveLoc(a,b,n - (int)(a - b), sz_of);
    }
  }
}

__device__ void MatAssemblyEnd_SeqAIJCUDA(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat            cudamat, *pCudaMat;
  //Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscInt       *aj,lastcol = -1, *ip, m = A->rmap->n;
  MatScalar      *ap=NULL,value=0.0,*aa;
  PetscBool      ignorezeroentries;
  PetscBool      roworiented;
  PetscInt       *imax,*ai,*ailen,i,k;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  //ierr = MatSeqAIJInvalidateDiagonal(A);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqAIJCUSPARSE *b = (Mat_SeqAIJCUSPARSE*)A->spptr;
    pCudaMat = &b->cudaMat;
  } else {
    Mat_SeqAIJCUSPARSETriFactors *b = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
    pCudaMat = &b->cudaMat;
  }
  if (*pCudaMat) cudamat = *pCudaMat;
  else {
    ierr = MatDuplicate_SeqAIJCUDA(A, MAT_COPY_VALUES, &cudamat);CHKERRQ(ierr);
    *pCudaMat = cudamat;
  } 
  // if (A->was_assembled && A->ass_nonzerostate == A->nonzerostate) PetscFunctionReturn(0);
  ignorezeroentries = cudamat->ignorezeroentries;
  n = cudamat->rmap->n;
  roworiented       = cudamat->roworiented;
  ai = cudamat->i;
  ailen = cudamat->ilen;
  aj = cudamat->j;
  aa = cudamat->a;
  imax = cudamat->imax;

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i=1; i<m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax    = rmax > ailen[i] ? rmax : ailen[i]; // PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i];
      ap = aa + ai[i];
      N  = ailen[i];
      ierr = MemmoveLoc(ip-fshift,ip,N,sizeof(PetscInt));CHKERRQ(ierr);
      ierr = MemmoveLoc(ap-fshift,ap,N,sizeof(MatScalar));CHKERRQ(ierr);
    }
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (m) {
    fshift += imax[m-1] - ailen[m-1];
    ai[m]   = ai[m-1] + ailen[m-1];
  }

  /* reset ilen and imax for each row */
  cudamat->nonzerorowcnt = 0;
  for (i=0; i<m; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
    cudamat->nonzerorowcnt += ((ai[i+1] - ai[i]) > 0);
  }
  cudamat->nz    = ai[m];
  cudamat->rmax  = rmax;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);

  if (B->i) CUDA_SAFE_CALL(cudaFree(B->i));
  if (B->ilen) CUDA_SAFE_CALL(cudaFree(B->ilen));
  if (B->j) CUDA_SAFE_CALL(cudaFree(B->j));
  if (B->a) CUDA_SAFE_CALL(cudaFree(B->a));
  if (B->imax) CUDA_SAFE_CALL(cudaFree(B->imax));

  CUDA_SAFE_CALL(cudaFree(B));

  PetscFunctionReturn(0);
}
//
// This is the constructor and copy at this point
//
PetscErrorCode MatDuplicate_SeqAIJCUDA(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *jaca = (Mat_SeqAIJ*)A->data; // input is a Mat_SeqAIJ
  static PetscInt  idcnt = 10000;
  Mat              newMat;
  const PetscInt   nnz = jaca->i[JacP->rmap->n], N = JacP->rmap->n;  /* serial */
  PetscBool        flgcusparse, flgaij;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(B,3);
  ierr = PetscObjectTypeCompare(((PetscObject)A),MATSEQAIJCUSPARSE,&flgcusparse);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare(((PetscObject)A),MATSEQAIJ,&flgaij);CHKERRQ(ierr);
  if (!flgcusparse && !flgaij) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Not for type %s",((PetscObject)A)->type_name);

  CUDA_SAFE_CALL(cudaMalloc((void **)&newMat, sizeof(struct _p_Mat)));
  CUDA_SAFE_CALL(cudaMemcpy(          newMat,   A, sizeof(struct _p_Mat), cudaMemcpyHostToDevice)); // this copies junk
  *B = newMat;
  ierr = PetscCommDuplicate(PetscObjectComm((PetscObject)A),&B->comm,&B->tag);CHKERRQ(ierr);        // need this?
  CUDA_SAFE_CALL(cudaMalloc((void **)&B->rmap, sizeof(struct _n_PetscLayout)));
  CUDA_SAFE_CALL(cudaMemcpy(          B->rmap,   A->rmap, sizeof(struct _n_PetscLayout), cudaMemcpyHostToDevice)); // skip cmap
  ierr = PetscStrallocpy(VECSTANDARD,&B->defaultvectype);CHKERRQ(ierr);        // need this?

  CUDA_SAFE_CALL(cudaMalloc((void **)&B->i,               (N+1)*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          B->i,    jaca->i,   (N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&B->ilen,            (N)*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          B->ilen, jaca->ilen,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&B->j,               (nnz)*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          B->j,    jaca->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&B->a,               (nnz)*sizeof(PetscScalar))); // kernel output
  if (cpvalues == MAT_COPY_VALUES) {
    CUDA_SAFE_CALL(cudaMemcpy(          B->a,    jaca->a,   (nnz)*sizeof(PetscScalar), cudaMemcpyHostToDevice));
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&B->imax,            (N)*sizeof(PetscInt))); // kernel output
  CUDA_SAFE_CALL(cudaMemcpy(          B->imax, jaca->imax,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));

  //B->congruentlayouts = PETSC_DECIDE;
  //B->preallocated     = PETSC_FALSE;
  B->ops->setvalues      = MatSetValues_SeqAIJCUDA;
  B->ops->assemblyend    = MatAssemblyEnd_SeqAIJCUDA;
  B->ops->destroy        = MatDestroy_SeqAIJCUDA;
  B->ops->duplicate      = MatDuplicate_SeqAIJCUDA; // ???

  B->nonzerostate  = A->nonzerostate;

  ierr = MatBindToCPU_SeqAIJCUSPARSE(B,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJCUDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //XXXXX ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  // ierr = MatCreate_SeqAIJ(B);CHKERRQ(ierr);
  SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MatCreate_MPIAIJCUDA not supported ???????");
  // ierr = MatConvert_SeqAIJ_SeqAIJCUDA(B,MATSEQAIJCUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUDA(Mat B)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)B),PETSC_ERR_SUP,"MatCreate_MPIAIJCUDA not supported");
  PetscFunctionReturn(0);
}

