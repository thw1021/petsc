static char help[] = "Serial test of Cuda matrix assemble with 1D Laplacian.\n\n";

#include <petscmat.h>
#include <../../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>

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

static __device__ 
void PetscMemmove_cuda(void *a, void *b, size_t n)
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

static __device__
void MatSetValues_SeqAIJCUDA_device(Mat_SeqAIJCUDA_GPUData *a, PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  PetscInt       *rp,k,low,high,t,row,nrow,i,col,l,rmax,N;
  PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  PetscInt       *aj = a->j,nonew = a->nonew,lastcol = -1;
  MatScalar      *ap=NULL,value=0.0,*aa = a->a;
  PetscBool      ignorezeroentries = (a->ignorezeroentries==0) ? PETSC_FALSE : PETSC_TRUE;
  PetscBool      inserted          = PETSC_FALSE;

  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    if (row >= a->n) printf("MatSetValues_SeqAIJCUDA_device: row >= a->n");
    rp   = aj + ai[row];
    ap = aa + ai[row];
    rmax = imax[row]; nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      if (in[l] >= a->n) printf("MatSetValues_SeqAIJCUDA_device: in[%d]=%d >= a->n (square serial)",l,in[l]);
      col = in[l];
      //if (v) value = roworiented ? v[l + k*n] : v[k + l*m];
      if (v) value = v[l + k*n];
      if (value == 0.0 && ignorezeroentries && is == ADD_VALUES && row != col) continue;
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
	  if (v) {
	    if (is == ADD_VALUES) {
	      ap[i] += value;
	    }
	    else ap[i] = value;
	    inserted = PETSC_TRUE;
	  }
	  low = i + 1;
          goto noinsert;
        }
      }
      if (value == 0.0 && ignorezeroentries && row != col) goto noinsert;
      if (nonew == 1) goto noinsert;
      if (nonew == -1) printf("MatSetValues_SeqAIJCUDA_device: Inserting a new nonzero at (%d,%d) in the matrix",row,col);
      if (nrow >= rmax) printf("ERROR, ran out of preallocated space in row %d\n",(int)row);
      N = nrow++ - 1; a->nz++; high++;
      /* shift up all the later entries in this row */
      PetscMemmove_cuda(rp+i+1,rp+i,(N-i+1)*sizeof(PetscInt));
      rp[i] = col;
      PetscMemmove_cuda(ap+i+1,ap+i,(N-i+1)*sizeof(PetscScalar));
      ap[i] = value;
      low = i + 1;
      a->nonzerostate++; // should be A
      inserted = PETSC_TRUE;
noinsert:;
    }
    ailen[row] = nrow;
  }
  if (a->offloadmask != PETSC_OFFLOAD_UNALLOCATED && inserted) a->offloadmask = (PetscInt)PETSC_OFFLOAD_GPU; // wrong logic here!!!!!!
}

__global__
void assemble(PetscInt  n, Mat_SeqAIJCUDA_GPUData *cuda_mat)
{
  const PetscInt  inc = blockDim.x, my0 = blockIdx.x;
  PetscInt        i;
  PetscScalar     values[] = {-1,2,-1};
  for (i=my0; i<n; i+=inc) {
    if (i==0) {
      PetscInt js[] = {0, 1};
      MatSetValues_SeqAIJCUDA_device(cuda_mat,(PetscInt)1,&i,(PetscInt)2,js,&values[1],INSERT_VALUES);
    } else if (i==n-1) {
      PetscInt js[] = {n-2, n-1};
      MatSetValues_SeqAIJCUDA_device(cuda_mat,(PetscInt)1,&i,(PetscInt)2,js,values,INSERT_VALUES);
    } else {
      PetscInt js[] = {i-1, i, i+1};
      MatSetValues_SeqAIJCUDA_device(cuda_mat,(PetscInt)1,&i,(PetscInt)3,js,values,INSERT_VALUES);
    }
  }
}

int main(int argc,char **args)
{
  PetscErrorCode         ierr;
  Mat                    A;
  PetscInt               n=2, nz=2,create_t = 1;
  Mat_SeqAIJCUDA_GPUData *cuda_mat;
  PetscLogEvent          event;
  Vec                    x,y;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-nz", &nz, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-create_type", &create_t, NULL);CHKERRQ(ierr);
  if (nz>n) nz=n;
  ierr = PetscLogEventRegister("GPU operator", MAT_CLASSID, &event);CHKERRQ(ierr);
  if (create_t==1) {
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetType(A,MATAIJCUDA);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  } else {
    ierr = MatCreateSeqAIJCUDA(PETSC_COMM_SELF,n,n,nz,NULL,&A);CHKERRQ(ierr);
  }
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCUSPARSEGetCudaData(A,(void**)&cuda_mat);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  assemble<<<512,1>>>(n,cuda_mat);CHECK_LAUNCH_ERROR();
  CUDA_SAFE_CALL (cudaDeviceSynchronize());
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  // assemble end on CPU for now
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  VecViewFromOptions(y,NULL,"-vec_view");

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: cuda

   testset:
      args: -mat_type aijcuda -vec_type cuda -mat_view -n 11
      output_file: output/ex5cu_1.out
      test:
        suffix: 1
        nsize:  1

TEST*/
