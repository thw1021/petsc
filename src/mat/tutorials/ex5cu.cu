static char help[] = "Serial test of Cuda matrix assemble with 1D Laplacian.\n\n";

#include <../../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>

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

__global__
void assemble(PetscInt  n, Mat_SeqAIJCUDA_GPUData *cuda_mat)
{
  const PetscInt  Nq = blockDim.x, myelem = blockIdx.x;
  PetscInt        i;
  PetscScalar     values[] = {-1,2,-1};

  for (i=0; i<n; i++) {
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
  PetscInt               n=2, nz=2;
  Mat_SeqAIJCUDA_GPUData *cuda_mat;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-nz", &nz, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL);CHKERRQ(ierr);
  if (nz>n) nz=n;
  ierr = MatCreateSeqAIJCUDA(PETSC_COMM_SELF,n,n,nz,NULL,&A);CHKERRQ(ierr);
  ierr = MatCUSPARSEGetCudaData(A,&cuda_mat);CHKERRQ(ierr);

  assemble<<<1,1>>>(n,cuda_mat);CHECK_LAUNCH_ERROR();

  // assemble end on CPU for now
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatViewFromOptions(A,NULL,"-mat_view");CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -mat_view -nz 3

TEST*/
