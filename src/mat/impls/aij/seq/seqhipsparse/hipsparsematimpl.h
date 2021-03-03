#if !defined(__HIPSPARSEMATIMPL)
#define __HIPSPARSEMATIMPL

/*
 * Note on hip versus roc:
 * For the basic hip language itself, we used the hip label in keeping with the
 * recommended porting guide and to enable testing with a cuda back end.
 * However, with the rocblas/rocsparse, the hip layers do not do a good job of
 * maintaining the compatibility so we abandon their hip layers and go straigt
 * to the roc layers.  This is recommended by conversations with AMD developers
 * themselves.  This means that we should enable vec_type=hip and
 * HIP_PLATFORM=cuda and use cusparse.  Permuations like this will be deferred
 * down the road.
 *
 * We do continue using the hip label at the PETSc level; i.e., internally we
 * use the hip moniker even if the underlying call is out to roc
 *
 */
#include <petscpkg_version.h>
#include <petsc/private/hipvecimpl.h>

#include <rocsparse.h>
/* csrsv2Info_t is defined in hipsparse and not rocsparse */
#include <hipsparse.h>

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/system/system_error.h>

#define CHKERRHIPSPARSE(stat) do {if (PetscUnlikely(stat)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"rocsparse error %d",(int)stat);} while (0)

#define PetscStackCallThrust(body) do {                                     \
    try {                                                                   \
      body;                                                                 \
    } catch(thrust::system_error& e) {                                      \
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Thrust %s",e.what());\
    }                                                                       \
  } while (0)

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    const hipComplex PETSC_HIPSPARSE_ONE        = {1.0f, 0.0f};
    const hipComplex PETSC_HIPSPARSE_ZERO       = {0.0f, 0.0f};
  #elif defined(PETSC_USE_REAL_DOUBLE)
    const hipDoubleComplex PETSC_HIPSPARSE_ONE  = {1.0, 0.0};
    const hipDoubleComplex PETSC_HIPSPARSE_ZERO = {0.0, 0.0};
  #endif
#else
  const PetscScalar PETSC_HIPSPARSE_ONE        = 1.0;
  const PetscScalar PETSC_HIPSPARSE_ZERO       = 0.0;
#endif

#define rocsparse_create_analysis_info  hipsparseCreateCsrsv2Info
#define rocsparse_destroy_analysis_info hipsparseDestroyCsrsv2Info
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define rocsparse_get_svbuffsize(a,b,c,d,e,f,g,h,i,j) hipsparseCcsrsv2_bufferSize(a,b,c,d,e,(hipComplex*)(f),g,h,i,j)
    #define rocsparse_analysis(a,b,c,d,e,f,g,h,i,j,k)     hipsparseCcsrsv2_analysis(a,b,c,d,e,(const hipComplex*)(f),g,h,i,j,k)
    #define rocsparse_solve(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  hipsparseCcsrsv2_solve(a,b,c,d,(const hipComplex*)(e),f,(const hipComplex*)(g),h,i,j,(const hipComplex*)(k),(hipComplex*)(l),m,n)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define rocsparse_get_svbuffsize(a,b,c,d,e,f,g,h,i,j) hipsparseZcsrsv2_bufferSize(a,b,c,d,e,(hipDoubleComplex*)(f),g,h,i,j)
    #define rocsparse_analysis(a,b,c,d,e,f,g,h,i,j,k)     hipsparseZcsrsv2_analysis(a,b,c,d,e,(const hipDoubleComplex*)(f),g,h,i,j,k)
    #define rocsparse_solve(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  hipsparseZcsrsv2_solve(a,b,c,d,(const hipDoubleComplex*)(e),f,(const hipDoubleComplex*)(g),h,i,j,(const hipDoubleComplex*)(k),(hipDoubleComplex*)(l),m,n)
  #endif
#else /* not complex */
  #if defined(PETSC_USE_REAL_SINGLE)
    #define rocsparse_get_svbuffsize hipsparseScsrsv2_bufferSize
    #define rocsparse_analysis       hipsparseScsrsv2_analysis
    #define rocsparse_solve          hipsparseScsrsv2_solve
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define rocsparse_get_svbuffsize hipsparseDcsrsv2_bufferSize
    #define rocsparse_analysis       hipsparseDcsrsv2_analysis
    #define rocsparse_solve          hipsparseDcsrsv2_solve
  #endif
#endif
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define rocsparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       hipsparseCcsrmv((a),(b),(c),(d),(e),(hipComplex*)(f),(g),(hipComplex*)(h),(i),(j),(hipComplex*)(k),(hipComplex*)(l),(hipComplex*)(m))
    #define rocsparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) hipsparseCcsrmm((a),(b),(c),(d),(e),(f),(hipComplex*)(g),(h),(hipComplex*)(i),(j),(k),(hipComplex*)(l),(m),(hipComplex*)(n),(hipComplex*)(o),(p))
    #define rocsparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          hipsparseCcsr2csc((a),(b),(c),(d),(hipComplex*)(e),(f),(g),(hipComplex*)(h),(i),(j),(k),(l))
    #define rocsparse_hyb_spmv(a,b,c,d,e,f,g,h)                 hipsparseChybmv((a),(b),(hipComplex*)(c),(d),(e),(hipComplex*)(f),(hipComplex*)(g),(hipComplex*)(h))
    #define rocsparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              hipsparseCcsr2hyb((a),(b),(c),(d),(hipComplex*)(e),(f),(g),(h),(i),(j))
    #define rocsparse_hyb2csr(a,b,c,d,e,f)                      rocsparseChyb2csr((a),(b),(c),(hipComplex*)(d),(e),(f))
    #define rocsparse_csr_spgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) hipsparseCcsrgemm(a,b,c,d,e,f,g,h,(hipComplex*)i,j,k,l,m,(hipComplex*)n,o,p,q,(hipComplex*)r,s,t)
    #define rocsparse_csr_spgeam(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s)   hipsparseCcsrgeam(a,b,c,(hipComplex*)d,e,f,(hipComplex*)g,h,i,(hipComplex*)j,k,l,(hipComplex*)m,n,o,p,(hipComplex*)q,r,s)
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define rocsparse_csr_spmv(a,b,c,d,e,f,g,h,i,j,k,l,m)       hipsparseZcsrmv((a),(b),(c),(d),(e),(hipDoubleComplex*)(f),(g),(hipDoubleComplex*)(h),(i),(j),(hipDoubleComplex*)(k),(hipDoubleComplex*)(l),(hipDoubleComplex*)(m))
    #define rocsparse_csr_spmm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) hipsparseZcsrmm((a),(b),(c),(d),(e),(f),(hipDoubleComplex*)(g),(h),(hipDoubleComplex*)(i),(j),(k),(hipDoubleComplex*)(l),(m),(hipDoubleComplex*)(n),(hipDoubleComplex*)(o),(p))
    #define rocsparse_csr2csc(a,b,c,d,e,f,g,h,i,j,k,l)          hipsparseZcsr2csc((a),(b),(c),(d),(hipDoubleComplex*)(e),(f),(g),(hipDoubleComplex*)(h),(i),(j),(k),(l))
    #define rocsparse_hyb_spmv(a,b,c,d,e,f,g,h)                 hipsparseZhybmv((a),(b),(hipDoubleComplex*)(c),(d),(e),(hipDoubleComplex*)(f),(hipDoubleComplex*)(g),(hipDoubleComplex*)(h))
    #define rocsparse_csr2hyb(a,b,c,d,e,f,g,h,i,j)              hipsparseZcsr2hyb((a),(b),(c),(d),(hipDoubleComplex*)(e),(f),(g),(h),(i),(j))
    #define rocsparse_hyb2csr(a,b,c,d,e,f)                      rocsparseZhyb2csr((a),(b),(c),(hipDoubleComplex*)(d),(e),(f))
    #define rocsparse_csr_spgemm(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) hipsparseZcsrgemm(a,b,c,d,e,f,g,h,(hipDoubleComplex*)i,j,k,l,m,(hipDoubleComplex*)n,o,p,q,(hipDoubleComplex*)r,s,t)
    #define rocsparse_csr_spgeam(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s)   hipsparseZcsrgeam(a,b,c,(hipDoubleComplex*)d,e,f,(hipDoubleComplex*)g,h,i,(hipDoubleComplex*)j,k,l,(hipDoubleComplex*)m,n,o,p,(hipDoubleComplex*)q,r,s)
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define rocsparse_csr_spmv hipsparseScsrmv
    #define rocsparse_csr_spmm hipsparseScsrmm
    #define rocsparse_csr2csc  hipsparseScsr2csc
    #define rocsparse_hyb_spmv hipsparseShybmv
    #define rocsparse_csr2hyb  hipsparseScsr2hyb
    #define rocsparse_hyb2csr  rocsparseShyb2csr
    #define rocsparse_csr_spgemm hipsparseScsrgemm
    #define rocsparse_csr_spgeam hipsparseScsrgeam
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define rocsparse_csr_spmv hipsparseDcsrmv
    #define rocsparse_csr_spmm hipsparseDcsrmm
    #define rocsparse_csr2csc  hipsparseDcsr2csc
    #define rocsparse_hyb_spmv hipsparseDhybmv
    #define rocsparse_csr2hyb  hipsparseDcsr2hyb
    #define rocsparse_hyb2csr  hipsparseDhyb2csr
    #define rocsparse_csr_spgemm hipsparseDcsrgemm
    #define rocsparse_csr_spgeam hipsparseDcsrgeam
  #endif
#endif

#define THRUSTINTARRAY32 thrust::device_vector<int>
#define THRUSTINTARRAY thrust::device_vector<PetscInt>
#define THRUSTARRAY thrust::device_vector<PetscScalar>

/* A CSR matrix structure */
struct CsrMatrix {
  PetscInt         num_rows;
  PetscInt         num_cols;
  PetscInt         num_entries;
  THRUSTINTARRAY32 *row_offsets;
  THRUSTINTARRAY32 *column_indices;
  THRUSTARRAY      *values;
};

/* This is struct holding the relevant data needed to a MatSolve */
struct Mat_SeqAIJHIPSPARSETriFactorStruct {
  /* Data needed for triangular solve */
  /* rocsparseMatDescr_t */
  rocsparse_mat_descr          descr;
  rocsparse_operation_         solveOp;
  /* rocsparseOperation_t */
  CsrMatrix                   *csrMat;
  csrsv2Info_t                solveInfo;
  rocsparse_solve_policy_     solvePolicy;     /* whether level information is generated and used */
  /* rocsparseSolvePolicy_t */
  int                         solveBufferSize;
  void                        *solveBuffer;
  size_t                      csr2cscBufferSize; /* to transpose the triangular factor (only used for CUDA >= 11.0) */
  void                        *csr2cscBuffer;
  PetscScalar                 *AA_h; /* managed host buffer for moving values to the GPU */
};

/* This is a larger struct holding all the triangular factors for a solve, transpose solve, and any indices used in a reordering */
struct Mat_SeqAIJHIPSPARSETriFactors {
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtr; /* pointer for lower triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtr; /* pointer for upper triangular (factored matrix) on GPU */
  Mat_SeqAIJHIPSPARSETriFactorStruct *loTriFactorPtrTranspose; /* pointer for lower triangular (factored matrix) on GPU for the transpose (useful for BiCG) */
  Mat_SeqAIJHIPSPARSETriFactorStruct *upTriFactorPtrTranspose; /* pointer for upper triangular (factored matrix) on GPU for the transpose (useful for BiCG)*/
  THRUSTINTARRAY                    *rpermIndices;  /* indices used for any reordering */
  THRUSTINTARRAY                    *cpermIndices;  /* indices used for any reordering */
  THRUSTARRAY                       *workVector;
  rocsparse_handle                  handle;   /* a handle to the rocsparse library */
  PetscInt                          nnz;      /* number of nonzeros ... need this for accurate logging between ICC and ILU */
};

struct Mat_rocsparseSpMV {
  PetscBool             initialized;    /* Don't rely on spmvBuffer != NULL to test if the struct is initialized, */
  size_t                spmvBufferSize; /* since I'm not sure if smvBuffer can be NULL even after rocsparseSpMV_bufferSize() */
  void                  *spmvBuffer;
};

/* This is struct holding the relevant data needed to a MatMult */
struct Mat_SeqAIJHIPSPARSEMultStruct {
  void               *mat;  /* opaque pointer to a matrix. This could be either a hipsparseHybMat_t or a CsrMatrix */
  rocsparse_mat_descr descr; /* Data needed to describe the matrix for a multiply */
  THRUSTINTARRAY     *cprowIndices;   /* compressed row indices used in the parallel SpMV */
  PetscScalar        *alpha_one; /* pointer to a device "scalar" storing the alpha parameter in the SpMV */
  PetscScalar        *beta_zero; /* pointer to a device "scalar" storing the beta parameter in the SpMV as zero*/
  PetscScalar        *beta_one; /* pointer to a device "scalar" storing the beta parameter in the SpMV as one */
};

/* This is a larger struct holding all the matrices for a SpMV, and SpMV Tranpose */
struct Mat_SeqAIJHIPSPARSE {
  Mat_SeqAIJHIPSPARSEMultStruct *mat;            /* pointer to the matrix on the GPU */
  Mat_SeqAIJHIPSPARSEMultStruct *matTranspose;   /* pointer to the matrix on the GPU (for the transpose ... useful for BiCG) */
  THRUSTARRAY                  *workVector;     /* pointer to a workvector to which we can copy the relevant indices of a vector we want to multiply */
  THRUSTINTARRAY32             *rowoffsets_gpu; /* rowoffsets on GPU in non-compressed-row format. It is used to convert CSR to CSC */
  PetscInt                     nrows;           /* number of rows of the matrix seen by GPU */
  MatHIPSPARSEStorageFormat     format;          /* the storage format for the matrix on the device */
  hipStream_t                 stream;          /* a stream for the parallel SpMV ... this is not owned and should not be deleted */
  rocsparse_handle             handle;          /* a handle to the rocsparse library ... this may not be owned (if we're working in parallel i.e. multiGPUs) */
  PetscObjectState             nonzerostate;    /* track nonzero state to possibly recreate the GPU matrix */
  PetscBool                    transgen;        /* whether or not to generate explicit transpose for MatMultTranspose operations */
  PetscBool                    transupdated;    /* whether or not the explicitly generated transpose is up-to-date */
  THRUSTINTARRAY               *csr2csc_i;
  PetscSplitCSRDataStructure   *deviceMat;       /* Matrix on device for, eg, assembly */
  THRUSTINTARRAY               *cooPerm;
  THRUSTINTARRAY               *cooPerm_a;
};

PETSC_INTERN PetscErrorCode MatHIPSPARSECopyToGPU(Mat);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetStream(Mat, const hipStream_t stream);
PETSC_INTERN PetscErrorCode MatHIPSPARSESetHandle(Mat, const rocsparse_handle handle);
PETSC_INTERN PetscErrorCode MatHIPSPARSEClearHandle(Mat);
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJHIPSPARSE(Mat,PetscInt,const PetscInt[],const PetscInt[]);
PETSC_INTERN PetscErrorCode MatSetValuesCOO_SeqAIJHIPSPARSE(Mat,const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSEGetArrayRead(Mat,const PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayRead(Mat,const PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSEGetArrayWrite(Mat,PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreArrayWrite(Mat,PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSEGetArray(Mat,PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSERestoreArray(Mat,PetscScalar**);
PETSC_INTERN PetscErrorCode MatSeqAIJHIPSPARSEMergeMats(Mat,Mat,MatReuse,Mat*);

PETSC_STATIC_INLINE bool isHipMem(const void *data)
{
  hipError_t                  cerr;
  struct hipPointerAttribute_t attr;
  enum hipMemoryType          mtype;
  cerr = hipPointerGetAttributes(&attr,data);
  hipGetLastError(); /* Reset the last error */
  mtype = attr.memoryType;
  if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) return true;
  else return false;
}

#endif
