#include "petsc/private/petscimpl.h"
#include <petscsystypes.h>
#include <petscerror.h>
#include <petscveckokkos.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spmv.hpp>
#include <../src/mat/impls/aij/seq/aij.h>

#include <../src/mat/impls/aij/seq/kokkos/aijkokkosimpl.hpp>
#include <petscmat.h>

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat); /* Forward declaration */

static PetscErrorCode MatAssemblyEnd_SeqAIJKokkos(Mat A,MatAssemblyType mode)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJKokkos  *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqAIJ(A,mode);CHKERRQ(ierr);
  if (aijkok && aijkok->device_mat_d.data()) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // in GPU mode, no going back. MatSetValues checks this
  }
  /* Don't build (or update) the Mat_SeqAIJKokkos struct. We delay it to the very last moment until we need it. */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqAIJ       *aijseq = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscInt         nrows   = A->rmap->n,ncols = A->cmap->n,nnz = aijseq->nz;

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  /* If aijkok is not built yet or the nonzero pattern on CPU has changed, build aijkok from the scratch */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) {
    delete aijkok;
    aijkok               = new Mat_SeqAIJKokkos(nrows,ncols,nnz,aijseq->i,aijseq->j,aijseq->a);
    aijkok->nonzerostate = A->nonzerostate;
    A->spptr             = aijkok;
  } else if (A->offloadmask == PETSC_OFFLOAD_CPU) { /* Copy values only */
    Kokkos::deep_copy(aijkok->a_d,aijkok->a_h);
  }
  A->offloadmask = PETSC_OFFLOAD_BOTH;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJKokkosSyncHost(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  PetscCheckTypeName(A,MATSEQAIJKOKKOS);
  if (A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (!aijkok) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing AIJKOK");
    Kokkos::deep_copy(aijkok->a_h,aijkok->a_d);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJKokkos(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncHost(A);CHKERRQ(ierr);
  *array = a->a;
  A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

// MatSeqAIJKokkosSetDeviceMat takes a PetscSplitCSRDataStructure with device data and copies it to the device. Note, "deep_copy" here is really a shallow copy
PETSC_EXTERN PetscErrorCode MatSeqAIJKokkosSetDeviceMat(Mat A, PetscSplitCSRDataStructure *h_mat)
{
  Mat_SeqAIJKokkos *aijkok;
  Kokkos::View<PetscSplitCSRDataStructure, Kokkos::HostSpace> h_mat_k(h_mat);

  PetscFunctionBegin;
  // ierr    = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  if (!aijkok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"no Mat_SeqAIJKokkos");
  aijkok->device_mat_d = create_mirror(DeviceMemorySpace(),h_mat_k);
  Kokkos::deep_copy (aijkok->device_mat_d, h_mat_k);
  PetscFunctionReturn(0);
}

// MatSeqAIJKokkosGetDeviceMat gets the device if it is here, otherwise it creates a place for it and returns NULL
PETSC_EXTERN PetscErrorCode MatSeqAIJKokkosGetDeviceMat(Mat A, PetscSplitCSRDataStructure **d_mat)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  if (aijkok && aijkok->device_mat_d.data()) {
    *d_mat = aijkok->device_mat_d.data();
  } else {
    PetscErrorCode   ierr;
    ierr    = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr); // create aijkok (we are making d_mat now so make a place for it)
    *d_mat  = NULL;
  }
  PetscFunctionReturn(0);
}

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr   = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr   = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr   = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A x + beta y */
  ierr   = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr   = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  /* 2.0*aijkok->csr.nnz()-aijkok->csr.numRows() seems more accurate here but assumes there are no zero-rows. So a little sloopy here. */
  ierr   = WaitForKokkos();CHKERRQ(ierr);
  ierr   = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("T",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A^T x + beta y */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv;
  PetscScalarViewDevice_t          yv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(yy,&yv);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("C",1.0/*alpha*/,aijkok->csr,xv,0.0/*beta*/,yv); /* y = alpha A^H x + beta y */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(yy,&yv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("N",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("T",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A^T x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode                   ierr;
  Mat_SeqAIJKokkos                 *aijkok;
  ConstPetscScalarViewDevice_t     xv,yv;
  PetscScalarViewDevice_t          zv;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosGetDeviceView(zz,&zv);CHKERRQ(ierr);
  if (zz != yy) Kokkos::deep_copy(zv,yv);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosSparse::spmv("C",1.0/*alpha*/,aijkok->csr,xv,1.0/*beta*/,zv); /* z = alpha A^H x + beta z */
  ierr = VecKokkosRestoreDeviceViewRead(xx,&xv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceViewRead(yy,&yv);CHKERRQ(ierr);
  ierr = VecKokkosRestoreDeviceView(zz,&zv);CHKERRQ(ierr);
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*aijkok->csr.nnz());CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode ierr;
  Mat            B;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) { /* Build a new mat */
    ierr = MatDuplicate(A,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) { /* Reuse the mat created before */
    ierr = MatCopy(A,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  B    = *newmat;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECKOKKOS,&B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSetOps_SeqAIJKokkos(B);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqAIJGetArray_C",MatSeqAIJGetArray_SeqAIJKokkos);CHKERRQ(ierr);
  /* TODO: see ViennaCL and CUSPARSE once we have a BindToCPU? */
  aij  = (Mat_SeqAIJ*)B->data;
  aij->inode.use = PETSC_FALSE;

  B->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqAIJKokkos(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,cpvalues,B);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(*B,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqAIJKokkos(Mat A)
{
  PetscErrorCode        ierr;
  Mat_SeqAIJKokkos      *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);

  PetscFunctionBegin;
  if (aijkok) {
    if (aijkok->device_mat_d.data()) {
      delete aijkok->colmap_d;
      delete aijkok->i_uncompressed_d;
    }
    if (aijkok->diag_d) delete aijkok->diag_d;
  }
  delete aijkok;
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqAIJGetArray_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode MatMatKernelHandleDestroy_Private(void* data)
{
  MatMatKernelHandle_t *kh = static_cast<MatMatKernelHandle_t *>(data);

  PetscFunctionBegin;
  delete kh;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product          *product = C->product;
  Mat                  A,B;
  MatProductType       ptype;
  Mat_SeqAIJKokkos     *akok,*bkok,*ckok;
  bool                 tA,tB;
  PetscErrorCode       ierr;
  MatMatKernelHandle_t *kh;
  Mat_SeqAIJ           *c;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  A = product->A;
  B = product->B;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ckok = static_cast<Mat_SeqAIJKokkos*>(C->spptr);
  kh   = static_cast<MatMatKernelHandle_t*>(C->product->data);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    tA = false;
    tB = false;
    break;
  case MATPRODUCT_AtB:
    tA = true;
    tB = false;
    break;
  case MATPRODUCT_ABt:
    tA = false;
    tB = true;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }

  KokkosSparse::spgemm_numeric(*kh, akok->csr, tA, bkok->csr, tB, ckok->csr);
  C->offloadmask = PETSC_OFFLOAD_GPU;
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  c = (Mat_SeqAIJ*)C->data;
  ierr = PetscInfo3(C,"Matrix size: %D X %D; storage space: 0 unneeded,%D used\n",C->rmap->n,C->cmap->n,c->nz);CHKERRQ(ierr);
  ierr = PetscInfo(C,"Number of mallocs during MatSetValues() is 0\n");CHKERRQ(ierr);
  ierr = PetscInfo1(C,"Maximum nonzeros in any row is %D\n",c->rmax);CHKERRQ(ierr);
  c->reallocs         = 0;
  C->info.mallocs    += 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  /* we can remove these calls when MatSeqAIJGetArray operations are used everywhere! */
  // TODO JZ, copy from device to host since most of Petsc code for AIJ matrices does not use MatSeqAIJGetArray()
  C->offloadmask = PETSC_OFFLOAD_BOTH;
  // Also, we should add support to copy back from device to host
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product          *product = C->product;
  Mat                  A,B;
  MatProductType       ptype;
  Mat_SeqAIJKokkos     *akok,*bkok,*ckok;
  PetscInt             m,n,k;
  bool                 tA,tB;
  PetscErrorCode       ierr;
  Mat_SeqAIJ           *c;
  MatMatKernelHandle_t *kh;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A = product->A;
  B = product->B;
  // TODO only copy the i,j data, not the values
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr);
  akok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  ptype = product->type;
  if (A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  if (B->symmetric && ptype == MATPRODUCT_ABt) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    tA = false;
    tB = false;
    m = A->rmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_AtB:
    tA = true;
    tB = false;
    m = A->cmap->n;
    n = B->cmap->n;
    break;
  case MATPRODUCT_ABt:
    tA = false;
    tB = true;
    m = A->rmap->n;
    n = B->rmap->n;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Unsupported product type %s",MatProductTypes[product->type]);
  }
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  c = (Mat_SeqAIJ*)C->data;

  kh = new MatMatKernelHandle_t;
  // TODO SZ: ADD RUNTIME SELECTION OF THESE
  kh->set_team_work_size(16);
  kh->set_dynamic_scheduling(true);
  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  kh->create_spgemm_handle(KokkosSparse::StringToSPGEMMAlgorithm(myalg));

  /////////////////////////////////////
  // TODO JZ
  ckok = NULL; //new Mat_SeqAIJKokkos();
  C->spptr = ckok;
  KokkosCsrMatrix_t ccsr; // here only to have the code compile
  KokkosSparse::spgemm_symbolic(*kh, akok->csr, tA, bkok->csr, tB, ccsr);
  //cerr = WaitForKOKKOS();CHKERRCUDA(cerr);
  //c->nz = get_nnz_from_ccsr
  //////////////////////////////////////
  c->singlemalloc = PETSC_FALSE;
  c->free_a       = PETSC_TRUE;
  c->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc1(m+1,&c->i);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->j);CHKERRQ(ierr);
  ierr = PetscMalloc1(c->nz,&c->a);CHKERRQ(ierr);
  ////////////////////////////////////
  // TODO JZ copy from device to c->i and c->j
  ////////////////////////////////////
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

  C->nonzerostate++;
  ierr = PetscLayoutSetUp(C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->cmap);CHKERRQ(ierr);
  ierr = MatMarkDiagonal_SeqAIJ(C);CHKERRQ(ierr);
  ckok->nonzerostate = C->nonzerostate;
  C->offloadmask   = PETSC_OFFLOAD_UNALLOCATED;
  C->preallocated  = PETSC_TRUE;
  C->assembled     = PETSC_FALSE;
  C->was_assembled = PETSC_FALSE;

  C->ops->productnumeric = MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos;
  C->product->data = kh;
  C->product->destroy = MatMatKernelHandleDestroy_Private;
  PetscFunctionReturn(0);
}

/* handles sparse matrix matrix ops */
PETSC_UNUSED static PetscErrorCode MatProductSetFromOptions_SeqAIJKokkos(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscErrorCode ierr;
  PetscBool      Biskok = PETSC_FALSE,Ciskok = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  ierr = PetscObjectTypeCompare((PetscObject)product->B,MATSEQAIJKOKKOS,&Biskok);CHKERRQ(ierr);
  if (product->type == MATPRODUCT_ABC) {
    ierr = PetscObjectTypeCompare((PetscObject)product->C,MATSEQAIJKOKKOS,&Ciskok);CHKERRQ(ierr);
  }
  if (Biskok && Ciskok) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos;
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
#endif

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_SeqAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqAIJKokkos(A,MATSEQAIJKOKKOS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_SeqAIJKokkos(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJKokkos  *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  PetscFunctionBegin;
  if (aijkok && aijkok->device_mat_d.data()) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mixing GPU and non-GPU assembly not supported");
  }
  ierr = MatSetValues_SeqAIJ(A,m,im,n,in,v,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqAIJKokkos(Mat A, PetscScalar a)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  KokkosBlas::scal(aijkok->a_d,a,aijkok->a_d);
  A->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = WaitForKokkos();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(aijkok->a_d.size());CHKERRQ(ierr);
  // TODO Remove: this can be removed once we implement matmat operations with KOKKOS
  ierr = MatSeqAIJKokkosSyncHost(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqAIJKokkos(Mat A)
{
  PetscErrorCode   ierr;
  PetscBool        both = PETSC_FALSE;
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr);
  Mat_SeqAIJ       *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (aijkok && aijkok->a_d.data()) {
    KokkosBlas::fill(aijkok->a_d,0.0);
    both = PETSC_TRUE;
  }
  ierr = PetscArrayzero(a->a,a->i[A->rmap->n]);CHKERRQ(ierr);
  ierr = MatSeqAIJInvalidateDiagonal(A);CHKERRQ(ierr);
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqAIJKokkos(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *x = (Mat_SeqAIJ*)X->data,*y = (Mat_SeqAIJ*)Y->data;

  PetscFunctionBegin;
  if (X->ops->axpy != Y->ops->axpy) {
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (str != SAME_NONZERO_PATTERN && x->nz == y->nz) {
    PetscBool e;
    ierr = PetscArraycmp(x->i,y->i,Y->rmap->n+1,&e);CHKERRQ(ierr);
    if (e) {
      ierr = PetscArraycmp(x->j,y->j,y->nz,&e);CHKERRQ(ierr);
      if (e) str = SAME_NONZERO_PATTERN;
    }
  }
  if (str != SAME_NONZERO_PATTERN) {
    ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else {
    if (Y->offloadmask == PETSC_OFFLOAD_CPU && X->offloadmask == PETSC_OFFLOAD_CPU) {
      ierr = MatAXPY_SeqAIJ(Y,a,X,str);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else {
      ierr = MatSeqAIJKokkosSyncDevice(X);CHKERRQ(ierr);
      ierr = MatSeqAIJKokkosSyncDevice(Y);CHKERRQ(ierr);
    }
    Mat_SeqAIJKokkos *aijkokY = static_cast<Mat_SeqAIJKokkos*>(Y->spptr);
    Mat_SeqAIJKokkos *aijkokX = static_cast<Mat_SeqAIJKokkos*>(X->spptr);
    if (aijkokY && aijkokX && aijkokY->a_d.data() && aijkokX->a_d.data()) {
      KokkosBlas::axpy(a,aijkokX->a_d,aijkokY->a_d);
      Y->offloadmask = PETSC_OFFLOAD_GPU;
      ierr = WaitForKokkos();CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(2.0*aijkokY->a_d.size());CHKERRQ(ierr);
      // TODO Remove: this can be removed once we implement matmat operations with KOKKOS
      ierr = MatSeqAIJKokkosSyncHost(Y);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"no Mat_SeqAIJKokkos ???");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  A->boundtocpu = PETSC_FALSE;

  A->ops->setvalues                 = MatSetValues_SeqAIJKokkos; /* protect with DEBUG, but MatSeqAIJSetTotalPreallocation defeats this ??? */
  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;
  A->ops->axpy                      = MatAXPY_SeqAIJKokkos;
  A->ops->scale                     = MatScale_SeqAIJKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqAIJKokkos;
  //A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJKokkos;
  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/
/*C
   MatCreateSeqAIJKokkos - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format). This matrix will ultimately be handled by
   Kokkos for calculations. For good matrix
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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATSeqAIJKokkos, MATAIJKOKKOS
@*/
PetscErrorCode  MatCreateSeqAIJKokkos(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJKOKKOS);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// factorizations
typedef Kokkos::TeamPolicy<>::member_type team_member;
//
// This factorization exploits block diagonal matrices with "Nf" attached to the matrix in a container.
// Use -pc_factor_mat_ordering_type rcm to order decouple blocks of size N/Nf for this optimization
//
static PetscErrorCode MatLUFactorNumeric_SeqAIJKOKKOS(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *b=(Mat_SeqAIJ*)B->data;
  Mat_SeqAIJKokkos   *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr), *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  IS                 isrow = b->row,isicol = b->icol;
  PetscErrorCode     ierr;
  const PetscInt     *r_h,*ic_h;
  const PetscInt     n=A->rmap->n, *ai_d=aijkok->i_d.data(), *aj_d=aijkok->j_d.data(), *bi_d=baijkok->i_d.data(), *bj_d=baijkok->j_d.data(), *bdiag_d = baijkok->diag_d->data();
  const PetscScalar  *aa_d = aijkok->a_d.data();
  PetscScalar        *ba_d = baijkok->a_d.data();
  PetscBool          row_identity,col_identity;
  PetscInt           Nf,nc;
  PetscContainer     container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "Nf", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscInt *pNf=NULL;
    ierr = PetscContainerGetPointer(container, (void **) &pNf);CHKERRQ(ierr);
    Nf = *pNf;
  } else Nf = 1;
  if (n%Nf) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"n % Nf != 0 %D %D",n,Nf);
  ierr = ISGetIndices(isrow,&r_h);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic_h);CHKERRQ(ierr);
  ierr = ISGetSize(isicol,&nc);CHKERRQ(ierr);
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE)
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#endif
  {
#define KOKKOS_SHARED_LEVEL 1
    const Kokkos::View<const PetscInt*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_r_k (r_h, n);
    Kokkos::View<PetscInt*, Kokkos::LayoutLeft> d_r_k ("r", n);
    const Kokkos::View<const PetscInt*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_ic_k (ic_h, nc);
    Kokkos::View<PetscInt*, Kokkos::LayoutLeft> d_ic_k ("ic", nc);
    double flops_h = 0.0;
    Kokkos::View<double, Kokkos::HostSpace> h_flops_k (&flops_h);
    Kokkos::View<double> d_flops_k ("flops");
    using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using scr_t = Kokkos::View<PetscScalar*, Kokkos::LayoutRight, scr_mem_t>;
    const int nloc = n/Nf, scr_bytes = scr_t::shmem_size(nloc), bnnz = b->nz;

    Kokkos::deep_copy (d_flops_k, h_flops_k);
    Kokkos::deep_copy (d_r_k, h_r_k);
    Kokkos::deep_copy (d_ic_k, h_ic_k);
    // LU factorization that outputs flops
    Kokkos::parallel_for(Kokkos::TeamPolicy<>( (int)Nf, 1, 32).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
        scr_t           rtmp(team.team_scratch(KOKKOS_SHARED_LEVEL),nloc);
        const PetscInt  start = team.league_rank()*nloc, end = start + nloc;
        const PetscInt  *ic = d_ic_k.data(), *r = d_r_k.data();
        double flops = 0;

        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, team.league_rank()*bnnz/Nf, (team.league_rank()+1)*bnnz/Nf), [=] (int idx) {ba_d[idx] = 0;});
        team.team_barrier();

        for (int i=start; i<end; i++) {
          const PetscScalar *v;
          PetscScalar       *pv;
          const PetscInt    *ajtmp,*bjtmp,*pj;
          PetscInt          nz,nzL;
          /* L part */
          nz    = bi_d[i+1] - bi_d[i];
          bjtmp = bj_d + bi_d[i];
          /* zero rtmp */
          // for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nz), [=] (int j) { rtmp(bjtmp[j]-start) = .0; });

          /* U part */
          nz    = bdiag_d[i]-bdiag_d[i+1];
          bjtmp = bj_d + bdiag_d[i+1]+1;
          //for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nz), [=] (int j) { rtmp(bjtmp[j]-start) = .0; });

          /* load in initial (unfactored row) */
          nz    = ai_d[r[i]+1] - ai_d[r[i]];
          ajtmp = aj_d + ai_d[r[i]];
          v     = aa_d + ai_d[r[i]];
          //for (j=0; j<nz; j++) {
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nz), [=] (int j) { rtmp(ic[ajtmp[j]]-start) = v[j]; });

          /* elimination - for all cols in row i */
          bjtmp = bj_d + bi_d[i];
          nzL   = bi_d[i+1] - bi_d[i];
          for (int k=0; k < nzL; k++) {
            //Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nzL), [=] (int kkk) {
            PetscScalar    *pc, multiplier;
            PetscInt       row = bjtmp[k], nz2;
            PetscScalar    *pv2;
            const PetscInt *pj2;

            pc = &rtmp(row - start);
            if (*pc == 0.0) printf("ERROR: zero diagonal\n");

            pv2         = ba_d + bdiag_d[row];

            multiplier = *pc * (*pv2);
            Kokkos::single(Kokkos::PerThread(team), [=]() { *pc = multiplier; });

            pj2 = bj_d + bdiag_d[row+1]+1; /* beginning of U(row,:) */
            pv2 = ba_d + bdiag_d[row+1]+1;
            nz2 = bdiag_d[row]-bdiag_d[row+1]-1; /* num of entries in U(row,:) excluding diag */
            // for (int j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nz2), [=] (const int &j) { rtmp(pj2[j]-start) -= multiplier * pv2[j]; });
            flops += (double)(1 + 2.0*nz2);
          }

          /* finished row so stick it into ba_d */
          /* L part */
          pv = ba_d + bi_d[i];
          pj = bj_d + bi_d[i];
          nz = bi_d[i+1] - bi_d[i];

          //for (j=0; j<nz; j++) { pv[j] = rtmp[pj[j]];
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nz), [=] (const int &j) { pv[j] = rtmp(pj[j]-start); });

          /* U part */
          pv = ba_d + bdiag_d[i+1]+1;
          pj = bj_d + bdiag_d[i+1]+1;
          nz = bdiag_d[i] - bdiag_d[i+1]-1;

          //for (j=0; j<nz; j++) { pv[j] = rtmp[pj[j]];
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team,nz), [=] (const int &j) { pv[j] = rtmp(pj[j]-start); });
          /* Invert diagonal for simpler triangular solves */
          pv = ba_d + bdiag_d[i];
          Kokkos::single(Kokkos::PerTeam(team), [=]() { *pv = 1.0/rtmp(i-start); });
        } /* endof for (i=0; i<n; i++) { */
        Kokkos::single(Kokkos::PerThread(team), [=]() { Kokkos::atomic_add( &d_flops_k(), flops); });
      }); // outer loop over species
    Kokkos::deep_copy (h_flops_k, d_flops_k);
    Kokkos::fence();
#if defined(PETSC_HAVE_DEVICE)
    ierr = PetscLogGpuFlops(h_flops_k());CHKERRQ(ierr);
#else
    ierr = PetscLogFlops(h_flops_k());CHKERRQ(ierr);
#endif
  }
#if defined(PETSC_HAVE_DEVICE)
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
  ierr = ISRestoreIndices(isicol,&ic_h);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r_h);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (b->inode.size) {
    B->ops->solve = MatSolve_SeqAIJ_Inode;
  } else if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJ; // at least this needs to be in Kokkos
  }
  B->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJKokkosSyncHost(B);CHKERRQ(ierr); // solve on CPU
  B->ops->solveadd          = MatSolveAdd_SeqAIJ; // and this
  B->ops->solvetranspose    = MatSolveTranspose_SeqAIJ;
  B->ops->solvetransposeadd = MatSolveTransposeAdd_SeqAIJ;
  B->ops->matsolve          = MatMatSolve_SeqAIJ;
  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;


  
ierr = ISGetIndices(isrow,&r_h);CHKERRQ(ierr);
ierr = ISGetIndices(isicol,&ic_h);CHKERRQ(ierr);
ierr = MatView(A,0);CHKERRQ(ierr);
for (int i=0;i<n;i++) {
  PetscScalar    *pv;
  const PetscInt *pj;
  printf("%d) diag=%d, b->i = %d,row perm = %d, inverse col perm = %d\n",i,(*baijkok->diag_d)(i), bi_d[i], r_h[i],ic_h[i]);
  // L
  pv = ba_d + bi_d[i];
  pj = bj_d + bi_d[i];
  int nz = bi_d[i+1] - bi_d[i];
  for (int j=0;j<nz;j++) {
    printf("L(%d,%d)=%f (%ld); ",i,pj[j],pv[j],&pv[j]-ba_d);
  }
  printf(";");
  pv = ba_d + bdiag_d[i+1]+1;
  pj = bj_d + bdiag_d[i+1]+1;
  nz = bdiag_d[i] - bdiag_d[i+1]; // -1
  for (int j=0;j<nz;j++) {
    printf("U(%d,%d)=%f (%ld); ",i,pj[j],pv[j],&pv[j]-ba_d);
  }
  printf("\n");
 }
 printf("%d) diag=%d, b->i = %d\n",n,(*baijkok->diag_d)(n),bi_d[n]);
ierr = ISRestoreIndices(isrow,&r_h);CHKERRQ(ierr);
ierr = ISRestoreIndices(isicol,&ic_h);CHKERRQ(ierr);



 
   PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumericV2_SeqAIJKOKKOS(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *b=(Mat_SeqAIJ*)B->data;
  Mat_SeqAIJKokkos   *aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr), *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
  IS                 isrow = b->row,isicol = b->icol;
  PetscErrorCode     ierr;
  const PetscInt     *r_h,*ic_h;
  const PetscInt     n=A->rmap->n, *ai_d=aijkok->i_d.data(), *aj_d=aijkok->j_d.data(), *bi_d=baijkok->i_d.data(), *bj_d=baijkok->j_d.data(), *bdiag_d = baijkok->diag_d->data();
  const PetscScalar  *aa_d = aijkok->a_d.data();
  PetscScalar        *ba_d = baijkok->a_d.data();
  PetscBool          row_identity,col_identity;
  PetscInt           nc; int Nf;
  PetscContainer     container;

  PetscFunctionBegin;
  if (A->rmap->n != n) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"square matrices only supported %D %D",A->rmap->n,n);
  ierr = PetscObjectQuery((PetscObject) A, "Nf", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscInt *pNf=NULL;
    ierr = PetscContainerGetPointer(container, (void **) &pNf);CHKERRQ(ierr);
    Nf = *pNf;
  } else Nf = 1;
  if (n%Nf) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"n % Nf != 0 %D %D",n,Nf);
  ierr = ISGetIndices(isrow,&r_h);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic_h);CHKERRQ(ierr);
  ierr = ISGetSize(isicol,&nc);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_USE_LOG)
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#endif
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr);
  {
#define KOKKOS_SHARED_LEVEL 1
#define NSM_FIELD 1
    const Kokkos::View<const PetscInt*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_r_k (r_h, n);
    Kokkos::View<PetscInt*, Kokkos::LayoutLeft> d_r_k ("r", n);
    const Kokkos::View<const PetscInt*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_ic_k (ic_h, nc);
    Kokkos::View<PetscInt*, Kokkos::LayoutLeft> d_ic_k ("ic", nc);
    int flops_h = 0.0;
    Kokkos::View<int, Kokkos::HostSpace> h_flops_k (&flops_h);
    Kokkos::View<int> d_flops_k ("flops");
    // using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    // using scr_t = Kokkos::View<PetscScalar*, Kokkos::LayoutRight, scr_mem_t>; , scr_bytes = scr_t::shmem_size(nloc)
    const int nloc = n/Nf, Ni_t = NSM_FIELD, Ni = (Ni_t > nloc/2+1) ? nloc/2+1 : Ni_t, bnnz = b->nz, bnzLoc = bnnz/Nf;
    const int conc = Kokkos::DefaultExecutionSpace().concurrency(), nVec = 32, team_size = conc > 8 ? 1 : 1; // 8 !!!!
    Kokkos::deep_copy (d_flops_k, h_flops_k);
    Kokkos::deep_copy (d_r_k, h_r_k);
    Kokkos::deep_copy (d_ic_k, h_ic_k);
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, team_size, nVec), KOKKOS_LAMBDA (const team_member team) {
        //scr_t  rtmp(team.team_scratch(KOKKOS_SHARED_LEVEL),nloc);
        const PetscInt  lg_rank = team.league_rank(), field = lg_rank/Ni, field_block = lg_rank%Ni;
        const PetscInt  nzlc_i =  (bnzLoc/Ni + !!(bnzLoc%Ni)), nzstr_i = field*bnzLoc + field_block*nzlc_i, nzed_i = (nzstr_i + nzlc_i)  > (field+1)*bnzLoc ? (field+1)*bnzLoc : (nzstr_i   +  nzlc_i);
        // zero out B (fill)
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nzstr_i, nzed_i), [=] (const int &idx) { ba_d[idx] = 0; //printf("%d/%d) zero B (%d)\n",idx,bnnz,field_block);
          });
      });
    Kokkos::fence();
    // fuse these two loops and for loe concurrency (CPU) force Ni=1
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, team_size, nVec), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt  lg_rank = team.league_rank(), field = lg_rank/Ni, field_block = lg_rank%Ni;
        const PetscInt  nloc_i =  (nloc/Ni   + !!(nloc%Ni)),   start_i = field*nloc   + field_block*nloc_i, end_i  = (start_i + nloc_i)  > (field+1)*nloc ?   (field+1)*nloc   : (start_i   +  nloc_i);
        const PetscInt  *ic = d_ic_k.data(), *r = d_r_k.data();
        printf("%d) local_i %d -- %d. %d == %d; %d == %d, nloc_i=%d %d %d\n",lg_rank,start_i,end_i,team.league_size(),Nf*Ni,team_size,team.team_size(),nloc_i,nloc/Ni,nloc%Ni);
        // copy A into B
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start_i, end_i), [=] (const int &rowb) {
            PetscInt          rowa = r[rowb], nza = ai_d[rowa+1] - ai_d[rowa];
            const PetscScalar *av    = aa_d + ai_d[rowa];
            const PetscInt    *ajtmp = aj_d + ai_d[rowa];
            /* load in initial (unfactored row) */
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nza), [=] (const int &ja) {
                PetscInt    colb = ic[ajtmp[ja]];
                PetscScalar vala = av[ja];
                if (colb == rowb) {
                  *(ba_d + bdiag_d[rowb]) = vala;
                  printf(" *B(%d,%d) = %g ",rowb,rowb,vala);
                } else {
                  const PetscInt    *pbj = bj_d + ((colb > rowb) ? bdiag_d[rowb+1]+1 : bi_d[rowb]);
                  PetscScalar       *pba = ba_d + ((colb > rowb) ? bdiag_d[rowb+1]+1 : bi_d[rowb]);
                  PetscInt          nz   = (colb > rowb) ? bdiag_d[rowb] - (bdiag_d[rowb+1]+1) : bi_d[rowb+1] - bi_d[rowb], set=0;
                  for (int j=0; j<nz ; j++) {
                    if (pbj[j] == colb) {
                      pba[j] = vala;
                      printf("  B(%d,%d) = %g [U=%d] ",rowb,colb,vala,colb > rowb);
                      set++;
                      break;
                    }
                  }
                  if (set!=1) printf("\t\t\t ERROR DID NOT SET ?????\n");
                }
              });
            printf("\n");
          });
      });
    Kokkos::fence();


for (int i=0;i<n;i++) {
  PetscScalar    *pv;
  const PetscInt *pj;
  printf("%d) diag=%d, bi=%d, row perm = %d, inverse col perm = %d\n",i,(*baijkok->diag_d)(i),bi_d[i],r_h[i],ic_h[i]);
  // L
  pv = ba_d + bi_d[i];
  pj = bj_d + bi_d[i];
  int nz = bi_d[i+1] - bi_d[i];
  for (int j=0;j<nz;j++) {
    printf("L(%d,%d)=%f (%ld); ",i,pj[j],pv[j],&pv[j]-ba_d);
  }
  printf(";");
  pv = ba_d + bdiag_d[i+1]+1;
  pj = bj_d + bdiag_d[i+1]+1;
  nz = bdiag_d[i] - bdiag_d[i+1]; // -1
  for (int j=0;j<nz;j++) {
    printf("U(%d,%d)=%f (%ld); ",i,pj[j],pv[j],&pv[j]-ba_d);
  }
  printf("\n");
 }
printf("%d) diag=%d, bi[n] = %d\n",n,(*baijkok->diag_d)(n),bi_d[n]);


    Kokkos::parallel_for(Kokkos::TeamPolicy<>(Nf*Ni, team_size, nVec), KOKKOS_LAMBDA (const team_member team) {
        //scr_t  rtmp(team.team_scratch(KOKKOS_SHARED_LEVEL),nloc);
        const PetscInt  lg_rank = team.league_rank(), field = lg_rank/Ni; //, field_offset = lg_rank%Ni;
        const PetscInt  start = field*nloc, end = start + nloc;
        int             flops = 0;

        // A22 panel update for each row A(1,:) and col A(:,1)
        for (int ii=start; ii<end-1; ii++) {
          const PetscInt    *bjUi = bj_d + bdiag_d[ii+1]+1, nzUi = bdiag_d[ii] - (bdiag_d[ii+1]+1); // vector, and vector size, of column indices of U(i,i+1:end)
          const PetscScalar *baUi = ba_d + bdiag_d[ii+1]+1; // vector of data  U(i,i+1:end)
          const PetscInt    nzUi_pad = (nVec*Ni)*(nzUi/(nVec*Ni) + !!(nzUi%(nVec*Ni))); // want all threads active in loop for barriers (needed?)
          const PetscScalar Bii = *(ba_d + bdiag_d[ii]); // diagonal in its special place
          printf("U(%d,%d) = %f\n",ii,ii,Bii);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nzUi_pad/Ni), [&] (const int outer_kIdx) {
              PetscInt kIdx = outer_kIdx*Ni + lg_rank%Ni;
              if (kIdx >= nzUi) {
                team.team_barrier(); // we need to match any barriers in the else clause, I think
              } else {
                const PetscInt myk = bjUi[kIdx]; // assume symmetric structure, need a transposed meta-data here in general
                if (kIdx==0 && myk != ii+1) printf("\t\t\t\t\t\t\tERROR: first k %d != i+1 %d\n",myk,ii+1);
                const PetscInt *pjL = bj_d + bi_d[myk]; // look for L(myk,ii) in start of row
                const PetscInt nzL  = bi_d[myk+1] - bi_d[myk]; // size of L_k(:)
                PetscInt       colkIdx = -1;
                PetscScalar    *pLki = NULL;
                // find and do L(k,i) = A(:k,i) / A(i,i)
                printf("\tLower B(%d,:) nzL = %d search\n",myk,nzL);
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nzL), [&] (const int &j) {
                    if (pjL[j] == ii) {
                      if (colkIdx != -1) printf("\t\t\t\t\t\t\tERROR:%d) found column %d already idx = %d. found in idx %d also\n",myk,ii,colkIdx,j);
                      colkIdx = j;
                      pLki = ba_d + bi_d[myk] + colkIdx;
                      printf("\t\t\t have Lx1[%d,%d], set to %g. Ax1=%g\n",myk,ii,*pLki/Bii,*pLki);
                      *pLki = *pLki/Bii; // column scaling:  L(k,i) = A(:k,i) / A(i,i)
                    }
                  });
                team.team_barrier(); // yuck. this is like a broadcast, I don't see a way to avoid it
                if (colkIdx == -1) printf("\t\t\t\t\t\t\tERROR: failed to find A(%d,%d)\n",myk,ii);
                else { // active row k, do  A_kj -= Lki * U_ij; j \in U(i,:) j != i
                  const PetscScalar L_ki = *pLki;
                  // U(i+1,:end)
                  int ndiag=0;
                  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,nzUi), [&/*=*/] (const int &uiIdx) { // index into i (U)
                      PetscScalar Uij = baUi[uiIdx];
                      PetscInt    col = bjUi[uiIdx];
                      printf("\t\t row %d search for col %d\n",myk,col);
                      if (col==myk) {
                        // A_kk = A_kk - L_ki * U_ij(k)
                        PetscScalar *Akkv = (ba_d + bdiag_d[myk]); // diagonal in its special place
                        *Akkv = *Akkv - L_ki * Uij; // UiK
                        printf("\t\t\tUpdate D A[%d,%d] = %g, with L_ki=%g Uik=%g. 1/D=%g\n", myk, bjUi[uiIdx], *Akkv, L_ki, Uij, 1./(*Akkv));
                        ndiag++;
                      } else {
                        PetscScalar    *start, *end, *pAkjv=NULL;
                        const PetscInt *startj;
                        if (col<myk) { // L
                          PetscInt idx = (pLki+1) - (ba_d + bi_d[myk]); // index into row
                          start = pLki+1; // start at pLki+1, A22(myk,1)
                          startj= bj_d + bi_d[myk] + idx;
                          end   = ba_d + bi_d[myk+1];
                        } else {
                          PetscInt idx = bdiag_d[myk+1]+1;
                          start = ba_d + idx;
                          startj= bj_d + idx;
                          end   = ba_d + bdiag_d[myk];
                        }
                        // search for 'col', use bisection search - TODO
                        printf("\t\t\t row %d search for col %d\n",myk,col);
                        for (pAkjv=start; pAkjv<end; pAkjv++) {
                          printf("\t\t\t\t check col %d for col %d\n",startj[pAkjv-start],col);
                          if (startj[pAkjv-start] == col) break;
                        }
                        if (pAkjv==end) printf("\t\t\t\t\t\t\t\t\t\t\tERROR: failed to find Akj(%d,%d)\n",myk,col);
                        *pAkjv = *pAkjv - L_ki * Uij; // A_kj = A_kj - L_ki * U_ij
                        printf("\t\t\tUpdate %c A[%d,%d] = %g with Uij=%g, Lik=%g\n", (col>myk) ? 'U' : 'L', myk, col, *pAkjv, Uij, L_ki);
                      }
                    });
                  if (ndiag!=1) printf("\t\t\t\t\t\t\tERROR: row %d found %d diagonals\n",myk,ndiag);
                  Kokkos::atomic_add( &flops, 2*nzUi+1);
                }
              }
            });
        } /* endof for (i=0; i<n; i++) { */

        /* Invert diagonal for simpler triangular solves */
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start, end), [=] (int i) {
            PetscScalar *pv = ba_d + bdiag_d[i];
            *pv = 1.0/(*pv);
            printf("\t\t\tInvert A[%d,%d==%d] = %g\n",i,i,*(bj_d + bdiag_d[i]),*pv);
          });
        Kokkos::single(Kokkos::PerThread(team), [=]() { Kokkos::atomic_add( &d_flops_k(), flops); });
      });
    Kokkos::deep_copy (h_flops_k, d_flops_k);
    Kokkos::fence();
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_USE_LOG)
    ierr = PetscLogGpuFlops((PetscLogDouble)h_flops_k());CHKERRQ(ierr);
#elif  defined(PETSC_USE_LOG)
    ierr = PetscLogFlops((PetscLogDouble)h_flops_k());CHKERRQ(ierr);
#endif
  }
#if defined(PETSC_HAVE_DEVICE) && defined(PETSC_USE_LOG)
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
  ierr = ISRestoreIndices(isicol,&ic_h);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r_h);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (b->inode.size) {
    B->ops->solve = MatSolve_SeqAIJ_Inode;
  } else if (row_identity && col_identity) {
    B->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJ; // at least this needs to be in Kokkos
  }
  B->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = MatSeqAIJKokkosSyncHost(B);CHKERRQ(ierr); // solve on CPU
  B->ops->solveadd          = MatSolveAdd_SeqAIJ; // and this
  B->ops->solvetranspose    = MatSolveTranspose_SeqAIJ;
  B->ops->solvetransposeadd = MatSolveTransposeAdd_SeqAIJ;
  B->ops->matsolve          = MatMatSolve_SeqAIJ;
  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;

  
ierr = ISGetIndices(isrow,&r_h);CHKERRQ(ierr);
ierr = ISGetIndices(isicol,&ic_h);CHKERRQ(ierr);
ierr = MatView(A,0);CHKERRQ(ierr);
for (int i=0;i<n;i++) {
  PetscScalar    *pv;
  const PetscInt *pj;
  printf("%d) diag=%d, bi=%d, row perm = %d, inverse col perm = %d\n",i,(*baijkok->diag_d)(i),bi_d[i],r_h[i],ic_h[i]);
  // L
  pv = ba_d + bi_d[i];
  pj = bj_d + bi_d[i];
  int nz = bi_d[i+1] - bi_d[i];
  for (int j=0;j<nz;j++) {
    printf("L(%d,%d)=%f (%ld); ",i,pj[j],pv[j],&pv[j]-ba_d);
  }
  printf(";");
  pv = ba_d + bdiag_d[i+1]+1;
  pj = bj_d + bdiag_d[i+1]+1;
  nz = bdiag_d[i] - bdiag_d[i+1]; // -1
  for (int j=0;j<nz;j++) {
    printf("U(%d,%d)=%f (%ld); ",i,pj[j],pv[j],&pv[j]-ba_d);
  }
  printf("\n");
 }
printf("%d) diag=%d, bi[n] = %d\n",n,(*baijkok->diag_d)(n),bi_d[n]);
ierr = ISRestoreIndices(isrow,&r_h);CHKERRQ(ierr);
ierr = ISRestoreIndices(isicol,&ic_h);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJKOKKOS(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  PetscErrorCode   ierr;
  Mat_SeqAIJ       *b=(Mat_SeqAIJ*)B->data;
  const PetscInt   nrows   = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SeqAIJ(B,A,isrow,iscol,info);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumericV2_SeqAIJKOKKOS;
  // move B data into Kokkos
  ierr = MatSeqAIJKokkosSyncDevice(B);CHKERRQ(ierr); // create aijkok
  ierr = MatSeqAIJKokkosSyncDevice(A);CHKERRQ(ierr); // create aijkok
  {
    Mat_SeqAIJKokkos *baijkok = static_cast<Mat_SeqAIJKokkos*>(B->spptr);
    if (!baijkok->diag_d) {
      const Kokkos::View<PetscInt*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_diag (b->diag,nrows+1);
      baijkok->diag_d = new Kokkos::View<PetscInt*>(Kokkos::create_mirror(DeviceMemorySpace(),h_diag));
      Kokkos::deep_copy (*baijkok->diag_d, h_diag);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijkokkos_kokkos(Mat,MatFactorType,Mat*);

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_KOKKOS(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERKOKKOS,MATSEQAIJKOKKOS,MAT_FACTOR_LU,MatGetFactor_seqaijkokkos_kokkos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_kokkos(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERKOKKOS;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERKOKKOS = "kokkos" - A matrix solver type providing triangular solvers for sequential matrices
  on a single GPU of type, seqaijkokkos, aijkokkos.

  Level: beginner

.seealso: PCFactorSetMatSolverType(), MatSolverType, MatCreateSeqAIJKOKKOS(), MATAIJKOKKOS, MatCreateAIJKOKKOS(), MatKOKKOSSetFormat(), MatKOKKOSStorageFormat, MatKOKKOSFormatOperation
M*/

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijkokkos_kokkos(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  (*B)->useordering = PETSC_TRUE;
  ierr = MatSetType(*B,MATSEQAIJKOKKOS);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU /* || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT*/) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    // (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJKOKKOS;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJKOKKOS;
    // } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    // (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJKOKKOS;
    // (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJKOKKOS;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for KOKKOS Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_kokkos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
