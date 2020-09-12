#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>

PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b              = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQAIJCUSPARSE matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusparseStruct->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusparseStruct->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->A,cusparseStruct->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->B,cusparseStruct->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->A,cusparseStruct->stream);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->B,cusparseStruct->stream);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->rmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCUSPARSESetFormat_MPIAIJCUSPARSE(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a               = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE * cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT_DIAG:
    cusparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_MULT_OFFDIAG:
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparseStruct->diagGPUMatFormat    = format;
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  MatCUSPARSEStorageFormat format;
  PetscErrorCode           ierr;
  PetscBool                flg;
  Mat_MPIAIJ               *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE       *cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPIAIJCUSPARSE options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_DIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_OFFDIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode             ierr;
  Mat_MPIAIJ                 *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE         *cusparseStruct = (Mat_MPIAIJCUSPARSE*)mpiaij->spptr;
  PetscSplitCSRDataStructure *d_mat = cusparseStruct->deviceMat;

  PetscFunctionBegin;
  if (d_mat) { // replicate MatAssemblyEnd_MPIAIJ semantics (yuck)
    Mat_SeqAIJ                 *jaca = (Mat_SeqAIJ*)mpiaij->A->data;
    Mat_SeqAIJ                 *jacb = (Mat_SeqAIJ*)mpiaij->B->data;
    PetscSplitCSRDataStructure h_mat;
    cudaError_t                err;
    PetscInt                   n = A->rmap->n, nnz;
    err = cudaMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    // A
    jaca->nz         = h_mat.diag.nz;
    mpiaij->A->nonzerostate  = h_mat.diag.nonzerostate;
    jaca->nonzerorowcnt = h_mat.diag.nonzerorowcnt;
    jaca->rmax          = h_mat.diag.rmax;
    err = cudaMemcpy( jaca->i,    h_mat.diag.i,    (n+1)*sizeof(PetscInt), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    nnz = jaca->i[n];
    if (jaca->nz != nnz) printf("ERROR, jaca->nz != nnz %d %d\n", jaca->nz, nnz);
    err = cudaMemcpy( jaca->ilen, h_mat.diag.ilen, (n)*sizeof(PetscInt),     cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    err = cudaMemcpy( jaca->imax, h_mat.diag.imax, (n)*sizeof(PetscInt),     cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    err = cudaMemcpy( jaca->j,    h_mat.diag.j,    (nnz)*sizeof(PetscInt),   cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    err = cudaMemcpy( jaca->a,    h_mat.diag.a,    (nnz)*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    // B
    jacb->nz         = h_mat.offdiag.nz;
    mpiaij->B->nonzerostate  = h_mat.offdiag.nonzerostate;
    jacb->nonzerorowcnt = h_mat.offdiag.nonzerorowcnt;
    jacb->rmax          = h_mat.offdiag.rmax;
    err = cudaMemcpy( jacb->i,    h_mat.offdiag.i,    (n+1)*sizeof(PetscInt), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    nnz = jacb->i[n];
    if (jacb->nz != nnz) printf("ERROR, jacb->nz != nnz %d %d\n", jacb->nz, nnz);
    err = cudaMemcpy( jacb->ilen, h_mat.offdiag.ilen, (n)*sizeof(PetscInt),     cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    err = cudaMemcpy( jacb->imax, h_mat.offdiag.imax, (n)*sizeof(PetscInt),     cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    err = cudaMemcpy( jacb->j,    h_mat.offdiag.j,    (nnz)*sizeof(PetscInt),   cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    err = cudaMemcpy( jacb->a,    h_mat.offdiag.a,    (nnz)*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    A->offloadmask = PETSC_OFFLOAD_CPU; // MatCUSPARSE can now copy data to its GPU data structure
    // replace semantics of MatAssemblyEnd_MPIAIJ because MatAssemblyEnd_SeqAIJ does not have the device data, assume not dissembled
    if (mpiaij->A->offloadmask == PETSC_OFFLOAD_CPU) mpiaij->A->offloadmask = PETSC_OFFLOAD_CPU;
    ierr = MatAssemblyBegin(mpiaij->A,mode);CHKERRQ(ierr); // MatAssemblyBegin ???
    ierr = MatAssemblyEnd_SeqAIJ(mpiaij->A,mode);CHKERRQ(ierr);

    if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
      ierr = MatSetUpMultiply_MPIAIJ(A);CHKERRQ(ierr);
    }
    ierr = MatSetOption(mpiaij->B,MAT_USE_INODES,PETSC_FALSE);CHKERRQ(ierr);
    if (A->offloadmask == PETSC_OFFLOAD_CPU && mpiaij->B->offloadmask != PETSC_OFFLOAD_UNALLOCATED) mpiaij->B->offloadmask = PETSC_OFFLOAD_CPU;
    ierr = MatAssemblyBegin(mpiaij->B,mode);CHKERRQ(ierr);  // MatAssemblyBegin ???
    ierr = MatAssemblyEnd_SeqAIJ(mpiaij->B,mode);CHKERRQ(ierr);
    // ???
    ierr = PetscFree2(mpiaij->rowvalues,mpiaij->rowindices);CHKERRQ(ierr);
    mpiaij->rowvalues = NULL;
    ierr = VecDestroy(&mpiaij->diag);CHKERRQ(ierr);
    // if (jaca->inode.size) A->ops->multdiagonalblock = MatMultDiagonalBlock_MPIAIJ;

    /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
    if ((!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqAIJ*)(mpiaij->A->data))->nonew) {
      PetscObjectState state = mpiaij->A->nonzerostate + mpiaij->B->nonzerostate;
      ierr = MPIU_Allreduce(&state,&A->nonzerostate,1,MPIU_INT64,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
    }
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  }
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQCUDA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a              = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;
  cudaError_t        err;
  cusparseStatus_t   stat;

  PetscFunctionBegin;
  if (cusparseStruct->deviceMat) {
    PetscSplitCSRDataStructure *d_mat = cusparseStruct->deviceMat, h_mat;
    err = cudaMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    if (h_mat.diag.i)    {err = cudaFree(h_mat.diag.i);CHKERRCUDA(err);}
    if (h_mat.diag.ilen) {err = cudaFree(h_mat.diag.ilen);CHKERRCUDA(err);}
    if (h_mat.diag.j)    {err = cudaFree(h_mat.diag.j);CHKERRCUDA(err);}
    if (h_mat.diag.a)    {err = cudaFree(h_mat.diag.a);CHKERRCUDA(err);}
    if (h_mat.diag.imax) {err = cudaFree(h_mat.diag.imax);CHKERRCUDA(err);}
    if (h_mat.offdiag.i)    {err = cudaFree(h_mat.offdiag.i);CHKERRCUDA(err);}
    if (h_mat.offdiag.ilen) {err = cudaFree(h_mat.offdiag.ilen);CHKERRCUDA(err);}
    if (h_mat.offdiag.j)    {err = cudaFree(h_mat.offdiag.j);CHKERRCUDA(err);}
    if (h_mat.offdiag.a)    {err = cudaFree(h_mat.offdiag.a);CHKERRCUDA(err);}
    if (h_mat.offdiag.imax) {err = cudaFree(h_mat.offdiag.imax);CHKERRCUDA(err);}
    if (h_mat.colmap) {err = cudaFree(h_mat.colmap);CHKERRCUDA(err);}
    err =                    cudaFree(d_mat);CHKERRCUDA(err);
  }
  try {
    if (a->A) { ierr = MatCUSPARSEClearHandle(a->A);CHKERRQ(ierr); }
    if (a->B) { ierr = MatCUSPARSEClearHandle(a->B);CHKERRQ(ierr); }
    stat = cusparseDestroy(cusparseStruct->handle);CHKERRCUSPARSE(stat);
    if (cusparseStruct->stream) {
      err = cudaStreamDestroy(cusparseStruct->stream);CHKERRCUDA(err);
    }
    delete cusparseStruct;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJCUSPARSE error: %s", ex);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a;
  Mat_MPIAIJCUSPARSE *cusparseStruct;
  cusparseStatus_t   stat;
  Mat                A;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(B,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  A = *newmat;

  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);

  a = (Mat_MPIAIJ*)A->data;
  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    a->spptr = new Mat_MPIAIJCUSPARSE;

    cusparseStruct                      = (Mat_MPIAIJCUSPARSE*)a->spptr;
    cusparseStruct->diagGPUMatFormat    = MAT_CUSPARSE_CSR;
    cusparseStruct->offdiagGPUMatFormat = MAT_CUSPARSE_CSR;
    cusparseStruct->stream              = 0;
    stat = cusparseCreate(&(cusparseStruct->handle));CHKERRCUSPARSE(stat);
  }

  A->ops->assemblyend    = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult           = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd        = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose  = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy        = MatDestroy_MPIAIJCUSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_MPIAIJCUSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJCUSPARSE(A,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateAIJCUSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to NVidia GPUs and use the CUSPARSE library for calculations. For good matrix
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
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJCUSPARSE, MATAIJCUSPARSE
@*/
PetscErrorCode  MatCreateAIJCUSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJCUSPARSE - MATMPIAIJCUSPARSE = "aijcusparse" = "mpiaijcusparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require CUDA 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the CUSPARSE library.

   This matrix type is identical to MATSEQAIJCUSPARSE when constructed with a single process communicator,
   and MATMPIAIJCUSPARSE otherwise.  As a result, for single process communicators,
   MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijcusparse - sets the matrix type to "mpiaijcusparse" during a call to MatSetFromOptions()
.  -mat_cusparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
.  -mat_cusparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

 .seealso: MatCreateAIJCUSPARSE(), MATSEQAIJCUSPARSE, MatCreateSeqAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M
M*/

// get GPU pointer to stripped down Mat. For both Seq and MPI Mat.
PetscErrorCode MatCUSPARSEGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure **B)
{
  PetscSplitCSRDataStructure **p_d_mat;
  PetscMPIInt                size;
  MPI_Comm                   comm;
  PetscErrorCode             ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    if (size == 1) {
      Mat_SeqAIJCUSPARSE *spptr = (Mat_SeqAIJCUSPARSE*)A->spptr;
      p_d_mat = &spptr->deviceMat;
    } else {
      Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)A->data;
      Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;
      p_d_mat = &spptr->deviceMat;
    }
  } else {
    *B = NULL;
    PetscFunctionReturn(0);
  }
  if (!*p_d_mat) {
    cudaError_t                 err;
    PetscSplitCSRDataStructure  *d_mat, h_mat;
    Mat_SeqAIJ                  *jaca;
    PetscInt                    n = A->rmap->n, nnz;
    if (!A->preallocated && A->ops->setup) {
      ierr = PetscInfo(A,"Warning not preallocating matrix storage\n");CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Matrix not preallocated");
    }
    // create and copy
    err = cudaMalloc((void **)&d_mat, sizeof(PetscSplitCSRDataStructure));CHKERRCUDA(err);
    err = cudaMemset( d_mat, 0,       sizeof(PetscSplitCSRDataStructure));CHKERRCUDA(err);
    *B = *p_d_mat = d_mat;
    if (size == 1) {
      jaca = (Mat_SeqAIJ*)A->data;
      h_mat.rstart = 0; h_mat.rend = A->rmap->n;
      h_mat.cstart = 0; h_mat.cend = A->cmap->n;
      h_mat.offdiag.i = h_mat.offdiag.ilen = h_mat.offdiag.imax = h_mat.offdiag.j = NULL;
      h_mat.offdiag.a = NULL;
    } else {
      Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)A->data;
      Mat_SeqAIJ  *jacb;
      if (aij->B->rmap->n != aij->A->rmap->n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only support aij->B->rmap->n == aij->A->rmap->n");
      aij->donotstash = PETSC_TRUE;       // no stashing now
      A->nooffprocentries = PETSC_TRUE;
      // allocate B copy data
      jaca = (Mat_SeqAIJ*)aij->A->data;
      jacb = (Mat_SeqAIJ*)aij->B->data;
      h_mat.rstart = A->rmap->rstart; h_mat.rend = A->rmap->rend;
      h_mat.cstart = A->cmap->rstart; h_mat.cend = A->cmap->rend;
      nnz = jacb->i[n];
      err = cudaMalloc((void **)&h_mat.offdiag.i,               (n+1)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
      err = cudaMemcpy(          h_mat.offdiag.i,    jacb->i,   (n+1)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      err = cudaMalloc((void **)&h_mat.offdiag.ilen,            (n)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
      err = cudaMemcpy(          h_mat.offdiag.ilen, jacb->ilen,(n)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      err = cudaMalloc((void **)&h_mat.offdiag.imax,            (n)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
      err = cudaMemcpy(          h_mat.offdiag.imax, jacb->imax,(n)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      err = cudaMalloc((void **)&h_mat.offdiag.j,               (nnz)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
      err = cudaMemcpy(          h_mat.offdiag.j,    jacb->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      err = cudaMalloc((void **)&h_mat.offdiag.a,               (nnz)*sizeof(PetscScalar));CHKERRCUDA(err); // kernel output
      err = cudaMemcpy(          h_mat.offdiag.a,    jacb->a,   (nnz)*sizeof(PetscScalar), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      h_mat.offdiag.nonew =jacb->nonew;
      h_mat.offdiag.ignorezeroentries = jacb->ignorezeroentries;
      h_mat.offdiag.nonzerorowcnt = jacb->nonzerorowcnt;
      h_mat.offdiag.nonzerostate = A->nonzerostate; // ???
      h_mat.offdiag.rmax = jacb->rmax;
      h_mat.offdiag.n = n;
      h_mat.offdiag.nz = 0;
    }
    // allocate A copy data
    nnz = jaca->i[n];
    h_mat.diag.n = n;
    h_mat.diag.nz = 0;
    h_mat.diag.ignorezeroentries = jaca->ignorezeroentries;
    h_mat.diag.nonew =jaca->nonew;
    h_mat.diag.nonzerostate = A->nonzerostate; // ???
    h_mat.diag.nonzerorowcnt = jaca->nonzerorowcnt;
    h_mat.diag.rmax = jaca->rmax;
    err = cudaMalloc((void **)&h_mat.diag.i,               (n+1)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
    err = cudaMemcpy(          h_mat.diag.i,    jaca->i,   (n+1)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMalloc((void **)&h_mat.diag.ilen,            (n)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
    err = cudaMemcpy(          h_mat.diag.ilen, jaca->ilen,(n)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMalloc((void **)&h_mat.diag.imax,            (n)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
    err = cudaMemcpy(          h_mat.diag.imax, jaca->imax,(n)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMalloc((void **)&h_mat.diag.j,               (nnz)*sizeof(PetscInt));CHKERRCUDA(err); // kernel input
    err = cudaMemcpy(          h_mat.diag.j,    jaca->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMalloc((void **)&h_mat.diag.a,               (nnz)*sizeof(PetscScalar));CHKERRCUDA(err); // kernel output
    err = cudaMemcpy(          h_mat.diag.a,    jaca->a,   (nnz)*sizeof(PetscScalar), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    err = cudaMemcpy(          d_mat, &h_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    ierr = PetscInfo4(A,"Create device Mat n=%D rmax=%D nonzerorowcnt=%D nnz=%D\n",h_mat.diag.n, h_mat.diag.rmax, h_mat.diag.nonzerorowcnt, nnz);CHKERRQ(ierr);
  } else {
    *B = *p_d_mat;
    if (A->assembled) {
      A->was_assembled = PETSC_TRUE;
      if (size > 1) {
	Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)A->data;
	if (!aij->colmap) { // this is done in matsetvalues, but we need to do it on the host
	  ierr = MatCreateColmap_MPIAIJ_Private(A);CHKERRQ(ierr);
	  ierr = PetscInfo(A,"Setup colmap\n");CHKERRQ(ierr);
	}
      }
    }
  }
  A->assembled = PETSC_FALSE; // ready to write with matsetvalues
  PetscFunctionReturn(0);
}
