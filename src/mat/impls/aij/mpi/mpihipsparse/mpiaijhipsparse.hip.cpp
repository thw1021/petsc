#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_HIPWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpihipsparse/mpihipsparsematimpl.h>
#include <thrust/advance.h>
#include <petscsf.h>

struct VecHIPEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

static PetscErrorCode MatSetValuesCOO_MPIAIJHIPSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ         *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE *hipsp = (Mat_MPIAIJHIPSPARSE*)a->spptr;
  PetscInt           n = hipsp->coo_nd + hipsp->coo_no;
  PetscErrorCode     ierr;
  hipError_t        cerr;

  PetscFunctionBegin;
  if (hipsp->coo_p && v) {
    thrust::device_ptr<const PetscScalar> d_v;
    THRUSTARRAY                           *w = NULL;

    if (isHipMem(v)) {
      d_v = thrust::device_pointer_cast(v);
    } else {
      w = new THRUSTARRAY(n);
      w->assign(v,v+n);
      ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr);
      d_v = w->data();
    }

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,hipsp->coo_p->begin()),
                                                              hipsp->coo_pw->begin()));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,hipsp->coo_p->end()),
                                                              hipsp->coo_pw->end()));
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    thrust::for_each(zibit,zieit,VecHIPEquals());
    cerr = WaitForHIP();CHKERRHIP(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    delete w;
    ierr = MatSetValuesCOO_SeqAIJHIPSPARSE(a->A,hipsp->coo_pw->data().get(),imode);CHKERRQ(ierr);
    ierr = MatSetValuesCOO_SeqAIJHIPSPARSE(a->B,hipsp->coo_pw->data().get()+hipsp->coo_nd,imode);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesCOO_SeqAIJHIPSPARSE(a->A,v,imode);CHKERRQ(ierr);
    ierr = MatSetValuesCOO_SeqAIJHIPSPARSE(a->B,v ? v+hipsp->coo_nd : NULL,imode);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  A->num_ass++;
  A->assembled        = PETSC_TRUE;
  A->ass_nonzerostate = A->nonzerostate;
  A->offloadmask      = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

template <typename Tuple>
struct IsNotOffDiagT
{
  PetscInt _cstart,_cend;

  IsNotOffDiagT(PetscInt cstart, PetscInt cend) : _cstart(cstart), _cend(cend) {}
  __host__ __device__
  inline bool operator()(Tuple t)
  {
    return !(thrust::get<1>(t) < _cstart || thrust::get<1>(t) >= _cend);
  }
};

struct IsOffDiag
{
  PetscInt _cstart,_cend;

  IsOffDiag(PetscInt cstart, PetscInt cend) : _cstart(cstart), _cend(cend) {}
  __host__ __device__
  inline bool operator() (const PetscInt &c)
  {
    return c < _cstart || c >= _cend;
  }
};

struct GlobToLoc
{
  PetscInt _start;

  GlobToLoc(PetscInt start) : _start(start) {}
  __host__ __device__
  inline PetscInt operator() (const PetscInt &c)
  {
    return c - _start;
  }
};

static PetscErrorCode MatSetPreallocationCOO_MPIAIJHIPSPARSE(Mat B, PetscInt n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_MPIAIJ             *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJHIPSPARSE     *hipsp = (Mat_MPIAIJHIPSPARSE*)b->spptr;
  PetscErrorCode         ierr;
  PetscInt               *jj;
  size_t                 noff = 0;
  THRUSTINTARRAY         d_i(n);
  THRUSTINTARRAY         d_j(n);
  ISLocalToGlobalMapping l2g;
  hipError_t            cerr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (b->A) { ierr = MatHIPSPARSEClearHandle(b->A);CHKERRQ(ierr); }
  if (b->B) { ierr = MatHIPSPARSEClearHandle(b->B);CHKERRQ(ierr); }
  ierr = PetscFree(b->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&b->lvec);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);

  ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
  d_i.assign(coo_i,coo_i+n);
  d_j.assign(coo_j,coo_j+n);
  delete hipsp->coo_p;
  delete hipsp->coo_pw;
  hipsp->coo_p = NULL;
  hipsp->coo_pw = NULL;
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  auto firstoffd = thrust::find_if(thrust::device,d_j.begin(),d_j.end(),IsOffDiag(B->cmap->rstart,B->cmap->rend));
  auto firstdiag = thrust::find_if_not(thrust::device,firstoffd,d_j.end(),IsOffDiag(B->cmap->rstart,B->cmap->rend));
  if (firstoffd != d_j.end() && firstdiag != d_j.end()) {
    hipsp->coo_p = new THRUSTINTARRAY(n);
    hipsp->coo_pw = new THRUSTARRAY(n);
    thrust::sequence(thrust::device,hipsp->coo_p->begin(),hipsp->coo_p->end(),0);
    auto fzipp = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin(),hipsp->coo_p->begin()));
    auto ezipp = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end(),hipsp->coo_p->end()));
    auto mzipp = thrust::partition(thrust::device,fzipp,ezipp,IsNotOffDiagT<thrust::tuple<PetscInt,PetscInt,PetscInt> >(B->cmap->rstart,B->cmap->rend));
    firstoffd = mzipp.get_iterator_tuple().get<1>();
  }
  hipsp->coo_nd = thrust::distance(d_j.begin(),firstoffd);
  hipsp->coo_no = thrust::distance(firstoffd,d_j.end());

  /* from global to local */
  thrust::transform(thrust::device,d_i.begin(),d_i.end(),d_i.begin(),GlobToLoc(B->rmap->rstart));
  thrust::transform(thrust::device,d_j.begin(),firstoffd,d_j.begin(),GlobToLoc(B->cmap->rstart));
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

  /* copy offdiag column indices to map on the CPU */
  ierr = PetscMalloc1(hipsp->coo_no,&jj);CHKERRQ(ierr);
  cerr = hipMemcpy(jj,d_j.data().get()+hipsp->coo_nd,hipsp->coo_no*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
  auto o_j = d_j.begin();
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  thrust::advance(o_j,hipsp->coo_nd);
  thrust::sort(thrust::device,o_j,d_j.end());
  auto wit = thrust::unique(thrust::device,o_j,d_j.end());
  cerr = WaitForHIP();CHKERRHIP(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  noff = thrust::distance(o_j,wit);
  ierr = PetscMalloc1(noff+1,&b->garray);CHKERRQ(ierr);
  cerr = hipMemcpy(b->garray,d_j.data().get()+hipsp->coo_nd,noff*sizeof(PetscInt),hipMemcpyDeviceToHost);CHKERRHIP(cerr);
  ierr = PetscLogGpuToCpu((noff+hipsp->coo_no)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,noff,b->garray,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(l2g,ISLOCALTOGLOBALMAPPINGHASH);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,hipsp->coo_no,jj,&n,jj);CHKERRQ(ierr);
  if (n != hipsp->coo_no) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected is size %D != %D coo size",n,hipsp->coo_no);
  ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
  ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(b->A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
  ierr = MatSetSizes(b->B,B->rmap->n,noff,B->rmap->n,noff);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);

  /* GPU memory, hipsparse specific call handles it internally */
  ierr = MatSetPreallocationCOO_SeqAIJHIPSPARSE(b->A,hipsp->coo_nd,d_i.data().get(),d_j.data().get());CHKERRQ(ierr);
  ierr = MatSetPreallocationCOO_SeqAIJHIPSPARSE(b->B,hipsp->coo_no,d_i.data().get()+hipsp->coo_nd,jj);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);

  ierr = MatHIPSPARSESetFormat(b->A,MAT_HIPSPARSE_MULT,hipsp->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetFormat(b->B,MAT_HIPSPARSE_MULT,hipsp->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetHandle(b->A,hipsp->handle);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetHandle(b->B,hipsp->handle);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetStream(b->A,hipsp->stream);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetStream(b->B,hipsp->stream);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_MPIAIJ(B);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  B->nonzerostate++;

  ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  B->assembled = PETSC_FALSE;
  B->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJHIPSPARSE(Mat A,MatReuse scall,IS *glob,Mat *A_loc)
{
  Mat            Ad,Ao;
  const PetscInt *cmap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&cmap);CHKERRQ(ierr);
  ierr = MatSeqAIJHIPSPARSEMergeMats(Ad,Ao,scall,A_loc);CHKERRQ(ierr);
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;

    ierr = MatGetLocalSize(Ad,NULL,&dn);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Ao,NULL,&on);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(A,&cst,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(dn+on,&gidx);CHKERRQ(ierr);
    for (i=0; i<dn; i++) gidx[i]    = cst + i;
    for (i=0; i<on; i++) gidx[i+dn] = cmap[i];
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)Ad),dn+on,gidx,PETSC_OWN_POINTER,glob);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJHIPSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJHIPSPARSE *hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)b->spptr;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG) && d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (PetscDefined(USE_DEBUG) && o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&b->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(b->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(b->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&b->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->Mvctx);CHKERRQ(ierr);
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);
  if (!b->A) {
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
  }
  if (!b->B) {
    PetscMPIInt size;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,size > 1 ? B->cmap->N : 0,B->rmap->n,size > 1 ? B->cmap->N : 0);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }
  ierr = MatSetType(b->A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetFormat(b->A,MAT_HIPSPARSE_MULT,hipsparseStruct->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetFormat(b->B,MAT_HIPSPARSE_MULT,hipsparseStruct->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetHandle(b->A,hipsparseStruct->handle);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetHandle(b->B,hipsparseStruct->handle);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetStream(b->A,hipsparseStruct->stream);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetStream(b->B,hipsparseStruct->stream);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   MatAIJHIPSPARSESetGenerateTranspose - Sets the flag to explicitly generate the transpose matrix before calling MatMultTranspose

   Not collective

   Input Parameters:
+  A - Matrix of type SEQAIJHIPSPARSE or MPIAIJHIPSPARSE
-  gen - the boolean flag

   Level: intermediate

.seealso: MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE
@*/
PetscErrorCode  MatAIJHIPSPARSESetGenerateTranspose(Mat A, PetscBool gen)
{
  PetscErrorCode ierr;
  PetscBool      ismpiaij;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  MatCheckPreallocated(A,1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (ismpiaij) {
    Mat A_d,A_o;

    ierr = MatMPIAIJGetSeqAIJ(A,&A_d,&A_o,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSESetGenerateTranspose(A_d,gen);CHKERRQ(ierr);
    ierr = MatSeqAIJHIPSPARSESetGenerateTranspose(A_o,gen);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJHIPSPARSESetGenerateTranspose(A,gen);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIAIJHIPSPARSE(Mat A)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_MPIAIJHIPSPARSE *spptr = (Mat_MPIAIJHIPSPARSE*)l->spptr;
    PetscSplitCSRDataStructure *d_mat = spptr->deviceMat;
    if (d_mat) {
      Mat_SeqAIJ   *a = (Mat_SeqAIJ*)l->A->data;
      Mat_SeqAIJ   *b = (Mat_SeqAIJ*)l->B->data;
      PetscInt     n = A->rmap->n, nnza = a->i[n], nnzb = b->i[n];
      hipError_t  err;
      PetscScalar  *vals;
      ierr = PetscInfo(A,"Zero device matrix diag and offfdiag\n");CHKERRQ(ierr);
      err = hipMemcpy( &vals, &d_mat->diag.a, sizeof(PetscScalar*), hipMemcpyDeviceToHost);CHKERRHIP(err);
      err = hipMemset( vals, 0, (nnza)*sizeof(PetscScalar));CHKERRHIP(err);
      err = hipMemcpy( &vals, &d_mat->offdiag.a, sizeof(PetscScalar*), hipMemcpyDeviceToHost);CHKERRHIP(err);
      err = hipMemset( vals, 0, (nnzb)*sizeof(PetscScalar));CHKERRHIP(err);
    }
  }
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
PetscErrorCode MatMultAdd_MPIAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->rmap->n,nt);
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHIPSPARSESetFormat_MPIAIJHIPSPARSE(Mat A,MatHIPSPARSEFormatOperation op,MatHIPSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a               = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE * hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_HIPSPARSE_MULT_DIAG:
    hipsparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_HIPSPARSE_MULT_OFFDIAG:
    hipsparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_HIPSPARSE_ALL:
    hipsparseStruct->diagGPUMatFormat    = format;
    hipsparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatHIPSPARSEFormatOperation. Only MAT_HIPSPARSE_MULT_DIAG, MAT_HIPSPARSE_MULT_DIAG, and MAT_HIPSPARSE_MULT_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPIAIJHIPSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  MatHIPSPARSEStorageFormat format;
  PetscErrorCode           ierr;
  PetscBool                flg;
  Mat_MPIAIJ               *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE       *hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)a->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPIAIJHIPSPARSE options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_hipsparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_MULT_DIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_hipsparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_MULT_OFFDIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_hipsparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_ALL,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJHIPSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode             ierr;
  Mat_MPIAIJ                 *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE         *hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)mpiaij->spptr;
  PetscSplitCSRDataStructure *d_mat = hipsparseStruct->deviceMat;
  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQHIP);CHKERRQ(ierr);
  }
  if (d_mat) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // if we assembled on the device
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJHIPSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE *hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)aij->spptr;
  hipError_t        err;
  rocsparse_status   stat;

  PetscFunctionBegin;
  if (!hipsparseStruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (hipsparseStruct->deviceMat) {
    Mat_SeqAIJ                 *jaca = (Mat_SeqAIJ*)aij->A->data;
    Mat_SeqAIJ                 *jacb = (Mat_SeqAIJ*)aij->B->data;
    PetscSplitCSRDataStructure *d_mat = hipsparseStruct->deviceMat, h_mat;
    ierr = PetscInfo(A,"Have device matrix\n");CHKERRQ(ierr);
    err = hipMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), hipMemcpyDeviceToHost);CHKERRHIP(err);
    if (jaca->compressedrow.use) {
      err = hipFree(h_mat.diag.i);CHKERRHIP(err);
    }
    if (jacb->compressedrow.use) {
      err = hipFree(h_mat.offdiag.i);CHKERRHIP(err);
    }
    err = hipFree(h_mat.colmap);CHKERRHIP(err);
    err = hipFree(d_mat);CHKERRHIP(err);
  }
  try {
    if (aij->A) { ierr = MatHIPSPARSEClearHandle(aij->A);CHKERRQ(ierr); }
    if (aij->B) { ierr = MatHIPSPARSEClearHandle(aij->B);CHKERRQ(ierr); }
    stat = hipsparseDestroy(hipsparseStruct->handle);CHKERRHIPSPARSE(stat);
    if (hipsparseStruct->stream) {
      err = hipStreamDestroy(hipsparseStruct->stream);CHKERRHIP(err);
    }
    delete hipsparseStruct->coo_p;
    delete hipsparseStruct->coo_pw;
    delete hipsparseStruct;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJHIPSPARSE error: %s", ex);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHIPSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJHIPSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a;
  Mat_MPIAIJHIPSPARSE *hipsparseStruct;
  rocsparse_status   stat;
  Mat                A;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(B,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  A = *newmat;
  A->boundtocpu = PETSC_FALSE;
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECHIP,&A->defaultvectype);CHKERRQ(ierr);

  a = (Mat_MPIAIJ*)A->data;
  if (a->A) { ierr = MatSetType(a->A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr); }
  if (a->B) { ierr = MatSetType(a->B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr); }
  if (a->lvec) {
    ierr = VecSetType(a->lvec,VECSEQHIP);CHKERRQ(ierr);
  }

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    a->spptr = new Mat_MPIAIJHIPSPARSE;

    hipsparseStruct                      = (Mat_MPIAIJHIPSPARSE*)a->spptr;
    hipsparseStruct->diagGPUMatFormat    = MAT_HIPSPARSE_CSR;
    hipsparseStruct->offdiagGPUMatFormat = MAT_HIPSPARSE_CSR;
    hipsparseStruct->coo_p               = NULL;
    hipsparseStruct->coo_pw              = NULL;
    hipsparseStruct->stream              = 0;
    stat = rocsparse_create(&(hipsparseStruct->handle));CHKERRHIPSPARSE(stat);
    hipsparseStruct->deviceMat = NULL;
  }

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJHIPSPARSE;
  A->ops->mult                  = MatMult_MPIAIJHIPSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJHIPSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJHIPSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJHIPSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJHIPSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJHIPSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",MatMPIAIJGetLocalMatMerge_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHIPSPARSESetFormat_C",MatHIPSPARSESetFormat_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJHIPSPARSE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHIPInitializeCheck();CHKERRQ(ierr);
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJHIPSPARSE(A,MATMPIAIJHIPSPARSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateAIJHIPSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
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

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJHIPSPARSE, MATAIJHIPSPARSE
@*/
PetscErrorCode  MatCreateAIJHIPSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJHIPSPARSE - MATMPIAIJHIPSPARSE = "aijhipsparse" = "mpiaijhipsparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require HIP 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the HIPSPARSE library.

   This matrix type is identical to MATSEQAIJHIPSPARSE when constructed with a single process communicator,
   and MATMPIAIJHIPSPARSE otherwise.  As a result, for single process communicators,
   MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijhipsparse - sets the matrix type to "mpiaijhipsparse" during a call to MatSetFromOptions()
.  -mat_hipsparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
.  -mat_hipsparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

 .seealso: MatCreateAIJHIPSPARSE(), MATSEQAIJHIPSPARSE, MatCreateSeqAIJHIPSPARSE(), MatHIPSPARSESetFormat(), MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
M
M*/

// get GPU pointer to stripped down Mat. For both Seq and MPI Mat.
PetscErrorCode MatHIPSPARSEGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure **B)
{
#if defined(PETSC_USE_CTABLE)
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device metadata does not support ctable (--with-ctable=0)");
#else
  PetscSplitCSRDataStructure **p_d_mat;
  PetscMPIInt                size,rank;
  MPI_Comm                   comm;
  PetscErrorCode             ierr;
  int                        *ai,*bi,*aj,*bj;
  PetscScalar                *aa,*ba;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    CsrMatrix *matrixA,*matrixB=NULL;
    if (size == 1) {
      Mat_SeqAIJHIPSPARSE *hipsparsestruct = (Mat_SeqAIJHIPSPARSE*)A->spptr;
      p_d_mat = &hipsparsestruct->deviceMat;
      Mat_SeqAIJHIPSPARSEMultStruct *matstruct = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestruct->mat;
      if (hipsparsestruct->format==MAT_HIPSPARSE_CSR) {
        matrixA = (CsrMatrix*)matstruct->mat;
        bi = bj = NULL; ba = NULL;
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat needs MAT_HIPSPARSE_CSR");
    } else {
      Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)A->data;
      Mat_MPIAIJHIPSPARSE *spptr = (Mat_MPIAIJHIPSPARSE*)aij->spptr;
      p_d_mat = &spptr->deviceMat;
      Mat_SeqAIJHIPSPARSE *hipsparsestructA = (Mat_SeqAIJHIPSPARSE*)aij->A->spptr;
      Mat_SeqAIJHIPSPARSE *hipsparsestructB = (Mat_SeqAIJHIPSPARSE*)aij->B->spptr;
      Mat_SeqAIJHIPSPARSEMultStruct *matstructA = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestructA->mat;
      Mat_SeqAIJHIPSPARSEMultStruct *matstructB = (Mat_SeqAIJHIPSPARSEMultStruct*)hipsparsestructB->mat;
      if (hipsparsestructA->format==MAT_HIPSPARSE_CSR) {
        if (hipsparsestructB->format!=MAT_HIPSPARSE_CSR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat B needs MAT_HIPSPARSE_CSR");
        matrixA = (CsrMatrix*)matstructA->mat;
        matrixB = (CsrMatrix*)matstructB->mat;
        bi = thrust::raw_pointer_cast(matrixB->row_offsets->data());
        bj = thrust::raw_pointer_cast(matrixB->column_indices->data());
        ba = thrust::raw_pointer_cast(matrixB->values->data());
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat A needs MAT_HIPSPARSE_CSR");
    }
    ai = thrust::raw_pointer_cast(matrixA->row_offsets->data());
    aj = thrust::raw_pointer_cast(matrixA->column_indices->data());
    aa = thrust::raw_pointer_cast(matrixA->values->data());
  } else {
    *B = NULL;
    PetscFunctionReturn(0);
  }
  // act like MatSetValues because not called on host
  if (A->assembled) {
    if (A->was_assembled) {
      ierr = PetscInfo(A,"Assemble more than once already\n");CHKERRQ(ierr);
    }
    A->was_assembled = PETSC_TRUE; // this is done (lazy) in MatAssemble but we are not calling it anymore - done in AIJ AssemblyEnd, need here?
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Need assemble matrix");
  }
  if (!*p_d_mat) {
    hipError_t                 err;
    PetscSplitCSRDataStructure  *d_mat, h_mat;
    Mat_SeqAIJ                  *jaca;
    PetscInt                    n = A->rmap->n, nnz;
    // create and copy
    ierr = PetscInfo(A,"Create device matrix\n");CHKERRQ(ierr);
    err = hipMalloc((void **)&d_mat, sizeof(PetscSplitCSRDataStructure));CHKERRHIP(err);
    err = hipMemset( d_mat, 0,       sizeof(PetscSplitCSRDataStructure));CHKERRHIP(err);
    *B = *p_d_mat = d_mat; // return it, set it in Mat, and set it up
    if (size == 1) {
      jaca = (Mat_SeqAIJ*)A->data;
      h_mat.rstart = 0; h_mat.rend = A->rmap->n;
      h_mat.cstart = 0; h_mat.cend = A->cmap->n;
      h_mat.offdiag.i = h_mat.offdiag.j = NULL;
      h_mat.offdiag.a = NULL;
    } else {
      Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)A->data;
      Mat_SeqAIJ  *jacb;
      jaca = (Mat_SeqAIJ*)aij->A->data;
      jacb = (Mat_SeqAIJ*)aij->B->data;
      if (!aij->garray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");
      if (aij->B->rmap->n != aij->A->rmap->n) SETERRQ(comm,PETSC_ERR_SUP,"Only support aij->B->rmap->n == aij->A->rmap->n");
      // create colmap - this is ussually done (lazy) in MatSetValues
      aij->donotstash = PETSC_TRUE;
      aij->A->nooffprocentries = aij->B->nooffprocentries = A->nooffprocentries = PETSC_TRUE;
      jaca->nonew = jacb->nonew = PETSC_TRUE; // no more dissassembly
      ierr = PetscCalloc1(A->cmap->N+1,&aij->colmap);CHKERRQ(ierr);
      aij->colmap[A->cmap->N] = -9;
      ierr = PetscLogObjectMemory((PetscObject)A,(A->cmap->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
      {
        PetscInt ii;
        for (ii=0; ii<aij->B->cmap->n; ii++) aij->colmap[aij->garray[ii]] = ii+1;
      }
      if (aij->colmap[A->cmap->N] != -9) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"aij->colmap[A->cmap->N] != -9");
      // allocate B copy data
      h_mat.rstart = A->rmap->rstart; h_mat.rend = A->rmap->rend;
      h_mat.cstart = A->cmap->rstart; h_mat.cend = A->cmap->rend;
      nnz = jacb->i[n];

      if (jacb->compressedrow.use) {
        err = hipMalloc((void **)&h_mat.offdiag.i,               (n+1)*sizeof(int));CHKERRHIP(err); // kernel input
        err = hipMemcpy(          h_mat.offdiag.i,    jacb->i,   (n+1)*sizeof(int), hipMemcpyHostToDevice);CHKERRHIP(err);
      } else h_mat.offdiag.i = bi;
      h_mat.offdiag.j = bj;
      h_mat.offdiag.a = ba;

      err = hipMalloc((void **)&h_mat.colmap,                  (A->cmap->N+1)*sizeof(PetscInt));CHKERRHIP(err); // kernel output
      err = hipMemcpy(          h_mat.colmap,    aij->colmap,  (A->cmap->N+1)*sizeof(PetscInt), hipMemcpyHostToDevice);CHKERRHIP(err);
      h_mat.offdiag.ignorezeroentries = jacb->ignorezeroentries;
      h_mat.offdiag.n = n;
    }
    // allocate A copy data
    nnz = jaca->i[n];
    h_mat.diag.n = n;
    h_mat.diag.ignorezeroentries = jaca->ignorezeroentries;
    ierr = MPI_Comm_rank(comm,&h_mat.rank);CHKERRMPI(ierr);
    if (jaca->compressedrow.use) {
      err = hipMalloc((void **)&h_mat.diag.i,               (n+1)*sizeof(int));CHKERRHIP(err); // kernel input
      err = hipMemcpy(          h_mat.diag.i,    jaca->i,   (n+1)*sizeof(int), hipMemcpyHostToDevice);CHKERRHIP(err);
    } else {
      h_mat.diag.i = ai;
    }
    h_mat.diag.j = aj;
    h_mat.diag.a = aa;
    // copy pointers and metdata to device
    err = hipMemcpy(          d_mat, &h_mat, sizeof(PetscSplitCSRDataStructure), hipMemcpyHostToDevice);CHKERRHIP(err);
    ierr = PetscInfo2(A,"Create device Mat n=%D nnz=%D\n",h_mat.diag.n, nnz);CHKERRQ(ierr);
  } else {
    *B = *p_d_mat;
  }
  A->assembled = PETSC_FALSE; // ready to write with matsetvalues - this done (lazy) in normal MatSetValues
  PetscFunctionReturn(0);
#endif
}
