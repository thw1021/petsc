/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include <petsc/private/matimpl.h>
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/

/*
   Produces a set of block column indices of the matrix row, one for each block represented in the original row

   n - the number of block indices in cc[]
   cc - the block indices (must be large enough to contain the indices)
*/
static inline PetscErrorCode MatCollapseRow(Mat Amat,PetscInt row,PetscInt bs,PetscInt *n,PetscInt *cc)
{
  PetscInt       cnt = -1,nidx,j;
  const PetscInt *idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(MatGetRow(Amat,row,&nidx,&idx,NULL));
  if (nidx) {
    cnt = 0;
    cc[cnt] = idx[0]/bs;
    for (j=1; j<nidx; j++) {
      if (cc[cnt] < idx[j]/bs) cc[++cnt] = idx[j]/bs;
    }
  }
  CHKERRQ(MatRestoreRow(Amat,row,&nidx,&idx,NULL));
  *n = cnt+1;
  PetscFunctionReturn(0);
}

/*
    Produces a set of block column indices of the matrix block row, one for each block represented in the original set of rows

    ncollapsed - the number of block indices
    collapsed - the block indices (must be large enough to contain the indices)
*/
static inline PetscErrorCode MatCollapseRows(Mat Amat,PetscInt start,PetscInt bs,PetscInt *w0,PetscInt *w1,PetscInt *w2,PetscInt *ncollapsed,PetscInt **collapsed)
{
  PetscInt       i,nprev,*cprev = w0,ncur = 0,*ccur = w1,*merged = w2,*cprevtmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(MatCollapseRow(Amat,start,bs,&nprev,cprev));
  for (i=start+1; i<start+bs; i++) {
    CHKERRQ(MatCollapseRow(Amat,i,bs,&ncur,ccur));
    CHKERRQ(PetscMergeIntArray(nprev,cprev,ncur,ccur,&nprev,&merged));
    cprevtmp = cprev; cprev = merged; merged = cprevtmp;
  }
  *ncollapsed = nprev;
  if (collapsed) *collapsed  = cprev;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCreateGraph - create simple scaled scalar graph from matrix

 Input Parameter:
 . Amat - matrix
 Output Parameter:
 . a_Gmaat - eoutput scalar graph (symmetric?)
 */
PetscErrorCode PCGAMGCreateGraph(Mat Amat, Mat *a_Gmat)
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,jj,kk,ncols,nloc,NN,MM,bs;
  MPI_Comm       comm;
  Mat            Gmat;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)Amat,&comm));
  CHKERRQ(MatGetOwnershipRange(Amat, &Istart, &Iend));
  CHKERRQ(MatGetSize(Amat, &MM, &NN));
  CHKERRQ(MatGetBlockSize(Amat, &bs));
  nloc = (Iend-Istart)/bs;

  CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0));

  /* TODO GPU: these calls are potentially expensive if matrices are large and we want to use the GPU */
  /* A solution consists in providing a new API, MatAIJGetCollapsedAIJ, and each class can provide a fast
     implementation */
  if (bs > 1) {
    const PetscScalar *vals;
    const PetscInt    *idx;
    PetscInt          *d_nnz, *o_nnz,*w0,*w1,*w2;
    PetscBool         ismpiaij,isseqaij;

    /*
       Determine the preallocation needed for the scalar matrix derived from the vector matrix.
    */

    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)Amat,MATSEQAIJ,&isseqaij));
    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)Amat,MATMPIAIJ,&ismpiaij));
    CHKERRQ(PetscMalloc2(nloc, &d_nnz,isseqaij ? 0 : nloc, &o_nnz));

    if (isseqaij) {
      PetscInt max_d_nnz;

      /*
          Determine exact preallocation count for (sequential) scalar matrix
      */
      CHKERRQ(MatSeqAIJGetMaxRowNonzeros(Amat,&max_d_nnz));
      max_d_nnz = PetscMin(nloc,bs*max_d_nnz);
      CHKERRQ(PetscMalloc3(max_d_nnz, &w0,max_d_nnz, &w1,max_d_nnz, &w2));
      for (Ii = 0, jj = 0; Ii < Iend; Ii += bs, jj++) {
        CHKERRQ(MatCollapseRows(Amat,Ii,bs,w0,w1,w2,&d_nnz[jj],NULL));
      }
      CHKERRQ(PetscFree3(w0,w1,w2));

    } else if (ismpiaij) {
      Mat            Daij,Oaij;
      const PetscInt *garray;
      PetscInt       max_d_nnz;

      CHKERRQ(MatMPIAIJGetSeqAIJ(Amat,&Daij,&Oaij,&garray));

      /*
          Determine exact preallocation count for diagonal block portion of scalar matrix
      */
      CHKERRQ(MatSeqAIJGetMaxRowNonzeros(Daij,&max_d_nnz));
      max_d_nnz = PetscMin(nloc,bs*max_d_nnz);
      CHKERRQ(PetscMalloc3(max_d_nnz, &w0,max_d_nnz, &w1,max_d_nnz, &w2));
      for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
        CHKERRQ(MatCollapseRows(Daij,Ii,bs,w0,w1,w2,&d_nnz[jj],NULL));
      }
      CHKERRQ(PetscFree3(w0,w1,w2));

      /*
         Over estimate (usually grossly over), preallocation count for off-diagonal portion of scalar matrix
      */
      for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
        o_nnz[jj] = 0;
        for (kk=0; kk<bs; kk++) { /* rows that get collapsed to a single row */
          CHKERRQ(MatGetRow(Oaij,Ii+kk,&ncols,NULL,NULL));
          o_nnz[jj] += ncols;
          CHKERRQ(MatRestoreRow(Oaij,Ii+kk,&ncols,NULL,NULL));
        }
        if (o_nnz[jj] > (NN/bs-nloc)) o_nnz[jj] = NN/bs-nloc;
      }

    } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Require AIJ matrix type");

    /* get scalar copy (norms) of matrix */
    CHKERRQ(MatCreate(comm, &Gmat));
    CHKERRQ(MatSetSizes(Gmat,nloc,nloc,PETSC_DETERMINE,PETSC_DETERMINE));
    CHKERRQ(MatSetBlockSizes(Gmat, 1, 1));
    CHKERRQ(MatSetType(Gmat, MATAIJ));
    CHKERRQ(MatSeqAIJSetPreallocation(Gmat,0,d_nnz));
    CHKERRQ(MatMPIAIJSetPreallocation(Gmat,0,d_nnz,0,o_nnz));
    CHKERRQ(PetscFree2(d_nnz,o_nnz));

    for (Ii = Istart; Ii < Iend; Ii++) {
      PetscInt dest_row = Ii/bs;
      CHKERRQ(MatGetRow(Amat,Ii,&ncols,&idx,&vals));
      for (jj=0; jj<ncols; jj++) {
        PetscInt    dest_col = idx[jj]/bs;
        PetscScalar sv       = PetscAbs(PetscRealPart(vals[jj]));
        CHKERRQ(MatSetValues(Gmat,1,&dest_row,1,&dest_col,&sv,ADD_VALUES));
      }
      CHKERRQ(MatRestoreRow(Amat,Ii,&ncols,&idx,&vals));
    }
    CHKERRQ(MatAssemblyBegin(Gmat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Gmat,MAT_FINAL_ASSEMBLY));
  } else {
    /* just copy scalar matrix - abs() not taken here but scaled later */
    CHKERRQ(MatDuplicate(Amat, MAT_COPY_VALUES, &Gmat));
  }
  CHKERRQ(MatPropagateSymmetryOptions(Amat, Gmat));

  CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0));

  *a_Gmat = Gmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*@C
   PCGAMGFilterGraph - filter (remove zero and possibly small values from the) graph and make it symmetric if requested

   Collective on Mat

   Input Parameters:
+   a_Gmat - the graph
.   vfilter - threshold parameter [0,1)
-   symm - make the result symmetric

   Level: developer

   Notes:
    This is called before graph coarsers are called.

.seealso: PCGAMGSetThreshold()
@*/
PetscErrorCode PCGAMGFilterGraph(Mat *a_Gmat,PetscReal vfilter,PetscBool symm)
{
  PetscErrorCode    ierr;
  PetscInt          Istart,Iend,Ii,jj,ncols,nnz0,nnz1, NN, MM, nloc;
  PetscMPIInt       rank;
  Mat               Gmat  = *a_Gmat, tGmat;
  MPI_Comm          comm;
  const PetscScalar *vals;
  const PetscInt    *idx;
  PetscInt          *d_nnz, *o_nnz;
  Vec               diag;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0));

  /* TODO GPU: optimization proposal, each class provides fast implementation of this
     procedure via MatAbs API */
  if (vfilter < 0.0 && !symm) {
    /* Just use the provided matrix as the graph but make all values positive */
    MatInfo     info;
    PetscScalar *avals;
    PetscBool isaij,ismpiaij;
    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)Gmat,MATSEQAIJ,&isaij));
    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)Gmat,MATMPIAIJ,&ismpiaij));
    PetscCheckFalse(!isaij && !ismpiaij,PETSC_COMM_WORLD,PETSC_ERR_USER,"Require (MPI)AIJ matrix type");
    if (isaij) {
      CHKERRQ(MatGetInfo(Gmat,MAT_LOCAL,&info));
      CHKERRQ(MatSeqAIJGetArray(Gmat,&avals));
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      CHKERRQ(MatSeqAIJRestoreArray(Gmat,&avals));
    } else {
      Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)Gmat->data;
      CHKERRQ(MatGetInfo(aij->A,MAT_LOCAL,&info));
      CHKERRQ(MatSeqAIJGetArray(aij->A,&avals));
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      CHKERRQ(MatSeqAIJRestoreArray(aij->A,&avals));
      CHKERRQ(MatGetInfo(aij->B,MAT_LOCAL,&info));
      CHKERRQ(MatSeqAIJGetArray(aij->B,&avals));
      for (jj = 0; jj<info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
      CHKERRQ(MatSeqAIJRestoreArray(aij->B,&avals));
    }
    CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0));
    PetscFunctionReturn(0);
  }

  /* TODO GPU: this can be called when filter = 0 -> Probably provide MatAIJThresholdCompress that compresses the entries below a threshold?
               Also, if the matrix is symmetric, can we skip this
               operation? It can be very expensive on large matrices. */
  CHKERRQ(PetscObjectGetComm((PetscObject)Gmat,&comm));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(MatGetOwnershipRange(Gmat, &Istart, &Iend));
  nloc = Iend - Istart;
  CHKERRQ(MatGetSize(Gmat, &MM, &NN));

  if (symm) {
    Mat matTrans;
    CHKERRQ(MatTranspose(Gmat, MAT_INITIAL_MATRIX, &matTrans));
    CHKERRQ(MatAXPY(Gmat, 1.0, matTrans, Gmat->structurally_symmetric ? SAME_NONZERO_PATTERN : DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatDestroy(&matTrans));
  }

  /* scale Gmat for all values between -1 and 1 */
  CHKERRQ(MatCreateVecs(Gmat, &diag, NULL));
  CHKERRQ(MatGetDiagonal(Gmat, diag));
  CHKERRQ(VecReciprocal(diag));
  CHKERRQ(VecSqrtAbs(diag));
  CHKERRQ(MatDiagonalScale(Gmat, diag, diag));
  CHKERRQ(VecDestroy(&diag));

  /* Determine upper bound on nonzeros needed in new filtered matrix */
  CHKERRQ(PetscMalloc2(nloc, &d_nnz,nloc, &o_nnz));
  for (Ii = Istart, jj = 0; Ii < Iend; Ii++, jj++) {
    CHKERRQ(MatGetRow(Gmat,Ii,&ncols,NULL,NULL));
    d_nnz[jj] = ncols;
    o_nnz[jj] = ncols;
    CHKERRQ(MatRestoreRow(Gmat,Ii,&ncols,NULL,NULL));
    if (d_nnz[jj] > nloc) d_nnz[jj] = nloc;
    if (o_nnz[jj] > (MM-nloc)) o_nnz[jj] = MM - nloc;
  }
  CHKERRQ(MatCreate(comm, &tGmat));
  CHKERRQ(MatSetSizes(tGmat,nloc,nloc,MM,MM));
  CHKERRQ(MatSetBlockSizes(tGmat, 1, 1));
  CHKERRQ(MatSetType(tGmat, MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(tGmat,0,d_nnz));
  CHKERRQ(MatMPIAIJSetPreallocation(tGmat,0,d_nnz,0,o_nnz));
  CHKERRQ(MatSetOption(tGmat,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  CHKERRQ(PetscFree2(d_nnz,o_nnz));

  for (Ii = Istart, nnz0 = nnz1 = 0; Ii < Iend; Ii++) {
    CHKERRQ(MatGetRow(Gmat,Ii,&ncols,&idx,&vals));
    for (jj=0; jj<ncols; jj++,nnz0++) {
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      if (PetscRealPart(sv) > vfilter) {
        nnz1++;
        CHKERRQ(MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,INSERT_VALUES));
      }
    }
    CHKERRQ(MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals));
  }
  CHKERRQ(MatAssemblyBegin(tGmat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(tGmat,MAT_FINAL_ASSEMBLY));
  if (symm) {
    CHKERRQ(MatSetOption(tGmat,MAT_SYMMETRIC,PETSC_TRUE));
  } else {
    CHKERRQ(MatPropagateSymmetryOptions(Gmat,tGmat));
  }
  CHKERRQ(PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0));

#if defined(PETSC_USE_INFO)
  {
    double t1 = (!nnz0) ? 1. : 100.*(double)nnz1/(double)nnz0, t2 = (!nloc) ? 1. : (double)nnz0/(double)nloc;
    CHKERRQ(PetscInfo(*a_Gmat,"\t %g%% nnz after filtering, with threshold %g, %g nnz ave. (N=%D)\n",t1,vfilter,t2,MM));
  }
#endif
  CHKERRQ(MatDestroy(&Gmat));
  *a_Gmat = tGmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGGetDataWithGhosts - hacks into Mat MPIAIJ so this must have size > 1

   Input Parameter:
   . Gmat - MPIAIJ matrix for scattters
   . data_sz - number of data terms per node (# cols in output)
   . data_in[nloc*data_sz] - column oriented data
   Output Parameter:
   . a_stride - numbrt of rows of output
   . a_data_out[stride*data_sz] - output data with ghosts
*/
PetscErrorCode PCGAMGGetDataWithGhosts(Mat Gmat,PetscInt data_sz,PetscReal data_in[],PetscInt *a_stride,PetscReal **a_data_out)
{
  PetscErrorCode ierr;
  Vec            tmp_crds;
  Mat_MPIAIJ     *mpimat = (Mat_MPIAIJ*)Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar    *data_arr;
  PetscReal      *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ));
  CHKERRQ(MatGetOwnershipRange(Gmat, &my0, &Iend));
  nloc      = Iend - my0;
  CHKERRQ(VecGetLocalSize(mpimat->lvec, &num_ghosts));
  nnodes    = num_ghosts + nloc;
  *a_stride = nnodes;
  CHKERRQ(MatCreateVecs(Gmat, &tmp_crds, NULL));

  CHKERRQ(PetscMalloc1(data_sz*nnodes, &datas));
  for (dir=0; dir<data_sz; dir++) {
    /* set local, and global */
    for (kk=0; kk<nloc; kk++) {
      PetscInt    gid = my0 + kk;
      PetscScalar crd = (PetscScalar)data_in[dir*nloc + kk]; /* col oriented */
      datas[dir*nnodes + kk] = PetscRealPart(crd);

      CHKERRQ(VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(tmp_crds));
    CHKERRQ(VecAssemblyEnd(tmp_crds));
    /* get ghost datas */
    CHKERRQ(VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecGetArray(mpimat->lvec, &data_arr));
    for (kk=nloc,jj=0;jj<num_ghosts;kk++,jj++) datas[dir*nnodes + kk] = PetscRealPart(data_arr[jj]);
    CHKERRQ(VecRestoreArray(mpimat->lvec, &data_arr));
  }
  CHKERRQ(VecDestroy(&tmp_crds));
  *a_data_out = datas;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableCreate(PetscInt a_size, PCGAMGHashTable *a_tab)
{
  PetscErrorCode ierr;
  PetscInt       kk;

  PetscFunctionBegin;
  a_tab->size = a_size;
  CHKERRQ(PetscMalloc2(a_size, &a_tab->table,a_size, &a_tab->data));
  for (kk=0; kk<a_size; kk++) a_tab->table[kk] = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableDestroy(PCGAMGHashTable *a_tab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscFree2(a_tab->table,a_tab->data));
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGHashTableAdd(PCGAMGHashTable *a_tab, PetscInt a_key, PetscInt a_data)
{
  PetscInt kk,idx;

  PetscFunctionBegin;
  PetscCheckFalse(a_key<0,PETSC_COMM_SELF,PETSC_ERR_USER,"Negative key %D.",a_key);
  for (kk = 0, idx = GAMG_HASH(a_key); kk < a_tab->size; kk++, idx = (idx==(a_tab->size-1)) ? 0 : idx + 1) {
    if (a_tab->table[idx] == a_key) {
      /* exists */
      a_tab->data[idx] = a_data;
      break;
    } else if (a_tab->table[idx] == -1) {
      /* add */
      a_tab->table[idx] = a_key;
      a_tab->data[idx]  = a_data;
      break;
    }
  }
  if (kk==a_tab->size) {
    /* this is not to efficient, waiting until completely full */
    PetscInt       oldsize = a_tab->size, new_size = 2*a_tab->size + 5, *oldtable = a_tab->table, *olddata = a_tab->data;
    PetscErrorCode ierr;

    a_tab->size = new_size;
    CHKERRQ(PetscMalloc2(a_tab->size, &a_tab->table,a_tab->size, &a_tab->data));
    for (kk=0;kk<a_tab->size;kk++) a_tab->table[kk] = -1;
    for (kk=0;kk<oldsize;kk++) {
      if (oldtable[kk] != -1) {
        CHKERRQ(PCGAMGHashTableAdd(a_tab, oldtable[kk], olddata[kk]));
       }
    }
    CHKERRQ(PetscFree2(oldtable,olddata));
    CHKERRQ(PCGAMGHashTableAdd(a_tab, a_key, a_data));
  }
  PetscFunctionReturn(0);
}