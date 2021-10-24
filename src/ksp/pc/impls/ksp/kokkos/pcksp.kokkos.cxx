#include <petscvec_kokkos.hpp>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscksp.h>            /*I "petscksp.h" I*/
#include <petscdmcomposite.h>
#include <Kokkos_Core.hpp>

typedef Kokkos::TeamPolicy<>::member_type team_member;

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkokkosimpl.hpp>

#define KOKKOS_SHARED_LEVEL 1
#define KOKKOS_VEC_SIZE 16

typedef enum {BICG_IDX,NUM_KSP_IDX} KSPIndex;
typedef struct {
  Vec                                              vec_diag;
  PetscInt                                         nBlocks; /* total number of blocks */
  PetscInt                                         maxBlkSize; /* largest grid */
  KSP                                              ksp; // Used just for options. Should have one for each block
  Kokkos::View<PetscInt*, Kokkos::LayoutRight>    *d_block_offsets;
  Kokkos::View<PetscScalar*, Kokkos::LayoutRight> *d_idiag;
  KSPIndex                                         ksp_idx;
  PetscInt                                         nwork;
} PC_KSPKOKKOS;

static PetscErrorCode  PCKSPKOKKOSCreateKSP_KSPKOKKOS(PC pc)
{
  PetscErrorCode ierr;
  const char     *prefix;
  PC_KSPKOKKOS   *jac = (PC_KSPKOKKOS*)pc->data;
  DM             dm;

  PetscFunctionBegin;
  ierr = KSPCreate(PetscObjectComm((PetscObject)pc),&jac->ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(jac->ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)jac->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(jac->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(jac->ksp,"ksp_");CHKERRQ(ierr);
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  if (dm) {
    ierr = KSPSetDM(jac->ksp, dm);CHKERRQ(ierr);
    ierr = KSPSetDMActive(jac->ksp, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// y <-- Ax
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMult(const team_member team,  const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt start, const PetscInt end, const PetscScalar *x, PetscScalar *y)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start,end), [=] (const int glb_row) {
      using Kokkos::parallel_reduce;
      int n = glb_Aai[glb_row+1] - glb_Aai[glb_row];
      const PetscInt    *aj  = glb_Aaj + glb_Aai[glb_row];
      const PetscScalar *aa  = glb_Aaa + glb_Aai[glb_row];
      PetscScalar sum = 0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team, n), [=] (const int i, PetscScalar& lsum) {
          lsum += aa[i] * x[aj[i]-start];
        }, sum);
      y[glb_row-start] = sum;
    });
  return 0;
}

// temp buffer per thread with reduction at end?
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMultTranspose(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt start, const PetscInt end, const PetscScalar *x, PetscScalar *y)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,end-start), [=] (int i) {y[i] = 0;});
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start,end), [=] (const int glb_row) {
      using Kokkos::parallel_reduce;
      int n = glb_Aai[glb_row+1] - glb_Aai[glb_row];
      const PetscInt    *aj  = glb_Aaj + glb_Aai[glb_row];
      const PetscScalar *aa  = glb_Aaa + glb_Aai[glb_row];
      const PetscScalar xx = x[glb_row-start];
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,n), [=] (const int &i) {
          PetscScalar val = aa[i] * xx;
          Kokkos::atomic_fetch_add(&y[aj[i]-start], val);
        });
    });
  return 0;
}

typedef struct PCKSP_MetaData_TAG
{
  PetscInt           its;
  KSPConvergedReason reason;
}PCKSP_MetaData;

// Solve Ax = y with biCG stabilized
KOKKOS_INLINE_FUNCTION PetscErrorCode PCKSPSolve_BICG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, PetscScalar *work_space, PetscReal rtol, PetscInt maxit, PCKSP_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x)
{
  PetscInt          Nblk = end-start, i;
  PetscReal         dp;
  const PetscScalar *B = &glb_b[start], *idiag = &glb_idiag[start];
  PetscScalar       *X = &glb_x[start], *ptr = work_space, dpi, a=1.0, beta, betaold=1.0, b,ma, r0;
  PetscScalar       *Rl = ptr; ptr += Nblk;
  PetscScalar       *Zl = ptr; ptr += Nblk;
  PetscScalar       *Pl = ptr; ptr += Nblk;
  PetscScalar       *Rr = ptr; ptr += Nblk;
  PetscScalar       *Zr = ptr; ptr += Nblk;
  PetscScalar       *Pr = ptr; ptr += Nblk;

  //ierr = VecCopy(B,Rr);CHKERRQ(ierr);           /*     r <- b (x is 0) */
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Rr[idx] = B[idx];});
  //ierr = VecCopy(Rr,Rl);CHKERRQ(ierr);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Rl[idx] = Rr[idx];});
  //ierr = KSP_PCApply(ksp,Rr,Zr);CHKERRQ(ierr);     /*     z <- Br         */
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Zr[idx] = idiag[idx]*Rr[idx];});
  //ierr = KSP_PCApplyHermitianTranspose(ksp,Rl,Zl);CHKERRQ(ierr);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Zl[idx] = idiag[idx]*Rl[idx];});
  //ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
  dp = 0;
  Kokkos::parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += PetscSqr(Rr[idx]);}, dp);
  r0 = dp = PetscSqrtReal(dp);
#if defined(PETSC_USE_DEBUG)
  printf("%7d KSP Residual norm %14.12e \n",0,(double)dp);
#endif
  if (dp < 1e-50) {metad->reason = KSP_CONVERGED_ATOL_NORMAL; return 0;}
  if (0 == maxit) {metad->reason = KSP_DIVERGED_ITS; return 0;}
  i = 0;
  do {
    //ierr = VecDot(Zr,Rl,&beta);CHKERRQ(ierr);       /*     beta <- r'z     */
    beta = 0;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += Zr[idx]*Rl[idx];}, beta);

    if (!i) {
      if (beta == 0.0) {
        metad->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        goto done;
      }
      //ierr = VecCopy(Zr,Pr);CHKERRQ(ierr);       /*     p <- z          */
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Pr[idx] = Zr[idx];});
      //ierr = VecCopy(Zl,Pl);CHKERRQ(ierr);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Pl[idx] = Zl[idx];});
    } else {
      b    = beta/betaold;
      //ierr = VecAYPX(Pr,b,Zr);CHKERRQ(ierr);  /*     p <- z + b* p   */
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Pr[idx] = b*Pr[idx] + Zr[idx];});
      b    = PetscConj(b);
      //ierr = VecAYPX(Pl,b,Zl);CHKERRQ(ierr);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Pl[idx] = b*Pl[idx] + Zl[idx];});
    }
    betaold = beta;
    //ierr    = KSP_MatMult(ksp,Amat,Pr,Zr);CHKERRQ(ierr); /*     z <- Kp         */
    MatMult         (team,glb_Aai,glb_Aaj,glb_Aaa,start,end,Pr,Zr);
    team.team_barrier();
    //ierr    = KSP_MatMultHermitianTranspose(ksp,Amat,Pl,Zl);CHKERRQ(ierr);
    MatMultTranspose(team,glb_Aai,glb_Aaj,glb_Aaa,start,end,Pl,Zl);
    team.team_barrier();
    //ierr    = VecDot(Zr,Pl,&dpi);CHKERRQ(ierr);            /*     dpi <- z'p      */
    dpi = 0;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += Zr[idx]*Pl[idx];}, dpi);
    //
    a       = beta/dpi;                           /*     a = beta/p'z    */
    //ierr    = VecAXPY(X,a,Pr);CHKERRQ(ierr);    /*     x <- x + ap     */
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {X[idx] = X[idx] + a*Pr[idx];});
    ma      = -a;
    //ierr    = VecAXPY(Rr,ma,Zr);CHKERRQ(ierr);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Rr[idx] = Rr[idx] + ma*Zr[idx];});
    ma      = PetscConj(ma);
    //ierr    = VecAXPY(Rl,ma,Zl);CHKERRQ(ierr);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Rl[idx] = Rl[idx] + ma*Zl[idx];});
    //ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
    dp = 0;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange (team, Nblk), [=] (const int idx, PetscScalar& lsum) {lsum += PetscSqr(Rr[idx]);}, dp);
    dp = PetscSqrtReal(dp);
#if defined(PETSC_USE_DEBUG)
    printf("%7d KSP Residual norm %14.12e \n",i+1,(double)dp);
#endif
    if (dp < 1e-50) {metad->reason = KSP_CONVERGED_ATOL_NORMAL; goto done;}
    if (dp/r0 < rtol) {metad->reason = KSP_CONVERGED_RTOL_NORMAL; goto done;}
    if (dp/r0 > 1.e5) {metad->reason = KSP_DIVERGED_DTOL; goto done;}
    if (i+1 == maxit) {metad->reason = KSP_DIVERGED_ITS; goto done;}
    //ierr = KSP_PCApply(ksp,Rr,Zr);CHKERRQ(ierr);  /* z <- Br  */
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Zr[idx] = idiag[idx]*Rr[idx];});
    //ierr = KSP_PCApplyHermitianTranspose(ksp,Rl,Zl);CHKERRQ(ierr);
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team,Nblk), [=] (int idx) {Zl[idx] = idiag[idx]*Rl[idx];});
    i++;
  } while (i<maxit);
 done:
  metad->its = i;
  return 0;
}

// KSP solver solve Ax = b
static PetscErrorCode PCApply_KSPKOKKOS(PC pc,Vec b,Vec x)
{
  PetscErrorCode    ierr;
  PC_KSPKOKKOS      *jac = (PC_KSPKOKKOS*)pc->data;
  Mat               A = pc->pmat;
  Mat_SeqAIJKokkos  *aijkok;
  PetscMemType      mtype;

  PetscFunctionBegin;
  if (!jac->vec_diag || !A) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not setup???? %p %p",jac->vec_diag,A);
  if (!(aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr))) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"No aijkok");
  else {
    using scr_mem_t  = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using vect2D_scr_t = Kokkos::View<PetscScalar**, Kokkos::LayoutLeft, scr_mem_t>;
    PetscInt          N, *d_block_offsets, maxit = jac->ksp->max_it, errsum;
    const PetscInt    conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp==0) ? 16 : 1;
    const PetscInt    nwork = jac->nwork,blkSz = jac->maxBlkSize, nBlk = jac->nBlocks;
    PetscScalar       *glb_xdata=NULL;
    PetscReal         rtol = jac->ksp->rtol;
    const PetscScalar *glb_idiag =jac->d_idiag->data(), *glb_bdata=NULL;
    const PetscInt    *glb_Aai = aijkok->i_d.data(), *glb_Aaj = aijkok->j_d.data();
    const PetscScalar *glb_Aaa = aijkok->a_d.data();
    Kokkos::View<PCKSP_MetaData*, Kokkos::DefaultExecutionSpace> d_metadata("solver meta data", nBlk);
    PCFailedReason    pcreason;

    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    ierr = VecGetArrayAndMemType(x,&glb_xdata,&mtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    if (mtype!=PETSC_MEMTYPE_DEVICE) SETERRQ2(PetscObjectComm((PetscObject) pc),PETSC_ERR_ARG_WRONG,"No GPU data for x %D != %D",mtype,PETSC_MEMTYPE_DEVICE);
#endif
    ierr = VecGetArrayReadAndMemType(b,&glb_bdata,&mtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    if (mtype!=PETSC_MEMTYPE_DEVICE) SETERRQ(PetscObjectComm((PetscObject) pc),PETSC_ERR_ARG_WRONG,"No GPU data for b");
#endif
    d_block_offsets = jac->d_block_offsets->data();
    // solve each block independently
    const int scr_bytes = vect2D_scr_t::shmem_size(blkSz*nBlk,nwork);
    PetscInfo6(pc,"\tN = %D. %d shared memory words. maxBlkSize = %d. rtol=%e team_size=%D, %D vector threads\n",N,scr_bytes/sizeof(PetscScalar),blkSz,rtol,team_size,KOKKOS_VEC_SIZE);
    Kokkos::parallel_for("Solve", Kokkos::TeamPolicy<>(nBlk, team_size, KOKKOS_VEC_SIZE).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt blkID = team.league_rank(), start = d_block_offsets[blkID], end = d_block_offsets[blkID+1];
        vect2D_scr_t work_vecs(team.team_scratch(KOKKOS_SHARED_LEVEL),end-start,nwork);
        switch (jac->ksp_idx) {
        case BICG_IDX:
          PCKSPSolve_BICG(team, glb_Aai, glb_Aaj, glb_Aaa, work_vecs.data(), rtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata);
          break;
        default:
#if defined(PETSC_USE_DEBUG)
          printf("Unknown KSP type %d\n",jac->ksp_idx);
#else
          /* void */;
#endif
        }
#if defined(PETSC_USE_DEBUG)
        if (d_metadata[blkID].reason<0) printf("Solver diverged %d\n",d_metadata[blkID].reason);
#endif
      });
    ierr = VecRestoreArrayAndMemType(x,&glb_xdata);CHKERRQ(ierr);
    ierr = VecRestoreArrayReadAndMemType(b,&glb_bdata);CHKERRQ(ierr);
    {
      auto h_metadata = Kokkos::create_mirror(Kokkos::HostSpace::memory_space(), d_metadata);
      Kokkos::deep_copy (h_metadata, d_metadata);
      for (int i=0;i<nBlk;i++) {
        PetscInfo3(pc,"%d) Solver reason %d, %d iterations\n",i, h_metadata[i].reason, h_metadata[i].its);
      }
    }
    errsum = 0;
    Kokkos::parallel_reduce (nBlk, [=] (const int idx, PetscInt& lsum) {
        if (d_metadata[idx].reason < 0 && d_metadata[idx].reason != KSP_DIVERGED_ITS && d_metadata[idx].reason != KSP_CONVERGED_ITS) lsum += 1;
      }, errsum);
    if (!errsum) pcreason = PC_NOERROR;
    else pcreason = PC_SUBPC_ERROR;
#if defined(PETSC_USE_DEBUG)
    printf("PCSetFailedReason %d\n",pcreason);
#endif
    ierr = PCSetFailedReason(pc,pcreason);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_KSPKOKKOS(PC pc)
{
  PetscErrorCode    ierr;
  PC_KSPKOKKOS      *jac = (PC_KSPKOKKOS*)pc->data;
  Mat               A = pc->pmat;
  Mat_SeqAIJKokkos  *aijkok;
  PetscBool         flg;

  PetscFunctionBegin;
  if (pc->useAmat) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"No support for using 'use_amat'");
  if (!A) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No matrix - A is used above");
  ierr = PetscObjectTypeCompareAny((PetscObject)A,&flg,MATSEQAIJKOKKOS,MATMPIAIJKOKKOS,MATAIJKOKKOS,"");CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"must use '-dm_mat_type aijkokkos -dm_vec_type kokkos' for -pc_type kspkokkos");
  if (!(aijkok = static_cast<Mat_SeqAIJKokkos*>(A->spptr))) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No aijkok");
  else {
    PetscInt          nrows=A->rmap->n, N, *d_block_offsets;
    const PetscInt    conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp==0) ? 16 : 1;
    const PetscInt    *d_ai=aijkok->i_d.data(), *d_aj=aijkok->j_d.data();
    const PetscScalar *d_aa = aijkok->a_d.data();
    PetscScalar       *d_idiag;
    DM                pack;

    if (!jac->ksp) {
      ierr = PCKSPKOKKOSCreateKSP_KSPKOKKOS(pc);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(jac->ksp);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)jac->ksp,&flg,KSPBICG,"");CHKERRQ(ierr);
      if (flg) {jac->ksp_idx = BICG_IDX; jac->nwork = 6;}
      else SETERRQ1(PetscObjectComm((PetscObject)jac->ksp),PETSC_ERR_ARG_WRONG,"unsupported type %s", ((PetscObject)jac->ksp)->type_name);
    }

    // get block sizes
    jac->nBlocks = 1;
    ierr = PCGetDM(pc, &pack);CHKERRQ(ierr);
    if (!pack) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"no DM. Requires a composite DM");
    ierr = PetscObjectTypeCompare((PetscObject)pack,DMCOMPOSITE,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PetscObjectComm((PetscObject)pack),PETSC_ERR_USER,"Not for type %s",((PetscObject)pack)->type_name);
    ierr = DMCompositeGetNumberDM(pack,&jac->nBlocks);CHKERRQ(ierr);
    if (!jac->vec_diag) {
      ierr = DMCreateGlobalVector(pack, &jac->vec_diag);CHKERRQ(ierr);
    }
    ierr = VecGetSize(jac->vec_diag,&N);CHKERRQ(ierr);
    ierr = PetscInfo5(pc, "Have %D blocks, N=%D nrows=%D rtol=%g type = %s\n", jac->nBlocks, N, nrows, jac->ksp->rtol, ((PetscObject)jac->ksp)->type_name);CHKERRQ(ierr);
    {
      Vec                                                             *subX;
      Kokkos::View<PetscInt*, Kokkos::LayoutRight, Kokkos::HostSpace> h_block_offsets("block_offsets", jac->nBlocks+1);
      ierr = PetscMalloc(jac->nBlocks*sizeof(Vec),&subX);CHKERRQ(ierr);
      ierr = DMCompositeGetAccessArray(pack, jac->vec_diag, jac->nBlocks, NULL, subX);CHKERRQ(ierr);
      h_block_offsets[0] = 0;
      jac->maxBlkSize = 0;
      for (PetscInt ii=0;ii<jac->nBlocks;ii++) {
        PetscInt nloc;
        ierr = VecGetSize(subX[ii],&nloc);CHKERRQ(ierr);
        h_block_offsets[ii+1] = h_block_offsets[ii] + nloc;
        ierr = PetscInfo1(pc,"\tAdd block with %D equations\n",nloc);CHKERRQ(ierr);
        if (nloc > jac->maxBlkSize) jac->maxBlkSize = nloc;
      }
      ierr = DMCompositeRestoreAccessArray(pack, jac->vec_diag, jac->nBlocks, NULL, subX);CHKERRQ(ierr);
      ierr = PetscFree(subX);CHKERRQ(ierr);
      jac->d_block_offsets = new Kokkos::View<PetscInt*, Kokkos::LayoutRight>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(),h_block_offsets));
      Kokkos::deep_copy (*jac->d_block_offsets, h_block_offsets);
    }
    d_block_offsets = jac->d_block_offsets->data();
    jac->d_idiag = new Kokkos::View<PetscScalar*, Kokkos::LayoutRight>("idiag", N);
    d_idiag = jac->d_idiag->data();
    // get diagonal
    Kokkos::parallel_for("Diag", Kokkos::TeamPolicy<>(jac->nBlocks, team_size, KOKKOS_VEC_SIZE), KOKKOS_LAMBDA (const team_member team) {
        const PetscInt blkID = team.league_rank();
        PetscInfo2(pc,"blkID = %d, nloc=%d\n",blkID,d_block_offsets[blkID+1] - d_block_offsets[blkID]);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team,d_block_offsets[blkID],d_block_offsets[blkID+1]), [=] (const int row) {
            const PetscInt    *rp1 = d_aj + d_ai[row];
            const PetscScalar *ap1 = d_aa + d_ai[row];
            const PetscInt     nrow1 = d_ai[row+1] - d_ai[row];
#if defined(PETSC_USE_DEBUG)
            int found = 0;
#endif
            Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, nrow1), [&] (const int& idx) {
                const PetscInt col = rp1[idx];
                if (col==row) {
                  d_idiag[row] = 1./ap1[idx];
#if defined(PETSC_USE_DEBUG)
                  found++;
                  if (col<d_block_offsets[blkID] || col >= d_block_offsets[blkID+1]) printf("ERROR A[%d,%d] not in block diagonal\n",(int)row,(int)col);
                  if (found>1) printf("ERROR A[%d,%d] twice\n",(int)row,(int)col);
#endif
                }
              });
          });
      });
  }
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_KSPKOKKOS(PC pc)
{
  PC_KSPKOKKOS   *jac = (PC_KSPKOKKOS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->vec_diag);CHKERRQ(ierr);
  if (jac->d_block_offsets) delete jac->d_block_offsets;
  if (jac->d_idiag) delete jac->d_idiag;
  jac->d_block_offsets = NULL;
  jac->d_idiag = NULL;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPKOKKOSGetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPKOKKOSSetKSP_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_KSPKOKKOS(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_KSPKOKKOS(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_KSPKOKKOS(PC pc,PetscViewer viewer)
{
  PC_KSPKOKKOS   *jac = (PC_KSPKOKKOS*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  if (!jac->ksp) {ierr = PCKSPKOKKOSCreateKSP_KSPKOKKOS(pc);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"????????????????????????? %D",iascii);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Portable linear solver: Krylov (KSP) method and preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"nwork = %D, rtol = %e, max it =%D, type = %s\n",jac->nwork,jac->ksp->rtol, jac->ksp->max_it,
                                  ((PetscObject)jac->ksp)->type_name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_KSPKOKKOS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PC KSPKOKKOS options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCKSPKOKKOSSetKSP_KSPKOKKOS(PC pc,KSP ksp)
{
  PC_KSPKOKKOS         *jac = (PC_KSPKOKKOS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  jac->ksp = ksp;
  PetscFunctionReturn(0);
}

/*@
   PCKSPKOKKOSSetKSP - Sets the KSP context for a KSP PC.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  ksp - the KSP solver

   Notes:
   The PC and the KSP must have the same communicator

   Level: advanced

@*/
PetscErrorCode  PCKSPKOKKOSSetKSP(PC pc,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(pc,1,ksp,2);
  ierr = PetscTryMethod(pc,"PCKSPKOKKOSSetKSP_C",(PC,KSP),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCKSPKOKKOSGetKSP_KSPKOKKOS(PC pc,KSP *ksp)
{
  PC_KSPKOKKOS         *jac = (PC_KSPKOKKOS*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac->ksp) {ierr = PCKSPKOKKOSCreateKSP_KSPKOKKOS(pc);CHKERRQ(ierr);}
  *ksp = jac->ksp;
  PetscFunctionReturn(0);
}

/*@
   PCKSPKOKKOSGetKSP - Gets the KSP context for a KSP PC.

   Not Collective but KSP returned is parallel if PC was parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  ksp - the KSP solver

   Notes:
   You must call KSPSetUp() before calling PCKSPKOKKOSGetKSP().

   If the PC is not a PCKSPKOKKOS object it raises an error

   Level: advanced

@*/
PetscErrorCode  PCKSPKOKKOSGetKSP(PC pc,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(pc,"PCKSPKOKKOSGetKSP_C",(PC,KSP*),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

/*MC
     PCKSPKOKKOS -  Defines a preconditioner that applies a Krylov solver and preconditioner to the blocks in a AIJASeq matrix on the GPU.

   Options Database Key:
.     -pc_kspkokkos_

   Level: intermediate

   Notes:
    For use with -ksp_type preonly to bypass any CPU work

   Developer Notes:

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCCOMPOSITE, PCSetUseAmat(), PCKSPKOKKOSGetKSP()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_KSPKOKKOS(PC pc)
{
  PetscErrorCode ierr;
  PC_KSPKOKKOS   *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  pc->data = (void*)jac;

  jac->ksp = NULL;
  jac->vec_diag = NULL;
  jac->d_block_offsets = NULL;
  jac->d_idiag = NULL;
  jac->nBlocks = 1; // one block by default (debugging)

  ierr = PetscMemzero(pc->ops,sizeof(struct _PCOps));CHKERRQ(ierr);
  pc->ops->apply           = PCApply_KSPKOKKOS;
  pc->ops->applytranspose  = NULL;
  pc->ops->setup           = PCSetUp_KSPKOKKOS;
  pc->ops->reset           = PCReset_KSPKOKKOS;
  pc->ops->destroy         = PCDestroy_KSPKOKKOS;
  pc->ops->setfromoptions  = PCSetFromOptions_KSPKOKKOS;
  pc->ops->view            = PCView_KSPKOKKOS;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPKOKKOSGetKSP_C",PCKSPKOKKOSGetKSP_KSPKOKKOS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPKOKKOSSetKSP_C",PCKSPKOKKOSSetKSP_KSPKOKKOS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
