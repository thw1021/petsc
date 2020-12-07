#include <petsc/private/cudavecimpl.h>
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

PETSC_STATIC_INLINE PetscErrorCode PetscNvshmemMalloc(size_t size, void**ptr)
{
  PetscFunctionBegin;
  *ptr = nvshmem_malloc(size);
  if (!*ptr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nvshmem_malloc() failed to allocate %zu bytes",size);
  PetscFunctionReturn(0);
}

/* Calloc <count> elements with each of <size> bytes */
PETSC_STATIC_INLINE PetscErrorCode PetscNvshmemCalloc(size_t count,size_t size, void**ptr)
{
  PetscFunctionBegin;
  *ptr = nvshmem_calloc(count,size);
  if (!*ptr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nvshmem_calloc() failed to allocate %zu bytes",count*size);
  PetscFunctionReturn(0);
}

#define PetscNvshmemFree(ptr)      ((ptr) && (nvshmem_free(ptr),(ptr)=NULL,0))

static PetscErrorCode PetscSFLinkDestroy_NVSHMEM(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!link->isbuiltin) {ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);}
  ierr = PetscNvshmemFree(link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->leafsig);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->rootsig);CHKERRQ(ierr);
  if (link->stream) {cudaError_t cerr = cudaStreamDestroy(link->stream);CHKERRCUDA(cerr); link->stream = NULL;}
  ierr = PetscFree(link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkCreate_NVSHMEM(PetscSF sf,MPI_Datatype unit,PetscMemType xrootmtype,const void *rootdata,PetscMemType xleafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFLink       *p,link;
  PetscBool         match;

  PetscFunctionBegin;
  /* Look for free nvshmem links in cache */
  for (p=&bas->avail; (link=*p); p=&link->next) {
    if (link->use_nvshmem) {
      ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
      if (match) {
        *p = link->next; /* Remove from available list */
        goto found;
      }
    }
  }
  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = PetscSFLinkSetUp_Host(sf,link,unit);CHKERRQ(ierr); /* Compute link->unitbytes, dup link->unit etc. */
  if (sf->backend == PETSCSF_BACKEND_CUDA) {ierr = PetscSFLinkSetUp_Cuda(sf,link,unit);CHKERRQ(ierr);} /* Setup pack routines */
 #if defined(PETSC_HAVE_KOKKOS)
  else if (sf->backend == PETSCSF_BACKEND_KOKKOS) {ierr = PetscSFLinkSetUp_Kokkos(sf,link,unit);CHKERRQ(ierr);}
 #endif

  if (!link->leafsig) {
    ierr              = PetscNvshmemMalloc(sf->leafbuflen_rmax*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr              = PetscNvshmemCalloc(sf->nranks_rmax*2,sizeof(uint64_t),(void**)&link->leafsig);CHKERRQ(ierr); /* Init signals to zero */
    link->leafsig_old = link->leafsig + sf->nranks_rmax;
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }
  if (!link->rootsig) {
    ierr              = PetscNvshmemMalloc(bas->rootbuflen_rmax*link->unitbytes,(void**)&link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr              = PetscNvshmemCalloc(bas->niranks_rmax*2,sizeof(uint64_t),(void**)&link->rootsig);CHKERRQ(ierr);
    link->rootsig_old = link->rootsig + bas->niranks_rmax;
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }

  link->rootmtype                  = PETSC_MEMTYPE_DEVICE; /* Only need 0/1-based mtype from now on */
  link->leafmtype                  = PETSC_MEMTYPE_DEVICE;
  link->rootdirect[PETSCSF_REMOTE] = PETSC_FALSE; /* For the remote part, we always need root/leaf buffers allocated by nvshmem_malloc*/
  link->leafdirect[PETSCSF_REMOTE] = PETSC_FALSE;
  link->rootdirect[PETSCSF_LOCAL]  = PETSC_FALSE; /* For the local part, we directly use Scatter (since rootmtype=leafmtype)... */
  link->leafdirect[PETSCSF_LOCAL]  = PETSC_FALSE; /* .., making root/leafdirect[PETSCSF_LOCAL] actually useless */
  link->use_nvshmem                = PETSC_TRUE;
  link->Destroy                    = PetscSFLinkDestroy_NVSHMEM;

found:
  link->rootdata  = rootdata; /* root/leafdata are keys to look up links in PetscSFXxxEnd */
  link->leafdata  = leafdata;
  link->next      = bas->inuse;
  bas->inuse      = link;
  *mylink         = link;
  PetscFunctionReturn(0);
}

/* Set up NVSHMEM related fields for an SF of type SFBASIC (only after PetscSFSetup_Basic() already set up dependant fields */
static PetscErrorCode PetscSFSetUp_Basic_NVSHMEM(PetscSF sf)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       i,nRemoteRootRanks,nRemoteLeafRanks;
  PetscMPIInt    tag;
  MPI_Comm       comm;
  MPI_Request    *rootreqs,*leafreqs;
  PetscInt       tmp,stmp[4],rtmp[4]; /* tmps for send/recv buffers */

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)sf,&tag);CHKERRQ(ierr);

  nRemoteRootRanks = sf->nranks-sf->ndranks;
  nRemoteLeafRanks = bas->niranks-bas->ndiranks;
  ierr = PetscMalloc2(nRemoteLeafRanks,&rootreqs,nRemoteRootRanks,&leafreqs);CHKERRQ(ierr);

  stmp[0] = nRemoteRootRanks;
  stmp[1] = sf->leafbuflen[PETSCSF_REMOTE];
  stmp[2] = nRemoteLeafRanks;
  stmp[3] = bas->rootbuflen[PETSCSF_REMOTE];

  ierr = MPIU_Allreduce(stmp,rtmp,4,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);

  sf->nranks_rmax      = rtmp[0];
  sf->leafbuflen_rmax  = rtmp[1];
  bas->niranks_rmax    = rtmp[2];
  bas->rootbuflen_rmax = rtmp[3];

  /* Total four rounds of MPI communications to set up the nvshmem fields */

  /* Root ranks to leaf ranks: send info about rootsigdisp[] and rootbufdisp[] */
  ierr = PetscMalloc2(nRemoteRootRanks,&sf->rootsigdisp,nRemoteRootRanks,&sf->rootbufdisp);CHKERRQ(ierr);
  for (i=0; i<nRemoteRootRanks; i++) {ierr = MPI_Irecv(&sf->rootsigdisp[i],1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm,&leafreqs[i]);CHKERRMPI(ierr);} /* Leaves recv */
  for (i=0; i<nRemoteLeafRanks; i++) {ierr = MPI_Send(&i,1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm);CHKERRMPI(ierr);} /* Roots send. Note i changes, so we use MPI_Send. */
  ierr = MPI_Waitall(nRemoteRootRanks,leafreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

  for (i=0; i<nRemoteRootRanks; i++) {ierr = MPI_Irecv(&sf->rootbufdisp[i],1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm,&leafreqs[i]);CHKERRMPI(ierr);} /* Leaves recv */
  for (i=0; i<nRemoteLeafRanks; i++) {
    tmp  = bas->ioffset[i+bas->ndiranks] - bas->ioffset[bas->ndiranks];
    ierr = MPI_Send(&tmp,1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm);CHKERRMPI(ierr);  /* Roots send. Note tmp changes, so we use MPI_Send. */
  }
  ierr = MPI_Waitall(nRemoteRootRanks,leafreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

  cerr = cudaMalloc((void**)&sf->rootsigdisp_d,nRemoteRootRanks*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&sf->rootbufdisp_d,nRemoteRootRanks*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&sf->ranks_d,nRemoteRootRanks*sizeof(PetscMPIInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&sf->roffset_d,(nRemoteRootRanks+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(sf->rootsigdisp_d,sf->rootsigdisp,nRemoteRootRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(sf->rootbufdisp_d,sf->rootbufdisp,nRemoteRootRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(sf->ranks_d,sf->ranks+sf->ndranks,nRemoteRootRanks*sizeof(PetscMPIInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(sf->roffset_d,sf->roffset+sf->ndranks,(nRemoteRootRanks+1)*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);

  /* Leaf ranks to root ranks: send info about leafsigdisp[] and leafbufdisp[] */
  ierr = PetscMalloc2(nRemoteLeafRanks,&bas->leafsigdisp,nRemoteLeafRanks,&bas->leafbufdisp);CHKERRQ(ierr);
  for (i=0; i<nRemoteLeafRanks; i++) {ierr = MPI_Irecv(&bas->leafsigdisp[i],1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm,&rootreqs[i]);CHKERRMPI(ierr);}
  for (i=0; i<nRemoteRootRanks; i++) {ierr = MPI_Send(&i,1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm);CHKERRMPI(ierr);}
  ierr = MPI_Waitall(nRemoteLeafRanks,rootreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

  for (i=0; i<nRemoteLeafRanks; i++) {ierr = MPI_Irecv(&bas->leafbufdisp[i],1,MPIU_INT,bas->iranks[i+bas->ndiranks],tag,comm,&rootreqs[i]);CHKERRMPI(ierr);}
  for (i=0; i<nRemoteRootRanks; i++) {
    tmp  = sf->roffset[i+sf->ndranks] - sf->roffset[sf->ndranks];
    ierr = MPI_Send(&tmp,1,MPIU_INT,sf->ranks[i+sf->ndranks],tag,comm);CHKERRMPI(ierr);
  }
  ierr = MPI_Waitall(nRemoteLeafRanks,rootreqs,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

  cerr = cudaMalloc((void**)&bas->leafsigdisp_d,nRemoteLeafRanks*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&bas->leafbufdisp_d,nRemoteLeafRanks*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&bas->iranks_d,nRemoteLeafRanks*sizeof(PetscMPIInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&bas->ioffset_d,(nRemoteLeafRanks+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(bas->leafsigdisp_d,bas->leafsigdisp,nRemoteLeafRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(bas->leafbufdisp_d,bas->leafbufdisp,nRemoteLeafRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(bas->iranks_d,bas->iranks+bas->ndiranks,nRemoteLeafRanks*sizeof(PetscMPIInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(bas->ioffset_d,bas->ioffset+bas->ndiranks,(nRemoteLeafRanks+1)*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);

  ierr = PetscFree2(rootreqs,leafreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemInitializeCheck(void)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!PetscNvshmemInitialized) { /* Note NVSHMEM does not provide a routine to check whether it is initialized */
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &PETSC_COMM_WORLD;
    ierr = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr);CHKERRQ(ierr);
    PetscNvshmemInitialized = PETSC_TRUE;
    PetscBeganNvshmem       = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkNvshmemCheck(PetscSF sf,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,PetscBool *use_nvshmem)
{
  PetscErrorCode   ierr;
  MPI_Comm         comm;
  PetscBool        isBasic;
  PetscMPIInt      result = MPI_UNEQUAL;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  /* Check if the sf is eligible for NVSHMEM, if we have not checked yet.
     Note the check result <use_nvshmem> must be the same over comm, since an SFLink must be collectively either NVSHMEM or MPI.
  */
  if (sf->use_nvshmem && !sf->checked_nvshmem_eligibility) {
    /* Only use NVSHMEM for SFBASIC on PETSC_COMM_WORLD  */
    ierr = PetscObjectTypeCompare((PetscObject)sf,PETSCSFBASIC,&isBasic);CHKERRQ(ierr);
    if (isBasic) {ierr = MPI_Comm_compare(PETSC_COMM_WORLD,comm,&result);CHKERRMPI(ierr);}
    if (!isBasic || (result != MPI_IDENT && result != MPI_CONGRUENT)) sf->use_nvshmem = PETSC_FALSE; /* If not eligible, clear the flag so that we don't try again */

    /* Do further check: If on a rank, both rootdata and leafdata are NULL, we would think they are PETSC_MEMTYPE_CUDA (or HOST)
       and then use NVSHMEM. But if root/leafmtypes on other ranks are PETSC_MEMTYPE_HOST (or DEVICE), this would lead to
       inconsistency on the return value <use_nvshmem>. To be safe, we simply disable nvshmem on these rare SFs.
    */
    if (sf->use_nvshmem) {
      PetscInt hasNullRank = (!rootdata && !leafdata) ? 1 : 0;
      ierr = MPI_Allreduce(MPI_IN_PLACE,&hasNullRank,1,MPIU_INT,MPI_LOR,comm);CHKERRMPI(ierr);
      if (hasNullRank) sf->use_nvshmem = PETSC_FALSE;
    }
    sf->checked_nvshmem_eligibility = PETSC_TRUE; /* If eligible, don't do above check again */
  }

  /* Check if rootmtype and leafmtype collectively are PETSC_MEMTYPE_CUDA */
  if (sf->use_nvshmem) {
    PetscInt oneCuda = (!rootdata || rootmtype == PETSC_MEMTYPE_CUDA) && (!leafdata || leafmtype == PETSC_MEMTYPE_CUDA) ? 1 : 0; /* Do I use cuda for both root&leafmtype? */
    PetscInt allCuda = oneCuda; /* Assume the same for all ranks. But if not, in opt mode, return value <use_nvshmem> won't be collective! */
   #if defined(PETSC_USE_DEBUG)  /* Check in dbg mode. Note MPI_Allreduce is expensive and GPU-blocking */
    ierr = MPI_Allreduce(&oneCuda,&allCuda,1,MPIU_INT,MPI_LAND,comm);CHKERRMPI(ierr);
    if (allCuda != oneCuda) SETERRQ(comm,PETSC_ERR_SUP,"root/leaf mtypes are inconsistent among ranks, which may lead to SF nvshmem failure in opt mode. Add -use_nvshmem 0 to disable it.");
   #endif
    if (allCuda) {
      ierr = PetscNvshmemInitializeCheck();CHKERRQ(ierr);
      if (!sf->setup_nvshmem) { /* Set up nvshmem related fields on this SF on-demand */
        ierr = PetscSFSetUp_Basic_NVSHMEM(sf);CHKERRQ(ierr);
        sf->setup_nvshmem = PETSC_TRUE;
      }
      *use_nvshmem = PETSC_TRUE;
    } else {
      *use_nvshmem = PETSC_FALSE;
    }
  } else {
    *use_nvshmem = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*  Put data in my send buffer to the receive buffer of a remote rank

  Input parameters:
  + sbuf      - Send buffer (with symmetric address)
  . sbufdisp  - Data to be send to the i-th remote rank starts in sbuf at offset (sbufdisp[i] - sbufdisp[0])*unitbyte. Note sbufdisp[0] is not necessarily 0
  . rbuf      - Receive buffer (with symmetric address)
  . rbufdisp  - Data sent to the i-th remote rank will be put at offset rdisp[i]*unitbytes of the rbuf on the remote rank
  . sig      -  Signal array for receiving data (with symmetric address)
  . sigdisp  -  I will use the sigdisp[i]-th signal on the i-th remote rank
  . ranks    -  Remote ranks
  - unitbytes - Used to scale offsets. It is the size of link->unit
 */
__global__ static void PetscNvshmemPut(char *sbuf,PetscInt* sbufdisp,char *rbuf,PetscInt *rbufdisp,uint64_t *sig,PetscInt *sigdisp,PetscMPIInt *ranks,size_t unitbytes)
{
  int               i        = blockIdx.x;
  char              *dest    = rbuf + rbufdisp[i]*unitbytes;
  char              *src     = sbuf + (sbufdisp[i]-sbufdisp[0])*unitbytes;
  size_t            nelems   = (sbufdisp[i+1]-sbufdisp[i])*unitbytes;
  uint64_t          *sigaddr = sig + sigdisp[i];
  int               pe       = ranks[i]; /* remote PE this thread block will put data to */

  /* Or nvshmem_char_put_signal_nbi(dest,src,nelems,signal,1,NVSHMEM_SIGNAL_ADD,pe) once it has a _block version */
  nvshmemx_putmem_nbi_block(dest,src,nelems,pe);
  nvshmem_quiet(); /* not nvshmem_fence() since we have to make sure the put is complete at the remote side */
  nvshmemx_signal_op(sigaddr,1,NVSHMEM_SIGNAL_ADD,pe);
}

/* Send out data in rootbuf */
PetscErrorCode PetscSFLinkPutRootData_NVSHMEM(PetscSF sf,PetscSFLink link)
{
  PetscSF_Basic     *bas  = (PetscSF_Basic*)sf->data;
  char              *sbuf,*rbuf;
  uint64_t          *sig;
  PetscInt          *sbufdisp,*rbufdisp,*sigdisp;
  PetscMPIInt       *ranks;
  PetscInt          nto;

  PetscFunctionBegin;
  sbuf      = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* send buf in symmetric heap */
  sbufdisp  = bas->ioffset_d;    /* offsets for sbuf */
  rbuf      = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* recv buf in symmetric heap */
  rbufdisp  = bas->leafbufdisp_d; /* offsets of recv buf at remote */
  sig       = link->leafsig;      /* signal array in symmetric heap */
  sigdisp   = bas->leafsigdisp_d; /* offsets of sig array at remote */
  ranks     = bas->iranks_d;      /* remote leaf ranks */
  nto       = bas->niranks-bas->ndiranks;
  /* Launch nto blocks of 256 threads. Each block puts data for a destination rank  */
  PetscNvshmemPut<<<nto,256,0,link->stream>>>(sbuf,sbufdisp,rbuf,rbufdisp,sig,sigdisp,ranks,link->unitbytes);
  PetscFunctionReturn(0);
}

/* Send out data in leafbuf */
PetscErrorCode PetscSFLinkPutLeafData_NVSHMEM(PetscSF sf,PetscSFLink link)
{
  char              *sbuf,*rbuf;
  uint64_t          *sig;
  PetscInt          *sbufdisp,*rbufdisp,*sigdisp;
  PetscMPIInt       *ranks;
  PetscInt          nto;

  PetscFunctionBegin;
  sbuf      = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  sbufdisp  = sf->roffset_d;
  rbuf      = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  rbufdisp  = sf->rootbufdisp_d;
  sig       = link->rootsig;
  sigdisp   = sf->rootsigdisp_d;
  ranks     = sf->ranks_d;
  nto       = sf->nranks-sf->ndranks;
  /* Launch nto blocks of 256 threads. Each block puts data for a destination rank  */
  PetscNvshmemPut<<<nto,256,0,link->stream>>>(sbuf,sbufdisp,rbuf,rbufdisp,sig,sigdisp,ranks,link->unitbytes);
  PetscFunctionReturn(0);
}

/*  Wait for completion of a send/put.

 Input parameters:
  + sig      - An array of signals at the receiver side. The signals will be updated (increased) by sender.
  - sig_old  - An array storing old values of signals before update
 */
__global__ static void PetscNvshmemWait(uint64_t *sig,uint64_t *sig_old)
{
  int          i = blockIdx.x;

  /* Wait until sig[i] > oldval, then store the return value (the new sig[i]) to sig_old[i] */
  sig_old[i] = nvshmem_signal_wait_until(sig+i,NVSHMEM_CMP_GT,sig_old[i]);
}

/* In a receiver's view, wait for all communications in the given direction to be completed */
PetscErrorCode PetscSFLinkWaitall_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas  = (PetscSF_Basic*)sf->data;
  uint64_t          *sig,*sig_old;
  PetscInt          nfrom;

  PetscFunctionBegin;
  if (direction == PETSCSF_LEAF2ROOT) { /* leaf to root reduce */
    sig       = link->rootsig; /* sig address at the receiver side */
    sig_old   = link->rootsig_old;
    nfrom     = sf->nranks-sf->ndranks;
  } else { /* root to leaf bcast */
    sig       = link->leafsig;
    sig_old   = link->leafsig_old;
    nfrom     = bas->niranks-bas->ndiranks;
  }

  /* Launch nfrom blocks of 1 thread.  Each block waits for a signal from a source rank */
  PetscNvshmemWait<<<nfrom,1,0,link->stream>>>(sig,sig_old);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemFinalize(void)
{
  PetscFunctionBegin;
  nvshmem_finalize();
  PetscFunctionReturn(0);
}

__global__ void PetscNvshmemNorm2(nvshmem_team_t team,float *alpha)
{
  alpha[0] = alpha[0]*alpha[0];
  nvshmem_float_sum_reduce(team,alpha,alpha,1);
  alpha[0] = sqrt(alpha[0]);
}

__global__ void PetscNvshmemNorm2(nvshmem_team_t team,double *alpha)
{
  alpha[0] = alpha[0]*alpha[0];
  nvshmem_double_sum_reduce(team,alpha,alpha,1);
  alpha[0] = sqrt(alpha[0]);
}

__global__ void PetscNvshmemSum(nvshmem_team_t team,float  *alpha) {nvshmem_float_sum_reduce(team,alpha,alpha,1);}
__global__ void PetscNvshmemSum(nvshmem_team_t team,double *alpha) {nvshmem_double_sum_reduce(team,alpha,alpha,1);}

__global__ void PetscNvshmemMax(nvshmem_team_t team,float  *alpha) {nvshmem_float_max_reduce(team,alpha,alpha,1);}
__global__ void PetscNvshmemMax(nvshmem_team_t team,double *alpha) {nvshmem_double_max_reduce(team,alpha,alpha,1);}

__global__ void PetscNvshmemNorm1And2(nvshmem_team_t team,float *alpha)
{
  alpha[1] = alpha[1]*alpha[1];
  nvshmem_float_sum_reduce(team,alpha,alpha,2);
  alpha[1] = sqrt(alpha[1]);
}

__global__ void PetscNvshmemNorm1And2(nvshmem_team_t team,double *alpha)
{
  alpha[1] = alpha[1]*alpha[1];
  nvshmem_double_sum_reduce(team,alpha,alpha,2);
  alpha[1] = sqrt(alpha[1]);
}

PetscErrorCode VecGetNormArray_MPICUDA_NVSHMEM(Vec xin,NormType type,PetscReal **alpha)
{
  PetscErrorCode          ierr;
  PetscInt                offset;

  PetscFunctionBegin;
  ierr = PetscNvshmemInitializeCheck();CHKERRQ(ierr);
  /* Must use zero the norms since some processes might have no vector entries */
  if (!xin->normArray_d) {ierr = PetscNvshmemCalloc(3,sizeof(PetscReal),(void**)&xin->normArray_d);CHKERRQ(ierr);}
  /* Compute offset of this norm in normArray[] */
  if (type == NORM_1 || type == NORM_1_AND_2)        offset = 0;
  else if (type == NORM_2 || type == NORM_FROBENIUS) offset = 1;
  else if (type == NORM_INFINITY)                    offset = 2;

  *alpha = &xin->normArray_d[offset];
  PetscFunctionReturn(0);
}

PetscErrorCode VecFreeNormArray_MPICUDA_NVSHMEM(Vec xin)
{
  PetscErrorCode          ierr;
  PetscFunctionBegin;
  ierr = PetscNvshmemFree(xin->normArray_d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The 'MPI' in 'MPICUDA' only means the vector is parallel. It does not mean we must use MPI for e.g., VecNorm */
PetscErrorCode VecNormAsync_MPICUDA_NVSHMEM(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode          ierr;
  cudaError_t             cerr;
  PetscInt                m = (type == NORM_1_AND_2) ? 2 : 1; /* count of norms */
  PetscReal               *alpha;

  PetscFunctionBegin;
  /* Compute the local norm and then the global norm */
  ierr = VecGetNormArray_MPICUDA_NVSHMEM(xin,type,&alpha);CHKERRQ(ierr);
  ierr = VecNormAsync_SeqCUDA(xin,type,alpha);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {PetscNvshmemNorm2<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}
  else if (type == NORM_1)                      {PetscNvshmemSum<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}
  else if (type == NORM_INFINITY)               {PetscNvshmemMax<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}
  else if (type == NORM_1_AND_2)                {PetscNvshmemNorm1And2<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}

  /* If user did not use the norm array provided by the vector, we need to do the extra copy */
  if (z != alpha) {cerr = cudaMemcpyAsync(z,alpha,sizeof(PetscReal)*m,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);}
  PetscFunctionReturn(0);
}
