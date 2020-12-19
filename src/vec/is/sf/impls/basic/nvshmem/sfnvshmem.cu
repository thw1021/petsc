#include "petsc/private/sfimpl.h"
#include "petsccublas.h"
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
  cudaError_t       cerr;

  PetscFunctionBegin;
  if (!link->isbuiltin) {ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);}
  ierr = PetscNvshmemFree(link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->leafsig);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->rootsig);CHKERRQ(ierr);

  cerr = cudaEventDestroy(link->input_ready);CHKERRCUDA(cerr);
  cerr = cudaEventDestroy(link->lscatter_end);CHKERRCUDA(cerr);
  cerr = cudaEventDestroy(link->recv_end);CHKERRCUDA(cerr);
  cerr = cudaStreamDestroy(link->send_stream);CHKERRCUDA(cerr);
  //cerr = cudaStreamDestroy(link->recv_stream);CHKERRCUDA(cerr);
  cerr = cudaStreamDestroy(link->lscatter_stream);CHKERRCUDA(cerr);

  // if (link->stream) {cudaError_t cerr = cudaStreamDestroy(link->stream);CHKERRCUDA(cerr); link->stream = NULL;}
  ierr = PetscFree(link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkCreate_NVSHMEM(PetscSF sf,MPI_Datatype unit,PetscMemType xrootmtype,const void *rootdata,PetscMemType xleafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  PetscErrorCode    ierr;
  cudaError_t       cerr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFLink       *p,link;
  PetscBool         match;
  int               leastPriority,greatestPriority;

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
    ierr = PetscNvshmemMalloc(sf->leafbuflen_rmax*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr = PetscNvshmemCalloc(sf->nranks_rmax,sizeof(uint64_t),(void**)&link->leafsig);CHKERRQ(ierr); /* Init signals to zero */
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }
  if (!link->rootsig) {
    ierr = PetscNvshmemMalloc(bas->rootbuflen_rmax*link->unitbytes,(void**)&link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    ierr = PetscNvshmemCalloc(bas->niranks_rmax,sizeof(uint64_t),(void**)&link->rootsig);CHKERRQ(ierr);
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }
  cerr = cudaEventCreate(&link->input_ready);CHKERRCUDA(cerr);
  cerr = cudaEventCreate(&link->lscatter_end);CHKERRCUDA(cerr);
  cerr = cudaEventCreate(&link->recv_end);CHKERRCUDA(cerr);
  cerr = cudaDeviceGetStreamPriorityRange(&leastPriority,&greatestPriority);CHKERRCUDA(cerr);

  cerr = cudaStreamCreateWithPriority(&link->send_stream,cudaStreamNonBlocking,greatestPriority);
  //cerr = cudaStreamCreateWithFlags(&link->recv_stream,cudaStreamNonBlocking);CHKERRCUDA(cerr);
  link->recv_stream = link->send_stream;
  cerr = cudaStreamCreateWithPriority(&link->lscatter_stream,cudaStreamNonBlocking,greatestPriority);CHKERRCUDA(cerr);

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

/* Send/Put signals to remote ranks

 Input parameters:
  + n        - Number of remote ranks
  . sig      - Signal address in symmetric heap
  . sigdisp  - To i-th rank, use its signal at offset sigdisp[i]
  . ranks    - remote ranks
  - newval   - Set signals to this value
*/
__global__ static void PetscNvshmemSendSignals(PetscInt n,uint64_t *sig,PetscInt *sigdisp,PetscMPIInt *ranks,uint64_t newval)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  /* Each thread puts one remote signal */
  if (i < n) {
    //printf("set signal at offset %d at remote rank %d to %llu\n",sigdisp[i],ranks[i],newval);
    nvshmemx_uint64_signal(sig+sigdisp[i],newval,ranks[i]);
  }
}

/* Wait until local signals equal to the expected value and then set them to a new value

 Input parameters:
  + n        - Number of signals
  . sig      - Local signal address
  . expval   - expected value
  - newval   - Set signals to this new value
*/
__global__ static void PetscNvshmemWaitSignals(PetscInt n,uint64_t *sig,uint64_t expval,uint64_t newval)
{
  //printf("Before WaitSignals: sig[0] = %llu, expval = %llu, newval = %llu\n", sig[0], expval, newval);
  nvshmem_uint64_wait_until_all(sig,n,NULL,NVSHMEM_CMP_EQ,expval);
  for (int i=0; i<n; i++) sig[i] = newval;
  //printf("After WaitSignals:  sig[0] = %llu, expval = %llu, newval = %llu\n", sig[0], expval, newval);
}

/* Receiver tells its senders that they are allowed to reuse their send buffer (since receiver has got data from their send buffer) */
PetscErrorCode PetscSFLinkSendSignalsToAllowReusingSbuf_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nfrom,*sigdisp;
  PetscMPIInt       *ranks;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* rootbuf is sbuf; leaf allows root to reuse rootbuf (e.g., in next iteration) */
    nfrom   = sf->nranks-sf->ndranks;
    sig     = link->rootsig;
    sigdisp = sf->rootsigdisp_d;
    ranks   = sf->ranks_d;
  } else { /* LEAF2ROOT, leafbuf is sbuf; root allows leaf to reuse leafbuf */
    nfrom   = bas->niranks-bas->ndiranks;
    sig     = link->leafsig;
    sigdisp = bas->leafsigdisp_d;
    ranks   = bas->iranks_d;
  }
  PetscNvshmemSendSignals<<<(nfrom+511)/512,512,0,link->recv_stream>>>(nfrom,sig,sigdisp,ranks,0); /* set signals to 0 afterwards */
  PetscFunctionReturn(0);
}

/* Sender waits until it can resue its send buffer */
PetscErrorCode PetscSFLinkWaitSignalsToStartReusingSbuf_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nto;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    sig = link->rootsig;              /* send from root */
    nto = bas->niranks-bas->ndiranks; /* I will send to nto remote ranks */
  } else { /* LEAF2ROOT */
    sig = link->leafsig;
    nto = sf->nranks-sf->ndranks;
  }
  PetscNvshmemWaitSignals<<<1,1,0,link->send_stream>>>(nto,sig,0,1); /* wait the signals to be 0, then set them to 1 */
  PetscFunctionReturn(0);
}

/* Sender tells its receivers that they are allowed to get data from his send buffer */
PetscErrorCode PetscSFLinkSendSignalsToAllowGettingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nto,*sigdisp;
  PetscMPIInt       *ranks;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* root allows leaf to get data */
    nto     = bas->niranks-bas->ndiranks; /* number of remote leaf ranks */
    sig     = link->leafsig;              /* signals along leaves */
    sigdisp = bas->leafsigdisp_d;
    ranks   = bas->iranks_d;
  } else { /* LEAF2ROOT */
    nto     = sf->nranks-sf->ndranks;
    sig     = link->rootsig;
    sigdisp = sf->rootsigdisp_d;
    ranks   = sf->ranks_d;
  }
  PetscNvshmemSendSignals<<<(nto+511)/512,512,0,link->send_stream>>>(nto,sig,sigdisp,ranks,1); /* set signals to 1 */
  PetscFunctionReturn(0);
}

/* Receiver wait for signals to allow him to start getting data from senders */
PetscErrorCode PetscSFLinkWaitSignalsToStartGettingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nfrom;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    sig   = link->leafsig;          /* gettting to leaf buf */
    nfrom = sf->nranks-sf->ndranks; /* I will get data from nfrom remote root ranks */
  } else { /* LEAF2ROOT */
    sig   = link->rootsig;
    nfrom = bas->niranks-bas->ndiranks;
  }
  PetscNvshmemWaitSignals<<<1,1,0,link->recv_stream>>>(nfrom,sig,1,0); /* wait the signals to be 1, then set them to 0 */
  PetscFunctionReturn(0);
}

#if 1
/* Send(Get) data in the given direction */
PetscErrorCode PetscSFLinkGetData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  char              *sbuf,*rbuf;
  PetscInt          *sbufdisp,*rbufdisp;
  PetscMPIInt       *srcranks;
  PetscInt          n;

  PetscFunctionBegin;
  ierr = PetscSFLinkWaitSignalsToStartGettingData_NVSHMEM(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  if (direction == PETSCSF_ROOT2LEAF) { /* In the view of leaf, who is the receiver */
    n        = sf->nranks-sf->ndranks;                              /* number of remote root ranks */
    sbuf     = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* root buf is the send buf; it is in symmetric heap */
    sbufdisp = sf->rootbufdisp;                                     /* for my i-th remote root rank, I will access its root buf at offset rootbufdisp[i] */
    rbuf     = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* recv buf is the local leaf buf, also in symmetric heap */
    rbufdisp = sf->roffset+sf->ndranks;                             /* offsets of the local recv buf. Note rbufdisp[0] is not necessarily 0 */
    srcranks = sf->ranks+sf->ndranks;                               /* remote root ranks */
  } else { /* LEAF2ROOT, so in the view of root, who is the receiver */
    n        = bas->niranks-bas->ndiranks;                          /* number of remote leaf ranks */
    sbuf     = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* leaf buf is the send buf */
    sbufdisp = bas->leafbufdisp;                                    /* for my i-th remote leaf rank, I will access its leaf buf at offset leafbufdisp[i] */
    rbuf     = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* the local root buf is the recv buf */
    rbufdisp = bas->ioffset+bas->ndiranks;                          /* offsets of the local recv buf */
    srcranks = bas->iranks+bas->ndiranks;                           /* remote leaf ranks */
  }
  for (int i=0; i<n; i++) {
    char   *src   = sbuf + sbufdisp[i]*link->unitbytes;
    char   *dst   = rbuf + (rbufdisp[i]-rbufdisp[0])*link->unitbytes;
    size_t nelems = (rbufdisp[i+1]-rbufdisp[i])*link->unitbytes;
    int    pe     = srcranks[i];
    nvshmemx_getmem_on_stream(dst,src,nelems,pe,link->send_stream);
  }
  //ierr = PetscSFLinkSendSignalsToAllowReusingSbuf_NVSHMEM(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
/* In receiver's view, clear flags of the recv buf and the send buf */
__global__ static void PetscClearSendRecvFlags(PetscInt n,PetscMPIInt *sranks,uint64_t *ssig,PetscInt *ssigdisp,uint64_t *rsig)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<n) {
    rsig[i] = 0;
    nvshmemx_uint64_signal(ssig+ssigdisp[i],0,sranks[i]);
  }
}

PetscErrorCode PetscSFLinkGetData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  char              *sbuf,*rbuf;
  PetscInt          *sbufdisp,*rbufdisp;
  PetscInt          *ssigdisp;
  PetscMPIInt       *sranks_h,*sranks_d;
  uint64_t          *ssig,*rsig;
  PetscInt          n;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* In the view of leaf, who is the receiver */
    n        = sf->nranks-sf->ndranks;                              /* number of remote root ranks */
    sbuf     = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* root buf is the send buf; it is in symmetric heap */
    sbufdisp = sf->rootbufdisp;                                     /* for my i-th remote root rank, I will access its root buf at offset rootbufdisp[i] */
    rbuf     = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* local leaf buf is the recv buf, also in symmetric heap */
    rbufdisp = sf->roffset+sf->ndranks;                             /* offsets of the local recv buf. Note rbufdisp[0] is not necessarily 0 */
    sranks_h = sf->ranks+sf->ndranks;                               /* remote src/root ranks on host */
    sranks_d = sf->ranks_d;                                         /* remote src/root ranks on device */
    rsig     = link->leafsig;                                       /* signals along local recv buf (leafbuf). Wait for them to be 1 to start gettting data */
    ssig     = link->rootsig;                                       /* signals along remote rootbuf */
    ssigdisp = sf->rootsigdisp_d;                                   /* signal disp along remote rootbuf */
  } else { /* LEAF2ROOT, so in the view of root, who is the receiver */
    n        = bas->niranks-bas->ndiranks;                          /* number of remote leaf ranks */
    sbuf     = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* leaf buf is the send buf */
    sbufdisp = bas->leafbufdisp;                                    /* for my i-th remote leaf rank, I will access its leaf buf at offset leafbufdisp[i] */
    rbuf     = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* the local root buf is the recv buf */
    rbufdisp = bas->ioffset+bas->ndiranks;                          /* offsets of the local recv buf */
    sranks_h = bas->iranks+bas->ndiranks;                           /* remote src/leaf ranks on host */
    sranks_d = bas->iranks_d;                                       /* remote src/leaf ranks on device */
    rsig     = link->rootsig;                                       /* signals along recv buf (rootbuf) */
    ssig     = link->leafsig;                                       /* signals along remote leafbuf */
    ssigdisp = bas->leafsigdisp_d;                                  /* signal disp along remote leafbuf */
  }

  /* Get data, i.e., copy data from sbuf to rbuf */
  for (int i=0; i<n; i++) {
    char   *src   = sbuf + sbufdisp[i]*link->unitbytes;
    char   *dst   = rbuf + (rbufdisp[i]-rbufdisp[0])*link->unitbytes;
    size_t nelems = (rbufdisp[i+1]-rbufdisp[i])*link->unitbytes;
    int    pe     = sranks_h[i];
    printf("waiting until sig is not 0\n");
    nvshmemx_uint64_wait_until_on_stream(rsig+i,NVSHMEM_CMP_EQ,1,link->recv_stream); /* wait until sig is 1 */
    nvshmemx_getmem_on_stream(dst,src,nelems,pe,link->recv_stream);
  }
  /* After getting data, clear flags of recv buf and send buf */
  PetscClearSendRecvFlags<<<(n+255)/256,256,0,link->recv_stream>>>(n,sranks_d,ssig,ssigdisp,rsig);
  PetscFunctionReturn(0);
}
#endif

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


__global__ static void PetscCudaSqr (PetscReal *a) {a[0] = a[0]*a[0];}
__global__ static void PetscCudaSqrt(PetscReal *a) {a[0] = sqrt(a[0]);}

void PetscNvshmemNorm2(nvshmem_team_t team,double *alpha)
{
  PetscCudaSqr<<<1,1>>>(alpha);
  nvshmemx_double_sum_reduce_on_stream(team,alpha,alpha,1,NULL);
  PetscCudaSqrt<<<1,1>>>(alpha);
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
  PetscInt                offset = 0;

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
  if (type == NORM_2 || type == NORM_FROBENIUS) {PetscNvshmemNorm2(NVSHMEM_TEAM_WORLD,alpha);}
  else if (type == NORM_1)                      {PetscNvshmemSum<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}
  else if (type == NORM_INFINITY)               {PetscNvshmemMax<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}
  else if (type == NORM_1_AND_2)                {PetscNvshmemNorm1And2<<<1,1>>>(NVSHMEM_TEAM_WORLD,alpha);}

  /* If user did not use the norm array provided by the vector, we need to do the extra copy */
  if (z != alpha) {cerr = cudaMemcpyAsync(z,alpha,sizeof(PetscReal)*m,cudaMemcpyDeviceToDevice);CHKERRCUDA(cerr);}
  PetscFunctionReturn(0);
}
