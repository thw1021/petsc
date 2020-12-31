#include "petsc/private/sfimpl.h"
#include "petsccublas.h"
#include "petscsystypes.h"
#include <petsc/private/cudavecimpl.h>
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

PetscErrorCode PetscNvshmemMalloc(size_t size, void** ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNvshmemInitializeCheck();CHKERRQ(ierr);
  *ptr = nvshmem_malloc(size);
  if (!*ptr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nvshmem_malloc() failed to allocate %zu bytes",size);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemCalloc(size_t size, void**ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNvshmemInitializeCheck();CHKERRQ(ierr);
  *ptr = nvshmem_calloc(size,1);
  if (!*ptr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"nvshmem_calloc() failed to allocate %zu bytes",size);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemFree_Private(void* ptr)
{
  PetscFunctionBegin;
  nvshmem_free(ptr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemFinalize(void)
{
  PetscFunctionBegin;
  nvshmem_finalize();
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
  cerr = cudaMalloc((void**)&sf->ranks_d,nRemoteRootRanks*sizeof(PetscMPIInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&sf->roffset_d,(nRemoteRootRanks+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(sf->rootsigdisp_d,sf->rootsigdisp,nRemoteRootRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
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
  cerr = cudaMalloc((void**)&bas->iranks_d,nRemoteLeafRanks*sizeof(PetscMPIInt));CHKERRCUDA(cerr);
  cerr = cudaMalloc((void**)&bas->ioffset_d,(nRemoteLeafRanks+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
  cerr = cudaMemcpyAsync(bas->leafsigdisp_d,bas->leafsigdisp,nRemoteLeafRanks*sizeof(PetscInt),cudaMemcpyHostToDevice,NULL);CHKERRCUDA(cerr);
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
    PetscInt oneCuda = (!rootdata || PetscMemTypeCUDA(rootmtype)) && (!leafdata || PetscMemTypeCUDA(leafmtype)) ? 1 : 0; /* Do I use cuda for both root&leafmtype? */
    PetscInt allCuda = oneCuda; /* Assume the same for all ranks. But if not, in opt mode, return value <use_nvshmem> won't be collective! */
   #if defined(PETSC_USE_DEBUG)  /* Check in debug mode. Note MPI_Allreduce is expensive, so only in debug mode */
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
__global__ static void NvshmemSendSignals(PetscInt n,uint64_t *sig,PetscInt *sigdisp,PetscMPIInt *ranks,uint64_t newval)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  /* Each thread puts one remote signal */
  if (i < n) nvshmemx_uint64_signal(sig+sigdisp[i],newval,ranks[i]);
}

/* Wait until local signals equal to the expected value and then set them to a new value

 Input parameters:
  + n        - Number of signals
  . sig      - Local signal address
  . expval   - expected value
  - newval   - Set signals to this new value
*/
__global__ static void NvshmemWaitSignals(PetscInt n,uint64_t *sig,uint64_t expval,uint64_t newval)
{
  nvshmem_uint64_wait_until_all(sig,n,NULL,NVSHMEM_CMP_EQ,expval);
  for (int i=0; i<n; i++) sig[i] = newval;
}

/* ===========================================================================================================

   A set of routines to support receiver initiated communication using the get method

    The getting protocol is:

    Sender has a send buf (sbuf) and a signal variable (ssig);  Receiver has a recv buf (rbuf) and a signal variable (rsig);
    All signal variables have an initial value 0.

    Sender:                                 |  Receiver:
                                            |
  1.  Wait ssig be 0, then set it to 1      |
  2.  Pack data into local sbuf             |
  3.  Put 1 to receiver's rsig              |
                                            |   a. Wait rsig to be 1, then set it 0
                                            |   b. Get data from remote sbuf to local rbuf
                                            |   c. Put 0 to sender's ssig // so sender can proceed at step 1 next time
                                            |   d. Unpack data from local rbuf
   ===========================================================================================================*/

/* Receiver tells its senders that they are allowed to reuse their send buffer (since receiver has got data from their send buffer) */
PetscErrorCode PetscSFLinkSendSignalsToAllowPackingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
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
  NvshmemSendSignals<<<(nfrom+511)/512,512,0,link->remote_comm_stream>>>(nfrom,sig,sigdisp,ranks,0); /* set signals to 0 afterwards */
  PetscFunctionReturn(0);
}

/* Sender waits until it can resue its send buffer */
PetscErrorCode PetscSFLinkWaitSignalsToStartPackingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
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
  NvshmemWaitSignals<<<1,1,0,link->remote_comm_stream>>>(nto,sig,0,1); /* wait the signals to be 0, then set them to 1 */
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
  NvshmemSendSignals<<<(nto+511)/512,512,0,link->remote_comm_stream>>>(nto,sig,sigdisp,ranks,1); /* set signals to 1 */
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
  NvshmemWaitSignals<<<1,1,0,link->remote_comm_stream>>>(nfrom,sig,1,0); /* wait the signals to be 1, then set them to 0 */
  PetscFunctionReturn(0);
}

/* Get data in the given direction */
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
    nvshmemx_getmem_on_stream(dst,src,nelems,pe,link->remote_comm_stream);
  }
  //ierr = PetscSFLinkSendSignalsToAllowPackingData_NVSHMEM(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
/* The code below tries to use nvshmemx_uint64_wait_until_on_stream() to see if there is performance benefit.
   But it hangs for reason I don't know. I keep it here for future reference.
*/

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
    nvshmemx_uint64_wait_until_on_stream(rsig+i,NVSHMEM_CMP_EQ,1,link->remote_comm_stream); /* wait until sig is 1 */
    nvshmemx_getmem_on_stream(dst,src,nelems,pe,link->remote_comm_stream);
  }
  /* After getting data, clear flags of recv buf and send buf */
  PetscClearSendRecvFlags<<<(n+255)/256,256,0,link->remote_comm_stream>>>(n,sranks_d,ssig,ssigdisp,rsig);
  PetscFunctionReturn(0);
}
#endif


/* ===========================================================================================================

   A set of routines to support sender initiated communication using the put method (the default)

    The putting protocol is:

    Sender has a send buf (sbuf) and a signal variable (ssig);  Receiver has a recv buf (rbuf) and a signal variable (rsig);
    All signal variables have an initial value 0.

    Sender:                                 |  Receiver:
                                            |
  1.  Pack data into local sbuf             |
  2.  Wait ssig be 0, then set it to 1      |
  3.  Put data to remote rbuf               |
  4.  Fence // make sure 5 happens after 3  |
  5.  Put 1 to receiver's rsig              |   a. Wait rsig to be 1, then set it 0
                                            |   b. Unpack data from local rbuf
                                            |   c. Put 0 to sender's ssig // so sender can proceed at step 2 next time
   ===========================================================================================================*/
 __global__ static void NvshmemFenceAndSendSignals(PetscInt n,uint64_t *sig,PetscInt *sigdisp,PetscMPIInt *ranks,uint64_t newval)
 {
   /* Each thread puts one remote signal */
   nvshmem_fence();

   for (int i=0; i<n; i++) nvshmemx_uint64_signal(sig+sigdisp[i],newval,ranks[i]);
 }

/* A sender tells its receivers that sent data is deliveried in their corresponding receive buffer */
PetscErrorCode PetscSFLinkSendSignalsOfCompletionOfPuttingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nto,*sigdisp;
  PetscMPIInt       *ranks;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* root sends to leaf */
    nto     = bas->niranks-bas->ndiranks; /* number of remote leaf ranks */
    sig     = link->leafsig;
    sigdisp = bas->leafsigdisp_d;
    ranks   = bas->iranks_d;
  } else { /* LEAF2ROOT, leaf sends to root */
    nto   = sf->nranks-sf->ndranks;
    sig     = link->rootsig;
    sigdisp = sf->rootsigdisp_d;
    ranks   = sf->ranks_d;
  }
  NvshmemFenceAndSendSignals<<<1,1,0,link->remote_comm_stream>>>(nto,sig,sigdisp,ranks,1); /* fence and set remote signals to 1 */
  PetscFunctionReturn(0);
}

/* A receiver waits until it can access its receive buffer */
PetscErrorCode PetscSFLinkWaitSignalsOfCompletionOfPuttingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nfrom;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    nfrom = sf->nranks-sf->ndranks; /* number of remote root ranks. I receive data from them */
    sig   = link->leafsig; /* sig along recv buffer (i.e., leafbuf) */
  } else { /* LEAF2ROOT */
    nfrom = bas->niranks-bas->ndiranks;
    sig   = link->rootsig;
  }
  NvshmemWaitSignals<<<1,1,0,link->remote_comm_stream>>>(nfrom,sig,1,0); /* wait signals to be 1, then clear them */
  PetscFunctionReturn(0);
}

/* A receiver tells its senders that they are allowed to put/send data to here (it implies recv buf is free to take new data) */
PetscErrorCode PetscSFLinkSendSignalsToAllowPuttingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nfrom,*sigdisp;
  PetscMPIInt       *ranks;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    nfrom   = sf->nranks-sf->ndranks; /* number of remote root ranks. I receive data from them */
    sig     = link->leafsig; /* sig along recv buffer (i.e., leafbuf) */
    sigdisp = sf->rootsigdisp_d;
    ranks   = sf->ranks_d;
  } else { /* LEAF2ROOT */
    nfrom   = bas->niranks-bas->ndiranks;
    sig     = link->rootsig;
    sigdisp = bas->leafsigdisp_d;
    ranks   = bas->iranks_d;
  }
  NvshmemSendSignals<<<1,1,0,link->remote_comm_stream>>>(nfrom,sig,sigdisp,ranks,0); /* Set remote signals to 0 */
  PetscFunctionReturn(0);
}

/* A sender waits until it can put/send data to all its receivers */
PetscErrorCode PetscSFLinkWaitSignalsToStartPuttingData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  uint64_t          *sig;
  PetscInt          nto;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* root sends to leaf */
    nto     = bas->niranks-bas->ndiranks; /* number of remote leaf ranks (i.e., # of puts ) */
    sig     = link->rootsig; /* signals along rootbuf */
  } else { /* LEAF2ROOT, leaf sends to root */
    nto     = sf->nranks-sf->ndranks;
    sig     = link->leafsig;
  }
  NvshmemWaitSignals<<<1,1,0,link->remote_comm_stream>>>(nto,sig,0,1); /* Wait signals to be 0, then set them to 1 */
  PetscFunctionReturn(0);
}

/* Sender initiated communication in the given direction  */
PetscErrorCode PetscSFLinkPutData_NVSHMEM(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  char              *sbuf,*rbuf;
  PetscInt          *sbufdisp,*rbufdisp;
  PetscMPIInt       i,j,rank,*dstranks;
  PetscInt          n; /* Number of remote destinaton ranks to put data */

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) { /* In the view of leaf, who is the receiver */
    n        = bas->niranks-bas->ndiranks;                          /* number of remote leaf ranks */
    sbuf     = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* the local root buf is the send buf */
    sbufdisp = bas->ioffset+bas->ndiranks;                          /* offsets of the local recv buf */
    rbuf     = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* leaf buf is the rev buf */
    rbufdisp = bas->leafbufdisp;                                    /* for my i-th remote leaf rank, I will access its leaf buf at offset leafbufdisp[i] */
    dstranks = bas->iranks+bas->ndiranks;                           /* remote leaf ranks */
  } else { /* LEAF2ROOT, so in the view of root, who is the receiver */
    n        = sf->nranks-sf->ndranks;                              /* number of remote root ranks */
    sbuf     = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* send buf is the local leaf buf, also in symmetric heap */
    sbufdisp = sf->roffset+sf->ndranks;                             /* offsets of the local recv buf. Note rbufdisp[0] is not necessarily 0 */
    rbuf     = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]; /* root buf is the recv buf; it is in symmetric heap */
    rbufdisp = sf->rootbufdisp;                                     /* for my i-th remote root rank, I will access its root buf at offset rootbufdisp[i] */
    dstranks = sf->ranks+sf->ndranks;                               /* remote root ranks */
  }

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRMPI(ierr);

  /* Find i such that rstranks[i] is my closest right neighbor. If not exist, set i to 0 */
  for (i=0; i<n; i++) if (dstranks[i]>rank) break;
  if (i == n) i = 0;

  for (j=0; j<n; j++,i=(i+1)%n) { /* Shift to avoid communication hot spot */
    char   *src   = sbuf + (sbufdisp[i]-sbufdisp[0])*link->unitbytes;
    char   *dst   = rbuf + rbufdisp[i]*link->unitbytes;
    size_t nelems = (sbufdisp[i+1]-sbufdisp[i])*link->unitbytes;
    int    pe     = dstranks[i];
    nvshmemx_putmem_on_stream(dst,src,nelems,pe,link->remote_comm_stream);
  }

  ierr = PetscSFLinkSendSignalsOfCompletionOfPuttingData_NVSHMEM(sf,link,PETSCSF_ROOT2LEAF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Destructor when the link uses nvshmem for communication on CUDA device */
static PetscErrorCode PetscSFLinkDestroy_NVSHMEM(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode    ierr;
  cudaError_t       cerr;

  PetscFunctionBegin;
  cerr = cudaEventDestroy(link->root_ready);CHKERRCUDA(cerr);
  cerr = cudaEventDestroy(link->leaf_ready);CHKERRCUDA(cerr);
  cerr = cudaEventDestroy(link->local_comm_end);CHKERRCUDA(cerr);
  cerr = cudaEventDestroy(link->remote_comm_end);CHKERRCUDA(cerr);
  cerr = cudaStreamDestroy(link->remote_comm_stream);CHKERRCUDA(cerr);
  cerr = cudaStreamDestroy(link->local_comm_stream);CHKERRCUDA(cerr);

  /* nvshmem does not need buffers on host, which should be NULL */
  ierr = PetscNvshmemFree(link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->leafsig);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
  ierr = PetscNvshmemFree(link->rootsig);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFLinkCreate_NVSHMEM(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,const void *leafdata,MPI_Op op,PetscSFOperation sfop,PetscSFLink *mylink)
{
  PetscErrorCode    ierr;
  PetscSF_Basic     *bas = (PetscSF_Basic*)sf->data;
  PetscSFLink       *p,link;
  PetscBool         match,rootdirect[2],leafdirect[2];;

  PetscFunctionBegin;
  /* Can we directly send/recv root/leafdata with the given sf, sfop and op?
     We only care root/leafdirect[PETSCSF_REMOTE], since we never need intermeidate buffers in local communication with NVSHMEM.
  */
  if (sfop == PETSCSF_BCAST) {
    rootdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(rootmtype) && bas->rootcontig[PETSCSF_REMOTE]) ? PETSC_TRUE : PETSC_FALSE; /* Pack roots */
    leafdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(leafmtype) && sf->leafcontig[PETSCSF_REMOTE] && op == MPIU_REPLACE) ? PETSC_TRUE : PETSC_FALSE;  /* Unpack leaves */
  } else if (sfop == PETSCSF_REDUCE) {
    leafdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(leafmtype) && sf->leafcontig[PETSCSF_REMOTE]) ? PETSC_TRUE : PETSC_FALSE;  /* Pack leaves */
    rootdirect[PETSCSF_REMOTE] = (PetscMemTypeNVSHMEM(rootmtype) && bas->rootcontig[PETSCSF_REMOTE] && op == MPIU_REPLACE) ? PETSC_TRUE : PETSC_FALSE; /* Unpack roots */
  } else { /* PETSCSF_FETCH */
    rootdirect[PETSCSF_REMOTE] = PETSC_FALSE; /* FETCH always need a separate rootbuf */
    leafdirect[PETSCSF_REMOTE] = PETSC_FALSE; /* We also force allocating a separate leafbuf so that leafdata and leafupdate can share mpi requests */
  }

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
  if (sf->backend == PETSCSF_BACKEND_CUDA) {ierr = PetscSFLinkSetUp_CUDA(sf,link,unit);CHKERRQ(ierr);} /* Setup pack routines, streams etc */
 #if defined(PETSC_HAVE_KOKKOS)
  else if (sf->backend == PETSCSF_BACKEND_KOKKOS) {ierr = PetscSFLinkSetUp_Kokkos(sf,link,unit);CHKERRQ(ierr);}
 #endif

  link->rootdirect[PETSCSF_LOCAL]  = PETSC_TRUE; /* For the local part we directly use root/leafdata */
  link->leafdirect[PETSCSF_LOCAL]  = PETSC_TRUE;

  if (!link->rootsig) {ierr = PetscNvshmemCalloc(bas->niranks_rmax*sizeof(uint64_t),(void**)&link->rootsig);CHKERRQ(ierr);}
  if (!link->leafsig) {ierr = PetscNvshmemCalloc(sf->nranks_rmax*sizeof(uint64_t),(void**)&link->leafsig);CHKERRQ(ierr);} /* Init signals to zero */

  link->use_nvshmem                = PETSC_TRUE;
  link->rootmtype                  = PETSC_MEMTYPE_DEVICE; /* Only need 0/1-based mtype from now on */
  link->leafmtype                  = PETSC_MEMTYPE_DEVICE;
  /* Overwrite some function pointers set by PetscSFLinkSetUp_CUDA */
  link->Destroy                    = PetscSFLinkDestroy_NVSHMEM;
  link->StartCommunication         = PetscSFLinkPutData_NVSHMEM;
  link->FinishCommunication        = PetscSFLinkWaitSignalsOfCompletionOfPuttingData_NVSHMEM;

found:
  if (rootdirect[PETSCSF_REMOTE]) {
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = (char*)rootdata + bas->rootstart[PETSCSF_REMOTE]*link->unitbytes;
  } else {
    if (!link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]) {
      ierr = PetscNvshmemMalloc(bas->rootbuflen_rmax*link->unitbytes,(void**)&link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    }
    link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->rootbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }

  if (leafdirect[PETSCSF_REMOTE]) {
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = (char*)leafdata + sf->leafstart[PETSCSF_REMOTE]*link->unitbytes;
  } else {
    if (!link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]) {
      ierr = PetscNvshmemMalloc(sf->leafbuflen_rmax*link->unitbytes,(void**)&link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE]);CHKERRQ(ierr);
    }
    link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE] = link->leafbuf_alloc[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  }

  link->rootdirect[PETSCSF_REMOTE] = rootdirect[PETSCSF_REMOTE];
  link->leafdirect[PETSCSF_REMOTE] = leafdirect[PETSCSF_REMOTE];
  link->rootdata                   = rootdata; /* root/leafdata are keys to look up links in PetscSFXxxEnd */
  link->leafdata                   = leafdata;
  link->next                       = bas->inuse;
  bas->inuse                       = link;
  *mylink                          = link;
  PetscFunctionReturn(0);
}

template<typename RealType> __global__ static void CudaSqr (RealType *a) {a[0] = a[0]*a[0];}
template<typename RealType> __global__ static void CudaSqrt(RealType *a) {a[0] = sqrt(a[0]);}

#if defined(PETSC_USE_REAL_SINGLE)
PetscErrorCode PetscNvshmemNorm2(float *alpha)
{
  PetscFunctionBegin;
  CudaSqr<<<1,1>>>(alpha);
  nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,1,NULL/*stream*/);
  CudaSqrt<<<1,1>>>(alpha);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemSum(float *alpha)
{
  PetscFunctionBegin;
  nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,1,NULL);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemMax(float *alpha)
{
  PetscFunctionBegin;
  nvshmemx_float_max_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,1,NULL);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemNorm1And2(float *alpha)
{
  PetscFunctionBegin;
  CudaSqr<<<1,1>>>(&alpha[1]);
  nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,2,NULL);
  CudaSqrt<<<1,1>>>(&alpha[1]);
  PetscFunctionReturn(0);
}
#elif defined(PETSC_USE_REAL_DOUBLE)
PetscErrorCode PetscNvshmemNorm2(double *alpha)
{
  PetscFunctionBegin;
  CudaSqr<<<1,1>>>(alpha);
  nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,1,NULL);
  CudaSqrt<<<1,1>>>(alpha);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemSum(double *alpha)
{
  PetscFunctionBegin;
  nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,1,NULL);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemMax(double *alpha)
{
  PetscFunctionBegin;
  nvshmemx_double_max_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,1,NULL);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscNvshmemNorm1And2(double *alpha)
{
  PetscFunctionBegin;
  CudaSqr<<<1,1>>>(&alpha[1]);
  nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD,alpha,alpha,2,NULL);
  CudaSqrt<<<1,1>>>(&alpha[1]);
  PetscFunctionReturn(0);
}
#endif
