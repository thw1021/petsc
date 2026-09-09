/*
   NCCL as the inter-process transport of PETSCSFBASIC for root and leaf data on CUDA devices.

   NCCL point-to-point operations are enqueued on a CUDA stream and never block the host, so an exchange over NCCL is
   fully stream-ordered: the pack kernel, the communication and the unpack kernel are chained by stream dependencies
   and the CPU only enqueues work. MPI is still used to bootstrap NCCL (ncclGetUniqueId() has to be broadcast out of
   band) and for everything that is not the inter-process data path of PETSCSFBASIC.

   One exchange costs one NCCL group: an event records the readiness of the packed send buffer on the compute stream
   (link->stream); a dedicated high-priority stream (link->remoteCommStream) waits for it and runs a single
   ncclGroupStart()/ncclGroupEnd() holding one ncclSend() per remote destination rank and one ncclRecv() per remote
   source rank; at PetscSFXxxEnd() the compute stream waits for an event recorded on the communication stream before
   the unpack kernel runs. Reductions (MPI_SUM etc.) are performed by the unpack kernels, so NCCL only moves bytes and
   no NCCL datatype or reduction operator is ever needed.

   NCCL does not support several ranks of one communicator on the same GPU, so a communicator whose ranks share a GPU
   stays on MPI. The decision whether a given exchange uses NCCL must be the same on every rank, see
   PetscSFLinkNcclCheck().
*/
#include <petscdevice_cuda.h>
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <nccl.h>

#define PetscCallNCCL(...) \
  do { \
    ncclResult_t nccl_ierr_ = __VA_ARGS__; \
    PetscCheck(nccl_ierr_ == ncclSuccess, PETSC_COMM_SELF, PETSC_ERR_LIB, "NCCL error %d: %s", (int)nccl_ierr_, ncclGetErrorString(nccl_ierr_)); \
  } while (0)

/* One NCCL communicator per MPI communicator. It is cached as an attribute of the (PETSc inner) MPI communicator and
   also kept in a list, so that all of them can be destroyed at PetscFinalize() while CUDA is still usable */
typedef struct _n_PetscNcclComm *PetscNcclComm;
struct _n_PetscNcclComm {
  ncclComm_t    comm;    /* the NCCL communicator; NULL when the MPI communicator cannot use NCCL (its ranks share a GPU) */
  MPI_Comm      mpicomm; /* the MPI communicator it is attached to */
  PetscNcclComm next;
};

static PetscMPIInt   Petsc_Nccl_keyval           = MPI_KEYVAL_INVALID;
static PetscNcclComm PetscNcclCommList           = NULL;
static PetscBool     PetscNcclFinalizeRegistered = PETSC_FALSE;

/* Called by MPI when a communicator carrying a NCCL communicator is freed, and through MPI_Comm_delete_attr() by PetscNcclFinalize() */
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_Nccl_Attr_DeleteFn(MPI_Comm comm, PetscMPIInt keyval, void *val, void *extra_state)
{
  PetscNcclComm  nc = (PetscNcclComm)val;
  PetscNcclComm *p  = &PetscNcclCommList;

  PetscFunctionBegin;
  while (*p && *p != nc) p = &(*p)->next;
  if (*p) *p = nc->next; /* unlink it */
  if (nc->comm) {
    PetscCallReturnMPI(PetscInfo(NULL, "Destroying the NCCL communicator attached to MPI_Comm %ld\n", (long)comm));
    if (ncclCommDestroy(nc->comm) != ncclSuccess) PetscFunctionReturn(MPI_ERR_OTHER);
  }
  PetscCallReturnMPI(PetscFree(nc));
  PetscFunctionReturn(MPI_SUCCESS);
}

/* Registered with PetscRegisterFinalize() when the first NCCL communicator is created */
static PetscErrorCode PetscNcclFinalize(void)
{
  PetscFunctionBegin;
  while (PetscNcclCommList) {
    PetscNcclComm head = PetscNcclCommList;

    PetscCallMPI(MPI_Comm_delete_attr(head->mpicomm, Petsc_Nccl_keyval)); /* invokes Petsc_Nccl_Attr_DeleteFn(), which unlinks and destroys it */
    PetscCheck(PetscNcclCommList != head, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to detach a NCCL communicator from its MPI_Comm");
  }
  if (Petsc_Nccl_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Nccl_keyval));
  Petsc_Nccl_keyval           = MPI_KEYVAL_INVALID;
  PetscNcclFinalizeRegistered = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* NCCL does not support several ranks of one communicator on the same GPU (it may hang), so compare the GPUs of the ranks sharing a node */
static PetscErrorCode PetscNcclCheckDistinctDevices(MPI_Comm comm, PetscBool *distinct)
{
  PetscInt dup = 0;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  {
    PetscShmComm pshmcomm;
    MPI_Comm     shmcomm;
    PetscMPIInt  shmsize, shmrank;
    char         busid[32], *allbusid;
    int          dev;

    PetscCall(PetscShmCommGet(comm, &pshmcomm));
    PetscCall(PetscShmCommGetMpiShmComm(pshmcomm, &shmcomm));
    PetscCallMPI(MPI_Comm_size(shmcomm, &shmsize));
    PetscCallMPI(MPI_Comm_rank(shmcomm, &shmrank));
    PetscCallCUDA(cudaGetDevice(&dev));
    PetscCallCUDA(cudaDeviceGetPCIBusId(busid, (int)sizeof(busid), dev));
    PetscCall(PetscMalloc1(shmsize * sizeof(busid), &allbusid));
    PetscCallMPI(MPI_Allgather(busid, (PetscMPIInt)sizeof(busid), MPI_CHAR, allbusid, (PetscMPIInt)sizeof(busid), MPI_CHAR, shmcomm));
    for (PetscMPIInt i = 0; i < shmsize; i++) {
      PetscBool same;

      if (i == shmrank) continue;
      PetscCall(PetscStrcmp(busid, allbusid + i * sizeof(busid), &same));
      if (same) dup = 1;
    }
    PetscCall(PetscFree(allbusid));
  }
#else
  dup = 1; /* cannot tell, so do not risk a hang */
#endif
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &dup, 1, MPIU_INT, MPI_MAX, comm));
  *distinct = dup ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get the NCCL communicator of an MPI communicator, creating it (collectively) on first use. Returns NULL if the communicator cannot use NCCL */
static PetscErrorCode PetscNcclCommGet(MPI_Comm comm, ncclComm_t *ncomm)
{
  PetscNcclComm nc = NULL;
  PetscMPIInt   flg;

  PetscFunctionBegin;
  if (Petsc_Nccl_keyval == MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_Nccl_Attr_DeleteFn, &Petsc_Nccl_keyval, NULL));
  if (!PetscNcclFinalizeRegistered) {
    PetscCall(PetscRegisterFinalize(PetscNcclFinalize));
    PetscNcclFinalizeRegistered = PETSC_TRUE;
  }
  PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_Nccl_keyval, &nc, &flg));
  if (!flg) { /* First use of NCCL on this communicator */
    PetscBool distinct;

    PetscCall(PetscNew(&nc));
    nc->mpicomm = comm;
    PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA)); /* NCCL binds each rank to its current CUDA device, which PETSc must have selected first */
    PetscCall(PetscNcclCheckDistinctDevices(comm, &distinct));
    if (distinct) {
      ncclUniqueId   id;
      PetscMPIInt    size, rank;
      int            version = 0;
      PetscLogDouble t0, t1;

      PetscCallMPI(MPI_Comm_size(comm, &size));
      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(PetscTime(&t0));
      if (rank == 0) PetscCallNCCL(ncclGetUniqueId(&id));
      PetscCallMPI(MPI_Bcast(&id, (PetscMPIInt)sizeof(id), MPI_BYTE, 0, comm));
      PetscCallNCCL(ncclCommInitRank(&nc->comm, size, id, rank));
      PetscCall(PetscTime(&t1));
      PetscCallNCCL(ncclGetVersion(&version));
      /* Creating the communicator is a one-time cost of the order of seconds, and NCCL connects the peers lazily during the first exchange, which costs about as much again */
      PetscCall(PetscInfo(NULL, "Created a NCCL (version %d) communicator with %d ranks on MPI_Comm %ld in %g seconds\n", version, size, (long)comm, t1 - t0));
    } else PetscCall(PetscInfo(NULL, "Some ranks of MPI_Comm %ld share a GPU, which NCCL does not support; PetscSF will use MPI on it\n", (long)comm));
    nc->next          = PetscNcclCommList;
    PetscNcclCommList = nc;
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Nccl_keyval, nc));
  }
  *ncomm = nc->comm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Decide whether the exchange the link is being created for uses NCCL. The result must be the same on all ranks of the sf's communicator */
PetscErrorCode PetscSFLinkNcclCheck(PetscSF sf, PetscSFLink link, PetscBool *use_nccl)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  *use_nccl = PETSC_FALSE;
  if (!sf->use_nccl) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  if (!sf->checked_nccl_eligibility) { /* Decide once, and collectively, whether links of this sf may use NCCL at all */
    PetscMPIInt size;
    PetscInt    bad;
    ncclComm_t  ncomm = NULL;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    /* A rank without root and leaf data cannot tell its memory type, so its per-call vote below could differ from the other ranks'. Such sfs stay on MPI, as do single-rank sfs, which have no inter-process communication */
    bad = (size == 1 || (!link->rootdata && !link->leafdata)) ? 1 : 0;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &bad, 1, MPIU_INT, MPI_MAX, comm));
    if (!bad) PetscCall(PetscNcclCommGet(comm, &ncomm));
    sf->nccl_eligible            = ncomm ? PETSC_TRUE : PETSC_FALSE;
    sf->checked_nccl_eligibility = PETSC_TRUE;
    PetscCall(PetscInfo(sf, "PetscSF %s use NCCL for the inter-process communication of device data\n", sf->nccl_eligible ? "will" : "will not"));
  }
  if (sf->nccl_eligible) {
    PetscInt onedev = ((!link->rootdata || PetscMemTypeDevice(link->rootmtype)) && (!link->leafdata || PetscMemTypeDevice(link->leafmtype))) ? 1 : 0; /* Are root and leaf data on device on this rank? */
    PetscInt alldev = onedev; /* Assume the same on all ranks. If not, in optimized mode the outcome would not be collective and the exchange would hang */

#if defined(PETSC_USE_DEBUG)
    PetscCallMPI(MPIU_Allreduce(&onedev, &alldev, 1, MPIU_INT, MPI_LAND, comm));
    PetscCheck(alldev == onedev, comm, PETSC_ERR_SUP, "root/leaf memory types are inconsistent among ranks, so NCCL cannot be used by all of them. Add -use_nccl 0 to disable it");
#endif
    *use_nccl = alldev ? PETSC_TRUE : PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The stream and events of an NCCL exchange are created lazily on the (MPI-style) link and destroyed with it */
static PetscErrorCode PetscSFLinkSetUp_NCCL(PetscSFLink link)
{
  int greatestPriority;

  PetscFunctionBegin;
  if (!link->ncclinited) {
    PetscCallCUDA(cudaDeviceGetStreamPriorityRange(NULL, &greatestPriority));
    PetscCallCUDA(cudaStreamCreateWithPriority(&link->remoteCommStream, cudaStreamNonBlocking, greatestPriority));
    PetscCallCUDA(cudaEventCreateWithFlags(&link->dataReady, cudaEventDisableTiming));
    PetscCallCUDA(cudaEventCreateWithFlags(&link->endRemoteComm, cudaEventDisableTiming));
    link->ncclinited = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSFLinkDestroy_NCCL(PetscSF sf, PetscSFLink link)
{
  PetscFunctionBegin;
  if (link->ncclinited) {
    PetscCallCUDA(cudaEventDestroy(link->dataReady));
    PetscCallCUDA(cudaEventDestroy(link->endRemoteComm));
    PetscCallCUDA(cudaStreamDestroy(link->remoteCommStream));
    link->ncclinited = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Start the inter-process communication: one NCCL group with all sends and receives of this rank, enqueued on the communication stream */
PetscErrorCode PetscSFLinkStartCommunication_NCCL(PetscSF sf, PetscSFLink link, PetscSFDirection direction)
{
  PetscSF_Basic     *bas = (PetscSF_Basic *)sf->data;
  PetscMPIInt        nrootranks, ndrootranks, nleafranks, ndleafranks;
  const PetscMPIInt *rootranks, *leafranks;
  const PetscInt    *rootoffset, *leafoffset;
  char              *rootbuf = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  char              *leafbuf = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
  ncclComm_t         ncomm;
  MPI_Comm           comm;
  PetscBool          rootsend = (direction == PETSCSF_ROOT2LEAF) ? PETSC_TRUE : PETSC_FALSE; /* roots send to leaves, or leaves send to roots */
  PetscLogDouble     nsend = 0, nrecv = 0, sendlen = 0, recvlen = 0;

  PetscFunctionBegin;
  if (!bas->rootbuflen[PETSCSF_REMOTE] && !sf->leafbuflen[PETSCSF_REMOTE]) PetscFunctionReturn(PETSC_SUCCESS); /* no remote neighbors */
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscNcclCommGet(comm, &ncomm));
  PetscCheck(ncomm, comm, PETSC_ERR_PLIB, "PetscSF was found eligible for NCCL but its communicator has no NCCL communicator");
  PetscCall(PetscSFLinkSetUp_NCCL(link));
  PetscCall(PetscSFGetRootInfo_Basic(sf, &nrootranks, &ndrootranks, &rootranks, &rootoffset, NULL));
  PetscCall(PetscSFGetLeafInfo_Basic(sf, &nleafranks, &ndleafranks, &leafranks, &leafoffset, NULL, NULL));

  /* The packed send buffer is produced on link->stream; the communication stream waits for it and the group is enqueued there */
  PetscCallCUDA(cudaEventRecord(link->dataReady, link->stream));
  PetscCallCUDA(cudaStreamWaitEvent(link->remoteCommStream, link->dataReady, 0));
  PetscCallNCCL(ncclGroupStart());
  for (PetscMPIInt i = ndrootranks; i < nrootranks; i++) { /* rootbuf is exchanged with the ranks whose leaves reference my roots */
    char  *buf   = rootbuf + (rootoffset[i] - rootoffset[ndrootranks]) * link->unitbytes;
    size_t bytes = (size_t)(rootoffset[i + 1] - rootoffset[i]) * link->unitbytes;

    if (!bytes) continue;
    if (rootsend) {
      PetscCallNCCL(ncclSend(buf, bytes, ncclInt8, rootranks[i], ncomm, link->remoteCommStream));
      nsend++;
      sendlen += (PetscLogDouble)bytes;
    } else {
      PetscCallNCCL(ncclRecv(buf, bytes, ncclInt8, rootranks[i], ncomm, link->remoteCommStream));
      nrecv++;
      recvlen += (PetscLogDouble)bytes;
    }
  }
  for (PetscMPIInt i = ndleafranks; i < nleafranks; i++) { /* leafbuf is exchanged with the ranks owning the roots my leaves reference */
    char  *buf   = leafbuf + (leafoffset[i] - leafoffset[ndleafranks]) * link->unitbytes;
    size_t bytes = (size_t)(leafoffset[i + 1] - leafoffset[i]) * link->unitbytes;

    if (!bytes) continue;
    if (rootsend) {
      PetscCallNCCL(ncclRecv(buf, bytes, ncclInt8, leafranks[i], ncomm, link->remoteCommStream));
      nrecv++;
      recvlen += (PetscLogDouble)bytes;
    } else {
      PetscCallNCCL(ncclSend(buf, bytes, ncclInt8, leafranks[i], ncomm, link->remoteCommStream));
      nsend++;
      sendlen += (PetscLogDouble)bytes;
    }
  }
  PetscCallNCCL(ncclGroupEnd());
  if (PetscDefined(USE_LOG)) { /* account for the messages as -log_view does for MPI sends and receives */
    PetscCall(PetscAddLogDouble(&petsc_isend_ct, &petsc_isend_ct_th, nsend));
    PetscCall(PetscAddLogDouble(&petsc_irecv_ct, &petsc_irecv_ct_th, nrecv));
    PetscCall(PetscAddLogDouble(&petsc_isend_len, &petsc_isend_len_th, sendlen));
    PetscCall(PetscAddLogDouble(&petsc_irecv_len, &petsc_irecv_len_th, recvlen));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Finish the inter-process communication: the compute stream waits for the NCCL kernels, so that the unpack kernel sees the received data and the next pack does not overwrite a send buffer still in use. The host does not block */
PetscErrorCode PetscSFLinkFinishCommunication_NCCL(PetscSF sf, PetscSFLink link, PetscSFDirection direction)
{
  PetscSF_Basic *bas = (PetscSF_Basic *)sf->data;

  PetscFunctionBegin;
  if (!bas->rootbuflen[PETSCSF_REMOTE] && !sf->leafbuflen[PETSCSF_REMOTE]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallCUDA(cudaEventRecord(link->endRemoteComm, link->remoteCommStream));
  PetscCallCUDA(cudaStreamWaitEvent(link->stream, link->endRemoteComm, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
