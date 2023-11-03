/*
    Code for allocating Unix shared memory on MPI rank 0 and later accessing it from other MPI processes
*/
#include <petscsys.h>

PetscBool PCMPIServerActive          = PETSC_FALSE; // PETSc is running in server mode
PetscBool PCMPIServerInSolve         = PETSC_FALSE; // A parallel server solve is occuring
PetscBool PCMPIServerUseSharedMemory = PETSC_TRUE;  // Use Unix shared memory for distributing objects

#if defined(PETSC_HAVE_SHARED_MEMORY)
  #include <sys/shm.h>
  #include <sys/mman.h>
  #include <errno.h>

typedef struct _PCMPIServerAllocation *PCMPIServerAllocation;
struct _PCMPIServerAllocation {
  void                 *addr; // address on this process; points to same physical address on all processes
  int                   shmkey, shmid;
  size_t                sz;
  PCMPIServerAllocation next;
};
static PCMPIServerAllocation allocations = NULL;

typedef struct {
  size_t shmkey[3];
  size_t sz[3];
} BcastInfo;

#endif

PetscErrorCode PCMPIServerAddressesDestroy(PCMPIServerAddresses *addresses)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHARED_MEMORY)
  PetscCall(PCMPIServerUnmapAddresses(addresses->n, addresses->addr));
  PetscCall(PetscFree(addresses));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCMPIServerMapAddresses - given shared address on the first MPI process determines the
  addresses on the other MPI processes that map to the same physical memory

  Input Parameters:
+ comm       - the `MPI_Comm` to scatter the address
. n          - the number of addresses, each obtained on MPI process zero by `PCMPIServerAllocateArray()`
- baseaddres - the addresses on the first MPI process, ignored on all but first process

  Output Parameter:
. addres - the addresses on each MPI process, the array of void * must already be allocated

  Level: developer

  Note:
  This routine does nothing if `PETSC_HAVE_SHARED_MEMORY` is not defined

.seealso: `PCMPIServerDeallocateArray()`, `PCMPIServerAllocateArray()`, `PCMPIServerUnmapAddresses()`
@*/
PetscErrorCode PCMPIServerMapAddresses(MPI_Comm comm, PetscInt n, const void **baseaddres, void **addres)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHARED_MEMORY)
  if (PetscGlobalRank == 0) {
    BcastInfo bcastinfo;
    for (PetscInt i = 0; i < n; i++) {
      PCMPIServerAllocation allocation = allocations;

      while (allocation) {
        if (allocation->addr == baseaddres[i]) {
          bcastinfo.shmkey[i] = allocation->shmkey;
          bcastinfo.sz[i]     = allocation->sz;
          // PetscCallMPI(MPI_Bcast(&allocation->shmkey, 1, MPI_INT, 0, comm));
          // PetscCallMPI(MPI_Bcast(&allocation->sz, 1, MPIU_SIZE_T, 0, comm)); // TODO combine broadcasts
          addres[i] = (void *)baseaddres[i];
          break;
        }
        allocation = allocation->next;
      }
      PetscCheck(allocation, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to locate allocated shared address %p", baseaddres[i]);
    }
    PetscCallMPI(MPI_Bcast(&bcastinfo, 2 * n, MPIU_SIZE_T, 0, comm));
  } else {
    BcastInfo bcastinfo = {
      {0, 0, 0},
      {0, 0, 0}
    };
    int    shmkey = 0;
    size_t sz     = 0;

    PetscCallMPI(MPI_Bcast(&bcastinfo, 2 * n, MPIU_SIZE_T, 0, comm));
    for (PetscInt i = 0; i < n; i++) {
      PCMPIServerAllocation next = allocations, previous = NULL;
      // PetscCallMPI(MPI_Bcast(&shmkey, 1, MPI_INT, 0, comm));
      // PetscCallMPI(MPI_Bcast(&sz, 1, MPIU_SIZE_T, 0, comm));
      shmkey = (int)bcastinfo.shmkey[i];
      sz     = bcastinfo.sz[i];
      while (next) {
        if (next->shmkey == shmkey) { addres[i] = (void *)next->addr; }
        previous = next;
        next     = next->next;
      }
      if (!next) {
        PCMPIServerAllocation allocation;
        PetscCall(PetscCalloc(sizeof(struct _PCMPIServerAllocation), &allocation));
        allocation->shmkey = shmkey;
        allocation->sz     = sz;
        allocation->shmid  = shmget(allocation->shmkey, allocation->sz, 0666);
        PetscCheck(allocation->shmid != -1, PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to map shared memory key %d of size %d", allocation->shmkey, (int)allocation->sz);
        allocation->addr = shmat(allocation->shmid, (void *)0, 0);
        PetscCheck(allocation->addr, PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to map shared memory key %d", allocation->shmkey);
        addres[i] = allocation->addr;
        if (previous) previous->next = allocation;
        else allocations = allocation;
        // printf("[%d] Mapping address from key %d %p\n",PetscGlobalRank,allocation->shmkey,allocation->addr);
      }
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCMPIServerUnmapAddresses - given shared addresses on a MPI process unlink it

  Input Parameters:
+ n      - the number of addresses, each obtained on MPI process zero by `PCMPIServerAllocateArray()`
- addres - the addresses

  Level: developer

  Note:
  This routine does nothing if `PETSC_HAVE_SHARED_MEMORY` is not defined

.seealso: `PCMPIServerDeallocateArray()`, `PCMPIServerAllocateArray()`, `PCMPIServerMapAddresses()`
@*/
PetscErrorCode PCMPIServerUnmapAddresses(PetscInt n, void **addres)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SHARED_MEMORY)
  if (PetscGlobalRank > 0) {
    for (PetscInt i = 0; i < n; i++) {
      PCMPIServerAllocation next = allocations, previous = NULL;
      PetscBool             found = PETSC_FALSE;

      while (next) {
        if (next->addr == addres[i]) {
          // printf("[%d] Unmapping key %d shmid %d %p\n",PetscGlobalRank,next->shmkey,next->shmid,next->addr);
          PetscCheck(!shmdt(next->addr), PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to shmdt() location %s", strerror(errno));
          if (previous) previous->next = next->next;
          else allocations = next->next;
          PetscCall(PetscFree(next));
          found = PETSC_TRUE;
          break;
        }
        previous = next;
        next     = next->next;
      }
      PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find address %p to unmap", addres[i]);
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCMPIServerAllocateArray - allocates shared memory accessable by all MPI processes in the server

  Not Collective, only called on the first MPI process

  Input Parameters:
+ sz  - the am
- asz - the size of an entry in the array, for example `sizeof(PetscScalar)`

  Output Parameters:
. addr - the address of the memory

  Level: developer

  Note:
  Uses `PetscMalloc()` if `PETSC_HAVE_SHARED_MEMORY` is not defined or the MPI linear solver server is not running

.seealso: [](sec_pcmpi), `PCMPIServerBegin()`, `PCMPI`, `KSPCheckPCMPI()`, `PCMPIServerDeallocateArray()`
@*/
PetscErrorCode PCMPIServerAllocateArray(size_t sz, size_t asz, void **addr)
{
  PetscFunctionBegin;
  if (!PCMPIServerUseSharedMemory || !PCMPIServerActive || PCMPIServerInSolve) PetscCall(PetscMalloc(sz * asz, addr));
#if defined(PETSC_HAVE_SHARED_MEMORY)
  else {
    PCMPIServerAllocation allocation;
    static int            shmkeys = 10;

    PetscCall(PetscCalloc(sizeof(struct _PCMPIServerAllocation), &allocation));
    allocation->shmkey = shmkeys++;
    allocation->sz     = sz * asz;
    allocation->shmid  = shmget(allocation->shmkey, allocation->sz, 0666 | IPC_CREAT);
    PetscCheck(allocation->shmid != -1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to schmget() of size %d with key %d %s", (int)allocation->sz, allocation->shmkey, strerror(errno));
    allocation->addr = shmat(allocation->shmid, (void *)0, 0);
    PetscCheck(allocation->addr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unable to shmat() of shmid %d %s", (int)allocation->shmid, strerror(errno));

    if (!allocations) allocations = allocation;
    else {
      PCMPIServerAllocation next = allocations;
      while (next->next) next = next->next;
      next->next = allocation;
    }
    *addr = allocation->addr;
    // printf("[0] Allocating key %d shmid %d length %d %p\n",allocation->shmkey,allocation->shmid,(int)sz,*addr);
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCMPIServerDeallocateArray - deallocates shared memory accessable by all MPI processes in the server

  Not Collective, only called on the first MPI process

  Input Parameter:
. addr - the address of array

  Level: developer

  Note:
  Uses `PetscFree()` if `PETSC_HAVE_SHARED_MEMORY` is not defined or the MPI linear solver server is not running

.seealso: [](sec_pcmpi), `PCMPIServerBegin()`, `PCMPI`, `KSPCheckPCMPI()`, `PCMPIServerAllocateArray()`
@*/
PetscErrorCode PCMPIServerDeallocateArray(void **addr)
{
  PetscFunctionBegin;
  if (!*addr) PetscFunctionReturn(PETSC_SUCCESS);
  if (!PCMPIServerUseSharedMemory || !PCMPIServerActive || PCMPIServerInSolve) PetscCall(PetscFree(*addr));
#if defined(PETSC_HAVE_SHARED_MEMORY)
  else {
    PCMPIServerAllocation next = allocations, previous = NULL;

    while (next) {
      if (next->addr == *addr) {
        // printf("[0] Deallocating key %d shmid %d array %p\n",next->shmkey, next->shmid,next->addr);
        PetscCheck(!shmctl(next->shmid, IPC_RMID, NULL), PETSC_COMM_SELF, PETSC_ERR_SYS, "Unable to free shared memory addr %p key %d shmid %d %s", *addr, next->shmkey, next->shmid, strerror(errno));
        *addr = NULL;
        if (previous) previous->next = next->next;
        else allocations = next->next;
        PetscCall(PetscFree(next));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      previous = next;
      next     = next->next;
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to locate allocated address %p", *addr);
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
