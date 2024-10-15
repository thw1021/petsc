#pragma once

#include <petscvec.h>     /*I "petscvec.h" I*/
#include <petscmat.h>     /*I      "petscmat.h"          I*/
#include <petscdmswarm.h> /*I      "petscdmswarm.h"    I*/
#include <petsc/private/dmimpl.h>

PETSC_EXTERN PetscBool  SwarmProjcite;
PETSC_EXTERN const char SwarmProjCitation[];

PETSC_EXTERN PetscLogEvent DMSWARM_Migrate;
PETSC_EXTERN PetscLogEvent DMSWARM_SetSizes;
PETSC_EXTERN PetscLogEvent DMSWARM_AddPoints;
PETSC_EXTERN PetscLogEvent DMSWARM_RemovePoints;
PETSC_EXTERN PetscLogEvent DMSWARM_Sort;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerTopologySetup;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerBegin;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerEnd;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerSendCount;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerPack;

typedef struct _p_CellDMInfo *CellDMInfo;
struct _p_CellDMInfo {
  DM         dm;         // The cell DM
  PetscInt   Nf;         // The number of DM fields
  char     **dmFields;   // Swarm fields defining this DM
  char      *coordField; // Swarm field for coordinates on this DM
  CellDMInfo next;       // Next struct down in the stack
};

/*
 Error checking to ensure the swarm type is correct and that a cell DM has been set
*/
#define DMSWARMPICVALID(obj) \
  do { \
    DM_Swarm *_swarm = (DM_Swarm *)(obj)->data; \
    PetscCheck(_swarm->swarm_type == DMSWARM_PIC, PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Valid only for DMSwarm-PIC. You must call DMSwarmSetType(dm,DMSWARM_PIC)"); \
    PetscCheck(_swarm->cellinfo && _swarm->cellinfo[0].dm, PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Valid only for DMSwarmPIC if the cell DM is set. You must call DMSwarmSetCellDM() or DMSwarmPushCellDM()"); \
  } while (0)

typedef struct {
  DMSwarmDataBucket db;
  PetscInt          refct;
  PetscBool         field_registration_initialized;
  PetscBool         field_registration_finalized;
  /* DMSwarmProjectMethod *swarm_project;*/ /* swarm, geometry, result */

  /* PetscInt overlap; */
  /* PetscErrorCode (*update_overlap)(void); */

  const char **vec_field_names;
  PetscInt     vec_field_num;
  PetscInt     vec_field_bs, vec_field_nlocal;
  const char  *coord_name;

  PetscBool          issetup;
  DMSwarmType        swarm_type;
  DMSwarmMigrateType migrate_type;
  DMSwarmCollectType collect_type;

  CellDMInfo cellinfo; // The stack of cell DMs

  PetscBool migrate_error_on_missing_point;

  PetscBool   collect_view_active;
  PetscInt    collect_view_reset_nlocal;
  DMSwarmSort sort_context;

  /* Support for PIC */
  PetscInt Ns; /* The number of particle species */

  PetscSimplePointFn *coordFunc; /* Function to set particle coordinates */
  PetscSimplePointFn *velFunc;   /* Function to set particle velocities */

  /* Debugging */
  PetscInt printCoords;
  PetscInt printWeights;
} DM_Swarm;

typedef struct {
  PetscInt point_index;
  PetscInt cell_index;
} SwarmPoint;

struct _p_DMSwarmSort {
  PetscBool   isvalid;
  PetscInt    ncells, npoints;
  PetscInt   *pcell_offsets;
  SwarmPoint *list;
};

PETSC_INTERN PetscErrorCode DMSwarmMigrate_Push_Basic(DM, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMigrate_CellDMScatter(DM, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMigrate_CellDMExact(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMSwarmReplace_Internal(DM, DM *);

/* DMSwarmDataBucket */
typedef enum {
  DATABUCKET_VIEW_STDOUT = 0,
  DATABUCKET_VIEW_ASCII,
  DATABUCKET_VIEW_BINARY,
  DATABUCKET_VIEW_HDF5
} DMSwarmDataBucketViewType;

struct _p_DMSwarmDataField {
  char         *registration_function;
  PetscInt      L, bs;
  PetscBool     active;
  size_t        atomic_size;
  char         *name; /* what are they called */
  void         *data; /* the data - an array of structs */
  PetscDataType petsc_type;
};

struct _p_DMSwarmDataBucket {
  PetscInt          L;         /* number in use */
  PetscInt          buffer;    /* memory buffer used for re-allocation */
  PetscInt          allocated; /* number allocated, this will equal datafield->L */
  PetscBool         finalised; /* DEPRECATED */
  PetscInt          nfields;   /* how many fields of this type */
  DMSwarmDataField *field;     /* the data */
};

#define DMSWARM_DATAFIELD_point_access(data, index, atomic_size)                (void *)((char *)(data) + (index) * (atomic_size))
#define DMSWARM_DATAFIELD_point_access_offset(data, index, atomic_size, offset) (void *)((char *)(data) + (index) * (atomic_size) + (offset))

PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringInList(const char[], const PetscInt, const DMSwarmDataField[], PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringFindInList(const char[], const PetscInt, const DMSwarmDataField[], PetscInt *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldCreate(const char[], const char[], const size_t, const PetscInt, DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldDestroy(DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreate(DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroy(DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketQueryForActiveFields(DMSwarmDataBucket, PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRegisterField(DMSwarmDataBucket, const char[], const char[], size_t, DMSwarmDataField *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetNumEntries(DMSwarmDataField, PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetBlockSize(DMSwarmDataField, PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetSize(DMSwarmDataField, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroBlock(DMSwarmDataField, const PetscInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAccess(const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPoint(const DMSwarmDataField, const PetscInt, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPointOffset(const DMSwarmDataField, const size_t, const PetscInt, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldRestoreAccess(DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldVerifyAccess(const DMSwarmDataField, const size_t);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAtomicSize(const DMSwarmDataField, size_t *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldInsertPoint(const DMSwarmDataField, const PetscInt, const void *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt, const DMSwarmDataField, const PetscInt, const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroPoint(const DMSwarmDataField, const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketFinalize(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetInitialSizes(DMSwarmDataBucket, const PetscInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetSizes(DMSwarmDataBucket, const PetscInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetSizes(DMSwarmDataBucket, PetscInt *, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetGlobalSizes(MPI_Comm, DMSwarmDataBucket, PetscInt *, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFields(DMSwarmDataBucket, PetscInt *, DMSwarmDataField *[]);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketCopyPoint(const DMSwarmDataBucket, const PetscInt, const DMSwarmDataBucket, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreateFromSubset(DMSwarmDataBucket, const PetscInt, const PetscInt[], DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketZeroPoint(const DMSwarmDataBucket, const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketView(MPI_Comm, DMSwarmDataBucket, const char[], DMSwarmDataBucketViewType);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketAddPoint(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePoint(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePointAtIndex(const DMSwarmDataBucket, const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketDuplicateFields(DMSwarmDataBucket, DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertValues(DMSwarmDataBucket, DMSwarmDataBucket);

/* helpers for parallel send/recv */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreatePackedArray(DMSwarmDataBucket, size_t *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroyPackedArray(DMSwarmDataBucket, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketFillPackedArray(DMSwarmDataBucket, const PetscInt, void *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertPackedArray(DMSwarmDataBucket, const PetscInt, void *);

/* DMSwarmDataEx */
typedef enum {
  DEOBJECT_INITIALIZED = 0,
  DEOBJECT_FINALIZED,
  DEOBJECT_STATE_UNKNOWN
} DMSwarmDEObjectState;

typedef struct _p_DMSwarmDataEx *DMSwarmDataEx;
struct _p_DMSwarmDataEx {
  PetscInt             instance;
  MPI_Comm             comm;
  PetscMPIInt          rank;
  PetscMPIInt          n_neighbour_procs;
  PetscMPIInt         *neighbour_procs;          /* [n_neighbour_procs] */
  PetscInt            *messages_to_be_sent;      /* [n_neighbour_procs] */
  PetscInt            *message_offsets;          /* [n_neighbour_procs] */
  PetscInt            *messages_to_be_recvieved; /* [n_neighbour_procs] */
  size_t               unit_message_size;
  void                *send_message;
  PetscInt             send_message_length;
  void                *recv_message;
  PetscInt             recv_message_length;
  PetscMPIInt         *send_tags, *recv_tags;
  PetscInt             total_pack_cnt;
  PetscInt            *pack_cnt; /* [n_neighbour_procs] */
  DMSwarmDEObjectState topology_status;
  DMSwarmDEObjectState message_lengths_status;
  DMSwarmDEObjectState packer_status;
  DMSwarmDEObjectState communication_status;
  MPI_Status          *_stats;
  MPI_Request         *_requests;
};

extern const char *status_names[];

PETSC_INTERN PetscErrorCode DMSwarmDataExCreate(MPI_Comm, const PetscInt, DMSwarmDataEx *);
PETSC_INTERN PetscErrorCode DMSwarmDataExView(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExDestroy(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyInitialize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyAddNeighbour(DMSwarmDataEx, const PetscMPIInt);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyFinalize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExInitializeSendCount(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExAddToSendCount(DMSwarmDataEx, const PetscMPIInt, const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataExFinalizeSendCount(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackInitialize(DMSwarmDataEx, size_t);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackData(DMSwarmDataEx, PetscMPIInt, PetscInt, void *);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackFinalize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExBegin(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExEnd(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetSendData(DMSwarmDataEx, PetscInt *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetRecvData(DMSwarmDataEx, PetscInt *, void **);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyGetNeighbours(DMSwarmDataEx, PetscMPIInt *, PetscMPIInt *[]);
