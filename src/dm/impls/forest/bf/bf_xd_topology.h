#if defined(PETSC_HAVE_P4EST)

  #include "bf_xd.h"

/* default definitions, to be overwritten when this files is included */
  #if !defined(DM_BF_XD_Topology)
typedef struct _p_DM_BF_XD_Topology DM_BF_XD_Topology;
  #endif

/* declare "virtual" functions that need to be implemented */
static PetscErrorCode DMBF_XD_ConnectivityCreate(DM, p4est_connectivity_t **);

static PetscErrorCode DMBF_XD_ConnectivityDestroy(DM dm, p4est_connectivity_t *connectivity)
{
  PetscFunctionBegin;
  PetscCallP4est(p4est_connectivity_destroy, (connectivity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct _p_DM_BF_XD_Topology {
  int                   refct;
  p4est_connectivity_t *connectivity;
};

  #if !defined(DMBF_XD_TopologyCreate)
static
  #endif
  PetscErrorCode
  DMBF_XD_TopologyCreate(DM dm, DM_BF_XD_Topology **topology, PetscErrorCode (*setUpUserFnAfterConnectivity)(DM, void *))
{
  PetscFunctionBegin;
  PetscCall(PetscNew(topology));
  PetscCall(DMBF_XD_ConnectivityCreate(dm, &(*topology)->connectivity));
  if (setUpUserFnAfterConnectivity) { PetscCall(setUpUserFnAfterConnectivity(dm, (void *)(*topology)->connectivity)); }
  (*topology)->refct = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #if !defined(DMBF_XD_TopologyDestroy)
static
  #endif
  PetscErrorCode
  DMBF_XD_TopologyDestroy(DM dm, DM_BF_XD_Topology *topology)
{
  PetscFunctionBegin;
  topology->refct -= 1;
  if (!topology->refct) {
    PetscCall(DMBF_XD_ConnectivityDestroy(dm, topology->connectivity));
    PetscCall(PetscFree(topology));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #if !defined(DMBF_XD_TopologyClone)
static
  #endif
  PetscErrorCode
  DMBF_XD_TopologyClone(DM_BF_XD_Topology *srcTopology, DM_BF_XD_Topology **trgTopology, DM trgDm)
{
  PetscFunctionBegin;
  (*trgTopology) = srcTopology;
  (*trgTopology)->refct += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #if !defined(DMBF_XD_TopologyGetConnectivity)
static
  #endif
  PetscErrorCode
  DMBF_XD_TopologyGetConnectivity(DM_BF_XD_Topology *topology, void *connectivity)
{
  PetscFunctionBegin;
  *(void **)connectivity = topology->connectivity;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* defined(PETSC_HAVE_P4EST) */
