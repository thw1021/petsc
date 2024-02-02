#if defined(PETSC_HAVE_P4EST)

  #include "bf_xd.h"

/* default definitions, to be overwritten if this files is included */
  #if !defined(DM_BF_XD_Topology)
typedef struct _p_DM_BF_XD_Topology DM_BF_XD_Topology;
  #endif

/* declare "virtual" functions that need to be implemented */
static PetscErrorCode DMBF_XD_ConnectivityCreate(DM, p4est_connectivity_t **);

static PetscErrorCode DMBF_XD_ConnectivityDestroy(DM dm, p4est_connectivity_t *connectivity)
{
  PetscFunctionBegin;
  PetscCallP4est(p4est_connectivity_destroy, (connectivity));
  PetscFunctionReturn(0);
}

struct _p_DM_BF_XD_Topology {
  int                   refct;
  p4est_connectivity_t *connectivity;
};

PetscErrorCode DMBF_XD_TopologyCreate(DM dm, DM_BF_XD_Topology **topology, PetscErrorCode (*setUpUserFnAfterConnectivity)(DM, void *))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(topology);
  CHKERRQ(ierr);
  ierr = DMBF_XD_ConnectivityCreate(dm, &(*topology)->connectivity);
  CHKERRQ(ierr);
  if (setUpUserFnAfterConnectivity) {
    ierr = setUpUserFnAfterConnectivity(dm, (void *)(*topology)->connectivity);
    CHKERRQ(ierr);
  }
  (*topology)->refct = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_TopologyDestroy(DM dm, DM_BF_XD_Topology *topology)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  topology->refct -= 1;
  if (!topology->refct) {
    ierr = DMBF_XD_ConnectivityDestroy(dm, topology->connectivity);
    CHKERRQ(ierr);
    ierr = PetscFree(topology);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_TopologyClone(DM_BF_XD_Topology *srcTopology, DM_BF_XD_Topology **trgTopology, DM trgDm)
{
  PetscFunctionBegin;
  (*trgTopology) = srcTopology;
  (*trgTopology)->refct += 1;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBF_XD_TopologyGetConnectivity(DM_BF_XD_Topology *topology, void *connectivity)
{
  PetscFunctionBegin;
  *(void **)connectivity = topology->connectivity;
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
