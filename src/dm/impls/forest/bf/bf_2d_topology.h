#if !defined(PETSCDMBF_2D_TOPOLOGY_H)
  #define PETSCDMBF_2D_TOPOLOGY_H

  #include <petscdmbf.h> /*I "petscdmbf.h" I*/

typedef struct _p_DM_BF_2D_Topology DM_BF_2D_Topology;

PetscErrorCode DMBF_2D_TopologyCreate(DM, DM_BF_2D_Topology **, PetscErrorCode (*)(DM, void *));
PetscErrorCode DMBF_2D_TopologyDestroy(DM, DM_BF_2D_Topology *);
PetscErrorCode DMBF_2D_TopologyClone(DM_BF_2D_Topology *, DM_BF_2D_Topology **, DM);

PetscErrorCode DMBF_2D_TopologyGetConnectivity(DM_BF_2D_Topology *, void *);

#endif /* defined(PETSCDMBF_2D_TOPOLOGY_H) */
