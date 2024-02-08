#if !defined(PETSCDMBF_3D_TOPOLOGY_H)
  #define PETSCDMBF_3D_TOPOLOGY_H

  #include <petscdmbf.h> /*I "petscdmbf.h" I*/

typedef struct _p_DM_BF_3D_Topology DM_BF_3D_Topology;

PETSC_EXTERN PetscErrorCode DMBF_3D_TopologyCreate(DM, DM_BF_3D_Topology **, PetscErrorCode (*)(DM, void *));
PETSC_EXTERN PetscErrorCode DMBF_3D_TopologyDestroy(DM, DM_BF_3D_Topology *);
PETSC_EXTERN PetscErrorCode DMBF_3D_TopologyClone(DM_BF_3D_Topology *, DM_BF_3D_Topology **, DM);

PETSC_EXTERN PetscErrorCode DMBF_3D_TopologyGetConnectivity(DM_BF_3D_Topology *, void *);

#endif /* defined(PETSCDMBF_3D_TOPOLOGY_H) */
