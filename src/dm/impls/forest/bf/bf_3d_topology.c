#include "bf_3d_topology.h"

#if defined(PETSC_HAVE_P4EST)

  /* convert to p8est for 3D domains */
  #if defined(PETSC_HAVE_MPIUNI)
    #undef MPI_SUCCESS
  #endif
  #if defined(PETSC_HAVE_P4EST)
    #include <p4est_to_p8est.h>
  #endif
  #if defined(PETSC_HAVE_MPIUNI)
    #define MPI_SUCCESS 0
  #endif

  /* rename generic functions that are the same for 2D and 3D */
  #define DMBF_XD_ConnectivityCreate  DMBF_3D_ConnectivityCreate
  #define DMBF_XD_ConnectivityDestroy DMBF_3D_ConnectivityDestroy

  #define DM_BF_XD_Topology    DM_BF_3D_Topology
  #define _p_DM_BF_XD_Topology _p_DM_BF_3D_Topology

  #define DMBF_XD_TopologyCreate          DMBF_3D_TopologyCreate
  #define DMBF_XD_TopologyDestroy         DMBF_3D_TopologyDestroy
  #define DMBF_XD_TopologyClone           DMBF_3D_TopologyClone
  #define DMBF_XD_TopologyGetConnectivity DMBF_3D_TopologyGetConnectivity

  /* include generic functions */
  #include "bf_xd_topology.h"

static PetscErrorCode DMBF_3D_ConnectivityCreate(DM dm, p4est_connectivity_t **connectivity)
{
  const char      *prefix;
  DMForestTopology topologyName;
  PetscBool        isBrick;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));

  /* get topology name */
  PetscCall(DMForestGetTopology(dm, &topologyName));
  PetscCheck(topologyName, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMBF needs a topology");
  PetscCall(PetscStrcmp((const char *)topologyName, "brick", &isBrick));

  if (isBrick && dm->setfromoptionscalled) { /* if brick topology with given uptions */
    PetscBool flgN, flgP, flgB, periodic = PETSC_FALSE;
    PetscInt  N[3] = {2, 2, 2}, P[3] = {0, 0, 0}, nretN = P4EST_DIM, nretP = P4EST_DIM, nretB = 2 * P4EST_DIM, i, j;
    PetscReal B[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0}, Lstart[3] = {0., 0., 0.}, L[3] = {-1.0, -1.0, -1.0}, maxCell[3] = {-1.0, -1.0, -1.0};

    /* get brick options */
    PetscCall(PetscOptionsGetIntArray(((PetscObject)dm)->options, prefix, "-dm_p4est_brick_size", N, &nretN, &flgN));
    PetscCall(PetscOptionsGetIntArray(((PetscObject)dm)->options, prefix, "-dm_p4est_brick_periodicity", P, &nretP, &flgP));
    PetscCall(PetscOptionsGetRealArray(((PetscObject)dm)->options, prefix, "-dm_p4est_brick_bounds", B, &nretB, &flgB));
    PetscCheck(!flgN || nretN == P4EST_DIM, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Need to give %d sizes in -dm_p4est_brick_size, gave %" PetscInt_FMT, P4EST_DIM, nretN);
    PetscCheck(!flgP || nretP == P4EST_DIM, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Need to give %d periodicities in -dm_p4est_brick_periodicity, gave %" PetscInt_FMT, P4EST_DIM, nretP);
    PetscCheck(!flgB || nretB == 2 * P4EST_DIM, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "Need to give %d bounds in -dm_p4est_brick_bounds, gave %" PetscInt_FMT, P4EST_DIM, nretP);

    /* update periodicity */
    for (i = 0; i < P4EST_DIM; i++) {
      P[i]     = (P[i] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE);
      periodic = (PetscBool)(P[i] || periodic);
      if (!flgB) B[2 * i + 1] = N[i];
      if (P[i]) {
        Lstart[i]  = B[2 * i + 0];
        L[i]       = B[2 * i + 1] - B[2 * i + 0];
        maxCell[i] = 1.1 * (L[i] / N[i]);
      }
    }
    if (periodic) PetscCall(DMSetPeriodicity(dm, maxCell, Lstart, L));

    /* create connectivity */
    PetscCallP4estReturn(*connectivity, p8est_connectivity_new_brick, ((int)N[0], (int)N[1], (int)N[2], (P[0] == DM_BOUNDARY_PERIODIC), (P[1] == DM_BOUNDARY_PERIODIC), (P[2] == DM_BOUNDARY_PERIODIC)));

    { /* scale to bounds */
      double *vertices = (*connectivity)->vertices;

      for (i = 0; i < 3 * (*connectivity)->num_vertices; i++) {
        j           = i % 3;
        vertices[i] = B[2 * j] + (vertices[i] / N[j]) * (B[2 * j + 1] - B[2 * j]);
      }
    }
  } else { /* otherwise call generic function */
    /* create connectivity */
    PetscCallP4estReturn(*connectivity, p8est_connectivity_new_byname, ((const char *)topologyName));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif /* defined(PETSC_HAVE_P4EST) */
