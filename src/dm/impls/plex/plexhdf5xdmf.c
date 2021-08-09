#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsclayouthdf5.h>

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode SplitPath_Private(char path[], char name[])
{
  char *tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrrchr(path,'/',&tmp);CHKERRQ(ierr);
  ierr = PetscStrcpy(name,tmp);CHKERRQ(ierr);
  if (tmp != path) {
    /* '/' found, name is substring of path after last occurence of '/'. */
    /* Trim the '/name' part from path just by inserting null character. */
    tmp--;
    *tmp = '\0';
  } else {
    /* '/' not found, name = path, path = "/". */
    ierr = PetscStrcpy(path,"/");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  - invert (involute) cells of some types according to XDMF/VTK numbering of vertices in a cells
  - cell type is identified using the number of vertices
*/
static PetscErrorCode DMPlexInvertCells_XDMF_Private(DM dm)
{
  PetscInt       dim, *cones, cHeight, cStart, cEnd, p;
  PetscSection   cs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim != 3) PetscFunctionReturn(0);
  ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
  ierr = DMPlexGetConeSection(dm, &cs);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cHeight, &cStart, &cEnd);CHKERRQ(ierr);
  for (p=cStart; p<cEnd; p++) {
    PetscInt numCorners, o;

    ierr = PetscSectionGetDof(cs, p, &numCorners);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cs, p, &o);CHKERRQ(ierr);
    switch (numCorners) {
      case 4: ierr = DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON,&cones[o]);CHKERRQ(ierr); break;
      case 6: ierr = DMPlexInvertCell(DM_POLYTOPE_TRI_PRISM,&cones[o]);CHKERRQ(ierr); break;
      case 8: ierr = DMPlexInvertCell(DM_POLYTOPE_HEXAHEDRON,&cones[o]);CHKERRQ(ierr); break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLoad_HDF5_Xdmf_Internal(DM dm, PetscViewer viewer)
{
  Vec             coordinates;
  IS              cells;
  PetscInt        spatialDim, topoDim = -1;
  PetscMPIInt     rank;
  MPI_Comm        comm;
  PetscErrorCode  ierr;
  char            topo_path[PETSC_MAX_PATH_LEN]="/viz/topology/cells", topo_name[PETSC_MAX_PATH_LEN];
  char            geom_path[PETSC_MAX_PATH_LEN]="/geometry/vertices",  geom_name[PETSC_MAX_PATH_LEN];
  PetscBool       seq = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5/XDMF Loader Options","PetscViewer");CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_hdf5_topology_path","HDF5 path of topology dataset",NULL,topo_path,topo_path,sizeof(topo_path),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_plex_hdf5_geometry_path","HDF5 path to geometry dataset",NULL,geom_path,geom_path,sizeof(geom_path),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_plex_hdf5_force_sequential","force sequential loading",NULL,seq,&seq,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = SplitPath_Private(topo_path, topo_name);CHKERRQ(ierr);
  ierr = SplitPath_Private(geom_path, geom_name);CHKERRQ(ierr);
  ierr = PetscInfo2(dm, "Topology group %s, name %s\n", topo_path, topo_name);CHKERRQ(ierr);
  ierr = PetscInfo2(dm, "Geometry group %s, name %s\n", geom_path, geom_name);CHKERRQ(ierr);

  /* Read topology */
  ierr = PetscViewerHDF5PushGroup(viewer, topo_path);CHKERRQ(ierr);
  ierr = ISCreate(comm, &cells);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) cells, topo_name);CHKERRQ(ierr);
  if (seq) {
    PetscInt n;
    ierr = PetscViewerHDF5ReadSizes(viewer, topo_name, NULL, &n);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(cells->map, n);CHKERRQ(ierr);
    n = !rank ? n : 0;
    ierr = PetscLayoutSetLocalSize(cells->map, n);CHKERRQ(ierr);
  }
  ierr = ISLoad(cells, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, topo_name, "cell_dim", PETSC_INT, &topoDim, &topoDim);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  /* Read geometry */
  ierr = PetscViewerHDF5PushGroup(viewer, geom_path);CHKERRQ(ierr);
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, geom_name);CHKERRQ(ierr);
  if (seq) {
    PetscInt n;
    ierr = PetscViewerHDF5ReadSizes(viewer, geom_name, NULL, &n);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(coordinates->map, n);CHKERRQ(ierr);
    n = !rank ? n : 0;
    ierr = PetscLayoutSetLocalSize(coordinates->map, n);CHKERRQ(ierr);
  }
  ierr = VecLoad(coordinates, viewer);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &spatialDim);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  {
    PetscSF           sfVert = NULL;

    ierr = DMSetDimension(dm, topoDim < 0 ? spatialDim : topoDim);CHKERRQ(ierr);
    /* coordinates->map can be directly used as DMPlexTopologyBuild() shrinks its blocksize to 1 */
    ierr = DMPlexTopologyBuild(dm, cells, coordinates->map, &sfVert);CHKERRQ(ierr);
    ierr = DMPlexInvertCells_XDMF_Private(dm);CHKERRQ(ierr);
    ierr = DMPlexGeometryBuild(dm, coordinates, sfVert);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfVert);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&cells);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  /* scale coordinates - unlike in DMPlexLoad_HDF5_Internal, this can only be done after DM is populated */
  {
    PetscReal lengthScale;

    ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
    ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
    ierr = VecScale(coordinates, 1.0/lengthScale);CHKERRQ(ierr);
  }

  /* Read Labels */
  /* TODO: this probably does not work as elements get permuted */
  /* ierr = DMPlexLabelsLoad_HDF5_Internal(dm, viewer);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}
#endif
