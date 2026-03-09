#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

#include <netcdf.h>
#include <exodusII.h>

#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerexodusiiimpl.h>

PetscErrorCode DMView_PlexExodusII(DM dm, PetscViewer viewer)
{
  enum ElemType {
    SEGMENT,
    TRI,
    QUAD,
    TET,
    HEX
  };
  MPI_Comm comm;
  PetscInt degree; /* the order of the mesh */
  /* Connectivity Variables */
  PetscInt cellsNotInConnectivity;
  /* Cell Sets */
  DMLabel         csLabel;
  IS              csIS;
  const PetscInt *csIdx;
  PetscInt        num_cs, cs;
  enum ElemType  *type;
  PetscBool       hasLabel;
  /* Coordinate Variables */
  DM                 cdm;
  PetscSection       coordSection;
  Vec                coord;
  PetscInt         **nodes;
  PetscInt           depth, d, dim, skipCells = 0;
  PetscInt           pStart, pEnd, p, cStart, cEnd, numCells, vStart, vEnd, numVertices, eStart, eEnd, numEdges, fStart, fEnd, numFaces, numNodes;
  PetscInt           num_vs, num_fs;
  PetscMPIInt        rank, size;
  const char        *dmName;
  PetscInt           nodesLineP1[4] = {2, 0, 0, 0};
  PetscInt           nodesLineP2[4] = {2, 0, 0, 1};
  PetscInt           nodesTriP1[4]  = {3, 0, 0, 0};
  PetscInt           nodesTriP2[4]  = {3, 3, 0, 0};
  PetscInt           nodesQuadP1[4] = {4, 0, 0, 0};
  PetscInt           nodesQuadP2[4] = {4, 4, 0, 1};
  PetscInt           nodesTetP1[4]  = {4, 0, 0, 0};
  PetscInt           nodesTetP2[4]  = {4, 6, 0, 0};
  PetscInt           nodesHexP1[4]  = {8, 0, 0, 0};
  PetscInt           nodesHexP2[4]  = {8, 12, 6, 1};
  PetscExodusIIInt   CPU_word_size, IO_word_size, EXO_mode;
  PetscExodusIIFloat EXO_version;

  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /*
    Creating coordSection is a collective operation so we do it somewhat out of sequence
  */
  PetscCall(PetscSectionCreate(comm, &coordSection));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  /*
    Check that all points are on rank 0 since we don't know how to save distributed DM in exodus format
  */
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  numCells    = cEnd - cStart;
  numEdges    = eEnd - eStart;
  numVertices = vEnd - vStart;
  PetscCheck(!(rank && (numCells || numEdges || numVertices)), PETSC_COMM_SELF, PETSC_ERR_SUP, "Writing distributed DM in ExodusII format not supported");
  if (rank == 0) {
    switch (exo->btype) {
    case FILE_MODE_READ:
    case FILE_MODE_APPEND:
    case FILE_MODE_UPDATE:
    case FILE_MODE_APPEND_UPDATE:
      /* ExodusII does not allow writing geometry to an existing file */
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "cannot add geometry to existing file %s", exo->filename);
    case FILE_MODE_WRITE:
      /* Create an empty file if one already exists*/
      EXO_mode = EX_CLOBBER;
#if defined(PETSC_USE_64BIT_INDICES)
      EXO_mode += EX_ALL_INT64_API;
#endif
      CPU_word_size = sizeof(PetscReal);
      IO_word_size  = sizeof(PetscReal);
      exo->exoid    = ex_create(exo->filename, EXO_mode, &CPU_word_size, &IO_word_size);
      PetscCheck(exo->exoid >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_create failed for %s", exo->filename);

      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
    }

    /* --- Get DM info --- */
    PetscCall(PetscObjectGetName((PetscObject)dm, &dmName));
    PetscCall(DMPlexGetDepth(dm, &depth));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    if (depth == 3) {
      numFaces = fEnd - fStart;
    } else {
      numFaces = 0;
    }
    PetscCall(DMGetLabelSize(dm, "Cell Sets", &num_cs));
    PetscCall(DMGetLabelSize(dm, "Vertex Sets", &num_vs));
    PetscCall(DMGetLabelSize(dm, "Face Sets", &num_fs));
    PetscCall(DMGetCoordinatesLocal(dm, &coord));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    if (num_cs > 0) {
      PetscCall(DMGetLabel(dm, "Cell Sets", &csLabel));
      PetscCall(DMLabelGetValueIS(csLabel, &csIS));
      PetscCall(ISGetIndices(csIS, &csIdx));
    }
    PetscCall(PetscMalloc1(num_cs, &nodes));
    /* Set element type for each block and compute total number of nodes */
    PetscCall(PetscMalloc1(num_cs, &type));
    numNodes = numVertices;

    PetscCall(PetscViewerExodusIIGetOrder(viewer, &degree));
    if (degree == 2) numNodes += numEdges;
    cellsNotInConnectivity = numCells;
    for (cs = 0; cs < num_cs; ++cs) {
      IS              stratumIS;
      const PetscInt *cells;
      PetscScalar    *xyz = NULL;
      PetscInt        csSize, closureSize;

      PetscCall(DMLabelGetStratumIS(csLabel, csIdx[cs], &stratumIS));
      PetscCall(ISGetIndices(stratumIS, &cells));
      PetscCall(ISGetSize(stratumIS, &csSize));
      PetscCall(DMPlexVecGetClosure(cdm, NULL, coord, cells[0], &closureSize, &xyz));
      switch (dim) {
      case 1:
        PetscCheck(closureSize == 2 * dim, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %" PetscInt_FMT " in dimension %" PetscInt_FMT " has no ExodusII type", closureSize / dim, dim);
        type[cs] = SEGMENT;
        break;
      case 2:
        if (closureSize == 3 * dim) {
          type[cs] = TRI;
        } else if (closureSize == 4 * dim) {
          type[cs] = QUAD;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %" PetscInt_FMT " in dimension %" PetscInt_FMT " has no ExodusII type", closureSize / dim, dim);
        break;
      case 3:
        if (closureSize == 4 * dim) {
          type[cs] = TET;
        } else if (closureSize == 8 * dim) {
          type[cs] = HEX;
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %" PetscInt_FMT " in dimension %" PetscInt_FMT " has no ExodusII type", closureSize / dim, dim);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not handled by ExodusII viewer", dim);
      }
      if ((degree == 2) && (type[cs] == SEGMENT)) numNodes += csSize;
      if ((degree == 2) && (type[cs] == QUAD)) numNodes += csSize;
      if ((degree == 2) && (type[cs] == HEX)) {
        numNodes += csSize;
        numNodes += numFaces;
      }
      PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coord, cells[0], &closureSize, &xyz));
      /* Set nodes and Element type */
      if (type[cs] == SEGMENT) {
        if (degree == 1) nodes[cs] = nodesLineP1;
        else if (degree == 2) nodes[cs] = nodesLineP2;
      } else if (type[cs] == TRI) {
        if (degree == 1) nodes[cs] = nodesTriP1;
        else if (degree == 2) nodes[cs] = nodesTriP2;
      } else if (type[cs] == QUAD) {
        if (degree == 1) nodes[cs] = nodesQuadP1;
        else if (degree == 2) nodes[cs] = nodesQuadP2;
      } else if (type[cs] == TET) {
        if (degree == 1) nodes[cs] = nodesTetP1;
        else if (degree == 2) nodes[cs] = nodesTetP2;
      } else if (type[cs] == HEX) {
        if (degree == 1) nodes[cs] = nodesHexP1;
        else if (degree == 2) nodes[cs] = nodesHexP2;
      }
      /* Compute the number of cells not in the connectivity table */
      cellsNotInConnectivity -= nodes[cs][3] * csSize;

      PetscCall(ISRestoreIndices(stratumIS, &cells));
      PetscCall(ISDestroy(&stratumIS));
    }
    if (num_cs) PetscCallExternal(ex_put_init, exo->exoid, dmName, dim, numNodes, numCells, num_cs, num_vs, num_fs);
    /* --- Connectivity --- */
    for (cs = 0; cs < num_cs; ++cs) {
      IS              stratumIS;
      const PetscInt *cells;
      PetscInt       *connect, off = 0;
      PetscInt        edgesInClosure = 0, facesInClosure = 0, verticesInClosure = 0;
      PetscInt        csSize, c, connectSize, closureSize;
      char           *elem_type        = NULL;
      char            elem_type_bar2[] = "BAR2", elem_type_bar3[] = "BAR3";
      char            elem_type_tri3[] = "TRI3", elem_type_quad4[] = "QUAD4";
      char            elem_type_tri6[] = "TRI6", elem_type_quad9[] = "QUAD9";
      char            elem_type_tet4[] = "TET4", elem_type_hex8[] = "HEX8";
      char            elem_type_tet10[] = "TET10", elem_type_hex27[] = "HEX27";

      PetscCall(DMLabelGetStratumIS(csLabel, csIdx[cs], &stratumIS));
      PetscCall(ISGetIndices(stratumIS, &cells));
      PetscCall(ISGetSize(stratumIS, &csSize));
      /* Set Element type */
      if (type[cs] == SEGMENT) {
        if (degree == 1) elem_type = elem_type_bar2;
        else if (degree == 2) elem_type = elem_type_bar3;
      } else if (type[cs] == TRI) {
        if (degree == 1) elem_type = elem_type_tri3;
        else if (degree == 2) elem_type = elem_type_tri6;
      } else if (type[cs] == QUAD) {
        if (degree == 1) elem_type = elem_type_quad4;
        else if (degree == 2) elem_type = elem_type_quad9;
      } else if (type[cs] == TET) {
        if (degree == 1) elem_type = elem_type_tet4;
        else if (degree == 2) elem_type = elem_type_tet10;
      } else if (type[cs] == HEX) {
        if (degree == 1) elem_type = elem_type_hex8;
        else if (degree == 2) elem_type = elem_type_hex27;
      }
      connectSize = nodes[cs][0] + nodes[cs][1] + nodes[cs][2] + nodes[cs][3];
      PetscCall(PetscMalloc1(PetscMax(27, connectSize) * csSize, &connect));
      PetscCallExternal(ex_put_block, exo->exoid, EX_ELEM_BLOCK, csIdx[cs], elem_type, csSize, connectSize, 0, 0, 1);
      /* Find number of vertices, edges, and faces in the closure */
      verticesInClosure = nodes[cs][0];
      if (depth > 1) {
        if (dim == 2) {
          PetscCall(DMPlexGetConeSize(dm, cells[0], &edgesInClosure));
        } else if (dim == 3) {
          PetscInt *closure = NULL;

          PetscCall(DMPlexGetConeSize(dm, cells[0], &facesInClosure));
          PetscCall(DMPlexGetTransitiveClosure(dm, cells[0], PETSC_TRUE, &closureSize, &closure));
          edgesInClosure = closureSize - facesInClosure - 1 - verticesInClosure;
          PetscCall(DMPlexRestoreTransitiveClosure(dm, cells[0], PETSC_TRUE, &closureSize, &closure));
        }
      }
      /* Get connectivity for each cell */
      for (c = 0; c < csSize; ++c) {
        PetscInt *closure = NULL;
        PetscInt  temp, i;

        PetscCall(DMPlexGetTransitiveClosure(dm, cells[c], PETSC_TRUE, &closureSize, &closure));
        for (i = 0; i < connectSize; ++i) {
          if (i < nodes[cs][0]) { /* Vertices */
            connect[i + off] = closure[(i + edgesInClosure + facesInClosure + 1) * 2] + 1;
            connect[i + off] -= cellsNotInConnectivity;
          } else if (i < nodes[cs][0] + nodes[cs][1]) { /* Edges */
            connect[i + off] = closure[(i - verticesInClosure + facesInClosure + 1) * 2] + 1;
            if (nodes[cs][2] == 0) connect[i + off] -= numFaces;
            connect[i + off] -= cellsNotInConnectivity;
          } else if (i < nodes[cs][0] + nodes[cs][1] + nodes[cs][3]) { /* Cells */
            connect[i + off] = closure[0] + 1;
            connect[i + off] -= skipCells;
          } else if (i < nodes[cs][0] + nodes[cs][1] + nodes[cs][3] + nodes[cs][2]) { /* Faces */
            connect[i + off] = closure[(i - edgesInClosure - verticesInClosure) * 2] + 1;
            connect[i + off] -= cellsNotInConnectivity;
          } else {
            connect[i + off] = -1;
          }
        }
        /* Tetrahedra are inverted */
        if (type[cs] == TET) {
          temp             = connect[0 + off];
          connect[0 + off] = connect[1 + off];
          connect[1 + off] = temp;
          if (degree == 2) {
            temp             = connect[5 + off];
            connect[5 + off] = connect[6 + off];
            connect[6 + off] = temp;
            temp             = connect[7 + off];
            connect[7 + off] = connect[8 + off];
            connect[8 + off] = temp;
          }
        }
        /* Hexahedra are inverted */
        if (type[cs] == HEX) {
          temp             = connect[1 + off];
          connect[1 + off] = connect[3 + off];
          connect[3 + off] = temp;
          if (degree == 2) {
            temp              = connect[8 + off];
            connect[8 + off]  = connect[11 + off];
            connect[11 + off] = temp;
            temp              = connect[9 + off];
            connect[9 + off]  = connect[10 + off];
            connect[10 + off] = temp;
            temp              = connect[16 + off];
            connect[16 + off] = connect[17 + off];
            connect[17 + off] = temp;
            temp              = connect[18 + off];
            connect[18 + off] = connect[19 + off];
            connect[19 + off] = temp;

            temp              = connect[12 + off];
            connect[12 + off] = connect[16 + off];
            connect[16 + off] = temp;
            temp              = connect[13 + off];
            connect[13 + off] = connect[17 + off];
            connect[17 + off] = temp;
            temp              = connect[14 + off];
            connect[14 + off] = connect[18 + off];
            connect[18 + off] = temp;
            temp              = connect[15 + off];
            connect[15 + off] = connect[19 + off];
            connect[19 + off] = temp;

            temp              = connect[23 + off];
            connect[23 + off] = connect[26 + off];
            connect[26 + off] = temp;
            temp              = connect[24 + off];
            connect[24 + off] = connect[25 + off];
            connect[25 + off] = temp;
            temp              = connect[25 + off];
            connect[25 + off] = connect[26 + off];
            connect[26 + off] = temp;
          }
        }
        off += connectSize;
        PetscCall(DMPlexRestoreTransitiveClosure(dm, cells[c], PETSC_TRUE, &closureSize, &closure));
      }
      PetscCallExternal(ex_put_conn, exo->exoid, EX_ELEM_BLOCK, csIdx[cs], connect, 0, 0);
      skipCells += (nodes[cs][3] == 0) * csSize;
      PetscCall(PetscFree(connect));
      PetscCall(ISRestoreIndices(stratumIS, &cells));
      PetscCall(ISDestroy(&stratumIS));
    }
    PetscCall(PetscFree(type));
    /* --- Coordinates --- */
    PetscCall(PetscSectionSetChart(coordSection, pStart, pEnd));
    if (num_cs) {
      for (d = 0; d < depth; ++d) {
        PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
        for (p = pStart; p < pEnd; ++p) PetscCall(PetscSectionSetDof(coordSection, p, nodes[0][d] > 0));
      }
    }
    for (cs = 0; cs < num_cs; ++cs) {
      IS              stratumIS;
      const PetscInt *cells;
      PetscInt        csSize, c;

      PetscCall(DMLabelGetStratumIS(csLabel, csIdx[cs], &stratumIS));
      PetscCall(ISGetIndices(stratumIS, &cells));
      PetscCall(ISGetSize(stratumIS, &csSize));
      for (c = 0; c < csSize; ++c) PetscCall(PetscSectionSetDof(coordSection, cells[c], nodes[cs][3] > 0));
      PetscCall(ISRestoreIndices(stratumIS, &cells));
      PetscCall(ISDestroy(&stratumIS));
    }
    if (num_cs) {
      PetscCall(ISRestoreIndices(csIS, &csIdx));
      PetscCall(ISDestroy(&csIS));
    }
    PetscCall(PetscFree(nodes));
    PetscCall(PetscSectionSetUp(coordSection));
    if (numNodes) {
      const char  *coordNames[3] = {"x", "y", "z"};
      PetscScalar *closure, *cval;
      PetscReal   *coords;
      PetscInt     hasDof, n = 0;

      /* There can't be more than 24 values in the closure of a point for the coord coordSection */
      PetscCall(PetscCalloc3(numNodes * 3, &coords, dim, &cval, 24, &closure));
      PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coord));
      PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
      for (p = pStart; p < pEnd; ++p) {
        PetscCall(PetscSectionGetDof(coordSection, p, &hasDof));
        if (hasDof) {
          PetscInt closureSize = 24, j;

          PetscCall(DMPlexVecGetClosure(cdm, NULL, coord, p, &closureSize, &closure));
          for (d = 0; d < dim; ++d) {
            cval[d] = 0.0;
            for (j = 0; j < closureSize / dim; j++) cval[d] += closure[j * dim + d];
            coords[d * numNodes + n] = PetscRealPart(cval[d]) * dim / closureSize;
          }
          ++n;
        }
      }
      PetscCallExternal(ex_put_coord, exo->exoid, &coords[0 * numNodes], &coords[1 * numNodes], &coords[2 * numNodes]);
      PetscCall(PetscFree3(coords, cval, closure));
      PetscCallExternal(ex_put_coord_names, exo->exoid, (char **)coordNames);
    }

    /* --- Node Sets/Vertex Sets --- */
    PetscCall(DMHasLabel(dm, "Vertex Sets", &hasLabel));
    if (hasLabel) {
      PetscInt        i, vs, vsSize;
      const PetscInt *vsIdx, *vertices;
      PetscInt       *nodeList;
      IS              vsIS, stratumIS;
      DMLabel         vsLabel;
      PetscCall(DMGetLabel(dm, "Vertex Sets", &vsLabel));
      PetscCall(DMLabelGetValueIS(vsLabel, &vsIS));
      PetscCall(ISGetIndices(vsIS, &vsIdx));
      for (vs = 0; vs < num_vs; ++vs) {
        PetscCall(DMLabelGetStratumIS(vsLabel, vsIdx[vs], &stratumIS));
        PetscCall(ISGetIndices(stratumIS, &vertices));
        PetscCall(ISGetSize(stratumIS, &vsSize));
        PetscCall(PetscMalloc1(vsSize, &nodeList));
        for (i = 0; i < vsSize; ++i) nodeList[i] = vertices[i] - skipCells + 1;
        PetscCallExternal(ex_put_set_param, exo->exoid, EX_NODE_SET, vsIdx[vs], vsSize, 0);
        PetscCallExternal(ex_put_set, exo->exoid, EX_NODE_SET, vsIdx[vs], nodeList, NULL);
        PetscCall(ISRestoreIndices(stratumIS, &vertices));
        PetscCall(ISDestroy(&stratumIS));
        PetscCall(PetscFree(nodeList));
      }
      PetscCall(ISRestoreIndices(vsIS, &vsIdx));
      PetscCall(ISDestroy(&vsIS));
    }
    /* --- Side Sets/Face Sets --- */
    PetscCall(DMHasLabel(dm, "Face Sets", &hasLabel));
    if (hasLabel) {
      PetscInt        i, j, fs, fsSize;
      const PetscInt *fsIdx, *faces;
      IS              fsIS, stratumIS;
      DMLabel         fsLabel;
      PetscInt        numPoints, *points;
      PetscInt        elem_list_size = 0;
      PetscInt       *elem_list, *elem_ind, *side_list;

      PetscCall(DMGetLabel(dm, "Face Sets", &fsLabel));
      /* Compute size of Node List and Element List */
      PetscCall(DMLabelGetValueIS(fsLabel, &fsIS));
      PetscCall(ISGetIndices(fsIS, &fsIdx));
      for (fs = 0; fs < num_fs; ++fs) {
        PetscCall(DMLabelGetStratumIS(fsLabel, fsIdx[fs], &stratumIS));
        PetscCall(ISGetSize(stratumIS, &fsSize));
        elem_list_size += fsSize;
        PetscCall(ISDestroy(&stratumIS));
      }
      if (num_fs) {
        PetscCall(PetscMalloc3(num_fs, &elem_ind, elem_list_size, &elem_list, elem_list_size, &side_list));
        elem_ind[0] = 0;
        for (fs = 0; fs < num_fs; ++fs) {
          PetscCall(DMLabelGetStratumIS(fsLabel, fsIdx[fs], &stratumIS));
          PetscCall(ISGetIndices(stratumIS, &faces));
          PetscCall(ISGetSize(stratumIS, &fsSize));
          /* Set Parameters */
          PetscCallExternal(ex_put_set_param, exo->exoid, EX_SIDE_SET, fsIdx[fs], fsSize, 0);
          /* Indices */
          if (fs < num_fs - 1) elem_ind[fs + 1] = elem_ind[fs] + fsSize;

          for (i = 0; i < fsSize; ++i) {
            /* Element List */
            points = NULL;
            PetscCall(DMPlexGetTransitiveClosure(dm, faces[i], PETSC_FALSE, &numPoints, &points));
            elem_list[elem_ind[fs] + i] = points[2] + 1;
            PetscCall(DMPlexRestoreTransitiveClosure(dm, faces[i], PETSC_FALSE, &numPoints, &points));

            /* Side List */
            points = NULL;
            PetscCall(DMPlexGetTransitiveClosure(dm, elem_list[elem_ind[fs] + i] - 1, PETSC_TRUE, &numPoints, &points));
            for (j = 1; j < numPoints; ++j) {
              if (points[j * 2] == faces[i]) break;
            }
            /* Convert HEX sides */
            if (numPoints == 27) {
              if (j == 1) {
                j = 5;
              } else if (j == 2) {
                j = 6;
              } else if (j == 3) {
                j = 1;
              } else if (j == 4) {
                j = 3;
              } else if (j == 5) {
                j = 2;
              } else if (j == 6) {
                j = 4;
              }
            }
            /* Convert TET sides */
            if (numPoints == 15) {
              --j;
              if (j == 0) j = 4;
            }
            side_list[elem_ind[fs] + i] = j;
            PetscCall(DMPlexRestoreTransitiveClosure(dm, elem_list[elem_ind[fs] + i] - 1, PETSC_TRUE, &numPoints, &points));
          }
          PetscCall(ISRestoreIndices(stratumIS, &faces));
          PetscCall(ISDestroy(&stratumIS));
        }
        PetscCall(ISRestoreIndices(fsIS, &fsIdx));
        PetscCall(ISDestroy(&fsIS));

        /* Put side sets */
        for (fs = 0; fs < num_fs; ++fs) PetscCallExternal(ex_put_set, exo->exoid, EX_SIDE_SET, fsIdx[fs], &elem_list[elem_ind[fs]], &side_list[elem_ind[fs]]);
        PetscCall(PetscFree3(elem_ind, elem_list, side_list));
      }
    }
    /*
      close the exodus file
    */
    ex_close(exo->exoid);
    exo->exoid = -1;
  }
  PetscCall(PetscSectionDestroy(&coordSection));

  /*
    reopen the file in parallel
  */
  EXO_mode = EX_WRITE;
#if defined(PETSC_USE_64BIT_INDICES)
  EXO_mode += EX_ALL_INT64_API;
#endif
  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);
  exo->exoid    = ex_open_par(exo->filename, EXO_mode, &CPU_word_size, &IO_word_size, &EXO_version, comm, MPI_INFO_NULL);
  PetscCheck(exo->exoid >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open_par failed for %s %d", exo->filename, exo->exoid);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecViewPlex_ExodusII_Nodal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset);
static PetscErrorCode VecViewPlex_ExodusII_Zonal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset);
static PetscErrorCode VecLoadPlex_ExodusII_Nodal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset);
static PetscErrorCode VecLoadPlex_ExodusII_Zonal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset);

PetscErrorCode VecView_PlexExodusII_Internal(Vec v, PetscViewer viewer)
{
  DM          dm;
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscExodusIIInt exoid, offsetN = -1, offsetZ = -1;
  const char      *vecname;
  PetscInt         step;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(PetscObjectGetName((PetscObject)v, &vecname));

  PetscCall(DMGetOutputSequenceNumber(dm, &step, NULL));
  PetscCall(PetscViewerExodusIIGetNodalVariableIndex(viewer, vecname, &offsetN));
  PetscCall(PetscViewerExodusIIGetZonalVariableIndex(viewer, vecname, &offsetZ));
  PetscCheck(!(offsetN >= 0 && offsetZ >= 0), comm, PETSC_ERR_FILE_UNEXPECTED, "Found both nodal and zonal variable %s in exodus file. ", vecname);
  if (offsetN >= 0) {
    PetscCall(VecViewPlex_ExodusII_Nodal_Internal(v, exoid, (PetscExodusIIInt)step + 1, offsetN + 1));
  } else if (offsetZ >= 0) {
    PetscCall(VecViewPlex_ExodusII_Zonal_Internal(v, exoid, (PetscExodusIIInt)step + 1, offsetZ + 1));
  } else SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Could not find nodal or zonal variable %s in exodus file. ", vecname);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecLoad_PlexExodusII_Internal(Vec v, PetscViewer viewer)
{
  DM          dm;
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscExodusIIInt exoid, offsetN = 0, offsetZ = 0;
  const char      *vecname;
  PetscInt         step;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerExodusIIGetId(viewer, &exoid));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(PetscObjectGetName((PetscObject)v, &vecname));

  PetscCall(DMGetOutputSequenceNumber(dm, &step, NULL));
  PetscCall(PetscViewerExodusIIGetNodalVariableIndex(viewer, vecname, &offsetN));
  PetscCall(PetscViewerExodusIIGetZonalVariableIndex(viewer, vecname, &offsetZ));
  PetscCheck(!(offsetN >= 0 && offsetZ >= 0), comm, PETSC_ERR_FILE_UNEXPECTED, "Found both nodal and zonal variable %s in exodus file. ", vecname);
  if (offsetN >= 0) PetscCall(VecLoadPlex_ExodusII_Nodal_Internal(v, exoid, (PetscExodusIIInt)step + 1, offsetN + 1));
  else if (offsetZ >= 0) PetscCall(VecLoadPlex_ExodusII_Zonal_Internal(v, exoid, (PetscExodusIIInt)step + 1, offsetZ + 1));
  else SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Could not find nodal or zonal variable %s in exodus file. ", vecname);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecViewPlex_ExodusII_Nodal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset)
{
  MPI_Comm           comm;
  PetscMPIInt        size;
  DM                 dm;
  Vec                vNatural, vComp;
  const PetscScalar *varray;
  PetscInt           xs, xe, bs;
  PetscBool          useNatural;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetUseNatural(dm, &useNatural));
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) {
    PetscCall(DMPlexCreateNaturalVector(dm, &vNatural));
    PetscCall(DMPlexGlobalToNaturalBegin(dm, v, vNatural));
    PetscCall(DMPlexGlobalToNaturalEnd(dm, v, vNatural));
  } else {
    vNatural = v;
  }

  /* Write local chunk of the result in the exodus file
     exodus stores each component of a vector-valued field as a separate variable.
     We assume that they are stored sequentially */
  PetscCall(VecGetOwnershipRange(vNatural, &xs, &xe));
  PetscCall(VecGetBlockSize(vNatural, &bs));
  if (bs == 1) {
    PetscCall(VecGetArrayRead(vNatural, &varray));
    PetscCallExternal(ex_put_partial_var, exoid, step, EX_NODAL, offset, 1, xs + 1, xe - xs, varray);
    PetscCall(VecRestoreArrayRead(vNatural, &varray));
  } else {
    IS       compIS;
    PetscInt c;

    PetscCall(ISCreateStride(comm, (xe - xs) / bs, xs, bs, &compIS));
    for (c = 0; c < bs; ++c) {
      PetscCall(ISStrideSetStride(compIS, (xe - xs) / bs, xs + c, bs));
      PetscCall(VecGetSubVector(vNatural, compIS, &vComp));
      PetscCall(VecGetArrayRead(vComp, &varray));
      PetscCallExternal(ex_put_partial_var, exoid, step, EX_NODAL, offset + c, 1, xs / bs + 1, (xe - xs) / bs, varray);
      PetscCall(VecRestoreArrayRead(vComp, &varray));
      PetscCall(VecRestoreSubVector(vNatural, compIS, &vComp));
    }
    PetscCall(ISDestroy(&compIS));
  }
  if (useNatural) PetscCall(VecDestroy(&vNatural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecLoadPlex_ExodusII_Nodal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset)
{
  MPI_Comm     comm;
  PetscMPIInt  size;
  DM           dm;
  Vec          vNatural, vComp;
  PetscScalar *varray;
  PetscInt     xs, xe, bs;
  PetscBool    useNatural;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetUseNatural(dm, &useNatural));
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) PetscCall(DMPlexCreateNaturalVector(dm, &vNatural));
  else vNatural = v;

  /* Read local chunk from the file */
  PetscCall(VecGetOwnershipRange(vNatural, &xs, &xe));
  PetscCall(VecGetBlockSize(vNatural, &bs));
  if (bs == 1) {
    PetscCall(VecGetArray(vNatural, &varray));
    PetscCallExternal(ex_get_partial_var, exoid, step, EX_NODAL, offset, 1, xs + 1, xe - xs, varray);
    PetscCall(VecRestoreArray(vNatural, &varray));
  } else {
    IS       compIS;
    PetscInt c;

    PetscCall(ISCreateStride(comm, (xe - xs) / bs, xs, bs, &compIS));
    for (c = 0; c < bs; ++c) {
      PetscCall(ISStrideSetStride(compIS, (xe - xs) / bs, xs + c, bs));
      PetscCall(VecGetSubVector(vNatural, compIS, &vComp));
      PetscCall(VecGetArray(vComp, &varray));
      PetscCallExternal(ex_get_partial_var, exoid, step, EX_NODAL, offset + c, 1, xs / bs + 1, (xe - xs) / bs, varray);
      PetscCall(VecRestoreArray(vComp, &varray));
      PetscCall(VecRestoreSubVector(vNatural, compIS, &vComp));
    }
    PetscCall(ISDestroy(&compIS));
  }
  if (useNatural) {
    PetscCall(DMPlexNaturalToGlobalBegin(dm, vNatural, v));
    PetscCall(DMPlexNaturalToGlobalEnd(dm, vNatural, v));
    PetscCall(VecDestroy(&vNatural));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecViewPlex_ExodusII_Zonal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset)
{
  MPI_Comm           comm;
  PetscMPIInt        size;
  DM                 dm;
  Vec                vNatural, vComp;
  const PetscScalar *varray;
  PetscInt           xs, xe, bs;
  PetscBool          useNatural;
  IS                 compIS;
  PetscInt          *csSize, *csID;
  PetscExodusIIInt   numCS, set, csxs = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetUseNatural(dm, &useNatural));
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) {
    PetscCall(DMPlexCreateNaturalVector(dm, &vNatural));
    PetscCall(DMPlexGlobalToNaturalBegin(dm, v, vNatural));
    PetscCall(DMPlexGlobalToNaturalEnd(dm, v, vNatural));
  } else {
    vNatural = v;
  }

  /* Write local chunk of the result in the exodus file
     exodus stores each component of a vector-valued field as a separate variable.
     We assume that they are stored sequentially
     Zonal variables are accessed one element block at a time, so we loop through the cell sets,
     but once the vector has been reordered to natural size, we cannot use the label information
     to figure out what to save where. */
  numCS = (PetscExodusIIInt)ex_inquire_int(exoid, EX_INQ_ELEM_BLK); // This is an int64_t
  PetscCall(PetscMalloc2(numCS, &csID, numCS, &csSize));
  PetscCallExternal(ex_get_ids, exoid, EX_ELEM_BLOCK, csID);
  for (set = 0; set < numCS; ++set) {
    ex_block block;

    block.id   = csID[set];
    block.type = EX_ELEM_BLOCK;
    PetscCallExternal(ex_get_block_param, exoid, &block);
    PetscCall(PetscIntCast(block.num_entry, &csSize[set])); // This is an int64_t
  }
  PetscCall(VecGetOwnershipRange(vNatural, &xs, &xe));
  PetscCall(VecGetBlockSize(vNatural, &bs));
  if (bs > 1) PetscCall(ISCreateStride(comm, (xe - xs) / bs, xs, bs, &compIS));
  for (set = 0; set < numCS; set++) {
    PetscInt csLocalSize, c;

    /* range of indices for set setID[set]: csxs:csxs + csSize[set]-1
       local slice of zonal values:         xs/bs,xm/bs-1
       intersection:                        max(xs/bs,csxs),min(xm/bs-1,csxs + csSize[set]-1) */
    csLocalSize = PetscMax(0, PetscMin(xe / bs, csxs + csSize[set]) - PetscMax(xs / bs, csxs));
    if (bs == 1) {
      PetscCall(VecGetArrayRead(vNatural, &varray));
      PetscCallExternal(ex_put_partial_var, exoid, step, EX_ELEM_BLOCK, offset, csID[set], PetscMax(xs - csxs, 0) + 1, csLocalSize, &varray[PetscMax(0, csxs - xs)]);
      PetscCall(VecRestoreArrayRead(vNatural, &varray));
    } else {
      for (c = 0; c < bs; ++c) {
        PetscCall(ISStrideSetStride(compIS, (xe - xs) / bs, xs + c, bs));
        PetscCall(VecGetSubVector(vNatural, compIS, &vComp));
        PetscCall(VecGetArrayRead(vComp, &varray));
        PetscCallExternal(ex_put_partial_var, exoid, step, EX_ELEM_BLOCK, offset + c, csID[set], PetscMax(xs / bs - csxs, 0) + 1, csLocalSize, &varray[PetscMax(0, csxs - xs / bs)]);
        PetscCall(VecRestoreArrayRead(vComp, &varray));
        PetscCall(VecRestoreSubVector(vNatural, compIS, &vComp));
      }
    }
    csxs += csSize[set];
  }
  PetscCall(PetscFree2(csID, csSize));
  if (bs > 1) PetscCall(ISDestroy(&compIS));
  if (useNatural) PetscCall(VecDestroy(&vNatural));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecLoadPlex_ExodusII_Zonal_Internal(Vec v, PetscExodusIIInt exoid, PetscExodusIIInt step, PetscExodusIIInt offset)
{
  MPI_Comm         comm;
  PetscMPIInt      size;
  DM               dm;
  Vec              vNatural, vComp;
  PetscScalar     *varray;
  PetscInt         xs, xe, bs;
  PetscBool        useNatural;
  IS               compIS;
  PetscInt        *csSize, *csID;
  PetscExodusIIInt numCS, set, csxs = 0;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)v, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecGetDM(v, &dm));
  PetscCall(DMGetUseNatural(dm, &useNatural));
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) PetscCall(DMPlexCreateNaturalVector(dm, &vNatural));
  else vNatural = v;

  /* Read local chunk of the result in the exodus file
     exodus stores each component of a vector-valued field as a separate variable.
     We assume that they are stored sequentially
     Zonal variables are accessed one element block at a time, so we loop through the cell sets,
     but once the vector has been reordered to natural size, we cannot use the label information
     to figure out what to save where. */
  numCS = (PetscExodusIIInt)ex_inquire_int(exoid, EX_INQ_ELEM_BLK); // This is an int64_t
  PetscCall(PetscMalloc2(numCS, &csID, numCS, &csSize));
  PetscCallExternal(ex_get_ids, exoid, EX_ELEM_BLOCK, csID);
  for (set = 0; set < numCS; ++set) {
    ex_block block;

    block.id   = csID[set];
    block.type = EX_ELEM_BLOCK;
    PetscCallExternal(ex_get_block_param, exoid, &block);
    PetscCall(PetscIntCast(block.num_entry, &csSize[set])); // This is an int64_t
  }
  PetscCall(VecGetOwnershipRange(vNatural, &xs, &xe));
  PetscCall(VecGetBlockSize(vNatural, &bs));
  if (bs > 1) PetscCall(ISCreateStride(comm, (xe - xs) / bs, xs, bs, &compIS));
  for (set = 0; set < numCS; ++set) {
    PetscInt csLocalSize, c;

    /* range of indices for set setID[set]: csxs:csxs + csSize[set]-1
       local slice of zonal values:         xs/bs,xm/bs-1
       intersection:                        max(xs/bs,csxs),min(xm/bs-1,csxs + csSize[set]-1) */
    csLocalSize = PetscMax(0, PetscMin(xe / bs, csxs + csSize[set]) - PetscMax(xs / bs, csxs));
    if (bs == 1) {
      PetscCall(VecGetArray(vNatural, &varray));
      PetscCallExternal(ex_get_partial_var, exoid, step, EX_ELEM_BLOCK, offset, csID[set], PetscMax(xs - csxs, 0) + 1, csLocalSize, &varray[PetscMax(0, csxs - xs)]);
      PetscCall(VecRestoreArray(vNatural, &varray));
    } else {
      for (c = 0; c < bs; ++c) {
        PetscCall(ISStrideSetStride(compIS, (xe - xs) / bs, xs + c, bs));
        PetscCall(VecGetSubVector(vNatural, compIS, &vComp));
        PetscCall(VecGetArray(vComp, &varray));
        PetscCallExternal(ex_get_partial_var, exoid, step, EX_ELEM_BLOCK, offset + c, csID[set], PetscMax(xs / bs - csxs, 0) + 1, csLocalSize, &varray[PetscMax(0, csxs - xs / bs)]);
        PetscCall(VecRestoreArray(vComp, &varray));
        PetscCall(VecRestoreSubVector(vNatural, compIS, &vComp));
      }
    }
    csxs += csSize[set];
  }
  PetscCall(PetscFree2(csID, csSize));
  if (bs > 1) PetscCall(ISDestroy(&compIS));
  if (useNatural) {
    PetscCall(DMPlexNaturalToGlobalBegin(dm, vNatural, v));
    PetscCall(DMPlexNaturalToGlobalEnd(dm, vNatural, v));
    PetscCall(VecDestroy(&vNatural));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ExodusGetCellType_Internal(const char *elem_type, DMPolytopeType *ct)
{
  PetscBool flg;

  PetscFunctionBegin;
  *ct = DM_POLYTOPE_UNKNOWN;
  PetscCall(PetscStrcmp(elem_type, "BAR2", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_SEGMENT;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "BAR3", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_SEGMENT;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "TRI", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_TRIANGLE;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "TRI3", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_TRIANGLE;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "QUAD", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_QUADRILATERAL;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "QUAD4", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_QUADRILATERAL;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "SHELL4", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_QUADRILATERAL;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "TETRA", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_TETRAHEDRON;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "TET4", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_TETRAHEDRON;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "WEDGE", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_TRI_PRISM;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "HEX", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_HEXAHEDRON;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "HEX8", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_HEXAHEDRON;
    goto done;
  }
  PetscCall(PetscStrcmp(elem_type, "HEXAHEDRON", &flg));
  if (flg) {
    *ct = DM_POLYTOPE_HEXAHEDRON;
    goto done;
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unrecognized element type %s", elem_type);
done:
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCreateExodus - Create a `DMPLEX` mesh from an ExodusII file ID.

  Collective

  Input Parameters:
+ comm        - The MPI communicator
. exoid       - The ExodusII id associated with a exodus file and obtained using ex_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm - The `DM` object representing the mesh

  Level: beginner

.seealso: [](ch_unstructured), `DM`, `PETSCVIEWEREXODUSII`, `DMPLEX`, `DMCreate()`
@*/
PetscErrorCode DMPlexCreateExodus(MPI_Comm comm, PetscExodusIIInt exoid, PetscBool interpolate, DM *dm)
{
  PetscMPIInt  num_proc, rank;
  DMLabel      cellSets = NULL, faceSets = NULL, vertSets = NULL;
  PetscSection coordSection;
  Vec          coordinates;
  PetscScalar *coords;
  PetscInt     coordSize, v;
  /* Read from ex_get_init() */
  char title[PETSC_MAX_PATH_LEN + 1];
  int  dim = 0, dimEmbed = 0, numVertices = 0, numCells = 0;
  int  num_cs = 0, num_vs = 0, num_fs = 0;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &num_proc));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  /* Open EXODUS II file and read basic information on rank 0, then broadcast to all processors */
  if (rank == 0) {
    PetscCall(PetscMemzero(title, PETSC_MAX_PATH_LEN + 1));
    PetscCallExternal(ex_get_init, exoid, title, &dimEmbed, &numVertices, &numCells, &num_cs, &num_vs, &num_fs);
    PetscCheck(num_cs, PETSC_COMM_SELF, PETSC_ERR_SUP, "Exodus file does not contain any cell set");
  }
  PetscCallMPI(MPI_Bcast(title, PETSC_MAX_PATH_LEN + 1, MPI_CHAR, 0, comm));
  PetscCallMPI(MPI_Bcast(&dim, 1, MPI_INT, 0, comm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, title));
  PetscCall(DMPlexSetChart(*dm, 0, numCells + numVertices));
  /*   We do not want this label automatically computed, instead we compute it here */
  PetscCall(DMCreateLabel(*dm, "celltype"));

  /* Read cell sets information */
  if (rank == 0) {
    PetscInt *cone;
    int       c, cs, ncs, c_loc, v, v_loc;
    /* Read from ex_get_elem_blk_ids() */
    int *cs_id, *cs_order;
    /* Read from ex_get_elem_block() */
    char buffer[PETSC_MAX_PATH_LEN + 1];
    int  num_cell_in_set, num_vertex_per_cell, num_hybrid, num_attr;
    /* Read from ex_get_elem_conn() */
    int *cs_connect;

    /* Get cell sets IDs */
    PetscCall(PetscMalloc2(num_cs, &cs_id, num_cs, &cs_order));
    PetscCallExternal(ex_get_ids, exoid, EX_ELEM_BLOCK, cs_id);
    /* Read the cell set connectivity table and build mesh topology
       EXO standard requires that cells in cell sets be numbered sequentially and be pairwise disjoint. */
    /* Check for a hybrid mesh */
    for (cs = 0, num_hybrid = 0; cs < num_cs; ++cs) {
      DMPolytopeType ct;
      char           elem_type[PETSC_MAX_PATH_LEN];

      PetscCall(PetscArrayzero(elem_type, sizeof(elem_type)));
      PetscCallExternal(ex_get_elem_type, exoid, cs_id[cs], elem_type);
      PetscCall(ExodusGetCellType_Internal(elem_type, &ct));
      dim = PetscMax(dim, DMPolytopeTypeGetDim(ct));
      PetscCallExternal(ex_get_block, exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, 0, 0, &num_attr);
      switch (ct) {
      case DM_POLYTOPE_TRI_PRISM:
        cs_order[cs] = cs;
        ++num_hybrid;
        break;
      default:
        for (c = cs; c > cs - num_hybrid; --c) cs_order[c] = cs_order[c - 1];
        cs_order[cs - num_hybrid] = cs;
      }
    }
    /* First set sizes */
    for (ncs = 0, c = 0; ncs < num_cs; ++ncs) {
      DMPolytopeType ct;
      char           elem_type[PETSC_MAX_PATH_LEN];
      const PetscInt cs = cs_order[ncs];

      PetscCall(PetscArrayzero(elem_type, sizeof(elem_type)));
      PetscCallExternal(ex_get_elem_type, exoid, cs_id[cs], elem_type);
      PetscCall(ExodusGetCellType_Internal(elem_type, &ct));
      PetscCallExternal(ex_get_block, exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, 0, 0, &num_attr);
      for (c_loc = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        PetscCall(DMPlexSetConeSize(*dm, c, num_vertex_per_cell));
        PetscCall(DMPlexSetCellType(*dm, c, ct));
      }
    }
    for (v = numCells; v < numCells + numVertices; ++v) PetscCall(DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT));
    PetscCall(DMSetUp(*dm));
    for (ncs = 0, c = 0; ncs < num_cs; ++ncs) {
      const PetscInt cs = cs_order[ncs];
      PetscCallExternal(ex_get_block, exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, 0, 0, &num_attr);
      PetscCall(PetscMalloc2(num_vertex_per_cell * num_cell_in_set, &cs_connect, num_vertex_per_cell, &cone));
      PetscCallExternal(ex_get_conn, exoid, EX_ELEM_BLOCK, cs_id[cs], cs_connect, NULL, NULL);
      /* EXO uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
      for (c_loc = 0, v = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        DMPolytopeType ct;

        for (v_loc = 0; v_loc < num_vertex_per_cell; ++v_loc, ++v) cone[v_loc] = cs_connect[v] + numCells - 1;
        PetscCall(DMPlexGetCellType(*dm, c, &ct));
        PetscCall(DMPlexInvertCell(ct, cone));
        PetscCall(DMPlexSetCone(*dm, c, cone));
        PetscCall(DMSetLabelValue_Fast(*dm, &cellSets, "Cell Sets", c, cs_id[cs]));
      }
      PetscCall(PetscFree2(cs_connect, cone));
    }
    PetscCall(PetscFree2(cs_id, cs_order));
  }
  {
    PetscInt ints[] = {dim, dimEmbed};

    PetscCallMPI(MPI_Bcast(ints, 2, MPIU_INT, 0, comm));
    PetscCall(DMSetDimension(*dm, ints[0]));
    PetscCall(DMSetCoordinateDim(*dm, ints[1]));
    dim      = ints[0];
    dimEmbed = ints[1];
  }
  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm = idm;
  }

  /* Create vertex set label */
  if (rank == 0 && (num_vs > 0)) {
    int vs, v;
    /* Read from ex_get_node_set_ids() */
    int *vs_id;
    /* Read from ex_get_node_set_param() */
    int num_vertex_in_set;
    /* Read from ex_get_node_set() */
    int *vs_vertex_list;

    /* Get vertex set ids */
    PetscCall(PetscMalloc1(num_vs, &vs_id));
    PetscCallExternal(ex_get_ids, exoid, EX_NODE_SET, vs_id);
    for (vs = 0; vs < num_vs; ++vs) {
      PetscCallExternal(ex_get_set_param, exoid, EX_NODE_SET, vs_id[vs], &num_vertex_in_set, NULL);
      PetscCall(PetscMalloc1(num_vertex_in_set, &vs_vertex_list));
      PetscCallExternal(ex_get_set, exoid, EX_NODE_SET, vs_id[vs], vs_vertex_list, NULL);
      for (v = 0; v < num_vertex_in_set; ++v) PetscCall(DMSetLabelValue_Fast(*dm, &vertSets, "Vertex Sets", vs_vertex_list[v] + numCells - 1, vs_id[vs]));
      PetscCall(PetscFree(vs_vertex_list));
    }
    PetscCall(PetscFree(vs_id));
  }
  /* Read coordinates */
  PetscCall(DMGetCoordinateSection(*dm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, dimEmbed));
  PetscCall(PetscSectionSetChart(coordSection, numCells, numCells + numVertices));
  for (v = numCells; v < numCells + numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, dimEmbed));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, dimEmbed));
  }
  PetscCall(PetscSectionSetUp(coordSection));
  PetscCall(PetscSectionGetStorageSize(coordSection, &coordSize));
  PetscCall(VecCreate(PETSC_COMM_SELF, &coordinates));
  PetscCall(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
  PetscCall(VecSetSizes(coordinates, coordSize, PETSC_DETERMINE));
  PetscCall(VecSetBlockSize(coordinates, dimEmbed));
  PetscCall(VecSetType(coordinates, VECSTANDARD));
  PetscCall(VecGetArray(coordinates, &coords));
  if (rank == 0) {
    PetscReal *x, *y, *z;

    PetscCall(PetscMalloc3(numVertices, &x, numVertices, &y, numVertices, &z));
    PetscCallExternal(ex_get_coord, exoid, x, y, z);
    if (dimEmbed > 0) {
      for (v = 0; v < numVertices; ++v) coords[v * dimEmbed + 0] = x[v];
    }
    if (dimEmbed > 1) {
      for (v = 0; v < numVertices; ++v) coords[v * dimEmbed + 1] = y[v];
    }
    if (dimEmbed > 2) {
      for (v = 0; v < numVertices; ++v) coords[v * dimEmbed + 2] = z[v];
    }
    PetscCall(PetscFree3(x, y, z));
  }
  PetscCall(VecRestoreArray(coordinates, &coords));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));

  /* Create side set label */
  if (rank == 0 && interpolate && (num_fs > 0)) {
    int fs, f, voff;
    /* Read from ex_get_side_set_ids() */
    int *fs_id;
    /* Read from ex_get_side_set_param() */
    int num_side_in_set;
    /* Read from ex_get_side_set_node_list() */
    int *fs_vertex_count_list, *fs_vertex_list, *fs_side_list;
    /* Read side set labels */
    char   fs_name[MAX_STR_LENGTH + 1];
    size_t fs_name_len;

    /* Get side set ids */
    PetscCall(PetscMalloc1(num_fs, &fs_id));
    PetscCallExternal(ex_get_ids, exoid, EX_SIDE_SET, fs_id);
    // Ids 1 and 2 are reserved by ExodusII for indicating things in 3D
    for (fs = 0; fs < num_fs; ++fs) {
      PetscCallExternal(ex_get_set_param, exoid, EX_SIDE_SET, fs_id[fs], &num_side_in_set, NULL);
      PetscCall(PetscMalloc3(num_side_in_set, &fs_vertex_count_list, num_side_in_set * 4, &fs_vertex_list, num_side_in_set, &fs_side_list));
      PetscCallExternal(ex_get_side_set_node_list, exoid, fs_id[fs], fs_vertex_count_list, fs_vertex_list);
      PetscCallExternal(ex_get_set, exoid, EX_SIDE_SET, fs_id[fs], NULL, fs_side_list);

      /* Get the specific name associated with this side set ID. */
      int fs_name_err = ex_get_name(exoid, EX_SIDE_SET, fs_id[fs], fs_name);
      if (!fs_name_err) {
        PetscCall(PetscStrlen(fs_name, &fs_name_len));
        if (fs_name_len == 0) PetscCall(PetscStrncpy(fs_name, "Face Sets", MAX_STR_LENGTH));
      }
      for (f = 0, voff = 0; f < num_side_in_set; ++f) {
        const PetscInt *faces    = NULL;
        PetscInt        faceSize = fs_vertex_count_list[f], numFaces;
        PetscInt        faceVertices[4], v;

        PetscCheck(faceSize <= 4, comm, PETSC_ERR_ARG_WRONG, "ExodusII side cannot have %" PetscInt_FMT " > 4 vertices", faceSize);
        for (v = 0; v < faceSize; ++v, ++voff) faceVertices[v] = fs_vertex_list[voff] + numCells - 1;
        PetscCall(DMPlexGetFullJoin(*dm, faceSize, faceVertices, &numFaces, &faces));
        PetscCheck(numFaces == 1, comm, PETSC_ERR_ARG_WRONG, "Invalid ExodusII side %d in set %d maps to %" PetscInt_FMT " faces", f, fs, numFaces);
        PetscCheck(dim == 1 || faces[0] >= numCells + numVertices, comm, PETSC_ERR_ARG_WRONG, "Invalid ExodusII side %d in set %d maps to point %" PetscInt_FMT " which is not a face", f, fs, faces[0]);
        PetscCall(DMSetLabelValue_Fast(*dm, &faceSets, "Face Sets", faces[0], fs_id[fs]));
        /* Only add the label if one has been detected for this side set. */
        if (!fs_name_err) PetscCall(DMSetLabelValue(*dm, fs_name, faces[0], fs_id[fs]));
        PetscCall(DMPlexRestoreJoin(*dm, faceSize, faceVertices, &numFaces, &faces));
      }
      PetscCall(PetscFree3(fs_vertex_count_list, fs_vertex_list, fs_side_list));
    }
    PetscCall(PetscFree(fs_id));
  }

  { /* Create Cell/Face/Vertex Sets labels at all processes */
    enum {
      n = 3
    };
    PetscBool flag[n];

    flag[0] = cellSets ? PETSC_TRUE : PETSC_FALSE;
    flag[1] = faceSets ? PETSC_TRUE : PETSC_FALSE;
    flag[2] = vertSets ? PETSC_TRUE : PETSC_FALSE;
    PetscCallMPI(MPI_Bcast(flag, n, MPI_C_BOOL, 0, comm));
    if (flag[0]) PetscCall(DMCreateLabel(*dm, "Cell Sets"));
    if (flag[1]) PetscCall(DMCreateLabel(*dm, "Face Sets"));
    if (flag[2]) PetscCall(DMCreateLabel(*dm, "Vertex Sets"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
