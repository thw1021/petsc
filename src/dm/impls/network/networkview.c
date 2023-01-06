#include <petsc/private/dmnetworkimpl.h> /*I  "petscdmnetwork.h"  I*/

static PetscErrorCode DMView_Network_CSV(DM dm, PetscViewer viewer)
{
  MPI_Comm        comm;
  PetscMPIInt     rank;
  DM              dmcoords;
  PetscInt        nsubnets, i, subnet, nvertices, nedges, vertex, edge;
  PetscInt        vertexOffsets[2], globalEdgeVertices[2];
  PetscScalar     vertexCoords[2];
  const PetscInt *vertices, *edges, *edgeVertices;
  Vec             allVertexCoords;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  // Get the network containing coordinate information
  PetscCall(DMGetCoordinateDM(dm, &dmcoords));
  // Get the coordinate vector for the network
  PetscCall(DMGetCoordinates(dm, &allVertexCoords));

  // Write the header
  PetscCall(PetscViewerASCIIPrintf(viewer, "Type,ID,X,Y,Z,Name,Color\n"));

  // Iterate each subnetwork (Note: We need to get the global number of subnets apparently)
  PetscCall(DMNetworkGetNumSubNetworks(dm, NULL, &nsubnets));
  for (subnet = 0; subnet < nsubnets; subnet++) {
    // Get the subnetwork's vertices and edges
    PetscCall(DMNetworkGetSubnetwork(dm, subnet, &nvertices, &nedges, &vertices, &edges));

    // Write out each vertex
    for (i = 0; i < nvertices; i++) {
      vertex = vertices[i];
      // Get the offset into the coordinate vector for the vertex
      PetscCall(DMNetworkGetLocalVecOffset(dmcoords, vertex, ALL_COMPONENTS, vertexOffsets));
      vertexOffsets[1] = vertexOffsets[0] + 1;
      // Remap vertex to the global value
      PetscCall(DMNetworkGetGlobalVertexIndex(dm, vertex, &vertex));
      // Get the vertex position from the coordinate vector
      PetscCall(VecGetValues(allVertexCoords, 2, vertexOffsets, vertexCoords));

      // TODO: Determine vertex color/name
      PetscCall(PetscViewerASCIIPrintf(viewer, "Node,%" PetscInt_FMT ",%lf,%lf,0,%" PetscInt_FMT "\n", vertex, (double)vertexCoords[0], (double)vertexCoords[1], vertex));
    }

    // Write out each edge
    for (i = 0; i < nedges; i++) {
      edge = edges[i];
      PetscCall(DMNetworkGetConnectedVertices(dm, edge, &edgeVertices));
      PetscCall(DMNetworkGetGlobalVertexIndex(dm, edgeVertices[0], &globalEdgeVertices[0]));
      PetscCall(DMNetworkGetGlobalVertexIndex(dm, edgeVertices[1], &globalEdgeVertices[1]));
      PetscCall(DMNetworkGetGlobalEdgeIndex(dm, edge, &edge));

      // TODO: Determine edge color/name
      PetscCall(PetscViewerASCIIPrintf(viewer, "Edge,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",0,%" PetscInt_FMT "\n", edge, globalEdgeVertices[0], globalEdgeVertices[1], edge));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Network_Matplotlib(DM dm)
{
  PetscMPIInt rank, size;
  MPI_Comm    comm;
  char        filename[FILENAME_MAX + 1], proccall[FILENAME_MAX + 500], scriptFile[FILENAME_MAX + 1];
  PetscViewer csvViewer;
  size_t      numChars, appendChars;

  PetscFunctionBegin;
  // Get the MPI communicator and this process' rank
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  // Acquire a temporary file to write to and open an ASCII/CSV viewer
  PetscCall(PetscStrcpy(filename, "/tmp/matplotlib-XXXXXX"));
  PetscCheck(mkstemp(filename) >= 0, comm, PETSC_ERR_SYS, "Could not acquire temporary file");
  // Note: We need to open with PETSC_COMM_SELF for each process to open a unique temporary file
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &csvViewer));
  PetscCall(PetscViewerPushFormat(csvViewer, PETSC_VIEWER_ASCII_CSV));

  // Use the CSV viewer to write out the local network
  DMView_Network_CSV(dm, csvViewer);

  // Close the viewer
  PetscCall(PetscViewerDestroy(&csvViewer));

  // Collect the temporary files on rank 0
  if (rank != 0) {
    // If not rank 0, send the file name
    PetscCallMPI(MPI_Send(filename, FILENAME_MAX, MPI_BYTE, 0, 0, comm));
  } else {
    // Get the value of $PETSC_DIR
    PetscCall(PetscStrreplace(PETSC_COMM_WORLD, "${PETSC_DIR}/share/petsc/dmnetwork_view.py", scriptFile, sizeof(scriptFile)));
    // Generate the system call for 'python3 $PETSC_DIR/share/petsc/dmnetwork_view.py file1 file2 ...'
    PetscCall(PetscArrayzero(proccall, sizeof(proccall)));
    PetscCall(PetscSNPrintfCount(proccall, sizeof(proccall), "%s %s %s", &numChars, PETSC_PYTHON_EXE, scriptFile, filename));

    filename[0] = ' ';
    // For every other rank, receive the file name and append with a space
    for (PetscMPIInt rank2 = 1; rank2 < size; rank2++) {
      PetscCallMPI(MPI_Recv(filename + 1, FILENAME_MAX, MPI_BYTE, rank2, 0, comm, MPI_STATUS_IGNORE));
      PetscCall(PetscStrlen(filename, &appendChars));
      numChars += appendChars;
      PetscCheck(numChars < sizeof(proccall), comm, PETSC_ERR_WRONG_MPI_SIZE, "Too many processes to invoke Matplotlib script");
      PetscCall(PetscStrlcat(proccall, filename, sizeof(proccall)));
    }

    // Perform the call to run the python script
    PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, proccall, "r", NULL));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Network(DM dm, PetscViewer viewer)
{
  PetscBool   iascii;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    const PetscInt   *cone, *vtx, *edges;
    PetscInt          vfrom, vto, i, j, nv, ne, nsv, p, nsubnet;
    DM_Network       *network = (DM_Network *)dm->data;
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_CSV) {
      PetscCall(DMView_Network_CSV(dm, viewer));
      PetscFunctionReturn(0);
    } else if (format == PETSC_VIEWER_ASCII_PYTHON) {
      PetscCall(DMView_Network_Matplotlib(dm));
      PetscFunctionReturn(0);
    }

    nsubnet = network->cloneshared->Nsubnet; /* num of subnetworks */
    if (!rank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  NSubnets: %" PetscInt_FMT "; NEdges: %" PetscInt_FMT "; NVertices: %" PetscInt_FMT "; NSharedVertices: %" PetscInt_FMT ".\n", nsubnet, network->cloneshared->NEdges, network->cloneshared->NVertices,
                            network->cloneshared->Nsvtx));
    }

    PetscCall(DMNetworkGetSharedVertices(dm, &nsv, NULL));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "  [%d] nEdges: %" PetscInt_FMT "; nVertices: %" PetscInt_FMT "; nSharedVertices: %" PetscInt_FMT "\n", rank, network->cloneshared->nEdges, network->cloneshared->nVertices, nsv));

    for (i = 0; i < nsubnet; i++) {
      PetscCall(DMNetworkGetSubnetwork(dm, i, &nv, &ne, &vtx, &edges));
      if (ne) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "     Subnet %" PetscInt_FMT ": nEdges %" PetscInt_FMT ", nVertices(include shared vertices) %" PetscInt_FMT "\n", i, ne, nv));
        for (j = 0; j < ne; j++) {
          p = edges[j];
          PetscCall(DMNetworkGetConnectedVertices(dm, p, &cone));
          PetscCall(DMNetworkGetGlobalVertexIndex(dm, cone[0], &vfrom));
          PetscCall(DMNetworkGetGlobalVertexIndex(dm, cone[1], &vto));
          PetscCall(DMNetworkGetGlobalEdgeIndex(dm, edges[j], &p));
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "       edge %" PetscInt_FMT ": %" PetscInt_FMT " ----> %" PetscInt_FMT "\n", p, vfrom, vto));
        }
      }
    }

    /* Shared vertices */
    PetscCall(DMNetworkGetSharedVertices(dm, NULL, &vtx));
    if (nsv) {
      PetscInt        gidx;
      PetscBool       ghost;
      const PetscInt *sv = NULL;

      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "     SharedVertices:\n"));
      for (i = 0; i < nsv; i++) {
        PetscCall(DMNetworkIsGhostVertex(dm, vtx[i], &ghost));
        if (ghost) continue;

        PetscCall(DMNetworkSharedVertexGetInfo(dm, vtx[i], &gidx, &nv, &sv));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "       svtx %" PetscInt_FMT ": global index %" PetscInt_FMT ", subnet[%" PetscInt_FMT "].%" PetscInt_FMT " ---->\n", i, gidx, sv[0], sv[1]));
        for (j = 1; j < nv; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "                                           ----> subnet[%" PetscInt_FMT "].%" PetscInt_FMT "\n", sv[2 * j], sv[2 * j + 1]));
      }
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else PetscCheck(iascii, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMNetwork writing", ((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}
