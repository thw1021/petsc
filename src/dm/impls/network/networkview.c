#include <petsc/private/dmnetworkimpl.h> /*I  "petscdmnetwork.h"  I*/

typedef struct {
  double    x;
  double    y;
  double    color;
} Coordinates;

typedef struct {
  Coordinates *sublines; /* dim = n_sublines */
  PetscInt    n_sublines;
} *Edge;

/*
  EdgeSublineCreate_coord - Create an edge subline for visulization

  Input parameters:
    dmnetwork - dm context
    dmclone - clone of dm, storing coordinates of vertices for visualization
    e - edge number

  Output parameters:
    edge_ptr -
    cord_ptr -
 */
static PetscErrorCode EdgeSublineCreate_coord(DM dmnetwork,PetscInt e,Edge *edge_ptr,Coordinates **cord_ptr)
{
  PetscInt       n_sublines=2,vStart,from,to,j,offset,rows[2];
  PetscScalar    val[2];
  Edge           edge = *edge_ptr;
  const PetscInt *connnodes;
  PetscReal      dx,dy,dcolor;
  Coordinates    *cord = *cord_ptr;
  Vec            coords;
  DM             dmclone;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetInt(NULL, ((PetscObject)dmnetwork)->prefix, "-nsublines", &n_sublines, NULL));
  edge->n_sublines = n_sublines;
  PetscCall(PetscCalloc1(n_sublines,&edge->sublines));

  PetscCall(DMGetCoordinateDM(dmnetwork,&dmclone));
  PetscCall(DMNetworkGetVertexRange(dmclone,&vStart,NULL));
  PetscCall(DMNetworkGetConnectedVertices(dmclone,e,&connnodes));
  from = connnodes[0] - vStart;
  to   = connnodes[1] - vStart;

  /* get cord[from] from vector coords */
  PetscCall(DMGetCoordinates(dmnetwork, &coords));
  PetscCall(DMNetworkGetLocalVecOffset(dmclone,connnodes[0],0,&offset));
  rows[0] = offset;
  rows[1] = offset + 1;
  PetscCall(VecGetValues(coords,2,rows,val));
  cord[from].x     = (double)val[0];
  cord[from].y     = (double)val[1];
  cord[from].color = 0.0;

  /* get cord[to] from vector coords */
  PetscCall(DMNetworkGetLocalVecOffset(dmclone,connnodes[1],0,&offset));
  rows[0] = offset;
  rows[1] = offset + 1;
  PetscCall(VecGetValues(coords,2,rows,val));
  cord[to].x     = (double)val[0];
  cord[to].y     = (double)val[1];
  cord[to].color = 0.0;

  /* after the offset is set we break up the edges into the elements we got from usr->ctx. */
  dx = (cord[to].x - cord[from].x)/(n_sublines-1);
  dy = (cord[to].y - cord[from].y)/(n_sublines-1);
  dcolor = 0.1/(n_sublines-1); /* coloring currently hard coded */

  for ( j = 0; j < n_sublines; j++) {
    edge->sublines[j].x     = cord[from].x + j*dx;
    edge->sublines[j].y     = cord[from].y + j*dy;
    edge->sublines[j].color = cord[from].color + j*dcolor;
  }
  PetscFunctionReturn(0);
}

/* num subedge points, s_edge1,c1, ..., s_edge2 */
static PetscErrorCode EdgeSublineWrite(Edge *edge_ptr,char *line,FILE *fp)
{
  Edge           edge = *edge_ptr;
  PetscInt       j;

  PetscFunctionBegin;
  sprintf(line,"%d,%f,%f\n",edge->n_sublines,edge->sublines[0].x,edge->sublines[0].y);
  fputs(line, fp);

  for (j = 1; j <  edge->n_sublines; j++) {
    sprintf(line,"%f,%f,%f\n",edge->sublines[j].x,edge->sublines[j].y,edge->sublines[j-1].color);
    fputs(line, fp);
  }
  PetscCall(PetscFree(edge->sublines));
  PetscCall(PetscFree(edge));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Network_python(DM dmnetwork)
{
  PetscMPIInt    mpi_rank,mpi_size;
  MPI_Comm       comm;
  Coordinates    *cord;
  Edge           *edges;
  PetscInt       nv=0,ne=0,i,vstart,vend,estart,eend;
  double         xmax,xmin,ymin,ymax,min[4],max[4];
  FILE           *fp;
  char           fileName[20], line[1000];
  PetscInt       globalIndex;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dmnetwork,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &mpi_rank));
  PetscCallMPI(MPI_Comm_size(comm,&mpi_size));

  PetscCall(DMNetworkGetVertexRange(dmnetwork,&vstart,&vend));
  nv = vend -vstart;
  PetscCall(DMNetworkGetEdgeRange(dmnetwork,&estart,&eend));
  ne = eend - estart;

  PetscCall(PetscCalloc2(ne,&edges,nv,&cord));
  for (i = 0; i < ne; i++) PetscCall(PetscCalloc1(1, &edges[i]));

  /*  Get edge sublines and vertex coordinates from dmclone */
  for (i = estart; i < eend; i++) PetscCall(EdgeSublineCreate_coord(dmnetwork,i,&edges[i-estart],&cord));

  /* Get xmin, xmax, ymin, ymax for 2D plot */
  for (i = 0; i < nv; i++) {
    if (i ==0) {
      max[0] = min[0] = cord[i].x; // for the 3d visualization we need the max and min
      max[1] = min[1] = cord[i].y;
      continue;
    }
    min[0] = PetscMin(min[0],cord[i].x);
    min[1] = PetscMin(min[1],cord[i].y);
    max[0] = PetscMax(max[0],cord[i].x);
    max[1] = PetscMax(max[1],cord[i].y);
  }

  /* Sync xmin, xmax, ymin, ymax over all processors */
  PetscCallMPI(MPI_Allreduce(min, min+2, 2, MPIU_REAL,MPIU_MIN,comm));
  xmin = min[2]; ymin = min[3];

  PetscCallMPI(MPI_Allreduce(max, max+2, 2, MPIU_REAL,MPIU_MAX,comm));
  xmax = max[2]; ymax = max[3];
  /* printf("[%d] xmin/max: %g, %g, ymin/max: %g, %g\n",mpi_rank,xmin,xmax,ymin,ymax);*/

  sprintf(fileName, "Net_proc%d_snet.txt",mpi_rank);
  fp = fopen(fileName, "r");
  if (fp == NULL) {
    fp = fopen(fileName, "a");
    /* min/max for 3D nv/ne used for data readin, size for # of networks, and 1 for the first time call */
    sprintf(line, "%f,%f,%f,%f\n",xmin,xmax,ymin,ymax);
    fputs(line, fp);
    sprintf(line, "%d,%d,%d\n",nv,ne,mpi_size);
    fputs(line, fp);
  }
  fclose(fp);

  /* Write Node Locations */
  fp = fopen(fileName, "a");
  fputs("Node Locations:\n", fp);
  for (i = 0; i < nv; i++) {
    PetscCall(DMNetworkGetGlobalVertexIndex(dmnetwork, i+vstart, &globalIndex));
    sprintf(line,"%f,%f,%f,%d\n",cord[i].x,cord[i].y,cord[i].color,globalIndex);
    fputs(line, fp);
  }

  /* Write Edges */
  fputs("Edges & SubEdges:\n", fp);
  for (i = 0; i < ne; i++) PetscCall(EdgeSublineWrite(&edges[i],line,fp));
  fclose(fp);

  PetscCall(PetscFree2(edges,cord));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Network_CSV(DM dm, PetscViewer viewer)
{
  PetscFunctionBegin;

  // Get the network containing coordinate information
  DM dmcoords;
  PetscCall(DMGetCoordinateDM(dm, &dmcoords));

  // Write the header
  PetscCall(PetscViewerASCIIPrintf(viewer, "Type,ID,X,Y,Z,Name,Color\n"));

  // Iterate each subnetwork
  PetscInt nsubnets;
  PetscCall(DMNetworkGetNumSubNetworks(dm, &nsubnets, NULL));
  for(PetscInt subnet = 0; subnet < nsubnets; subnet++) {
    // Get the subnetwork's vertices and edges
    PetscInt nvertices, nedges;
    const PetscInt *vertices, *edges;
    PetscCall(DMNetworkGetSubnetwork(dm, subnet, &nvertices, &nedges, &vertices, &edges));

    // Get the coordinate vector for the network
    Vec allVertexCoords;
    PetscCall(DMGetCoordinates(dm, &allVertexCoords));

    // Write out each vertex
    PetscInt vertexOffsets[2];
    PetscScalar vertexCoords[2];
    for(PetscInt i = 0; i < nvertices; i++) {
      PetscInt vertex = vertices[i];
      // Get the offset into the coordinate vector for the vertex
      PetscCall(DMNetworkGetLocalVecOffset(dmcoords, vertex, ALL_COMPONENTS, vertexOffsets));
      vertexOffsets[1] = vertexOffsets[0] + 1;
      // Get the vertex position from the coordinate vector
      PetscCall(VecGetValues(allVertexCoords, 2, vertexOffsets, vertexCoords));
      // TODO: Determine vertex color/name
      
      PetscCall(PetscViewerASCIIPrintf(viewer, "Node,%" PetscInt_FMT ",%lf,%lf,0,%" PetscInt_FMT "\n", vertex, (double)vertexCoords[0], (double)vertexCoords[1], vertex));
    }

    // Write out each edge
    const PetscInt* edgeVertices;
    for(PetscInt i = 0; i < nedges; i++) {
      PetscInt edge = edges[i];
      PetscCall(DMNetworkGetConnectedVertices(dm, edge, &edgeVertices));
      // TODO: Determine edge color/name

      PetscCall(PetscViewerASCIIPrintf(viewer, "Edge,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",0,%" PetscInt_FMT "\n", edge, edgeVertices[0], edgeVertices[1], edge));
    }
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Network_Matplotlib(DM dm)
{
  PetscFunctionBegin;

  // Get the MPI communicator and this process' rank
  PetscMPIInt rank, numranks;
  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &numranks));

  // Acquire a temporary file to write to and open an ASCII/CSV viewer
  char filename[L_tmpnam+1];
  tmpnam(filename);
  PetscViewer csvViewer;
  PetscCall(PetscViewerASCIIOpen(comm, filename, &csvViewer));
  PetscCall(PetscViewerPushFormat(csvViewer, PETSC_VIEWER_ASCII_CSV));

  // Use the CSV viewer to write out the local network
  DMView_Network_CSV(dm, csvViewer);
  
  // Close the viewer
  PetscCall(PetscViewerDestroy(&csvViewer));

  // Collect the temporary files on rank 0
  if (rank != 0) {
    // If not rank 0, send the file name
    PetscCallMPI(MPI_Send(filename, L_tmpnam, MPI_BYTE, 0, 0, comm));
  } else {
    // Generate the system call for 'python3 $PETSC_DIR/lib/petsc/bin/dmnetwork_view.py file1 file2 ...'
    // TODO: This currently uses unsafe string functions and could overflow if enough processes are used
    char proccall[2000];
    memset(proccall, 0, sizeof(proccall));
    snprintf(proccall, sizeof(proccall), "python3 %s/lib/petsc/bin/dmnetwork_view.py %s", getenv("PETSC_DIR"), filename);

    filename[0] = ' ';
    // For every other rank, receive the file name and append with a space
    for(PetscMPIInt rank2 = 1; rank2 < numranks; rank2++) {
      PetscCallMPI(MPI_Recv(filename+1, L_tmpnam, MPI_BYTE, rank2, 0, comm, MPI_STATUS_IGNORE));
      strcat(proccall, filename);
    }

    // Perform the call to run the python script
    system(proccall);
  }

  // Introduce a barrier to make sure all ranks maintain sync before returning
  PetscCallMPI(MPI_Barrier(comm));
  
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
    const PetscInt    *cone, *vtx, *edges;
    PetscInt          vfrom, vto, i, j, nv, ne, nsv, p, nsubnet;
    DM_Network        *network = (DM_Network *)dm->data;
    PetscViewerFormat format;

    // TODO: Why is format not preserved?
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_CSV));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_CSV) {
      PetscCall(DMView_Network_CSV(dm, viewer));
      PetscFunctionReturn(0);
    }

    nsubnet = network->cloneshared->Nsubnet; /* num of subnetworks */
    if (rank == 0) {
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
      PetscInt       gidx;
      PetscBool      ghost;
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

