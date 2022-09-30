#include <petsc/private/dmnetworkimpl.h> /*I  "petscdmnetwork.h"  I*/

typedef struct {
  double    x;
  double    y;
  PetscReal color;
} Coordinates;

typedef struct {
  PetscInt    from[2];   /* x,y from */
  PetscInt    to[2];     /* x, y to */
  Coordinates *sublines; /* dim = n_sublines */
  PetscInt    n_sublines;
  PetscReal   *color;
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
  PetscInt       n_sublines,vStart,from,to,j,offset,rows[2];
  PetscScalar    val[4];
  Edge           edge = *edge_ptr;
  const PetscInt *connnodes;
  PetscReal      dx,dy;
  Coordinates    *cord = *cord_ptr;
  Vec            coords;
  DM             dmclone;

  PetscFunctionBegin;
  edge->n_sublines = n_sublines = 26;
  PetscCall(PetscCalloc2(n_sublines-1,&edge->color,n_sublines,&edge->sublines));

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

  PetscCall(DMNetworkGetLocalVecOffset(dmclone,connnodes[1],0,&offset));
  rows[0] = offset;
  rows[1] = offset + 1;
  PetscCall(VecGetValues(coords,2,rows,val+2));

  /* get cord[from] and cord[to] */
  cord[from].x = (double)val[0];
  cord[from].y = (double)val[1];
  cord[to].x   = (double)val[2];
  cord[to].y   = (double)val[3];

  /* cord[]-+1: offset in (x,y) of plot centering */
  if (cord[from].y > cord[to].y) {
    if (cord[from].x > cord[to].x) {
      edge->from[0] = cord[from].x-1;  edge->from[1] = cord[from].y-1;
      edge->to[0]   = cord[to].x-1;    edge->to[1]   = cord[to].y-1;
    } else if (cord[from].x <= cord[to].x) {
      edge->from[0] = cord[from].x-1;  edge->from[1] = cord[from].y + 1;
      edge->to[0]   = cord[to].x-1;    edge->to[1]   = cord[to].y + 1;
    }
  } else if (cord[from].y <= cord[to].y) {
    if (cord[from].x > cord[to].x) {
      edge->from[0] = cord[from].x + 1;  edge->from[1] = cord[from].y-1;
      edge->to[0]   = cord[to].x + 1;    edge->to[1]   = cord[to].y-1;
    } else if (cord[from].x <= cord[to].x) {
      edge->from[0] = cord[from].x + 1;  edge->from[1] = cord[from].y + 1;
      edge->to[0]   = cord[to].x + 1;    edge->to[1]   = cord[to].y + 1;
    }
  }

  // after the offset is set we break up the edges into the elements we got from usr->ctx.
  dx = ( edge->to[0]- edge->from[0])/(n_sublines-1);
  dy = ( edge->to[1]- edge->from[1])/(n_sublines-1);

  for ( j = 0; j <  n_sublines; j++) {
    edge->sublines[j].x = edge->from[0] + j*dx;
    edge->sublines[j].y = edge->from[1] + j*dy;
    edge->color[j]      = 0.0; /* coloring currently hard coded */
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
    sprintf(line,"%f,%f,%f\n",edge->sublines[j].x,edge->sublines[j].y,edge->color[j-1]); //x,y,color
    fputs(line, fp);
  }
  PetscCall(PetscFree2(edge->color, edge->sublines));
  PetscCall(PetscFree(edge));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Network_python(DM dmnetwork)
{
  PetscMPIInt    mpi_rank,mpi_size;
  MPI_Comm       comm;
  Coordinates    *cord;
  Edge           *edges;
  PetscInt       nv=0,ne=0,i;
  double         xmax,xmin,ymin,ymax,min[4],max[4];
  PetscInt       Nsubnet,net;
  const PetscInt *vtx,*e;
  FILE           *fp;
  char           fileName[20], line[1000];
  PetscInt       globalIndex;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dmnetwork,&comm));
  PetscCallMPI(MPI_Comm_rank(comm, &mpi_rank));
  PetscCallMPI(MPI_Comm_size(comm,&mpi_size));

  PetscCall(DMNetworkGetNumSubNetworks(dmnetwork,NULL,&Nsubnet));
  for (net=0; net<Nsubnet; net++) {
    PetscCall(DMNetworkGetSubnetwork(dmnetwork,net,&nv,&ne,&vtx,&e));
    if (nv) PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] DMNetworkView_py: subnet %d: nv %d, ne %d\n",mpi_rank,net,nv,ne));
    if (nv == 0) continue;

    PetscCall(PetscCalloc2(ne,&edges,nv,&cord));
    for (i = 0; i < ne; i++) PetscCall(PetscCalloc1(1, &edges[i]));

    /*  Get edge sublines and vertex coordinates from dmclone */
    for (i = 0; i < ne; i++) PetscCall(EdgeSublineCreate_coord(dmnetwork,e[i],&edges[i],&cord));

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
    /* printf("[%d] xmin/max: %g, %g, ymin/max: %g, %g\n",mpi_rank,xmin,xmax,ymin,ymax); */

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
      PetscCall(DMNetworkGetGlobalVertexIndex(dmnetwork, vtx[i], &globalIndex));
      sprintf(line,"%f,%f,%f,%d\n",cord[i].x,cord[i].y,cord[i].color,globalIndex);
      fputs(line, fp);
    }

    /* Write Edges */
    fputs("Edges & SubEdges:\n", fp);
    for (i = 0; i < ne; i++) PetscCall(EdgeSublineWrite(&edges[i],line,fp));
    fclose(fp);

    PetscCall(PetscFree2(edges,cord));
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
    PetscViewerFormat format;
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_PYTHON) {
      PetscCall(DMView_Network_python(dm));
      PetscFunctionReturn(0);
    }

    const PetscInt *cone, *vtx, *edges;
    PetscInt        vfrom, vto, i, j, nv, ne, nsv, p, nsubnet;
    DM_Network     *network = (DM_Network *)dm->data;

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

