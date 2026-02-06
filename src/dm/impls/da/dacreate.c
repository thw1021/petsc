#include <petsc/private/dmdaimpl.h> /*I   "petscdmda.h"   I*/

static PetscErrorCode DMSetFromOptions_DA(DM da, PetscOptionItems PetscOptionsObject)
{
  DM_DA         *dd     = (DM_DA *)da->data;
  PetscInt       refine = 0, dim = da->dim, maxnlevels = 100, refx[100], refy[100], refz[100], n, i;
  DMBoundaryType bt = DM_BOUNDARY_NONE;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCheck(!da->setupcalled, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cannot call DMSetFromOptions() after DMSetUp() for DMDA");
  PetscCheck(dd->M >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");
  PetscCheck(dd->N >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");
  PetscCheck(dd->P >= 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Dimension must be non-negative, call DMSetFromOptions() if you want to change the value at runtime");

  PetscOptionsHeadBegin(PetscOptionsObject, "DMDA Options");
  PetscCall(PetscOptionsBoundedInt("-da_grid_x", "Number of grid points in x direction", "DMDASetSizes", dd->M, &dd->M, NULL, 1));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_grid_y", "Number of grid points in y direction", "DMDASetSizes", dd->N, &dd->N, NULL, 1));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_grid_z", "Number of grid points in z direction", "DMDASetSizes", dd->P, &dd->P, NULL, 1));

  PetscCall(PetscOptionsBoundedInt("-da_overlap", "Decomposition overlap in all directions", "DMDASetOverlap", dd->xol, &dd->xol, &flg, 0));
  if (flg) PetscCall(DMDASetOverlap(da, dd->xol, dd->xol, dd->xol));
  PetscCall(PetscOptionsBoundedInt("-da_overlap_x", "Decomposition overlap in x direction", "DMDASetOverlap", dd->xol, &dd->xol, NULL, 0));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_overlap_y", "Decomposition overlap in y direction", "DMDASetOverlap", dd->yol, &dd->yol, NULL, 0));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_overlap_z", "Decomposition overlap in z direction", "DMDASetOverlap", dd->zol, &dd->zol, NULL, 0));

  PetscCall(PetscOptionsBoundedInt("-da_local_subdomains", "", "DMDASetNumLocalSubdomains", dd->Nsub, &dd->Nsub, &flg, PETSC_DECIDE));
  if (flg) PetscCall(DMDASetNumLocalSubDomains(da, dd->Nsub));

  /* Handle DMDA parallel distribution */
  PetscCall(PetscOptionsBoundedInt("-da_processors_x", "Number of processors in x direction", "DMDASetNumProcs", dd->m, &dd->m, NULL, PETSC_DECIDE));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_processors_y", "Number of processors in y direction", "DMDASetNumProcs", dd->n, &dd->n, NULL, PETSC_DECIDE));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_processors_z", "Number of processors in z direction", "DMDASetNumProcs", dd->p, &dd->p, NULL, PETSC_DECIDE));
  // Handle boundaries
  PetscCall(PetscOptionsEnum("-da_bd_x", "Boundary type for x direction", "DMDASetBoundaryType", DMBoundaryTypes, (PetscEnum)dd->bx, (PetscEnum *)&dd->bx, NULL));
  if (dim > 1) PetscCall(PetscOptionsEnum("-da_bd_y", "Boundary type for y direction", "DMDASetBoundaryType", DMBoundaryTypes, (PetscEnum)dd->by, (PetscEnum *)&dd->by, NULL));
  if (dim > 2) PetscCall(PetscOptionsEnum("-da_bd_z", "Boundary type for z direction", "DMDASetBoundaryType", DMBoundaryTypes, (PetscEnum)dd->bz, (PetscEnum *)&dd->bz, NULL));
  PetscCall(PetscOptionsEnum("-da_bd_all", "Boundary type for every direction", "DMDASetBoundaryType", DMBoundaryTypes, (PetscEnum)bt, (PetscEnum *)&bt, &flg));
  if (flg) PetscCall(DMDASetBoundaryType(da, bt, bt, bt));
  /* Handle DMDA refinement */
  PetscCall(PetscOptionsBoundedInt("-da_refine_x", "Refinement ratio in x direction", "DMDASetRefinementFactor", dd->refine_x, &dd->refine_x, NULL, 1));
  if (dim > 1) PetscCall(PetscOptionsBoundedInt("-da_refine_y", "Refinement ratio in y direction", "DMDASetRefinementFactor", dd->refine_y, &dd->refine_y, NULL, 1));
  if (dim > 2) PetscCall(PetscOptionsBoundedInt("-da_refine_z", "Refinement ratio in z direction", "DMDASetRefinementFactor", dd->refine_z, &dd->refine_z, NULL, 1));
  dd->coarsen_x = dd->refine_x;
  dd->coarsen_y = dd->refine_y;
  dd->coarsen_z = dd->refine_z;

  /* Get refinement factors, defaults taken from the coarse DMDA */
  PetscCall(DMDAGetRefinementFactor(da, &refx[0], &refy[0], &refz[0]));
  for (i = 1; i < maxnlevels; i++) {
    refx[i] = refx[0];
    refy[i] = refy[0];
    refz[i] = refz[0];
  }
  n = maxnlevels;
  PetscCall(PetscOptionsIntArray("-da_refine_hierarchy_x", "Refinement factor for each level", "None", refx, &n, &flg));
  if (flg) {
    dd->refine_x        = refx[0];
    dd->refine_x_hier_n = n;
    PetscCall(PetscMalloc1(n, &dd->refine_x_hier));
    PetscCall(PetscArraycpy(dd->refine_x_hier, refx, n));
  }
  if (dim > 1) {
    n = maxnlevels;
    PetscCall(PetscOptionsIntArray("-da_refine_hierarchy_y", "Refinement factor for each level", "None", refy, &n, &flg));
    if (flg) {
      dd->refine_y        = refy[0];
      dd->refine_y_hier_n = n;
      PetscCall(PetscMalloc1(n, &dd->refine_y_hier));
      PetscCall(PetscArraycpy(dd->refine_y_hier, refy, n));
    }
  }
  if (dim > 2) {
    n = maxnlevels;
    PetscCall(PetscOptionsIntArray("-da_refine_hierarchy_z", "Refinement factor for each level", "None", refz, &n, &flg));
    if (flg) {
      dd->refine_z        = refz[0];
      dd->refine_z_hier_n = n;
      PetscCall(PetscMalloc1(n, &dd->refine_z_hier));
      PetscCall(PetscArraycpy(dd->refine_z_hier, refz, n));
    }
  }

  PetscCall(PetscOptionsBoundedInt("-da_refine", "Uniformly refine DA one or more times", "None", refine, &refine, NULL, 0));
  PetscCall(PetscOptionsBool("-da_use_section", "Create a PetscSection for the data layout", "None", dd->useSection, &dd->useSection, NULL));
  PetscOptionsHeadEnd();

  while (refine--) {
    if (dd->bx == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0) {
      PetscCall(PetscIntMultError(dd->refine_x, dd->M, &dd->M));
    } else {
      PetscCall(PetscIntMultError(dd->refine_x, dd->M - 1, &dd->M));
      dd->M += 1;
    }
    if (dim > 1 && (dd->by == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0)) {
      PetscCall(PetscIntMultError(dd->refine_y, dd->N, &dd->N));
    } else if (dim > 1) {
      PetscCall(PetscIntMultError(dd->refine_y, dd->N - 1, &dd->N));
      dd->N += 1;
    }
    if (dim > 2 && (dd->bz == DM_BOUNDARY_PERIODIC || dd->interptype == DMDA_Q0)) {
      PetscCall(PetscIntMultError(dd->refine_z, dd->P, &dd->P));
    } else if (dim > 2) {
      PetscCall(PetscIntMultError(dd->refine_z, dd->P - 1, &dd->P));
      dd->P += 1;
    }
    da->levelup++;
    if (da->levelup - da->leveldown >= 0) {
      dd->refine_x = refx[da->levelup - da->leveldown];
      dd->refine_y = refy[da->levelup - da->leveldown];
      dd->refine_z = refz[da->levelup - da->leveldown];
    }
    if (da->levelup - da->leveldown >= 1) {
      dd->coarsen_x = refx[da->levelup - da->leveldown - 1];
      dd->coarsen_y = refy[da->levelup - da->leveldown - 1];
      dd->coarsen_z = refz[da->levelup - da->leveldown - 1];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMLoad_DA(DM da, PetscViewer viewer)
{
  PetscInt        dim, m, n, p, dof, swidth;
  DMDAStencilType stencil;
  DMBoundaryType  bx, by, bz;
  PetscBool       coors;
  DM              dac;
  Vec             c;

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryRead(viewer, &dim, 1, NULL, PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer, &m, 1, NULL, PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer, &n, 1, NULL, PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer, &p, 1, NULL, PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer, &dof, 1, NULL, PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer, &swidth, 1, NULL, PETSC_INT));
  PetscCall(PetscViewerBinaryRead(viewer, &bx, 1, NULL, PETSC_ENUM));
  PetscCall(PetscViewerBinaryRead(viewer, &by, 1, NULL, PETSC_ENUM));
  PetscCall(PetscViewerBinaryRead(viewer, &bz, 1, NULL, PETSC_ENUM));
  PetscCall(PetscViewerBinaryRead(viewer, &stencil, 1, NULL, PETSC_ENUM));

  PetscCall(DMSetDimension(da, dim));
  PetscCall(DMDASetSizes(da, m, n, p));
  PetscCall(DMDASetBoundaryType(da, bx, by, bz));
  PetscCall(DMDASetDof(da, dof));
  PetscCall(DMDASetStencilType(da, stencil));
  PetscCall(DMDASetStencilWidth(da, swidth));
  PetscCall(DMSetUp(da));
  PetscCall(PetscViewerBinaryRead(viewer, &coors, 1, NULL, PETSC_BOOL));
  if (coors) {
    PetscCall(DMGetCoordinateDM(da, &dac));
    PetscCall(DMCreateGlobalVector(dac, &c));
    PetscCall(VecLoad(c, viewer));
    PetscCall(DMSetCoordinates(da, c));
    PetscCall(VecDestroy(&c));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateSubDM_DA(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  DM_DA *da = (DM_DA *)dm->data;

  PetscFunctionBegin;
  if (subdm) {
    PetscSF sf;
    Vec     coords;
    void   *ctx;
    /* Cannot use DMClone since the dof stuff is mixed in. Ugh
    PetscCall(DMClone(dm, subdm)); */
    PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), subdm));
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(DMSetPointSF(*subdm, sf));
    PetscCall(DMGetApplicationContext(dm, &ctx));
    PetscCall(DMSetApplicationContext(*subdm, ctx));
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    if (coords) {
      PetscCall(DMSetCoordinatesLocal(*subdm, coords));
    } else {
      PetscCall(DMGetCoordinates(dm, &coords));
      if (coords) PetscCall(DMSetCoordinates(*subdm, coords));
    }

    PetscCall(DMSetType(*subdm, DMDA));
    PetscCall(DMSetDimension(*subdm, dm->dim));
    PetscCall(DMDASetSizes(*subdm, da->M, da->N, da->P));
    PetscCall(DMDASetNumProcs(*subdm, da->m, da->n, da->p));
    PetscCall(DMDASetBoundaryType(*subdm, da->bx, da->by, da->bz));
    PetscCall(DMDASetDof(*subdm, numFields));
    PetscCall(DMDASetStencilType(*subdm, da->stencil_type));
    PetscCall(DMDASetStencilWidth(*subdm, da->s));
    PetscCall(DMDASetOwnershipRanges(*subdm, da->lx, da->ly, da->lz));
  }
  if (is) {
    PetscInt *indices, cnt = 0, dof = da->w, i, j;

    PetscCall(PetscMalloc1(da->Nlocal * numFields / dof, &indices));
    for (i = da->base / dof; i < (da->base + da->Nlocal) / dof; ++i) {
      for (j = 0; j < numFields; ++j) indices[cnt++] = dof * i + fields[j];
    }
    PetscCheck(cnt == da->Nlocal * numFields / dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Count %" PetscInt_FMT " does not equal expected value %" PetscInt_FMT, cnt, da->Nlocal * numFields / dof);
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), cnt, indices, PETSC_OWN_POINTER, is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateFieldDecomposition_DA(DM dm, PetscInt *len, char ***namelist, IS **islist, DM **dmlist)
{
  PetscInt i;
  DM_DA   *dd  = (DM_DA *)dm->data;
  PetscInt dof = dd->w;

  PetscFunctionBegin;
  if (len) *len = dof;
  if (islist) {
    Vec      v;
    PetscInt rstart, n;

    PetscCall(DMGetGlobalVector(dm, &v));
    PetscCall(VecGetOwnershipRange(v, &rstart, NULL));
    PetscCall(VecGetLocalSize(v, &n));
    PetscCall(DMRestoreGlobalVector(dm, &v));
    PetscCall(PetscMalloc1(dof, islist));
    for (i = 0; i < dof; i++) PetscCall(ISCreateStride(PetscObjectComm((PetscObject)dm), n / dof, rstart + i, dof, &(*islist)[i]));
  }
  if (namelist) {
    PetscCall(PetscMalloc1(dof, namelist));
    PetscCheck(dd->fieldname, PETSC_COMM_SELF, PETSC_ERR_SUP, "Currently DMDA must have fieldnames");
    for (i = 0; i < dof; i++) PetscCall(PetscStrallocpy(dd->fieldname[i], &(*namelist)[i]));
  }
  if (dmlist) {
    DM da;

    PetscCall(DMDACreate(PetscObjectComm((PetscObject)dm), &da));
    PetscCall(DMSetDimension(da, dm->dim));
    PetscCall(DMDASetSizes(da, dd->M, dd->N, dd->P));
    PetscCall(DMDASetNumProcs(da, dd->m, dd->n, dd->p));
    PetscCall(DMDASetBoundaryType(da, dd->bx, dd->by, dd->bz));
    PetscCall(DMDASetDof(da, 1));
    PetscCall(DMDASetStencilType(da, dd->stencil_type));
    PetscCall(DMDASetStencilWidth(da, dd->s));
    PetscCall(DMSetUp(da));
    PetscCall(PetscMalloc1(dof, dmlist));
    for (i = 0; i < dof - 1; i++) PetscCall(PetscObjectReference((PetscObject)da));
    for (i = 0; i < dof; i++) (*dmlist)[i] = da;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMClone_DA(DM dm, DM *newdm)
{
  DM_DA *da = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscCall(DMSetType(*newdm, DMDA));
  PetscCall(DMSetDimension(*newdm, dm->dim));
  PetscCall(DMDASetSizes(*newdm, da->M, da->N, da->P));
  PetscCall(DMDASetNumProcs(*newdm, da->m, da->n, da->p));
  PetscCall(DMDASetBoundaryType(*newdm, da->bx, da->by, da->bz));
  PetscCall(DMDASetDof(*newdm, da->w));
  PetscCall(DMDASetStencilType(*newdm, da->stencil_type));
  PetscCall(DMDASetStencilWidth(*newdm, da->s));
  PetscCall(DMDASetOwnershipRanges(*newdm, da->lx, da->ly, da->lz));
  PetscCall(DMSetUp(*newdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMHasCreateInjection_DA(DM dm, PetscBool *flg)
{
  DM_DA *da = (DM_DA *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  *flg = da->interptype == DMDA_Q1 ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetDimPoints_DA(DM dm, PetscInt dim, PetscInt *pStart, PetscInt *pEnd)
{
  PetscFunctionBegin;
  PetscCall(DMDAGetDepthStratum(dm, dim, pStart, pEnd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetNeighbors_DA(DM dm, PetscInt *nranks, const PetscMPIInt *ranks[])
{
  PetscInt        dim;
  DMDAStencilType st;

  PetscFunctionBegin;
  PetscCall(DMDAGetNeighbors(dm, ranks));
  PetscCall(DMDAGetInfo(dm, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &st));

  switch (dim) {
  case 1:
    *nranks = 3;
    /* if (st == DMDA_STENCIL_STAR) *nranks = 3; */
    break;
  case 2:
    *nranks = 9;
    /* if (st == DMDA_STENCIL_STAR) *nranks = 5; */
    break;
  case 3:
    *nranks = 27;
    /* if (st == DMDA_STENCIL_STAR) *nranks = 7; */
    break;
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscsf.h>

static PetscErrorCode PetscSFAddFace_2D_Private(DM dm, PetscMPIInt nrank, PetscInt xmin, PetscInt xmax, PetscInt ymin, PetscInt ymax, const PetscInt bases[], const PetscInt glx[], const PetscInt glxs[], const PetscInt glys[], PetscInt *l, PetscInt local[], PetscSFNode remote[])
{
  PetscInt pm, dof, x, y, gx, gy, gm;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(dm, NULL, NULL, NULL, NULL, &pm, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(dm, &x, &y, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(dm, &gx, &gy, NULL, &gm, NULL, NULL));
  const PetscInt nrm = nrank % pm; // The x-component of the neighbor rank, in [0, m)
  const PetscInt nrn = nrank / pm; // The y-component of the neighbor rank, in [0, n)

  if (nrank < 0) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt j = ymin; j < ymax; ++j) {
    for (PetscInt i = xmin; i < xmax; ++i) {
      local[*l]        = (j - gy) * gm + (i - gx);
      remote[*l].rank  = nrank;
      remote[*l].index = (j - glys[nrn] * dof) * glx[nrm] * dof + (i - glxs[nrm] * dof);
      ++(*l);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode PetscSFAddFace_1D_Private(DM dm, PetscMPIInt nrank, PetscInt xmin, PetscInt xmax, const PetscInt bases[], const PetscInt glx[], const PetscInt glxs[], PetscInt *l, PetscInt local[], PetscSFNode remote[])
{
  PetscInt dof, x, gx, gm, pm;

  PetscFunctionBegin;
  if (nrank < 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMDAGetInfo(dm, NULL, NULL, NULL, NULL, &pm, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(dm, &x, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetGhostCorners(dm, &gx, NULL, NULL, &gm, NULL, NULL));

  const PetscInt nrm = nrank;

  for (PetscInt i = xmin; i < xmax; ++i) {
    local[*l]        = i - gx;
    remote[*l].rank  = nrank;
    remote[*l].index = (i - glxs[nrm]) * dof;
    ++(*l);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateLocalSection_DA(DM dm)
{
  MPI_Comm     comm;
  PetscMPIInt  size, rank, left=-1, right=-1;
  PetscSection s;
  PetscSF      sf;
  PetscInt     x, y = 0, z = 0, m, n = 1, p = 1;
  PetscInt     gx, gy = 0, gz = 0, gm, gn = 1, gp = 1;
  PetscInt     dim, dof, M, N = 1, P = 1, pm, pn = 1, pp = 1, Nv, gNv;

  DM_DA        *dd = (DM_DA *)dm->data;
  DMDA_PointBC *bc = dd->bc;

  PetscFunctionBegin;
  if (!dd->useSection) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMDAGetInfo(dm, &dim, &M, &N, &P, &pm, &pn, &pp, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(dm, &x, &y, &z, &m, &n, &p));
  PetscCall(DMDAGetGhostCorners(dm, &gx, &gy, &gz, &gm, &gn, &gp));
  Nv  = m * n * p;
  gNv = gm * gn * gp;
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &s));
  PetscCall(PetscSectionSetChart(s, 0, gNv));
  for (PetscInt v = 0; v < gNv; ++v) PetscCall(PetscSectionSetDof(s, v, dof));

  // If DM_BOUNDARY_GHOSTED is used, mark ghosted vertices in this region as
  // "constrained" so they are excluded from the SF
  PetscBool ghostX = (dd->bx == DM_BOUNDARY_GHOSTED);
  PetscBool ghostY = (dim > 1 && dd->by == DM_BOUNDARY_GHOSTED);

  if (ghostX || ghostY) {
      for (PetscInt j = gy; j < gy + gn; j++) {
        for (PetscInt i = gx; i < gx + gm; i++) {

          PetscBool isGhostedBoundary =
            (ghostX && (i < 0 || i >= M)) ||
            (ghostY && (dim > 1) && (j < 0 || j >= N));

          if (isGhostedBoundary) {
            PetscInt p =
              (j - gy) * gm + (i - gx);
            PetscCall(PetscSectionSetConstraintDof(s, p, dof));
          }
        }
      }
  }

  if (bc) {
    // Set BC dofs
    for (PetscInt b = 0; b < bc->numBC; b++) {
      const PetscInt *pts;
      PetscInt        npts = 0, ncmp = 0;

      PetscCall(ISGetLocalSize(bc->bcPoints[b], &npts));
      PetscCall(ISGetIndices(bc->bcPoints[b], &pts));

      if (bc->bcComps[b]) {
        PetscCall(ISGetLocalSize(bc->bcComps[b], &ncmp));
      } else {
        ncmp = dof; /* constrain all components/dofs at the point */
      }
      for (PetscInt i = 0; i < npts; i++) {
        PetscInt p = pts[i];
        PetscCall(PetscSectionSetConstraintDof(s, p, ncmp));
      }
      PetscCall(ISRestoreIndices(bc->bcPoints[b], &pts));
    }
  }

  PetscCall(PetscSectionSetFromOptions(s));
  PetscCall(PetscSectionSetUp(s));

  if (bc) {
    // Set BC indices
    for (PetscInt b = 0; b < bc->numBC; b++) {
      const PetscInt *pts, *cmp;
      PetscInt        npts = 0, ncmp = 0;

      PetscCall(ISGetLocalSize(bc->bcPoints[b], &npts));
      PetscCall(ISGetIndices(bc->bcPoints[b], &pts));

      if (bc->bcComps[b]) {
        PetscCall(ISGetLocalSize(bc->bcComps[b], &ncmp));
        PetscCall(ISGetIndices(bc->bcComps[b], &cmp));
      } else {
        cmp = NULL;
      }
      for (PetscInt i = 0; i < npts; i++) {
        PetscInt p = pts[i];
        PetscCall(PetscSectionSetConstraintIndices(s, p, cmp));
      }
      PetscCall(ISRestoreIndices(bc->bcPoints[b], &pts));
      if (bc->bcComps[b]) PetscCall(ISRestoreIndices(bc->bcComps[b], &cmp));
    }
  }
  // number of boundary vertices
  PetscInt Nbv = 0;
  if (ghostX || ghostY) {
    PetscInt comps[dof];
    for (PetscInt c = 0; c < dof; c++) comps[c] = c;

      for (PetscInt j = gy; j < gy + gn; j++) {
        for (PetscInt i = gx; i < gx + gm; i++) {

          PetscBool isGhostedBoundary =
            (ghostX && (i < 0 || i >= M)) ||
            (ghostY && (j < 0 || j >= N));

          if (isGhostedBoundary) { 
            Nbv ++;
            PetscInt p =
              (j - gy) * gm + (i - gx);
            PetscCall(PetscSectionSetConstraintIndices(s, p, comps));
          }
        }
    }
  }
  PetscCall(DMSetLocalSection(dm, s));
  // DMView(dm, NULL);
  // PetscSectionView(s, NULL);
  PetscCall(PetscSectionDestroy(&s));

  // Create point SF
  const PetscMPIInt *neigh;  // Neighbor ranks
  PetscInt          *local;  // Local indices of shared vertices
  PetscSFNode       *remote; // Remote indices of shared vertices
  PetscInt          *bases;  // First vertex on each rank
  PetscInt          *ldims;  // Number of vertices on each rank
  const PetscInt    *glx;    // The number of ghosted vertices along x in each of the m procs
  const PetscInt    *gly;    // The number of ghosted vertices along y in each of the n procs
  const PetscInt    *glz;    // The number of ghosted vertices along z in each of the p procs
  const PetscInt    *glxs;   // The first ghosted vertex along x in each of the m procs
  const PetscInt    *glys;   // The first ghosted vertex along y in each of the n procs
  const PetscInt    *glzs;   // The first ghosted vertex along z in each of the p procs
  PetscInt           Nl=0, l = 0;
  PetscCall(PetscSFCreate(comm, &sf));
  Nl = gNv - Nv - Nbv;

  // Compute starting point of each process
  PetscCall(PetscMalloc2(size + 1, &bases, size, &ldims));
  PetscCallMPI(MPI_Allgather(&Nv, 1, MPIU_INT, ldims, 1, MPIU_INT, comm));
  bases[0] = 0;
  for (PetscInt i = 1; i <= size; i++) bases[i] = ldims[i - 1];
  for (PetscInt i = 1; i <= size; i++) bases[i] += bases[i - 1];
  // Compute local and remote points for each leaf
  PetscCall(DMDAGetGhostOwnershipRanges(dm, &glx, &glxs, &gly, &glys, &glz, &glzs));


  if (dim == 1) {
      if (rank > 0) left  = rank - 1;
      if (rank < size-1) right = rank + 1;
  } else {
      PetscCall(DMDAGetNeighbors(dm, &neigh));
  }

  PetscCall(PetscMalloc2(Nl, &local, Nl, &remote));
  // for (PetscInt i = 0; i < pm; ++i) PetscSynchronizedPrintf(comm, "[%d]glx %d\n", rank, glx[i]);
  // for (PetscInt i = 0; i < pn; ++i) PetscSynchronizedPrintf(comm, "[%d]gly %d\n", rank, gly[i]);
  switch (dim) {
  case 1:
    // Left
    PetscCall(PetscSFAddFace_1D_Private(dm, left, gx, x, bases, glx, glxs, &l, local, remote));
    // Right
    PetscCall(PetscSFAddFace_1D_Private(dm, right, x + m, gx + gm, bases, glx, glxs, &l, local, remote));
    break;
  case 2:
    // Lower left
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[0], gx, x, gy, y, bases, glx, glxs, glys, &l, local, remote));
    // Lower middle
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[1], x, x + m, gy, y, bases, glx, glxs, glys, &l, local, remote));
    // Lower right
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[2], x + m, gx + gm, gy, y, bases, glx, glxs, glys, &l, local, remote));
    // Middle left
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[3], gx, x, y, y + n, bases, glx, glxs, glys, &l, local, remote));
    // Middle right
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[5], x + m, gx + gm, y, y + n, bases, glx, glxs, glys, &l, local, remote));
    // Upper left
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[6], gx, x, y + n, gy + gn, bases, glx, glxs, glys, &l, local, remote));
    // Upper middle
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[7], x, x + m, y + n, gy + gn, bases, glx, glxs, glys, &l, local, remote));
    // Upper right
    PetscCall(PetscSFAddFace_2D_Private(dm, neigh[8], x + m, gx + gm, y + n, gy + gn, bases, glx, glxs, glys, &l, local, remote));
    break;
  default:
    PetscCheck(0, comm, PETSC_ERR_SUP, "No support for dimension %" PetscInt_FMT, dim);
  }
  PetscCall(PetscFree2(bases, ldims));
  PetscCall(DMDARestoreGhostOwnershipRanges(dm, &glx, &glxs, &gly, &glys, &glz, &glzs));

  // for (PetscInt l = 0; l < Nl; ++l) PetscSynchronizedPrintf(comm, "[%d]local: %d remote %d %d\n", rank, local[l], remote[l].rank, remote[l].index);
  // PetscSynchronizedFlush(comm, NULL);
  PetscCall(PetscSFSetGraph(sf, gNv, Nl, local, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
  // view the SF
  // PetscCall(PetscSFView(sf, NULL));
  PetscCall(DMSetPointSF(dm, sf));

  PetscCall(PetscSFDestroy(&sf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMDASetPointBC(DM dm, PetscInt numBC, IS bcPoints[], IS bcComps[])
{
  DM_DA        *dd = (DM_DA *)dm->data;
  DMDA_PointBC *bc = NULL;

  PetscFunctionBegin;
  PetscCall(PetscNew(&bc));
  bc->numBC = numBC;

  PetscCall(PetscMalloc1(numBC, &bc->bcPoints));
  PetscCall(PetscMalloc1(numBC, &bc->bcComps));

  for (PetscInt b = 0; b < numBC; b++) {
    PetscCall(ISDuplicate(bcPoints[b], &bc->bcPoints[b]));
    if (bcComps && bcComps[b]) {
      PetscCall(ISDuplicate(bcComps[b], &bc->bcComps[b]));
    } else {
      bc->bcComps[b] = NULL; /* constrain all components */
    }
  }
  dd->bc = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   DMDA = "da" - A `DM` object that is used to help solve PDEs on a structured grid (or mesh) in 1, 2, or 3 dimensions.

   Level: intermediate

   Notes:
   In the global representation of the vectors each process stores a non-overlapping rectangular (or slab in 3d) portion of the grid points.
   In the local representation these rectangular regions (slabs) are extended in all directions by a stencil width set with `DMDASetStencilWidth()`.

   The vectors can be thought of as either cell centered or vertex centered on the grid (or mesh). But some variables cannot be cell centered and others
   vertex centered; see the documentation for `DMSTAG`, a similar `DM` implementation which supports more general staggered grids.

  Periodic boundary conditions can be handled by using a `DMBoundaryType` of `DM_BOUNDARY_PERIODIC` provided with `DMDASetBoundaryType()`.
  Other `DMBoundaryType`values allow for different handling of terms along the boundary of the grid (or mesh).

.seealso: [](sec_struct), `DMType`, `DMCOMPOSITE`, `DMSTAG`, `DMDACreate()`, `DMCreate()`, `DMSetType()`, `DMDASetStencilWidth()`, `DMDASetStencilType()`,
          `DMDAStencilType`
M*/

PETSC_EXTERN PetscErrorCode DMCreate_DA(DM da)
{
  DM_DA *dd;

  PetscFunctionBegin;
  PetscAssertPointer(da, 1);
  PetscCall(PetscNew(&dd));
  da->data = dd;

  da->dim        = -1;
  dd->interptype = DMDA_Q1;
  dd->refine_x   = 2;
  dd->refine_y   = 2;
  dd->refine_z   = 2;
  dd->coarsen_x  = 2;
  dd->coarsen_y  = 2;
  dd->coarsen_z  = 2;
  dd->fieldname  = NULL;
  dd->nlocal     = -1;
  dd->Nlocal     = -1;
  dd->M          = -1;
  dd->N          = -1;
  dd->P          = -1;
  dd->m          = -1;
  dd->n          = -1;
  dd->p          = -1;
  dd->w          = -1;
  dd->s          = -1;

  dd->xs = -1;
  dd->xe = -1;
  dd->ys = -1;
  dd->ye = -1;
  dd->zs = -1;
  dd->ze = -1;
  dd->Xs = -1;
  dd->Xe = -1;
  dd->Ys = -1;
  dd->Ye = -1;
  dd->Zs = -1;
  dd->Ze = -1;

  dd->Nsub = 1;
  dd->xol  = 0;
  dd->yol  = 0;
  dd->zol  = 0;
  dd->xo   = 0;
  dd->yo   = 0;
  dd->zo   = 0;
  dd->Mo   = -1;
  dd->No   = -1;
  dd->Po   = -1;

  dd->gtol = NULL;
  dd->ltol = NULL;
  dd->ao   = NULL;
  PetscCall(PetscStrallocpy(AOBASIC, (char **)&dd->aotype));
  dd->base         = -1;
  dd->bx           = DM_BOUNDARY_NONE;
  dd->by           = DM_BOUNDARY_NONE;
  dd->bz           = DM_BOUNDARY_NONE;
  dd->stencil_type = DMDA_STENCIL_BOX;
  dd->interptype   = DMDA_Q1;
  dd->lx           = NULL;
  dd->ly           = NULL;
  dd->lz           = NULL;

  dd->elementtype = DMDA_ELEMENT_Q1;

  da->ops->globaltolocalbegin        = DMGlobalToLocalBegin_DA;
  da->ops->globaltolocalend          = DMGlobalToLocalEnd_DA;
  da->ops->localtoglobalbegin        = DMLocalToGlobalBegin_DA;
  da->ops->localtoglobalend          = DMLocalToGlobalEnd_DA;
  da->ops->localtolocalbegin         = DMLocalToLocalBegin_DA;
  da->ops->localtolocalend           = DMLocalToLocalEnd_DA;
  da->ops->createglobalvector        = DMCreateGlobalVector_DA;
  da->ops->createlocalvector         = DMCreateLocalVector_DA;
  da->ops->createinterpolation       = DMCreateInterpolation_DA;
  da->ops->getcoloring               = DMCreateColoring_DA;
  da->ops->creatematrix              = DMCreateMatrix_DA;
  da->ops->refine                    = DMRefine_DA;
  da->ops->coarsen                   = DMCoarsen_DA;
  da->ops->refinehierarchy           = DMRefineHierarchy_DA;
  da->ops->coarsenhierarchy          = DMCoarsenHierarchy_DA;
  da->ops->createinjection           = DMCreateInjection_DA;
  da->ops->hascreateinjection        = DMHasCreateInjection_DA;
  da->ops->destroy                   = DMDestroy_DA;
  da->ops->view                      = NULL;
  da->ops->setfromoptions            = DMSetFromOptions_DA;
  da->ops->setup                     = DMSetUp_DA;
  da->ops->createlocalsection        = DMCreateLocalSection_DA;
  da->ops->clone                     = DMClone_DA;
  da->ops->load                      = DMLoad_DA;
  da->ops->createcoordinatedm        = DMCreateCoordinateDM_DA;
  da->ops->createcellcoordinatedm    = NULL;
  da->ops->createsubdm               = DMCreateSubDM_DA;
  da->ops->createfielddecomposition  = DMCreateFieldDecomposition_DA;
  da->ops->createdomaindecomposition = DMCreateDomainDecomposition_DA;
  da->ops->createddscatters          = DMCreateDomainDecompositionScatters_DA;
  da->ops->getdimpoints              = DMGetDimPoints_DA;
  da->ops->getneighbors              = DMGetNeighbors_DA;
  da->ops->getlocalboundingbox       = DMGetLocalBoundingBox_DA;
  da->ops->locatepoints              = DMLocatePoints_DA_Regular;
  da->ops->getcompatibility          = DMGetCompatibility_DA;
  PetscCall(PetscObjectComposeFunction((PetscObject)da, "DMSetUpGLVisViewer_C", DMSetUpGLVisViewer_DMDA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMDACreate - Creates a `DMDA` object for managing structured grids.

  Collective

  Input Parameter:
. comm - The communicator for the `DMDA` object

  Output Parameter:
. da - the `DMDA` object

  Level: advanced

  Notes:
  See [](sec_struct) for details on the construction of a `DMDA`

  `DMDACreate1d()`, `DMDACreate2d()`, and `DMDACreate3d()` are convenience routines to quickly completely create a `DMDA`

.seealso: [](sec_struct), `DM`, `DMDA`, `DMSetUp()`, `DMDASetSizes()`, `DMClone()`, `DMDACreate1d()`, `DMDACreate2d()`, `DMDACreate3d()`
@*/
PetscErrorCode DMDACreate(MPI_Comm comm, DM *da)
{
  PetscFunctionBegin;
  PetscAssertPointer(da, 2);
  PetscCall(DMCreate(comm, da));
  PetscCall(DMSetType(*da, DMDA));
  PetscFunctionReturn(PETSC_SUCCESS);
}
