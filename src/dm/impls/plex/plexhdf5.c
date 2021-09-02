#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/viewerhdf5impl.h>
#include <petsclayouthdf5.h>

PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);

//TODO move to PetscLayout, publish PetscLayoutView()
static PetscErrorCode PetscLayoutView_ASCII(PetscLayout l, const char name[], PetscViewer v)
{
  MPI_Comm comm = l->comm;
  PetscMPIInt rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPushSynchronized(v);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d] PetscLayout %s n N rstart rend %D %D %D %D\n", rank, name, l->n, l->N, l->rstart, l->rend);CHKERRQ(ierr);
  ierr = PetscViewerFlush(v);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode DMSequenceView_HDF5(DM dm, const char *seqname, PetscInt seqnum, PetscScalar value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp);CHKERRQ(ierr);
  ierr = VecSetBlockSize(stamp, 1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) stamp, seqname);CHKERRQ(ierr);
  if (!rank) {
    PetscReal timeScale;
    PetscBool istime;

    ierr = PetscStrncmp(seqname, "time", 5, &istime);CHKERRQ(ierr);
    if (istime) {ierr = DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale);CHKERRQ(ierr); value *= timeScale;}
    ierr = VecSetValue(stamp, 0, value, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(stamp);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(stamp);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr); /* seqnum < 0 jumps out above */
  ierr = VecView(stamp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&stamp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSequenceLoad_HDF5_Internal(DM dm, const char *seqname, PetscInt seqnum, PetscScalar *value, PetscViewer viewer)
{
  Vec            stamp;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (seqnum < 0) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
  ierr = VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp);CHKERRQ(ierr);
  ierr = VecSetBlockSize(stamp, 1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) stamp, seqname);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);  /* seqnum < 0 jumps out above */
  ierr = VecLoad(stamp, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  if (!rank) {
    const PetscScalar *a;
    PetscReal timeScale;
    PetscBool istime;

    ierr = VecGetArrayRead(stamp, &a);CHKERRQ(ierr);
    *value = a[0];
    ierr = VecRestoreArrayRead(stamp, &a);CHKERRQ(ierr);
    ierr = PetscStrncmp(seqname, "time", 5, &istime);CHKERRQ(ierr);
    if (istime) {ierr = DMPlexGetScale(dm, PETSC_UNIT_TIME, &timeScale);CHKERRQ(ierr); *value /= timeScale;}
  }
  ierr = VecDestroy(&stamp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateCutVertexLabel_Private(DM dm, DMLabel cutLabel, DMLabel *cutVertexLabel)
{
  IS              cutcells = NULL;
  const PetscInt *cutc;
  PetscInt        cellHeight, vStart, vEnd, cStart, cEnd, c;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!cutLabel) PetscFunctionReturn(0);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  /* Label vertices that should be duplicated */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Cut Vertices", cutVertexLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(cutLabel, 2, &cutcells);CHKERRQ(ierr);
  if (cutcells) {
    PetscInt n;

    ierr = ISGetIndices(cutcells, &cutc);CHKERRQ(ierr);
    ierr = ISGetLocalSize(cutcells, &n);CHKERRQ(ierr);
    for (c = 0; c < n; ++c) {
      if ((cutc[c] >= cStart) && (cutc[c] < cEnd)) {
        PetscInt *closure = NULL;
        PetscInt  closureSize, cl, value;

        ierr = DMPlexGetTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (cl = 0; cl < closureSize*2; cl += 2) {
          if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) {
            ierr = DMLabelGetValue(cutLabel, closure[cl], &value);CHKERRQ(ierr);
            if (value == 1) {
              ierr = DMLabelSetValue(*cutVertexLabel, closure[cl], 1);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cutc[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(cutcells, &cutc);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&cutcells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_Local_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM                      dm;
  DM                      dmBC;
  PetscSection            section, sectionGlobal;
  Vec                     gv;
  const char             *name;
  PetscViewerVTKFieldType ft;
  PetscViewerFormat       format;
  PetscInt                seqnum;
  PetscReal               seqval;
  PetscBool               isseq;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum, &seqval);CHKERRQ(ierr);
  ierr = DMSequenceView_HDF5(dm, "time", seqnum, (PetscScalar) seqval, viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = DMGetOutputDM(dm, &dmBC);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmBC, &sectionGlobal);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmBC, &gv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) gv, name);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmBC, v, INSERT_VALUES, gv);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dmBC, v, INSERT_VALUES, gv);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) gv, VECSEQ, &isseq);CHKERRQ(ierr);
  if (isseq) {ierr = VecView_Seq(gv, viewer);CHKERRQ(ierr);}
  else       {ierr = VecView_MPI(gv, viewer);CHKERRQ(ierr);}
  if (format == PETSC_VIEWER_HDF5_VIZ) {
    /* Output visualization representation */
    PetscInt numFields, f;
    DMLabel  cutLabel, cutVertexLabel = NULL;

    ierr = PetscSectionGetNumFields(section, &numFields);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      Vec         subv;
      IS          is;
      const char *fname, *fgroup;
      char        subname[PETSC_MAX_PATH_LEN];
      PetscInt    pStart, pEnd;

      ierr = DMPlexGetFieldType_Internal(dm, section, f, &pStart, &pEnd, &ft);CHKERRQ(ierr);
      fgroup = (ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_POINT_FIELD) ? "/vertex_fields" : "/cell_fields";
      ierr = PetscSectionGetFieldName(section, f, &fname);CHKERRQ(ierr);
      if (!fname) continue;
      ierr = PetscViewerHDF5PushGroup(viewer, fgroup);CHKERRQ(ierr);
      if (cutLabel) {
        const PetscScalar *ga;
        PetscScalar       *suba;
        PetscInt           Nc, gstart, subSize = 0, extSize = 0, subOff = 0, newOff = 0, p;

        ierr = DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldComponents(section, f, &Nc);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, fdof = 0, val;

          ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
          if (gdof > 0) {ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);}
          subSize += fdof;
          ierr = DMLabelGetValue(cutVertexLabel, p, &val);CHKERRQ(ierr);
          if (val == 1) extSize += fdof;
        }
        ierr = VecCreate(PetscObjectComm((PetscObject) gv), &subv);CHKERRQ(ierr);
        ierr = VecSetSizes(subv, subSize+extSize, PETSC_DETERMINE);CHKERRQ(ierr);
        ierr = VecSetBlockSize(subv, Nc);CHKERRQ(ierr);
        ierr = VecSetType(subv, VECSTANDARD);CHKERRQ(ierr);
        ierr = VecGetOwnershipRange(gv, &gstart, NULL);CHKERRQ(ierr);
        ierr = VecGetArrayRead(gv, &ga);CHKERRQ(ierr);
        ierr = VecGetArray(subv, &suba);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
          PetscInt gdof, goff, val;

          ierr = PetscSectionGetDof(sectionGlobal, p, &gdof);CHKERRQ(ierr);
          if (gdof > 0) {
            PetscInt fdof, fc, f2, poff = 0;

            ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
            /* Can get rid of this loop by storing field information in the global section */
            for (f2 = 0; f2 < f; ++f2) {
              ierr  = PetscSectionGetFieldDof(section, p, f2, &fdof);CHKERRQ(ierr);
              poff += fdof;
            }
            ierr = PetscSectionGetFieldDof(section, p, f, &fdof);CHKERRQ(ierr);
            for (fc = 0; fc < fdof; ++fc, ++subOff) suba[subOff] = ga[goff+poff+fc - gstart];
            ierr = DMLabelGetValue(cutVertexLabel, p, &val);CHKERRQ(ierr);
            if (val == 1) {
              for (fc = 0; fc < fdof; ++fc, ++newOff) suba[subSize+newOff] = ga[goff+poff+fc - gstart];
            }
          }
        }
        ierr = VecRestoreArrayRead(gv, &ga);CHKERRQ(ierr);
        ierr = VecRestoreArray(subv, &suba);CHKERRQ(ierr);
        ierr = DMLabelDestroy(&cutVertexLabel);CHKERRQ(ierr);
      } else {
        ierr = PetscSectionGetField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);
      }
      ierr = PetscStrncpy(subname, name,sizeof(subname));CHKERRQ(ierr);
      ierr = PetscStrlcat(subname, "_",sizeof(subname));CHKERRQ(ierr);
      ierr = PetscStrlcat(subname, fname,sizeof(subname));CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) subv, subname);CHKERRQ(ierr);
      if (isseq) {ierr = VecView_Seq(subv, viewer);CHKERRQ(ierr);}
      else       {ierr = VecView_MPI(subv, viewer);CHKERRQ(ierr);}
      if ((ft == PETSC_VTK_POINT_VECTOR_FIELD) || (ft == PETSC_VTK_CELL_VECTOR_FIELD)) {
        ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, "vector_field_type", PETSC_STRING, "vector");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) subv, "vector_field_type", PETSC_STRING, "scalar");CHKERRQ(ierr);
      }
      if (cutLabel) {ierr = VecDestroy(&subv);CHKERRQ(ierr);}
      else          {ierr = PetscSectionRestoreField_Internal(section, sectionGlobal, gv, f, pStart, pEnd, &is, &subv);CHKERRQ(ierr);}
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreGlobalVector(dmBC, &gv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  PetscObject    isZero;
  const char    *name;
  PetscReal      time;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) v, "__Vec_bc_zero__", &isZero);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", isZero);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, v, INSERT_VALUES, locv);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, NULL, &time);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locv, time, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_VIZ);CHKERRQ(ierr);
  ierr = VecView_Plex_Local_HDF5_Internal(locv, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) locv, "__Vec_bc_zero__", NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecView_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  PetscBool      isseq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) v, VECSEQ, &isseq);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  if (isseq) {ierr = VecView_Seq(v, viewer);CHKERRQ(ierr);}
  else       {ierr = VecView_MPI(v, viewer);CHKERRQ(ierr);}
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_HDF5_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  Vec            locv;
  const char    *name;
  PetscInt       seqnum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locv);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) locv, name);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  }
  ierr = VecLoad_Plex_Local(locv, viewer);CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, locv, INSERT_VALUES, v);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, locv, INSERT_VALUES, v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecLoad_Plex_HDF5_Native_Internal(Vec v, PetscViewer viewer)
{
  DM             dm;
  PetscInt       seqnum;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetOutputSequenceNumber(dm, &seqnum, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/fields");CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
  }
  ierr = VecLoad_Default(v, viewer);CHKERRQ(ierr);
  if (seqnum >= 0) {
    ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTopologyView_HDF5_Private(DM dm, IS globalPointNumbers, PetscViewer viewer, PetscInt pStart, PetscInt pEnd, const char pointsName[], const char coneSizesName[], const char conesName[], const char orientationsName[])
{
  IS              coneSizesIS, conesIS, orientationsIS;
  const PetscInt *gpoint;
  PetscInt       *coneSizes, *cones, *orientations;
  PetscInt        nPoints = 0, conesSize = 0;
  PetscInt        p, c, s;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    if (gpoint[p] >= 0) {
      PetscInt coneSize;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      nPoints += 1;
      conesSize += coneSize;
    }
  }
  ierr = PetscMalloc1(nPoints, &coneSizes);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &cones);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &orientations);CHKERRQ(ierr);
  for (p = pStart, c = 0, s = 0; p < pEnd; ++p) {
    //TODO Can gpoint[] elements really be negative? What does it mean? Should be explained in DMPlexCreateNumbering() manpage.
    if (gpoint[p] >= 0) {
      const PetscInt *cone, *ornt;
      PetscInt        coneSize, cp;

      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
      coneSizes[s] = coneSize;
      for (cp = 0; cp < coneSize; ++cp, ++c) {
        cones[c] = gpoint[cone[cp]] < 0 ? -(gpoint[cone[cp]]+1) : gpoint[cone[cp]];
        orientations[c] = ornt[cp];
      }
      ++s;
    }
  }
  if (s != nPoints) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of points %d != %d", s, nPoints);
  if (c != conesSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cone points %d != %d", c, conesSize);
  ierr = ISCreateGeneral(comm, nPoints, coneSizes, PETSC_OWN_POINTER, &coneSizesIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, conesSize, cones, PETSC_OWN_POINTER, &conesIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm, conesSize, orientations, PETSC_OWN_POINTER, &orientationsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) conesIS, conesName);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orientationsIS, orientationsName);CHKERRQ(ierr);
  ierr = ISView(coneSizesIS, viewer);CHKERRQ(ierr);
  ierr = ISView(conesIS, viewer);CHKERRQ(ierr);
  ierr = ISView(orientationsIS, viewer);CHKERRQ(ierr);
  ierr = ISDestroy(&coneSizesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&orientationsIS);CHKERRQ(ierr);
  if (pointsName) {
    IS        pointsIS;
    PetscInt  *points;

    ierr = PetscMalloc1(nPoints, &points);CHKERRQ(ierr);
    for (p = pStart, c = 0, s = 0; p < pEnd; ++p) {
      if (gpoint[p] >= 0) {
        points[s] = gpoint[p];
        ++s;
      }
    }
    ierr = ISCreateGeneral(comm, nPoints, points, PETSC_OWN_POINTER, &pointsIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) pointsIS, pointsName);CHKERRQ(ierr);
    ierr = ISView(pointsIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&pointsIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTopologyView_HDF5_v1(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const char     *pointsName, *coneSizesName, *conesName, *orientationsName;
  PetscInt        pStart, pEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  pointsName        = "order";
  coneSizesName     = "cones";
  conesName         = "cells";
  orientationsName  = "orientation";
  ierr = PetscViewerHDF5PushGroup(viewer, "/topology");CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexTopologyView_HDF5_Private(dm, globalPointNumbers, viewer, pStart, pEnd, pointsName, coneSizesName, conesName, orientationsName);CHKERRQ(ierr);
  {
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, conesName, "cell_dim", PETSC_INT, (void *) &dim);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//TODO get this numbering right away without needing this function
/* Renumber global point numbers so that they are 0-based per stratum */
static PetscErrorCode RenumberGlobalPointNumbersPerStratum_Private(DM dm, IS globalPointNumbers, IS *newGlobalPointNumbers, IS *strataPermutation)
{
  PetscInt        d, depth, p, n;
  PetscInt       *offsets;
  const PetscInt *gpn;
  PetscInt       *ngpn;
  MPI_Comm        comm;
  PetscBool       debug = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_topology_view_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(globalPointNumbers, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(globalPointNumbers, &gpn);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &ngpn);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc1(depth+1, &offsets);CHKERRQ(ierr);
  for (d = 0; d <= depth; d++) {
    PetscInt pStart, pEnd;

    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    offsets[d] = PETSC_MAX_INT;
    for (p = pStart; p < pEnd; p++) {
      if (gpn[p] >= 0 && gpn[p] < offsets[d]) offsets[d] = gpn[p];
    }
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE, offsets, depth+1, MPIU_INT, MPI_MIN, comm);CHKERRQ(ierr);
  for (d = 0; d <= depth; d++) {
    PetscInt pStart, pEnd;

    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      ngpn[p] = gpn[p] - PetscSign(gpn[p]) * offsets[d];
    }
  }
  ierr = ISRestoreIndices(globalPointNumbers, &gpn);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)globalPointNumbers), n, ngpn, PETSC_OWN_POINTER, newGlobalPointNumbers);CHKERRQ(ierr);
  {
    PetscInt *perm;

    ierr = PetscMalloc1(depth+1, &perm);CHKERRQ(ierr);
    for (d = 0; d <= depth; d++) perm[d] = d;
    ierr = PetscSortIntWithPermutation(depth+1, offsets, perm);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, depth+1, perm, PETSC_OWN_POINTER, strataPermutation);CHKERRQ(ierr);
  }
  if (debug) {
    PetscViewer v = PETSC_VIEWER_STDOUT_(comm);

    ierr = PetscViewerASCIIPrintf(v, "RenumberGlobalPointNumbersPerStratum_Private offsets:\n");CHKERRQ(ierr);
    ierr = PetscIntView(depth+1, offsets, v);CHKERRQ(ierr);
  }
  ierr = PetscFree(offsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTopologyView_HDF5_v2(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  IS              globalPointNumbers0, strataPermutation;
  const char     *coneSizesName, *conesName, *orientationsName;
  PetscInt        depth, d;
  PetscBool       debug = PETSC_FALSE;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  coneSizesName     = "cone_sizes";
  conesName         = "cones";
  orientationsName  = "orientations";
  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_topology_view_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "topology");CHKERRQ(ierr);
  {
    PetscInt ver = 2, dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "cell_dim", PETSC_INT, &dim);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "depth", PETSC_INT, &depth);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "version", PETSC_INT, &ver);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer, "strata");CHKERRQ(ierr);
  ierr = RenumberGlobalPointNumbersPerStratum_Private(dm, globalPointNumbers, &globalPointNumbers0, &strataPermutation);
  /* TODO dirty trick to save serial IS using the same parallel viewer */
  {
    IS              spOnComm;
    PetscInt        n = 0, N;
    const PetscInt  *idx = NULL;
    const PetscInt *old;
    PetscMPIInt     rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    ierr = ISGetLocalSize(strataPermutation, &N);CHKERRQ(ierr);
    ierr = ISGetIndices(strataPermutation, &old);CHKERRQ(ierr);
    if (!rank) {
      n   = N;
      idx = old;
    }
    ierr = ISCreateGeneral(comm, n, idx, PETSC_COPY_VALUES, &spOnComm);CHKERRQ(ierr);
    ierr = ISRestoreIndices(strataPermutation, &old);CHKERRQ(ierr);
    ierr = ISDestroy(&strataPermutation);CHKERRQ(ierr);
    strataPermutation = spOnComm;
  }
  ierr = PetscObjectSetName((PetscObject) strataPermutation, "permutation");CHKERRQ(ierr);
  ierr = ISView(strataPermutation, viewer);CHKERRQ(ierr);
  if (debug) {
    PetscViewer v = PETSC_VIEWER_STDOUT_(comm);
    ierr = PetscObjectSetName((PetscObject) globalPointNumbers, "globalPointNumbers");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) globalPointNumbers0, "globalPointNumbers0");CHKERRQ(ierr);
    ierr = ISView(globalPointNumbers, v);CHKERRQ(ierr);
    ierr = ISView(globalPointNumbers0, v);CHKERRQ(ierr);
    ierr = ISView(strataPermutation, v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
  }
  for (d = 0; d <= depth; d++) {
    PetscInt pStart, pEnd;
    char     group[128];

    ierr = PetscSNPrintf(group, sizeof(group), "%D", d);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexTopologyView_HDF5_Private(dm, globalPointNumbers0, viewer, pStart, pEnd, NULL, coneSizesName, conesName, orientationsName);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr); /* strata */
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr); /* topology */
  ierr = ISDestroy(&globalPointNumbers0);CHKERRQ(ierr);
  ierr = ISDestroy(&strataPermutation);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTopologyView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  PetscInt        version = 1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),((PetscObject)dm)->prefix,"DMPlex HDF5 Loader Options","PetscViewer");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_plex_hdf5_topology_version","version of DMPlex HDF5 topology serialization",NULL,version,&version,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  switch (version) {
    case 1: ierr = DMPlexTopologyView_HDF5_v1(dm, globalPointNumbers, viewer);CHKERRQ(ierr); break;
    case 2: ierr = DMPlexTopologyView_HDF5_v2(dm, globalPointNumbers, viewer);CHKERRQ(ierr); break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DMPlexTopologyView() for topology version %D not implemented yet", version);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateConesIS_Private(DM dm, PetscInt cStart, PetscInt cEnd, IS globalCellNumbers, PetscInt *numCorners, IS *cellIS)
{
  PetscSF         sfPoint;
  DMLabel         cutLabel, cutVertexLabel = NULL;
  IS              globalVertexNumbers, cutvertices = NULL;
  const PetscInt *gcell, *gvertex, *cutverts = NULL;
  PetscInt       *vertices;
  PetscInt        conesSize = 0;
  PetscInt        dim, numCornersLocal = 0, cell, vStart, vEnd, vExtra = 0, v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  *numCorners = 0;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(globalCellNumbers, &gcell);CHKERRQ(ierr);

  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, v, Nc = 0;

    if (gcell[cell] < 0) continue;
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (v = 0; v < closureSize*2; v += 2) {
      if ((closure[v] >= vStart) && (closure[v] < vEnd)) ++Nc;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    conesSize += Nc;
    if (!numCornersLocal)           numCornersLocal = Nc;
    else if (numCornersLocal != Nc) numCornersLocal = 1;
  }
  ierr = MPIU_Allreduce(&numCornersLocal, numCorners, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
  if (numCornersLocal && (numCornersLocal != *numCorners || *numCorners == 1)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Visualization topology currently only supports identical cell shapes");
  /* Handle periodic cuts by identifying vertices which should be duplicated */
  ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);
  ierr = DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel);CHKERRQ(ierr);
  if (cutVertexLabel) {ierr = DMLabelGetStratumIS(cutVertexLabel, 1, &cutvertices);CHKERRQ(ierr);}
  if (cutvertices) {
    ierr = ISGetIndices(cutvertices, &cutverts);CHKERRQ(ierr);
    ierr = ISGetLocalSize(cutvertices, &vExtra);CHKERRQ(ierr);
  }
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  if (cutLabel) {
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    PetscInt           nroots, nleaves;

    ierr = PetscSFGetGraph(sfPoint, &nroots, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
    if (nleaves < 0) {
      ierr = PetscObjectReference((PetscObject) sfPoint);CHKERRQ(ierr);
    } else {
      ierr = PetscSFCreate(PetscObjectComm((PetscObject) sfPoint), &sfPoint);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(sfPoint, nroots+vExtra, nleaves, ilocal, PETSC_USE_POINTER, iremote, PETSC_USE_POINTER);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject) sfPoint);CHKERRQ(ierr);
  }
  /* Number all vertices */
  ierr = DMPlexCreateNumbering_Plex(dm, vStart, vEnd+vExtra, 0, NULL, sfPoint, &globalVertexNumbers);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfPoint);CHKERRQ(ierr);
  /* Create cones */
  ierr = ISGetIndices(globalVertexNumbers, &gvertex);CHKERRQ(ierr);
  ierr = PetscMalloc1(conesSize, &vertices);CHKERRQ(ierr);
  for (cell = cStart, v = 0; cell < cEnd; ++cell) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, Nc = 0, p, value = -1;
    PetscBool replace;

    if (gcell[cell] < 0) continue;
    if (cutLabel) {ierr = DMLabelGetValue(cutLabel, cell, &value);CHKERRQ(ierr);}
    replace = (value == 2) ? PETSC_TRUE : PETSC_FALSE;
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (p = 0; p < closureSize*2; p += 2) {
      if ((closure[p] >= vStart) && (closure[p] < vEnd)) {
        closure[Nc++] = closure[p];
      }
    }
    ierr = DMPlexReorderCell(dm, cell, closure);CHKERRQ(ierr);
    for (p = 0; p < Nc; ++p) {
      PetscInt nv, gv = gvertex[closure[p] - vStart];

      if (replace) {
        ierr = PetscFindInt(closure[p], vExtra, cutverts, &nv);CHKERRQ(ierr);
        if (nv >= 0) gv = gvertex[vEnd - vStart + nv];
      }
      vertices[v++] = gv < 0 ? -(gv+1) : gv;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(globalVertexNumbers, &gvertex);CHKERRQ(ierr);
  ierr = ISDestroy(&globalVertexNumbers);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalCellNumbers, &gcell);CHKERRQ(ierr);
  if (cutvertices) {ierr = ISRestoreIndices(cutvertices, &cutverts);CHKERRQ(ierr);}
  ierr = ISDestroy(&cutvertices);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&cutVertexLabel);CHKERRQ(ierr);
  if (v != conesSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Total number of cell vertices %d != %d", v, conesSize);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), conesSize, vertices, PETSC_OWN_POINTER, cellIS);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize((*cellIS)->map, *numCorners);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *cellIS, "cells");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteTopology_Vertices_HDF5_Static(DM dm, IS globalCellNumbers, PetscViewer viewer)
{
  DM              cdm;
  DMLabel         depthLabel, ctLabel;
  IS              cellIS;
  PetscInt        dim, depth, cellHeight, c;
  hid_t           fileId, groupId;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Gclose,(groupId));

  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMPlexGetCellTypeLabel(dm, &ctLabel);CHKERRQ(ierr);
  for (c = 0; c < DM_NUM_POLYTOPES; ++c) {
    const DMPolytopeType ict = (DMPolytopeType) c;
    PetscInt             pStart, pEnd, dep, numCorners, n = 0;
    PetscBool            output = PETSC_FALSE, doOutput;

    if (ict == DM_POLYTOPE_FV_GHOST) continue;
    ierr = DMLabelGetStratumBounds(ctLabel, ict, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart >= 0) {
      ierr = DMLabelGetValue(depthLabel, pStart, &dep);CHKERRQ(ierr);
      if (dep == depth - cellHeight) output = PETSC_TRUE;
    }
    ierr = MPI_Allreduce(&output, &doOutput, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject) dm));CHKERRMPI(ierr);
    if (!doOutput) continue;
    ierr = CreateConesIS_Private(dm, pStart, pEnd, globalCellNumbers, &numCorners,  &cellIS);CHKERRQ(ierr);
    if (!n) {
      ierr = PetscViewerHDF5PushGroup(viewer, "/viz/topology");CHKERRQ(ierr);
    } else {
      char group[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/viz/topology_%D", n);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
    }
    ierr = ISView(cellIS, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellIS, "cell_corners", PETSC_INT, (void *) &numCorners);CHKERRQ(ierr);
    ierr = PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject) cellIS, "cell_dim",     PETSC_INT, (void *) &dim);CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ++n;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCoordinatesView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  DM             cdm;
  Vec            coordinates, newcoords;
  PetscReal      lengthScale;
  PetscInt       m, M, bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) coordinates), &newcoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newcoords, "vertices");CHKERRQ(ierr);
  ierr = VecGetSize(coordinates, &M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &m);CHKERRQ(ierr);
  ierr = VecSetSizes(newcoords, m, M);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);
  ierr = VecSetType(newcoords,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecCopy(coordinates, newcoords);CHKERRQ(ierr);
  ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
  /* Did not use DMGetGlobalVector() in order to bypass default group assignment */
  ierr = PetscViewerHDF5PushGroup(viewer, "/geometry");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = VecView(newcoords, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexWriteCoordinates_Vertices_HDF5_Static(DM dm, PetscViewer viewer)
{
  DM               cdm;
  Vec              coordinatesLocal, newcoords;
  PetscSection     cSection, cGlobalSection;
  PetscScalar     *coords, *ncoords;
  DMLabel          cutLabel, cutVertexLabel = NULL;
  const PetscReal *L;
  const DMBoundaryType *bd;
  PetscReal        lengthScale;
  PetscInt         vStart, vEnd, v, bs, N, coordSize, dof, off, d;
  PetscBool        localized, embedded;
  hid_t            fileId, groupId;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinatesLocal);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinatesLocal, &bs);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalized(dm, &localized);CHKERRQ(ierr);
  if (localized == PETSC_FALSE) PetscFunctionReturn(0);
  ierr = DMGetPeriodicity(dm, NULL, NULL, &L, &bd);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &cSection);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(cdm, &cGlobalSection);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "periodic_cut", &cutLabel);CHKERRQ(ierr);
  N    = 0;

  ierr = DMPlexCreateCutVertexLabel_Private(dm, cutLabel, &cutVertexLabel);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &newcoords);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(cSection, vStart, &dof);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "DOF: %D\n", dof);CHKERRQ(ierr);
  embedded  = (PetscBool) (L && dof == 2 && !cutLabel);
  if (cutVertexLabel) {
    ierr = DMLabelGetStratumSize(cutVertexLabel, 1, &v);CHKERRQ(ierr);
    N   += dof*v;
  }
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(cGlobalSection, v, &dof);CHKERRQ(ierr);
    if (dof < 0) continue;
    if (embedded) N += dof+1;
    else          N += dof;
  }
  if (embedded) {ierr = VecSetBlockSize(newcoords, bs+1);CHKERRQ(ierr);}
  else          {ierr = VecSetBlockSize(newcoords, bs);CHKERRQ(ierr);}
  ierr = VecSetSizes(newcoords, N, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(newcoords, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesLocal, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(newcoords,        &ncoords);CHKERRQ(ierr);
  coordSize = 0;
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(cGlobalSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cSection, v, &off);CHKERRQ(ierr);
    if (dof < 0) continue;
    if (embedded) {
      if ((bd[0] == DM_BOUNDARY_PERIODIC) && (bd[1] == DM_BOUNDARY_PERIODIC)) {
        PetscReal theta, phi, r, R;
        /* XY-periodic */
        /* Suppose its an y-z circle, then
             \hat r = (0, cos(th), sin(th)) \hat x = (1, 0, 0)
           and the circle in that plane is
             \hat r cos(phi) + \hat x sin(phi) */
        theta = 2.0*PETSC_PI*PetscRealPart(coords[off+1])/L[1];
        phi   = 2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0];
        r     = L[0]/(2.0*PETSC_PI * 2.0*L[1]);
        R     = L[1]/(2.0*PETSC_PI);
        ncoords[coordSize++] =  PetscSinReal(phi) * r;
        ncoords[coordSize++] = -PetscCosReal(theta) * (R + r * PetscCosReal(phi));
        ncoords[coordSize++] =  PetscSinReal(theta) * (R + r * PetscCosReal(phi));
      } else if ((bd[0] == DM_BOUNDARY_PERIODIC)) {
        /* X-periodic */
        ncoords[coordSize++] = -PetscCosReal(2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0])*(L[0]/(2.0*PETSC_PI));
        ncoords[coordSize++] = coords[off+1];
        ncoords[coordSize++] = PetscSinReal(2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0])*(L[0]/(2.0*PETSC_PI));
      } else if ((bd[1] == DM_BOUNDARY_PERIODIC)) {
        /* Y-periodic */
        ncoords[coordSize++] = coords[off+0];
        ncoords[coordSize++] = PetscSinReal(2.0*PETSC_PI*PetscRealPart(coords[off+1])/L[1])*(L[1]/(2.0*PETSC_PI));
        ncoords[coordSize++] = -PetscCosReal(2.0*PETSC_PI*PetscRealPart(coords[off+1])/L[1])*(L[1]/(2.0*PETSC_PI));
      } else if ((bd[0] == DM_BOUNDARY_TWIST)) {
        PetscReal phi, r, R;
        /* Mobius strip */
        /* Suppose its an x-z circle, then
             \hat r = (-cos(phi), 0, sin(phi)) \hat y = (0, 1, 0)
           and in that plane we rotate by pi as we go around the circle
             \hat r cos(phi/2) + \hat y sin(phi/2) */
        phi   = 2.0*PETSC_PI*PetscRealPart(coords[off+0])/L[0];
        R     = L[0];
        r     = PetscRealPart(coords[off+1]) - L[1]/2.0;
        ncoords[coordSize++] = -PetscCosReal(phi) * (R + r * PetscCosReal(phi/2.0));
        ncoords[coordSize++] =  PetscSinReal(phi/2.0) * r;
        ncoords[coordSize++] =  PetscSinReal(phi) * (R + r * PetscCosReal(phi/2.0));
      } else SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Cannot handle periodicity in this domain");
    } else {
      for (d = 0; d < dof; ++d, ++coordSize) ncoords[coordSize] = coords[off+d];
    }
  }
  if (cutVertexLabel) {
    IS              vertices;
    const PetscInt *verts;
    PetscInt        n;

    ierr = DMLabelGetStratumIS(cutVertexLabel, 1, &vertices);CHKERRQ(ierr);
    if (vertices) {
      ierr = ISGetIndices(vertices, &verts);CHKERRQ(ierr);
      ierr = ISGetLocalSize(vertices, &n);CHKERRQ(ierr);
      for (v = 0; v < n; ++v) {
        ierr = PetscSectionGetDof(cSection, verts[v], &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(cSection, verts[v], &off);CHKERRQ(ierr);
        for (d = 0; d < dof; ++d) ncoords[coordSize++] = coords[off+d] + ((bd[d] == DM_BOUNDARY_PERIODIC) ? L[d] : 0.0);
      }
      ierr = ISRestoreIndices(vertices, &verts);CHKERRQ(ierr);
      ierr = ISDestroy(&vertices);CHKERRQ(ierr);
    }
  }
  if (coordSize != N) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatched sizes: %D != %D", coordSize, N);
  ierr = DMLabelDestroy(&cutVertexLabel);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesLocal, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(newcoords,        &ncoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newcoords, "vertices");CHKERRQ(ierr);
  ierr = VecScale(newcoords, lengthScale);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Gclose,(groupId));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/viz/geometry");CHKERRQ(ierr);
  ierr = VecView(newcoords, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&newcoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLabelsView_HDF5_Internal(DM dm, IS globalPointNumbers, PetscViewer viewer)
{
  const PetscInt   *gpoint;
  PetscInt          numLabels, l;
  hid_t             fileId, groupId;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = ISGetIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/labels");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  if (groupId != fileId) PetscStackCallHDF5(H5Gclose,(groupId));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label;
    const char     *name;
    IS              valueIS, pvalueIS, globalValueIS;
    const PetscInt *values;
    PetscInt        numValues, v;
    PetscBool       isDepth, output;
    char            group[PETSC_MAX_PATH_LEN];

    ierr = DMGetLabelName(dm, l, &name);CHKERRQ(ierr);
    ierr = DMGetLabelOutput(dm, name, &output);CHKERRQ(ierr);
    ierr = PetscStrncmp(name, "depth", 10, &isDepth);CHKERRQ(ierr);
    if (isDepth || !output) continue;
    ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
    ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s", name);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
    ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
    if (groupId != fileId) PetscStackCallHDF5(H5Gclose,(groupId));
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    /* Must copy to a new IS on the global comm */
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), numValues, values, PETSC_COPY_VALUES, &pvalueIS);CHKERRQ(ierr);
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISAllGather(pvalueIS, &globalValueIS);CHKERRQ(ierr);
    ierr = ISDestroy(&pvalueIS);CHKERRQ(ierr);
    ierr = ISSortRemoveDups(globalValueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(globalValueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(globalValueIS, &values);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      IS              stratumIS, globalStratumIS;
      const PetscInt *spoints = NULL;
      PetscInt       *gspoints, n = 0, gn, p;
      const char     *iname = "indices";

      ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s/%d", name, values[v]);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, values[v], &stratumIS);CHKERRQ(ierr);

      if (stratumIS) {ierr = ISGetLocalSize(stratumIS, &n);CHKERRQ(ierr);}
      if (stratumIS) {ierr = ISGetIndices(stratumIS, &spoints);CHKERRQ(ierr);}
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) ++gn;
      ierr = PetscMalloc1(gn,&gspoints);CHKERRQ(ierr);
      for (gn = 0, p = 0; p < n; ++p) if (gpoint[spoints[p]] >= 0) gspoints[gn++] = gpoint[spoints[p]];
      if (stratumIS) {ierr = ISRestoreIndices(stratumIS, &spoints);CHKERRQ(ierr);}
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), gn, gspoints, PETSC_OWN_POINTER, &globalStratumIS);CHKERRQ(ierr);
      if (stratumIS) {ierr = PetscObjectGetName((PetscObject) stratumIS, &iname);CHKERRQ(ierr);}
      ierr = PetscObjectSetName((PetscObject) globalStratumIS, iname);CHKERRQ(ierr);

      ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
      ierr = ISView(globalStratumIS, viewer);CHKERRQ(ierr);
      ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&globalStratumIS);CHKERRQ(ierr);
      ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(globalValueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&globalValueIS);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(globalPointNumbers, &gpoint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* We only write cells and vertices. Does this screw up parallel reading? */
PetscErrorCode DMPlexView_HDF5_Internal(DM dm, PetscViewer viewer)
{
  IS                globalPointNumbers;
  PetscViewerFormat format;
  PetscBool         viz_geom=PETSC_FALSE, xdmf_topo=PETSC_FALSE, petsc_topo=PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPlexCreatePointNumbering(dm, &globalPointNumbers);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesView_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
  ierr = DMPlexLabelsView_HDF5_Internal(dm, globalPointNumbers, viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "petsc_version_git", PETSC_STRING, PETSC_VERSION_GIT);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  switch (format) {
    case PETSC_VIEWER_HDF5_VIZ:
      viz_geom    = PETSC_TRUE;
      xdmf_topo   = PETSC_TRUE;
      break;
    case PETSC_VIEWER_HDF5_XDMF:
      xdmf_topo   = PETSC_TRUE;
      break;
    case PETSC_VIEWER_HDF5_PETSC:
      petsc_topo  = PETSC_TRUE;
      break;
    case PETSC_VIEWER_DEFAULT:
    case PETSC_VIEWER_NATIVE:
      viz_geom    = PETSC_TRUE;
      xdmf_topo   = PETSC_TRUE;
      petsc_topo  = PETSC_TRUE;
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "PetscViewerFormat %s not supported for HDF5 output.", PetscViewerFormats[format]);
  }

  if (viz_geom)   {ierr = DMPlexWriteCoordinates_Vertices_HDF5_Static(dm, viewer);CHKERRQ(ierr);}
  if (xdmf_topo)  {ierr = DMPlexWriteTopology_Vertices_HDF5_Static(dm, globalPointNumbers, viewer);CHKERRQ(ierr);}
  if (petsc_topo) {ierr = DMPlexTopologyView_HDF5_Internal(dm, globalPointNumbers, viewer);CHKERRQ(ierr);}

  ierr = ISDestroy(&globalPointNumbers);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexSectionView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm)
{
  MPI_Comm       comm;
  const char    *topologydm_name;
  const char    *sectiondm_name;
  PetscSection   gsection;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sectiondm, &comm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(sectiondm, &gsection);CHKERRQ(ierr);
  /* Save raw section */
  ierr = PetscSectionView(gsection, viewer);CHKERRQ(ierr);
  /* Save plex wrapper */
  {
    PetscInt        pStart, pEnd, p, n;
    IS              globalPointNumbers;
    const PetscInt *gpoints;
    IS              orderIS;
    PetscInt       *order;

    ierr = PetscSectionGetChart(gsection, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexCreatePointNumbering(dm, &globalPointNumbers);CHKERRQ(ierr);
    ierr = ISGetIndices(globalPointNumbers, &gpoints);CHKERRQ(ierr);
    for (p = pStart, n = 0; p < pEnd; ++p) if (gpoints[p] >= 0) n++;
    /* "order" is an array of global point numbers.
       When loading, it is used with topology/order array
       to match section points with plex topology points. */
    ierr = PetscMalloc1(n, &order);CHKERRQ(ierr);
    for (p = pStart, n = 0; p < pEnd; ++p) if (gpoints[p] >= 0) order[n++] = gpoints[p];
    ierr = ISRestoreIndices(globalPointNumbers, &gpoints);CHKERRQ(ierr);
    ierr = ISDestroy(&globalPointNumbers);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, n, order, PETSC_OWN_POINTER, &orderIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)orderIS, "order");CHKERRQ(ierr);
    ierr = ISView(orderIS, viewer);CHKERRQ(ierr);
    ierr = ISDestroy(&orderIS);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexGlobalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  const char     *topologydm_name;
  const char     *sectiondm_name;
  const char     *vec_name;
  PetscInt        bs;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;

    ierr = DMGetPointSF(dm, &pointsf);CHKERRQ(ierr);
    ierr = DMGetPointSF(sectiondm, &pointsf1);CHKERRQ(ierr);
    if (pointsf1 != pointsf) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  ierr = PetscObjectGetName((PetscObject)dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)vec, &vec_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "vecs");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vec_name);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vec, &bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *) &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec, 1);CHKERRQ(ierr);
  /* VecView(vec, viewer) would call (*vec->opt->view)(vec, viewer), but,    */
  /* if vec was created with DMGet{Global, Local}Vector(), vec->opt->view    */
  /* is set to VecView_Plex, which would save vec in a predefined location.  */
  /* To save vec in where we want, we create a new Vec (temp) with           */
  /* VecCreate(), wrap the vec data in temp, and call VecView(temp, viewer). */
  {
    Vec                temp;
    const PetscScalar *array;
    PetscLayout        map;

    ierr = VecCreate(PetscObjectComm((PetscObject)vec), &temp);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)temp, vec_name);CHKERRQ(ierr);
    ierr = VecGetLayout(vec, &map);CHKERRQ(ierr);
    ierr = VecSetLayout(temp, map);CHKERRQ(ierr);
    ierr = VecSetUp(temp);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec, &array);CHKERRQ(ierr);
    ierr = VecPlaceArray(temp, array);CHKERRQ(ierr);
    ierr = VecView(temp, viewer);CHKERRQ(ierr);
    ierr = VecResetArray(temp);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec, &array);CHKERRQ(ierr);
    ierr = VecDestroy(&temp);CHKERRQ(ierr);
  }
  ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexLocalVectorView_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, Vec vec)
{
  MPI_Comm        comm;
  const char     *topologydm_name;
  const char     *sectiondm_name;
  const char     *vec_name;
  PetscSection    section;
  PetscBool       includesConstraints;
  Vec             gvec;
  PetscInt        m, bs;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;

    ierr = DMGetPointSF(dm, &pointsf);CHKERRQ(ierr);
    ierr = DMGetPointSF(sectiondm, &pointsf1);CHKERRQ(ierr);
    if (pointsf1 != pointsf) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
  }
  ierr = PetscObjectGetName((PetscObject)dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)vec, &vec_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "vecs");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vec_name);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vec, &bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5WriteAttribute(viewer, NULL, "blockSize", PETSC_INT, (void *) &bs);CHKERRQ(ierr);
  ierr = VecCreate(comm, &gvec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)gvec, vec_name);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(sectiondm, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(section, &includesConstraints);CHKERRQ(ierr);
  if (includesConstraints) {ierr = PetscSectionGetStorageSize(section, &m);CHKERRQ(ierr);}
  else {ierr = PetscSectionGetConstrainedStorageSize(section, &m);CHKERRQ(ierr);}
  ierr = VecSetSizes(gvec, m, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetUp(gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(sectiondm, vec, INSERT_VALUES, gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(sectiondm, vec, INSERT_VALUES, gvec);CHKERRQ(ierr);
  ierr = VecView(gvec, viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscMPIInt rank;
  DM          dm;
  PetscViewer viewer;
  DMLabel     label;
} LabelCtx;

static herr_t ReadLabelStratumHDF5_Static(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)
{
  PetscViewer     viewer = ((LabelCtx *) op_data)->viewer;
  DMLabel         label  = ((LabelCtx *) op_data)->label;
  IS              stratumIS;
  const PetscInt *ind;
  PetscInt        value, N, i;
  const char     *lname;
  char            group[PETSC_MAX_PATH_LEN];
  PetscErrorCode  ierr;

  ierr = PetscOptionsStringToInt(name, &value);CHKERRQ(ierr);
  ierr = ISCreate(PetscObjectComm((PetscObject) viewer), &stratumIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) stratumIS, "indices");CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) label, &lname);CHKERRQ(ierr);
  ierr = PetscSNPrintf(group, PETSC_MAX_PATH_LEN, "/labels/%s/%s", lname, name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, group);CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, "indices", NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(stratumIS->map, !((LabelCtx *) op_data)->rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(stratumIS->map, N);CHKERRQ(ierr);
  }
  ierr = ISLoad(stratumIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = ISGetLocalSize(stratumIS, &N);CHKERRQ(ierr);
  ierr = ISGetIndices(stratumIS, &ind);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {ierr = DMLabelSetValue(label, ind[i], value);CHKERRQ(ierr);}
  ierr = ISRestoreIndices(stratumIS, &ind);CHKERRQ(ierr);
  ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
  return 0;
}

static herr_t ReadLabelHDF5_Static(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)
{
  DM             dm  = ((LabelCtx *) op_data)->dm;
  hsize_t        idx = 0;
  PetscErrorCode ierr;
  herr_t         err;

  ierr = DMCreateLabel(dm, name); if (ierr) return (herr_t) ierr;
  ierr = DMGetLabel(dm, name, &((LabelCtx *) op_data)->label); if (ierr) return (herr_t) ierr;
  PetscStackCall("H5Literate_by_name",err = H5Literate_by_name(g_id, name, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelStratumHDF5_Static, op_data, 0));
  return err;
}

PetscErrorCode DMPlexLabelsLoad_HDF5_Internal(DM dm, PetscViewer viewer)
{
  LabelCtx        ctx;
  hid_t           fileId, groupId;
  hsize_t         idx = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &ctx.rank);CHKERRMPI(ierr);
  ctx.dm     = dm;
  ctx.viewer = viewer;
  ierr = PetscViewerHDF5PushGroup(viewer, "/labels");CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &fileId, &groupId);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Literate,(groupId, H5_INDEX_NAME, H5_ITER_NATIVE, &idx, ReadLabelHDF5_Static, &ctx));
  PetscStackCallHDF5(H5Gclose,(groupId));
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTopologyLoad_HDF5_v1(DM dm, PetscViewer viewer)
{
  MPI_Comm        comm;
  const char     *pointsName, *coneSizesName, *conesName, *orientationsName;
  IS              pointsIS, coneSizesIS, conesIS, orientationsIS;
  const PetscInt *points, *coneSizes, *cones, *orientations;
  PetscInt       *cone, *ornt;
  PetscInt        dim, N, Np, pEnd, p, q, maxConeSize = 0, c;
  PetscMPIInt     size, rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  pointsName        = "order";
  coneSizesName     = "cones";
  conesName         = "cells";
  orientationsName  = "orientation";
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  /* Read topology */
  ierr = PetscViewerHDF5PushGroup(viewer, "/topology");CHKERRQ(ierr);
  ierr = ISCreate(comm, &pointsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) pointsIS, pointsName);CHKERRQ(ierr);
  ierr = ISCreate(comm, &coneSizesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName);CHKERRQ(ierr);
  ierr = ISCreate(comm, &conesIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) conesIS, conesName);CHKERRQ(ierr);
  ierr = ISCreate(comm, &orientationsIS);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) orientationsIS, orientationsName);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadObjectAttribute(viewer, (PetscObject) conesIS, "cell_dim", PETSC_INT, NULL, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  {
    /* Force serial load */
    ierr = PetscViewerHDF5ReadSizes(viewer, pointsName, NULL, &Np);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(pointsIS->map, !rank ? Np : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(pointsIS->map, Np);CHKERRQ(ierr);
    pEnd = !rank ? Np : 0;
    ierr = PetscViewerHDF5ReadSizes(viewer, coneSizesName, NULL, &Np);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(coneSizesIS->map, !rank ? Np : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(coneSizesIS->map, Np);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, conesName, NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(conesIS->map, !rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(conesIS->map, N);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, orientationsName, NULL, &N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(orientationsIS->map, !rank ? N : 0);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(orientationsIS->map, N);CHKERRQ(ierr);
  }
  ierr = ISLoad(pointsIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(coneSizesIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(conesIS, viewer);CHKERRQ(ierr);
  ierr = ISLoad(orientationsIS, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  /* Create Plex */
  ierr = DMPlexSetChart(dm, 0, pEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = ISGetIndices(coneSizesIS, &coneSizes);CHKERRQ(ierr);
  for (p = 0; p < pEnd; ++p) {
    ierr = DMPlexSetConeSize(dm, points[p], coneSizes[p]);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, coneSizes[p]);
  }
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = ISGetIndices(conesIS, &cones);CHKERRQ(ierr);
  ierr = ISGetIndices(orientationsIS, &orientations);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxConeSize,&cone,maxConeSize,&ornt);CHKERRQ(ierr);
  for (p = 0, q = 0; p < pEnd; ++p) {
    for (c = 0; c < coneSizes[p]; ++c, ++q) {
      cone[c] = cones[q];
      ornt[c] = orientations[q];
    }
    ierr = DMPlexSetCone(dm, points[p], cone);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(dm, points[p], ornt);CHKERRQ(ierr);
  }
  ierr = PetscFree2(cone,ornt);CHKERRQ(ierr);
  /* Clean-up */
  ierr = ISRestoreIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = ISRestoreIndices(coneSizesIS, &coneSizes);CHKERRQ(ierr);
  ierr = ISRestoreIndices(conesIS, &cones);CHKERRQ(ierr);
  ierr = ISRestoreIndices(orientationsIS, &orientations);CHKERRQ(ierr);
  ierr = ISDestroy(&pointsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&coneSizesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&orientationsIS);CHKERRQ(ierr);
  /* Fill in the rest of the topology structure */
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Representation of two DMPlex strata in 0-based global numbering */
struct _n_PlexLayer {
  PetscInt        d;
  IS              conesIS, orientationsIS;
  PetscSection    coneSizesSection;
  PetscLayout     vertexLayout;
  PetscSF         overlapSF, l2gSF; //TODO maybe confusing names (in DMPlex in general pointSF -> overlapSF, vertexSF -> localToGlobalSF)
  PetscInt        offset, conesOffset, leafOffset;
};
typedef struct _n_PlexLayer* PlexLayer;

static PetscErrorCode PlexLayerDestroy(PlexLayer *layer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!*layer) PetscFunctionReturn(0);
  ierr = PetscSectionDestroy(&(*layer)->coneSizesSection);CHKERRQ(ierr);
  ierr = ISDestroy(&(*layer)->conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&(*layer)->orientationsIS);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*layer)->overlapSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(*layer)->l2gSF);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&(*layer)->vertexLayout);CHKERRQ(ierr);
  ierr = PetscFree(*layer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PlexLayerCreate_Private(PlexLayer *layer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNew(layer);CHKERRQ(ierr);
  (*layer)->d = -1;
  (*layer)->offset = -1;
  (*layer)->conesOffset = -1;
  (*layer)->leafOffset = -1;
  PetscFunctionReturn(0);
}

static PetscErrorCode PlexLayerLoad_Private(PlexLayer layer, PetscViewer viewer, PetscInt d, PetscLayout pointsLayout)
{
  char            path[128];
  MPI_Comm        comm;
  const char     *coneSizesName, *conesName, *orientationsName;
  IS              coneSizesIS, conesIS, orientationsIS;
  PetscSection    coneSizesSection;
  PetscLayout     vertexLayout = NULL;
  PetscInt        s;
  PetscBool       debug = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  coneSizesName     = "cone_sizes";
  conesName         = "cones";
  orientationsName  = "orientations";

  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_topology_load_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer, &comm);CHKERRQ(ierr);

  /* query size of next lower depth stratum (next lower dimension) */
  if (d > 0) {
    PetscInt NVertices;
    ierr = PetscSNPrintf(path, sizeof(path), "%D/%s", d-1, coneSizesName);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadSizes(viewer, path, NULL, &NVertices);CHKERRQ(ierr);
    ierr = PetscLayoutCreate(comm, &vertexLayout);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(vertexLayout, NVertices);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(vertexLayout);CHKERRQ(ierr);
  }

  ierr = PetscSNPrintf(path, sizeof(path), "%D", d);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, path);CHKERRQ(ierr);

  /* create coneSizesSection from stored IS coneSizes */
  {
    const PetscInt *coneSizes;

    ierr = ISCreate(comm, &coneSizesIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) coneSizesIS, coneSizesName);CHKERRQ(ierr);
    if (pointsLayout) {
      ierr = ISSetLayout(coneSizesIS, pointsLayout);CHKERRQ(ierr);
    }
    ierr = ISLoad(coneSizesIS, viewer);CHKERRQ(ierr);
    if (!pointsLayout) {
      ierr = ISGetLayout(coneSizesIS, &pointsLayout);CHKERRQ(ierr);
    }
    ierr = ISGetIndices(coneSizesIS, &coneSizes);CHKERRQ(ierr);
    ierr = PetscSectionCreate(comm, &coneSizesSection);CHKERRQ(ierr);
    //TODO different start ?
    ierr = PetscSectionSetChart(coneSizesSection, 0, pointsLayout->n);CHKERRQ(ierr);
    for (s = 0; s < pointsLayout->n; ++s) {
      ierr = PetscSectionSetDof(coneSizesSection, s, coneSizes[s]);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(coneSizesSection);CHKERRQ(ierr);
    ierr = ISRestoreIndices(coneSizesIS, &coneSizes);CHKERRQ(ierr);
    {
      PetscLayout tmp=NULL;
      /* We need to keep the layout until the end of function */
      ierr = PetscLayoutReference((PetscLayout)pointsLayout, &tmp);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&coneSizesIS);CHKERRQ(ierr);
  }

  /* use value layout of coneSizesSection as layout of cones and orientations */
  {
    PetscLayout conesLayout;

    ierr = PetscSectionGetValueLayout(comm, coneSizesSection, &conesLayout);CHKERRQ(ierr);
    ierr = ISCreate(comm, &conesIS);CHKERRQ(ierr);
    ierr = ISCreate(comm, &orientationsIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) conesIS, conesName);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) orientationsIS, orientationsName);CHKERRQ(ierr);
    ierr = PetscLayoutDuplicate(conesLayout, &conesIS->map);CHKERRQ(ierr);
    ierr = PetscLayoutDuplicate(conesLayout, &orientationsIS->map);CHKERRQ(ierr);
    ierr = ISLoad(conesIS, viewer);CHKERRQ(ierr);
    ierr = ISLoad(orientationsIS, viewer);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&conesLayout);CHKERRQ(ierr);
  }

  /* check assertion that layout of points is the same as point layout of coneSizesSection */
  {
    PetscLayout pointsLayout0;
    PetscBool   flg;
    ierr = PetscSectionGetPointLayout(comm, coneSizesSection, &pointsLayout0);CHKERRQ(ierr);
    ierr = PetscLayoutCompare(pointsLayout, pointsLayout0, &flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(comm, PETSC_ERR_PLIB, "points layout != coneSizesSection point layout");
    ierr = PetscLayoutDestroy(&pointsLayout0);CHKERRQ(ierr);
  }
  if (debug) {
    const char *group;
    PetscViewer v = PETSC_VIEWER_STDOUT_(comm);

    ierr = PetscViewerHDF5GetGroup(viewer, &group);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "group %s\n", group);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscLayoutView_ASCII(pointsLayout, "pointsLayout", v);CHKERRQ(ierr);
    ierr = PetscSectionView(coneSizesSection, v);CHKERRQ(ierr);
    ierr = ISView(conesIS, v);CHKERRQ(ierr);
    ierr = ISView(orientationsIS, v);CHKERRQ(ierr);
    if (vertexLayout) {
      ierr = PetscLayoutView_ASCII(vertexLayout, "vertexLayout", v);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&pointsLayout);CHKERRQ(ierr);

  layer->d                = d;
  layer->conesIS          = conesIS;
  layer->coneSizesSection = coneSizesSection;
  layer->orientationsIS   = orientationsIS;
  layer->vertexLayout     = vertexLayout;
  PetscFunctionReturn(0);
}

static PetscErrorCode PlexLayerDistribute_Private(PlexLayer layer, PetscSF cellLocalToGlobalSF)
{
  IS              newConesIS, newOrientationsIS;
  PetscSection    newConeSizesSection;
  PetscBool       debug = PETSC_FALSE;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_topology_load_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)cellLocalToGlobalSF, &comm);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &newConeSizesSection);CHKERRQ(ierr);
  //TODO rename to something like ISDistribute() with optional PetscSection argument, allow multiple ISs at once
  ierr = DMPlexDistributeFieldIS(NULL, cellLocalToGlobalSF, layer->coneSizesSection, layer->conesIS, newConeSizesSection, &newConesIS);CHKERRQ(ierr);
  ierr = DMPlexDistributeFieldIS(NULL, cellLocalToGlobalSF, layer->coneSizesSection, layer->orientationsIS, newConeSizesSection, &newOrientationsIS);CHKERRQ(ierr);

  if (debug) {
    PetscViewer v = PETSC_VIEWER_STDOUT_(comm);

    ierr = PetscViewerASCIIPrintf(v, "PlexLayerDistribute_Private depth %D:\n", layer->d);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newConeSizesSection, "newConeSizesSection");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newConesIS, "newConesIS");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) newOrientationsIS, "newOrientationsIS");CHKERRQ(ierr);
    ierr = PetscSectionView(newConeSizesSection, v);CHKERRQ(ierr);
    ierr = ISView(newConesIS, v);CHKERRQ(ierr);
    ierr = ISView(newOrientationsIS, v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }

  ierr = PetscObjectSetName((PetscObject) newConeSizesSection, ((PetscObject)layer->coneSizesSection)->name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newConesIS, ((PetscObject)layer->conesIS)->name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) newOrientationsIS, ((PetscObject)layer->orientationsIS)->name);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&layer->coneSizesSection);CHKERRQ(ierr);
  ierr = ISDestroy(&layer->conesIS);CHKERRQ(ierr);
  ierr = ISDestroy(&layer->orientationsIS);CHKERRQ(ierr);
  layer->coneSizesSection = newConeSizesSection;
  layer->conesIS          = newConesIS;
  layer->orientationsIS   = newOrientationsIS;
  PetscFunctionReturn(0);
}

//TODO share code with DMPlexTopologyBuildTwoStrata
#include <petsc/private/hashseti.h>
static PetscErrorCode PlexLayerCreateSFs_Private(PlexLayer layer, PetscSF *vertexOverlapSF, PetscSF *vertexLocalToGlobalSF)
{
  PetscLayout     vertexLayout = layer->vertexLayout;
  PetscSection    coneSection = layer->coneSizesSection;
  IS              cellVertexData = layer->conesIS;
  IS              coneOrientations = layer->orientationsIS;
  PetscSF         vl2gSF, vOverlapSF;
  PetscInt        *verticesAdj;
  PetscInt        i, n, numVerticesAdj;
  const PetscInt  *cvd, *co = NULL;
  PetscBool       debug = PETSC_FALSE;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_topology_load_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)coneSection, &comm);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coneSection, &n);CHKERRQ(ierr);
  {
    PetscInt n0;
    ierr = ISGetLocalSize(cellVertexData, &n0);CHKERRQ(ierr);
    if (n != n0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local size of IS cellVertexData = %D != %D = storage size of PetscSection coneSection",n0,n);
    ierr = ISGetIndices(cellVertexData, &cvd);CHKERRQ(ierr);
  }
  if (coneOrientations) {
    PetscInt n0;
    ierr = ISGetLocalSize(coneOrientations, &n0);CHKERRQ(ierr);
    if (n != n0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local size of IS coneOrientations = %D != %D = storage size of PetscSection coneSection",n0,n);
    ierr = ISGetIndices(coneOrientations, &co);CHKERRQ(ierr);
  }
  /* Get/check global number of vertices */
  {
    PetscInt NVerticesInCells = PETSC_MIN_INT;

    /* NVerticesInCells = max(cellVertexData) + 1 */
    for (i=0; i<n; i++) if (cvd[i] > NVerticesInCells) NVerticesInCells = cvd[i];
    ++NVerticesInCells;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &NVerticesInCells, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);

    if (vertexLayout->n == PETSC_DECIDE && vertexLayout->N == PETSC_DECIDE) vertexLayout->N = NVerticesInCells;
    else if (vertexLayout->N != PETSC_DECIDE && vertexLayout->N < NVerticesInCells) SETERRQ2(comm, PETSC_ERR_ARG_SIZ, "Specified global number of vertices %D must be greater than or equal to the number of vertices in cells %D",vertexLayout->N,NVerticesInCells);
    ierr = PetscLayoutSetUp(vertexLayout);CHKERRQ(ierr);
  }
  /* Find locally unique vertices in cellVertexData */
  /* We keep the order of first encounter to improve consistency of local numbering */
  {
    PetscHSetI  vhash;
    PetscInt    off = 0;
    PetscBool   missing;
    PetscInt   *verticesAdjTmp;

    ierr = PetscMalloc1(n, &verticesAdjTmp);CHKERRQ(ierr);
    ierr = PetscHSetICreate(&vhash);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      ierr = PetscHSetIQueryAdd(vhash, cvd[i], &missing);CHKERRQ(ierr);
      if (missing) {
        verticesAdjTmp[off] = cvd[i];
        off++;
      }
    }
    ierr = PetscHSetIGetSize(vhash, &numVerticesAdj);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&vhash);CHKERRQ(ierr);
    if (off != numVerticesAdj) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of local vertices %D should be %D", off, numVerticesAdj);
    ierr = PetscMalloc1(numVerticesAdj, &verticesAdj);CHKERRQ(ierr);
    ierr = PetscMemcpy(verticesAdj, verticesAdjTmp, numVerticesAdj * sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscFree(verticesAdjTmp);CHKERRQ(ierr);
  }

  //TODO maybe this could play with ISLocalToGlobalMapping() somehow
  ierr = PetscSFCreateByMatchingIndices(vertexLayout, numVerticesAdj, verticesAdj, NULL, 0, numVerticesAdj, verticesAdj, NULL, 0, &vl2gSF, &vOverlapSF);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) vOverlapSF, "overlapSF");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vl2gSF, "localToGlobalSF");CHKERRQ(ierr);

  if (debug) {
    PetscViewer v = PETSC_VIEWER_STDOUT_(comm);
    ISLocalToGlobalMapping l2g;

    ierr = PetscViewerASCIIPrintf(v, "PlexLayerCreateSFs_Private depth %D:\n", layer->d);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "verticesAdj:\n");CHKERRQ(ierr);
    ierr = PetscIntView(numVerticesAdj, verticesAdj, v);CHKERRQ(ierr);
    ierr = PetscSFView(vOverlapSF, v);CHKERRQ(ierr);
    ierr = PetscSFView(vl2gSF, v);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateSF(vl2gSF, vertexLayout->rstart, &l2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingView(l2g, v);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  ierr = PetscFree(verticesAdj);CHKERRQ(ierr);
  *vertexOverlapSF        = vOverlapSF;
  *vertexLocalToGlobalSF  = vl2gSF;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTopologyBuildFromLayers_Private(DM dm, PetscInt depth, PlexLayer *layers, IS strataPermutation)
{
  const PetscInt *permArr;
  PetscInt        d, nPoints;
  MPI_Comm        comm;
  PetscViewer     dbgv = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  {
    PetscViewerFormat f;
    ierr = PetscOptionsGetViewer(comm, NULL, NULL, "-dm_plex_build_from_layers_debug", &dbgv, &f, NULL);CHKERRQ(ierr);
    if (dbgv) {ierr = PetscViewerPushFormat(dbgv, f);CHKERRQ(ierr);}
  }
  ierr = ISGetIndices(strataPermutation, &permArr);CHKERRQ(ierr);

  /* Count points, strata offsets and cones offsets (taking strataPermutation into account) */
  {
    PetscInt stratumOffset  = 0;
    PetscInt conesOffset    = 0;

    for (d = 0; d <= depth; d++) {
      const PetscInt  e = permArr[d];
      const PlexLayer l = layers[e];
      PetscInt        lo, n, size;

      ierr = PetscSectionGetChart(l->coneSizesSection, &lo, &n);CHKERRQ(ierr);
      ierr = PetscSectionGetStorageSize(l->coneSizesSection, &size);CHKERRQ(ierr);
      if (lo) SETERRQ1(comm, PETSC_ERR_PLIB, "starting point should be 0 in coneSizesSection %D", d);
      l->offset       = stratumOffset;
      l->conesOffset  = conesOffset;
      stratumOffset  += n;
      conesOffset    += size;
    }
    nPoints = stratumOffset;
  }

  /* Set interval for all plex points */
  //TODO we should store starting point of plex
  ierr = DMPlexSetChart(dm, 0, nPoints);CHKERRQ(ierr);

  /* Set up plex coneSection from layer coneSections */
  {
    PetscSection  coneSection;

    ierr = DMPlexGetConeSection(dm, &coneSection);CHKERRQ(ierr);
    for (d = 0; d <= depth; d++) {
      const PlexLayer  l = layers[d];
      PetscInt         n, q;

      ierr = PetscSectionGetChart(l->coneSizesSection, NULL, &n);CHKERRQ(ierr);
      for (q = 0; q < n; q++) {
        const PetscInt p = l->offset + q;
        PetscInt       coneSize;

        ierr = PetscSectionGetDof(l->coneSizesSection, q, &coneSize);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(coneSection, p, coneSize);CHKERRQ(ierr);
      }
    }
  }
  //TODO this is terrible, DMSetUp_Plex() should be DMPlexSetUpSections() or so
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  /* Renumber cones points from layer-global numbering to plex-local numbering */
  {
    PetscInt     *cones, *ornts;

    ierr = DMPlexGetCones(dm, &cones);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientations(dm, &ornts);CHKERRQ(ierr);
    for (d = 1; d <= depth; d++) {
      const PlexLayer  l = layers[d];
      PetscInt         i, lConesSize;
      PetscInt        *lCones;
      const PetscInt  *lOrnts;
      PetscInt        *pCones = &cones[l->conesOffset];
      PetscInt        *pOrnts = &ornts[l->conesOffset];

      ierr = PetscSectionGetStorageSize(l->coneSizesSection, &lConesSize);CHKERRQ(ierr);
      /* Get cones in local plex numbering */
      {
        ISLocalToGlobalMapping  l2g;
        PetscLayout             vertexLayout = l->vertexLayout;
        PetscSF                 vertexSF = layers[d-1]->l2gSF; /* vertices of this layer are cells of previous layer */
        const PetscInt         *gCones;
        PetscInt                lConesSize0;

        ierr = ISGetLocalSize(l->conesIS, &lConesSize0);CHKERRQ(ierr);
        if (lConesSize0 != lConesSize) SETERRQ3(comm, PETSC_ERR_PLIB, "layer %D size(conesIS) = %D != %D = storageSize(coneSizesSection)", d, lConesSize0, lConesSize);
        ierr = ISGetLocalSize(l->orientationsIS, &lConesSize0);CHKERRQ(ierr);
        if (lConesSize0 != lConesSize) SETERRQ3(comm, PETSC_ERR_PLIB, "layer %D size(orientationsIS) = %D != %D = storageSize(coneSizesSection)", d, lConesSize0, lConesSize);

        ierr = PetscMalloc1(lConesSize, &lCones);CHKERRQ(ierr);
        ierr = ISGetIndices(l->conesIS, &gCones);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreateSF(vertexSF, vertexLayout->rstart, &l2g);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApply(l2g, IS_GTOLM_MASK, lConesSize, gCones, &lConesSize0, lCones);CHKERRQ(ierr);
        if (lConesSize0 != lConesSize) SETERRQ2(comm, PETSC_ERR_PLIB, "global to local does not cover all indices (%D of %D)", lConesSize0, lConesSize);
        if (dbgv) {
          ierr = PetscViewerASCIIPrintf(dbgv, "DMPlexBuildFromLayers_Private depth %D\n", d);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPushTab(dbgv);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject)l2g, "l2g");CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingView(l2g, dbgv);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(dbgv, "gCones\n");CHKERRQ(ierr);
          ierr = PetscIntView(lConesSize, gCones, dbgv);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(dbgv, "lCones\n");CHKERRQ(ierr);
          ierr = PetscIntView(lConesSize, lCones, dbgv);CHKERRQ(ierr);
        }
        ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);
        ierr = ISRestoreIndices(l->conesIS, &gCones);CHKERRQ(ierr);
      }
      ierr = ISGetIndices(l->orientationsIS, &lOrnts);CHKERRQ(ierr);
      /* Set cones, need to add stratum offset */
      for (i = 0; i < lConesSize; i++) {
        pCones[i] = lCones[i] + layers[d-1]->offset; /* cone points of current layer are points of previous layer */
        pOrnts[i] = lOrnts[i];
      }
      if (dbgv) {
        PetscMPIInt rank;
        ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
        ierr = PetscViewerASCIIPrintf(dbgv, "point offset of previous layer, cones offset:\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushSynchronized(dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(dbgv, "[%d] %2D, %2D\n", rank, layers[d-1]->offset, l->conesOffset);CHKERRQ(ierr);
        ierr = PetscViewerFlush(dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopSynchronized(dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbgv, "pCones\n");CHKERRQ(ierr);
        ierr = PetscIntView(lConesSize, pCones, dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbgv, "pOrnts\n");CHKERRQ(ierr);
        ierr = PetscIntView(lConesSize, pOrnts, dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(dbgv);CHKERRQ(ierr);
      }
      ierr = PetscFree(lCones);CHKERRQ(ierr);
      ierr = ISRestoreIndices(l->orientationsIS, &lOrnts);CHKERRQ(ierr);
    }
  }

  if (dbgv) {
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    for (d = 0; d <= depth; d++) {
      const PlexLayer   l = layers[d];
      PetscInt          n, q;

      ierr = PetscSectionGetChart(l->coneSizesSection, NULL, &n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(dbgv, "DMPlexBuildFromLayers_Private depth %D\n", d);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(dbgv);CHKERRQ(ierr);
      for (q = 0; q < n; q++) {
        const PetscInt p = l->offset + q;
        PetscInt       coneSize, cp;
        const PetscInt *cone, *ornt;

        ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(dbgv, "  [%2d] point %2D coneSize %D cone", rank, p, coneSize);CHKERRQ(ierr);
        for (cp = 0; cp < coneSize; cp++) {
          ierr = PetscViewerASCIISynchronizedPrintf(dbgv, " %2D", cone[cp]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(dbgv, "  orientation");CHKERRQ(ierr);
        for (cp = 0; cp < coneSize; cp++) {
          ierr = PetscViewerASCIISynchronizedPrintf(dbgv, " %2D", ornt[cp]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(dbgv, "\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(dbgv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(dbgv);CHKERRQ(ierr);
    }

  }
  ierr = DMPlexSymmetrize(dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(dm);CHKERRQ(ierr);

  {
    PetscSF       pointsf_new;
    PetscInt      i, nLeaves;
    PetscInt     *ilocal_new;
    PetscSFNode  *iremote_new;
    PetscMPIInt   rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    /* Count leaves and layer offsets */
    {
      PetscInt leafOffset = 0;

      for (d = 0; d < depth; d++) {
        const PlexLayer l   = layers[d];
        PetscInt        nl  = 0;

        ierr = PetscSFGetGraph(l->overlapSF, NULL, &nl, NULL, NULL);CHKERRQ(ierr);
        l->leafOffset   = leafOffset;
        leafOffset     += nl;
      }
      nLeaves = leafOffset;
    }
    if (dbgv) {
      ierr = PetscViewerASCIIPrintf(dbgv, "DMPlexBuildFromLayers_Private create pointSF\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(dbgv);CHKERRQ(ierr);
    }
    /* Renumber and concatenate local leaves */
    ierr = PetscMalloc1(nLeaves, &ilocal_new);CHKERRQ(ierr);
    for (i = 0; i < nLeaves; i++) ilocal_new[i] = -1;
    for (d = 0; d < depth; d++) {
      const PlexLayer   l = layers[d];
      const PetscInt   *ilocal;
      PetscInt         *ilocal_l = &ilocal_new[l->leafOffset];
      PetscInt          i, nleaves_l;

      ierr = PetscSFGetGraph(l->overlapSF, NULL, &nleaves_l, &ilocal, NULL);CHKERRQ(ierr);
      for (i=0; i<nleaves_l; i++) ilocal_l[i] = ilocal[i] + layers[d]->offset;
    }
    /* Renumber and concatenate remote roots */
    ierr = PetscMalloc1(nLeaves, &iremote_new);CHKERRQ(ierr);
    for (i = 0; i < nLeaves; i++) {
      iremote_new[i].rank   = -1;
      iremote_new[i].index  = -1;
    }
    for (d = 0; d < depth; d++) {
      const PlexLayer     l = layers[d];
      PetscInt            nl, nroots;
      PetscSF             sfTemp;
      const PetscSFNode  *iremote;
      PetscSFNode        *rootdata;
      PetscSFNode        *leafdata = &iremote_new[l->leafOffset];

      ierr = PetscSFGetGraph(l->overlapSF, &nroots, &nl, NULL, &iremote);CHKERRQ(ierr);
      ierr = PetscSFCreate(comm, &sfTemp);CHKERRQ(ierr);
      /* create SF with contiguous leaves */
      ierr = PetscSFSetGraph(sfTemp, nroots, nl, NULL, PETSC_USE_POINTER, iremote, PETSC_USE_POINTER);CHKERRQ(ierr);
      ierr = PetscSFSetUp(sfTemp);CHKERRQ(ierr);
      ierr = PetscMalloc1(nroots, &rootdata);CHKERRQ(ierr);
      for (i = 0; i < nroots; i++) {
        rootdata[i].index = i + layers[d]->offset;
        rootdata[i].rank  = (PetscInt) rank;
      }
      if (dbgv) {
        ierr = PetscViewerASCIIPrintf(dbgv, "depth %D\n", d);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(dbgv);CHKERRQ(ierr);
        ierr = PetscSFView(l->overlapSF, dbgv);CHKERRQ(ierr);
        ierr = PetscSFView(sfTemp, dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbgv, "old leafdata:\n");CHKERRQ(ierr);
        ierr = PetscIntView(2*nl, (PetscInt*) leafdata, dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(dbgv, "rootdata:\n");CHKERRQ(ierr);
        ierr = PetscIntView(2*nroots, (PetscInt*) rootdata, dbgv);CHKERRQ(ierr);
      }
      ierr = PetscSFBcastBegin(sfTemp, MPIU_2INT, rootdata, leafdata, MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(  sfTemp, MPIU_2INT, rootdata, leafdata, MPI_REPLACE);CHKERRQ(ierr);
      if (dbgv) {
        ierr = PetscViewerASCIIPrintf(dbgv, "new leafdata:\n");CHKERRQ(ierr);
        ierr = PetscIntView(2*nl, (PetscInt*) leafdata, dbgv);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(dbgv);CHKERRQ(ierr);
      }
      ierr = PetscSFDestroy(&sfTemp);CHKERRQ(ierr);
      ierr = PetscFree(rootdata);CHKERRQ(ierr);
    }
    if (dbgv) {
      ierr = PetscViewerASCIIPrintf(dbgv, "ilocal_new:\n");CHKERRQ(ierr);
      ierr = PetscIntView(nLeaves, ilocal_new, dbgv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(dbgv, "iremote_ind:\n");CHKERRQ(ierr);
      ierr = PetscIntView(2*nLeaves, (PetscInt*) iremote_new, dbgv);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(dbgv);CHKERRQ(ierr);
    }
    /* Build the new pointSF */
    ierr = PetscSFCreate(comm, &pointsf_new);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(pointsf_new, nPoints, nLeaves, ilocal_new, PETSC_OWN_POINTER, iremote_new, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetUp(pointsf_new);CHKERRQ(ierr);
    ierr = DMSetPointSF(dm, pointsf_new);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&pointsf_new);CHKERRQ(ierr);
  }

  if (dbgv) {
    ierr = DMView(dm, dbgv);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(dbgv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dbgv);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(strataPermutation, &permArr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTopologyLoad_HDF5_v2(DM dm, PetscViewer viewer, PetscSF *vertexLocalToGlobalSF)
{
  PlexLayer      *layers;
  IS              strataPermutation;
  PetscLayout     pointsLayout = NULL;
  PetscInt        depth;
  PetscInt        d;
  MPI_Comm        comm;
  PetscMPIInt     size, rank;
  PetscBool       debug = PETSC_FALSE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL, NULL, "-dm_plex_topology_load_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topology");CHKERRQ(ierr);
  {
    PetscInt dim;
    ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "depth", PETSC_INT, NULL, &depth);CHKERRQ(ierr);
    ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "cell_dim", PETSC_INT, NULL, &dim);CHKERRQ(ierr);
    ierr = DMSetDimension(dm, dim);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "strata");CHKERRQ(ierr);
  {
    IS              spOnComm;

    ierr = ISCreate(comm, &spOnComm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) spOnComm, "permutation");CHKERRQ(ierr);
    ierr = ISLoad(spOnComm, viewer);CHKERRQ(ierr);
    /* have the same serial IS on every rank */
    ierr = ISAllGather(spOnComm, &strataPermutation);CHKERRQ(ierr);
    //TODO PetscObjectCopyName((PetscObject) spOnComm, (PetscObject) strataPermutation);
    ierr = PetscObjectSetName((PetscObject) strataPermutation, ((PetscObject)spOnComm)->name);CHKERRQ(ierr);
    ierr = ISDestroy(&spOnComm);CHKERRQ(ierr);
  }
  if (debug && !rank) {
    ierr = ISView(strataPermutation, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = PetscMalloc1(depth+1, &layers);CHKERRQ(ierr);
  for (d = depth; d >= 0; d--) {
    ierr = PlexLayerCreate_Private(&layers[d]);CHKERRQ(ierr);
    ierr = PlexLayerLoad_Private(layers[d], viewer, d, pointsLayout);CHKERRQ(ierr);
    pointsLayout = layers[d]->vertexLayout;
  }
  for (d = depth; d >= 0; d--) {
    if (d < depth) {
      ierr = PlexLayerDistribute_Private(layers[d], layers[d]->l2gSF);CHKERRQ(ierr);
    }
    if (d > 0) {
      ierr = PlexLayerCreateSFs_Private(layers[d], &layers[d-1]->overlapSF, &layers[d-1]->l2gSF);CHKERRQ(ierr);
    }
  }

  ierr = DMPlexTopologyBuildFromLayers_Private(dm, depth, layers, strataPermutation);CHKERRQ(ierr);

  *vertexLocalToGlobalSF = layers[0]->l2gSF;
  ierr = PetscObjectReference((PetscObject) *vertexLocalToGlobalSF);CHKERRQ(ierr);
  for (d = depth; d >= 0; d--) {
    ierr = PlexLayerDestroy(&layers[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(layers);CHKERRQ(ierr);
  ierr = ISDestroy(&strataPermutation);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr); /* strata */
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr); /* topology */
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTopologyLoad_HDF5_Internal(DM dm, PetscViewer viewer, PetscSF *vertexLocalToGlobalSF)
{
  PetscInt        version = 1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5ReadAttribute(viewer, "topology", "version", PETSC_INT, &version, &version);CHKERRQ(ierr);
  if (vertexLocalToGlobalSF) *vertexLocalToGlobalSF = NULL;
  switch (version) {
    case 1: ierr = DMPlexTopologyLoad_HDF5_v1(dm, viewer);CHKERRQ(ierr); break;
    case 2: ierr = DMPlexTopologyLoad_HDF5_v2(dm, viewer, vertexLocalToGlobalSF);CHKERRQ(ierr); break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DMPlexTopologyLoad() for topology version %D not implemented yet", version);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexCoordinatesLoad_HDF5_Internal(DM dm, PetscSF vertexLocalToGlobalSF, PetscViewer viewer)
{
  PetscLayout     vertexLayout;
  Vec             coordinates;
  PetscReal       lengthScale;
  PetscInt        spatialDim, N, vStart, vEnd;
  PetscInt        nVertices, nVerticesAdj;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
  /* Read geometry */
  ierr = PetscViewerHDF5PushGroup(viewer, "/geometry");CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject) dm), &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "vertices");CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes(viewer, ((PetscObject)coordinates)->name, &spatialDim, &N);CHKERRQ(ierr);
  if (vertexLocalToGlobalSF) {
    ierr = PetscSFGetGraph(vertexLocalToGlobalSF, &nVertices, &nVerticesAdj, NULL, NULL);CHKERRQ(ierr);
    /* Correspondance between vertexLocalToGlobalSF and DMPlex topology is checked in DMPlexGeometryBuild() */
  } else {
    nVertices = nVerticesAdj = vEnd - vStart;
  }
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject) dm), nVertices * spatialDim, PETSC_DECIDE, spatialDim, &vertexLayout);CHKERRQ(ierr);
  {
    PetscInt lN;
    ierr = PetscLayoutGetSize(vertexLayout, &lN);CHKERRQ(ierr);
    if (N != lN) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Number of vertices in coordinate dataset %D does not match global number of topological vertices %D", N/spatialDim, lN/spatialDim);
  }
  ierr = VecSetLayout(coordinates, vertexLayout);CHKERRQ(ierr);
  ierr = VecLoad(coordinates, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  //TODO should be in DMPlexGeometryBuild?
  ierr = DMPlexGetScale(dm, PETSC_UNIT_LENGTH, &lengthScale);CHKERRQ(ierr);
  ierr = VecScale(coordinates, 1.0/lengthScale);CHKERRQ(ierr);

  ierr = DMPlexGeometryBuild(dm, coordinates, vertexLocalToGlobalSF);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&vertexLayout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The first version will read everything onto proc 0, letting the user distribute
   The next will create a naive partition, and then rebalance after reading
*/
PetscErrorCode DMPlexLoad_HDF5_Internal(DM dm, PetscViewer viewer)
{
  PetscSF         vertexSF;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexTopologyLoad_HDF5_Internal(dm, viewer, &vertexSF);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesLoad_HDF5_Internal(dm, vertexSF, viewer);CHKERRQ(ierr);
  ierr = DMPlexLabelsLoad_HDF5_Internal(dm, viewer);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&vertexSF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexSectionLoad_HDF5_Internal_CreateDataSF(PetscSection rootSection, PetscLayout layout, PetscInt globalOffsets[], PetscSection leafSection, PetscSF *sectionSF)
{
  MPI_Comm        comm;
  PetscInt        pStart, pEnd, p, m;
  PetscInt       *goffs, *ilocal;
  PetscBool       rootIncludeConstraints, leafIncludeConstraints;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)leafSection, &comm);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(leafSection, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(rootSection, &rootIncludeConstraints);CHKERRQ(ierr);
  ierr = PetscSectionGetIncludesConstraints(leafSection, &leafIncludeConstraints);CHKERRQ(ierr);
  if (rootIncludeConstraints && leafIncludeConstraints) {ierr = PetscSectionGetStorageSize(leafSection, &m);CHKERRQ(ierr);}
  else {ierr = PetscSectionGetConstrainedStorageSize(leafSection, &m);CHKERRQ(ierr);}
  ierr = PetscMalloc1(m, &ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &goffs);CHKERRQ(ierr);
  /* Currently, PetscSFDistributeSection() returns globalOffsets[] only */
  /* for the top-level section (not for each field), so one must have   */
  /* rootSection->pointMajor == PETSC_TRUE.                             */
  if (!rootSection->pointMajor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for field major ordering");
  /* Currently, we also assume that leafSection->pointMajor == PETSC_TRUE. */
  if (!leafSection->pointMajor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for field major ordering");
  for (p = pStart, m = 0; p < pEnd; ++p) {
    PetscInt        dof, cdof, i, j, off, goff;
    const PetscInt *cinds;

    ierr = PetscSectionGetDof(leafSection, p, &dof);CHKERRQ(ierr);
    if (dof < 0) continue;
    goff = globalOffsets[p-pStart];
    ierr = PetscSectionGetOffset(leafSection, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(leafSection, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintIndices(leafSection, p, &cinds);CHKERRQ(ierr);
    for (i = 0, j = 0; i < dof; ++i) {
      PetscBool constrained = (PetscBool) (j < cdof && i == cinds[j]);

      if (!constrained || (leafIncludeConstraints && rootIncludeConstraints)) {ilocal[m] = off++; goffs[m++] = goff++;}
      else if (leafIncludeConstraints && !rootIncludeConstraints) ++off;
      else if (!leafIncludeConstraints &&  rootIncludeConstraints) ++goff;
      if (constrained) ++j;
    }
  }
  ierr = PetscSFCreate(comm, sectionSF);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sectionSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(*sectionSF, layout, m, ilocal, PETSC_OWN_POINTER, goffs);CHKERRQ(ierr);
  ierr = PetscFree(goffs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexSectionLoad_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sfXB, PetscSF *gsf, PetscSF *lsf)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  const char    *topologydm_name;
  const char    *sectiondm_name;
  PetscSection   sectionA, sectionB;
  PetscInt       NX, nX, n, i;
  PetscSF        sfAB;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscObjectGetName((PetscObject)dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes(viewer, "/topology/order", NULL, &NX);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  /* A: on-disk points                        */
  /* X: list of global point numbers, [0, NX) */
  /* B: plex points                           */
  /* Load raw section (sectionA)              */
  ierr = PetscSectionCreate(comm, &sectionA);CHKERRQ(ierr);
  ierr = PetscSectionLoad(sectionA, viewer);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(sectionA, NULL, &n);CHKERRQ(ierr);
  /* Create sfAB: A -> B */
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt  N, N1;

    ierr = PetscViewerHDF5ReadSizes(viewer, "order", NULL, &N1);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&n, &N, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    if (N1 != N) SETERRQ2(comm, PETSC_ERR_ARG_SIZ, "Mismatching sizes: on-disk order array size (%D) != number of loaded section points (%D)", N1, N);
  }
#endif
  {
    IS              orderIS;
    const PetscInt *gpoints;
    PetscSF         sfXA, sfAX;
    PetscLayout     layout;
    PetscSFNode    *owners, *buffer;
    PetscInt        nleaves;
    PetscInt       *ilocal;
    PetscSFNode    *iremote;

    /* Create sfAX: A -> X */
    ierr = ISCreate(comm, &orderIS);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)orderIS, "order");CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(orderIS->map, n);CHKERRQ(ierr);
    ierr = ISLoad(orderIS, viewer);CHKERRQ(ierr);
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(layout, NX);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm, &sfXA);CHKERRQ(ierr);
    ierr = ISGetIndices(orderIS, &gpoints);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(sfXA, layout, n, NULL, PETSC_OWN_POINTER, gpoints);CHKERRQ(ierr);
    ierr = ISRestoreIndices(orderIS, &gpoints);CHKERRQ(ierr);
    ierr = ISDestroy(&orderIS);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfXA, &nX, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(n, &owners);CHKERRQ(ierr);
    ierr = PetscMalloc1(nX, &buffer);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {owners[i].rank = rank; owners[i].index = i;}
    for (i = 0; i < nX; ++i) {buffer[i].rank = -1; buffer[i].index = -1;}
    ierr = PetscSFReduceBegin(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sfXA, MPIU_2INT, owners, buffer, MPI_MAXLOC);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfXA);CHKERRQ(ierr);
    ierr = PetscFree(owners);CHKERRQ(ierr);
    for (i = 0, nleaves = 0; i < nX; ++i) if (buffer[i].rank >= 0) nleaves++;
    ierr = PetscMalloc1(nleaves, &ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves, &iremote);CHKERRQ(ierr);
    for (i = 0, nleaves = 0; i < nX; ++i) {
      if (buffer[i].rank >= 0) {
        ilocal[nleaves] = i;
        iremote[nleaves].rank = buffer[i].rank;
        iremote[nleaves].index = buffer[i].index;
        nleaves++;
      }
    }
    ierr = PetscSFCreate(comm, &sfAX);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(sfAX);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfAX, n, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    /* Fix PetscSFCompose() and replace the code-block below with:  */
    /* ierr = PetscSFCompose(sfAX, sfXB, &sfAB);CHKERRQ(ierr);      */
    /* which currently causes segmentation fault due to sparse map. */
    {
      PetscInt     npoints;
      PetscInt     mleaves;
      PetscInt    *jlocal;
      PetscSFNode *jremote;

      ierr = PetscSFGetGraph(sfXB, NULL, &npoints, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(npoints, &owners);CHKERRQ(ierr);
      for (i = 0; i < npoints; ++i) {owners[i].rank = -1; owners[i].index = -1;}
      ierr = PetscSFBcastBegin(sfXB, MPIU_2INT, buffer, owners, MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sfXB, MPIU_2INT, buffer, owners, MPI_REPLACE);CHKERRQ(ierr);
      for (i = 0, mleaves = 0; i < npoints; ++i) if (owners[i].rank >= 0) mleaves++;
      ierr = PetscMalloc1(mleaves, &jlocal);CHKERRQ(ierr);
      ierr = PetscMalloc1(mleaves, &jremote);CHKERRQ(ierr);
      for (i = 0, mleaves = 0; i < npoints; ++i) {
        if (owners[i].rank >= 0) {
          jlocal[mleaves] = i;
          jremote[mleaves].rank = owners[i].rank;
          jremote[mleaves].index = owners[i].index;
          mleaves++;
        }
      }
      ierr = PetscSFCreate(comm, &sfAB);CHKERRQ(ierr);
      ierr = PetscSFSetFromOptions(sfAB);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(sfAB, n, mleaves, jlocal, PETSC_OWN_POINTER, jremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree(owners);CHKERRQ(ierr);
    }
    ierr = PetscFree(buffer);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfAX);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  /* Create plex section (sectionB) */
  ierr = DMGetLocalSection(sectiondm, &sectionB);CHKERRQ(ierr);
  if (lsf || gsf) {
    PetscLayout  layout;
    PetscInt     M, m;
    PetscInt    *offsetsA;
    PetscBool    includesConstraintsA;

    ierr = PetscSFDistributeSection(sfAB, sectionA, &offsetsA, sectionB);CHKERRQ(ierr);
    ierr = PetscSectionGetIncludesConstraints(sectionA, &includesConstraintsA);CHKERRQ(ierr);
    if (includesConstraintsA) {ierr = PetscSectionGetStorageSize(sectionA, &m);CHKERRQ(ierr);}
    else {ierr = PetscSectionGetConstrainedStorageSize(sectionA, &m);CHKERRQ(ierr);}
    ierr = MPI_Allreduce(&m, &M, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(layout, M);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    if (lsf) {
      PetscSF lsfABdata;

      ierr = DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, sectionB, &lsfABdata);CHKERRQ(ierr);
      *lsf = lsfABdata;
    }
    if (gsf) {
      PetscSection  gsectionB, gsectionB1;
      PetscBool     includesConstraintsB;
      PetscSF       gsfABdata, pointsf;

      ierr = DMGetGlobalSection(sectiondm, &gsectionB1);CHKERRQ(ierr);
      ierr = PetscSectionGetIncludesConstraints(gsectionB1, &includesConstraintsB);CHKERRQ(ierr);
      ierr = DMGetPointSF(sectiondm, &pointsf);CHKERRQ(ierr);
      ierr = PetscSectionCreateGlobalSection(sectionB, pointsf, includesConstraintsB, PETSC_TRUE, &gsectionB);CHKERRQ(ierr);
      ierr = DMPlexSectionLoad_HDF5_Internal_CreateDataSF(sectionA, layout, offsetsA, gsectionB, &gsfABdata);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&gsectionB);CHKERRQ(ierr);
      *gsf = gsfABdata;
    }
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    ierr = PetscFree(offsetsA);CHKERRQ(ierr);
  } else {
    ierr = PetscSFDistributeSection(sfAB, sectionA, NULL, sectionB);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sfAB);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexVecLoad_HDF5_Internal(DM dm, PetscViewer viewer, DM sectiondm, PetscSF sf, Vec vec)
{
  MPI_Comm           comm;
  const char        *topologydm_name;
  const char        *sectiondm_name;
  const char        *vec_name;
  Vec                vecA;
  PetscInt           mA, m, bs;
  const PetscInt    *ilocal;
  const PetscScalar *src;
  PetscScalar       *dest;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm, &topologydm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)sectiondm, &sectiondm_name);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)vec, &vec_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "topologies");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, topologydm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "dms");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, sectiondm_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "vecs");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, vec_name);CHKERRQ(ierr);
  ierr = VecCreate(comm, &vecA);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)vecA, vec_name);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &mA, &m, &ilocal, NULL);CHKERRQ(ierr);
  /* Check consistency */
  {
    PetscSF   pointsf, pointsf1;
    PetscInt  m1, i, j;

    ierr = DMGetPointSF(dm, &pointsf);CHKERRQ(ierr);
    ierr = DMGetPointSF(sectiondm, &pointsf1);CHKERRQ(ierr);
    if (pointsf1 != pointsf) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mismatching point SFs for dm and sectiondm");
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt  MA, MA1;

      ierr = MPIU_Allreduce(&mA, &MA, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
      ierr = PetscViewerHDF5ReadSizes(viewer, vec_name, NULL, &MA1);CHKERRQ(ierr);
      if (MA1 != MA) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total SF root size (%D) != On-disk vector data size (%D)", MA, MA1);
    }
#endif
    ierr = VecGetLocalSize(vec, &m1);CHKERRQ(ierr);
    if (m1 < m) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Target vector size (%D) < SF leaf size (%D)", m1, m);
    for (i = 0; i < m; ++i) {
      j = ilocal ? ilocal[i] : i;
      if (j < 0 || j >= m1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Leaf's %D-th index, %D, not in [%D, %D)", i, j, 0, m1);
    }
  }
  ierr = VecSetSizes(vecA, mA, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecLoad(vecA, viewer);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vecA, &src);CHKERRQ(ierr);
  ierr = VecGetArray(vec, &dest);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_SCALAR, src, dest, MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_SCALAR, src, dest, MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec, &dest);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(vecA, &src);CHKERRQ(ierr);
  ierr = VecDestroy(&vecA);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadAttribute(viewer, NULL, "blockSize", PETSC_INT, NULL, (void *) &bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vec, bs);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
