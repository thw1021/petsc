#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

/*
Numbering:

  Since we can guarantee that interpolated faces are numbered last, we do not have to change the algorithm for offsets. However, it will no longer be true that

    face = off + (c - cStart) * Nf + r

  Shit! With different kinds of faces, the offsets are still too optimistic. We will need to actually count, but we can start running without this yet.
*/

static PetscErrorCode DMPlexTransformView_Interpolate(DMPlexTransform tr, PetscViewer viewer)
{
  DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;
  PetscBool                    isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    DM          dm;
    DMLabel     active;
    PetscInt    dim;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(DMPlexTransformGetDM(tr, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexTransformGetActive(tr, &active));

    PetscCall(PetscViewerASCIIPrintf(viewer, "Interpolation transformation %s\n", name ? name : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  face dim: %" PetscInt_FMT "\n", in->faceDim));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_Interpolate(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;
  PetscBool                    faceDim, flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlexTransform Interpolation Options");
  PetscCall(PetscOptionsBool("-dm_plex_transform_interpolate_face_dim", "Create tensor cells", "", in->faceDim, &faceDim, &flg));
  if (flg) PetscCall(DMPlexTransformInterpolateSetFaceDim(tr, faceDim));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_TRIANGLE produces
     1 triangle, and
     3 segments
*/
static PetscErrorCode DMPlexTransformInterpolateSetUp_Triangle(DMPlexTransform_Interpolate *in)
{
  const DMPolytopeType ct = DM_POLYTOPE_TRIANGLE;
  PetscInt             Nc, No, coff, ooff;

  PetscFunctionBegin;
  in->Nt[ct] = 2;
  Nc         = 9 + 8 * 3;
  No         = 3 + 2 * 3;
  PetscCall(PetscMalloc4(in->Nt[ct], &in->target[ct], in->Nt[ct], &in->size[ct], Nc, &in->cone[ct], No, &in->ornt[ct]));
  in->target[ct][0] = DM_POLYTOPE_TRIANGLE;
  in->target[ct][1] = DM_POLYTOPE_SEGMENT;
  in->size[ct][0]   = 1;
  in->size[ct][1]   = 3;
  /*   cones for triangle */
  in->cone[ct][0] = DM_POLYTOPE_SEGMENT;
  in->cone[ct][1] = 0;
  in->cone[ct][2] = 0;
  in->cone[ct][3] = DM_POLYTOPE_SEGMENT;
  in->cone[ct][4] = 0;
  in->cone[ct][5] = 1;
  in->cone[ct][6] = DM_POLYTOPE_SEGMENT;
  in->cone[ct][7] = 0;
  in->cone[ct][8] = 2;
  for (PetscInt i = 0; i < 3; ++i) in->ornt[ct][i] = 0;
  /*   cones for segments */
  coff = 9;
  ooff = 3;
  for (PetscInt i = 0; i < 3; ++i) {
    in->cone[ct][coff + 8 * i + 0] = DM_POLYTOPE_POINT;
    in->cone[ct][coff + 8 * i + 1] = 1;
    in->cone[ct][coff + 8 * i + 2] = i;
    in->cone[ct][coff + 8 * i + 3] = 0;
    in->cone[ct][coff + 8 * i + 4] = DM_POLYTOPE_POINT;
    in->cone[ct][coff + 8 * i + 5] = 1;
    in->cone[ct][coff + 8 * i + 6] = (i + 1) % 3;
    in->cone[ct][coff + 8 * i + 7] = 0;
    in->ornt[ct][ooff + 2 * i + 0] = 0;
    in->ornt[ct][ooff + 2 * i + 1] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The refine types for interpolation are:

*/
static PetscErrorCode DMPlexTransformSetUp_Interpolate(DMPlexTransform tr)
{
  DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;
  DM                           dm;
  PetscInt                     ict, dim;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscMalloc5(DM_NUM_POLYTOPES, &in->Nt, DM_NUM_POLYTOPES, &in->target, DM_NUM_POLYTOPES, &in->size, DM_NUM_POLYTOPES, &in->cone, DM_NUM_POLYTOPES, &in->ornt));
  for (ict = 0; ict < DM_NUM_POLYTOPES; ++ict) {
    in->Nt[ict]     = -1;
    in->target[ict] = NULL;
    in->size[ict]   = NULL;
    in->cone[ict]   = NULL;
    in->ornt[ict]   = NULL;
  }
  PetscCall(DMPlexTransformInterpolateSetUp_Triangle(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_Quadrilateral(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_SegPrismTensor(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_Tetrahedron(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_Hexahedron(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_TriPrism(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_TriPrismTensor(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_QuadPrismTensor(in));
  //PetscCall(DMPlexTransformExtrudeSetUp_Pyramid(in));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_Interpolate(DMPlexTransform tr)
{
  DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(in));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_Interpolate(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  //DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, so);
  if (!so) PetscFunctionReturn(PETSC_SUCCESS);
    switch (sct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      switch (tct) {
      case DM_POLYTOPE_SEGMENT:
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        *onew = DMPolytopeTypeComposeOrientation(tct, o, so ? -3 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    // We need to handle identity extrusions from volumes (TET, HEX, etc) when boundary faces are being extruded
    case DM_POLYTOPE_TRIANGLE:
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_Interpolate(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexTransform_Interpolate *in       = (DMPlexTransform_Interpolate *)tr->data;
  PetscBool                    identity = DMPolytopeTypeGetDim(source) == in->faceDim + 1 ? PETSC_FALSE : PETSC_TRUE;

  PetscFunctionBegin;
  if (rt) *rt = 0;
  if (identity) {
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
  } else {
    *Nt     = in->Nt[source];
    *target = in->target[source];
    *size   = in->size[source];
    *cone   = in->cone[source];
    *ornt   = in->ornt[source];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_Interpolate(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_Interpolate;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_Interpolate;
  tr->ops->setup                 = DMPlexTransformSetUp_Interpolate;
  tr->ops->destroy               = DMPlexTransformDestroy_Interpolate;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_Internal;
  tr->ops->celltransform         = DMPlexTransformCellTransform_Interpolate;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_Interpolate;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Interpolate(DMPlexTransform tr)
{
  DMPlexTransform_Interpolate *in;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&in));
  tr->data = in;
  PetscCall(DMPlexTransformInitialize_Interpolate(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformInterpolateGetFaceDim - Get the dimension of interpolated faces

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. faceDim - The dimension of added faces

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformInterpolateSetFaceDim()`
@*/
PetscErrorCode DMPlexTransformInterpolateGetFaceDim(DMPlexTransform tr, PetscInt *faceDim)
{
  DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(faceDim, 2);
  *faceDim = in->faceDim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformInterpolateSetFaceDim - Set the dimension of interpolated faces

  Not Collective

  Input Parameters:
+ tr      - The `DMPlexTransform`
- faceDim - The dimension of added faces

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformInterpolateGetFaceDim()`
@*/
PetscErrorCode DMPlexTransformInterpolateSetFaceDim(DMPlexTransform tr, PetscInt faceDim)
{
  DMPlexTransform_Interpolate *in = (DMPlexTransform_Interpolate *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  in->faceDim = faceDim;
  PetscFunctionReturn(PETSC_SUCCESS);
}
