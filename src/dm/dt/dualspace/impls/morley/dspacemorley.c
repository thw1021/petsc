#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>



/*-------------------------- Implementation for our PetscDualSpace Implementations -------------------------------*/

static PetscErrorCode PetscDualSpaceMorleyGetSize_Morley(PetscDualSpace sp, PetscInt *size)
{
  PetscDualSpace_Morley *mor = (PetscDualSpace_Morley *) sp->data;

  PetscFunctionBegin;
  *size = mor->size;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceMorleySetSize_Morley(PetscDualSpace sp, PetscInt size)
{
  PetscDualSpace_Morley *mor = (PetscDualSpace_Morley *) sp->data;

  PetscFunctionBegin;
  mor->size = size;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceMorleySetNumDof_Morley(PetscDualSpace sp, PetscInt dim)
{
  DM             dm;
  PetscInt       pStart, pEnd, vStart, vEnd, v, fStart, fEnd, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &sp->pointSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sp->pointSection, pStart, pEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {ierr = PetscSectionAddDof(sp->pointSection, v, 1);CHKERRQ(ierr);}
  for (f = fStart; f < fEnd; ++f) {ierr = PetscSectionAddDof(sp->pointSection, f, 1); CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(sp->pointSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSetUp_Morley(PetscDualSpace sp)
{
  PetscReal     *qpoints, *qweights;
  PetscReal      eps = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       size, dim = 0, f = 0, d, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(sp->dm, &dim); CHKERRQ(ierr);
  if (sp->order != 2) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Morley Elements only are defined for quadratic spaces");
  if (dim < 2 || dim > 3) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Morley Elements only work for spatial dimensions 2 and 3 in Petsc");

  ierr = PetscDualSpaceMorleySetNumDof_Morley(sp, dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &sp->functional);CHKERRQ(ierr);

  if (dim == 2) {
    PetscScalar   vertexCoords[6] = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0};
    PetscScalar   qp[5]           = {0.0, -1.0+eps, 0.0, -1.0-eps, 0.0};
    PetscInt      nvertex         = 3;

    /* Bottom of Reference Triangle */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*dim, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(2,     &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 2, qpoints, qweights);CHKERRQ(ierr);
    for (d = 0; d < 4; ++d) {qpoints[d] = PetscRealPart(qp[d]);}
    qweights[0] = -1.0/(2.0*eps);
    qweights[1] =  1.0/(2.0*eps);
    ++f;

    /* Diagonal for Reference Triangle */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(4*dim, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(4,     &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 4, qpoints, qweights);CHKERRQ(ierr);
    PetscScalar   qp2[8] = {eps, 0.0, -1.0*eps, 0.0, 0.0, eps, 0.0, -1.0*eps};
    PetscScalar   qw[4] = {1.0/(2.0*PetscSqrtReal(2)*eps), -1.0/(2.0*PetscSqrtReal(2)*eps), 1.0/(2.0*PetscSqrtReal(2)*eps), -1.0/(2.0*PetscSqrtReal(2)*eps)};
    for (d = 0; d < 4; ++d) {
      qpoints[2*d]   = PetscRealPart(qp2[2*d]);
      qpoints[2*d+1] = PetscRealPart(qp2[2*d+1];
      qweights[d]    = PetscRealPart(qw[d]);
    }
    ++f;

    /* Left of Reference Triangle */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*dim, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(2,     &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 2, qpoints, qweights);CHKERRQ(ierr);
    for (d = 0; d < 4; ++d) {qpoints[d] = PetscRealPart(qp[d+1]);}
    qweights[0] = -1.0/(2.0*eps);
    qweights[1] =  1.0/(2.0*eps);
    ++f;

    /* TODO: Get these coordinates from the DM */
    for (v = 0; v < nvertex; ++v, ++f) {
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
      ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
      ierr = PetscMalloc1(1,   &qweights);CHKERRQ(ierr);
      ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 1, qpoints, qweights);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) qpoints[d] = PetscRealPart(vertexCoords[v*dim + d]);
      qweights[0] = 1.0;
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "Morley Elements have not been implemented yet in 3D!");

        /* Face Dual Elements */

    PetscScalar centroidCoords[12]     = {-1.0/3.0, -1.0/3.0, -1.0, -1.0/3.0, -1.0, -1.0/3.0, -1.0, -1.0/3.0, -1.0/3.0, -1.0/3.0, -1.0/3.0, -1.0/3.0};

    f = 0;

        /* Gradient(centroid) \cdot Normal to Face z = -1 */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(6, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(2,   &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 2, qpoints, qweights);CHKERRQ(ierr);
    PetscScalar qpZ[6] = {centroidCoords[0],centroidCoords[1],centroidCoords[2]+eps,centroidCoords[0],centroidCoords[1],centroidCoords[2]-eps};
    PetscScalar qwZ[2] = {-1.0/(2.0*eps), 1.0/(2.0*eps)};
    for (d = 0; d < 2; ++d) {
      qpoints[dim*d]   = PetscRealPart(qpZ[dim*d]);
      qpoints[dim*d+1] = PetscRealPart(qpZ[dim*d+1]);
      qpoints[dim*d+2] = PetscRealPart(qpZ[dim*d+2]);
      qweights[d]     = PetscRealPart(qwZ[d]);
    }
    ++f;

        /* Gradient(centroid) \cdot Normal to Face y = -1 */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(6, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(2,   &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 2, qpoints, qweights);CHKERRQ(ierr);
    PetscScalar qpY[6] = {centroidCoords[3],centroidCoords[4]+eps,centroidCoords[5],centroidCoords[3],centroidCoords[4]-eps,centroidCoords[5]};
    PetscScalar qwY[2] = {-1.0/(2.0*eps), 1.0/(2.0*eps)};
    for (d = 0; d < 2; ++d) {
      qpoints[dim*d]   = PetscRealPart(qpY[dim*d]);
      qpoints[dim*d+1] = PetscRealPart(qpY[dim*d+1]);
      qpoints[dim*d+2] = PetscRealPart((qpY[dim*d+2]);
      qweights[d]      = PetscRealPart(qwY[d]);
    }
    ++f;

        /* Gradient(centroid) \cdot Normal to Face x = -1 */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(6, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(2,   &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 2, qpoints, qweights);CHKERRQ(ierr);
    PetscScalar qpX[6] = {centroidCoords[6]+eps,centroidCoords[7],centroidCoords[8],centroidCoords[6]-eps,centroidCoords[7],centroidCoords[8]};
    PetscScalar qwX[2] = {-1.0/(2.0*eps), 1.0/(2.0*eps)};
    for (d = 0; d < 2; ++d) {
      qpoints[dim*d]   = PetscRealPart(qpX[dim*d]);
      qpoints[dim*d+1] = PetscRealPart(qpX[dim*d+1]);
      qpoints[dim*d+2] = PetscRealPart(qpX[dim*d+2]);
      qweights[d]      = PetscRealPart(qwX[d]);
    }
    ++f;

        /* Gradient(centroid) \cdot Normal to Diagonal Face */
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
    ierr = PetscMalloc1(18, &qpoints);CHKERRQ(ierr);
    ierr = PetscMalloc1(6,   &qweights);CHKERRQ(ierr);
    ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, 1, 2, qpoints, qweights);CHKERRQ(ierr);
    PetscScalar qp0[18] = {centroidCoords[9],centroidCoords[10],centroidCoords[11]+eps,centroidCoords[9],centroidCoords[10],centroidCoords[11]-eps,
                           centroidCoords[9],centroidCoords[10]+eps,centroidCoords[11],centroidCoords[9],centroidCoords[10]-eps,centroidCoords[11],
                           centroidCoords[9]+eps,centroidCoords[10],centroidCoords[11],centroidCoords[9]-eps,centroidCoords[10],centroidCoords[11] };
    PetscScalar qw0[6] = {1.0/(2.0*sqrt(3)*eps), -1.0/(2.0*sqrt(3)*eps), 1.0/(2.0*sqrt(3)*eps), -1.0/(2.0*sqrt(3)*eps), 1.0/(2.0*sqrt(3)*eps), -1.0/(2.0*sqrt(3)*eps)};
    for (d = 0; d < 6; ++d) {
      qpoints[dim*d]   = PetscRealPart(qp0[dim*d]);
      qpoints[dim*d+1] = PetscRealPart(qp0[dim*d+1]);
      qpoints[dim*d+2] = PetscRealPart(qp0[dim*d+2]);
      qweights[d]      = PetscRealPart(qw0[d]);
    }
    ++f;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceDestroy_Morley(PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(sp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceDuplicate_Morley(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscInt       order;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceSetType(spNew, PETSCDUALSPACEMORLEY);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(spNew, order);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceSetFromOptions_Morley(PetscOptionItems *PetscOptionsObject, PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscDualSpace Morley Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode PetscDualSpaceGetNumDof_Morley(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Morley *mor = (PetscDualSpace_Morley *) sp->data;

  PetscFunctionBegin;
  *numDof = mor->numDof;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceCreateHeightSubspace_Morley(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscFunctionBegin;
  *bdsp = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDualSpaceInitialize_Morley(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscDualSpaceSetFromOptions_Morley;
  sp->ops->setup             = PetscDualSpaceSetUp_Morley;
  sp->ops->view              = NULL;
  sp->ops->destroy           = PetscDualSpaceDestroy_Morley;
  sp->ops->duplicate         = PetscDualSpaceDuplicate_Morley;
  sp->ops->createheightsubspace = PetscDualSpaceCreateHeightSubspace_Morley;
  sp->ops->getsymmetries     = NULL;
  sp->ops->apply	     = PetscDualSpaceApplyDefault;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACEMATT = "mor" - A PetscDualSpace object that just does nothing

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Morley(PetscDualSpace sp)
{
  PetscDualSpace_Morley *mor;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp, &mor);CHKERRQ(ierr);
  sp->data = mor;

  mor->size = 0;

  ierr = PetscDualSpaceInitialize_Morley(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceMorleyGetSize_C", PetscDualSpaceMorleyGetSize_Morley);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceMorleySetSize_C", PetscDualSpaceMorleySetSize_Morley);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
