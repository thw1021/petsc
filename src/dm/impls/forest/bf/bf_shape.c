#include <petsc/private/dmbfimpl.h>
#include <petsc/private/petscimpl.h>

PetscErrorCode DMBFShapeSetUp(DM_BF_Shape *shape, size_t n, size_t dim)
{
  size_t         i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertPointer(shape,1);
  /* clear existing content */
  ierr = DMBFShapeClear(shape);CHKERRQ(ierr);
  //TODO no need to destroy if list dimensions stay the same
  /* return if nothing to do */
  if (!(0<n && 0<dim)) {
    PetscFunctionReturn(0);
  }
  /* allocate and setup elements of shape object */
  ierr = PetscMalloc1(n+1,&shape->list);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*dim,&shape->list[0]);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    shape->list[i] = shape->list[i-1] + dim;
  }
  shape->list[n] = PETSC_NULLPTR;
  /* allocate padding */
  ierr = PetscMalloc1(n+1,&shape->pad);CHKERRQ(ierr);
  shape->pad[n] = 0;
  /* set sizes */
  shape->n    = n;
  shape->dim  = dim;
  shape->size = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeClear(DM_BF_Shape *shape)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertPointer(shape,1);
  if (shape->list && !shape->list[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Shape list is not been properly initialized");
  if (shape->list && shape->list[0]) {
    ierr = PetscFree(shape->list[0]);CHKERRQ(ierr);
    ierr = PetscFree(shape->list);CHKERRQ(ierr);
  }
  if (shape->pad) {
    ierr = PetscFree(shape->pad);CHKERRQ(ierr);
  }
  shape->list = PETSC_NULLPTR;
  shape->pad  = PETSC_NULLPTR;
  shape->n    = 0;
  shape->dim  = 0;
  shape->size = 0;
  PetscFunctionReturn(0);
}

static inline size_t _p_DMBFShapeNElements(const DM_BF_Shape *shape)
{
  return shape->n*shape->dim;
}

static inline size_t _p_DMBFShapeSize(const DM_BF_Shape *shape)
{
  size_t        i, j, s, size=0;

  if (!shape->list) {
    return size;
  }

  for (i=0; i<shape->n; i++) {
    s = shape->list[i][0];
    for (j=1; j<shape->dim; j++) {
      if (0 < shape->list[i][j]) {
        s *= shape->list[i][j];
      }
    }
    size += s + shape->pad[i];
  }
  return size;
}

PetscErrorCode DMBFShapeIsSetUp(const DM_BF_Shape *shape, PetscBool *isSetUp)
{
  PetscFunctionBegin;
  PetscAssertPointer(shape,1);
  PetscAssertPointer(isSetUp,2);
  if (shape->list && shape->pad && 0 < shape->n && 0 < shape->dim) {
    *isSetUp = PETSC_TRUE;
  } else if (!shape->list || !shape->list[0] || !shape->pad || shape->n <= 0 || shape->dim <= 0) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Shape is not properly initialized or cleared");
  } else {
    *isSetUp = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeIsValid(const DM_BF_Shape *shape, PetscBool *isValid)
{
  PetscFunctionBegin;
  PetscAssertPointer(shape,1);
  PetscAssertPointer(isValid,2);
  *isValid = PETSC_TRUE;
  CHKERRQ( DMBFShapeIsSetUp(shape,isValid) );
  if (!(*isValid)) {
    PetscFunctionReturn(0);
  }
  *isValid = (shape->size == _p_DMBFShapeSize(shape));
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeCheckSetUp(const DM_BF_Shape *shape)
{
  PetscBool      isSetUp;

  PetscFunctionBegin;
  CHKERRQ( DMBFShapeIsSetUp(shape,&isSetUp) );
  if (!isSetUp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Shape is not set up");
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeCheckValid(const DM_BF_Shape *shape)
{
  PetscBool      isValid;

  PetscFunctionBegin;
  CHKERRQ( DMBFShapeIsValid(shape,&isValid) );
  if (!isValid) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Shape is invalid");
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeCompare(const DM_BF_Shape *refShape, const DM_BF_Shape *chkShape, PetscBool *similar)
{
  PetscBool      refSetUp, chkSetUp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscAssertPointer(refShape,1);
  PetscAssertPointer(chkShape,2);
  PetscAssertPointer(similar,3);
  ierr = DMBFShapeIsSetUp(refShape,&refSetUp);CHKERRQ(ierr);
  ierr = DMBFShapeIsSetUp(chkShape,&chkSetUp);CHKERRQ(ierr);
  *similar = (refSetUp && chkSetUp && refShape->n == chkShape->n && refShape->dim == chkShape->dim);
//{ //###DEV###
//  PetscPrintf(PETSC_COMM_SELF,"DMBFShapeCompare: refSetUp=%i chkSetUp=%i refValid=%i chkValid=%i "
//              "refShape->(n,dim)=(%i,%i) chkShape->(n,dim)=(%i,%i)\n",
//              refSetUp,chkSetUp,refValid,chkValid,
//              refShape->n,refShape->dim,chkShape->n,chkShape->dim);
//}
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeCopy(DM_BF_Shape *trgShape, const DM_BF_Shape *srcShape)
{
  PetscBool      isOK;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check input */
  PetscAssertPointer(trgShape,1);
  PetscAssertPointer(srcShape,2);
  ierr = DMBFShapeIsSetUp(srcShape,&isOK);CHKERRQ(ierr);
  if (!isOK) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Source shape was not set up");
  ierr = DMBFShapeIsValid(srcShape,&isOK);CHKERRQ(ierr);
  if (!isOK) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Source shape is invalid");
  ierr = DMBFShapeIsSetUp(trgShape,&isOK);CHKERRQ(ierr);
  if (!isOK) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Target shape was not set up");
  ierr = DMBFShapeCompare(trgShape,srcShape,&isOK);CHKERRQ(ierr);
  if (!isOK) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Target and source shapes are not similar");
  /* copy */
  ierr = PetscArraycpy(trgShape->list[0],srcShape->list[0],_p_DMBFShapeNElements(srcShape));CHKERRQ(ierr);
  ierr = PetscArraycpy(trgShape->pad,srcShape->pad,srcShape->n);CHKERRQ(ierr);
  trgShape->size = srcShape->size;
  /* check output */
  ierr = DMBFShapeIsValid(trgShape,&isOK);CHKERRQ(ierr);
  if (!isOK) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Target shape is invalid");
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeSet(DM_BF_Shape *shape, const size_t *elements, const size_t *pad)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape,1);
  PetscAssertPointer(elements,2);
  ierr = DMBFShapeCheckSetUp(shape);CHKERRQ(ierr);
  /* set */
  ierr = PetscArraycpy(shape->list[0],elements,_p_DMBFShapeNElements(shape));CHKERRQ(ierr);
  ierr = PetscArraycpy(shape->pad,pad,shape->n);CHKERRQ(ierr);
  shape->size = _p_DMBFShapeSize(shape);
  ierr = DMBFShapeCheckValid(shape);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeGet(const DM_BF_Shape *shape, size_t **elements, size_t **pad, size_t *n, size_t *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape,1);
  PetscAssertPointer(elements,2);
  PetscAssertPointer(n,3);
  PetscAssertPointer(dim,4);
  ierr = DMBFShapeCheckSetUp(shape);CHKERRQ(ierr);
  /* get */
  ierr = PetscMalloc1(_p_DMBFShapeNElements(shape),elements);CHKERRQ(ierr);
  ierr = PetscArraycpy(*elements,shape->list[0],_p_DMBFShapeNElements(shape));CHKERRQ(ierr);
  ierr = PetscMalloc1(shape->n,pad);CHKERRQ(ierr);
  ierr = PetscArraycpy(*pad,shape->pad,shape->n);CHKERRQ(ierr);
  *n   = shape->n;
  *dim = shape->dim;
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeSetFromInt(DM_BF_Shape *shape, const PetscInt *elements)
{
  size_t         k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape,1);
  PetscAssertPointer(elements,2);
  ierr = DMBFShapeCheckSetUp(shape);CHKERRQ(ierr);
  /* set */
  for (k=0; k<shape->n*shape->dim; k++) {
    shape->list[0][k] = (size_t)elements[k];
  }
  for (k=0; k<shape->n; k++) {
    shape->pad[k] = 0;
  }
  shape->size = _p_DMBFShapeSize(shape);
  ierr = DMBFShapeCheckValid(shape);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMBFShapeGetToInt(const DM_BF_Shape *shape, PetscInt **elements, PetscInt *n, PetscInt *dim)
{
  size_t         k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape,1);
  PetscAssertPointer(elements,2);
  PetscAssertPointer(n,3);
  PetscAssertPointer(dim,4);
  ierr = DMBFShapeCheckSetUp(shape);CHKERRQ(ierr);
  /* get */
  ierr = PetscMalloc1(_p_DMBFShapeNElements(shape),elements);CHKERRQ(ierr);
  for (k=0; k<shape->n*shape->dim; k++) {
    (*elements)[k] = (PetscInt)shape->list[0][k];
  }
  *n   = shape->n;
  *dim = shape->dim;
  PetscFunctionReturn(0);
}
