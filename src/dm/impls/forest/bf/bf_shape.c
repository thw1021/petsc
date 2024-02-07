#include <petsc/private/dmbfimpl.h>
#include <petsc/private/petscimpl.h>

PetscErrorCode DMBFShapeSetUp(DM_BF_Shape *shape, size_t n, size_t dim)
{
  size_t i;

  PetscFunctionBegin;
  PetscAssertPointer(shape, 1);
  /* clear existing content */
  PetscCall(DMBFShapeClear(shape));
  //TODO no need to destroy if list dimensions stay the same
  /* return if nothing to do */
  if (!(0 < n && 0 < dim)) { PetscFunctionReturn(PETSC_SUCCESS); }
  /* allocate and setup elements of shape object */
  PetscCall(PetscMalloc1(n + 1, &shape->list));
  PetscCall(PetscMalloc1(n * dim, &shape->list[0]));
  for (i = 1; i < n; i++) { shape->list[i] = shape->list[i - 1] + dim; }
  shape->list[n] = PETSC_NULLPTR;
  /* allocate padding */
  PetscCall(PetscMalloc1(n + 1, &shape->pad));
  shape->pad[n] = 0;
  /* set sizes */
  shape->n    = n;
  shape->dim  = dim;
  shape->size = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeClear(DM_BF_Shape *shape)
{
  PetscFunctionBegin;
  PetscAssertPointer(shape, 1);
  PetscCheck(!shape->list || shape->list[0], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Shape list was not properly initialized");
  if (shape->list && shape->list[0]) {
    PetscCall(PetscFree(shape->list[0]));
    PetscCall(PetscFree(shape->list));
  }
  if (shape->pad) { PetscCall(PetscFree(shape->pad)); }
  shape->list = PETSC_NULLPTR;
  shape->pad  = PETSC_NULLPTR;
  shape->n    = 0;
  shape->dim  = 0;
  shape->size = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline size_t _p_DMBFShapeNElements(const DM_BF_Shape *shape)
{
  return shape->n * shape->dim;
}

static inline size_t _p_DMBFShapeSize(const DM_BF_Shape *shape)
{
  size_t i, j, s, size = 0;

  if (!shape->list) { return size; }

  for (i = 0; i < shape->n; i++) {
    s = shape->list[i][0];
    for (j = 1; j < shape->dim; j++) {
      if (0 < shape->list[i][j]) { s *= shape->list[i][j]; }
    }
    size += s + shape->pad[i];
  }
  return size;
}

PetscErrorCode DMBFShapeIsSetUp(const DM_BF_Shape *shape, PetscBool *isSetUp)
{
  PetscFunctionBegin;
  PetscAssertPointer(shape, 1);
  PetscAssertPointer(isSetUp, 2);
  if (shape->list && shape->pad && 0 < shape->n && 0 < shape->dim) {
    *isSetUp = PETSC_TRUE;
  } else if (!shape->list || !shape->list[0] || !shape->pad || shape->n <= 0 || shape->dim <= 0) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Shape is not properly initialized or cleared");
  } else {
    *isSetUp = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeIsValid(const DM_BF_Shape *shape, PetscBool *isValid)
{
  PetscFunctionBegin;
  PetscAssertPointer(shape, 1);
  PetscAssertPointer(isValid, 2);
  *isValid = PETSC_TRUE;
  CHKERRQ(DMBFShapeIsSetUp(shape, isValid));
  if (!(*isValid)) { PetscFunctionReturn(PETSC_SUCCESS); }
  *isValid = (PetscBool)(shape->size == _p_DMBFShapeSize(shape));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeCheckSetUp(const DM_BF_Shape *shape)
{
  PetscBool isSetUp;

  PetscFunctionBegin;
  CHKERRQ(DMBFShapeIsSetUp(shape, &isSetUp));
  PetscCheck(isSetUp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Shape is not set up");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeCheckValid(const DM_BF_Shape *shape)
{
  PetscBool isValid;

  PetscFunctionBegin;
  CHKERRQ(DMBFShapeIsValid(shape, &isValid));
  PetscCheck(isValid, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Shape is invalid");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeCompare(const DM_BF_Shape *refShape, const DM_BF_Shape *chkShape, PetscBool *similar)
{
  PetscBool refSetUp, chkSetUp;

  PetscFunctionBegin;
  PetscAssertPointer(refShape, 1);
  PetscAssertPointer(chkShape, 2);
  PetscAssertPointer(similar, 3);
  PetscCall(DMBFShapeIsSetUp(refShape, &refSetUp));
  PetscCall(DMBFShapeIsSetUp(chkShape, &chkSetUp));
  *similar = (PetscBool)(refSetUp && chkSetUp && refShape->n == chkShape->n && refShape->dim == chkShape->dim);
  //{ //###DEV###
  //  PetscPrintf(PETSC_COMM_SELF,"DMBFShapeCompare: refSetUp=%i chkSetUp=%i refValid=%i chkValid=%i "
  //              "refShape->(n,dim)=(%i,%i) chkShape->(n,dim)=(%i,%i)\n",
  //              refSetUp,chkSetUp,refValid,chkValid,
  //              refShape->n,refShape->dim,chkShape->n,chkShape->dim);
  //}
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeCopy(DM_BF_Shape *trgShape, const DM_BF_Shape *srcShape)
{
  PetscBool isOK;

  PetscFunctionBegin;
  /* check input */
  PetscAssertPointer(trgShape, 1);
  PetscAssertPointer(srcShape, 2);
  PetscCall(DMBFShapeIsSetUp(srcShape, &isOK));
  PetscCheck(isOK, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Source shape was not set up");
  PetscCall(DMBFShapeIsValid(srcShape, &isOK));
  PetscCheck(isOK, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Source shape is invalid");
  PetscCall(DMBFShapeIsSetUp(trgShape, &isOK));
  PetscCheck(isOK, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Target shape was not set up");
  PetscCall(DMBFShapeCompare(trgShape, srcShape, &isOK));
  PetscCheck(isOK, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Target and source shapes are not similar");
  /* copy */
  PetscCall(PetscArraycpy(trgShape->list[0], srcShape->list[0], _p_DMBFShapeNElements(srcShape)));
  PetscCall(PetscArraycpy(trgShape->pad, srcShape->pad, srcShape->n));
  trgShape->size = srcShape->size;
  /* check output */
  PetscCall(DMBFShapeIsValid(trgShape, &isOK));
  PetscCheck(isOK, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Target shape is invalid");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeSet(DM_BF_Shape *shape, const size_t *elements, const size_t *pad)
{
  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape, 1);
  PetscAssertPointer(elements, 2);
  PetscCall(DMBFShapeCheckSetUp(shape));
  /* set */
  PetscCall(PetscArraycpy(shape->list[0], elements, _p_DMBFShapeNElements(shape)));
  PetscCall(PetscArraycpy(shape->pad, pad, shape->n));
  shape->size = _p_DMBFShapeSize(shape);
  PetscCall(DMBFShapeCheckValid(shape));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeGet(const DM_BF_Shape *shape, size_t **elements, size_t **pad, size_t *n, size_t *dim)
{
  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape, 1);
  PetscAssertPointer(elements, 2);
  PetscAssertPointer(pad, 3);
  PetscAssertPointer(n, 4);
  PetscAssertPointer(dim, 5);
  PetscCall(DMBFShapeCheckSetUp(shape));
  /* get */
  PetscCall(PetscMalloc1(_p_DMBFShapeNElements(shape), elements));
  PetscCall(PetscArraycpy(*elements, shape->list[0], _p_DMBFShapeNElements(shape)));
  PetscCall(PetscMalloc1(shape->n, pad));
  PetscCall(PetscArraycpy(*pad, shape->pad, shape->n));
  *n   = shape->n;
  *dim = shape->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeSetFromInt(DM_BF_Shape *shape, const PetscInt *elements)
{
  size_t k;

  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape, 1);
  PetscAssertPointer(elements, 2);
  PetscCall(DMBFShapeCheckSetUp(shape));
  /* set */
  for (k = 0; k < shape->n * shape->dim; k++) { shape->list[0][k] = (size_t)elements[k]; }
  for (k = 0; k < shape->n; k++) { shape->pad[k] = 0; }
  shape->size = _p_DMBFShapeSize(shape);
  PetscCall(DMBFShapeCheckValid(shape));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMBFShapeGetToInt(const DM_BF_Shape *shape, PetscInt **elements, PetscInt *n, PetscInt *dim)
{
  size_t k;

  PetscFunctionBegin;
  /* check */
  PetscAssertPointer(shape, 1);
  PetscAssertPointer(elements, 2);
  PetscAssertPointer(n, 3);
  PetscAssertPointer(dim, 4);
  PetscCall(DMBFShapeCheckSetUp(shape));
  /* get */
  PetscCall(PetscMalloc1(_p_DMBFShapeNElements(shape), elements));
  for (k = 0; k < shape->n * shape->dim; k++) { (*elements)[k] = (PetscInt)shape->list[0][k]; }
  *n   = shape->n;
  *dim = shape->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}
