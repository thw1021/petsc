#include <petsc/private/characteristicimpl.h> /*I "petsccharacteristic.h" I*/
#include <petscviewer.h>

PetscClassId  CHARACTERISTIC_CLASSID;
PetscLogEvent CHARACTERISTIC_SetUp, CHARACTERISTIC_Solve, CHARACTERISTIC_QueueSetup, CHARACTERISTIC_DAUpdate;
PetscLogEvent CHARACTERISTIC_HalfTimeLocal, CHARACTERISTIC_HalfTimeRemote, CHARACTERISTIC_HalfTimeExchange;
PetscLogEvent CHARACTERISTIC_FullTimeLocal, CHARACTERISTIC_FullTimeRemote, CHARACTERISTIC_FullTimeExchange;
/*
   Contains the list of registered characteristic routines
*/
PetscFunctionList CharacteristicList              = NULL;
PetscBool         CharacteristicRegisterAllCalled = PETSC_FALSE;

static PetscErrorCode CharacteristicHeapSort(Characteristic, Queue, PetscInt);
static PetscErrorCode CharacteristicSiftDown(Characteristic, Queue, PetscInt, PetscInt);

static PetscErrorCode CharacteristicView(Characteristic c, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)c), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(c, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (!isascii) PetscTryTypeMethod(c, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  CharacteristicDestroy - Destroys a `Characteristic` context created with `CharacteristicCreate()`

  Collective

  Input Parameter:
. c - the `Characteristic` context

  Level: beginner

.seealso: `Characteristic`, `CharacteristicCreate()`
@*/
PetscErrorCode CharacteristicDestroy(Characteristic *c)
{
  PetscFunctionBegin;
  if (!*c) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*c, CHARACTERISTIC_CLASSID, 1);
  if (--((PetscObject)*c)->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscTryTypeMethod(*c, destroy);
  PetscCallMPI(MPI_Type_free(&(*c)->itemType));
  PetscCall(PetscFree((*c)->queue));
  PetscCall(PetscFree((*c)->queueLocal));
  PetscCall(PetscFree((*c)->queueRemote));
  PetscCall(PetscFree((*c)->neighbors));
  PetscCall(PetscFree((*c)->needCount));
  PetscCall(PetscFree((*c)->localOffsets));
  PetscCall(PetscFree((*c)->fillCount));
  PetscCall(PetscFree((*c)->remoteOffsets));
  PetscCall(PetscFree((*c)->request));
  PetscCall(PetscFree((*c)->status));
  PetscCall(PetscHeaderDestroy(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  CharacteristicCreate - Creates a `Characteristic` context for use with the Method of Characteristics

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. c - the `Characteristic` context

  Level: beginner

.seealso: `Characteristic`, `CharacteristicDestroy()`
@*/
PetscErrorCode CharacteristicCreate(MPI_Comm comm, Characteristic *c)
{
  Characteristic newC;

  PetscFunctionBegin;
  PetscAssertPointer(c, 2);
  *c = NULL;
  PetscCall(CharacteristicInitializePackage());

  PetscCall(PetscHeaderCreate(newC, CHARACTERISTIC_CLASSID, "Characteristic", "Characteristic", "Characteristic", comm, CharacteristicDestroy, CharacteristicView));
  *c = newC;

  newC->structured          = PETSC_TRUE;
  newC->numIds              = 0;
  newC->velocityDA          = NULL;
  newC->velocity            = NULL;
  newC->velocityOld         = NULL;
  newC->numVelocityComp     = 0;
  newC->velocityComp        = NULL;
  newC->velocityInterp      = NULL;
  newC->velocityInterpLocal = NULL;
  newC->velocityCtx         = NULL;
  newC->fieldDA             = NULL;
  newC->field               = NULL;
  newC->numFieldComp        = 0;
  newC->fieldComp           = NULL;
  newC->fieldInterp         = NULL;
  newC->fieldInterpLocal    = NULL;
  newC->fieldCtx            = NULL;
  newC->itemType            = 0;
  newC->queue               = NULL;
  newC->queueSize           = 0;
  newC->queueMax            = 0;
  newC->queueLocal          = NULL;
  newC->queueLocalSize      = 0;
  newC->queueLocalMax       = 0;
  newC->queueRemote         = NULL;
  newC->queueRemoteSize     = 0;
  newC->queueRemoteMax      = 0;
  newC->numNeighbors        = 0;
  newC->neighbors           = NULL;
  newC->needCount           = NULL;
  newC->localOffsets        = NULL;
  newC->fillCount           = NULL;
  newC->remoteOffsets       = NULL;
  newC->request             = NULL;
  newC->status              = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  CharacteristicSetType - Builds Characteristic for a particular solver.

  Logically Collective

  Input Parameters:
+ c    - the method of characteristics context
- type - a known method

  Options Database Key:
. -characteristic_type method - Sets the method; use -help for a list
    of available methods

  Level: intermediate

  Note:
  See "include/petsccharacteristic.h" for available methods

.seealso: [](ch_ts), `CharacteristicType`
@*/
PetscErrorCode CharacteristicSetType(Characteristic c, CharacteristicType type)
{
  PetscBool match;
  PetscErrorCode (*r)(Characteristic);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)c, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  if (c->data) {
    /* destroy the old private Characteristic context */
    PetscUseTypeMethod(c, destroy);
    c->ops->destroy = NULL;
    c->data         = NULL;
  }

  PetscCall(PetscFunctionListFind(CharacteristicList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)c), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Characteristic type given: %s", type);
  c->setupcalled = PETSC_FALSE;
  PetscCall((*r)(c));
  PetscCall(PetscObjectChangeTypeName((PetscObject)c, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  CharacteristicSetUp - Sets up the internal data structures for the
  later use of a `Charactoristic` .

  Collective

  Input Parameter:
. c - context obtained from CharacteristicCreate()

  Level: developer

.seealso: [](ch_ts), `Characteristic`, `CharacteristicCreate()`, `CharacteristicSolve()`, `CharacteristicDestroy()`
@*/
PetscErrorCode CharacteristicSetUp(Characteristic c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);

  if (!((PetscObject)c)->type_name) PetscCall(CharacteristicSetType(c, CHARACTERISTICDA));

  if (c->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_SetUp, c, NULL, NULL, NULL));
  if (!c->setupcalled) PetscUseTypeMethod(c, setup);
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_SetUp, c, NULL, NULL, NULL));
  c->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  CharacteristicRegister -  Adds an approarch to the method of characteristics package.

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new approach
- function - routine to create method context

  Level: advanced

  Example Usage:
.vb
    CharacteristicRegister("my_char", MyCharCreate);
.ve

  Then, your Characteristic type can be chosen with the procedural interface via
.vb
    CharacteristicCreate(MPI_Comm, Characteristic* &char);
    CharacteristicSetType(char,"my_char");
.ve
  or at runtime via the option
.vb
    -characteristic_type my_char
.ve

  Notes:
  `CharacteristicRegister()` may be called multiple times to add several approaches.

.seealso: [](ch_ts), `CharacteristicRegisterAll()`, `CharacteristicRegisterDestroy()`
@*/
PetscErrorCode CharacteristicRegister(const char sname[], PetscErrorCode (*function)(Characteristic))
{
  PetscFunctionBegin;
  PetscCall(CharacteristicInitializePackage());
  PetscCall(PetscFunctionListAdd(&CharacteristicList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSetVelocityInterpolation(Characteristic c, DM da, Vec v, Vec vOld, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(Vec, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), PetscCtx ctx)
{
  PetscFunctionBegin;
  c->velocityDA      = da;
  c->velocity        = v;
  c->velocityOld     = vOld;
  c->numVelocityComp = numComponents;
  c->velocityComp    = components;
  c->velocityInterp  = interp;
  c->velocityCtx     = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSetVelocityInterpolationLocal(Characteristic c, DM da, Vec v, Vec vOld, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(void *, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), PetscCtx ctx)
{
  PetscFunctionBegin;
  c->velocityDA          = da;
  c->velocity            = v;
  c->velocityOld         = vOld;
  c->numVelocityComp     = numComponents;
  c->velocityComp        = components;
  c->velocityInterpLocal = interp;
  c->velocityCtx         = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSetFieldInterpolation(Characteristic c, DM da, Vec v, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(Vec, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), PetscCtx ctx)
{
  PetscFunctionBegin;
#if 0
  PetscCheck(numComponents <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP, "Fields with more than 2 components are not supported. Send mail to petsc-maint@mcs.anl.gov.");
#endif
  c->fieldDA      = da;
  c->field        = v;
  c->numFieldComp = numComponents;
  c->fieldComp    = components;
  c->fieldInterp  = interp;
  c->fieldCtx     = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSetFieldInterpolationLocal(Characteristic c, DM da, Vec v, PetscInt numComponents, PetscInt components[], PetscErrorCode (*interp)(void *, PetscReal[], PetscInt, PetscInt[], PetscScalar[], void *), PetscCtx ctx)
{
  PetscFunctionBegin;
#if 0
  PetscCheck(numComponents <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP, "Fields with more than 2 components are not supported. Send mail to petsc-maint@mcs.anl.gov.");
#endif
  c->fieldDA          = da;
  c->field            = v;
  c->numFieldComp     = numComponents;
  c->fieldComp        = components;
  c->fieldInterpLocal = interp;
  c->fieldCtx         = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  CharacteristicSolve - Apply the Method of Characteristics solver

  Collective

  Input Parameters:
+ c        - context obtained from `CharacteristicCreate()`
. dt       - the time-step
- solution - vector holding the solution

  Level: developer

.seealso: [](ch_ts), `Characteristic`, `CharacteristicCreate()`, `CharacteristicDestroy()`
@*/
PetscErrorCode CharacteristicSolve(Characteristic c, PetscReal dt, Vec solution)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(c, CHARACTERISTIC_CLASSID, 1);
  PetscValidHeaderSpecific(solution, VEC_CLASSID, 3);
  PetscUseTypeMethod(c, solve, dt, solution);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSetNeighbors(Characteristic c, PetscInt numNeighbors, PetscMPIInt neighbors[])
{
  PetscFunctionBegin;
  PetscCall(PetscMPIIntCast(numNeighbors, &c->numNeighbors));
  PetscCall(PetscFree(c->neighbors));
  PetscCall(PetscMalloc1(numNeighbors, &c->neighbors));
  PetscCall(PetscArraycpy(c->neighbors, neighbors, numNeighbors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicAddPoint(Characteristic c, CharacteristicPointDA2D *point)
{
  PetscFunctionBegin;
  PetscCheck(c->queueSize < c->queueMax, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exceeded maximum queue size %" PetscInt_FMT, c->queueMax);
  c->queue[c->queueSize++] = *point;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSendCoordinatesBegin(Characteristic c)
{
  PetscMPIInt rank, tag = 121;
  PetscInt    i, n;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c), &rank));
  PetscCall(CharacteristicHeapSort(c, c->queue, c->queueSize));
  PetscCall(PetscArrayzero(c->needCount, c->numNeighbors));
  for (i = 0; i < c->queueSize; i++) c->needCount[c->queue[i].proc]++;
  c->fillCount[0] = 0;
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPIU_Irecv(&c->fillCount[n], 1, MPIU_INT, c->neighbors[n], tag, PetscObjectComm((PetscObject)c), &c->request[n - 1]));
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPIU_Send(&c->needCount[n], 1, MPIU_INT, c->neighbors[n], tag, PetscObjectComm((PetscObject)c)));
  PetscCallMPI(MPI_Waitall(c->numNeighbors - 1, c->request, c->status));
  /* Initialize the remote queue */
  c->queueLocalMax = c->localOffsets[0] = 0;
  c->queueRemoteMax = c->remoteOffsets[0] = 0;
  for (n = 1; n < c->numNeighbors; n++) {
    c->remoteOffsets[n] = c->queueRemoteMax;
    c->queueRemoteMax += c->fillCount[n];
    c->localOffsets[n] = c->queueLocalMax;
    c->queueLocalMax += c->needCount[n];
  }
  /* HACK BEGIN */
  for (n = 1; n < c->numNeighbors; n++) c->localOffsets[n] += c->needCount[0];
  c->needCount[0] = 0;
  /* HACK END */
  if (c->queueRemoteMax) PetscCall(PetscMalloc1(c->queueRemoteMax, &c->queueRemote));
  else c->queueRemote = NULL;
  c->queueRemoteSize = c->queueRemoteMax;

  /* Send and Receive requests for values at t_n+1/2, giving the coordinates for interpolation */
  for (n = 1; n < c->numNeighbors; n++) {
    PetscCall(PetscInfo(NULL, "Receiving %" PetscInt_FMT " requests for values from proc %d\n", c->fillCount[n], c->neighbors[n]));
    PetscCallMPI(MPIU_Irecv(&(c->queueRemote[c->remoteOffsets[n]]), c->fillCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c), &c->request[n - 1]));
  }
  for (n = 1; n < c->numNeighbors; n++) {
    PetscCall(PetscInfo(NULL, "Sending %" PetscInt_FMT " requests for values from proc %d\n", c->needCount[n], c->neighbors[n]));
    PetscCallMPI(MPIU_Send(&(c->queue[c->localOffsets[n]]), c->needCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicSendCoordinatesEnd(Characteristic c)
{
#if 0
  PetscMPIInt rank;
  PetscInt    n;
#endif

  PetscFunctionBegin;
  PetscCallMPI(MPI_Waitall(c->numNeighbors - 1, c->request, c->status));
#if 0
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c), &rank));
  for (n = 0; n < c->queueRemoteSize; n++) PetscCheck(c->neighbors[c->queueRemote[n].proc] != rank,PETSC_COMM_SELF,PETSC_ERR_PLIB, "This is messed up, n = %d proc = %d", n, c->queueRemote[n].proc);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicGetValuesBegin(Characteristic c)
{
  PetscMPIInt tag = 121;
  PetscInt    n;

  PetscFunctionBegin;
  /* SEND AND RECEIVE FILLED REQUESTS for velocities at t_n+1/2 */
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPIU_Irecv(&(c->queue[c->localOffsets[n]]), c->needCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c), &c->request[n - 1]));
  for (n = 1; n < c->numNeighbors; n++) PetscCallMPI(MPIU_Send(&(c->queueRemote[c->remoteOffsets[n]]), c->fillCount[n], c->itemType, c->neighbors[n], tag, PetscObjectComm((PetscObject)c)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CharacteristicGetValuesEnd(Characteristic c)
{
  PetscFunctionBegin;
  PetscCallMPI(MPI_Waitall(c->numNeighbors - 1, c->request, c->status));
  /* Free queue of requests from other procs */
  PetscCall(PetscFree(c->queueRemote));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Based on code from http://linux.wku.edu/~lamonml/algor/sort/heap.html
*/
static PetscErrorCode CharacteristicHeapSort(Characteristic c, Queue queue, PetscInt size)
{
  CharacteristicPointDA2D temp;
  PetscInt                n;

  PetscFunctionBegin;
  if (0) { /* Check the order of the queue before sorting */
    PetscCall(PetscInfo(NULL, "Before Heap sort\n"));
    for (n = 0; n < size; n++) PetscCall(PetscInfo(NULL, "%" PetscInt_FMT " %d\n", n, queue[n].proc));
  }

  /* SORTING PHASE */
  for (n = (size / 2) - 1; n >= 0; n--) PetscCall(CharacteristicSiftDown(c, queue, n, size - 1)); /* Rich had size-1 here, Matt had size*/
  for (n = size - 1; n >= 1; n--) {
    temp     = queue[0];
    queue[0] = queue[n];
    queue[n] = temp;
    PetscCall(CharacteristicSiftDown(c, queue, 0, n - 1));
  }
  if (0) { /* Check the order of the queue after sorting */
    PetscCall(PetscInfo(NULL, "Avter  Heap sort\n"));
    for (n = 0; n < size; n++) PetscCall(PetscInfo(NULL, "%" PetscInt_FMT " %d\n", n, queue[n].proc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Based on code from http://linux.wku.edu/~lamonml/algor/sort/heap.html
*/
static PetscErrorCode CharacteristicSiftDown(Characteristic c, Queue queue, PetscInt root, PetscInt bottom)
{
  PetscBool               done = PETSC_FALSE;
  PetscInt                maxChild;
  CharacteristicPointDA2D temp;

  PetscFunctionBegin;
  while ((root * 2 <= bottom) && (!done)) {
    if (root * 2 == bottom) maxChild = root * 2;
    else if (queue[root * 2].proc > queue[root * 2 + 1].proc) maxChild = root * 2;
    else maxChild = root * 2 + 1;

    if (queue[root].proc < queue[maxChild].proc) {
      temp            = queue[root];
      queue[root]     = queue[maxChild];
      queue[maxChild] = temp;
      root            = maxChild;
    } else done = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
