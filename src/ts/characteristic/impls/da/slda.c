#include <../src/ts/characteristic/impls/da/slda.h> /*I  "petsccharacteristic.h"  I*/
#include <petscdmda.h>
#include <petscviewer.h>

static PetscErrorCode CharacteristicView_DA(Characteristic c, PetscViewer viewer)
{
  Characteristic_DA *da = (Characteristic_DA *)c->data;
  PetscBool          isascii, isstring;

  PetscFunctionBegin;
  /* Pull out field names from DM */
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  DMDA: dummy=%" PetscInt_FMT "\n", da->dummy));
  } else if (isstring) {
    PetscCall(PetscViewerStringSPrintf(viewer, "dummy %" PetscInt_FMT, da->dummy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CharacteristicDestroy_DA(Characteristic c)
{
  Characteristic_DA *da = (Characteristic_DA *)c->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CharacteristicSetUp_DA(Characteristic c)
{
  PetscMPIInt  blockLen[2];
  MPI_Aint     indices[2];
  MPI_Datatype oldtypes[2];
  PetscInt     dim, numValues;

  PetscCall(DMDAGetInfo(c->velocityDA, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (c->structured) PetscCall(PetscMPIIntCast(dim, &c->numIds));
  else c->numIds = 3;
  PetscCheck(c->numFieldComp <= MAX_COMPONENTS, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "The maximum number of fields allowed is %d, you have %" PetscInt_FMT ". You can recompile after increasing MAX_COMPONENTS.", MAX_COMPONENTS, c->numFieldComp);
  numValues = 4 + MAX_COMPONENTS;

  /* Create new MPI datatype for communication of characteristic point structs */
  blockLen[0] = 1 + c->numIds;
  indices[0]  = 0;
  oldtypes[0] = MPIU_INT;
  PetscCall(PetscMPIIntCast(numValues, &blockLen[1]));
  indices[1]  = (1 + c->numIds) * sizeof(PetscInt);
  oldtypes[1] = MPIU_SCALAR;
  PetscCallMPI(MPI_Type_create_struct(2, blockLen, indices, oldtypes, &c->itemType));
  PetscCallMPI(MPI_Type_commit(&c->itemType));

  /* Initialize the local queue for char foot values */
  PetscCall(VecGetLocalSize(c->velocity, &c->queueMax));
  PetscCall(PetscMalloc1(c->queueMax, &c->queue));
  c->queueSize = 0;

  /* Allocate communication structures */
  PetscCheck(c->numNeighbors > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Invalid number of neighbors %d. Call CharactersiticSetNeighbors() before setup.", c->numNeighbors);
  PetscCall(PetscMalloc1(c->numNeighbors, &c->needCount));
  PetscCall(PetscMalloc1(c->numNeighbors, &c->localOffsets));
  PetscCall(PetscMalloc1(c->numNeighbors, &c->fillCount));
  PetscCall(PetscMalloc1(c->numNeighbors, &c->remoteOffsets));
  PetscCall(PetscMalloc1(c->numNeighbors - 1, &c->request));
  PetscCall(PetscMalloc1(c->numNeighbors - 1, &c->status));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* [center, left, top-left, top, top-right, right, bottom-right, bottom, bottom-left] */
static PetscErrorCode DMDAGetNeighborsRank(DM da, PetscMPIInt neighbors[])
{
  DMBoundaryType bx, by;
  PetscBool      IPeriodic = PETSC_FALSE, JPeriodic = PETSC_FALSE;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscMPIInt  **procs, pi, pj, pim, pip, pjm, pjp, PIi, PJi;
  PetscInt       PI, PJ;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMDAGetInfo(da, NULL, NULL, NULL, NULL, &PI, &PJ, NULL, NULL, NULL, &bx, &by, NULL, NULL));
  PetscCall(PetscMPIIntCast(PI, &PIi));
  PetscCall(PetscMPIIntCast(PJ, &PJi));
  if (bx == DM_BOUNDARY_PERIODIC) IPeriodic = PETSC_TRUE;
  if (by == DM_BOUNDARY_PERIODIC) JPeriodic = PETSC_TRUE;

  neighbors[0] = rank;
  rank         = 0;
  PetscCall(PetscMalloc1(PJ, &procs));
  for (pj = 0; pj < PJ; pj++) {
    PetscCall(PetscMalloc1(PI, &procs[pj]));
    for (pi = 0; pi < PI; pi++) {
      procs[pj][pi] = rank;
      rank++;
    }
  }

  pi  = neighbors[0] % PI;
  pj  = neighbors[0] / PI;
  pim = pi - 1;
  if (pim < 0) pim = PIi - 1;
  pip = (pi + 1) % PIi;
  pjm = pj - 1;
  if (pjm < 0) pjm = PJi - 1;
  pjp = (pj + 1) % PJi;

  neighbors[1] = procs[pj][pim];
  neighbors[2] = procs[pjp][pim];
  neighbors[3] = procs[pjp][pi];
  neighbors[4] = procs[pjp][pip];
  neighbors[5] = procs[pj][pip];
  neighbors[6] = procs[pjm][pip];
  neighbors[7] = procs[pjm][pi];
  neighbors[8] = procs[pjm][pim];

  if (!IPeriodic) {
    if (pi == 0) neighbors[1] = neighbors[2] = neighbors[8] = neighbors[0];
    if (pi == PI - 1) neighbors[4] = neighbors[5] = neighbors[6] = neighbors[0];
  }

  if (!JPeriodic) {
    if (pj == 0) neighbors[6] = neighbors[7] = neighbors[8] = neighbors[0];
    if (pj == PJ - 1) neighbors[2] = neighbors[3] = neighbors[4] = neighbors[0];
  }

  for (pj = 0; pj < PJ; pj++) PetscCall(PetscFree(procs[pj]));
  PetscCall(PetscFree(procs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  SUBDOMAIN NEIGHBORHOOD PROCESS MAP:
    2 | 3 | 4
    __|___|__
    1 | 0 | 5
    __|___|__
    8 | 7 | 6
      |   |
*/
static PetscMPIInt DMDAGetNeighborRelative(DM da, PetscReal ir, PetscReal jr)
{
  DMDALocalInfo info;
  PetscReal     is, ie, js, je;

  PetscCallAbort(PETSC_COMM_SELF, DMDAGetLocalInfo(da, &info));
  is = (PetscReal)info.xs - 0.5;
  ie = (PetscReal)info.xs + info.xm - 0.5;
  js = (PetscReal)info.ys - 0.5;
  je = (PetscReal)info.ys + info.ym - 0.5;

  if (ir >= is && ir <= ie) { /* center column */
    if (jr >= js && jr <= je) return 0;
    else if (jr < js) return 7;
    else return 3;
  } else if (ir < is) { /* left column */
    if (jr >= js && jr <= je) return 1;
    else if (jr < js) return 8;
    else return 2;
  } else { /* right column */
    if (jr >= js && jr <= je) return 5;
    else if (jr < js) return 6;
    else return 4;
  }
}

static PetscErrorCode CharacteristicSolve_DA(Characteristic c, PetscReal dt, Vec solution)
{
  CharacteristicPointDA2D Qi;
  DM                      da = c->velocityDA;
  Vec                     velocityLocal, velocityLocalOld;
  Vec                     fieldLocal;
  DMDALocalInfo           info;
  PetscScalar           **solArray;
  void                   *velocityArray;
  void                   *velocityArrayOld;
  void                   *fieldArray;
  PetscScalar            *interpIndices;
  PetscScalar            *velocityValues, *velocityValuesOld;
  PetscScalar            *fieldValues;
  PetscMPIInt             rank;
  PetscInt                dim;
  PetscMPIInt             neighbors[9];
  PetscInt                dof;
  PetscInt                gx, gy;
  PetscInt                n, is, ie, js, je, comp;

  PetscFunctionBegin;
  c->queueSize = 0;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c), &rank));
  PetscCall(DMDAGetNeighborsRank(da, neighbors));
  PetscCall(CharacteristicSetNeighbors(c, 9, neighbors));
  PetscCall(CharacteristicSetUp(c));
  /* global and local grid info */
  PetscCall(DMDAGetInfo(da, &dim, &gx, &gy, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetLocalInfo(da, &info));
  is = info.xs;
  ie = info.xs + info.xm;
  js = info.ys;
  je = info.ys + info.ym;
  /* Allocation */
  PetscCall(PetscMalloc1(dim, &interpIndices));
  PetscCall(PetscMalloc1(c->numVelocityComp, &velocityValues));
  PetscCall(PetscMalloc1(c->numVelocityComp, &velocityValuesOld));
  PetscCall(PetscMalloc1(c->numFieldComp, &fieldValues));
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_Solve, NULL, NULL, NULL, NULL));

  /*
     PART 1, AT t-dt/2
    */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_QueueSetup, NULL, NULL, NULL, NULL));
  /* GET POSITION AT HALF TIME IN THE PAST */
  if (c->velocityInterpLocal) {
    PetscCall(DMGetLocalVector(c->velocityDA, &velocityLocal));
    PetscCall(DMGetLocalVector(c->velocityDA, &velocityLocalOld));
    PetscCall(DMGlobalToLocalBegin(c->velocityDA, c->velocity, INSERT_VALUES, velocityLocal));
    PetscCall(DMGlobalToLocalEnd(c->velocityDA, c->velocity, INSERT_VALUES, velocityLocal));
    PetscCall(DMGlobalToLocalBegin(c->velocityDA, c->velocityOld, INSERT_VALUES, velocityLocalOld));
    PetscCall(DMGlobalToLocalEnd(c->velocityDA, c->velocityOld, INSERT_VALUES, velocityLocalOld));
    PetscCall(DMDAVecGetArray(c->velocityDA, velocityLocal, &velocityArray));
    PetscCall(DMDAVecGetArray(c->velocityDA, velocityLocalOld, &velocityArrayOld));
  }
  PetscCall(PetscInfo(NULL, "Calculating position at t_{n - 1/2}\n"));
  for (Qi.j = js; Qi.j < je; Qi.j++) {
    for (Qi.i = is; Qi.i < ie; Qi.i++) {
      interpIndices[0] = Qi.i;
      interpIndices[1] = Qi.j;
      if (c->velocityInterpLocal) PetscCallBack("Charactoristic callback velocityInterpLocal",(*c->velocityInterpLocal)(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      else PetscCall(c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      Qi.x = Qi.i - velocityValues[0] * dt / 2.0;
      Qi.y = Qi.j - velocityValues[1] * dt / 2.0;

      /* Determine whether the position at t - dt/2 is local */
      Qi.proc = DMDAGetNeighborRelative(da, Qi.x, Qi.y);

      /* Check for Periodic boundaries and move all periodic points back onto the domain */
      PetscCall(DMDAMapCoordsToPeriodicDomain(da, &Qi.x, &Qi.y));
      PetscCall(CharacteristicAddPoint(c, &Qi));
    }
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_QueueSetup, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesBegin(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeLocal, NULL, NULL, NULL, NULL));
  /* Calculate velocity at t_n+1/2 (local values) */
  PetscCall(PetscInfo(NULL, "Calculating local velocities at t_{n - 1/2}\n"));
  for (n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    if (c->neighbors[Qi.proc] == rank) {
      interpIndices[0] = Qi.x;
      interpIndices[1] = Qi.y;
      if (c->velocityInterpLocal) {
        PetscCall(c->velocityInterpLocal(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
        PetscCall(c->velocityInterpLocal(velocityArrayOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
      } else {
        PetscCall(c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
        PetscCall(c->velocityInterp(c->velocityOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
      }
      Qi.x = 0.5 * (velocityValues[0] + velocityValuesOld[0]);
      Qi.y = 0.5 * (velocityValues[1] + velocityValuesOld[1]);
    }
    c->queue[n] = Qi;
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeLocal, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesEnd(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));

  /* Calculate velocity at t_n+1/2 (fill remote requests) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeRemote, NULL, NULL, NULL, NULL));
  PetscCall(PetscInfo(NULL, "Calculating %" PetscInt_FMT " remote velocities at t_{n - 1/2}\n", c->queueRemoteSize));
  for (n = 0; n < c->queueRemoteSize; n++) {
    Qi               = c->queueRemote[n];
    interpIndices[0] = Qi.x;
    interpIndices[1] = Qi.y;
    if (c->velocityInterpLocal) {
      PetscCall(c->velocityInterpLocal(velocityArray, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      PetscCall(c->velocityInterpLocal(velocityArrayOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
    } else {
      PetscCall(c->velocityInterp(c->velocity, interpIndices, c->numVelocityComp, c->velocityComp, velocityValues, c->velocityCtx));
      PetscCall(c->velocityInterp(c->velocityOld, interpIndices, c->numVelocityComp, c->velocityComp, velocityValuesOld, c->velocityCtx));
    }
    Qi.x              = 0.5 * (velocityValues[0] + velocityValuesOld[0]);
    Qi.y              = 0.5 * (velocityValues[1] + velocityValuesOld[1]);
    c->queueRemote[n] = Qi;
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeRemote, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicGetValuesBegin(c));
  PetscCall(CharacteristicGetValuesEnd(c));
  if (c->velocityInterpLocal) {
    PetscCall(DMDAVecRestoreArray(c->velocityDA, velocityLocal, &velocityArray));
    PetscCall(DMDAVecRestoreArray(c->velocityDA, velocityLocalOld, &velocityArrayOld));
    PetscCall(DMRestoreLocalVector(c->velocityDA, &velocityLocal));
    PetscCall(DMRestoreLocalVector(c->velocityDA, &velocityLocalOld));
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_HalfTimeExchange, NULL, NULL, NULL, NULL));

  /*
     PART 2, AT t-dt
  */

  /* GET POSITION AT t_n (local values) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));
  PetscCall(PetscInfo(NULL, "Calculating position at t_{n}\n"));
  for (n = 0; n < c->queueSize; n++) {
    Qi   = c->queue[n];
    Qi.x = Qi.i - Qi.x * dt;
    Qi.y = Qi.j - Qi.y * dt;

    /* Determine whether the position at t-dt is local */
    Qi.proc = DMDAGetNeighborRelative(da, Qi.x, Qi.y);

    /* Check for Periodic boundaries and move all periodic points back onto the domain */
    PetscCall(DMDAMapCoordsToPeriodicDomain(da, &Qi.x, &Qi.y));

    c->queue[n] = Qi;
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesBegin(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));

  /* GET VALUE AT FULL TIME IN THE PAST (LOCAL REQUESTS) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));
  if (c->fieldInterpLocal) {
    PetscCall(DMGetLocalVector(c->fieldDA, &fieldLocal));
    PetscCall(DMGlobalToLocalBegin(c->fieldDA, c->field, INSERT_VALUES, fieldLocal));
    PetscCall(DMGlobalToLocalEnd(c->fieldDA, c->field, INSERT_VALUES, fieldLocal));
    PetscCall(DMDAVecGetArray(c->fieldDA, fieldLocal, &fieldArray));
  }
  PetscCall(PetscInfo(NULL, "Calculating local field at t_{n}\n"));
  for (n = 0; n < c->queueSize; n++) {
    if (c->neighbors[c->queue[n].proc] == rank) {
      interpIndices[0] = c->queue[n].x;
      interpIndices[1] = c->queue[n].y;
      if (c->fieldInterpLocal) PetscCall(c->fieldInterpLocal(fieldArray, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
      else PetscCall(c->fieldInterp(c->field, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
      for (comp = 0; comp < c->numFieldComp; comp++) c->queue[n].field[comp] = fieldValues[comp];
    }
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeLocal, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicSendCoordinatesEnd(c));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));

  /* GET VALUE AT FULL TIME IN THE PAST (REMOTE REQUESTS) */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeRemote, NULL, NULL, NULL, NULL));
  PetscCall(PetscInfo(NULL, "Calculating %" PetscInt_FMT " remote field points at t_{n}\n", c->queueRemoteSize));
  for (n = 0; n < c->queueRemoteSize; n++) {
    interpIndices[0] = c->queueRemote[n].x;
    interpIndices[1] = c->queueRemote[n].y;

    /* for debugging purposes */
    if (1) { /* hacked bounds test...let's do better */
      PetscScalar im = interpIndices[0];
      PetscScalar jm = interpIndices[1];

      PetscCheck((im >= (PetscScalar)is - 1.) && (im <= (PetscScalar)ie) && (jm >= (PetscScalar)js - 1.) && (jm <= (PetscScalar)je), PETSC_COMM_SELF, PETSC_ERR_LIB, "Nonlocal point: (%g,%g)", (double)PetscAbsScalar(im), (double)PetscAbsScalar(jm));
    }

    if (c->fieldInterpLocal) PetscCall(c->fieldInterpLocal(fieldArray, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
    else PetscCall(c->fieldInterp(c->field, interpIndices, c->numFieldComp, c->fieldComp, fieldValues, c->fieldCtx));
    for (comp = 0; comp < c->numFieldComp; comp++) c->queueRemote[n].field[comp] = fieldValues[comp];
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeRemote, NULL, NULL, NULL, NULL));

  PetscCall(PetscLogEventBegin(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));
  PetscCall(CharacteristicGetValuesBegin(c));
  PetscCall(CharacteristicGetValuesEnd(c));
  if (c->fieldInterpLocal) {
    PetscCall(DMDAVecRestoreArray(c->fieldDA, fieldLocal, &fieldArray));
    PetscCall(DMRestoreLocalVector(c->fieldDA, &fieldLocal));
  }
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_FullTimeExchange, NULL, NULL, NULL, NULL));

  /* Return field of characteristics at t_n-1 */
  PetscCall(PetscLogEventBegin(CHARACTERISTIC_DAUpdate, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetInfo(c->fieldDA, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAVecGetArray(c->fieldDA, solution, &solArray));
  for (n = 0; n < c->queueSize; n++) {
    Qi = c->queue[n];
    for (comp = 0; comp < c->numFieldComp; comp++) solArray[Qi.j][Qi.i * dof + c->fieldComp[comp]] = Qi.field[comp];
  }
  PetscCall(DMDAVecRestoreArray(c->fieldDA, solution, &solArray));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_DAUpdate, NULL, NULL, NULL, NULL));
  PetscCall(PetscLogEventEnd(CHARACTERISTIC_Solve, NULL, NULL, NULL, NULL));

  /* Cleanup */
  PetscCall(PetscFree(interpIndices));
  PetscCall(PetscFree(velocityValues));
  PetscCall(PetscFree(velocityValuesOld));
  PetscCall(PetscFree(fieldValues));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode CharacteristicCreate_DA(Characteristic c)
{
  Characteristic_DA *da;

  PetscFunctionBegin;
  PetscCall(PetscNew(&da));
  PetscCall(PetscMemzero(da, sizeof(Characteristic_DA)));
  c->data = (void *)da;

  c->ops->setup   = CharacteristicSetUp_DA;
  c->ops->destroy = CharacteristicDestroy_DA;
  c->ops->view    = CharacteristicView_DA;
  c->ops->solve   = CharacteristicSolve_DA;

  da->dummy = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Checks for periodicity of a DM and Maps points outside of a domain back onto the domain
   using appropriate periodicity. At the moment assumes only a 2-D DMDA.
 */
PetscErrorCode DMDAMapCoordsToPeriodicDomain(DM da, PetscScalar *x, PetscScalar *y)
{
  DMBoundaryType bx, by;
  PetscInt       dim, gx, gy;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, &dim, &gx, &gy, NULL, NULL, NULL, NULL, NULL, NULL, &bx, &by, NULL, NULL));

  if (bx == DM_BOUNDARY_PERIODIC) {
    while (*x >= (PetscScalar)gx) *x -= (PetscScalar)gx;
    while (*x < 0.0) *x += (PetscScalar)gx;
  }
  if (by == DM_BOUNDARY_PERIODIC) {
    while (*y >= (PetscScalar)gy) *y -= (PetscScalar)gy;
    while (*y < 0.0) *y += (PetscScalar)gy;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
