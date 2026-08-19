/*
      Objects which encapsulate solver adaptation operations based on a mesh
*/
#pragma once

#include <petscdm.h>
#include <petscksptypes.h>
#include <petscsnestypes.h>
#include <petsctstypes.h>
#include <petsctaotypes.h>
#include <petscdmsolveadaptortypes.h>

/* SUBMANSEC = DM */

PETSC_EXTERN PetscClassId DMSOLVEADAPTOR_CLASSID;

/*J
   DMSolveAdaptorType - String with the name of a PETSc DMSolveAdaptor type

   Level: beginner

.seealso: [](dm_adaptor_table), [](ch_unstructured), `DMSolveAdaptorCreate()`, `DMSolveAdaptor`, `DMSolveAdaptorRegister()`
J*/
typedef const char *DMSolveAdaptorType;
#define DMSOLVEADAPTORRESIDUAL "residual"

PETSC_EXTERN PetscFunctionList DMSolveAdaptorList;

PETSC_EXTERN PetscErrorCode DMSolveAdaptorCreate(MPI_Comm, DMSolveAdaptor *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetType(DMSolveAdaptor, DMSolveAdaptorType);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorGetType(DMSolveAdaptor, DMSolveAdaptorType *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorRegister(const char[], PetscErrorCode (*)(DMSolveAdaptor));
PETSC_EXTERN PetscErrorCode DMSolveAdaptorRegisterAll(void);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetOptionsPrefix(DMSolveAdaptor, const char[]);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetFromOptions(DMSolveAdaptor);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetUp(DMSolveAdaptor);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorView(DMSolveAdaptor, PetscViewer);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorDestroy(DMSolveAdaptor *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorGetSequenceLength(DMSolveAdaptor, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetSequenceLength(DMSolveAdaptor, PetscInt);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorGetLinearSolver(DMSolveAdaptor, KSP *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetLinearSolver(DMSolveAdaptor, KSP);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorGetNonlinearSolver(DMSolveAdaptor, SNES *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetNonlinearSolver(DMSolveAdaptor, SNES);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorGetTimestepper(DMSolveAdaptor, TS *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetTimestepper(DMSolveAdaptor, TS);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorGetOptimizer(DMSolveAdaptor, Tao *);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorSetOptimizer(DMSolveAdaptor, Tao);
PETSC_EXTERN PetscErrorCode DMSolveAdaptorAdapt(DMSolveAdaptor);
