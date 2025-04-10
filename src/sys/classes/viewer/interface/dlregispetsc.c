#include <petscdraw.h>
#include <petscviewer.h>
#include <petsc/private/viewerimpl.h>

static PetscBool PetscSysPackageInitialized = PETSC_FALSE;

/*@C
  PetscSysFinalizePackage - This function destroys everything in the system library portion of PETSc.
  It is called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscSysInitializePackage()`, `PetscFinalize()`
@*/
PetscErrorCode PetscSysFinalizePackage(void)
{
  PetscFunctionBegin;
  if (Petsc_Seq_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Seq_keyval));
  PetscSysPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscMPIInt PETSC_MPI_THREAD_REQUIRED;
PETSC_EXTERN MPI_Comm    petsc_yaml_comm;
PETSC_EXTERN PetscMPIInt Petsc_Reduction_keyval;

/*@C
  PetscSysInitializePackage - This function initializes everything in the system library portion of PETSc. It is called
  from `PetscDLLibraryRegister_petsc()` when using dynamic libraries, and in the call to `PetscInitialize()`
  when using shared or static libraries.

  Level: developer

  Note:
  This function never needs to be called by PETSc users.

.seealso: `PetscSysFinalizePackage()`, `PetscInitialize()`
@*/
PetscErrorCode PetscSysInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PetscSysPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  petsc_yaml_comm       = MPI_COMM_NULL;
  PETSC_MPI_ERROR_CLASS = MPI_ERR_LASTCODE;
#if PetscDefined(HAVE_MPI_INIT_THREAD)
  PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_FUNNELED;
#else
  PETSC_MPI_THREAD_REQUIRED = 0;
#endif
  Petsc_Counter_keyval       = MPI_KEYVAL_INVALID;
  Petsc_InnerComm_keyval     = MPI_KEYVAL_INVALID;
  Petsc_OuterComm_keyval     = MPI_KEYVAL_INVALID;
  Petsc_ShmComm_keyval       = MPI_KEYVAL_INVALID;
  Petsc_CreationIdx_keyval   = MPI_KEYVAL_INVALID;
  Petsc_Garbage_HMap_keyval  = MPI_KEYVAL_INVALID;
  Petsc_SharedWD_keyval      = MPI_KEYVAL_INVALID;
  Petsc_SharedTmp_keyval     = MPI_KEYVAL_INVALID;
  Petsc_Reduction_keyval     = MPI_KEYVAL_INVALID;
  Petsc_Seq_keyval           = MPI_KEYVAL_INVALID;
  Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;
  Petsc_Viewer_keyval        = MPI_KEYVAL_INVALID;
  Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;
  Petsc_Viewer_Binary_keyval = MPI_KEYVAL_INVALID;
  Petsc_Viewer_Draw_keyval   = MPI_KEYVAL_INVALID;
#if defined(PETSC_HAVE_HDF5)
  Petsc_Viewer_HDF5_keyval = MPI_KEYVAL_INVALID;
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
  Petsc_Viewer_Socket_keyval MPI_KEYVAL_INVALID;
#endif
  PetscSysPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Object", &PETSC_OBJECT_CLASSID));
  PetscCall(PetscClassIdRegister("Container", &PETSC_CONTAINER_CLASSID));

  /* Register Events */
  PetscCall(PetscLogEventRegister("PetscBarrier", PETSC_SMALLEST_CLASSID, &PETSC_Barrier));
  PetscCall(PetscLogEventRegister("BuildTwoSided", PETSC_SMALLEST_CLASSID, &PETSC_BuildTwoSided));
  PetscCall(PetscLogEventRegister("BuildTwoSidedF", PETSC_SMALLEST_CLASSID, &PETSC_BuildTwoSidedF));
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSC_SMALLEST_CLASSID;
    PetscCall(PetscInfoProcessClass("sys", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("null", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSC_SMALLEST_CLASSID));
  }
  PetscCall(PetscRegisterFinalize(PetscSysFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

  #if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscvec(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscmat(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscdm(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscksp(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsnes(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscts(void);
  #endif

  /*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the system level objects.

 */
  #if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsc(void)
  #else
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsys(void)
  #endif
{
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  PetscCall(PetscSysInitializePackage());
  PetscCall(PetscDrawInitializePackage());
  PetscCall(PetscViewerInitializePackage());
  PetscCall(PetscRandomInitializePackage());

  #if defined(PETSC_USE_SINGLE_LIBRARY)
  PetscCall(PetscDLLibraryRegister_petscvec());
  PetscCall(PetscDLLibraryRegister_petscmat());
  PetscCall(PetscDLLibraryRegister_petscdm());
  PetscCall(PetscDLLibraryRegister_petscksp());
  PetscCall(PetscDLLibraryRegister_petscsnes());
  PetscCall(PetscDLLibraryRegister_petscts());
  #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
