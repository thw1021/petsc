#pragma once

#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_INTERN PetscBool      PetscViewerRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscViewerRegisterAll(void);

struct _PetscViewerOps {
  PetscErrorCode (*destroy)(PetscViewer);
  PetscErrorCode (*view)(PetscViewer, PetscViewer);
  PetscErrorCode (*flush)(PetscViewer);
  PetscErrorCode (*getsubviewer)(PetscViewer, MPI_Comm, PetscViewer *);
  PetscErrorCode (*restoresubviewer)(PetscViewer, MPI_Comm, PetscViewer *);
  PetscErrorCode (*read)(PetscViewer, void *, PetscInt, PetscInt *, PetscDataType);
  PetscErrorCode (*setfromoptions)(PetscViewer, PetscOptionItems);
  PetscErrorCode (*setup)(PetscViewer);
};

#define PETSCVIEWERCREATEVIEWEROFFPUSHESMAX 25

#define PETSCVIEWERFORMATPUSHESMAX 25
/*
   Defines the viewer data structure.
*/
typedef struct _PetscViewerOps *PetscViewerOps;
struct _p_PetscViewer {
  PETSCHEADER(struct _PetscViewerOps);
  PetscViewerFormat format, formats[PETSCVIEWERFORMATPUSHESMAX];
  int               iformat; /* number of formats that have been pushed on formats[] stack */
  void             *data;
  PetscBool         setupcalled;
};

PETSC_INTERN PetscMPIInt Petsc_Viewer_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Stdout_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Stderr_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Binary_keyval;
PETSC_INTERN PetscMPIInt Petsc_Viewer_Draw_keyval;
#if defined(PETSC_HAVE_HDF5)
PETSC_INTERN PetscMPIInt Petsc_Viewer_HDF5_keyval;
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
PETSC_INTERN PetscMPIInt Petsc_Viewer_Socket_keyval;
#endif

typedef enum {
  PETSC_COLOR_DATA,
  PETSC_COLOR_ERROR,
  PETSC_COLOR_INFO,
  PETSC_COLOR_SUCCESS,
  PETSC_COLOR_WARNING,
} PetscColorType;

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerASCIIGetColor(PetscViewer, PetscColorType, const char ***);

#define PetscIntColor_FMT "s%" PetscInt_FMT "%s"

#define PetscColorFmt(f) "%s" f "%s"
#define PetscColorArg(color,...) (color) ? (color[0]) : "", __VA_ARGS__, (color) ? (color[1]) : ""

enum {
  PETSC_REAL_FMT_DEFAULT      = 0,
  PETSC_REAL_FMT_SHORT        = 1,
  PETSC_REAL_FMT_SIGNED       = 2,
  PETSC_REAL_FMT_INF_CUTOFF   = 4,
  PETSC_REAL_FMT_SMALL_CUTOFF = 8
};

#define PetscRealFmtShort(v)       ((v) & PETSC_REAL_FMT_SHORT)
#define PetscRealFmtSigned(v)      ((v) & PETSC_REAL_FMT_SIGNED)
#define PetscRealFmtInfCutoff(v)   ((v) & PETSC_REAL_FMT_INF_CUTOFF)
#define PetscRealFmtSmallCutoff(v) ((v) & PETSC_REAL_FMT_SMALL_CUTOFF)

#define PETSC_MONITOR_REAL_LENGTH 128

PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode PetscViewerASCIIFormatMonitorReal(PetscViewer, PetscReal, PetscInt, char[]);
