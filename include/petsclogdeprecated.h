#if !defined(PETSCLOGDEPRECATED_H)
#define PETSCLOGDEPRECATED_H

#include <petscsystypes.h>
#include <petscconf.h>

PETSC_DEPRECATED_TYPEDEF("PetscClassRegInfo is deprecated (since 3.20)") typedef struct {
  char        *name;
  PetscClassId classid;
} PetscClassRegInfo;

PETSC_DEPRECATED_TYPEDEF("PetscClassRegLog is deprecated (since 3.20)") typedef struct _n_PetscClassRegLog *PetscClassRegLog;
struct PETSC_DEPRECATED_STRUCT("_n_PetscClassRegLog is deprecated (since 3.20)") _n_PetscClassRegLog {
  int                numClasses;
  int                maxClasses;
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscClassRegInfo *classInfo);
};

PETSC_DEPRECATED_TYPEDEF("PetscClassPerfInfo is deprecated (since 3.20)") typedef struct {
  PetscClassId   id;
  int            creations;
  int            destructions;
  PetscLogDouble mem;
  PetscLogDouble descMem;
} PetscClassPerfInfo;

PETSC_DEPRECATED_TYPEDEF("PetscClassPerfLog is deprecated (since 3.20)") typedef struct _n_PetscClassPerfLog *PetscClassPerfLog;
struct PETSC_DEPRECATED_STRUCT("_n_PetscClassPerfLog is deprecated (since 3.20)") _n_PetscClassPerfLog {
  int                 numClasses;
  int                 maxClasses;
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscClassPerfInfo *classInfo);
};

PETSC_DEPRECATED_TYPEDEF("PetscEventRegInfo is deprecated (since 3.20)") typedef struct {
  char        *name;
  PetscClassId classid;
  PetscBool    collective;
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer;
#endif
#if defined(PETSC_HAVE_MPE)
  int mpe_id_begin;
  int mpe_id_end;
#endif
} PetscEventRegInfo;

PETSC_DEPRECATED_TYPEDEF("PetscEventRegLog is deprecated (since 3.20)") typedef struct _n_PetscEventRegLog *PetscEventRegLog;
struct PETSC_DEPRECATED_STRUCT("_n_PetscEventRegLog is deprecated (since 3.20)") _n_PetscEventRegLog {
  int                numEvents;
  int                maxEvents;
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscEventRegInfo *eventInfo); /* The registration information for each event */
};

PETSC_DEPRECATED_TYPEDEF("PetscEventPerfLog is deprecated (since 3.20)") typedef struct _n_PetscEventPerfLog *PetscEventPerfLog;
struct PETSC_DEPRECATED_STRUCT("_n_PetscEventRegLog is deprecated (since 3.20)") _n_PetscEventPerfLog {
  int                 numEvents;
  int                 maxEvents;
  PetscEventPerfInfo *eventInfo;
};

PETSC_DEPRECATED_TYPEDEF("PetscStageInfo is deprecated (since 3.20)") typedef struct PETSC_DEPRECATED_STRUCT("_PetscStageInfo is deprecated (since 3.20)") _PetscStageInfo {
  char              *name;
  PetscBool          used;
  PetscEventPerfInfo perfInfo;
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscEventPerfLog  eventLog);
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscClassPerfLog  classLog);
#if defined(PETSC_HAVE_TAU_PERFSTUBS)
  void *timer;
#endif
} PetscStageInfo;

PETSC_DEPRECATED_TYPEDEF("PetscStageLog is deprecated (since 3.20)") typedef struct _n_PetscStageLog *PetscStageLog;
struct PETSC_DEPRECATED_STRUCT("PetscStageLog is deprecated (since 3.20)") _n_PetscStageLog {
  int              numStages;
  int              maxStages;
  PetscIntStack    stack;
  int              curStage;
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscStageInfo  *stageInfo);
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscEventRegLog eventLog);
  PETSC_DEPRECATED_FIELD_IN_DEPRECATED_STRUCT(PetscClassRegLog classLog);
};



#endif /* define PETSCLOGDEPRECATED_H */
