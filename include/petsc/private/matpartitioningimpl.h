
#ifndef __MATPARTITIONINGIMPL_H
#define __MATPARTITIONINGIMPL_H

#include <petscmatpartitioning.h>
#include <petsc/private/matimpl.h>

PETSC_EXTERN PetscBool MatPartitioningRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MatPartitioningRegisterAll(void);

typedef struct _MatPartitioningOps *MatPartitioningOps;
struct _MatPartitioningOps {
  PetscErrorCode (*apply)(MatPartitioning,IS*);
  PetscErrorCode (*applynd)(MatPartitioning,IS*);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,MatPartitioning);
  PetscErrorCode (*destroy)(MatPartitioning);
  PetscErrorCode (*view)(MatPartitioning,PetscViewer);
  PetscErrorCode (*improve)(MatPartitioning,IS*);
  PetscErrorCode (*setup)(MatPartitioning);
  PetscErrorCode (*reset)(MatPartitioning);
};

struct _p_MatPartitioning {
  PETSCHEADER(struct _MatPartitioningOps);
  Mat         adj;
  PetscInt    *vertex_weights;
  PetscBool   use_vertex_weights;
  PetscReal   *part_weights;
  PetscBool   use_part_weights;
  PetscInt    n;                                 /* number of partitions */
  void        *data;
  PetscBool   setupcalled;
  PetscBool   use_edge_weights;  /* A flag indicates whether or not to use edge weights */
  PetscBool   parallel;
};

/* needed for parallel nested dissection by ParMetis and PTSCOTCH */
PETSC_INTERN PetscErrorCode MatPartitioningSizesToSep_Private(PetscInt,PetscInt[],PetscInt[],PetscInt[]);

PETSC_EXTERN PetscLogEvent MAT_Partitioning;
PETSC_EXTERN PetscLogEvent MAT_PartitioningND;

#endif
