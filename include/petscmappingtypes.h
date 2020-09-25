#ifndef PETSCMAPPINGTYPES_H
#define PETSCMAPPINGTYPES_H

typedef struct _p_PetscMapping *PetscMapping;

typedef enum {
  IM_INVALID,
  IM_CONTIG,
  IM_DISCONTIG
} PetscMappingState;

#endif
