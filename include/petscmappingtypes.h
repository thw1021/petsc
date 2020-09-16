#ifndef PETSCMAPPINGTYPES_H
#define PETSCMAPPINGTYPES_H

typedef struct _p_PetscMapping *PetscMapping;

typedef enum {
  NONE_VALID = -1,
  KEY_VALID = 0,
  MAPS_VALID,
  ALL_VALID,
  INDICES_VALID
} PetscMappingState;

#endif
