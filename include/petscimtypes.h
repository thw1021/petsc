#ifndef IMTYPES_H
#define IMTYPES_H

typedef struct _p_IM *IM;

typedef enum {
  IM_INVALID = 0,
  IM_CONTIGUOUS = 1,
  IM_ARRAY = 2
} IMState;

typedef enum {
  IM_LOCAL = 0,
  IM_GLOBAL = 1,
  IM_MAX_MODE = 2
} IMOpMode;
#endif
