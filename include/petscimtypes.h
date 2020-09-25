#ifndef IMTYPES_H
#define IMTYPES_H

typedef struct _p_IM *IM;

typedef enum {
  IM_INVALID = 0,
  IM_CONTIG = 1,
  IM_DISCONTIG = 2
} IMState;

typedef enum {
  IM_LOCAL = 0,
  IM_GLOBAL = 1
} IMOpMode;
#endif
