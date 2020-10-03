#if !defined(PETSCIMTYPES_H)
#define PETSCIMTYPES_H

typedef struct _p_IM *IM;

typedef enum {
  IM_INVALID = -1,
  IM_INTERVAL = 0,
  IM_ARRAY = 1,
  IM_STATE_MAX = 2
} IMState;

typedef enum {
  IM_MIN_MODE = -1,
  IM_LOCAL = 0,
  IM_GLOBAL = 1,
  IM_MAX_MODE = 2
} IMOpMode;
#endif /* PETSCIMTYPES_H */
