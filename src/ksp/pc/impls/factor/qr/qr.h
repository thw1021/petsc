/*
   Private data structure for QR preconditioner.
*/
#if !defined(__QR_H)
#define __QR_H

#include <../src/ksp/pc/impls/factor/factor.h>

typedef struct {
  PC_Factor hdr;
  IS        col;            /* index sets used for reordering */
} PC_QR;

#endif
