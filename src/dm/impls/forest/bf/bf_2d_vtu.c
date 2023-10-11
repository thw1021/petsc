#include "bf_2d_vtu.h"

#if defined(PETSC_HAVE_P4EST)

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_VTKWriteAll DMBF_2D_VTKWriteAll

/* include generic functions */
#include "bf_xd_vtu.h"

#endif /* defined(PETSC_HAVE_P4EST) */
