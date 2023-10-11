#include "bf_3d_vtu.h"

#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h> /* convert to p8est for 3D domains */

/* rename generic functions that are the same for 2D and 3D */
#define DMBF_XD_VTKWriteAll DMBF_3D_VTKWriteAll

/* include generic functions */
#include "bf_xd_vtu.h"

#endif /* defined(PETSC_HAVE_P4EST) */
