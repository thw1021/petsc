#if !defined(PETSCDMBF_XD_H)
#define PETSCDMBF_XD_H

#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/

#if defined(PETSC_HAVE_P4EST)
#include "../p4est/petsc_p4est_package.h"

#if !defined(P4_TO_P8)
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_ghost.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#include <p4est_mesh.h>
#include <p4est_search.h>
#else
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#include <p8est_mesh.h>
#include <p8est_search.h>
#endif /* !defined(P4_TO_P8) */

#endif /* defined(PETSC_HAVE_P4EST) */

#endif /* defined(PETSCDMBF_XD_H) */
