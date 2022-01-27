#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"
/* petsclog.h */
/* Fortran interface file */

/*
* This file was generated automatically by bfort from the C source
* file.  
 */

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" { 
#endif 
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
} 
#endif 

#else

#define PetscToPointer(a) (*(PetscFortranAddr *)(a))
#define PetscFromPointer(a) (PetscFortranAddr)(a)
#define PetscRmPointer(a)
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petsclogflops_ PETSCLOGFLOPS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petsclogflops_ petsclogflops
#endif
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscloggpuflops_ PETSCLOGGPUFLOPS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscloggpuflops_ petscloggpuflops
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif

PETSC_STATIC_INLINE PetscErrorCode  petsclogflops_(PetscLogDouble *n, int *__ierr)
{
*__ierr = PetscLogFlops(*n);
}
PETSC_STATIC_INLINE PetscErrorCode  petscloggpuflops_(PetscLogDouble *n, int *__ierr)
{
*__ierr = PetscLogGpuFlops(*n);
}
#if defined(__cplusplus)
}
#endif
