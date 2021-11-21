#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define veccreatempiwitharray_        VECCREATEMPIWITHARRAY
#define veccreateghostblockwitharray_ VECCREATEGHOSTBLOCKWITHARRAY
#define veccreateghostwitharray_      VECCREATEGHOSTWITHARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define veccreatempiwitharray_        veccreatempiwitharray
#define veccreateghostblockwitharray_ veccreateghostblockwitharray
#define veccreateghostwitharray_      veccreateghostwitharray
#endif

PETSC_EXTERN void veccreatempiwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscScalar *s,Vec *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateMPIWithArray(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*bs,*n,*N,s,V);
}

PETSC_EXTERN void veccreateghostblockwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,
                                                PetscScalar *array,PetscInt *nextra,Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostBlockWithArray(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*bs,*n,*N,*nghost,ghosts,*nextra,array,vv);
}

PETSC_EXTERN void veccreateghostwitharray_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,PetscInt *nextra,
                                           PetscScalar *array,Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostWithArray(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*n,*N,*nghost,ghosts,*nextra,array,vv);
}
