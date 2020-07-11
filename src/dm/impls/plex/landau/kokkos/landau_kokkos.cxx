#include <petscdmplex.h>

//
// First Kokkos::View (multidimensional array) example:
//   1. Start up Kokkos
//   2. Allocate a Kokkos::View
//   3. Execute a parallel_for and a parallel_reduce over that View's data
//   4. Shut down Kokkos
//
// Compare this example to 03_simple_view, which uses functors to
// define the loop bodies of the parallel_for and parallel_reduce.
//

//#include <Kokkos_Core.hpp>
//#include <cstdio>

PetscErrorCode LandauKokkosJacobian( DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[],
                                     const PetscReal invMass[], const PetscReal Eq_m[], const PetscReal * const IPDataGlobal,
                                     const PetscReal wiGlobal[], const PetscReal invJj[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain, 
                                     Mat JacP)
{
  PetscErrorCode    ierr;
  /* PetscInt          ii,ej,*Nbf,Nb,nip_dim2,cStart,cEnd,Nf,dim,numGCells,totDim,nip,szf=sizeof(PetscReal); */
  /* PetscReal         *d_TabBD,*d_invJj,*d_wiGlobal,*d_nu_alpha,*d_nu_beta,*d_invMass,*d_Eq_m; */
  /* PetscScalar       *elemMats,*d_elemMats,  *iTab; */
  /* PetscLogDouble    flops; */
  /* PetscTabulation   *Tf; */
  /* PetscDS           prob; */
  /* PetscSection      section, globalSection; */
  /* PetscReal        *d_IPDataGlobal; */
  /* PetscContainer    container = NULL; */
  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
#endif



#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr); // remove in real application
#endif
  PetscFunctionReturn(0);
}
