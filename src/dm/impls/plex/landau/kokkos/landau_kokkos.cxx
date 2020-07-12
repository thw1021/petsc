/*
   Implements the Kokkos kernel
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#if defined(PETSC_HAVE_OPENMP)
#include <omp.h>
#endif
#define PETSC_THREAD_SYNC
#define PETSC_DEVICE_FUNC_DECL static
#define PETSC_DEVICE_DATA_DECL static
#include "../land_kernel.h"

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

PetscErrorCode LandKokkosJacobian( DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[],
                                   const PetscReal invMass[], const PetscReal Eq_m[], const PetscReal * const IPDataGlobal,
                                   const PetscReal wiGlobal[], const PetscReal invJ_a[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain,
                                   Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          *Nbf,Nb,cStart,cEnd,Nf,dim,numCells,totDim,fieldA,ej;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscLogDouble    flops;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=LAND_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "LAND_DIM != dim");
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  flops = (PetscLogDouble)numCells*(PetscLogDouble)Nq*(PetscLogDouble)(5*dim*dim*Nf*Nf + 165);
  // create data
  {
    const PetscReal *invJ;
    PetscReal       *iTab,*Tables;
    PetscScalar     *elemMat;
    const PetscInt  nip = numCells*Nq;
    ierr = PetscMalloc2(Nf*Nq*Nb*(1+dim), &Tables, totDim*totDim, &elemMat);CHKERRQ(ierr);
    for (fieldA=0,iTab=Tables;fieldA<Nf;fieldA++,iTab += Nq*Nb*(1+dim)) {
      ierr = PetscMemcpy(iTab,         Tf[fieldA]->T[0], Nq*Nb*sizeof(PetscReal));CHKERRQ(ierr);
      ierr = PetscMemcpy(&iTab[Nq*Nb], Tf[fieldA]->T[1], Nq*Nb*dim*sizeof(PetscReal));CHKERRQ(ierr);
    }
    for (ej = cStart, invJ = invJ_a; ej < cEnd; ++ej, invJ += Nq*dim*dim) {
      PetscInt     qj,d,f;
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      ierr = PetscMemzero(elemMat, totDim *totDim * sizeof(PetscScalar));CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      for (qj = 0; qj < Nq; ++qj) {
        PetscReal       g2[1][LAND_MAX_SUB_THREAD_BLOCKS][LAND_MAX_SPECIES][LAND_DIM], g3[1][LAND_MAX_SUB_THREAD_BLOCKS][LAND_MAX_SPECIES][LAND_DIM][LAND_DIM];
        const PetscInt  jpidx = Nq*(ej-cStart) + qj, one = 1, zero = 0; /* length of inner global interation, outer integration point */
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogFlops(flops);CHKERRQ(ierr);
#endif
        landau_inner_integral(zero, one, zero, one, zero, nip, 1, jpidx, Nf, dim, IPDataGlobal, wiGlobal, &invJ[qj*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, quarter3DDomain, Nq, Nb, qj, qj+1, Tables, elemMat, g2, g3);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
#endif
      } /* qj loop */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
      /* assemble matrix */
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elemMat, ADD_VALUES);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
      if (ej==-1) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "CPU Element matrix\n");CHKERRQ(ierr);
        for (d = 0; d < totDim; ++d){
          for (f = 0; f < totDim; ++f) {
            int i = d, j = f;
            ierr = PetscPrintf(PETSC_COMM_SELF, " %19.12e", PetscRealPart(elemMat[i*totDim + j]));CHKERRQ(ierr);
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
        exit(13);
      }
    } /* ej cells loop, not cuda */
    ierr = PetscFree2(Tables,elemMat);CHKERRQ(ierr);
  }

#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr); // remove in real application
#endif
  PetscFunctionReturn(0);
}
