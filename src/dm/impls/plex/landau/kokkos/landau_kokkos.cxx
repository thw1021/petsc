/*
   Implements the Kokkos kernel
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/


//#include <Kokkos_Core.hpp>
//#include <cstdio>

PetscErrorCode LandKokkosJacobian_xxx( DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[],
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
  const PetscReal   *invJ;
  PetscReal         *iTab,*Tables;
  PetscScalar       *elemMats, *elMat;
  const PetscInt    nip = numCells*Nq;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
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
  flops = (PetscLogDouble)numCells*Nq*(5*dim*dim*Nf*Nf + 165);
  ierr = PetscMalloc2(Nf*Nq*Nb*(1+dim), &Tables, numCells*totDim*totDim, &elemMats);CHKERRQ(ierr);
  for (fieldA=0,iTab=Tables;fieldA<Nf;fieldA++,iTab += Nq*Nb*(1+dim)) {
    ierr = PetscMemcpy(iTab,         Tf[fieldA]->T[0], Nq*Nb*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemcpy(&iTab[Nq*Nb], Tf[fieldA]->T[1], Nq*Nb*dim*sizeof(PetscReal));CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
  // make elems




  ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  // ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
  // cleanup
  // ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
  /* assembly */
  ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
  for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
    ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
    if (ej==-1) {
      int d,f;
      printf("Kokkos Element matrix\n");
      for (d = 0; d < totDim; ++d){
        for (f = 0; f < totDim; ++f) printf(" %17.10e",  PetscRealPart(elMat[d*totDim + f]));
        printf("\n");
      }
      exit(12);
    }
  }
  ierr = PetscFree2(Tables,elemMats);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
