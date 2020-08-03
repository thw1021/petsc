/*
   Implements the Kokkos kernel
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/

#define PETSC_SKIP_CXX_COMPLEX_FIX
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petscts.h>
#include <Kokkos_Core.hpp>
#include <cstdio>
typedef Kokkos::TeamPolicy<>::member_type team_member;
#define PETSC_DEVICE_FUNC_DECL KOKKOS_INLINE_FUNCTION
#include "../land_tensors.h"

namespace landau_inner_red {  // namespace helps with name resolution in reduction identity
  template< class ScalarType, int Nf >
  struct array_type {
    ScalarType gg2[Nf][LAND_DIM];
    ScalarType gg3[Nf][LAND_DIM][LAND_DIM];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
    array_type() {
      for (int i = 0; i < Nf; i++ ){
        for (int j = 0; j < LAND_DIM; j++ ){
          gg2[i][j] = 0;
          for (int k = 0; k < LAND_DIM; k++ ){
            gg3[i][j][k] = 0;
          }
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) {
      for (int i = 0; i < Nf; i++ ){
        for (int j = 0; j < LAND_DIM; j++ ){
          gg2[i][j] = rhs.gg2[i][j];
          for (int k = 0; k < LAND_DIM; k++ ){
            gg3[i][j][k] = rhs.gg3[i][j][k];
          }
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src) {
      for (int i = 0; i < Nf; i++ ){
        for (int j = 0; j < LAND_DIM; j++ ){
          gg2[i][j] += src.gg2[i][j];
          for (int k = 0; k < LAND_DIM; k++ ){
            gg3[i][j][k] += src.gg3[i][j][k];
          }
        }
      }
      return *this;
    }
    KOKKOS_INLINE_FUNCTION   // volatile add operator
    void operator += (const volatile array_type& src) volatile {
      for (int i = 0; i < Nf; i++ ){
        for (int j = 0; j < LAND_DIM; j++ ){
          gg2[i][j] += src.gg2[i][j];
          for (int k = 0; k < LAND_DIM; k++ ){
            gg3[i][j][k] += src.gg3[i][j][k];
          }
        }
      }
    }
  };
  typedef array_type<PetscReal,LAND_MAX_SPECIES> ValueType;  // used to simplify code below
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
  template<>
  struct reduction_identity< landau_inner_red::ValueType > {
    KOKKOS_FORCEINLINE_FUNCTION static landau_inner_red::ValueType sum() {
      return landau_inner_red::ValueType();
    }
  };
}

extern "C"  {
PetscErrorCode LandKokkosJacobian( DM plex, const PetscInt Nq, PetscReal nu_alpha[], PetscReal nu_beta[],
                                   PetscReal invMass[], PetscReal Eq_m[], PetscReal * const IPDataGlobal,
                                   PetscReal wiGlobal[], PetscReal invJ[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain,
                                   Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          *Nbf,Nb,cStart,cEnd,Nf,dim,numCells,totDim,ej,nip;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscLogDouble    flops;
  PetscReal         *BB,*DD;
  LandCtx           *ctx;
  PetscFunctionBegin;
  ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=LAND_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "LAND_DIM != dim");
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  nip = numCells*Nq;
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  BB   = Tf[0]->T[0]; DD = Tf[0]->T[1];
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  flops = (PetscLogDouble)numCells*Nq*(5*dim*dim*Nf*Nf + 165);
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
#else
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "no KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA");
#endif
  {
    Kokkos::View<PetscScalar**> d_elem_mats( "element matrices", numCells, totDim*totDim);
    Kokkos::View<PetscScalar**> h_elem_mats = Kokkos::create_mirror_view(d_elem_mats);
    {
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_alpha (nu_alpha, Nf);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_alpha ("nu_alpha", Nf);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_beta (nu_beta, Nf);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_beta ("nu_beta", Nf);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invMass (invMass,Nf);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_invMass ("invMass", Nf);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_Eq_m (Eq_m,Nf);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_Eq_m ("Eq_m", Nf);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_BB (BB,Nq*Nb);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_BB ("BB", Nq*Nb);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_DD (DD,Nq*Nb*dim);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_DD ("DD", Nq*Nb*dim);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_wiGlobal (wiGlobal,Nq*numCells);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_wiGlobal ("wiGlobal", Nq*numCells);
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_ipdata (IPDataGlobal,nip*(dim + Nf*(dim+1)));
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_ipdata ("ipdata", nip*(dim + Nf*(dim+1)));
      const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invJ (invJ,Nq*numCells*dim*dim);
      Kokkos::View<PetscReal*, Kokkos::LayoutLeft> d_invJ ("invJ", Nq*numCells*dim*dim);
      ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
      Kokkos::deep_copy (d_alpha, h_alpha);
      Kokkos::deep_copy (d_beta, h_beta);
      Kokkos::deep_copy (d_invMass, h_invMass);
      Kokkos::deep_copy (d_Eq_m, h_Eq_m);
      Kokkos::deep_copy (d_BB, h_BB);
      Kokkos::deep_copy (d_DD, h_DD);
      Kokkos::deep_copy (d_wiGlobal, h_wiGlobal);
      Kokkos::deep_copy (d_ipdata, h_ipdata);
      Kokkos::deep_copy (d_invJ, h_invJ);
      ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_VIENNACL) // add Kokkos-GPU ???
      ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
      if (ctx->deviceType == LAND_CPU) PetscInfo(plex, "Warning: Landau selected CPU but no support for Kokkos using GPU\n");
#else
      ierr = PetscLogFlops(flops*nip);CHKERRQ(ierr);
#endif
      {
        int team_size = Kokkos::DefaultExecutionSpace().concurrency() > Nq ? Nq : 1;
        parallel_for("landau_e", Kokkos::TeamPolicy<>( (int)numCells, team_size, Nf*Nb > Nq*nip ? Nf*Nb : Nq*nip ).set_scratch_size(0, Kokkos::PerTeam(Nq*Nf*dim*(1+dim)*sizeof(PetscReal))), KOKKOS_LAMBDA (const team_member team_0) {
            const PetscInt  myelem = team_0.league_rank();
            PetscReal       *g2a = (PetscReal*) team_0.team_shmem().get_shmem(Nq*Nf*dim*sizeof(PetscReal));
            PetscReal       *g3a = (PetscReal*) team_0.team_shmem().get_shmem(Nq*Nf*dim*dim*sizeof(PetscReal));
            PetscReal       (*g2)[/*LAND_MAX_SPECIES*/][LAND_MAX_NQ][LAND_DIM] =           (PetscReal (*)[/*LAND_MAX_SPECIES*/][LAND_MAX_NQ][LAND_DIM])           g2a;
            PetscReal       (*g3)[/*LAND_MAX_SPECIES*/][LAND_MAX_NQ][LAND_DIM][LAND_DIM] = (PetscReal (*)[/*LAND_MAX_SPECIES*/][LAND_MAX_NQ][LAND_DIM][LAND_DIM]) g3a;
            PetscScalar     *elMat = &d_elem_mats(myelem,0);
            // get g2[] & g3[]
            parallel_for( TeamThreadRange(team_0,0,Nq), [&] (int ip) {
                using Kokkos::parallel_reduce;
                const PetscInt              myQi = ip;
                const PetscInt              jpidx = myQi + myelem * Nq;
                PetscReal                   *invJj = &d_invJ(jpidx*dim*dim);
                const PetscInt              ipdata_sz = (dim + Nf*(1+dim));
                PetscInt                    dp,d3,d,d2;
                const LandPointData * const fplpt_j = (LandPointData*)(IPDataGlobal + jpidx*ipdata_sz);
                const PetscReal * const     vj = fplpt_j->crd, wj = wiGlobal[jpidx];
                // reduce on g22 and g33 for IP jpidx
                landau_inner_red::ValueType gg;
                parallel_reduce( ThreadVectorRange (team_0, nip), KOKKOS_LAMBDA ( const int& ipidx, landau_inner_red::ValueType & ggg) {
                    const LandPointData * const fplpt = (LandPointData*)(IPDataGlobal + ipidx*ipdata_sz);
                    const LandFDF * const       fdf = &fplpt->fdf[0];
                    const PetscReal             wi = wiGlobal[ipidx];
                    PetscInt                    fieldA,fieldB,d2,d3;
#if LAND_DIM==2
                    PetscReal                   Ud[2][2], Uk[2][2];
                    LandauTensor2D(vj, fplpt->crd[0], fplpt->crd[1], Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
                    for (fieldA = 0; fieldA < Nf; ++fieldA) {
                      for (fieldB = 0; fieldB < Nf; ++fieldB) {
                        for (d2 = 0; d2 < 2; ++d2) {
                          for (d3 = 0; d3 < 2; ++d3) {
                            /* K = U * grad(f): g2=e: i,A */
                            ggg.gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * Uk[d2][d3] * fdf[fieldB].df[d3] * wi;
                            /* D = -U * (I \kron (fx)): g3=f: i,j,A */
                            ggg.gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * Ud[d2][d3] * fdf[fieldB].f * wi;
                          }
                        }
                      }
                    }
#else
                    PetscReal                   U[3][3];
                    LandauTensor3D(vj, fplpt->crd[0], fplpt->crd[1], fplpt->crd[2], U, (ipidx==jpidx) ? 0. : 1.);
                    for (fieldA = 0; fieldA < Nf; ++fieldA) {
                      for (fieldB = 0; fieldB < Nf; ++fieldB) {
                        for (d2 = 0; d2 < 3; ++d2) {
                          for (d3 = 0; d3 < 3; ++d3) {
                            /* K = U * grad(f): g2 = e: i,A */
                            ggg.gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * U[d2][d3] * fplpt->fdf[fieldB].df[d3] * wi;
                            /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                            ggg.gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * fplpt->fdf[fieldB].f * wi;
                          }
                        }
                      }
                    }
#endif
                  }, Kokkos::Sum<landau_inner_red::ValueType>(gg) );
                /* add electric field term once per IP */
                {
                  int f;
                  for (f = 0; f < Nf; ++f) gg.gg2[f][dim-1] += Eq_m[f];
                }
                /* Jacobian transform - g2, g3 - per thread (2D) */
                parallel_for( ThreadVectorRange(team_0,0,Nf), [&] (int fieldA) {
                    for (d = 0; d < dim; ++d) {
                      (*g2)[fieldA][myQi][d] = 0.0;
                      for (d2 = 0; d2 < dim; ++d2) {
                        (*g2)[fieldA][myQi][d] += invJj[d*dim+d2]*gg.gg2[fieldA][d2];
                        //printf("\t:g2[%d][%d][%d]=%g\n",fieldA,myQi,d,(*g2)[fieldA][myQi][d]);
                        (*g3)[fieldA][myQi][d][d2] = 0.0;
                        for (d3 = 0; d3 < dim; ++d3) {
                          for (dp = 0; dp < dim; ++dp) {
                            (*g3)[fieldA][myQi][d][d2] += invJj[d*dim + d3]*gg.gg3[fieldA][d3][dp]*invJj[d2*dim + dp];
                            //printf("\t\t\t: g3=%g\n",(*g3)[fieldA][myQi][d][d2]);
                          }
                        }
                        (*g3)[fieldA][myQi][d][d2] *= wj;
                      }
                      (*g2)[fieldA][myQi][d] *= wj;
                    }
                  });
              });
            /* assemble - on the diagonal (I,I) */
            parallel_for(Kokkos::TeamThreadRange(team_0,0,Nf), [&] (int fieldA) {
                parallel_for(Kokkos::ThreadVectorRange(team_0,0,Nb), [&] (int blk_i) {
                    int blk_j,qj,d,d2;
                    const PetscInt i = fieldA*Nb + blk_i; /* Element matrix row */
                    for (blk_j = 0; blk_j < Nb; ++blk_j) {
                      const PetscInt j    = fieldA*Nb + blk_j; /* Element matrix column */
                      const PetscInt fOff = i*totDim + j;
                      for ( qj = 0 ; qj < Nq ; qj++) { // look at others integration points
                        const PetscReal *BJq = &BB[qj*Nb], *DIq = &DD[qj*Nb*dim];
                        for (d = 0; d < dim; ++d) {
                          elMat[fOff] += DIq[blk_i*dim+d]*(*g2)[fieldA][qj][d]*BJq[blk_j];
                          //printf("\t\t: mat[%d]=%g D[%d]=%g g2[%d][%d][%d]=%g B=%g\n",fOff,elMat[fOff],blk_i*dim+d,DIq[blk_i*dim+d],fieldA,qj,d,(*g2)[fieldA][qj][d],BJq[blk_j]);
                          for (d2 = 0; d2 < dim; ++d2) {
                            elMat[fOff] += DIq[blk_i*dim + d]*(*g3)[fieldA][qj][d][d2]*DIq[blk_j*dim + d2];
                          }
                        }
                      }
                    }
                  });
              });
            if (myelem==-1) {
              int d,f;
              printf("Kokkos Element matrix\n");
              for (d = 0; d < totDim; ++d){
                for (f = 0; f < totDim; ++f) printf(" %17.10e",  PetscRealPart(elMat[d*totDim + f]));
                printf("\n");
              }
              //exit(12);
            }
          });
      }
      ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
    }
    Kokkos::deep_copy (h_elem_mats, d_elem_mats);
    ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
    for (ej = cStart ; ej < cEnd; ++ej) {
      const PetscScalar *elMat = &h_elem_mats(ej-cStart,0);
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
}
