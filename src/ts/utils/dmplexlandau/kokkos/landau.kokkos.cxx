/*
  Implements the Kokkos kernel
*/
#include <petscconf.h>
#include <petscvec_kokkos.hpp>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petsclandau.h>
#include <petscts.h>

#include <Kokkos_Core.hpp>
#include <cstdio>
typedef Kokkos::TeamPolicy<>::member_type team_member;
#include "../land_tensors.h"
#include <petscaijdevice.h>

namespace landau_inner_red {  // namespace helps with name resolution in reduction identity
  template< class ScalarType >
  struct array_type {
    ScalarType gg2[LANDAU_DIM];
    ScalarType gg3[LANDAU_DIM][LANDAU_DIM];

    KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
    array_type() {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] = 0;
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] = 0;
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // Copy Constructor
    array_type(const array_type & rhs) {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] = rhs.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] = rhs.gg3[j][k];
        }
      }
    }
    KOKKOS_INLINE_FUNCTION   // add operator
    array_type& operator += (const array_type& src)
    {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] += src.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] += src.gg3[j][k];
        }
      }
      return *this;
    }
    KOKKOS_INLINE_FUNCTION   // volatile add operator
    void operator += (const volatile array_type& src) volatile
    {
      for (int j = 0; j < LANDAU_DIM; j++) {
        gg2[j] += src.gg2[j];
        for (int k = 0; k < LANDAU_DIM; k++) {
          gg3[j][k] += src.gg3[j][k];
        }
      }
    }
  };
  typedef array_type<PetscReal> TensorValueType;  // used to simplify code below
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
  template<>
  struct reduction_identity< landau_inner_red::TensorValueType > {
    KOKKOS_FORCEINLINE_FUNCTION static landau_inner_red::TensorValueType sum() {
      return landau_inner_red::TensorValueType();
    }
  };
}

extern "C"  {
PetscErrorCode LandauKokkosCreateMatMaps(P4estVertexMaps maps[], pointInterpolationP4est (*pointMaps)[LANDAU_MAX_Q_FACE], PetscInt Nf[], PetscInt Nq, PetscInt num_grids)
{
  PetscFunctionBegin;
  for (PetscInt grid=0;grid<num_grids;grid++) {
    P4estVertexMaps   h_maps;  /* host container */
    const Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >    h_points ((pointInterpolationP4est*)pointMaps, maps[grid].num_reduced);
    const Kokkos::View< LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_gidx ((LandauIdx*)maps[grid].gIdx, maps[grid].num_elements);
    Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>   *d_points = new Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>("points", maps[grid].num_reduced);
    Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight> *d_gidx = new Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight>("gIdx", maps[grid].num_elements);

    Kokkos::deep_copy (*d_gidx, h_gidx);
    Kokkos::deep_copy (*d_points, h_points);
    h_maps.num_elements = maps[grid].num_elements;
    h_maps.num_face = maps[grid].num_face;
    h_maps.num_reduced = maps[grid].num_reduced;
    h_maps.deviceType = maps[grid].deviceType;
    h_maps.numgrids = maps[grid].numgrids;
    h_maps.Nf = Nf[grid];
    h_maps.Nq = Nq;
    h_maps.c_maps = (pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE]) d_points->data();
    maps[grid].vp1 = (void*)d_points;
    h_maps.gIdx = (LandauIdx (*)[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ]) d_gidx->data();
    maps[grid].vp2 = (void*)d_gidx;
    {
      Kokkos::View<P4estVertexMaps, Kokkos::HostSpace> h_maps_k(&h_maps);
      Kokkos::View<P4estVertexMaps>                    *d_maps_k = new Kokkos::View<P4estVertexMaps>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(),h_maps_k));
      Kokkos::deep_copy (*d_maps_k, h_maps_k);
      maps[grid].data = d_maps_k->data();
      maps[grid].vp3 = (void*)d_maps_k;
    }
  }
  PetscFunctionReturn(0);
}
PetscErrorCode LandauKokkosDestroyMatMaps(P4estVertexMaps maps[], PetscInt num_grids)
{
  PetscFunctionBegin;
  for (PetscInt grid=0;grid<num_grids;grid++) {
    Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>  *a = (Kokkos::View<pointInterpolationP4est*[LANDAU_MAX_Q_FACE], Kokkos::LayoutRight>*)maps[grid].vp1;
    Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight>*b = (Kokkos::View<LandauIdx*[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ], Kokkos::LayoutRight>*)maps[grid].vp2;
    Kokkos::View<P4estVertexMaps*>                                                  *c = (Kokkos::View<P4estVertexMaps*>*)maps[grid].vp3;
    delete a;  delete b;  delete c;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode LandauKokkosStaticDataSet(DM plex, const PetscInt Nq, const PetscInt Nf, const PetscInt nip, PetscReal nu_alpha[], PetscReal nu_beta[], PetscReal a_invMass[], PetscReal a_invJ[], PetscReal a_mass_w[],
                                         PetscReal a_x[], PetscReal a_y[], PetscReal a_z[], PetscReal a_w[], LandauGeomData *SData_d)
{
  PetscReal       *BB,*DD;
  PetscErrorCode  ierr;
  PetscTabulation *Tf;
  PetscInt        dim,Nb=Nq;
  PetscDS         prob;

  PetscFunctionBegin;
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  if (LANDAU_DIM != dim) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  BB   = Tf[0]->T[0]; DD = Tf[0]->T[1];
  ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
  {
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_alpha (nu_alpha, Nf);
    auto alpha = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("alpha", Nf);
    SData_d->alpha = static_cast<void*>(alpha);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_beta (nu_beta, Nf);
    auto beta = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("beta", Nf);
    SData_d->beta = static_cast<void*>(beta);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invMass (a_invMass,Nf);
    auto invMass = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("invMass", Nf);
    SData_d->invMass = static_cast<void*>(invMass);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_BB (BB,Nq*Nb);
    auto B = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("B", Nq*Nb);
    SData_d->B = static_cast<void*>(B);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_DD (DD,Nq*Nb*dim);
    auto D = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("D", Nq*Nb*dim);
    SData_d->D = static_cast<void*>(D);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_mass_w (a_mass_w, nip);
    auto mass_w = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("mass_w", nip);
    SData_d->mass_w = static_cast<void*>(mass_w);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_invJ (a_invJ, nip*dim*dim);
    auto invJ = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("invJ", nip*dim*dim);
    SData_d->invJ = static_cast<void*>(invJ);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_x (a_x, nip);
    auto x = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("x", nip);
    SData_d->x = static_cast<void*>(x);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_y (a_y, nip);
    auto y = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("y", nip);
    SData_d->y = static_cast<void*>(y);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_w (a_w, nip);
    auto w = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("w", nip);
    SData_d->w = static_cast<void*>(w);
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_z (a_z , dim==3 ? nip : 0);
    auto z = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("z", dim==3 ? nip : 0);
    SData_d->z = static_cast<void*>(z);
    Kokkos::deep_copy (*mass_w, h_mass_w);
    Kokkos::deep_copy (*alpha, h_alpha);
    Kokkos::deep_copy (*beta, h_beta);
    Kokkos::deep_copy (*invMass, h_invMass);
    Kokkos::deep_copy (*B, h_BB);
    Kokkos::deep_copy (*D, h_DD);
    Kokkos::deep_copy (*invJ, h_invJ);
    Kokkos::deep_copy (*x, h_x);
    Kokkos::deep_copy (*y, h_y);
    Kokkos::deep_copy (*z, h_z);
    Kokkos::deep_copy (*w, h_w);
    auto Eq_m = new Kokkos::View<PetscReal*, Kokkos::LayoutLeft> ("Eq_m",Nf); // allocate but do not set
    SData_d->Eq_m = static_cast<void*>(Eq_m);
  }
  PetscFunctionReturn(0);
}

  PetscErrorCode LandauKokkosStaticDataClear(LandauGeomData *SData_d)
  {
    PetscFunctionBegin;
    if (SData_d->alpha) {
      auto alpha = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->alpha);
      delete alpha;
      auto beta = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->beta);
      delete beta;
      auto invMass = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invMass);
      delete invMass;
      auto B = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->B);
      delete B;
      auto D = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->D);
      delete D;
      auto mass_w = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->mass_w);
      delete mass_w;
      auto invJ = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invJ);
      delete invJ;
      auto x = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->x);
      delete x;
      auto y = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->y);
      delete y;
      auto z = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->z);
      delete z;
      auto w = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->w);
      delete w;
      auto Eq_m = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->Eq_m);
      delete Eq_m;
    }
    PetscFunctionReturn(0);
  }

PetscErrorCode LandauKokkosJacobian(DM plex[], const PetscInt Nq, const PetscInt num_grids, const PetscInt numCells[], const PetscInt Nf_g[], PetscReal a_Eq_m[], PetscScalar a_IPf[],
                                    const PetscInt N, const PetscScalar a_xarray[], const LandauGeomData *SData_d, const PetscInt num_sub_blocks, const PetscReal shift,
                                    const PetscLogEvent events[], const PetscInt grid_mat_offsets[], const PetscInt species_grid_offset[], Mat subJ[], Mat JacP)
{
    using scr_mem_t = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using fieldMats_scr_t = Kokkos::View<PetscScalar**, Kokkos::LayoutRight, scr_mem_t>;
    using idx_scr_t = Kokkos::View<PetscInt**, Kokkos::LayoutRight, scr_mem_t>;
    using scale_scr_t = Kokkos::View<PetscReal**, Kokkos::LayoutRight, scr_mem_t>;
    using g2_scr_t = Kokkos::View<PetscReal***, Kokkos::LayoutRight, scr_mem_t>;
    using g3_scr_t = Kokkos::View<PetscReal****, Kokkos::LayoutRight, scr_mem_t>;
    PetscErrorCode    ierr;
    PetscInt          Nb=Nq,dim,global_elem_mat_sz,nfaces=0,nip_global,ip_offset[LANDAU_MAX_GRIDS];
    LandauCtx         *ctx;
    PetscReal         *d_Eq_m=NULL;
    PetscScalar       *d_IPf=NULL;
    P4estVertexMaps   *d_maps[LANDAU_MAX_GRIDS];
    PetscSplitCSRDataStructure d_mat;
    PetscContainer    container = NULL;
    const int         conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp==0) ? Nq : 1;
    auto              d_alpha_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->alpha); //static data
    const PetscReal   *d_alpha = d_alpha_k->data();
    const PetscInt    Nftot = d_alpha_k->size(); // total number of species
    auto              d_beta_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->beta);
    const PetscReal   *d_beta = d_beta_k->data();
    auto              d_invMass_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invMass);
    const PetscReal   *d_invMass = d_invMass_k->data();
    auto              d_B_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->B);
    const PetscReal   *d_BB = d_B_k->data();
    auto              d_D_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->D);
    const PetscReal   *d_DD = d_D_k->data();
    auto              d_mass_w_k = *static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->mass_w);
    auto              d_invJ_k = *static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->invJ); // use Kokkos vector in kernels
    auto              d_x_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->x); //static data
    const PetscReal   *d_x = d_x_k->data();
    auto              d_y_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->y); //static data
    const PetscReal   *d_y = d_y_k->data();
    auto              d_z_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->z); //static data
    const PetscReal   *d_z = d_z_k->data();
    auto              d_w_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->w); //static data
    const PetscReal   *d_w = d_w_k->data();
    auto              d_Eq_m_k = static_cast<Kokkos::View<PetscReal*, Kokkos::LayoutLeft>*>(SData_d->Eq_m); // static storage, dynamci data - E(t)

    PetscFunctionBegin;
    ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(plex[0], &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
    ierr = DMGetDimension(plex[0], &dim);CHKERRQ(ierr);                                        PetscPrintf(ctx->comm," LandauKokkosJacobian start Nftot=%D\n",Nftot);
    if (LANDAU_DIM != dim) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "dim %D != LANDAU_DIM %d",dim,LANDAU_DIM);
    if (ctx->gpu_assembly) {
      ierr = PetscObjectQuery((PetscObject) JacP, "assembly_maps", (PetscObject *) &container);CHKERRQ(ierr);
      if (container) { // not here first call
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
        P4estVertexMaps   *h_maps=NULL;
        ierr = PetscContainerGetPointer(container, (void **) &h_maps);CHKERRQ(ierr);
        for (PetscInt grid=0 ; grid<num_grids ; grid++) {
          if (h_maps[grid].data) {
            d_maps[grid] = h_maps[grid].data;
            nfaces = h_maps[grid].num_face;
          } else {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly but no metadata in container");
          }
        }
        // this does the setup the first time called
        ierr = MatKokkosGetDeviceMatWrite(JacP,&d_mat);CHKERRQ(ierr);
#else
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "GPU assembly w/o kokkos kernels -- should not be here");
#endif
      } else { // kernel output - first call assembled on device
        for (PetscInt grid=0 ; grid<num_grids ; grid++) d_maps[grid] = NULL;
      }
    } else {
      for (PetscInt grid=0 ; grid<num_grids ; grid++) d_maps[grid] = NULL;
    }
    const Kokkos::View<PetscReal*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >  h_Eq_m_k (a_Eq_m, (a_IPf || a_xarray) ? Nftot : 0);
    Kokkos::deep_copy (*d_Eq_m_k, h_Eq_m_k);
    d_Eq_m = d_Eq_m_k->data();
    ierr = PetscKokkosInitializeCheck();CHKERRQ(ierr);
    ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);

    ip_offset[0] = 0;
    nip_global = 0;
    for (PetscInt grid=0 ; grid<num_grids ; grid++) {
      nip_global += numCells[grid]*Nq;
      ip_offset[grid+1] = nip_global;
    }
    // loop over all grids and assemble
    PetscInt IPfdf_idx = 0, IP_idx = 0;
    for (PetscInt grid = 0 ; grid < ctx->num_grids ; grid++) { // IPfdf_idx += nip_loc*Nfloc, IP_idx += nip_loc
      const PetscInt moffset = grid_mat_offsets[grid], nip_loc = numCells[grid]*Nq, Nfloc = species_grid_offset[grid+1] - species_grid_offset[grid], totDim = Nfloc*Nq, elemMatSize = totDim*totDim;
      const PetscInt f_off = ctx->species_grid_offset[grid];
      global_elem_mat_sz = (container!=NULL) ? 0 : numCells[grid];
      Kokkos::View<PetscScalar**, Kokkos::LayoutRight> d_elem_mats("element matrices", global_elem_mat_sz, elemMatSize);
      if (a_IPf || a_xarray) { // Jacobian, create f & df
        const int num_face = nfaces;
        const int scr_bytes = 2*(g2_scr_t::shmem_size(dim,Nfloc,Nq) + g3_scr_t::shmem_size(dim,dim,Nfloc,Nq))+fieldMats_scr_t::shmem_size(Nb,Nb)+idx_scr_t::shmem_size(Nb,num_face)+scale_scr_t::shmem_size(Nb,num_face);
#if defined(LANDAU_LAYOUT_LEFT)
        Kokkos::View<PetscReal***, Kokkos::LayoutLeft >  d_fdf_k("df", dim+1, Nfloc, nip_loc); // integration point f, df/dx
#else
        Kokkos::View<PetscReal***, Kokkos::LayoutRight > d_fdf_k("df", dim+1, Nfloc, nip_loc);
#endif
        const Kokkos::View<PetscScalar*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_IPf_k (a_IPf + IPfdf_idx, a_IPf!=NULL ? nip_loc*Nfloc : 0); // Vertex data for each element
        auto d_IPf_k = (a_IPf!=NULL) ? new Kokkos::View<PetscScalar*, Kokkos::LayoutLeft> ("IPf",nip_loc*Nfloc) : NULL; // Nq==Nb
        // copy dynamic data to device
        ierr = PetscLogEventBegin(events[1],0,0,0,0);CHKERRQ(ierr); // nothing timer
        if (a_IPf) {
          Kokkos::deep_copy (*d_IPf_k, h_IPf_k);
          d_IPf  = d_IPf_k->data();
        } else {
          d_IPf = (PetscScalar*)a_xarray; // global
        }
        ierr = PetscLogEventEnd(events[1],0,0,0,0);CHKERRQ(ierr);
#define KOKKOS_SHARED_LEVEL 1
        // get f and df
        ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        Kokkos::parallel_for("f, df", Kokkos::TeamPolicy<>(numCells[grid], team_size, /* Kokkos::AUTO */ 16), KOKKOS_LAMBDA (const team_member team) {
            const PetscInt    elem = team.league_rank();
            const PetscScalar *coef;
            PetscScalar       coef_buff[LANDAU_MAX_SPECIES*LANDAU_MAX_NQ];
            // un pack IPData
            if (!d_maps[grid]) {
              coef = &d_IPf[elem*Nb*Nfloc]; // grid local pointer
            } else {
              coef = coef_buff;
              for (int f = 0; f < Nfloc; ++f) {
                LandauIdx *const Idxs = &d_maps[grid]->gIdx[elem][f][0];
                for (int b = 0; b < Nb; ++b) {
                  PetscInt idx = Idxs[b];
                  if (idx >= 0) {
                    coef_buff[f*Nb+b] = d_IPf[idx];
                  } else {
                    idx = -idx - 1;
                    coef_buff[f*Nb+b] = 0;
                    for (int q = 0; q < d_maps[grid]->num_face; q++) {
                      PetscInt    id = d_maps[grid]->c_maps[idx][q].gid;
                      PetscScalar scale = d_maps[grid]->c_maps[idx][q].scale;
                      coef_buff[f*Nb+b] += scale*d_IPf[id];
                    }
                  }
                }
              }
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
                const PetscInt          ipidx = myQi + elem * Nq;
                const PetscReal *const  invJj = &d_invJ_k( (IP_idx+ipidx)*dim*dim );
                const PetscReal         *Bq = &d_BB[myQi*Nb], *Dq = &d_DD[myQi*Nb*dim];
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,(int)Nfloc), [=] (int f) {
                    PetscInt   b, e, d;
                    PetscReal  refSpaceDer[LANDAU_DIM];
                    printf("\t%d.%d.%d) set d_fdf_k(0,%d,%d)\n",grid,myQi,f,f,ipidx);
                    d_fdf_k(0,f,ipidx) = 0.0;
                    for (d = 0; d < LANDAU_DIM; ++d) refSpaceDer[d] = 0.0;
                    for (b = 0; b < Nb; ++b) {
                      const PetscInt    cidx = b;
                      d_fdf_k(0,f,ipidx) += Bq[cidx]*PetscRealPart(coef[f*Nb+cidx]);
                      for (d = 0; d < dim; ++d) refSpaceDer[d] += Dq[cidx*dim+d]*PetscRealPart(coef[f*Nb+cidx]);
                    }
                    for (d = 0; d < dim; ++d) {
                      for (e = 0, d_fdf_k(d+1,f,ipidx) = 0.0; e < dim; ++e) {
                        d_fdf_k(d+1,f,ipidx) += invJj[e*dim+d]*refSpaceDer[e];
                      }
                    }
                  }); // Nfloc
              }); // Nq
          }); // elems
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
        ierr = PetscLogGpuFlops(nip_loc*(PetscLogDouble)(2*Nb*(1+dim)));CHKERRQ(ierr);
#else
        ierr = PetscLogFlops(nip_loc*(PetscLogDouble)(2*Nb*(1+dim)));CHKERRQ(ierr);
#endif
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
        // Jacobian
        ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
        ierr = PetscLogGpuFlops(nip_loc*(PetscLogDouble)((nip_loc*(11*Nfloc+ 4*dim*dim) + 6*Nfloc*dim*dim*dim + 10*Nfloc*dim*dim + 4*Nfloc*dim + Nb*Nfloc*Nb*Nq*dim*dim*5)));CHKERRQ(ierr);
        if (ctx->deviceType == LANDAU_CPU) PetscInfo(plex[grid], "Warning: Landau selected CPU but no support for Kokkos using CPU\n");
#else
        ierr = PetscLogFlops(nip_loc*(PetscLogDouble)((nip_loc*(11*Nfloc+ 4*dim*dim) + 6*Nfloc*dim*dim*dim + 10*Nfloc*dim*dim + 4*Nfloc*dim + Nb*Nfloc*Nb*Nq*dim*dim*5)));CHKERRQ(ierr);
#endif
        ierr = PetscInfo5(plex[grid], "Jacobian shared memory size: %d bytes in level %d conc=%D team size=%D #face=%D\n",scr_bytes,KOKKOS_SHARED_LEVEL,conc,team_size,nfaces);CHKERRQ(ierr);
        Kokkos::parallel_for("Landau Jacobian", Kokkos::TeamPolicy<>(numCells[grid], team_size,/*Kokkos::AUTO*/16).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)),KOKKOS_LAMBDA (const team_member team) {
            const PetscInt  elem = team.league_rank();
            g2_scr_t        g2(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,Nfloc,Nq);
            g3_scr_t        g3(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,dim,Nfloc,Nq);
            g2_scr_t        gg2(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,Nfloc,Nq);
            g3_scr_t        gg3(team.team_scratch(KOKKOS_SHARED_LEVEL),dim,dim,Nfloc,Nq);
            // get g2[] & g3[]
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nq), [=] (int myQi) {
                using Kokkos::parallel_reduce;
                const PetscInt                    jpidx_g = myQi + elem * Nq, jpidx = IP_idx + jpidx_g;
                const PetscReal* const            invJj = &d_invJ_k(jpidx*dim*dim);
                const PetscReal                   vj[3] = {d_x[jpidx], d_y[jpidx], d_z ? d_z[jpidx] : 0}, wj = d_w[jpidx];
                landau_inner_red::TensorValueType gg_temp; // reduce on part of gg2 and g33 for IP jpidx_g
                Kokkos::parallel_reduce(Kokkos::ThreadVectorRange (team, (int)nip_global), [=] (const int& ipidx, landau_inner_red::TensorValueType & ggg) {
                    const PetscReal wi = d_w[ipidx], x = d_x[ipidx], y = d_y[ipidx];
                    PetscReal       temp1[3] = {0, 0, 0}, temp2 = 0;
                    PetscInt        fieldA,d2,d3,f_off_r,grid_r;
#if LANDAU_DIM==2
                    PetscReal Ud[2][2], Uk[2][2], mask = (PetscAbs(vj[0]-x) < PETSC_MACHINE_EPSILON && PetscAbs(vj[1]-y) < PETSC_MACHINE_EPSILON) ? 0. : 1.;
                    LandauTensor2D(vj, x, y, Ud, Uk, mask);
#else
                    PetscReal U[3][3], z = d_z[ipidx], mask = (PetscAbs(vj[0]-x) < PETSC_MACHINE_EPSILON && PetscAbs(vj[1]-y) < PETSC_MACHINE_EPSILON && PetscAbs(vj[2]-z) < PETSC_MACHINE_EPSILON) ? 0. : 1.;
                    LandauTensor3D(vj, x, y, z, U, mask);
#endif
                    grid_r = 0;
                    while (ipidx >= ip_offset[grid_r]) grid_r++; // yuck search for grid
                    f_off_r = species_grid_offset[grid_r];
                    printf("\t\t%d.%d) f_off_r=%d\n",grid,grid_r,f_off_r);
                    for (fieldA = 0; fieldA < Nfloc; ++fieldA) {
                      temp1[0] += d_fdf_k(1,fieldA,ipidx)*d_beta[fieldA+f_off_r]*d_invMass[fieldA+f_off_r];
                      temp1[1] += d_fdf_k(2,fieldA,ipidx)*d_beta[fieldA+f_off_r]*d_invMass[fieldA+f_off_r];
#if LANDAU_DIM==3
                      temp1[2] += d_fdf_k(3,fieldA,ipidx)*d_beta[fieldA+f_off_r]*d_invMass[fieldA+f_off_r];
#endif
                      temp2    += d_fdf_k(0,fieldA,ipidx)*d_beta[fieldA+f_off_r];
                    }
                    temp1[0] *= wi;
                    temp1[1] *= wi;
#if LANDAU_DIM==3
                    temp1[2] *= wi;
#endif
                    temp2    *= wi;
#if LANDAU_DIM==2
                    for (d2 = 0; d2 < 2; d2++) {
                      for (d3 = 0; d3 < 2; ++d3) {
                        /* K = U * grad(f): g2=e: i,A */
                        ggg.gg2[d2] += Uk[d2][d3]*temp1[d3];
                        /* D = -U * (I \kron (fx)): g3=f: i,j,A */
                        ggg.gg3[d2][d3] += Ud[d2][d3]*temp2;
                      }
                    }
#else
                    for (d2 = 0; d2 < 3; ++d2) {
                      for (d3 = 0; d3 < 3; ++d3) {
                        /* K = U * grad(f): g2 = e: i,A */
                        ggg.gg2[d2] += U[d2][d3]*temp1[d3];
                        /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                        ggg.gg3[d2][d3] += U[d2][d3]*temp2;
                      }
                    }
#endif
                  }, Kokkos::Sum<landau_inner_red::TensorValueType>(gg_temp));
                // add alpha and put in gg2/3
                Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nfloc), [&] (const int& fieldA) {
                    PetscInt d2,d3;
                    for (d2 = 0; d2 < dim; d2++) {
                      gg2(d2,fieldA,myQi) = gg_temp.gg2[d2]*d_alpha[fieldA+f_off];
                      for (d3 = 0; d3 < dim; d3++) {
                        gg3(d2,d3,fieldA,myQi) = -gg_temp.gg3[d2][d3]*d_alpha[fieldA+f_off]*d_invMass[fieldA+f_off];
                      }
                    }
                  });
                /* add electric field term once per IP */
                Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nfloc), [&] (const int& fieldA) {
                    gg2(dim-1,fieldA,myQi) += d_Eq_m[fieldA+f_off];
                  });
                Kokkos::parallel_for(Kokkos::ThreadVectorRange (team, (int)Nfloc), [=] (const int& fieldA) {
                    int d,d2,d3,dp;
                    /* Jacobian transform - g2, g3 - per thread (2D) */
                    for (d = 0; d < dim; ++d) {
                      g2(d,fieldA,myQi) = 0;
                      for (d2 = 0; d2 < dim; ++d2) {
                        g2(d,fieldA,myQi) += invJj[d*dim+d2]*gg2(d2,fieldA,myQi);
                        g3(d,d2,fieldA,myQi) = 0;
                        for (d3 = 0; d3 < dim; ++d3) {
                          for (dp = 0; dp < dim; ++dp) {
                            g3(d,d2,fieldA,myQi) += invJj[d*dim + d3]*gg3(d3,dp,fieldA,myQi)*invJj[d2*dim + dp];
                          }
                        }
                        g3(d,d2,fieldA,myQi) *= wj;
                      }
                      g2(d,fieldA,myQi) *= wj;
                    }
                  });
              }); // Nq
            team.team_barrier();
            { /* assemble */
              fieldMats_scr_t s_fieldMats(team.team_scratch(KOKKOS_SHARED_LEVEL),Nq,Nq); // Only used for GPU assembly (ie, not first pass)
              idx_scr_t       s_idx(team.team_scratch(KOKKOS_SHARED_LEVEL),Nq,num_face);
              scale_scr_t     s_scale(team.team_scratch(KOKKOS_SHARED_LEVEL),Nq,num_face);
              PetscInt        fieldA;
              for (fieldA = 0; fieldA < Nfloc; fieldA++) {
                /* assemble */
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int f) {
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,Nb), [=] (int g) {
                        PetscScalar t = 0;
                        for (int qj = 0 ; qj < Nq ; qj++) { // look at others integration points
                          const PetscReal *BJq = &d_BB[qj*Nb], *DIq = &d_DD[qj*Nb*dim];
                          for (int d = 0; d < dim; ++d) {
                            t += DIq[f*dim+d]*g2(d,fieldA,qj)*BJq[g];
                            for (int d2 = 0; d2 < dim; ++d2) {
                              t += DIq[f*dim + d]*g3(d,d2,fieldA,qj)*DIq[g*dim + d2];
                            }
                          }
                        }
                        if (global_elem_mat_sz) { // CPU assembly
                          const PetscInt fOff = (fieldA*Nb + f)*totDim + fieldA*Nb + g;
                          d_elem_mats(elem,fOff) = t;
                        } else s_fieldMats(f,g) = t;
                      });
                  });
                if (!global_elem_mat_sz) { // GPU assembly
                  const LandauIdx *const Idxs = &d_maps[grid]->gIdx[elem][fieldA][0];
                  team.team_barrier();
                  Kokkos::parallel_for(Kokkos::TeamVectorRange(team,0,Nb), [=] (int f) {
                      PetscInt q,idx = Idxs[f];
                      if (idx >= 0) {
                        s_idx(f,0) = idx + moffset;
                        s_scale(f,0) = 1.;
                      } else {
                        idx = -idx - 1;
                        for (q = 0; q < nfaces; q++) {
                          s_idx(f,q) = d_maps[grid]->c_maps[idx][q].gid + moffset;
                          s_scale(f,q) = d_maps[grid]->c_maps[idx][q].scale;
                        }
                      }
                    });
                  team.team_barrier();
                  Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int f) {
                      PetscInt nr,idx = Idxs[f];
                      if (idx >= 0) {
                        nr = 1;
                      } else {
                        nr = nfaces;
                      }
                      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,Nb), [=] (int g) {
                          PetscScalar     vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE];
                          PetscInt        q,d,nc,idx = Idxs[g];
                          if (idx >= 0) {
                            nc = 1;
                          } else {
                            nc = nfaces;
                          }
                          for (q = 0; q < nr; q++) {
                            for (d = 0; d < nc; d++) {
                              vals[q*nc + d] = s_scale(f,q)*s_scale(g,d)*s_fieldMats(f,g);
                            }
                          }
                          MatSetValuesDevice(d_mat,nr,&s_idx(f,0),nc,&s_idx(g,0),vals,ADD_VALUES);
                        });
                    });
                }
              }
            }
          });
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
        if (d_IPf_k) delete d_IPf_k;
      } else { // mass
        int scr_bytes = fieldMats_scr_t::shmem_size(Nq,Nq) + idx_scr_t::shmem_size(Nb,nfaces) + scale_scr_t::shmem_size(Nb,nfaces);
        ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
        ierr = PetscLogGpuFlops(nip_loc*(PetscLogDouble)(Nb*Nfloc*Nb*Nq*4));CHKERRQ(ierr);
        if (ctx->deviceType == LANDAU_CPU) PetscInfo(plex[grid], "Warning: Landau selected CPU but no support for Kokkos using CPU\n");
#else
        ierr = PetscLogFlops(nip_loc*(PetscLogDouble)(Nb*Nfloc*Nb*Nq*4));CHKERRQ(ierr);
#endif
        ierr = PetscInfo5(plex[grid], "Mass shared memory size: %d bytes in level %d conc=%D team size=%D #face=%D\n",scr_bytes,KOKKOS_SHARED_LEVEL,conc,team_size,nfaces);CHKERRQ(ierr);
        Kokkos::parallel_for("Landau mass", Kokkos::TeamPolicy<>(numCells[grid], team_size, /*Kokkos::AUTO*/ 16).set_scratch_size(KOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes)), KOKKOS_LAMBDA (const team_member team) {
            const PetscInt  elem = team.league_rank();
            fieldMats_scr_t s_fieldMats(team.team_scratch(KOKKOS_SHARED_LEVEL),Nb,Nb);
            idx_scr_t       s_idx(team.team_scratch(KOKKOS_SHARED_LEVEL),Nb,nfaces);
            scale_scr_t     s_scale(team.team_scratch(KOKKOS_SHARED_LEVEL),Nb,nfaces);
            for (int fieldA = 0; fieldA < Nfloc; fieldA++) {
              /* assemble */
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int f) {
                  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,Nb), [=] (int g) {
                      PetscScalar    t = 0;
                      for (int qj = 0 ; qj < Nq ; qj++) { // look at others integration points
                        const PetscReal *BJq = &d_BB[qj*Nb];
                        const PetscInt jpidx = qj + elem * Nq;
                        t += BJq[f] * d_mass_w_k(jpidx) * shift * BJq[g];
                      }
                      if (global_elem_mat_sz) {
                        const PetscInt fOff = (fieldA*Nb + f)*totDim + fieldA*Nb + g;
                        d_elem_mats(elem,fOff) = t;
                      } else s_fieldMats(f,g) = t;
                    });
                });
              if (!global_elem_mat_sz) { // device assembly
                const LandauIdx *const Idxs = &d_maps[grid]->gIdx[elem][fieldA][0];
                team.team_barrier(); // needed ????
                Kokkos::parallel_for(Kokkos::TeamVectorRange(team,0,Nb), [=] (int f) {
                    PetscInt q,idx = Idxs[f];
                    if (idx >= 0) {
                      s_idx(f,0) = idx + moffset;
                      s_scale(f,0) = 1.;
                    } else {
                      idx = -idx - 1;
                      for (q = 0; q < nfaces; q++) {
                        s_idx(f,q) = d_maps[grid]->c_maps[idx][q].gid + moffset;
                        s_scale(f,q) = d_maps[grid]->c_maps[idx][q].scale;
                      }
                    }
                  });
                team.team_barrier();
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,Nb), [=] (int f) {
                    PetscInt nr,idx = Idxs[f];
                    if (idx >= 0) {
                      nr = 1;
                    } else {
                      nr = nfaces;
                    }
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,Nb), [=] (int g) {
                        PetscScalar vals[LANDAU_MAX_Q_FACE*LANDAU_MAX_Q_FACE];
                        PetscInt    q,d,nc,idx = Idxs[g];
                        if (idx >= 0) {
                          nc = 1;
                        } else {
                          nc = nfaces;
                        }
                        for (q = 0; q < nr; q++) {
                          for (d = 0; d < nc; d++) {
                            vals[q*nc + d] = s_scale(f,q)*s_scale(g,d)*s_fieldMats(f,g);
                          }
                        }
                        MatSetValuesDevice(d_mat,nr,&s_idx(f,0),nc,&s_idx(g,0),vals,ADD_VALUES);
                      });
                  });
              }
            }
          });
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
      } // mass
      Kokkos::fence();
      if (global_elem_mat_sz) { // CPU assembly
        PetscSection                                                 section, globalSection;
        PetscInt                                                     cStart,cEnd;
        Kokkos::View<PetscScalar**, Kokkos::LayoutRight>::HostMirror h_elem_mats = Kokkos::create_mirror_view(d_elem_mats);

        ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
        ierr = DMPlexGetHeightStratum(plex[grid],0,&cStart,&cEnd);CHKERRQ(ierr);
        ierr = DMGetLocalSection(plex[grid], &section);CHKERRQ(ierr);
        ierr = DMGetGlobalSection(plex[grid], &globalSection);CHKERRQ(ierr);
        Kokkos::deep_copy (h_elem_mats, d_elem_mats);
        ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
        for (PetscInt ej = cStart ; ej < cEnd; ++ej) {
          const PetscScalar *elMat = &h_elem_mats(ej-cStart,0);
          ierr = DMPlexMatSetClosure(plex[grid], section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
          if (ej==-1) {
            int d,f;
            PetscPrintf(PETSC_COMM_SELF,"Kokkos Element matrix %d/%d\n",1,(int)numCells[grid]);
            for (d = 0; d < totDim; ++d){
              for (f = 0; f < totDim; ++f) PetscPrintf(PETSC_COMM_SELF," %12.5e",  PetscRealPart(elMat[d*totDim + f]));
              PetscPrintf(PETSC_COMM_SELF,"\n");
            }
            exit(14);
          }
        }
        ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
        // transition to use of maps for a Kokkos VecGetClosure
        if (ctx->gpu_assembly) {
          if (!(a_IPf || a_xarray)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "transition without Jacobian");
        }
      }
      IPfdf_idx += nip_loc*Nfloc;
      IP_idx += nip_loc;
    } // grids
    PetscFunctionReturn(0);
  }
} // extern "C"
