#include <Kokkos_Core.hpp>
#include <petscdmda_kokkos.hpp>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>


#include <petsc/private/matimpl.h>


#include "ex5.h"

using namespace Kokkos;
using DefaultMemorySpace                 = Kokkos::DefaultExecutionSpace::memory_space;
using ConstPetscScalarKokkosOffsetView2D = Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,DefaultMemorySpace>;
using PetscScalarKokkosOffsetView2D      = Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,DefaultMemorySpace>;

using PetscCountKokkosView           = Kokkos::View<PetscCount*,DefaultMemorySpace>;
using PetscIntKokkosView             = Kokkos::View<PetscInt*,DefaultMemorySpace>;
using PetscCountKokkosViewHost       = Kokkos::View<PetscCount*,Kokkos::HostSpace>;
using PetscScalarKokkosView          = Kokkos::View<PetscScalar*,DefaultMemorySpace>;


KOKKOS_INLINE_FUNCTION PetscErrorCode MMSSolution1(AppCtx *user,const DMDACoor2d *c,PetscScalar *u)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  u[0] = x*(1 - x)*y*(1 - y);
  return 0;
}

KOKKOS_INLINE_FUNCTION PetscErrorCode MMSForcing1(PetscReal user_param,const DMDACoor2d *c,PetscScalar *f)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  f[0] = 2*x*(1 - x) + 2*y*(1 - y) - user_param*PetscExpReal(x*(1 - x)*y*(1 - y));
  return 0;
}

PetscErrorCode FormFunctionLocalExt_Kokkos(DMDALocalInfo *info,Vec x,Vec f,AppCtx *user)
{
  PetscReal      lambda,hx,hy,hxdhy,hydhx;
  PetscInt       xs = info->xs,ys = info->ys,xm = info->xm,ym = info->ym,mx = info->mx,my = info->my;
  PetscReal      user_param = user->param;

  ConstPetscScalarKokkosOffsetView2D xv;
  PetscScalarKokkosOffsetView2D      fv;

  PetscFunctionBeginUser;
  lambda = user->param;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy;
  hydhx  = hy/hx;
  /*
     Compute function over the locally owned part of the grid
  */
  PetscCallCXX(DMDAVecGetKokkosOffsetView(info->da,x,&xv));
  PetscCallCXX(DMDAVecGetKokkosOffsetViewWrite(info->da,f,&fv));

  PetscCallCXX(Kokkos::parallel_for ("FormFunctionLocalVec_Kokkos",
    MDRangePolicy <Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA (PetscInt j,PetscInt i)
  {
    DMDACoor2d   c;
    PetscScalar  u,ue,uw,un,us,uxx,uyy,mms_solution,mms_forcing;

    if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
      c.x = i*hx; c.y = j*hy;
      MMSSolution1(user,&c,&mms_solution);
      fv(j,i) = 2.0*(hydhx+hxdhy)*(xv(j,i) - mms_solution);
    } else {
      u  = xv(j,i);
      uw = xv(j,i-1);
      ue = xv(j,i+1);
      un = xv(j-1,i);
      us = xv(j+1,i);

      /* Enforce boundary conditions at neighboring points -- setting these values causes the Jacobian to be symmetric. */
      if (i-1 == 0) {c.x = (i-1)*hx; c.y = j*hy; MMSSolution1(user,&c,&uw);}
      if (i+1 == mx-1) {c.x = (i+1)*hx; c.y = j*hy; MMSSolution1(user,&c,&ue);}
      if (j-1 == 0) {c.x = i*hx; c.y = (j-1)*hy; MMSSolution1(user,&c,&un);}
      if (j+1 == my-1) {c.x = i*hx; c.y = (j+1)*hy; MMSSolution1(user,&c,&us);}

      uxx     = (2.0*u - uw - ue)*hydhx;
      uyy     = (2.0*u - un - us)*hxdhy;
      mms_forcing = 0;
      c.x = i*hx; c.y = j*hy;
      MMSForcing1(user_param,&c,&mms_forcing);
      fv(j,i) = uxx + uyy - hx*hy*(lambda*PetscExpScalar(u) + mms_forcing);
    }
  }));

  PetscCallCXX(DMDAVecRestoreKokkosOffsetView(info->da,x,&xv));
  PetscCallCXX(DMDAVecRestoreKokkosOffsetViewWrite(info->da,f,&fv));

  PetscCall(PetscLogFlops(11.0*info->ym*info->xm));
  PetscFunctionReturn(0);
}

PetscErrorCode FormObjectiveLocalExt_Kokkos(DMDALocalInfo *info,Vec x,PetscReal *obj,AppCtx *user)
{
  PetscInt       xs = info->xs,ys = info->ys,xm = info->xm,ym = info->ym,mx = info->mx,my = info->my;
  PetscReal      lambda,hx,hy,hxdhy,hydhx,sc,lobj=0;
  MPI_Comm       comm;

  ConstPetscScalarKokkosOffsetView2D xv;

  PetscFunctionBeginUser;
  *obj   = 0;
  PetscCall(PetscObjectGetComm((PetscObject)info->da,&comm));
  lambda = user->param;
  hx     = 1.0/(PetscReal)(mx-1);
  hy     = 1.0/(PetscReal)(my-1);
  sc     = hx*hy*lambda;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;
  /*
     Compute function over the locally owned part of the grid
  */
  PetscCallCXX(DMDAVecGetKokkosOffsetView(info->da,x,&xv));

  PetscCallCXX(Kokkos::parallel_reduce("FormObjectiveLocalExt_Kokkos",
    MDRangePolicy <Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA (PetscInt j,PetscInt i,PetscReal& update)
  {
    PetscScalar    u,ue,uw,un,us,uxux,uyuy;
    if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
      update += PetscRealPart((hydhx + hxdhy)*xv(j,i)*xv(j,i));
    } else {
      u  = xv(j,i);
      uw = xv(j,i-1);
      ue = xv(j,i+1);
      un = xv(j-1,i);
      us = xv(j+1,i);

      if (i-1 == 0)    uw = 0.;
      if (i+1 == mx-1) ue = 0.;
      if (j-1 == 0)    un = 0.;
      if (j+1 == my-1) us = 0.;

      /* F[u] = 1/2\int_{\omega}\nabla^2u(x)*u(x)*dx */

      uxux = u*(2.*u - ue - uw)*hydhx;
      uyuy = u*(2.*u - un - us)*hxdhy;

      update += PetscRealPart(0.5*(uxux + uyuy) - sc*PetscExpScalar(u));
    }
  },lobj));

  PetscCallCXX(DMDAVecRestoreKokkosOffsetView(info->da,x,&xv));

  PetscCall(PetscLogFlops(12.0*info->ym*info->xm));
  PetscCallMPI(MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,comm));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetCOOLocal(Mat mat,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],PetscInt coo_i[],PetscInt coo_j[])
{
  PetscInt       buf[8192],*bufr=NULL,*bufc=NULL;
  const PetscInt *irowm,*icolm;

  PetscFunctionBeginUser;
  if (!nrow || !ncol) PetscFunctionReturn(0); /* no values to insert */

  if ((!mat->rmap->mapping && !mat->cmap->mapping) || (nrow+ncol) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
    bufr  = buf;
    bufc  = buf + nrow;
    irowm = bufr;
    icolm = bufc;
  } else {
    PetscCall(PetscMalloc2(nrow,&bufr,ncol,&bufc));
    irowm = bufr;
    icolm = bufc;
  }
  if (mat->rmap->mapping) PetscCall(ISLocalToGlobalMappingApply(mat->rmap->mapping,nrow,irow,bufr));
  else irowm = irow;

  if (mat->cmap->mapping) {
    if (mat->cmap->mapping != mat->rmap->mapping || ncol != nrow || icol != irow) {
      PetscCall(ISLocalToGlobalMappingApply(mat->cmap->mapping,ncol,icol,bufc));
    } else icolm = irowm;
  } else icolm = icol;

  PetscInt k = 0;
  for (PetscInt j=0; j<ncol; j++) { /* The order in putting coo_i/j[] must be the same as puting coo_v[] !!! */
    for (PetscInt i=0; i<nrow; i++) {
      coo_i[k] = irowm[i];
      coo_j[k] = icolm[j];
      k++;
    }
  }
  if (bufr != buf) PetscCall(PetscFree2(bufr,bufc));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetCOOStencil(Mat mat,PetscInt m,const MatStencil idxm[],PetscInt n,const MatStencil idxn[],PetscInt coo_i[],PetscInt coo_j[])
{
  PetscInt       buf[8192],*bufm=NULL,*bufn=NULL,*jdxm,*jdxn;
  PetscInt       j,i,dim = mat->stencil.dim,*dims = mat->stencil.dims+1,tmp;
  PetscInt       *starts = mat->stencil.starts,*dxm = (PetscInt*)idxm,*dxn = (PetscInt*)idxn,sdim = dim - (1 - (PetscInt)mat->stencil.noc);

  PetscFunctionBeginUser;
  if (!m || !n) PetscFunctionReturn(0); /* no values to insert */

  if ((m+n) <= (PetscInt)(sizeof(buf)/sizeof(PetscInt))) {
    jdxm = buf; jdxn = buf+m;
  } else {
    PetscCall(PetscMalloc2(m,&bufm,n,&bufn));
    jdxm = bufm; jdxn = bufn;
  }
  for (i=0; i<m; i++) {
    for (j=0; j<3-sdim; j++) dxm++;
    tmp = *dxm++ - starts[0];
    for (j=0; j<dim-1; j++) {
      if ((*dxm++ - starts[j+1]) < 0 || tmp < 0) tmp = -1;
      else                                       tmp = tmp*dims[j] + *(dxm-1) - starts[j+1];
    }
    if (mat->stencil.noc) dxm++;
    jdxm[i] = tmp;
  }
  for (i=0; i<n; i++) {
    for (j=0; j<3-sdim; j++) dxn++;
    tmp = *dxn++ - starts[0];
    for (j=0; j<dim-1; j++) {
      if ((*dxn++ - starts[j+1]) < 0 || tmp < 0) tmp = -1;
      else                                       tmp = tmp*dims[j] + *(dxn-1) - starts[j+1];
    }
    if (mat->stencil.noc) dxn++;
    jdxn[i] = tmp;
  }
  PetscCall(MatGetCOOLocal(mat,m,jdxm,n,jdxn,coo_i,coo_j));
  PetscCall(PetscFree2(bufm,bufn));
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianLocalExt_Kokkos(DMDALocalInfo *info,Vec x,Mat jac,Mat jacpre,AppCtx *user)
{
  PetscInt       i,j,k,q;
  PetscInt       xs = info->xs,ys = info->ys,xm = info->xm,ym = info->ym,mx = info->mx,my = info->my;
  MatStencil     col[5],row;
  PetscScalar    lambda,hx,hy,hxdhy,hydhx,sc;
  DM             coordDA;
  Vec            coordinates;
  DMDACoor2d     **coords;

  PetscFunctionBeginUser;
  lambda = user->param;
  /* Extract coordinates */
  PetscCall(DMGetCoordinateDM(info->da, &coordDA));
  PetscCall(DMGetCoordinates(info->da, &coordinates));

  PetscCall(DMDAVecGetArray(coordDA, coordinates, &coords));
  hx     = xm > 1 ? PetscRealPart(coords[ys][xs+1].x) - PetscRealPart(coords[ys][xs].x) : 1.0;
  hy     = ym > 1 ? PetscRealPart(coords[ys+1][xs].y) - PetscRealPart(coords[ys][xs].y) : 1.0;
  PetscCall(DMDAVecRestoreArray(coordDA, coordinates, &coords));

  hxdhy  = hx/hy;
  hydhx  = hy/hx;
  sc     = hx*hy*lambda;

  PetscCount *offsets;
  PetscInt   *coo_i,*coo_j,*ip,*jp;

  PetscCall(PetscMalloc1(xm*ym+1,&offsets)); /* +1 for CSR-like data structure */
  PetscCall(PetscMalloc2(xm*ym*5,&coo_i,xm*ym*5,&coo_j)); /* 5-point stencil such that each row has at most 5 nonzeros */
  offsets[0] = 0;
  q  = 0; /* row counter */
  ip = coo_i;
  jp = coo_j;
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.j = j; row.i = i;
      k = 0; /* count nonzeros in the row */
      /* boundary points */
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        k++;
        PetscCall(MatGetCOOStencil(jacpre,1,&row,1,&row,ip,jp));
      } else {
        /* interior grid points */
        if (j-1 != 0) {
          col[k].j = j - 1; col[k].i = i;
          k++;
        }
        if (i-1 != 0) {
          col[k].j = j;     col[k].i = i-1;
          k++;
        }

        col[k].j = row.j; col[k].i = row.i; k++;

        if (i+1 != mx-1) {
          col[k].j = j;     col[k].i = i+1;
          k++;
        }
        if (j+1 != mx-1) {
          col[k].j = j + 1; col[k].i = i;
          k++;
        }
        PetscCall(MatGetCOOStencil(jacpre,1,&row,k,col,ip,jp));
      }
      offsets[q+1] = offsets[q] + k;
      ip += k;
      jp += k;
      q++;
    }
  }

  PetscCall(MatSetPreallocationCOO(jacpre,offsets[q],coo_i,coo_j));
  PetscCall(PetscFree2(coo_i,coo_j));

  PetscCountKokkosView               offsetsv = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),PetscCountKokkosViewHost(offsets,offsets[q]));
  PetscScalarKokkosView              coo_v("coo_v",offsets[q]);
  ConstPetscScalarKokkosOffsetView2D xv;

  PetscCallCXX(DMDAVecGetKokkosOffsetView(info->da,x,&xv));

  PetscCallCXX(Kokkos::parallel_for ("FormFunctionLocalVec_Kokkos",
    MDRangePolicy <Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA (PetscInt j,PetscInt i)
  {
    PetscInt q = (j-ys)*xm + (i-xs);
    PetscInt p = offsetsv(q);
    /* boundary points */
    if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
      coo_v(p++) =  2.0*(hydhx + hxdhy);
    } else {
      /* interior grid points */
      if (j-1 != 0) {
        coo_v(p++)     = -hxdhy;
      }
      if (i-1 != 0) {
        coo_v(p++)     = -hydhx;
      }

      coo_v(p++) = 2.0*(hydhx + hxdhy) - sc*PetscExpScalar(xv(j,i));

      if (i+1 != mx-1) {
        coo_v(p++)     = -hydhx;
      }
      if (j+1 != mx-1) {
        coo_v(p++)     = -hxdhy;
      }
    }
  }));
  PetscCall(MatSetValuesCOO(jacpre,coo_v.data(),INSERT_VALUES));
  PetscCallCXX(DMDAVecRestoreKokkosOffsetView(info->da,x,&xv));
  PetscCall(PetscFree(offsets));
  PetscFunctionReturn(0);
}