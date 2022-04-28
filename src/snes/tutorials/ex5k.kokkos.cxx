#include <Kokkos_Core.hpp>
#include <petscdmda_kokkos.hpp>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include "ex5.h"

using namespace Kokkos;
using DefaultMemorySpace                 = Kokkos::DefaultExecutionSpace::memory_space;
using ConstPetscScalarKokkosOffsetView2D = Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,DefaultMemorySpace>;
using PetscScalarKokkosOffsetView2D      = Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,DefaultMemorySpace>;

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

  PetscCall(PetscLogFlops(12.0*info->ym*info->xm));
  PetscCallMPI(MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,comm));
  PetscFunctionReturn(0);
}

