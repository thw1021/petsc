#include <petsc/private/tsimpl.h>       /*I   "petscts.h"   I*/
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <petsc/private/vecimpl.h>      /* put CUDA stuff in veccuda */
#include <petscdm.h>
#include <petscdmforest.h>
#include <omp.h>

/* Landau collision operator */

#if defined(HAVE_VTUNE) && defined(__INTEL_COMPILER)
#include <ittnotify.h>
#endif

#if defined(__INTEL_COMPILER)
#if defined(PETSC_USE_REAL_SINGLE)
#define MYINVSQRT(q) invsqrtf(q)
#define MYSQRT(q) sqrtf(q)
#else
#define MYINVSQRT(q) invsqrt(q)
#define MYSQRT(q) sqrt(q)
#endif
#else
#define MYINVSQRT(q) (1./PetscSqrtReal(q))
#define MYSQRT(q) PetscSqrtReal(q)
#endif

#if defined(PETSC_USE_REAL_SINGLE)
#define MYSMALL 1.e-12F
#define MYVERYSMALL 1.e-30F
#define MYLOG logf
#else
#define MYSMALL 1.e-12
#define MYVERYSMALL 1.e-300
#define MYLOG log
#endif

#if defined(PETSC_USE_REAL_SINGLE)
static PetscReal P2[] = {
  1.53552577301013293365E-4F,
  2.50888492163602060990E-3F,
  8.68786816565889628429E-3F,
  1.07350949056076193403E-2F,
  7.77395492516787092951E-3F,
  7.58395289413514708519E-3F,
  1.15688436810574127319E-2F,
  2.18317996015557253103E-2F,
  5.68051945617860553470E-2F,
  4.43147180560990850618E-1F,
  1.00000000000000000299E0F
};
static PetscReal Q2[] = {
  3.27954898576485872656E-5F,
  1.00962792679356715133E-3F,
  6.50609489976927491433E-3F,
  1.68862163993311317300E-2F,
  2.61769742454493659583E-2F,
  3.34833904888224918614E-2F,
  4.27180926518931511717E-2F,
  5.85936634471101055642E-2F,
  9.37499997197644278445E-2F,
  2.49999999999888314361E-1F
};
#else
static PetscReal P2[] = {
  1.53552577301013293365E-4,
  2.50888492163602060990E-3,
  8.68786816565889628429E-3,
  1.07350949056076193403E-2,
  7.77395492516787092951E-3,
  7.58395289413514708519E-3,
  1.15688436810574127319E-2,
  2.18317996015557253103E-2,
  5.68051945617860553470E-2,
  4.43147180560990850618E-1,
  1.00000000000000000299E0
};
static PetscReal Q2[] = {
  3.27954898576485872656E-5,
  1.00962792679356715133E-3,
  6.50609489976927491433E-3,
  1.68862163993311317300E-2,
  2.61769742454493659583E-2,
  3.34833904888224918614E-2,
  4.27180926518931511717E-2,
  5.85936634471101055642E-2,
  9.37499997197644278445E-2,
  2.49999999999888314361E-1
};
#endif
#if defined(PETSC_USE_REAL_SINGLE)
static PetscReal P1[] =
{
 1.37982864606273237150E-4F,
 2.28025724005875567385E-3F,
 7.97404013220415179367E-3F,
 9.85821379021226008714E-3F,
 6.87489687449949877925E-3F,
 6.18901033637687613229E-3F,
 8.79078273952743772254E-3F,
 1.49380448916805252718E-2F,
 3.08851465246711995998E-2F,
 9.65735902811690126535E-2F,
 1.38629436111989062502E0F
};
static PetscReal Q1[] =
{
 2.94078955048598507511E-5F,
 9.14184723865917226571E-4F,
 5.94058303753167793257E-3F,
 1.54850516649762399335E-2F,
 2.39089602715924892727E-2F,
 3.01204715227604046988E-2F,
 3.73774314173823228969E-2F,
 4.88280347570998239232E-2F,
 7.03124996963957469739E-2F,
 1.24999999999870820058E-1F,
 4.99999999999999999821E-1F
};
#else
static PetscReal P1[] =
{
 1.37982864606273237150E-4,
 2.28025724005875567385E-3,
 7.97404013220415179367E-3,
 9.85821379021226008714E-3,
 6.87489687449949877925E-3,
 6.18901033637687613229E-3,
 8.79078273952743772254E-3,
 1.49380448916805252718E-2,
 3.08851465246711995998E-2,
 9.65735902811690126535E-2,
 1.38629436111989062502E0
};
static PetscReal Q1[] =
{
 2.94078955048598507511E-5,
 9.14184723865917226571E-4,
 5.94058303753167793257E-3,
 1.54850516649762399335E-2,
 2.39089602715924892727E-2,
 3.01204715227604046988E-2,
 3.73774314173823228969E-2,
 4.88280347570998239232E-2,
 7.03124996963957469739E-2,
 1.24999999999870820058E-1,
 4.99999999999999999821E-1
};
#endif

/* elliptic functions
 */
PETSC_STATIC_INLINE PetscReal polevl_10( PetscReal x, PetscReal coef[] )
{
  PetscReal ans;
  int       i;
  ans = coef[0];
  for (i=1; i<11; i++) ans = ans * x + coef[i];
  return( ans );
}
PETSC_STATIC_INLINE PetscReal polevl_9( PetscReal x, PetscReal coef[] )
{
  PetscReal ans;
  int       i;
  ans = coef[0];
  for (i=1; i<10; i++) ans = ans * x + coef[i];
  return( ans );
}
/*
 *	Complete elliptic integral of the second kind
 */
PETSC_STATIC_INLINE void ellipticE(PetscReal x,PetscReal *ret)
{
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P2) - MYLOG(x) * (x * polevl_9(x,Q2));
}
/*
 *	Complete elliptic integral of the first kind
 */
PETSC_STATIC_INLINE void ellipticK(PetscReal x,PetscReal *ret)
{
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P1) - MYLOG(x) * polevl_10(x,Q1);
}

/* integration point functions */
/* Evaluates the tensor U=(I-(x-y)(x-y)/(x-y)^2)/|x-y| at point x,y */
/* if x==y we will return zero. This is not the correct result */
/* since the tensor diverges for x==y but when integrated */
/* the divergent part is antisymmetric and vanishes. This is not  */
/* trivial, but can be proven. */
PETSC_STATIC_INLINE void LandauTensor2D(const PetscReal x[], const PetscReal rp, const PetscReal zp, PetscReal Ud[][2], PetscReal Uk[][2])
{
  PetscReal l,s,r=x[0],z=x[1],i1func,i2func,i3func,ks,es,pi4pow,sqrt_1s,r2,rp2,r2prp2,zmzp,zmzp2,tt;
  PetscReal mask /* = !!(r!=rp || z!=zp) */;
  /* !!(zmzp2 > 1.e-12 || (r-rp) >  1.e-12 || (r-rp) < -1.e-12); */
  r2=PetscSqr(r);
  zmzp=z-zp;
  rp2=PetscSqr(rp);
  zmzp2=PetscSqr(zmzp);
  r2prp2=r2+rp2;
  l = r2 + rp2 + zmzp2;
  if      ( zmzp2 >  MYSMALL) mask = 1;
  else if ( (tt=(r-rp)) >  MYSMALL) mask = 1;
  else if (  tt         < -MYSMALL) mask = 1;
  else mask = 0;
  s = mask*2*r*rp/l; /* mask for vectorization */
  tt = 1./(1+s);
  pi4pow = 4*M_PI*MYINVSQRT(PetscSqr(l)*l);
  sqrt_1s = MYSQRT(1.+s);
   /* sp.ellipe(2.*s/(1.+s)) */
  ellipticE(2*s*tt,&es); /* 44 flops * 2 + 75 = 163 flops including 2 logs, 1 sqrt, 1 pow, 21 mult */
  /* sp.ellipk(2.*s/(1.+s)) */
  ellipticK(2*s*tt,&ks); /* 44 flops + 75 in rest, 21 mult */
  /* mask is needed here just for single precision */
  i2func = 2./((1-s)*sqrt_1s) * es;
  i1func = 4./(PetscSqr(s)*sqrt_1s + MYVERYSMALL) * mask * ( ks - (1.+s) * es);
  i3func = 2./((1-s)*(s)*sqrt_1s + MYVERYSMALL) * (es - (1-s) * ks);
  Ud[0][0]=                    pi4pow*(rp2*i1func+PetscSqr(zmzp)*i2func);
  Ud[0][1]=Ud[1][0]=Uk[0][1]= -pi4pow*(zmzp)*(r*i2func-rp*i3func);
  Uk[1][1]=Ud[1][1]=           pi4pow*((r2prp2)*i2func-2*r*rp*i3func)*mask;
  Uk[0][0]=                    pi4pow*(zmzp2*i3func+r*rp*i1func);
  Uk[1][0]=                   -pi4pow*(zmzp)*(r*i3func-rp*i2func); /* 48 mults + 21 + 21 = 90 mults and divs */
}

/* integration point functions */
/* Evaluates the tensor U=(I-(x-y)(x-y)/(x-y)^2)/|x-y| at point x,y */
/* if x==y we will return zero. This is not the correct result */
/* since the tensor diverges for x==y but when integrated */
/* the divergent part is antisymmetric and vanishes. This is not  */
/* trivial, but can be proven. */
#if FP_DIM==3
PETSC_STATIC_INLINE void LandauTensor3D(const PetscReal x1[], const PetscReal xp, const PetscReal yp, const PetscReal zp, PetscReal U[][3], PetscReal mask)
{
  PetscReal dx[3],inorm3,inorm,inorm2,norm2,x2[] = {xp,yp,zp};
  PetscInt  d;
  for (d = 0, norm2 = MYVERYSMALL; d < 3; ++d) {
    dx[d] = x2[d] - x1[d];
    norm2 += dx[d] * dx[d];
  }
  inorm2 = mask/norm2;
  inorm = MYSQRT(inorm2);
  inorm3 = inorm2*inorm;
  for (d = 0; d < 3; ++d) U[d][d] = inorm - inorm3 * dx[d] * dx[d];
  U[1][0] = U[0][1] = -inorm3 * dx[0] * dx[1];
  U[1][2] = U[2][1] = -inorm3 * dx[2] * dx[1];
  U[2][0] = U[0][2] = -inorm3 * dx[0] * dx[2];
}
#endif
#define LAND_VL  1
static PetscErrorCode FPLandPointDataCreate(PetscReal **IPData, PetscInt dim, PetscInt nip, PetscInt Ns)
{
  PetscErrorCode  ierr, d, s, jj, nip_pad = LAND_VL*(nip/LAND_VL + !!(nip%LAND_VL)), pnt_sz = (dim + Ns*(1+dim));
  PetscReal       *pdata;
  PetscFunctionBeginUser;
  ierr = PetscMalloc(nip_pad*pnt_sz*sizeof(PetscReal),IPData);CHKERRQ(ierr);
  /* debug */
  for (jj=0, pdata = *IPData; jj<nip; jj++, pdata += pnt_sz){
    FPLandPointData *fplpt = (FPLandPointData*)pdata; /* [dim + NS*(1+dim)] */
    for(d=0;d<dim;d++) fplpt->crd[d] = 0./0.;
    for(s=0;s<Ns;s++) {
      fplpt->fdf[s].f = 0./0.;
      for(d=0;d<dim;d++) fplpt->fdf[s].df[d] = 0./0.;
    }
  }
  /* pad with zeros in case we vectorize into this */
  for (jj=nip, pdata = *IPData + nip*pnt_sz; jj < nip_pad; jj++, pdata += pnt_sz){
    FPLandPointData *fplpt = (FPLandPointData*)pdata; /* [dim + NS*(1+dim)] */
    for(d=0;d<dim;d++) fplpt->crd[d] = -1;
    for(s=0;s<Ns;s++) {
      fplpt->fdf[s].f = 0;
      for(d=0;d<dim;d++) fplpt->fdf[s].df[d] = 0;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FPLandPointDataDestroy(PetscReal *IPData)
{
  PetscErrorCode   ierr;
  PetscFunctionBeginUser;
  ierr = PetscFree(IPData);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
  FormLandau - Evaluates Jacobian matrix.

  Input Parameters:
  .  globX - input vector
  .  actx - optional user-defined context

  Output Parameters:
  .  JacP - Jacobian matrix
*/
PetscErrorCode FormLandau(Vec a_X, Mat JacP, const PetscInt dim, LandCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          cStart, cEnd, elemMatSize;
  DM                plex = 0;
  PetscDS           prob;
  PetscSection      section,globsection;
  PetscScalar       *elemMat;
  PetscInt          numCells,totDim,ej,Nq,*Nbf,*Ncf,Nb,Nc,Nfx,d,f,fieldA,fieldB,Nip,NipVec,ipdata_sz;
  PetscQuadrature   quad;
  PetscTabulation   *Tf;
  PetscReal         *wiGlob, nu_alpha[FP_MAX_SPECIES], nu_beta[FP_MAX_SPECIES];
  const PetscReal   *quadPoints, *quadWeights, *BB, *DD;
  PetscReal         *IPData;
  PetscReal         invMass[FP_MAX_SPECIES],Eq_m[FP_MAX_SPECIES],m_0=ctx->m_0; /* normalize mass -- not needed! */
  PetscLogDouble    flops;
  PetscReal         vj[FP_MAX_NQ*3],Jj[FP_MAX_NQ*9],invJj[FP_MAX_NQ*9], detJj[FP_MAX_NQ];
  Vec               locX;
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(a_X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(JacP,MAT_CLASSID,2);
  PetscValidPointer(ctx,4);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMConvert(ctx->dmv, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(plex, &locX);CHKERRQ(ierr);
  ierr = VecZeroEntries(locX);CHKERRQ(ierr); /* zero BCs so don't set */
  ierr = DMGlobalToLocalBegin(plex, a_X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (plex, a_X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globsection);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr); // Bf, &Df
  BB = Tf[0]->T[0];
  DD = Tf[0]->T[1];
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0]; /* number of vertices*S */
  ierr = PetscSectionGetNumFields(section, &Nfx);CHKERRQ(ierr);  if(Nfx!=1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nf!=1 %D",Nfx);
  ierr = PetscDSGetComponents(prob, &Ncf);CHKERRQ(ierr); Nc = Ncf[0]; if(Nc!=ctx->num_species) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nc!=S %D",Nc);
  for (fieldA=0;fieldA<Nc;fieldA++) {
    invMass[fieldA] = m_0/ctx->masses[fieldA];
    Eq_m[fieldA] = -ctx->Ez * ctx->t_0 * ctx->charges[fieldA] / (ctx->v_0 * ctx->masses[fieldA]); /* normalize dimensionless */
    if (dim==2) Eq_m[fieldA] *=  2 * M_PI; /* add the 2pi term that is not in Landau */
    nu_alpha[fieldA] = PetscSqr(ctx->charges[fieldA]/m_0)*m_0/ctx->masses[fieldA];
    nu_beta[fieldA] = PetscSqr(ctx->charges[fieldA]/ctx->epsilon0)*ctx->lnLam / (8*M_PI) * ctx->t_0*ctx->n_0/pow(ctx->v_0,3);
  }
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscFEGetQuadrature(ctx->fe, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (Nb!=Nq*ctx->num_species) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nb!=Nq %D %D over integration or simplices? Tf[0]->Nb=%D dim=%D",Nb,Nq,Tf[0]->Nb,dim);
  if (Nq >FP_MAX_NQ) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > FP_MAX_NQ (%D)",Nq,FP_MAX_NQ);
  Nip = numCells*Nq;
  NipVec = LAND_VL*(Nip/LAND_VL + !!(Nip%LAND_VL));
  flops = (PetscLogDouble)numCells*(PetscLogDouble)Nq*(PetscLogDouble)(5*dim*dim*Nc*Nc + 165);
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  elemMatSize = totDim*totDim;
  {
    PetscScalar       *uu,*u_x;
    ierr = PetscDSGetEvaluationArrays(prob, &uu, NULL, &u_x);CHKERRQ(ierr);
    /* collect f data */
    if (ctx->verbose > 3) {
      PetscInt N;
      VecGetSize(locX,&N);
      PetscPrintf(PETSC_COMM_WORLD,"[%D]%s: %D IPs, %D cells, %s elements, totDim=%D, Nb=%D, Nq=%D, elemMatSize=%D, dim=%D, Tab: Nb=%D Nc=%D Np=%D cdim=%D N=%D\n",
                  0,"FormLandau",Nq*numCells,numCells,ctx->simplex ? "SIMPLEX" : "TENSOR", totDim, Nb, Nq, elemMatSize, dim, Tf[0]->Nb, Tf[0]->Nc, Tf[0]->Np, Tf[0]->cdim, N);
    }
    ierr = FPLandPointDataCreate(&IPData, dim, Nq*numCells, Nc);CHKERRQ(ierr);
    ipdata_sz = (dim + Nc*(1+dim));
    ierr = PetscMalloc2(elemMatSize,&elemMat,NipVec,&wiGlob);CHKERRQ(ierr);
    /* cache geometry and x, f and df/dx at IPs */
    for (ej = 0; ej < numCells; ++ej) {
      PetscInt     qj,c,b,e;
      PetscScalar *coef = NULL;
      ierr = DMPlexComputeCellGeometryFEM(plex, cStart+ej, quad, vj, Jj, invJj, detJj);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plex, section, locX, cStart+ej, NULL, &coef);CHKERRQ(ierr);
      /* create point data for cell i for Landau tensor: x, f(x), grad f(x) */
      for (qj = 0; qj < Nq; ++qj) {
        PetscInt    gidx = (ej*Nq + qj);
        FPLandPointData *pnt_data = (FPLandPointData*)(IPData + gidx*ipdata_sz);
        PetscScalar refSpaceDer[3*FP_MAX_SPECIES];
        const PetscReal *Bq = &BB[qj*Nb*Nc], *Dq = &DD[qj*Nb*Nc*dim];
        for (d = 0; d < dim; ++d) pnt_data->crd[d] = vj[qj * dim + d]; /* coordinate */
        wiGlob[gidx] = detJj[qj] * quadWeights[qj];
#if FP_DIM==2
        wiGlob[gidx] *= pnt_data->r;  /* cylindrical coordinate, w/o 2pi */
#endif
        /* get u & du (EvaluateFieldJets) */
        for (c = 0; c < Nc; ++c) uu[c] = 0.0;
        for (d = 0; d < dim*Nc; ++d) refSpaceDer[d] = 0.0;
        for (c = 0; c < Nc; ++c) {
          for (b = 0; b < Nb; ++b) {
            const PetscInt cidx = b*Nc+c;
            uu[c] += Bq[cidx]*coef[b];
            for (d = 0; d < dim; ++d) refSpaceDer[c*dim+d] += Dq[cidx*dim+d]*coef[b];
          }
        }
        for (c = 0; c < Nc; ++c) for (d = 0; d < dim; ++d) for (e = 0, u_x[c*dim+d] = 0.0; e < dim; ++e) {
              u_x[c*dim+d] += invJj[e*dim+d]*refSpaceDer[c*dim+e];
              //printf("\t\t%d) u_x=%g invJj=%g u_x=%g\n",c*dim+d,u_x[c*dim+d],invJj[e*dim+d],refSpaceDer[c*dim+e]);
            }
        /* copy to IPDataLocal */
        for (c=0;c<Nc;c++) {
          pnt_data->fdf[c].f = uu[c];
          for (d = 0; d < dim; ++d) pnt_data->fdf[c].df[d] = u_x[c*dim+d];
        }
      } /* q */
      ierr = DMPlexVecRestoreClosure(plex, section, locX, cStart+ej, NULL, &coef);CHKERRQ(ierr);
    } /* e */
  }
  ierr = DMRestoreLocalVector(plex, &locX);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif
  /* outer element loop j is like a regular assembly loop */
#if defined(HAVE_VTUNE) && defined(__INTEL_COMPILER)
  __SSC_MARK(0x111); // start SDE tracing, note it uses 2 underscores
  __itt_resume(); // start VTune, again use 2 underscores
#endif
#if defined(PETSC_HAVE_CUDA)
  if (ctx->useCUDA) {
    ierr = FPLandauCUDAJacobian(plex,quad,nu_alpha,nu_beta,invMass,Eq_m,&IPData,wiGlob,ctx->subThreadBlockSize,ctx->events,ctx->quarter3DDomain,JacP);
    CHKERRQ(ierr);
  } else
#endif
  for (ej = cStart; ej < cEnd; ++ej) {
    PetscInt     qj,ipidx;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = DMPlexComputeCellGeometryFEM(plex, ej, quad, vj, Jj, invJj, detJj);CHKERRQ(ierr);
    ierr = PetscMemzero(elemMat, totDim *totDim * sizeof(PetscScalar));CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
    for (qj = 0; qj < Nq; ++qj) {
      PetscScalar     gg2[FP_MAX_SPECIES][3],gg3[FP_MAX_SPECIES][3][3];
      PetscScalar     g2[FP_MAX_SPECIES][3], g3[FP_MAX_SPECIES][3][3];
      const PetscInt  nip = numCells*Nq, jpidx = Nq*(ej-cStart) + qj; /* length of inner global interation, outer integration point */
      PetscInt        d2,dp,d3;
      const PetscReal wj = wiGlob[jpidx];
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
#endif
      for (d=0;d<dim;d++) {
	for (f=0;f<Nc;f++) {
	  gg2[f][d] = 0;
	  for (d2=0;d2<dim;d2++) gg3[f][d][d2] = 0;
	}
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogFlops(flops);CHKERRQ(ierr);
#endif
#pragma omp simd
      for (ipidx = 0; ipidx < nip; ++ipidx) {
	const PetscReal wi = wiGlob[ipidx];
        const FPLandPointData * const __restrict__ fplpt = (FPLandPointData*)(IPData + ipidx*ipdata_sz);
	if (dim==3) {
#if FP_DIM==3
	  PetscReal U[3][3], R[2][2] = {{-1,1},{1,-1}};
          if (!ctx->quarter3DDomain) {
#pragma forceinline recursive
          LandauTensor3D(&vj[qj*dim], fplpt->x, fplpt->y, fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
          for (fieldA = 0; fieldA < Nc; ++fieldA) {
            for (fieldB = 0; fieldB < Nc; ++fieldB) {
              for (d2 = 0; d2 < dim; ++d2) {
                for (d3 = 0; d3 < dim; ++d3) {
                  /* K = U * grad(f): g2=e: i,A */
                  gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * U[d2][d3] * fplpt->fdf[fieldB].df[d3] * wi;
                  /* D = -U * (I \kron (fx)): g3=f: i,j,A */
                  gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * fplpt->fdf[fieldB].f * wi;
                }
              }
            }
          }
          } else {
            PetscReal lxx[2] = {fplpt->x, fplpt->y};
            PetscReal ldf[3][FP_MAX_SPECIES];
            for (fieldB = 0; fieldB < Nc; ++fieldB) for (d3 = 0; d3 < 3; ++d3) ldf[d3][fieldB] = fplpt->fdf[fieldB].df[d3] * wi * invMass[fieldB];
            for (dp=0;dp<4;dp++) {
              LandauTensor3D(&vj[qj*dim], lxx[0], lxx[1], fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
              for (fieldA = 0; fieldA < Nc; ++fieldA) {
                for (fieldB = 0; fieldB < Nc; ++fieldB) {
                  for (d2 = 0; d2 < 3; ++d2) {
                    for (d3 = 0; d3 < 3; ++d3) {
                      /* K = U * grad(f): g2 = e: i,A */
                      gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * U[d2][d3] * ldf[d3][fieldB];
                      /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                      gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * fplpt->fdf[fieldB].f * wi;
                    }
                  }
                }
              }
              for (d3 = 0; d3 < 2; ++d3) {
                lxx[d3] *= R[d3][dp%2];
                for (fieldB = 0; fieldB < Nc; ++fieldB) {
                  ldf[d3][fieldB] *= R[d3][dp%2];
                }
              }
            }
          }
#endif
	} else {
	  PetscReal Ud[2][2], Uk[2][2];
#pragma forceinline recursive
	  LandauTensor2D(&vj[qj * dim],  fplpt->r, fplpt->z ,Ud, Uk);
	  for (fieldA = 0; fieldA < Nc; ++fieldA) {
	    for (fieldB = 0; fieldB < Nc; ++fieldB) {
	      for (d2 = 0; d2 < 2; ++d2) {
		for (d3 = 0; d3 < 2; ++d3) {
		  /* K = U * grad(f): g2=e: i,A */
		  gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * Uk[d2][d3] * fplpt->fdf[fieldB].df[d3] * wi;
		  /* D = -U * (I \kron (fx)): g3=f: i,j,A */
		  gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * Ud[d2][d3] * fplpt->fdf[fieldB].f * wi;
                }
	      }
	    }
	  }
	} /* D */
      } /* IPs */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
      /* Jacobian transform */
#pragma omp simd
      for (fieldA = 0; fieldA < Nc; ++fieldA) {
        gg2[fieldA][1] += Eq_m[fieldA]; /* add electric field term */
	for (d = 0; d < dim; ++d) {
	  g2[fieldA][d] = 0.0;
	  for (d2 = 0; d2 < dim; ++d2) {
	    g2[fieldA][d] += invJj[qj * dim * dim + d*dim+d2]*gg2[fieldA][d2];
	  }
	  g2[fieldA][d] *= wj;
	}
#pragma omp ordered simd
	for (d = 0; d < dim; ++d) {
	  for (dp = 0; dp < dim; ++dp) {
	    g3[fieldA][d][dp] = 0.0;
	    for (d2 = 0; d2 < dim; ++d2) {
	      for (d3 = 0; d3 < dim; ++d3) {
		g3[fieldA][d][dp] += invJj[qj * dim * dim + d*dim + d2] * gg3[fieldA][d2][d3] * invJj[qj * dim * dim + dp*dim + d3];
	      }
	    }
	    g3[fieldA][d][dp] *= wj;
	  }
	}
      }
      /* assemble */
      {
        const PetscReal *Bq = &BB[qj*Nb*Nc], *Dq = &DD[qj*Nb*Nc*dim];
        for (f = 0; f < Nb; ++f) {
          int fc,g,gc,df,dg;
          for (fc = 0; fc < Nc; ++fc) {
            const PetscInt fidx = f*Nc+fc; /* Test function basis index */
            const PetscInt i    = f; /* Element matrix row */
            for (g = 0; g < Nb; ++g) {
              //for (gc = 0; gc < Nc; ++gc) {
              gc = fc;
              const PetscInt gidx = g*Nc+gc; /* Trial function basis index */
              const PetscInt j    = g; /* Element matrix column */
              const PetscInt fOff = i*totDim+j;
              /* elemMat[fOff] += tmpBasisI[fidx]*g0[fc*NcJ+gc]*tmpBasisJ[gidx]; */
              for (df = 0; df < dim; ++df) {
                /* elemMat[fOff] += tmpBasisI[fidx]*g1[(fc*Nc+gc)*dim+df]*tmpBasisDerJ[gidx*dim+df]; */
                /* elemMat[fOff] += tmpBasisDerI[fidx*dim+df]*g2[(fc*Nc+gc)*dim+df]*tmpBasisJ[gidx]; */
                elemMat[fOff] += Dq[fidx*dim+df]*g2[gc][df]*Bq[gidx];
                for (dg = 0; dg < dim; ++dg) {
                  // elemMat[fOff] += tmpBasisDerI[fidx*dim+df]*g3[((fc*NcJ+gc)*dim+df)*dim+dg]*tmpBasisDerJ[gidx*dim+dg];
                  elemMat[fOff] += Dq[fidx*dim+df]*g3[gc][df][dg]*Dq[gidx*dim+dg];
                }
              }
            }
          }
        }
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
    } /* qj loop */
    if (ej==-6) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "CPU Element matrix\n");CHKERRQ(ierr);
      for (d = 0; d < totDim; ++d){
        for (f = 0; f < totDim; ++f) {
          int ci = d/Nq, cj = f/Nq, qi = d%Nq, qj = f%Nq, i = Nc*qi + ci, j = Nc*qj + cj;
          ierr = PetscPrintf(PETSC_COMM_SELF, " %19.12e", PetscRealPart(elemMat[i*totDim + j]));CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
      //exit(13);
    }
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
#endif
    /* assemble matrix */
    ierr = DMPlexMatSetClosure(plex, section, globsection, JacP, ej, elemMat, ADD_VALUES);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  } /* ej cells loop, not cuda */
#if defined(HAVE_VTUNE) && defined(__INTEL_COMPILER)
  __itt_pause(); // stop VTune
  __SSC_MARK(0x222); // stop SDE tracing
#endif
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
  /* assemble matrix or vector */
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatScale(JacP, -1.0);CHKERRQ(ierr); /* The code reflect the papers: du/dt = C, whereas PETSc use the form G(u) = du/dt - C(u) = 0 */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
  /* clean up */
  ierr = PetscFree2(elemMat,wiGlob);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  /* ierr = DMDestroy(&Gplex);CHKERRQ(ierr); */
  ierr = FPLandPointDataDestroy(IPData);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  FPLandIFunction
@*/
PetscErrorCode FPLandIFunction(TS ts,PetscReal time_dummy,Vec X,Vec X_t,Vec F,void *actx)
{
  PetscErrorCode ierr;
  LandCtx        *ctx=(LandCtx*)actx;
  PetscScalar    unorm;
  PetscInt       dim;
  PetscFunctionBeginUser;
  if (PETSC_TRUE) {
    DM dm;
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  }
  ierr = VecNorm(X,NORM_2,&unorm);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetDimension(ctx->dmv, &dim);CHKERRQ(ierr);
  if (ctx->normJ!=unorm) {
    ctx->normJ = unorm;
    ierr = FormLandau(X,ctx->J,dim,ctx);CHKERRQ(ierr);
    ctx->aux_bool = PETSC_TRUE; /* debug: set flag that we made a new Jacobian */
  } else ctx->aux_bool = PETSC_FALSE;
  /* mat vec for op */
  ierr = MatMult(ctx->J,X,F);CHKERRQ(ierr);CHKERRQ(ierr); /* C*f */
  /* add time term */
  if (X_t) {
    ierr = MatMultAdd(ctx->M,X_t,F,F);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@
  FPLandIJacobian
@*/
PetscErrorCode FPLandIJacobian(TS ts,PetscReal time_dummy,Vec X,Vec U_tdummy,PetscReal shift,Mat Amat,Mat Pmat,void *actx)
{
  PetscErrorCode ierr;
  LandCtx        *ctx=NULL;
  PetscScalar    unorm;
  PetscInt       dim;
  PetscFunctionBeginUser;
  if (1) {
    DM dm;
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
    if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  }
  if (Amat!=Pmat || Amat!=ctx->J) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Amat!=Pmat || Amat!=ctx->J");
  ierr = DMGetDimension(ctx->dmv, &dim);CHKERRQ(ierr);
  /* get collision Jacobian into A */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = VecNorm(X,NORM_2,&unorm);CHKERRQ(ierr);
  if (ctx->normJ!=unorm) {
    ierr = FormLandau(X,ctx->J,dim,ctx); CHKERRQ(ierr);
    ctx->normJ = unorm;
    ctx->aux_bool = PETSC_TRUE; /* debug: set flag that we made a new Jacobian */
  } else ctx->aux_bool = PETSC_FALSE;
  /* add C */
  ierr = MatCopy(ctx->J,Pmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  /* add mass */
  ierr = MatAXPY(Pmat,shift,ctx->M,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* < v, u > */
static void g0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt ii;
  for(ii=0;ii<numConstants;ii++) g0[ii*numConstants+ii] = 1.;
}

/* < v, u > */
static void g0_r(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt ii;
  for(ii=0;ii<numConstants;ii++) g0[ii*numConstants+ii] = 2.*M_PI*x[0];
}


/* #define LAND_ADD_BCS */
#if defined(LAND_ADD_BCS)
static void zero_bc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  PetscInt ii;
  for(ii=0;ii<numConstants;ii++) uexact[ii] = 0;
}

#endif
#define MATVEC2(__a,__x,__p) {int i,j; for (i=0.; i<2; i++) {__p[i] = 0; for (j=0.; j<2; j++) __p[i] += __a[i][j]*__x[j]; }}
static void CircleInflate(PetscReal r1, PetscReal r2, PetscReal r0, PetscInt num_sections, PetscReal x, PetscReal y,
			  PetscReal *outX, PetscReal *outY)
{
  PetscReal rr = PetscSqrtReal(x*x + y*y), outfact, efact;
  if (rr < r1 + 1.e-8) {
    *outX = x; *outY = y;
  } else {
    const PetscReal xy[2] = {x,y}, sinphi=y/rr, cosphi=x/rr;
    PetscReal cth,sth,xyprime[2],Rth[2][2],rotcos,newrr;
    if (num_sections==2) {
      rotcos = 0.70710678118654;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/4,pi/4) */
      if (sinphi >= 0.) {         /* top cell, -pi/2 */
	cth = 0.707106781186548; sth = -0.707106781186548;
      } else {                    /* bottom cell -pi/8 */
	cth = 0.707106781186548; sth = .707106781186548;
      }
    } else if (num_sections==3) {
      rotcos = 0.86602540378443;
      outfact = 1.5; efact = 2.5;
      /* rotate normalized vector into [-pi/6,pi/6) */
      if (sinphi >= 0.5) {         /* top cell, -pi/3 */
	cth = 0.5; sth = -0.866025403784439;
      } else if (sinphi >= -.5) {  /* mid cell 0 */
	cth = 1.; sth = .0;
      } else { /* bottom cell +pi/3 */
	cth = 0.5; sth = 0.866025403784439;
      }
    } else if (num_sections==4) {
      rotcos = 0.9238795325112;
      outfact = 1.5; efact = 3;
      /* rotate normalized vector into [-pi/8,pi/8) */
      if (sinphi >= 0.707106781186548) {         /* top cell, -3pi/8 */
	cth = 0.38268343236509; sth = -0.923879532511287;
      } else if (sinphi >= 0.) {                 /* mid top cell -pi/8 */
	cth = 0.923879532511287; sth = -.38268343236509;
      } else if (sinphi >= -0.707106781186548) { /* mid bottom cell + pi/8 */
	cth = 0.923879532511287; sth = 0.38268343236509;
      } else {                                   /* bottom cell + 3pi/8 */
	cth = 0.38268343236509; sth = .923879532511287;
      }
    } else {
      cth = 0.; sth = 0.; rotcos = 0; efact = 0;
    }
    Rth[0][0] = cth; Rth[0][1] =-sth;
    Rth[1][0] = sth; Rth[1][1] = cth;
    MATVEC2(Rth,xy,xyprime);
    if (num_sections==2) {
      newrr = xyprime[0]/rotcos;
    } else {
      PetscReal newcosphi=xyprime[0]/rr, rin = r1, rout = rr - rin;
      PetscReal routmax = r0*rotcos/newcosphi - rin, nroutmax = r0 - rin, routfrac = rout/routmax;
      newrr = rin + routfrac*nroutmax;
    }
    *outX = cosphi*newrr; *outY = sinphi*newrr;
    /* grade */
    PetscReal fact,tt,rs,re, rr = PetscSqrtReal(PetscSqr(*outX) + PetscSqr(*outY));
    if (rr > r2) { rs = r2; re = r0; fact = outfact;} /* outer zone */
    else {         rs = r1; re = r2; fact = efact;} /* electron zone */
    tt = (rs + pow((rr - rs)/(re - rs),fact) * (re-rs)) / rr;
    *outX *= tt;
    *outY *= tt;
  }
}

static PetscErrorCode GeometryDMLandau(DM base, PetscInt point, PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
{
  LandCtx     *ctx = (LandCtx*)a_ctx;
  PetscReal   r = abc[0], z = abc[1];
  if (ctx->inflate) {
    PetscReal absR, absZ;
    absR = PetscAbsReal(r);
    absZ = PetscAbsReal(z);
    CircleInflate(ctx->i_radius,ctx->e_radius,ctx->radius,ctx->num_sections,absR,absZ,&absR,&absZ);
    r = (r > 0) ? absR : -absR;
    z = (z > 0) ? absZ : -absZ;
  }
  xyz[0] = r;
  xyz[1] = z;
  if (dim==3) xyz[2] = abc[2];

  PetscFunctionReturn(0);
}

static PetscErrorCode ErrorIndicator_Simple(PetscInt dim, PetscReal volume, PetscReal x[], PetscInt Nf, const PetscInt Nc[], const PetscScalar u[], const PetscScalar u_x[], PetscReal *error, void *actx)
{
  PetscReal err = 0.0;
  PetscInt  f = *(PetscInt*)actx, j;
  PetscFunctionBeginUser;
  for (j = 0; j < dim; ++j) {
    err += PetscSqr(PetscRealPart(u_x[f*dim+j]));
  }
  err = u[f]; /* just use rho */
  *error = volume * err; /* * (ctx->axisymmetric ? 2.*M_PI * r : 1); */
  PetscFunctionReturn(0);
}

static PetscErrorCode LandDMCreateVMesh(MPI_Comm comm, const PetscInt dim, const char prefix[], LandCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;
  PetscReal      radius = ctx->radius;
  size_t         len;
  char           fname[128] = ""; /* we can add a file if we want */
  PetscFunctionBegin;
  /* create DM */
  ierr = PetscStrlen(fname, &len);CHKERRQ(ierr);
  if (len) {
    PetscInt dim2;
    ierr = DMPlexCreateFromFile(comm, fname, ctx->interpolate, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim2);CHKERRQ(ierr);
  } else {    /* p4est, quads */
    /* Create plex mesh of Landau domain */
    if (!ctx->sphere) {
      PetscInt    cells[] = {4,4,4};
      PetscReal   lo[] = {-radius,-radius,-radius}, hi[] = {radius,radius,radius};
      DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, dim==2 ? DM_BOUNDARY_NONE : DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
      if (dim==2) { lo[0] = 0; cells[0] = 2; }
      else if (ctx->quarter3DDomain) { lo[0] = lo[1] = 0; cells[0] = cells[1] = 2; }
      ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, lo, hi, periodicity, PETSC_TRUE, dm);CHKERRQ(ierr);
      ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
      if (dim==3) ierr = PetscObjectSetName((PetscObject) *dm, "cube");
      else ierr = PetscObjectSetName((PetscObject) *dm, "half-plane");
      CHKERRQ(ierr);
    } else if (dim==2) {
      PetscInt       numCells,cells[16][4],i,j;
      PetscInt       numVerts;
      PetscReal      inner_radius1 = ctx->i_radius, inner_radius2 = ctx->e_radius;
      double         *flatCoords = NULL;
      int            *flatCells = NULL, *pcell;
      if (ctx->num_sections==2) {
#if 1
	numCells = 5;
	numVerts = 10;
	int cells2[][4] = { {0,1,4,3},
			    {1,2,5,4},
			    {3,4,7,6},
			    {4,5,8,7},
			    {6,7,8,9} };
	for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
	ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
	{
	  double (*coords)[2] = (double (*) [2]) flatCoords;
	  for (j = 0; j < numVerts-1; j++) {
	    double z, r, theta = -M_PI_2 + (j%3) * M_PI/2;
	    double rad = (j >= 6) ? inner_radius1 : (j >= 3) ? inner_radius2 : ctx->radius;
	    z = rad * sin(theta);
	    coords[j][1] = z;
	    r = rad * cos(theta);
	    coords[j][0] = r;
	  }
	  coords[numVerts-1][0] = coords[numVerts-1][1] = 0;
	}
#else
	numCells = 4;
	numVerts = 8;
	static int     cells2[][4] = {{0,1,2,3},
				     {4,5,1,0},
				     {5,6,2,1},
				     {6,7,3,2}};
        for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
	ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
	{
	  double (*coords)[2] = (double (*) [2]) flatCoords;
	  PetscInt j;
	  for (j = 0; j < 8; j++) {
            double z, r;
	    double theta = -M_PI_2 + (j%4) * M_PI/3.;
	    double rad = ctx->radius * ((j < 4) ? 0.5 : 1.0);
	    z = rad * sin(theta);
	    coords[j][1] = z;
	    r = rad * cos(theta);
	    coords[j][0] = r;
	  }
	}
#endif
      } else if (ctx->num_sections==3) {
	numCells = 7;
	numVerts = 12;
	int cells2[][4] = { {0,1,5,4},
			    {1,2,6,5},
			    {2,3,7,6},
			    {4,5,9,8},
			    {5,6,10,9},
			    {6,7,11,10},
			    {8,9,10,11} };
	for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
	ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
	{
	  double (*coords)[2] = (double (*) [2]) flatCoords;
	  for (j = 0; j < numVerts; j++) {
	    double z, r, theta = -M_PI_2 + (j%4) * M_PI/3;
	    double rad = (j >= 8) ? inner_radius1 : (j >= 4) ? inner_radius2 : ctx->radius;
	    z = rad * sin(theta);
	    coords[j][1] = z;
	    r = rad * cos(theta);
	    coords[j][0] = r;
	  }
	}
      } else if (ctx->num_sections==4) {
	numCells = 10;
	numVerts = 16;
	int cells2[][4] = { {0,1,6,5},
			    {1,2,7,6},
			    {2,3,8,7},
			    {3,4,9,8},
			    {5,6,11,10},
			    {6,7,12,11},
			    {7,8,13,12},
			    {8,9,14,13},
			    {10,11,12,15},
			    {12,13,14,15}};
	for (i = 0; i < numCells; i++) for (j = 0; j < 4; j++) cells[i][j] = cells2[i][j];
	ierr = PetscMalloc2(numVerts * 2, &flatCoords, numCells * 4, &flatCells);CHKERRQ(ierr);
	{
	  double (*coords)[2] = (double (*) [2]) flatCoords;
	  for (j = 0; j < numVerts-1; j++) {
	    double z, r, theta = -M_PI_2 + (j%5) * M_PI/4;
	    double rad = (j >= 10) ? inner_radius1 : (j >= 5) ? inner_radius2 : ctx->radius;
	    z = rad * sin(theta);
	    coords[j][1] = z;
	    r = rad * cos(theta);
	    coords[j][0] = r;
	  }
	  coords[numVerts-1][0] = coords[numVerts-1][1] = 0;
	}
      }
      else {
        numCells = 0;
	numVerts = 0;
      }
      for (j = 0, pcell = flatCells; j < numCells; j++, pcell += 4) {
	pcell[0] = cells[j][0]; pcell[1] = cells[j][1];
	pcell[2] = cells[j][2]; pcell[3] = cells[j][3];
      }
      ierr = DMPlexCreateFromCellList(comm,2,numCells,numVerts,4,ctx->interpolate,flatCells,2,flatCoords,dm);CHKERRQ(ierr);
      ierr = PetscFree2(flatCoords,flatCells);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) *dm, "semi-circle");CHKERRQ(ierr);
    } else { /* cubed sphere, dim==3 */
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Velocity space meshes does not support cubed sphere");
    }
  }
  ierr = PetscObjectSetOptionsPrefix((PetscObject)*dm,prefix);CHKERRQ(ierr);
#if defined(LAND_ADD_BCS)
  if (1) { /* mark BCs */
    DMLabel        label;
    PetscInt       fStart, fEnd, f;
    ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(*dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      PetscInt supportSize;
      ierr = DMPlexGetSupportSize(*dm, f, &supportSize);CHKERRQ(ierr);
      if (supportSize == 1) {
	PetscReal c[3];
	ierr = DMPlexComputeCellGeometryFVM(*dm, f, NULL, c, NULL);CHKERRQ(ierr);
	if (PetscAbsReal(c[0]) >1.e-12) {
	  ierr = DMLabelSetValue(label, f, 1);CHKERRQ(ierr);
	}
      }
    }
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
  }
#endif
  /* distribute */
  /* ierr = DMPlexDistribute(*dm, 0, NULL, &dm2);CHKERRQ(ierr); */
  /* if (dm2) { */
  /*   ierr = PetscObjectSetOptionsPrefix((PetscObject)dm2,prefix);CHKERRQ(ierr); */
  /*   ierr = DMDestroy(dm);CHKERRQ(ierr); */
  /*   *dm = dm2; */
  /* } */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr); /* Plex refine */

  { /* p4est? */
    char convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, prefix, "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_type","Convert DMPlex to another format (should not be Plex!)","ex6f.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmforest;
      ierr = DMConvert(*dm,convType,&dmforest);CHKERRQ(ierr);
      if (dmforest) {
        PetscBool isForest;
        ierr = PetscObjectSetOptionsPrefix((PetscObject)dmforest,prefix);CHKERRQ(ierr);
        ierr = DMIsForest(dmforest,&isForest);CHKERRQ(ierr);
        if (isForest) {
          if (ctx->sphere && ctx->inflate) {
            ierr = DMForestSetBaseCoordinateMapping(dmforest,GeometryDMLandau,ctx);CHKERRQ(ierr);
	  }
	  ierr = DMDestroy(dm);CHKERRQ(ierr);
	  *dm = dmforest;
          ctx->errorIndicator = ErrorIndicator_Simple; /* flag for Forest */
        } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDS(DM dm, PetscInt dim, LandCtx *ctx)
{
  PetscErrorCode  ierr;
  PetscFunctionBeginUser;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, ctx->num_species, ctx->simplex, NULL, PETSC_DECIDE, &ctx->fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) ctx->fe, "f");CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) ctx->fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
#if defined(LAND_ADD_BCS)
  {
    PetscDS prob;
    PetscInt id=1;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)()) zero_bc, 1, &id, ctx);CHKERRQ(ierr);
  }
#endif
  if (1) {
    PetscInt        ii;
    PetscSection    section;
    ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
    for(ii=0;ii<ctx->num_species;ii++ ){
      char buf[256];
      if (ii==0) ierr = PetscSNPrintf(buf, 256, "e");
      else ierr = PetscSNPrintf(buf, 256, "i%D", ii);
      ierr = PetscSectionSetComponentName(section, 0, ii, buf);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode LandCreateMassMatrix(LandCtx *ctx, Vec X, DM a_dm, Mat *Amat)
{
  DM             massDM;
  PetscDS        prob;
  PetscInt       dim,N1=1,N2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(a_dm, &dim);CHKERRQ(ierr);
  ierr = DMClone(a_dm, &massDM);CHKERRQ(ierr);
  ierr = DMCopyFields(a_dm, massDM);CHKERRQ(ierr);
  ierr = DMCreateDS(massDM);CHKERRQ(ierr);
  ierr = DMGetDS(massDM, &prob);CHKERRQ(ierr);
  if (dim==3) {ierr = PetscDSSetJacobian(prob, 0, 0, g0_1, NULL, NULL, NULL);CHKERRQ(ierr);}
  else        {ierr = PetscDSSetJacobian(prob, 0, 0, g0_r, NULL, NULL, NULL);CHKERRQ(ierr);}
  ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr);
#if defined(LAND_ADD_BCS)
  ierr = DMAddBoundary(massDM, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)()) zero_bc, 1, &N1, ctx);CHKERRQ(ierr);
#endif
  ierr = DMViewFromOptions(massDM,NULL,"-mass_dm_view");CHKERRQ(ierr);
  ierr = DMCreateMatrix(massDM, Amat);CHKERRQ(ierr);
  {
    Vec locX;
    DM  plex;
    ierr = DMConvert(massDM, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetLocalVector(massDM, &locX);CHKERRQ(ierr);
    /* Mass matrix is independent of the input, so no need to fill locX */
    ierr = DMPlexSNESComputeJacobianFEM(plex, locX, *Amat, *Amat, ctx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(massDM, &locX);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&massDM);CHKERRQ(ierr);
  ierr = MatGetSize(ctx->J, &N1, NULL);CHKERRQ(ierr);
  ierr = MatGetSize(*Amat, &N2, NULL);CHKERRQ(ierr);
  if (N1 != N2) SETERRQ2(PetscObjectComm((PetscObject) a_dm), PETSC_ERR_PLIB, "Incorrect matrix sizes: |Jacobian| = %D, |Mass|=%D",N1,N2);
  ierr = MatViewFromOptions(*Amat,NULL,"-mass_mat_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Define a Maxwellian function for testing out the operator. */

 /* Using cartesian velocity space coordinates, the particle */
 /* density, [1/m^3], is defined according to */

 /* $$ n=\int_{R^3} dv^3 \left(\frac{m}{2\pi T}\right)^{3/2}\exp [- mv^2/(2T)] $$ */

 /* Using some constant, c, we normalize the velocity vector into a */
 /* dimensionless variable according to v=c*x. Thus the density, $n$, becomes */

 /* $$ n=\int_{R^3} dx^3 \left(\frac{mc^2}{2\pi T}\right)^{3/2}\exp [- mc^2/(2T)*x^2] $$ */

 /* Defining $\theta=2T/mc^2$, we thus find that the probability density */
 /* for finding the particle within the interval in a box dx^3 around x is */

 /* f(x;\theta)=\left(\frac{1}{\pi\theta}\right)^{3/2} \exp [ -x^2/\theta ] */

typedef struct {
  LandCtx   *ctx;
  PetscReal kT_m[FP_MAX_SPECIES];
  PetscReal n[FP_MAX_SPECIES];
  PetscReal shift;
} MaxwellianCtx;

static PetscErrorCode maxwellian(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  MaxwellianCtx  *mctxs = (MaxwellianCtx*)actx;
  LandCtx        *ctx = mctxs->ctx;
  PetscInt        i,ii;
  PetscFunctionBeginUser;
  for (ii=0;ii<ctx->num_species;ii++) {
    PetscReal     v2 = 0, theta = 2*mctxs->kT_m[ii]/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 */
    /* compute the exponents, v^2 */
    for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
    /* evaluate the Maxwellian */
    u[ii] = mctxs->n[ii]*pow(M_PI*theta,-1.5)*(exp(-v2/theta));
    if (ii==0 && mctxs->shift!=0.) {
      v2 = 0;
      for (i = 0; i < dim-1; ++i) v2 += x[i]*x[i];
      v2 += (x[dim-1]-mctxs->shift)*(x[dim-1]-mctxs->shift);
      /* evaluate the shifted Maxwellian */
      u[ii] += mctxs->n[ii]*pow(M_PI*theta,-1.5)*(exp(-v2/theta));
    }
  }
  PetscFunctionReturn(0);
}

/*@
 DMPlexFPAddMaxwellians -

 Input Parameters:
 .   dm

 Output Parameter:
 .   X  -

 Level: beginner
 @*/
PetscErrorCode DMPlexFPAddMaxwellians(DM dm, Vec X, PetscReal time, PetscReal temps[], PetscReal ns[], void *actx)
{
  LandCtx        *ctx = (LandCtx*)actx;
  PetscErrorCode (*initu[2])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar [], void *);
  PetscErrorCode ierr,ii;
  PetscInt       dim;
  MaxwellianCtx  mctxs,*amctxs[2];
  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (!ctx) { ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr); }
  for (ii=0;ii<ctx->num_species;ii++) {
    mctxs.kT_m[ii] = ctx->k*temps[ii]/ctx->masses[ii]; /* kT/m */
    mctxs.n[ii] = ns[ii];
  }
  mctxs.ctx = ctx;
  mctxs.shift = ctx->electronShift;
  /* need to make ADD_ALL_VALUES work - TODO */
  initu[0] = &maxwellian;
  amctxs[0] = &mctxs;
  ierr = DMProjectFunction(dm, time, &initu[0], (void**)amctxs, INSERT_ALL_VALUES, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 FPSetInitialCondition -

 Input Parameters:
 .   dm

 Output Parameter:
 .   X  -

 Level: beginner
 */
static PetscErrorCode FPSetInitialCondition(DM dm, Vec X, void *actx)
{
  LandCtx        *ctx = (LandCtx*)actx;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if (!ctx) { ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr); }
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = DMPlexFPAddMaxwellians(dm, X, 0.0, ctx->thermal_temps, ctx->n, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode adaptToleranceFEM(PetscFE fem, Vec sol, PetscReal refineTol[], PetscReal coarsenTol[], PetscInt type, LandCtx *ctx, DM *newDM)
{
  DM               dm, plex, adaptedDM = NULL;
  PetscDS          prob;
  PetscBool        isForest;
  PetscQuadrature  quad;
  PetscInt         Nq, *Nb, *Nc, cStart, cEnd, c, dim, qj, k;
  PetscScalar     *u, *u_x;
  DMLabel          adaptLabel = NULL;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  ierr = VecGetDM(sol, &dm);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* ierr = DMGetSection(dm, &section);CHKERRQ(ierr); */
  ierr = DMIsForest(dm, &isForest);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  /* ierr = DMCreateLocalVector(plex, &locX);CHKERRQ(ierr); */
  /* ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, time, NULL, NULL, NULL);CHKERRQ(ierr); */
  /* ierr = DMGlobalToLocalBegin(plex, sol, INSERT_VALUES, locX);CHKERRQ(ierr); */
  /* ierr = DMGlobalToLocalEnd  (plex, sol, INSERT_VALUES, locX);CHKERRQ(ierr); */
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF,"adapt",&adaptLabel);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, 0, 0 );CHKERRQ(ierr);
  if (Nq >FP_MAX_NQ) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Order too high. Nq = %D > FP_MAX_NQ (%D)",Nq,FP_MAX_NQ);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  if (type==4) {
    for (c = cStart; c < cEnd; c++) {
      ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
    }
    ierr = PetscInfo1(sol, "Phase:%s: Uniform refinement\n","adaptToleranceFEM");
  } else if (type==2) {
    PetscInt  rCellIdx[8], eCellIdx[64], iCellIdx[64], eMaxIdx = -1, iMaxIdx = -1, nr = 0, nrmax = (dim==3 && !ctx->quarter3DDomain) ? 8 : 2;
    PetscReal minRad = 1.e100, r, eMinRad = 1.e100, iMinRad = 1.e100;
    for (c = 0; c < 64; c++) { eCellIdx[c] = iCellIdx[c] = -1; }
    for (c = cStart; c < cEnd; c++) {
      PetscReal    tt, v0[FP_MAX_NQ*3], detJ[FP_MAX_NQ];
      ierr = DMPlexComputeCellGeometryFEM(plex, c, quad, v0, NULL, NULL, detJ);CHKERRQ(ierr);
      for (qj = 0; qj < Nq; ++qj) {
        tt = PetscSqr(v0[dim*qj+0]) + PetscSqr(v0[dim*qj+1]) + PetscSqr(((dim==3) ? v0[dim*qj+2] : 0));
	r = PetscSqrtReal(tt);
        if (r < minRad - 1.e-6) {
          minRad = r;
	  nr = 0;
          rCellIdx[nr++]= c;
          ierr = PetscInfo4(sol, "\t\tPhase: adaptToleranceFEM Found first inner r=%e, cell %D, qp %D/%D\n", r, c, qj+1, Nq);CHKERRQ(ierr);
        } else if ((r-minRad) < 1.e-8 && nr < nrmax) {
	  for (k=0;k<nr;k++) if (c == rCellIdx[k]) break;
	  if (k==nr) {
	    rCellIdx[nr++]= c;
	    ierr = PetscInfo5(sol, "\t\t\tPhase: adaptToleranceFEM Found another inner r=%e, cell %D, qp %D/%D, d=%e\n", r, c, qj+1, Nq, r-minRad);CHKERRQ(ierr);
	  }
        }
        if (ctx->sphere) {
          if ((tt=r-ctx->e_radius) > 0) {
            PetscInfo2(sol, "\t\t\t %D cell r=%g\n",c,tt);
            if (tt < eMinRad - 1.e-5) {
              eMinRad = tt;
              eMaxIdx = 0;
              eCellIdx[eMaxIdx++] = c;
            }
            else if (eMaxIdx > 0 && (tt-eMinRad) <= 1.e-5 && c != eCellIdx[eMaxIdx-1]) {
              eCellIdx[eMaxIdx++] = c;
            }
          }
          if ((tt=r-ctx->i_radius) > 0) {
            if (tt < iMinRad - 1.e-5) {
              iMinRad = tt;
              iMaxIdx = 0;
              iCellIdx[iMaxIdx++] = c;
            }
            else if ( iMaxIdx > 0 && (tt-iMinRad) <= 1.e-5  && c != iCellIdx[iMaxIdx-1]) {
              iCellIdx[iMaxIdx++] = c;
            }
          }
        }
      }
    }
    for (k=0;k<nr;k++) {
      ierr = DMLabelSetValue(adaptLabel, rCellIdx[k], DM_ADAPT_REFINE);CHKERRQ(ierr);
    }
    if (ctx->sphere) {
      for (c = 0; c < eMaxIdx; c++) {
        ierr = DMLabelSetValue(adaptLabel, eCellIdx[c], DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscInfo3(sol, "\t\tPhase:%s: refine sphere e cell %D r=%g\n","adaptToleranceFEM",eCellIdx[c],eMinRad);
      }
      for (c = 0; c < iMaxIdx; c++) {
        ierr = DMLabelSetValue(adaptLabel, iCellIdx[c], DM_ADAPT_REFINE);CHKERRQ(ierr);
        ierr = PetscInfo3(sol, "\t\tPhase:%s: refine sphere i cell %D r=%g\n","adaptToleranceFEM",iCellIdx[c],iMinRad);
      }
    }
    ierr = PetscInfo4(sol, "Phase:%s: Adaptive refine origin cells %D,%D r=%g\n","adaptToleranceFEM",rCellIdx[0],rCellIdx[1],minRad);
  } else if (type==0 || type==1 || type==3) { /* refine along r=0 axis */
    PetscScalar  *coef = NULL;
    Vec          coords;
    PetscInt     csize,Nv,d,nz;
    DM           cdm;
    PetscSection cs;
    ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetLocalSection(cdm, &cs);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt doit = 0, outside = 0;
      ierr = DMPlexVecGetClosure(cdm, cs, coords, c, &csize, &coef);CHKERRQ(ierr);
      Nv = csize/dim;
      for (nz = d = 0; d < Nv; d++) {
        PetscReal z = coef[d*dim + (dim-1)], x = PetscSqr(coef[d*dim + 0]) + PetscSqr(((dim==3) ? coef[d*dim + 1] : 0));
	x = PetscSqrtReal(x);
        if (x < 1e-12 && PetscAbsReal(z)<1e-12) doit = 1;             /* refine origin */
        else if (type==0 && (z < -1e-12 || z > ctx->re_radius+1e-12)) outside++;   /* first pass don't refine bottom */
        else if (type==1 && (z > ctx->vperp0_radius1 || z < -ctx->vperp0_radius1)) outside++; /* don't refine outside electron refine radius */
        else if (type==3 && (z > ctx->vperp0_radius2 || z < -ctx->vperp0_radius2)) outside++; /* don't refine outside ion refine radius */
        if (x < 1e-12) nz++;
      }
      ierr = DMPlexVecRestoreClosure(cdm, cs, coords, c, &csize, &coef);CHKERRQ(ierr);
      if (doit || (outside<Nv && nz)) {
        ierr = DMLabelSetValue(adaptLabel, c, DM_ADAPT_REFINE);CHKERRQ(ierr);
      }
    }
    ierr = PetscInfo1(sol, "Phase:%s: RE refinement\n","adaptToleranceFEM");
  }
  /* ierr = VecDestroy(&locX);CHKERRQ(ierr); */
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = DMAdaptLabel(dm, adaptLabel, &adaptedDM);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  *newDM = adaptedDM;
  if (adaptedDM) {
    if (isForest) {
      ierr = DMForestSetAdaptivityForest(adaptedDM,NULL);CHKERRQ(ierr);
    }
    ierr = DMConvert(adaptedDM, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = PetscInfo2(sol, "\tPhase: adaptToleranceFEM: %D cells, %d total quadrature points\n",cEnd-cStart,Nq*(cEnd-cStart));CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode adapt(DM *dm, LandCtx *ctx, Vec *uu)
{
  PetscErrorCode  ierr;
  PetscInt        type, limits[5] = {ctx->numRERefine,ctx->nZRefine1,ctx->maxRefIts,ctx->nZRefine2,ctx->postAMRRefine};
  PetscInt        adaptIter;
  PetscFunctionBeginUser;
  for (type=0;type<5;type++) {
    for (adaptIter = 0; adaptIter<limits[type];adaptIter++) {
      DM  dmNew = NULL;
      ierr = adaptToleranceFEM(ctx->fe, *uu, ctx->refineTol, ctx->coarsenTol, type, ctx, &dmNew);CHKERRQ(ierr);
      if (!dmNew) {
        exit(13);
        break;
      } else {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        ierr = VecDestroy(uu);CHKERRQ(ierr);
        ierr = DMCreateGlobalVector(dmNew,uu);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) *uu, "u");CHKERRQ(ierr);
        ierr = FPSetInitialCondition(dmNew, *uu, ctx);CHKERRQ(ierr);
        *dm = dmNew;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(LandCtx *ctx, const char prefix[])
{
  PetscErrorCode  ierr;
  PetscBool       flg, sph_flg;
  PetscInt        ii,nt,nm,nc;
  DM              dummy;
  PetscFunctionBeginUser;
  ierr = DMCreate(PETSC_COMM_WORLD,&dummy);CHKERRQ(ierr);
  /* get options - initialize context */
  ctx->verbose = 3;
  ctx->interpolate = PETSC_TRUE;
  ctx->simplex = PETSC_FALSE;
  ctx->sphere = PETSC_FALSE;
  ctx->inflate = PETSC_FALSE;
  ctx->electronShift = 0;
  ctx->errorIndicator = NULL;
  ctx->radius = 5.; /* electron thermal radius (velocity) */
  ctx->re_radius = 0.;
  ctx->vperp0_radius1 = 0;
  ctx->vperp0_radius2 = 0;
  ctx->e_radius = .1;
  ctx->i_radius = .01;
  ctx->maxRefIts = 5;
  ctx->postAMRRefine = 0;
  ctx->nZRefine1 = 0;
  ctx->nZRefine2 = 0;
  ctx->numRERefine = 0;
  ctx->num_sections = 3; /* 2, 3 or 4 */
  /* species - [0] electrons, [1] one ion species eg, duetarium, [2] heavy impurity ion, ... */
  ctx->charges[0] = -1;  /* electron charge (MKS) */
  ctx->masses[0] = 1/1835.5; /* temporary value in proton mass */
  ctx->n[0] = 1;
  /* constants, etc. */
  ctx->epsilon0 = 8.8542e-12; /* permittivity of free space (MKS) F/m */
  ctx->k = 1.38064852e-23; /* Boltzmann constant (MKS) J/K */
  ctx->lnLam = 10;         /* cross section ratio large - small angle collisions */
  ctx->n_0 = 1.e20;        /* typical plasma n, but could set it to 1 */
  ctx->Ez = 0;
  ctx->v_0 = 1; /* in electron thermal velocity */
  ctx->subThreadBlockSize = 1;
  ctx->quarter3DDomain = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, prefix, "Options for Fokker-Plank-Landau collision operator", "none");CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  ctx->useCUDA = PETSC_TRUE;
#else
  ctx->useCUDA = PETSC_FALSE;
#if defined(PETSC_HAVE_OPENMP)
  if (1) {
    int thread_id,hwthread,num_threads;
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);
#pragma omp parallel default(shared) private(hwthread, thread_id)
    {
      thread_id = omp_get_thread_num();
      hwthread = -1; //sched_getcpu();
      num_threads = omp_get_num_threads();
      PetscPrintf(PETSC_COMM_SELF,"MPI Rank %03d of %03d on HWThread %03d of Node %s, OMP_threadID %d of %d\n", 0, 1, hwthread, name, thread_id, num_threads);
    }
  }
#endif
#endif
  ierr = PetscOptionsBool("-use_cuda", "Use CUDA kernels", "xgc_dmplex.c", ctx->useCUDA, &ctx->useCUDA, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-electron_shift","Shift in thermal velocity of electrons","none",ctx->electronShift,&ctx->electronShift, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "interpolate grid points in refinement", "xgc_dmplex.c", ctx->interpolate, &ctx->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sphere", "use sphere/semi-circle domain instead of rectangle", "xgc_dmplex.c", ctx->sphere, &ctx->sphere, &sph_flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-inflate", "With sphere, inflate for curved edges (no AMR)", "xgc_dmplex.c", ctx->inflate, &ctx->inflate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-quarter_3d_domain", "Use symmetry in 3D to model 1/4 of domain", "xgc_dmplex.c", ctx->quarter3DDomain, &ctx->quarter3DDomain, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-amr_re_levels", "Number of levels to refine along v_perp=0, z>0", "xgc_dmplex.c", ctx->numRERefine, &ctx->numRERefine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-amr_z_refine1",  "Number of levels to refine along v_perp=0", "xgc_dmplex.c", ctx->nZRefine1, &ctx->nZRefine1, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-amr_z_refine2",  "Number of levels to refine along v_perp=0", "xgc_dmplex.c", ctx->nZRefine2, &ctx->nZRefine2, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-amr_levels_max", "Number of AMR levels of refinement around origin after r=0 refinements", "xgc_dmplex.c", ctx->maxRefIts, &ctx->maxRefIts, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-amr_post_refine", "Number of levels to uniformly refine after AMR", "xgc_dmplex.c", ctx->postAMRRefine, &ctx->postAMRRefine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-verbose", "", "xgc_dmplex.c", ctx->verbose, &ctx->verbose, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-re_radius","velocity range to refine on positive (z>0) r=0 axis for runaways","xgc_dmplex.c",ctx->re_radius,&ctx->re_radius, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-z_radius1","velocity range to refine r=0 axis (for electrons)","xgc_dmplex.c",ctx->vperp0_radius1,&ctx->vperp0_radius1, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-z_radius2","velocity range to refine r=0 axis (for ions) after origin AMR","xgc_dmplex.c",ctx->vperp0_radius2,&ctx->vperp0_radius2, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Ez","Initial parallel electric field in unites of Conner-Hastie criticle field","xgc_dmplex.c",ctx->Ez,&ctx->Ez, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-n_0","Normalization constant for number density","xgc_dmplex.c",ctx->n_0,&ctx->n_0, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ln_lambda","Cross section parameter","xgc_dmplex.c",ctx->lnLam,&ctx->lnLam, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_sections", "Number of tangential section in (2D) grid, 2, 3, of 4", "xgc_dmplex.c", ctx->num_sections, &ctx->num_sections, NULL);CHKERRQ(ierr);
  flg = PETSC_FALSE;
  ierr = PetscOptionsBool("-petscspace_poly_tensor", "xgc_dmplex.c", "xgc_dmplex.c", flg, &flg, NULL);CHKERRQ(ierr);
  ctx->simplex = flg ? PETSC_FALSE : PETSC_TRUE;
  /* get num species */
  {
    PetscReal arr[100];
    nt = 100;
    ierr = PetscOptionsRealArray("-thermal_temps", "Temperature of each species [e,i_0,i_1,...] in keV", "xgc_dmplex.c", arr, &nt, &flg);CHKERRQ(ierr);
    if (flg && nt > FP_MAX_SPECIES) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-thermal_temps ,t1,t2,.. number of species %D > MAX %D",nt,FP_MAX_SPECIES);
  }
  nt = FP_MAX_SPECIES;
  for (ii=0;ii<FP_MAX_SPECIES;ii++) ctx->thermal_temps[ii] = 1.;
  ierr = PetscOptionsRealArray("-thermal_temps", "Temperature of each species [e,i_0,i_1,...] in keV", "xgc_dmplex.c", ctx->thermal_temps, &nt, &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInfo1(dummy, "num_species set to number of thermal temps provided (%D)\n",nt);
    ctx->num_species = nt;
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"-thermal_temps ,t1,t2,.. must be provided to set the number of species");
  for (ii=0;ii<ctx->num_species;ii++) ctx->thermal_temps[ii] *= 1.1604525e7; /* convert to Kelvin */
  nm = FP_MAX_SPECIES-1;
  ierr = PetscOptionsRealArray("-ion_masses", "Mass of each species in units of proton mass [i_0=2,i_1=40...]", "xgc_dmplex.c", &ctx->masses[1], &nm, &flg);CHKERRQ(ierr);
  if (flg && nm != ctx->num_species-1) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"num ion masses %D != num species %D",nm,ctx->num_species-1);
  }
  nm = FP_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-n", "Normalized (by -n_0) number density of each species", "xgc_dmplex.c", ctx->n, &nm, &flg);CHKERRQ(ierr);
  if (flg && nm != ctx->num_species) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"wrong num n: %D != num species %D",nm,ctx->num_species);
  }
  ctx->n_0 *= ctx->n[0]; /* normalized number density */
  for (ii=1;ii<ctx->num_species;ii++) ctx->n[ii] = ctx->n[ii]/ctx->n[0];
  ctx->n[0] = 1;
  for (ii=0;ii<FP_MAX_SPECIES;ii++) ctx->masses[ii] *= 1.6720e-27; /* scale by proton mass kg */
  ctx->masses[0] = 9.10938356e-31; /* electron mass kg (should be about right already) */
  ctx->m_0 = ctx->masses[0]; /* arbitrary reference mass, electrons */
  ierr = PetscOptionsReal("-v_0","Velocity to normalize with in units of initial electrons thermal velocity (not recommended to change default)","xgc_dmplex.c",ctx->v_0,&ctx->v_0, NULL);CHKERRQ(ierr);
  ctx->v_0 *= PetscSqrtReal(ctx->k*ctx->thermal_temps[0]/(ctx->masses[0])); /* electron mean velocity in 1D (need 3D form in computing T from FE integral) */
  nc = FP_MAX_SPECIES-1;
  ierr = PetscOptionsRealArray("-ion_charges", "Charge of each species in units of proton charge [i_0=2,i_1=18,...]", "main.c", &ctx->charges[1], &nc, &flg);CHKERRQ(ierr);
  if (flg && nc != ctx->num_species-1) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"num charges %D != num species %D",nc,ctx->num_species-1);
  }
  for (ii=0;ii<FP_MAX_SPECIES;ii++) ctx->charges[ii] *= 1.6022e-19; /* electron/proton charge (MKS) */
  ctx->t_0 = 8*M_PI*PetscSqr(ctx->epsilon0*ctx->m_0/PetscSqr(ctx->charges[0]))/ctx->lnLam/ctx->n_0*pow(ctx->v_0,3); /* note, this t_0 makes nu[0,0]=1 */
  /* geometry */
  for (ii=0;ii<ctx->num_species;ii++) ctx->refineTol[ii]  = PETSC_MAX_REAL;
  for (ii=0;ii<ctx->num_species;ii++) ctx->coarsenTol[ii] = 0.;
  ii = FP_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-refine_tol","tolerance for refining cells in AMR","xgc_dmplex.c",ctx->refineTol, &ii, &flg);CHKERRQ(ierr);
  if (flg && ii != ctx->num_species) ierr = PetscInfo2(dummy, "Phase: Warning, #refine_tol %D != num_species %D\n",ii,ctx->num_species);CHKERRQ(ierr);
  ii = FP_MAX_SPECIES;
  ierr = PetscOptionsRealArray("-coarsen_tol","tolerance for coarsening cells in AMR","xgc_dmplex.c",ctx->coarsenTol, &ii, &flg);CHKERRQ(ierr);
  if (flg && ii != ctx->num_species) ierr = PetscInfo2(dummy, "Phase: Warning, #coarsen_tol %D != num_species %D\n",ii,ctx->num_species);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-domain_radius","Phase space size in units of electron thermal velocity","xgc_dmplex.c",ctx->radius,&ctx->radius, &flg);CHKERRQ(ierr);
  if (flg && ctx->radius <= 0) { /* negative is ratio of c */
    if (ctx->radius == 0) ctx->radius = 0.75;
    else ctx->radius = -ctx->radius;
    ctx->radius = ctx->radius*299792458/ctx->v_0;
    ierr = PetscInfo1(dummy, "Change domain radius to %e\n",ctx->radius);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-i_radius","Ion thermal velocity, used for circular meshes","xgc_dmplex.c",ctx->i_radius,&ctx->i_radius, &flg);CHKERRQ(ierr);
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE; /* you gave me an ion radius but did not set sphere, user error really */
  if (!flg) {
    ctx->i_radius = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[1]/ctx->masses[1]/M_PI)/ctx->v_0; /* normalized radius with thermal velocity of first ion */
    /* ierr = PetscInfo1(dummy, "Phase: Warning i_radius not provided, using 2.5 * first ion thermal temp %e\n",ctx->i_radius);CHKERRQ(ierr); */
  }
  ierr = PetscOptionsReal("-e_radius","Electron thermal velocity, used for circular meshes","xgc_dmplex.c",ctx->e_radius,&ctx->e_radius, &flg);CHKERRQ(ierr);
  if (flg && !sph_flg) ctx->sphere = PETSC_TRUE; /* you gave me an e radius but did not set sphere, user error really */
  if (!flg) {
    ctx->e_radius = 1.5*PetscSqrtReal(8*ctx->k*ctx->thermal_temps[0]/ctx->masses[0]/M_PI)/ctx->v_0; /* normalized radius with thermal velocity of electrons */
    /* ierr = PetscInfo1(dummy, "Phase: Warning e_radius not provided, using 2.5 * electron thermal temp %e\n",ctx->masses[0]);CHKERRQ(ierr); */
  }
  /* ierr = PetscInfo2(dummy, "Phase: electron radius = %g, ion radius = %g\n",ctx->e_radius,ctx->i_radius);CHKERRQ(ierr); */
  if (ctx->sphere && (ctx->e_radius <= ctx->i_radius || ctx->radius <= ctx->e_radius)) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"bad radii: %g < %g < %g",ctx->i_radius,ctx->e_radius,ctx->radius);
  ierr = PetscOptionsInt("-sub_thread_block_size", "Number of threads in CUDA integration point subblock", "xgc_dmplex.c", ctx->subThreadBlockSize, &ctx->subThreadBlockSize, NULL);CHKERRQ(ierr);
  if (ctx->subThreadBlockSize > FP_MAX_SUB_THREAD_BLOCKS) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"num sub threads %D > MAX %D",ctx->subThreadBlockSize,FP_MAX_SUB_THREAD_BLOCKS);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  for (ii=ctx->num_species;ii<FP_MAX_SPECIES;ii++) ctx->masses[ii] = ctx->thermal_temps[ii]  = ctx->charges[ii] = 0;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "masses:        e=%10.3e; ions in proton mass units:   %10.3e %10.3e ...\n",ctx->masses[0],ctx->masses[1]/1.6720e-27,ctx->num_species>2 ? ctx->masses[2]/1.6720e-27 : 0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "charges:       e=%10.3e; charges in elementary units: %10.3e %10.3e\n", ctx->charges[0],-ctx->charges[1]/ctx->charges[0],ctx->num_species>2 ? -ctx->charges[2]/ctx->charges[0] : 0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "thermal T (K): e=%10.3e i=%10.3e imp=%10.3e. v_0=%10.3e n_0=%10.3e t_0=%10.3e domain=%10.3e\n",ctx->thermal_temps[0],ctx->thermal_temps[1],ctx->num_species>2 ? ctx->thermal_temps[2] : 0,ctx->v_0,ctx->n_0,ctx->t_0,ctx->radius);
  CHKERRQ(ierr);
  ierr = DMDestroy(&dummy);CHKERRQ(ierr);
  {
    PetscMPIInt    rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    /* PetscLogStage  setup_stage; */
    ierr = PetscLogEventRegister("Landau Operator", DM_CLASSID, &ctx->events[0]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister(" Jacobian-setup", DM_CLASSID, &ctx->events[1]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" Jacobian-kern-i", DM_CLASSID, &ctx->events[3]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister(" Jacobian-kernel", DM_CLASSID, &ctx->events[4]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" Jacobian-trans", DM_CLASSID, &ctx->events[5]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" Jacobian-assem", DM_CLASSID, &ctx->events[6]);CHKERRQ(ierr); /* 6 */
    ierr = PetscLogEventRegister(" Jacobian-end", DM_CLASSID, &ctx->events[7]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister("  Jac-geo-color", DM_CLASSID, &ctx->events[8]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister("  Jac-cuda-sum", DM_CLASSID, &ctx->events[2]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister("Landau Jacobian", DM_CLASSID, &ctx->events[9]);CHKERRQ(ierr); /* 9 */
    if (rank) { /* turn off output stuff for duplicate runs - do we need to add the prefix to all this? */
      ierr = PetscOptionsClearValue(NULL,"-snes_converged_reason");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ksp_converged_reason");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-snes_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ksp_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-ts_adapt_monitor");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-amr_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-amr_vec_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-mass_mat_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-mass_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-pre_dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-pre_vec_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-info");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexFPCreateVelocitySpace - Create a DMPlex velocity space mesh

  Collective on comm

  Input Parameters:
+   comm  - The MPI communicator
.   dim - velocity space dimension (2 for axisymmetric, 3 for full 3X + 3V solver)
-   prefix -

  Output Parameter:
.   dm  - The DM object representing the mesh
+   X - A vector (user destroys)
-   J - Matrix (object destroys)

  Level: beginner

.keywords: mesh
.seealso: DMPlexCreate()
@*/
PetscErrorCode DMPlexFPCreateVelocitySpace(MPI_Comm comm, PetscInt dim, const char prefix[], Vec *X, Mat *J, DM *dm)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  LandCtx        *ctx;
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size!=1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Velocity space meshes should be serial (but should work in parallel)");
  if (dim!=2 && dim!=3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Only 2D and 3D supported");
  ctx = malloc(sizeof(LandCtx));
  /* process options */
  ierr = ProcessOptions(ctx,prefix);CHKERRQ(ierr);
  /* Create Mesh */
  ierr = LandDMCreateVMesh(comm, dim, prefix, ctx, dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm,NULL,"-pre_dm_view");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, ctx);CHKERRQ(ierr);
  /* create FEM */
  ierr = SetupDS(*dm,dim,ctx);CHKERRQ(ierr);
  /* set initial state */
  ierr = DMCreateGlobalVector(*dm,X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *X, "u");CHKERRQ(ierr);
  /* initial static refinement, no solve */
  ierr = FPSetInitialCondition(*dm, *X, ctx);CHKERRQ(ierr);
  ierr = VecViewFromOptions(*X, NULL, "-pre_vec_view");CHKERRQ(ierr);
  /* forest refinement */
  if (ctx->errorIndicator) {
    /* AMR */
    ierr = adapt(dm,ctx,X);CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm,NULL,"-amr_dm_view");CHKERRQ(ierr);
    ierr = VecViewFromOptions(*X, NULL, "-amr_vec_view");CHKERRQ(ierr);
  }
  ierr = DMSetApplicationContext(*dm, ctx);CHKERRQ(ierr);
  ctx->dmv = *dm;
  ierr = DMCreateMatrix(ctx->dmv, &ctx->J);CHKERRQ(ierr);
  *J = ctx->J;
  ierr = LandCreateMassMatrix(ctx,*X,ctx->dmv,&ctx->M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexFPDestroyVelocitySpace - Destroy a DMPlex velocity space mesh

  Input/Output Parameters:
  .   dm

  Level: beginner
@*/
PetscErrorCode DMPlexFPDestroyVelocitySpace(DM *dm)
{
  PetscErrorCode ierr;
  LandCtx        *ctx;
  PetscContainer container = NULL;
  PetscFunctionBegin;
  ierr = DMGetApplicationContext(*dm, &ctx);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ctx->J,"coloring", (PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ctx->M);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->J);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&ctx->fe);CHKERRQ(ierr);
  free(ctx);
  ierr = DMDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* < v, ru > */
static void f0_s_den(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)constants[0];
  // for(ii=0;ii<numConstants;ii++)
  f0[0] = u[ii];
}

/* < v, ru > */
static void f0_s_mom(PetscInt dim, PetscInt Nf, PetscInt NfAux,
		    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
		    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
		    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)constants[0], jj = (PetscInt)constants[1];
  //for(ii=0;ii<numConstants;ii++)
  f0[0] = x[jj]*u[ii]; /* x momentum */
}

static void f0_s_v2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt i, ii = (PetscInt)constants[0];
  //for(ii=0;ii<numConstants;ii++) {
  double tmp1 = 0.;
  for (i = 0; i < dim; ++i) tmp1 += x[i]*x[i];
  f0[0] = tmp1*u[ii];
}

/* < v, ru > */
static void f0_s_rden(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)constants[0];
  //for(ii=0;ii<numConstants;ii++)
  f0[0] = 2.*M_PI*x[0]*u[ii];
}

/* < v, ru > */
static void f0_s_rmom(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)constants[0];
  //for(ii=0;ii<numConstants;ii++)
  f0[0] = 2.*M_PI*x[0]*x[1]*u[ii];
}

static void f0_s_rv2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii = (PetscInt)constants[0];
  //for(ii=0;ii<numConstants;ii++)
  f0[0] =              2.*M_PI*x[0]*(x[0]*x[0] + x[1]*x[1])*u[ii];
  if (dim==3) f0[0] += 2.*M_PI*x[0]*(x[2]*x[2]            )*u[ii];
}

/*@
  DMPlexFPPrintNorms

  Input/Output Parameters:
.   X

  Level: beginner
@*/
PetscErrorCode DMPlexFPPrintNorms(Vec X, PetscInt stepi)
{
  PetscErrorCode ierr;
  LandCtx        *ctx;
  PetscDS        prob;
  DM             plex,dm;
  PetscInt       cStart, cEnd, dim, ii;
  PetscScalar    xmomentumtot=0, ymomentumtot=0, zmomentumtot=0, energytot=0, densitytot=0, tt;
  PetscScalar    xmomentum[FP_MAX_SPECIES],  ymomentum[FP_MAX_SPECIES],  zmomentum[FP_MAX_SPECIES], energy[FP_MAX_SPECIES], density[FP_MAX_SPECIES];
  PetscFunctionBegin;
  ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no DM");
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = DMConvert(ctx->dmv, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMCreateDS(plex);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  /* print momentum and energy */
  for (ii=0;ii<ctx->num_species;ii++) {
    PetscReal user[2] = {ii,ctx->charges[ii]};
    if (dim==2) { /* 2/3X + 3V (cylindrical coordinates) */
      ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
      ierr = PetscDSSetObjective(prob, 0, &f0_s_rden);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      density[ii] = tt*ctx->n_0*ctx->charges[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_rmom);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      zmomentum[ii] = tt*ctx->n_0*ctx->v_0*ctx->masses[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_rv2);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      energy[ii] = tt*0.5*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
      PetscPrintf(PETSC_COMM_WORLD, "%3D) species-%D: charge density= %20.13e z-momentum= %20.13e energy= %20.13e\n",stepi,ii,density[ii],zmomentum[ii],energy[ii]);
      zmomentumtot += zmomentum[ii];
      energytot  += energy[ii];
      densitytot += density[ii];
    } else { /* 2/3X + 3V */
      ierr = PetscDSSetConstants(prob, 2, user);CHKERRQ(ierr);
      ierr = PetscDSSetObjective(prob, 0, &f0_s_den);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      density[ii] = tt*ctx->n_0*ctx->charges[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_mom);CHKERRQ(ierr);
      user[1] = 0;
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      xmomentum[ii]  = tt*ctx->n_0*ctx->v_0*ctx->masses[ii];
      user[1] = 1;
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      ymomentum[ii] = tt*ctx->n_0*ctx->v_0*ctx->masses[ii];
      user[1] = 2;
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      zmomentum[ii] = tt*ctx->n_0*ctx->v_0*ctx->masses[ii];
      ierr = PetscDSSetObjective(prob, 0, &f0_s_v2);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,&tt,ctx);CHKERRQ(ierr);
      energy[ii]    = 0.5*tt*ctx->n_0*ctx->v_0*ctx->v_0*ctx->masses[ii];
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%3D) species %D: density=%20.13e, x-momentum=%20.13e, y-momentum=%20.13e, z-momentum=%20.13e, energy=%21.13e\n",
                         stepi,ii,density[ii],xmomentum[ii],ymomentum[ii],zmomentum[ii],energy[ii]);
      CHKERRQ(ierr);
      xmomentumtot += xmomentum[ii];
      ymomentumtot += ymomentum[ii];
      zmomentumtot += zmomentum[ii];
      energytot  += energy[ii];
      densitytot += density[ii];
    }
  }
  /* totals */
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  if (ctx->num_species>1) {
    if (dim==2) {
      PetscPrintf(PETSC_COMM_WORLD, "\t%3D) Total: charge density=%21.13e, momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %D cells)",
                  stepi,densitytot,zmomentumtot,energytot,ctx->masses[1]/ctx->masses[0],cEnd-cStart);
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "\t%3D) Total: charge density=%21.13e, x-momentum=%21.13e, y-momentum=%21.13e, z-momentum=%21.13e, energy=%21.13e (m_i[0]/m_e = %g, %D cells)",
                  stepi,densitytot,xmomentumtot,ymomentumtot,zmomentumtot,energytot,ctx->masses[1]/ctx->masses[0],cEnd-cStart);
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD, " -- %D cells",cEnd-cStart);
  }
  if (ctx->useCUDA) PetscPrintf(PETSC_COMM_WORLD, ", %D sub threads\n",ctx->subThreadBlockSize);
  else PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscFunctionReturn(0);
}
