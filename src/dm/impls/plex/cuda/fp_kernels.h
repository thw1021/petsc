

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
PETSC_DEVICE_DATA_DECL PetscReal P2[] = {
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
PETSC_DEVICE_DATA_DECL PetscReal Q2[] = {
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
PETSC_DEVICE_DATA_DECL PetscReal P2[] = {
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
PETSC_DEVICE_DATA_DECL PetscReal Q2[] = {
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
PETSC_DEVICE_DATA_DECL PetscReal P1[] =
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
PETSC_DEVICE_DATA_DECL PetscReal Q1[] =
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
PETSC_DEVICE_DATA_DECL PetscReal P1[] =
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
PETSC_DEVICE_DATA_DECL PetscReal Q1[] =
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
PETSC_DEVICE_FUNC_DECL PetscReal polevl_10( PetscReal x, PetscReal coef[] )
{
  PetscReal ans;
  int       i;
  ans = coef[0];
  for (i=1; i<11; i++) ans = ans * x + coef[i];
  return( ans );
}
PETSC_DEVICE_FUNC_DECL PetscReal polevl_9( PetscReal x, PetscReal coef[] )
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
PETSC_DEVICE_FUNC_DECL void ellipticE(PetscReal x,PetscReal *ret)
{
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P2) - MYLOG(x) * (x * polevl_9(x,Q2));
}
/*
 *	Complete elliptic integral of the first kind
 */
PETSC_DEVICE_FUNC_DECL void ellipticK(PetscReal x,PetscReal *ret)
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
PETSC_DEVICE_FUNC_DECL void LandauTensor3D(const PetscReal x1[], const PetscReal xp, const PetscReal yp, const PetscReal zp, PetscReal U[][3], PetscReal mask)
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

PETSC_DEVICE_FUNC_DECL void LandauTensor2D(const PetscReal x[], const PetscReal rp, const PetscReal zp, PetscReal Ud[][2], PetscReal Uk[][2], const PetscReal mask)
{
  PetscReal l,s,r=x[0],z=x[1],i1func,i2func,i3func,ks,es,pi4pow,sqrt_1s,r2,rp2,r2prp2,zmzp,zmzp2,tt;
  //PetscReal mask /* = !!(r!=rp || z!=zp) */;
  /* !!(zmzp2 > 1.e-12 || (r-rp) >  1.e-12 || (r-rp) < -1.e-12); */
  r2=PetscSqr(r);
  zmzp=z-zp;
  rp2=PetscSqr(rp);
  zmzp2=PetscSqr(zmzp);
  r2prp2=r2+rp2;
  l = r2 + rp2 + zmzp2;
  /* if      ( zmzp2 >  MYSMALL) mask = 1; */
  /* else if ( (tt=(r-rp)) >  MYSMALL) mask = 1; */
  /* else if (  tt         < -MYSMALL) mask = 1; */
  /* else mask = 0; */
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

/* landau_inner_integral() */
/* Compute g2 and g3 for element */
PETSC_DEVICE_FUNC_DECL void
landau_inner_integral( const PetscInt myqi, const PetscInt mySubBlk, const PetscInt nSubBlocks, const PetscInt ip_start, const PetscInt ip_end, const PetscInt jpidx,
                       const PetscInt Nc, const PetscInt dim, const PetscReal * const IPDataGlobal, const PetscReal wiGlobal[], const PetscReal invJj[],
                       const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[], PetscBool quarter3DDomain,
                       PetscReal g2[/* FP_MAX_NQ */][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM], PetscReal g3[/* FP_MAX_NQ */][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] )
{
  PetscReal       gg2[FP_MAX_SPECIES][FP_DIM],gg3[FP_MAX_SPECIES][FP_DIM][FP_DIM];
  const PetscInt  ipdata_sz = (dim + Nc*(1+dim));
  PetscInt        d,f,d2,dp,d3,fieldB,ipidx,fieldA;
  const FPLandPointData * const __restrict__ fplpt_j = (FPLandPointData*)(IPDataGlobal + jpidx*ipdata_sz);
  const PetscReal * const vj = fplpt_j->crd, wj = wiGlobal[jpidx];
  // create g2 & g3
  for (d=0;d<dim;d++) { // clear accumulation data D & K
    for (f=0;f<Nc;f++) {
      gg2[f][d] = 0;
      for (d2=0;d2<dim;d2++) gg3[f][d][d2] = 0;
    }
  }
  for (ipidx = ip_start; ipidx < ip_end; ++ipidx) {
    const FPLandPointData * const __restrict__ fplpt = (FPLandPointData*)(IPDataGlobal + ipidx*ipdata_sz);
    const FPLandFDF * const __restrict__ fdf = &fplpt->fdf[0];
    const PetscReal wi = wiGlobal[ipidx];
#if FP_DIM==2
    PetscReal       Ud[2][2], Uk[2][2];
    LandauTensor2D(vj, fplpt->r, fplpt->z, Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
    for (fieldB = 0; fieldB < Nc; ++fieldB) {
      for (fieldA = 0; fieldA < Nc; ++fieldA) {
        for (d2 = 0; d2 < 2; ++d2) {
          for (d3 = 0; d3 < 2; ++d3) {
            /* K = U * grad(f): g2=e: i,A */
            gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * Uk[d2][d3] * fdf[fieldB].df[d3] * wi;
            /* D = -U * (I \kron (fx)): g3=f: i,j,A */
            gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * Ud[d2][d3] * fdf[fieldB].f * wi;
          }
        }
      }
    }
#else
    PetscReal U[3][3];
    if (!quarter3DDomain) {
      LandauTensor3D(vj, fplpt->x, fplpt->y, fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
      for (fieldA = 0; fieldA < Nc; ++fieldA) {
        for (fieldB = 0; fieldB < Nc; ++fieldB) {
          for (d2 = 0; d2 < 3; ++d2) {
            for (d3 = 0; d3 < 3; ++d3) {
              /* K = U * grad(f): g2 = e: i,A */
              gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldB] * U[d2][d3] * fplpt->fdf[fieldB].df[d3] * wi;
              /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
              gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * fplpt->fdf[fieldB].f * wi;
            }
          }
        }
      }
    } else {
      PetscReal lxx[] = {fplpt->x, fplpt->y}, R[2][2] = {{-1,1},{1,-1}};
      PetscReal ldf[3*FP_MAX_SPECIES];
      for (fieldB = 0; fieldB < Nc; ++fieldB) for (d3 = 0; d3 < 3; ++d3) ldf[d3 + fieldB*3] = fplpt->fdf[fieldB].df[d3] * wi * invMass[fieldB];
      for (dp=0;dp<4;dp++) {
        LandauTensor3D(vj, lxx[0], lxx[1], fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
        for (fieldA = 0; fieldA < Nc; ++fieldA) {
          for (fieldB = 0; fieldB < Nc; ++fieldB) {
            for (d2 = 0; d2 < 3; ++d2) {
              for (d3 = 0; d3 < 3; ++d3) {
                /* K = U * grad(f): g2 = e: i,A */
                gg2[fieldA][d2] += nu_alpha[fieldA]*nu_beta[fieldB] * U[d2][d3] * ldf[d3 + fieldB*3];
                /* D = -U * (I \kron (fx)): g3 = f: i,j,A */
                gg3[fieldA][d2][d3] -= nu_alpha[fieldA]*nu_beta[fieldB] * invMass[fieldA] * U[d2][d3] * f[fieldB] * wi;
              }
            }
          }
        }
        for (d3 = 0; d3 < 2; ++d3) {
          lxx[d3] *= R[d3][dp%2];
          for (fieldB = 0; fieldB < Nc; ++fieldB) {
            ldf[d3 + fieldB*3] *= R[d3][dp%2];
          }
        }
      }
    }
#endif
  } /* IPs */
  /* Jacobian transform - g2 */
  for (fieldA = 0; fieldA < Nc; ++fieldA) {
    if (mySubBlk==0) gg2[fieldA][dim-1] += Eq_m[fieldA]; /* add electric field term once per IP */
    for (d = 0; d < dim; ++d) {
      g2[myqi][mySubBlk][fieldA][d] = 0.0;
      for (d2 = 0; d2 < dim; ++d2) {
        g2[myqi][mySubBlk][fieldA][d] += invJj[d*dim+d2]*gg2[fieldA][d2];
      }
      g2[myqi][mySubBlk][fieldA][d] *= wj;
    }
  }
  /* g3 */
  for (fieldA = 0; fieldA < Nc; ++fieldA) {
    for (d = 0; d < dim; ++d) {
      for (dp = 0; dp < dim; ++dp) {
	g3[myqi][mySubBlk][fieldA][d][dp] = 0.0;
	for (d2 = 0; d2 < dim; ++d2) {
	  for (d3 = 0; d3 < dim; ++d3) {
	    g3[myqi][mySubBlk][fieldA][d][dp] += invJj[d*dim + d2]*gg3[fieldA][d2][d3]*invJj[dp*dim + d3];
	  }
	}
	g3[myqi][mySubBlk][fieldA][d][dp] *= wj;
      }
    }
  }
  // Synchronize (ensure all the data is available) and sum g2 & g3
  PETSC_DEVICE_SYNC;
  if (mySubBlk==0) { /* on one thread, sum up g2 & g3 (noop with one subblock) */
    for (fieldA = 0; fieldA < Nc; ++fieldA) {
      for (d = 0; d < dim; ++d) {
	for (d3 = 1; d3 < nSubBlocks; ++d3) {
	  g2[myqi][0][fieldA][d] += g2[myqi][d3][fieldA][d];
	  for (dp = 0; dp < dim; ++dp) {
	    g3[myqi][0][fieldA][d][dp] += g3[myqi][d3][fieldA][d][dp];
	  }
	}
      }
    }
  }
}
