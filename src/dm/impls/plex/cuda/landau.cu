/*
   Implements the Landau kernal
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <../src/mat/impls/aij/seq/aij.h>  /* put CUDA SeqAIJ */
#include <petsc/private/kernels/petscaxpy.h>
#include <omp.h>

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

texture<int, 2, cudaReadModeElementType> tex;

__global__ void kernel (int m, int n)
{
    int val;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            val = tex2D (tex, col+0.5f, row+0.5f);
            printf ("%3d  ", val);
        }
        printf ("\n");
    }
}

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
__constant__ PetscReal P2[] = {
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
__constant__ PetscReal Q2[] = {
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
__constant__ PetscReal P2[] = {
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
__constant__ PetscReal Q2[] = {
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
__constant__ PetscReal P1[] =
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
__constant__ PetscReal Q1[] =
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
__constant__ PetscReal P1[] =
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
__constant__ PetscReal Q1[] =
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
__device__ PetscReal polevl_10( PetscReal x, PetscReal coef[] )
{
  PetscReal ans;
  int       i;
  ans = coef[0];
  for (i=1; i<11; i++) ans = ans * x + coef[i];
  return( ans );
}
__device__ PetscReal polevl_9( PetscReal x, PetscReal coef[] )
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
__device__ void ellipticE(PetscReal x,PetscReal *ret)
{
  x = 1 - x; /* where m = 1 - m1 */
  *ret = polevl_10(x,P2) - MYLOG(x) * (x * polevl_9(x,Q2));
}
/*
 *	Complete elliptic integral of the first kind
 */
__device__ void ellipticK(PetscReal x,PetscReal *ret)
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
__device__ void LandauTensor3D(const PetscReal x1[], const PetscReal xp, const PetscReal yp, const PetscReal zp, PetscReal U[][3], PetscReal mask)
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

__device__ void LandauTensor2D(const PetscReal x[], const PetscReal rp, const PetscReal zp, PetscReal Ud[][2], PetscReal Uk[][2], const PetscReal mask)
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

// #define FP_USE_SHARED_GPU_MEM
//
// The GPU Landau kernel
//
__global__
void land_kernel(const PetscInt nip, const PetscInt dim, const PetscInt totDim, const PetscInt Nc, const PetscInt Nb,
		 const PetscReal vj[], const PetscReal Jj[], const PetscReal invJj[],
		 const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
		 const PetscReal * const BB, const PetscReal * const DD, const PetscReal quadWeights[],
		 const PetscReal * const IPDataGlobal, const PetscReal wiGlobal[],
#if !defined(FP_USE_SHARED_GPU_MEM)
		 PetscReal *g2arr, PetscReal *g3arr,
#endif
		 PetscBool quarter3DDomain, PetscScalar elemMats_out[])
{
  const PetscInt  Nq = blockDim.x, myelem = blockIdx.x;
#if defined(FP_USE_SHARED_GPU_MEM)
  extern __shared__ PetscReal g2_g3_qi[]; // Nq * { [NSubBlocks][Nc][dim] ; [NSubBlocks][Nc][dim][dim] }
  PetscReal       (*g2)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM]         = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM])         &g2_g3_qi[0];
  PetscReal       (*g3)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM]) &g2_g3_qi[FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM];
#else
  PetscReal       (*g2)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM]         = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM])         &g2arr[myelem*FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM       ];
  PetscReal       (*g3)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM]) &g3arr[myelem*FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*FP_DIM];
#endif
  const PetscInt  mythread = threadIdx.x + blockDim.x*threadIdx.y, myqi = threadIdx.x, mySubBlk = threadIdx.y, nSubBlocks = blockDim.y;
  const PetscInt  jpidx = myqi + myelem * Nq;
  const PetscInt  ipdata_sz = (dim + Nc*(1+dim)); // x[dim], f[Ns], df[dim*Nc]
  const PetscInt  subblocksz = nip/nSubBlocks + !!(nip%nSubBlocks), ip_start = mySubBlk*subblocksz, ip_end = (mySubBlk+1)*subblocksz > nip ? nip : (mySubBlk+1)*subblocksz; /* this could be wrong with very few global IPs */
  const PetscReal *pvj = &vj[jpidx*dim];
  const PetscReal wj = wiGlobal[jpidx];
  PetscReal       gg2[FP_MAX_SPECIES][FP_DIM],gg3[FP_MAX_SPECIES][FP_DIM][FP_DIM];
  PetscInt        d,f,d2,dp,d3,fieldB,ipidx,fieldA;
  // create g2 & g3
  for (d=0;d<dim;d++) { // clear accumulation data D & K
    for (f=0;f<Nc;f++) {
      gg2[f][d] = 0;
      for (d2=0;d2<dim;d2++) gg3[f][d][d2] = 0;
    }
  }
  for (ipidx = ip_start; ipidx < ip_end; ++ipidx) {
    const FPLandPointData * const __restrict__ fplpt = (FPLandPointData*)(IPDataGlobal + ipidx*ipdata_sz);
    const PetscReal wi = wiGlobal[ipidx];
    if (dim==2) {
      PetscReal       Ud[2][2], Uk[2][2];
      LandauTensor2D(pvj, fplpt->r, fplpt->z, Ud, Uk, (ipidx==jpidx) ? 0. : 1.);
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
    } else {
      PetscReal U[3][3];
      if (!quarter3DDomain) {
#if FP_DIM==3
	LandauTensor3D(pvj, fplpt->x, fplpt->y, fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
#endif
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
#if FP_DIM==3
	PetscReal lxx[] = {fplpt->x, fplpt->y}, R[2][2] = {{-1,1},{1,-1}};
	PetscReal ldf[3*FP_MAX_SPECIES];
	for (fieldB = 0; fieldB < Nc; ++fieldB) for (d3 = 0; d3 < 3; ++d3) ldf[d3 + fieldB*3] = fplpt->fdf[fieldB].df[d3] * wi * invMass[fieldB];
	for (dp=0;dp<4;dp++) {
	  LandauTensor3D(pvj, lxx[0], lxx[1], fplpt->z, U, (ipidx==jpidx) ? 0. : 1.);
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
#endif
      }
    }
  } /* IPs */
  /* Jacobian transform - g2 */
  for (fieldA = 0; fieldA < Nc; ++fieldA) {
    if (mySubBlk==0) gg2[fieldA][dim-1] += Eq_m[fieldA]; /* add electric field term once per IP */
    for (d = 0; d < dim; ++d) {
      (*g2)[myqi][mySubBlk][fieldA][d] = 0.0;
      for (d2 = 0; d2 < dim; ++d2) {
	(*g2)[myqi][mySubBlk][fieldA][d] += invJj[jpidx * dim * dim + d*dim+d2]*gg2[fieldA][d2];
      }
      (*g2)[myqi][mySubBlk][fieldA][d] *= wj;
    }
  }
  /* g3 */
  for (fieldA = 0; fieldA < Nc; ++fieldA) {
    for (d = 0; d < dim; ++d) {
      for (dp = 0; dp < dim; ++dp) {
	(*g3)[myqi][mySubBlk][fieldA][d][dp] = 0.0;
	for (d2 = 0; d2 < dim; ++d2) {
	  for (d3 = 0; d3 < dim; ++d3) {
	    (*g3)[myqi][mySubBlk][fieldA][d][dp] += invJj[jpidx * dim * dim + d*dim + d2]*gg3[fieldA][d2][d3]*invJj[jpidx * dim * dim + dp*dim + d3];
	  }
	}
	(*g3)[myqi][mySubBlk][fieldA][d][dp] *= wj;
      }
    }
  }
  // Synchronize (ensure all the data is available) and sum g2 & g3
  __syncthreads();
  if (mySubBlk==0) { /* on one thread, sum up */
    for (fieldA = 0; fieldA < Nc; ++fieldA) {
      for (d = 0; d < dim; ++d) {
	for (ipidx = 1; ipidx < nSubBlocks; ++ipidx) {
	  (*g2)[myqi][0][fieldA][d] += (*g2)[myqi][ipidx][fieldA][d];
	  for (dp = 0; dp < dim; ++dp) {
	    (*g3)[myqi][0][fieldA][d][dp] += (*g3)[myqi][ipidx][fieldA][d][dp];
	  }
	}
      }
    }
  }
  // Synchronize (ensure all the data is available) and sum IP matrices
  __syncthreads();
  if (mythread==0) { // on one thread, sum up
    int ii,qj;
    PetscScalar *elemMat  = &elemMats_out[myelem*totDim*totDim]; /* my output */
    for (ii=0;ii<totDim*totDim;ii++) elemMat[ii] = 0;
    for (qj=0;qj<Nq;qj++) {
      const PetscReal *Bq = &BB[qj*Nb*Nc], *Dq = &DD[qj*Nb*Nc*dim];
      for (f = 0; f < Nb; ++f) {
	int fc,g,gc,df,dg;
	for (fc = 0; fc < Nc; ++fc) {
	  const PetscInt fidx = f*Nc+fc; /* Test function basis index */
	  const PetscInt i    = f; /* Element matrix row */
	  for (g = 0; g < Nb; ++g) {
	    gc = fc;
	    const PetscInt gidx = g*Nc+gc; /* Trial function basis index */
	    const PetscInt j    = g; /* Element matrix column */
	    const PetscInt fOff = i*totDim+j;
	    /* elemMat[fOff] += tmpBasisI[fidx]*g0[fc*NcJ+gc]*tmpBasisJ[gidx]; */
	    for (df = 0; df < dim; ++df) {
	      /* elemMat[fOff] += tmpBasisI[fidx]*g1[(fc*Nc+gc)*dim+df]*tmpBasisDerJ[gidx*dim+df]; */
	      /* elemMat[fOff] += tmpBasisDerI[fidx*dim+df]*g2[(fc*Nc+gc)*dim+df]*tmpBasisJ[gidx]; */
	      // elemMat[fOff] += Dq[fidx*dim+df]*g2[gc][df]*Bq[gidx];
	      elemMat[fOff] += Dq[fidx*dim+df]*(*g2)[qj][0][gc][df]*Bq[gidx];
	      for (dg = 0; dg < dim; ++dg) {
		// elemMat[fOff] += tmpBasisDerI[fidx*dim+df]*g3[((fc*NcJ+gc)*dim+df)*dim+dg]*tmpBasisDerJ[gidx*dim+dg];
		// elemMat[fOff] += Dq[fidx*dim+df]*g3[gc][df][dg]*Dq[gidx*dim+dg];
		elemMat[fOff] += Dq[fidx*dim + df]*(*g3)[qj][0][gc][df][dg]*Dq[gidx*dim + dg];
	      }
	    }
	  }
	}
      }
    }
    if (myelem==-6) {
      printf("GPU Element matrix\n");
      for (d = 0; d < totDim; ++d){
        for (f = 0; f < totDim; ++f)
          printf(" %17.10e", elemMat[d*totDim + f]);
        printf("\n");
      }
    }
  }
}

struct _ISColoring_ctx {
  ISColoring coloring;
};
typedef struct _ISColoring_ctx * ISColoring_ctx;

static PetscErrorCode destroy_coloring (void *obj)
{
  PetscErrorCode    ierr;
  ISColoring_ctx    coloring_ctx = (ISColoring_ctx)obj;
  ISColoring        isc = coloring_ctx->coloring;
  PetscFunctionBegin;
  ierr = ISColoringDestroy(&isc);CHKERRQ(ierr);
  ierr = PetscFree(coloring_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

__global__ void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[]);
static PetscErrorCode assemble_omp_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container);
static PetscErrorCode assemble_cuda_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container, const PetscLogEvent events[]);

PetscErrorCode FPLandauCUDAJacobian( DM plex, PetscQuadrature quad, const PetscReal nu_alpha[],const PetscReal nu_beta[],
				     const PetscReal invMass[], const PetscReal Eq_m[], const PetscReal * const IPDataGlobal,
				     const PetscReal wiGlobal[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain, 
				     Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          ii,ej,*Nbf,Nb,nqdimGC,nqdim2GC,cStart,cEnd,Nfx,Nc,dim,numGCells,Nq,totDim,nip,szf=sizeof(PetscReal);
  PetscReal         *vj,*Jj,*invJj,*vj_a,*Jj_a,*invJj_a;
  const PetscReal   *quadWeights, *BB, *DD;
  PetscReal         *d_quadWeights,*d_BB,*d_DD;
  PetscReal         *d_vj,*d_Jj,*d_invJj,*d_wiGlobal,*d_nu_alpha,*d_nu_beta,*d_invMass,*d_Eq_m;
  PetscScalar       *elemMats,*d_elemMats;
  PetscLogDouble    flops;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscReal        *d_IPDataGlobal;
  PetscContainer    container = NULL;
  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=FP_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "FP_DIM != dim");
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numGCells = cEnd - cStart;
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, &quadWeights);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_quadWeights,Nq*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_quadWeights, quadWeights, Nq*szf, cudaMemcpyHostToDevice));
  nip  = numGCells*Nq; /* length of inner global iteration */
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  ierr = PetscDSGetNumFields(prob, &Nfx);CHKERRQ(ierr); if(Nfx!=1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nf!=1 %D",Nfx);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  Nc = Tf[0]->Nc;
  if (Nb!=Nq*Nc)SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nb!=Nq*Nc");
  BB = Tf[0]->T[0];
  DD = Tf[0]->T[1];
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  // create data
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_IPDataGlobal, nip*(dim + Nc*(dim+1))*szf )); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_alpha, Nc*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_beta,  Nc*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invMass,  Nc*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Eq_m,     Nc*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_IPDataGlobal, IPDataGlobal, nip*(dim + Nc*(dim+1))*szf, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_alpha, nu_alpha, Nc*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_beta,  nu_beta,  Nc*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invMass,  invMass,  Nc*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Eq_m,     Eq_m,     Nc*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_BB,    Nc*Nq*Nb*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_DD,    Nc*Nq*Nb*dim*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_BB, BB, Nq*Nb*Nc*szf,     cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_DD, DD, Nq*Nb*Nc*dim*szf, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_wiGlobal,           Nq*numGCells*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_wiGlobal, wiGlobal, Nq*numGCells*szf,   cudaMemcpyHostToDevice));
  // collect geometry
  flops = (PetscLogDouble)numGCells*(PetscLogDouble)Nq*(PetscLogDouble)(5.*dim*dim*Nc*Nc + 165.);
  nqdim2GC = Nq*dim*dim*numGCells;
  nqdimGC  = Nq*dim*numGCells;
  ierr = PetscMalloc3(nqdimGC,&vj_a,nqdim2GC,&Jj_a,nqdim2GC,&invJj_a);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_vj,    nqdimGC*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Jj,    nqdim2GC*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invJj, nqdim2GC*szf)); // kernel input
  for (ej = cStart, vj = vj_a, Jj = Jj_a, invJj = invJj_a;
       ej < cEnd;
       ++ej, vj += Nq*dim, Jj += Nq*dim*dim, invJj += Nq*dim*dim) {
    PetscReal  detJ[FP_MAX_NQ];
    ierr = DMPlexComputeCellGeometryFEM(plex, cStart+ej, quad, vj, Jj, invJj, detJ);CHKERRQ(ierr);
  }
  CUDA_SAFE_CALL(cudaMemcpy(d_vj,       vj_a, nqdimGC*szf,        cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Jj,       Jj_a, nqdim2GC*szf,       cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invJj, invJj_a, nqdim2GC*szf,       cudaMemcpyHostToDevice));
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
#endif
  {
    PetscReal  *d_g2g3;
    dim3 dimBlock(Nq,num_sub_blocks);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output
    ii = FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*(1+FP_DIM)*FP_MAX_SUB_THREAD_BLOCKS;
#if defined(FP_USE_SHARED_GPU_MEM)
    /* PetscPrintf(PETSC_COMM_SELF,"Call land_kernel with %D words shared memory\n",ii); */
    land_kernel<<<numGCells,dimBlock,ii*szf>>>( nip,dim,totDim,Nc,Nb,d_vj,d_Jj,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
						d_BB, d_DD, d_quadWeights, d_IPDataGlobal, d_wiGlobal, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
#else
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_g2g3, ii*szf*numGCells)); // kernel input
    PetscReal  *g2 = &d_g2g3[0];
    PetscReal  *g3 = &d_g2g3[FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*numGCells];
    land_kernel<<<numGCells,dimBlock>>>( nip,dim,totDim,Nc,Nb,d_vj,d_Jj,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
					 d_BB, d_DD, d_quadWeights, d_IPDataGlobal, d_wiGlobal, g2, g3, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL (cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaFree(d_g2g3));
  }
#endif
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
#endif
  // delete device data
  CUDA_SAFE_CALL(cudaFree(d_IPDataGlobal));
  CUDA_SAFE_CALL(cudaFree(d_vj));
  CUDA_SAFE_CALL(cudaFree(d_Jj));
  CUDA_SAFE_CALL(cudaFree(d_invJj));
  CUDA_SAFE_CALL(cudaFree(d_quadWeights));
  CUDA_SAFE_CALL(cudaFree(d_wiGlobal));
  CUDA_SAFE_CALL(cudaFree(d_nu_alpha));
  CUDA_SAFE_CALL(cudaFree(d_nu_beta));
  CUDA_SAFE_CALL(cudaFree(d_invMass));
  CUDA_SAFE_CALL(cudaFree(d_Eq_m));
  CUDA_SAFE_CALL(cudaFree(d_BB));
  CUDA_SAFE_CALL(cudaFree(d_DD));
  ierr = PetscFree3(vj_a,Jj_a,invJj_a);CHKERRQ(ierr);
  ierr = PetscMalloc1(totDim*totDim*numGCells,&elemMats);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMemcpy(elemMats, d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_elemMats));
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
#endif
  /* coloring */
  ierr = PetscObjectQuery((PetscObject)JacP,"coloring",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) {
    PetscInt        cell,i,nc,Nv;
    ISColoring      iscoloring = NULL;
    ISColoring_ctx  coloring_ctx = NULL;
    Mat             G,Q;
    PetscScalar     ones[64];
    MatColoring     mc;
    IS             *is;
    PetscInt        csize,colour,j,k;
    const PetscInt *indices;
    PetscInt       numComp[1];
    PetscInt       numDof[4];
    PetscFE        fe;
    DM             colordm;
    PetscSection   csection;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
#endif
    /* create cell centered DM */
    ierr = DMClone(plex, &colordm);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) plex), dim, 1, PETSC_FALSE, "color_", PETSC_DECIDE, &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, "color");CHKERRQ(ierr);
    ierr = DMSetField(colordm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    for (i = 0; i < (dim+1); ++i) numDof[i] = 0;
    numDof[dim] = 1;
    numComp[0] = 1;
    ierr = DMPlexCreateSection(colordm, NULL, numComp, numDof, 0, 0, NULL, NULL, NULL, &csection);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(csection, 0, "color");CHKERRQ(ierr);
    ierr = DMSetLocalSection(colordm, csection);CHKERRQ(ierr);
    ierr = DMViewFromOptions(colordm,NULL,"-color_dm_view");CHKERRQ(ierr);
    /* get vertex to element map Q and colroing graph G */
    ierr = MatGetSize(JacP,NULL,&Nv);CHKERRQ(ierr);
    ierr = MatCreateAIJ(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,numGCells,Nv,totDim,NULL,0,NULL,&Q);CHKERRQ(ierr);
    for(i=0;i<64;i++) ones[i] = 1.0;
    for (cell = cStart, ej = 0 ; cell < cEnd; ++cell, ++ej) {
      PetscInt numindices,*indices;
      ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, NULL);CHKERRQ(ierr);
      ierr = MatSetValues(Q,1,&ej,numindices,indices,ones,ADD_VALUES);CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatTransposeMult(Q,Q,MAT_INITIAL_MATRIX,4.0,&G);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) Q, "Q");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) G, "coloring graph");CHKERRQ(ierr);
    ierr = MatViewFromOptions(G,NULL,"-coloring_mat_view");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Q,NULL,"-coloring_mat_view");CHKERRQ(ierr);
    ierr = MatDestroy(&Q);CHKERRQ(ierr);
    /* coloring */
    ierr = MatColoringCreate(G,&mc);CHKERRQ(ierr);
    ierr = MatColoringSetDistance(mc,1);CHKERRQ(ierr);
    ierr = MatColoringSetType(mc,MATCOLORINGJP);CHKERRQ(ierr);
    ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
    ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
    ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
    /* view */
    ierr = ISColoringViewFromOptions(iscoloring,NULL,"-coloring_is_view");CHKERRQ(ierr);
    ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
    if (0) {
      PetscViewer    viewer;
      Vec            color_vec, eidx_vec;
      ierr = DMGetGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
      for (colour=0; colour<nc; colour++) {
	ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
	ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
	for (j=0; j<csize; j++) {
	  PetscScalar v = (PetscScalar)colour;
	  k = indices[j];
	  ierr = VecSetValues(color_vec,1,&k,&v,INSERT_VALUES);
	  v = (PetscScalar)k;
	  ierr = VecSetValues(eidx_vec,1,&k,&v,INSERT_VALUES);
	}
	ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
      }
      /* view */
      //ierr = VecViewFromOptions(color_vec, NULL, "-color_vec_view");CHKERRQ(ierr);
      //ierr = VecViewFromOptions(eidx_vec, NULL, "-eidx_vec_view");CHKERRQ(ierr);
      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "color.vtk");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) color_vec, "color");CHKERRQ(ierr);
      ierr = VecView(color_vec, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "eidx.vtk");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) eidx_vec, "element-idx");CHKERRQ(ierr);
      ierr = VecView(eidx_vec, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&csection);CHKERRQ(ierr);
    ierr = DMDestroy(&colordm);CHKERRQ(ierr);
    ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
    ierr = MatDestroy(&G);CHKERRQ(ierr);
    /* stash coloring */
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscNew(&coloring_ctx);CHKERRQ(ierr);
    coloring_ctx->coloring = iscoloring;
    ierr = PetscContainerSetPointer(container,(void*)coloring_ctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, destroy_coloring);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)JacP,"coloring",(PetscObject)container);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP)
    if (1) {
      int thread_id,num_threads;
      //char name[MPI_MAX_PROCESSOR_NAME];
      // int resultlength;
      //MPI_Get_processor_name(name, &resultlength);
#pragma omp parallel default(shared) private(thread_id)
      {
	thread_id = omp_get_thread_num();
	num_threads = omp_get_num_threads();
	PetscPrintf(PETSC_COMM_WORLD, "Made coloring with %D colors. OMP_threadID %d of %d\n", nc, thread_id, num_threads);
      }
    }
#endif
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
#endif
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  if (0) {
    PetscScalar *elMat;
    for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
    }
  } else if (0) { /* OMP assembly */
    ierr = assemble_omp_private(cStart, cEnd, totDim, plex, section, globalSection, JacP, elemMats, container);CHKERRQ(ierr);
  } else {  /* gpu assembly */
    ierr = assemble_cuda_private(cStart, cEnd, totDim, plex, section, globalSection, JacP, elemMats, container, events);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMats);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

__global__
void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[])
{
  const PetscInt     myelem = (gridDim.x==1) ? threadIdx.x : blockIdx.x;
  Mat_SeqAIJ         a = mats[colors[myelem]]; /* copy to GPU */
  const PetscScalar *v = el_mats[myelem];
  const PetscInt    *in = idx_arr[myelem], *im = idx_arr[myelem], n = nidx_arr[myelem], m = nidx_arr[myelem];
  /* mat set values */
  PetscInt          *rp,k,low,high,t,row,nrow,i,col,l;
  PetscInt          *ai = a.i,*ailen = a.ilen;
  PetscInt          *aj = a.j,lastcol = -1;
  MatScalar         *ap=NULL,value=0.0,*aa = a.a;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    rp   = aj + ai[row];
    ap = aa + ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      /* if (in[l] < 0) { */
      /* 	printf("\t\tin[l] < 0 ?????\n"); */
      /* 	continue; */
      /* } */
      col = in[l];
      value = v[l + k*n];
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        // if (rp[i] > col) break;
        if (rp[i] == col) {
	  ap[i] += value;
	  low = i + 1;
          goto noinsert;
        }
      }
      printf("\t\t\t ERROR in assemble_kernel\n");
    noinsert:;
    }
  }
}
static PetscErrorCode assemble_omp_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container)
{
  PetscErrorCode  ierr;
  IS             *is;
  PetscInt        nc,colour,j;
  const PetscInt *clr_idxs;
  ISColoring_ctx  coloring_ctx = NULL;
  ISColoring      iscoloring;
  ierr = PetscContainerGetPointer(container,(void**)&coloring_ctx);CHKERRQ(ierr);
  iscoloring = coloring_ctx->coloring;
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
  for (colour=0; colour<nc; colour++) {
    PetscInt    *idx_arr[64];
    PetscScalar *new_el_mats[64];
    PetscInt     idx_size[64],csize;
    ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
    if (csize>64) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements in color. %D > %D",csize,64);
    ierr = ISGetIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    /* get indices and mats */
    for (j=0; j<csize; j++) {
      PetscInt cell = cStart + clr_idxs[j];
      PetscInt numindices,*indices;
      PetscScalar *elMat = &elemMats[clr_idxs[j]*totDim*totDim];
      PetscScalar *valuesOrig = elMat;
      ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
      idx_size[j] = numindices;
      ierr = PetscMalloc2(numindices,&idx_arr[j],numindices*numindices,&new_el_mats[j]);CHKERRQ(ierr);
      ierr = PetscMemcpy(idx_arr[j],indices,numindices*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(new_el_mats[j],elMat,numindices*numindices*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
      if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);}
    }
    /* assemble matrix */
#pragma omp parallel for shared(JacP,idx_size,idx_arr,new_el_mats,colour,clr_idxs) private(j) schedule(static)
    for (j=0; j<csize; j++) {
      PetscInt numindices = idx_size[j], *indices = idx_arr[j];
      PetscScalar *elMat = new_el_mats[j];
      MatSetValues(JacP,numindices,indices,numindices,indices,elMat,ADD_VALUES);
    }
    /* free */
    ierr = ISRestoreIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    for (j=0; j<csize; j++) {
      ierr = PetscFree2(idx_arr[j],new_el_mats[j]);CHKERRQ(ierr);
    }
  }
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode assemble_cuda_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container, const PetscLogEvent events[])
{
  PetscErrorCode    ierr;
#define FP_MAX_COLORS 16
#define FP_MAX_ELEMS 512
  Mat_SeqAIJ             h_mats[FP_MAX_COLORS], *jaca = (Mat_SeqAIJ *)JacP->data, *d_mats;
  const PetscInt         nelems = cEnd - cStart, nnz = jaca->i[JacP->rmap->n], N = JacP->rmap->n;  /* serial */
  const ISColoringValue *colors;
  ISColoringValue       *d_colors,colour;
  PetscInt              *h_idx_arr[FP_MAX_ELEMS], h_nidx_arr[FP_MAX_ELEMS], *d_nidx_arr, **d_idx_arr,nc,ej,j,cell;
  PetscScalar           *h_new_el_mats[FP_MAX_ELEMS], *val_buf, **d_new_el_mats;
  ISColoring_ctx         coloring_ctx = NULL;
  ISColoring             iscoloring;
  ierr = PetscContainerGetPointer(container,(void**)&coloring_ctx);CHKERRQ(ierr);
  iscoloring = coloring_ctx->coloring;
  /* get colors */
  ierr = ISColoringGetColors(iscoloring, &j, &nc, &colors);CHKERRQ(ierr);
  if (nelems>FP_MAX_ELEMS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements. %D > %D",nelems,FP_MAX_ELEMS);
  if (nc>FP_MAX_COLORS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many colors. %D > %D",nc,FP_MAX_COLORS);
  /* colors for kernel */
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors,         nelems*sizeof(ISColoringValue))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_colors, colors, nelems*sizeof(ISColoringValue), cudaMemcpyHostToDevice));
  /* get indices and element matrices */
  for (cell = cStart, ej = 0 ; cell < cEnd; ++cell, ++ej) {
    PetscInt numindices,*indices;
    PetscScalar *elMat = &elemMats[ej*totDim*totDim];
    PetscScalar *valuesOrig = elMat;
    ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
    h_nidx_arr[ej] = numindices;
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_arr[ej],            numindices*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_idx_arr[ej],   indices, numindices*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_new_el_mats[ej],        numindices*numindices*sizeof(PetscScalar))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_new_el_mats[ej], elMat, numindices*numindices*sizeof(PetscScalar), cudaMemcpyHostToDevice));
    ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
    if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);CHKERRQ(ierr);}
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nidx_arr,                  nelems*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_nidx_arr,    h_nidx_arr,   nelems*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_arr,                   nelems*sizeof(PetscInt*))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_idx_arr,     h_idx_arr,    nelems*sizeof(PetscInt*), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_new_el_mats,               nelems*sizeof(PetscScalar*))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_new_el_mats, h_new_el_mats,nelems*sizeof(PetscScalar*), cudaMemcpyHostToDevice));
  /* make matrix buffers */
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    /* create on GPU and copy to GPU */
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->i,               (N+1)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->i,    jaca->i,   (N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->ilen,            (N)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->ilen, jaca->ilen,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->j,               (nnz)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->j,    jaca->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->a,               (nnz)*sizeof(PetscScalar))); // kernel output
    CUDA_SAFE_CALL(cudaMemset(          a->a, 0,            (nnz)*sizeof(PetscScalar)));
  }
  CUDA_SAFE_CALL(cudaMalloc(&d_mats,         nc*sizeof(Mat_SeqAIJ))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy( d_mats, h_mats, nc*sizeof(Mat_SeqAIJ), cudaMemcpyHostToDevice));
  /* do it */
  assemble_kernel<<<nelems,1>>>(d_nidx_arr, d_idx_arr, d_new_el_mats, d_colors, d_mats);
  CHECK_LAUNCH_ERROR();
  /* cleanup */
  CUDA_SAFE_CALL(cudaFree(d_colors));
  CUDA_SAFE_CALL(cudaFree(d_nidx_arr));
  for (ej = cStart ; ej < nelems; ++ej) {
    CUDA_SAFE_CALL(cudaFree(h_idx_arr[ej]));
    CUDA_SAFE_CALL(cudaFree(h_new_el_mats[ej]));
  }
  CUDA_SAFE_CALL(cudaFree(d_idx_arr));
  CUDA_SAFE_CALL(cudaFree(d_new_el_mats));
  /* copy & add Mat data back to CPU to JacP */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = PetscMalloc1(nnz,&val_buf);CHKERRQ(ierr);
  ierr = PetscMemzero(jaca->a,nnz*sizeof(PetscScalar));CHKERRQ(ierr);
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    CUDA_SAFE_CALL(cudaMemcpy(val_buf, a->a, (nnz)*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    PetscKernelAXPY(jaca->a,1.0,val_buf,nnz);
  }
  ierr = PetscFree(val_buf);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    /* destroy mat */
    CUDA_SAFE_CALL(cudaFree(a->i));
    CUDA_SAFE_CALL(cudaFree(a->ilen));
    CUDA_SAFE_CALL(cudaFree(a->j));
    CUDA_SAFE_CALL(cudaFree(a->a));
  }
  CUDA_SAFE_CALL(cudaFree(d_mats));
  PetscFunctionReturn(0);
}
