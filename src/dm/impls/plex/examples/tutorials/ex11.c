static char help[] = "Runaway electron model with Landau collision operator\n\n";

#include <petsc/private/dmpleximpl.h>
#include <petscts.h>

/* Things to do */
/*    * DEBUG Spitzer */
/*    * DEBUG Coronal model (Dylan) */
/*    * Coronal model numerics */
/*    * Induction equation to compute E */
/*    * Add control of E (Dylan) */
/*    * Add relativistic terms to Landau */

/* data for runaway electron model */
typedef struct REctx_struct {
  PetscErrorCode (*test)(TS, Vec, DM, PetscInt, PetscReal,  LandCtx *, struct REctx_struct *);
  PetscErrorCode (*impuritySrc)(PetscReal, PetscReal *, LandCtx*);
  PetscErrorCode (*electricField)(Vec, Vec, PetscReal *, LandCtx*);
  PetscReal     T_cold;        /* temperature of newly iomnized electrons and impurity ions */
  PetscReal     ion_potential; /* ionization potential of impurity */
  PetscInt      Ne_ion;        /* number of electrons shed in ioization of impurity */
  PetscReal     Ez_initial;
  PetscReal     L;             /* inductance */
  Vec           X_0;
  PetscReal     dp_dt;          /* cached values for diagnostics */
  PetscReal     dp2_dt;
} REctx;

static const PetscReal kev_joul = 6.241506479963235e+15;

#undef __FUNCT__
#define __FUNCT__ "f0_runners"
/* < v, ru > */
/* static void f0_runners(PetscInt dim, PetscInt Nf, PetscInt NfAux, */
/*                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], */
/*                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], */
/*                        PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0) */
/* { */
/*   PetscReal n_e = u[0]; */
/*   if (1) { */
/*     if (x[1] > 3.) { /\* simply a cutoff for REs. v_|| > 3 v(T_e) *\/ */
/*       if (dim==2)  *f0 = n_e * 2.*M_PI*x[0]; */
/*       else         *f0 = n_e; */
/*     } else *f0 = 0; */
/*   } else { */
/*     if (dim==2)  *f0 = n_e * 2.*M_PI*x[0] * x[1]*x[1]*x[1]*0.008; /\* n * r * x^3/5^3 -- TODO do something better *\/ */
/*     else { */
/*       PetscReal v = PetscSqrtReal(PetscSqr(x[0]) + PetscSqr(x[1]) + PetscSqr(x[2])); */
/*       *f0 = n_e * v*v*v*0.008; /\* n * x^3/5^3 -- TODO do something better *\/ */
/*     } */
/*   } */
/* } */

#undef __FUNCT__
#define __FUNCT__ "PrintnRE"
/* static PetscErrorCode PrintnRE(Vec X, PetscInt stepi, PetscReal *anre) */
/* { */
/*   PetscErrorCode    ierr; */
/*   DM                dm,plex; */
/*   PetscScalar       nre[FP_MAX_SPECIES]; */
/*   PetscDS           prob; */
/*   PetscFunctionBegin; */
/*   ierr = VecGetDM(X, &dm);CHKERRQ(ierr); */
/*   if (!dm) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no DM"); */
/*   ierr = DMGetDS(dm, &prob);CHKERRQ(ierr); */
/*   ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr); */
/*   ierr = PetscDSSetObjective(prob, 0, &f0_runners);CHKERRQ(ierr); */
/*   ierr = DMPlexComputeIntegralFEM(plex,X,nre,NULL);CHKERRQ(ierr); */
/*   ierr = DMDestroy(&plex);CHKERRQ(ierr); */
/*   ierr = PetscPrintf(PETSC_COMM_SELF, "%3D) Runaway electrons: %20.13e\n",stepi,nre[0]);CHKERRQ(ierr); */
/*   if (anre) *anre = nre[0]; */
/*   PetscFunctionReturn(0); */
/* } */

/* < v, u*v*q > */
static void f0_jz( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii;
  *f0 = 0;
  if (dim==2) {
    for(ii=0;ii<numConstants;ii++) f0[0] += u[ii] * 2.*M_PI*x[0] * x[1] * constants[ii]; /* n * r * v_|| * q */
  } else {
    for(ii=0;ii<numConstants;ii++) f0[0] += u[ii] * x[2] * constants[ii]; /* n * v_|| * q  */
  }
}
static void f0_0_jz( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii=0;
  if (dim==2) {
    *f0 = u[ii] * 2.*M_PI*x[0] * x[1] * constants[ii]; /* n * r * v_|| * q */
  } else {
    *f0 = u[ii] *                x[2] * constants[ii]; /* n * r * v_|| * q */
  }
}

static void f0_1_jz( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  PetscInt ii=1;
  if (dim==2) {
    *f0 = u[ii] * 2.*M_PI*x[0] * x[1] * constants[ii]; /* n * r * v_|| * q */
  } else {
    *f0 = u[ii] *                x[2] * constants[ii]; /* n * r * v_|| * q */
  }
}

/* static void f0_2_j( PetscInt dim, PetscInt Nf, PetscInt NfAux, */
/*                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], */
/*                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], */
/*                     PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0) */
/* { */
/*   PetscInt ii=2; */
/*   if (dim==2) { */
/*     *f0 = u[ii] * 2.*M_PI*x[0] * x[1] * constants[ii]; /\* n * r * v_|| * q *\/ */
/*   } else { */
/*     *f0 = u[ii] *                x[2] * constants[ii]; /\* n * r * v_|| * q *\/ */
/*   } */
/* } */

/* < v, n_e > */
static void f0_0_n( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = 2.*M_PI*x[0]*u[0];
  else        *f0 =              u[0];
}
static void f0_1_n( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = 2.*M_PI*x[0]*u[1];
  else        *f0 =              u[1];
}
static void f0_2_n( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = 2.*M_PI*x[0]*u[2];
  else        *f0 =              u[2];
}

/* < v, n_e v_|| > */
static void f0_0_vz( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = u[0] * 2.*M_PI*x[0] * x[1]; /* n r v_|| */
  else        *f0 = u[0] *                x[2]; /* n v_|| */
}
/* < v, n_e v_|| ^2 > */
/* static void f0_0_vz2( PetscInt dim, PetscInt Nf, PetscInt NfAux, */
/*                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], */
/*                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], */
/*                      PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0) */
/* { */
/*   if (dim==2) *f0 = u[0] * 2.*M_PI*x[0] * x[1]*x[1]; /\* n r v_||^2 *\/ */
/*   else        *f0 = u[0]                * x[2]*x[2]; /\* n v_||^2 *\/ */
/* } */

/* < v, n_e v > */
static void f0_0_v( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = u[0] * 2.*M_PI*x[0] * PetscSqrtReal(x[0]*x[0] + x[1]*x[1]);             /* n r v */
  else        *f0 = u[0] *                PetscSqrtReal(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); /* n v */
}
/* < v, n_i v > */
static void f0_1_v( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = u[1] * 2.*M_PI*x[0] * PetscSqrtReal(x[0]*x[0] + x[1]*x[1]);             /* n r v */
  else        *f0 = u[1] *                PetscSqrtReal(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); /* n v */
}
/* < v, n_imp v > */
static void f0_2_v( PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[],  PetscInt y, const PetscScalar xx[], PetscScalar *f0)
{
  if (dim==2) *f0 = u[2] * 2.*M_PI*x[0] * PetscSqrtReal(x[0]*x[0] + x[1]*x[1]);             /* n r v */
  else        *f0 = u[2] *                PetscSqrtReal(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); /* n v */
}

static PetscErrorCode getT_kev(DM plex, Vec X, PetscInt idx, PetscReal *a_n, PetscReal *a_Tkev)
{
  PetscErrorCode ierr;
  PetscDS        prob;
  LandCtx        *ctx;
  PetscReal      tt[FP_MAX_SPECIES],v2, v, n, mass;
  PetscFunctionBeginUser;
  ierr = DMGetApplicationContext(plex, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  mass = ctx->masses[idx];
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  switch (idx) {
  case 0:
    ierr = PetscDSSetObjective(prob, 0, &f0_0_n);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    n = ctx->n_0*tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_0_v);CHKERRQ(ierr); break;
  case 1:
    ierr = PetscDSSetObjective(prob, 0, &f0_1_n);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    n = ctx->n_0*tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_1_v);CHKERRQ(ierr); break;
  case 2:
    ierr = PetscDSSetObjective(prob, 0, &f0_2_n);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    n = ctx->n_0*tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_2_v);CHKERRQ(ierr); break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "case %D not supported",idx);
  }
  ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
  v = ctx->n_0*ctx->v_0*tt[0]/n;         /* remove number density to get velocity */
  v2 = PetscSqr(v);                      /* use real space: m^2 / s^2 */
  if (a_Tkev) *a_Tkev = (v2*mass*M_PI/8)*kev_joul; /* temperature in kev */
  if (a_n) *a_n = n;
  PetscFunctionReturn(0);
}
/* CalculateEForCurrent  Calculate the electric field required to reach the */
 /*                   given current density. */

 /* j0   -- Desired plasma current density */
 /* T    -- Electron temperature */
 /* n    -- Electron density */
static PetscReal CalculateEForCurrent(PetscReal j0, PetscReal Tev, PetscReal n, PetscReal lnLambda, PetscReal eps0, PetscReal *E)
{
  PetscReal         c,e,m;
  PetscFunctionBegin;
  c = 299792458;
  e = 1.602176e-19;
  m = 9.10938e-31;
  if (0) {
    PetscReal betath,Ec,EHat;
    betath = sqrt(2*Tev*e/(m*c*c));
    EHat = (j0 / (n*e*c)) / (7/(sqrt(2)*2)*pow(betath,3));
    Ec =  n*lnLambda*pow(e,3) / (4*M_PI*pow(eps0,2)*m*c*c);
    *E = EHat*Ec;
  } else {
    PetscReal Ed,vth;
    vth = PetscSqrtReal(8*Tev*e/(m*M_PI));
    Ed =  n*lnLambda*pow(e,3) / (4*M_PI*pow(eps0,2)*m*vth*vth);
    *E = Ed;
  }
  PetscFunctionReturn(0);
}

static PetscReal Spitzer(PetscReal m_e, PetscReal e, PetscReal Z, PetscReal epsilon0,  PetscReal lnLam, PetscReal kTe_joules)
{
  PetscReal Fz = (1+1.198*Z+0.222*Z*Z)/(1+2.966*Z+0.753*Z*Z), eta;
  eta = Fz*4./3.*PetscSqrtReal(2.*M_PI)*Z*PetscSqrtReal(m_e)*PetscSqr(e)*lnLam*pow(4*M_PI*epsilon0,-2.)*pow(kTe_joules,-1.5);
  /* PetscPrintf(PETSC_COMM_SELF, "Fz=%20.13e SpitzEr=%10.3e Z=%g\n",Fz,eta,Z); */
  return eta;
}

/*  */
static PetscErrorCode testNone(TS ts, Vec X, DM plex, PetscInt stepi, PetscReal time, LandCtx *ctx, REctx *rectx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*  */
static PetscErrorCode testSpitzer(TS ts, Vec X, DM plex, PetscInt stepi, PetscReal time, LandCtx *ctx, REctx *rectx)
{
  PetscErrorCode    ierr;
  PetscDS           prob;
  PetscScalar       J,tt[FP_MAX_SPECIES];
  static PetscReal  old_ratio = 0;
  PetscBool         done=PETSC_FALSE;
  PetscReal         spit_eta,Te_kev,E,ratio,Z;
  PetscFunctionBegin;
  if (ctx->num_species!=2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "ctx->num_species!=2");
  Z = -ctx->charges[1]/ctx->charges[0];
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_jz);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
  J = -ctx->n_0*ctx->v_0*tt[0];
  ierr = getT_kev(plex, X, 0, NULL, &Te_kev);CHKERRQ(ierr);
  spit_eta = Spitzer(ctx->masses[0],-ctx->charges[0],Z,ctx->epsilon0,ctx->lnLam,Te_kev/kev_joul); /* kev --> J (kT) */
  E = ctx->Ez; /* keep real E */
  ratio = E/J/spit_eta;
  done = (old_ratio-ratio < 1.e-4 && stepi>20 &&0);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s %D) time=%10.3e J= %10.3e E/J= %10.3e Spitzer_eta=%10.3e T_e(t)=%10.3e T_e(0)=%10.3e (kev). E/J to eta ratio=%g (diff=%g)\n",
                     done ? "DONE" : "----",stepi,time,J,E/J,spit_eta,Te_kev,ctx->thermal_temps[0]*ctx->k*kev_joul,ratio,old_ratio-ratio);CHKERRQ(ierr);
  if (done) {
    ierr = TSSetConvergedReason(ts,TS_CONVERGED_USER);CHKERRQ(ierr);
    old_ratio = 0;
  } else {
    TSConvergedReason reason;
    ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
    old_ratio = ratio;
    if (reason) done = PETSC_TRUE;
  }
  if (done) { /* test integration */
    PetscReal Te_kev, n_e, v, v_z, v2, tt[FP_MAX_SPECIES], j_0, j_1 = 0;
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_0_n);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    n_e = ctx->n_0*tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_0_v);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    v = ctx->n_0*ctx->v_0*tt[0]/n_e;
    ierr = PetscDSSetObjective(prob, 0, &f0_0_vz);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    v_z = ctx->n_0*ctx->v_0*tt[0]/n_e;
    ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_0_jz);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    j_0 = -ctx->n_0*ctx->v_0*tt[0];
    if (ctx->num_species>1) {
      ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr);
      ierr = PetscDSSetObjective(prob, 0, &f0_1_jz);CHKERRQ(ierr);
      ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
      j_1 = -ctx->n_0*ctx->v_0*tt[0];
    }
    /* ierr = getT_kev(plex, X, 0, &n_e, &Te_kev);CHKERRQ(ierr); */
    v2 = v*v;
    Te_kev = (v2*ctx->masses[0]*M_PI/8)*kev_joul; /* temperature in kev */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "ooooo T_e(kev)=%20.13e J_0=%10.3e J_1=%10.3e n_e=%10.3e v_z=%10.3e eta_s=%10.3e E/J=%10.3e Z= %d\n",
                       Te_kev, j_0, j_1, n_e, v_z, spit_eta, E/J, (int)Z);CHKERRQ(ierr);
  }
  if (time==0) {
    PetscReal E, j0 = 1.e6, Tev = ctx->thermal_temps[0]*8.621738e-5, n = ctx->n_0*ctx->n[0];
    CalculateEForCurrent(j0, Tev, n, ctx->lnLam, ctx->epsilon0, &E);
    ctx->Ez *= E;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "***** new E=%10.3e for j0=%10.3e -- scaling Ez by %g\n",ctx->Ez,j0,E);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static const double ppp = 2;
static void f0_0_diff_lp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  LandCtx        *ctx = (LandCtx *)constants;
  REctx          *rectx = rectx = (REctx*)ctx->data;
  PetscInt        ii = rectx->Ne_ion, i;
  const PetscReal kT_m = ctx->k*ctx->thermal_temps[ii]/ctx->masses[ii]; /* kT/m */
  const PetscReal n = ctx->n[ii];
  PetscReal       diff, f_maxwell, v2 = 0, theta = 2*kT_m/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  f_maxwell = n*pow(M_PI*theta,-1.5)*(exp(-v2/theta));
  diff = 2.*M_PI*x[0]*(u[ii] - f_maxwell);
  f0[0] = pow(diff,ppp);
}
static void f0_0_maxwellian_lp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  LandCtx        *ctx = (LandCtx *)constants;
  REctx          *rectx = rectx = (REctx*)ctx->data;
  PetscInt        ii = rectx->Ne_ion, i;
  const PetscReal kT_m = ctx->k*ctx->thermal_temps[ii]/ctx->masses[ii]; /* kT/m */
  const PetscReal n = ctx->n[ii];
  PetscReal       f_maxwell, v2 = 0, theta = 2*kT_m/(ctx->v_0*ctx->v_0); /* theta = 2kT/mc^2 */
  for (i = 0; i < dim; ++i) v2 += x[i]*x[i];
  f_maxwell = 2.*M_PI*x[0] * n*pow(M_PI*theta,-1.5)*(exp(-v2/theta));
  f0[0] = pow(f_maxwell,ppp);
}

/*  */
static PetscErrorCode testStable(TS ts, Vec X, DM plex, PetscInt stepi, PetscReal time, LandCtx *ctx, REctx *rectx)
{
  PetscErrorCode    ierr;
  PetscDS           prob;
  Vec               X2;
  PetscReal         ediff,idiff=0,tt[FP_MAX_SPECIES],lpm0,lpm1=1;
  DM                dm;
  PetscFunctionBegin;
  ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X2);CHKERRQ(ierr);
  ierr = VecCopy(X,X2);CHKERRQ(ierr);
  ierr = VecAXPY(X,-1.0,rectx->X_0);CHKERRQ(ierr);
  ierr = PetscDSSetConstants(prob, sizeof(LandCtx)/sizeof(PetscScalar), (PetscScalar*)ctx);CHKERRQ(ierr);
  rectx->Ne_ion = 0;
  ierr = PetscDSSetObjective(prob, 0, &f0_0_diff_lp);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X2,tt,NULL);CHKERRQ(ierr);
  ediff = pow(tt[0],1./ppp);
  ierr = PetscDSSetObjective(prob, 0, &f0_0_maxwellian_lp);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X2,tt,NULL);CHKERRQ(ierr);
  lpm0 = pow(tt[0],1./ppp);
  if (ctx->num_species>1) {
    rectx->Ne_ion = 1;
    ierr = PetscDSSetObjective(prob, 0, &f0_0_diff_lp);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X2,tt,NULL);CHKERRQ(ierr);
    idiff = pow(tt[0],1./ppp);
    ierr = PetscDSSetObjective(prob, 0, &f0_0_maxwellian_lp);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X2,tt,NULL);CHKERRQ(ierr);
    lpm1 = pow(tt[0],1./ppp);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s %D) time=%10.3e n-%d norm electrons/max=%20.13e ions/max=%20.13e\n", "----",stepi,time,(int)ppp,ediff/lpm0,idiff/lpm1);CHKERRQ(ierr);
  /* view */
  ierr = DMSetOutputSequenceNumber(dm, ((time/ctx->plotDt - (PetscReal)ctx->plotIdx) >= 0.0) ? ctx->plotIdx : ctx->plotIdx + 1, time*ctx->t_0);CHKERRQ(ierr);
  ierr = VecViewFromOptions(X,NULL,"-vec_view_diagnostics");CHKERRQ(ierr);
  ierr = VecCopy(X2,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*  */
static PetscErrorCode testShift(TS ts, Vec X, DM plex, PetscInt stepi, PetscReal time, LandCtx *ctx, REctx *rectx)
{
  PetscErrorCode    ierr;
  PetscDS           prob;
  PetscReal Te_kev, n_e, v, v_z, tt[FP_MAX_SPECIES], j_0, j_1 = 0, Z = -ctx->charges[1]/ctx->charges[0];
  PetscFunctionBegin;
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_0_n);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
  n_e = ctx->n_0*tt[0];
  ierr = PetscDSSetObjective(prob, 0, &f0_0_v);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
  v = ctx->n_0*ctx->v_0*tt[0]/n_e;
  ierr = PetscDSSetObjective(prob, 0, &f0_0_vz);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
  v_z = ctx->n_0*ctx->v_0*tt[0]/n_e;
  ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_0_jz);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
  j_0 = -ctx->n_0*ctx->v_0*tt[0];
  if (ctx->num_species>1) {
    ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_1_jz);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    j_1 = -ctx->n_0*ctx->v_0*tt[0];
  }
  Te_kev = (v*v*ctx->masses[0]*M_PI/8)*kev_joul; /* temperature in kev */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "++++++ T_e(kev)=%20.13e J_0=%10.3e J_1=%10.3e n_e=%10.3e v_z=%10.3e (v_0) Z= %d\n",Te_kev, j_0, j_1, n_e, v_z/ctx->v_0, (int)Z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PostStep"
static PetscErrorCode PostStep(TS ts)
{
  PetscErrorCode    ierr;
  PetscInt          stepi;
  Vec               X;
  DM                dm,plex;
  PetscDS           prob;
  PetscReal         time;
  LandCtx           *ctx;
  REctx            *rectx;
  TSConvergedReason reason;
  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts, &stepi);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(ts, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if ((time/ctx->plotDt - (PetscReal)ctx->plotIdx) >= 0.0 || reason) {
    rectx = (REctx*)ctx->data;
    ierr = TSGetSolution(ts, &X);CHKERRQ(ierr);
    ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
    /* print norms */
    ierr = DMPlexFPPrintNorms(X, stepi);CHKERRQ(ierr);
    ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    /* diagnostics */
    ierr = rectx->test(ts,X,plex,stepi,time,ctx,rectx);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
    /* view */
    ierr = DMSetOutputSequenceNumber(dm, ((time/ctx->plotDt - (PetscReal)ctx->plotIdx) >= 0.0) ? ctx->plotIdx : ctx->plotIdx + 1, time*ctx->t_0);CHKERRQ(ierr);
    ierr = VecViewFromOptions(X,NULL,"-vec_view");CHKERRQ(ierr);
    ctx->plotIdx++;
  }
  if (reason) {
    PetscReal    val,rval;
    PetscMPIInt    rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    ierr = TSGetSolution(ts, &X);CHKERRQ(ierr);
    ierr = VecNorm(X,NORM_2,&val);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&val,&rval,1,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
    if (rval != val) {
      PetscPrintf(PETSC_COMM_SELF, " ***** [%D] ERROR max |x| = %e, my |x| = %20.13e\n",rank,rval,val);CHKERRQ(ierr);
    } else {
      PetscPrintf(PETSC_COMM_SELF, "[%D] parallel consistency check OK\n",rank);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* model for source of ionized impurities, profile provided by model */
/* E = eta_spitzer(J-J_re) */
/* static PetscErrorCode ESpitzer(Vec X,  Vec X_t, PetscReal *a_E, LandCtx *ctx) */
/* { */
/*   PetscErrorCode    ierr; */
/*   PetscReal         spit_eta,Te_kev,J,J_re=0,tt[FP_MAX_SPECIES]; */
/*   PetscDS           prob; */
/*   DM                dm,plex; */
/*   PetscFunctionBegin; */
/*   ierr = VecGetDM(X, &dm);CHKERRQ(ierr); */
/*   ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr); */
/*   ierr = DMGetDS(plex, &prob);CHKERRQ(ierr); */
/*   ierr = getT_kev(plex, X, 0, NULL, &Te_kev);CHKERRQ(ierr); */
/*   spit_eta = Spitzer(ctx->masses[0],-ctx->charges[0],-ctx->charges[1]/ctx->charges[0],ctx->epsilon0,ctx->lnLam,Te_kev/kev_joul); /\* kev --> J (kT) *\/ */
/*   /\* J *\/ */
/*   ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr); */
/*   ierr = PetscDSSetObjective(prob, 0, &f0_j);CHKERRQ(ierr); */
/*   ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr); */
/*   J = -ctx->n_0*ctx->v_0*tt[0]; */
/*   /\* E = eta_spitzer(J-J_re) *\/ */
/*   *a_E = spit_eta*(J-J_re); */
/*   /\* cleanup *\/ */
/*   ierr = DMDestroy(&plex);CHKERRQ(ierr); */
/*   PetscFunctionReturn(0); */
/* } */

/* #undef __FUNCT__ */
/* #define __FUNCT__ "EInduction" */
/* static PetscErrorCode EInduction(Vec X, Vec X_t, PetscReal *a_E, LandCtx *ctx) */
/* { */
/*   REctx            *rectx = (REctx*)ctx->data; */
/*   PetscErrorCode    ierr; */
/*   DM                dm,plex; */
/*   PetscScalar       dJ_dt,tt[FP_MAX_SPECIES]; */
/*   PetscDS           prob; */
/*   PetscFunctionBegin; */
/*   ierr = VecGetDM(X, &dm);CHKERRQ(ierr); */
/*   ierr = DMGetDS(dm, &prob);CHKERRQ(ierr); */
/*   ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr); */
/*   /\* get d current / dt *\/ */
/*   ierr = PetscDSSetConstants(prob, ctx->num_species, ctx->charges);CHKERRQ(ierr); */
/*   ierr = PetscDSSetObjective(prob, 0, &f0_j);CHKERRQ(ierr); */
/*   ierr = DMPlexComputeIntegralFEM(plex,X_t,tt,NULL);CHKERRQ(ierr); */
/*   dJ_dt = -ctx->n_0*ctx->v_0*tt[0]/ctx->t_0; */
/*   /\* E induction *\/ */
/*   *a_E = -rectx->L*dJ_dt + rectx->Ez_initial; */
/*   ierr = DMDestroy(&plex);CHKERRQ(ierr); */
/*   PetscFunctionReturn(0); */
/* } */

static PetscErrorCode EConst(Vec X,  Vec X_t, PetscReal *a_E, LandCtx *ctx)
{
  PetscFunctionBegin;
  *a_E = ctx->Ez;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "REIFunction"
PetscErrorCode REIFunction(TS ts,PetscReal time,Vec X,Vec X_t,Vec F,void *actx)
{
  PetscErrorCode ierr;
  LandCtx        *ctx;
  DM             dm,plex;
  PetscReal      n_imp_new;
  REctx         *rectx;
  PetscFunctionBeginUser;
  /* check seed RE run */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (!ctx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no context");
  rectx = (REctx*)ctx->data;
  if (!rectx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "no re context");
  /* update E */
  ierr = rectx->electricField(X, X_t, &ctx->Ez, ctx);CHKERRQ(ierr);
  /* Add Landau part */
  ierr = FPLandIFunction(ts,time,X,X_t,F,actx);CHKERRQ(ierr);
  ctx->aux_bool = PETSC_FALSE;
  /* check for impurities */
  ierr = rectx->impuritySrc(time,&n_imp_new,ctx);CHKERRQ(ierr);
  if (n_imp_new != 0 && ctx->num_species==3) {
    PetscInt       dim;
    PetscReal      Le,Limp,n_imp,n_e,Te_kev,tilda_ns[3],temps[3],tt[FP_MAX_SPECIES]; /* hardwired for e, ions, impurity ions */
    Vec            F_src;
    PetscDS        prob;
    if (ctx->num_species!=3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "impurity hard wired for 3 species (%D)",ctx->num_species);
    ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    /* Add L(f_e) & s */
    ierr = getT_kev(plex, X, 0, &n_e, &Te_kev);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_2_n);CHKERRQ(ierr);/* could put this in getT_kev */
    ierr = DMPlexComputeIntegralFEM(plex,X,tt,NULL);CHKERRQ(ierr);
    n_imp = ctx->n_0*tt[0];
    /* corona model for rate of electron production from ionization - Aleynikov & Briezman */
    Le = n_imp * n_e * M_PI * pow(ctx->charges[0],4) / pow(4*M_PI*ctx->epsilon0,2) / rectx->ion_potential * PetscSqrtReal(8/(M_PI*ctx->m_0*Te_kev)); /* what is m_0 ? */
    ierr = PetscInfo5(dm, "***** Ionization model: T_e(kev)=%10.3e, n_e=%10.3e, n_imp=%10.3e. n_imp new %10.3, electron src %g\n",Te_kev,n_e,n_imp,n_imp_new,Le);CHKERRQ(ierr);
    Limp = Le/rectx->Ne_ion; /* Do we need to keep track of the unionized impurity? */
    ierr = VecDuplicate(F,&F_src);CHKERRQ(ierr);
    ierr = VecZeroEntries(F_src);CHKERRQ(ierr);
    tilda_ns[0] = Le/ctx->n_0;   tilda_ns[1] = 0; tilda_ns[2] = Limp/ctx->n_0;
    temps[0]    = rectx->T_cold; temps[1]    = 0;    temps[2] = rectx->T_cold;
    /* add it */
    ierr = DMPLexFPAddMaxwellians(dm,F_src,time,temps,tilda_ns,ctx);CHKERRQ(ierr);
    ierr = VecAXPY(F,1,F_src);CHKERRQ(ierr);
    /* clean up */
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
    ierr = VecDestroy(&F_src);CHKERRQ(ierr);
  }
  if (ctx->electronShift!=0) {
    PetscDS        prob;
    PetscReal      tt[FP_MAX_SPECIES];
    ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_0_vz);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(plex,X_t,tt,NULL);CHKERRQ(ierr);
    rectx->dp_dt = ctx->n_0*ctx->v_0*tt[0]*ctx->masses[0]/ctx->t_0; /* cache electron dp/dt */
    /* ierr = PetscDSSetObjective(prob, 0, &f0_0_vz2);CHKERRQ(ierr); */
    /* ierr = DMPlexComputeIntegralFEM(plex,X_t,tt,NULL);CHKERRQ(ierr); */
    /* rectx->dp2_dt = ctx->n_0*ctx->v_0*ctx->v_0*tt[0]*ctx->masses[0]; */ /* cache electron dp^2/dt */
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "REIJacobian"
PetscErrorCode REIJacobian(TS ts,PetscReal time,Vec X,Vec U_t,PetscReal shift,Mat Amat,Mat Pmat,void *actx)
{
  PetscErrorCode ierr;
  LandCtx        *ctx;
  DM             dm;
  PetscFunctionBeginUser;
  /* Add Landau part */
  ierr = FPLandIJacobian(ts,time,X,U_t,shift,Amat,Pmat,actx);CHKERRQ(ierr);
  /* check for noop */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  if (ctx->aux_bool) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Does this ever make Landau Jacobian here");
  PetscFunctionReturn(0);
}

/* model for source of non-ionized impurities, profile provided by model */
static PetscErrorCode stepSrc(PetscReal time, PetscReal *rho, LandCtx *ctx)
{
  PetscFunctionBegin;
  if (time > 1.e+5) *rho = 1.; /* turned off */
  else *rho = 0.;
  PetscFunctionReturn(0);
}
static PetscErrorCode zeroSrc(PetscReal time, PetscReal *rho, LandCtx *ctx)
{
  PetscFunctionBegin;
  *rho = 0.;
  PetscFunctionReturn(0);
}
static PetscErrorCode pulseSrc(PetscReal time, PetscReal *rho, LandCtx *ctx)
{
  static int step = 0;
  PetscFunctionBegin;
  *rho = 0;
  if (step++==3) *rho = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessREOptions"
static PetscErrorCode ProcessREOptions(REctx *rectx, const LandCtx *ctx, DM dm, const char prefix[])
{
  PetscErrorCode    ierr;
  PetscFunctionList plist = NULL, testlist = NULL;
  char              pname[256],testname[256];
  DM                dummy;
  PetscFunctionBeginUser;
  ierr = DMCreate(PETSC_COMM_WORLD,&dummy);CHKERRQ(ierr);
  rectx->Ne_ion = 2; /* number of electrons given up by imourity ion */
  rectx->T_cold = .005; /* kev - units? */
  rectx->ion_potential = 15; /* ev - units? */
  rectx->Ez_initial = ctx->Ez;     /* cache for induction caclulation - applied E field */
  rectx->L = 2;
  rectx->dp_dt = 0;
  rectx->dp2_dt = 0;
  /* Register the available impurity sources */
  ierr = PetscFunctionListAdd(&plist,"step",&stepSrc);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&plist,"none",&zeroSrc);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&plist,"pulse",&pulseSrc);CHKERRQ(ierr);
  ierr = PetscStrcpy(pname,"none");CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&testlist,"none",&testNone);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&testlist,"spitzer",&testSpitzer);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&testlist,"stable",&testStable);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&testlist,"bimaxwellian",&testShift);CHKERRQ(ierr);
  ierr = PetscStrcpy(testname,"none");CHKERRQ(ierr);
  /* electric field function - can switch at runtime */
  rectx->electricField = EConst;
  ierr = PetscOptionsBegin(PETSC_COMM_SELF, prefix, "Options for Runaway/seed electron model", "none");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-impurity_source_type","Name of impurity source to run","",plist,pname,pname,sizeof(pname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-test_type","Name of test to run","",testlist,testname,testname,sizeof(pname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n_imp_e", "number of electrons donated by ionization", "none", rectx->Ne_ion, &rectx->Ne_ion, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t_cold","Temperature of cold electron and ions after ionization in keV","none",rectx->T_cold,&rectx->T_cold, NULL);CHKERRQ(ierr);
  if (ctx->electronShift==0 && !strcmp(pname,"bimaxwellian") ) PetscPrintf(PETSC_COMM_WORLD, "Warning -electron_shift 0 and 'bimaxwellian' test -- rates will not be cached\n");
  rectx->T_cold *= 1.16e7; /* convert to Kelvin */
  ierr = PetscOptionsReal("-ion_potential","Potential to ionize impurity (should be array) in ev","none",rectx->ion_potential,&rectx->ion_potential, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-inductance","","none",rectx->L,&rectx->L, NULL);CHKERRQ(ierr);
  ierr = PetscInfo5(dummy, "Num electrons from ions=%D, T_cold=%10.3e, ion potential=%10.3e, E_z=%10.3e v_0=%10.3e\n",rectx->Ne_ion,rectx->T_cold,rectx->ion_potential,ctx->Ez,ctx->v_0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* get impurity source function */
  ierr = PetscFunctionListFind(plist,pname,&rectx->impuritySrc);CHKERRQ(ierr);
  if (!rectx->impuritySrc) SETERRQ1(PETSC_COMM_SELF,1,"No impurity source function found '%s'",pname);
  ierr = PetscFunctionListFind(testlist,testname,&rectx->test);CHKERRQ(ierr);
  if (!rectx->test) SETERRQ1(PETSC_COMM_SELF,1,"No impurity source function found '%s'",testname);
  ierr = PetscFunctionListDestroy(&plist);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&testlist);CHKERRQ(ierr);
  {
    PetscMPIInt    rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    if (rank) { /* turn off output stuff for duplicate runs */
      ierr = PetscOptionsClearValue(NULL,"-dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-vec_view");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-dm_view_diagnostics");CHKERRQ(ierr);
      ierr = PetscOptionsClearValue(NULL,"-vec_view_diagnostics");CHKERRQ(ierr);
    }
  }
  ierr = DMDestroy(&dummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  Vec            X;
  PetscErrorCode ierr;
  PetscInt       dim = 2, numSpecies;
  TS             ts;
  Mat            J;
  PetscDS        prob;
  LandCtx        *ctx;
  REctx         *rectx;
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
  /* Create a mesh */
  numSpecies = 3;
  ierr = DMPlexFPCreateVelocitySpace(PETSC_COMM_SELF, dim, numSpecies, "", &X, &dm); CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &numSpecies);CHKERRQ(ierr);
  /* context */
  rectx = (REctx*)(ctx->data = malloc(sizeof(REctx)));
  ierr = ProcessREOptions(rectx,ctx,dm,"");CHKERRQ(ierr);
  ierr = VecDuplicate(X,&rectx->X_0);CHKERRQ(ierr);
  ierr = VecCopy(X,rectx->X_0);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, 0.0);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-dm_view_diagnostics");CHKERRQ(ierr);
  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,REIFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,REIJacobian,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts, ctx);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts, PostStep);CHKERRQ(ierr);
  ierr = PostStep(ts);CHKERRQ(ierr);
  /* go */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  /* clean up */
  ierr = VecDestroy(&rectx->X_0);CHKERRQ(ierr);
  free(rectx);
  ierr = DMPlexFPDestroyPhaseSpace(&dm);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: p4est
    args: -Ez 0 -petscspace_degree 4 -mass_petscspace_degree 4 -petscspace_poly_tensor 1 -mass_petscspace_poly_tensor 1 -dm_type p4est -info -info_exclude null,vec,mat,pc,ksp,snes,p4est,vecscatter,is -ion_masses 2 -ion_charges 1 -thermal_temps 5,5 -n 2,2 -n_0 5e19 -ts_monitor -snes_rtol 1.e-12 -snes_stol 1.e-12 -snes_monitor -snes_converged_reason -snes_max_it 10 -ts_type arkimex -ts_arkimex_type 1bee -ts_max_snes_failures -1 -ts_rtol 1e-6 -ts_dt 1.e-4 -ts_max_time 1 -ts_adapt_clip .5,1.25 -ts_max_steps 2 -ts_adapt_scale_solve_failed -ts_adapt_time_step_increase_delay 5 -pc_type lu -ksp_type preonly -amr_levels_max 11 -domain_radius 5 -impurity_source_type pulse -malloc_debug

TEST*/
