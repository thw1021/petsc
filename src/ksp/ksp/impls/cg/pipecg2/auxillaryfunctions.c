/* This file contains the auxillary functions to be used by pipecg2.c. 
   Merged vector operations that load vectors from memory once and use
   the data multiple times by performing vector operations element-wise.
*/
/*   VecMergedDot function merges the dot products for gamma, delta and dp */
PetscErrorCode VecMergedDot(Vec U,Vec W,Vec R,PetscInt normtype,PetscScalar *ru, PetscScalar *wu, PetscScalar *uu)
{ 
  const PetscScalar *PETSC_RESTRICT PU, *PETSC_RESTRICT PW, *PETSC_RESTRICT PR;
  PetscScalar       sumru = 0.0, sumwu = 0.0, sumuu = 0.0;
  PetscInt          j, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,(const PetscScalar**)&PU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(W,(const PetscScalar**)&PW);CHKERRQ(ierr);
  ierr = VecGetArrayRead(R,(const PetscScalar**)&PR);CHKERRQ(ierr);
  ierr = VecGetLocalSize(U,&n);CHKERRQ(ierr);

  if(normtype==KSP_NORM_PRECONDITIONED) {
    #pragma _CRI ivdep
    for(j=0; j<n; j++) {
      sumwu += PU[j] * PW[j];
      sumru += PU[j] * PR[j];
      sumuu += PU[j] * PU[j];
      }
  } else if(normtype==KSP_NORM_UNPRECONDITIONED) {
      #pragma _CRI ivdep
      for(j=0; j<n; j++) {
        sumwu += PU[j] * PW[j];
        sumru += PU[j] * PR[j];
        sumuu += PR[j] * PR[j];
      }
  } else if(normtype==KSP_NORM_NATURAL) {
      #pragma _CRI ivdep
      for(j=0; j<n; j++) {
        sumwu += PU[j] * PW[j];
        sumru += PU[j] * PR[j];
      }
      sumuu = sumru;
  }

  *ru = sumru;
  *wu = sumwu;
  *uu = sumuu;

  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&PU);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(W,(const PetscScalar**)&PW);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(R,(const PetscScalar**)&PR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*   VecMergedDot2 function merges the dot products for lambda_1 and lambda_4 */
PetscErrorCode VecMergedDot2(Vec N,Vec M,Vec W,PetscScalar *wm, PetscScalar *nm)
{
  const PetscScalar *PETSC_RESTRICT PN, *PETSC_RESTRICT PM, *PETSC_RESTRICT PW;
  PetscScalar       sumwm = 0.0, sumnm = 0.0;
  PetscInt          j, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(W,(const PetscScalar**)&PW);CHKERRQ(ierr);
  ierr = VecGetArrayRead(N,(const PetscScalar**)&PN);CHKERRQ(ierr);
  ierr = VecGetArrayRead(M,(const PetscScalar**)&PM);CHKERRQ(ierr);  
  ierr = VecGetLocalSize(N,&n);CHKERRQ(ierr);
   
  #pragma _CRI ivdep
  for(j=0; j<n; j++){
    sumwm += PM[j] * PW[j];
    sumnm += PN[j] * PM[j];
  }
 
  *wm = sumwm;
  *nm = sumnm;

  ierr = VecRestoreArrayRead(W,(const PetscScalar**)&PW);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(N,(const PetscScalar**)&PN);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(M,(const PetscScalar**)&PM);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/*   VecMergedOpsShort function merges the dot products, AXPY and SAXPY operations for all vectors for iteration 0  */
PetscErrorCode VecMergedOpsShort(Vec vx,Vec vr,Vec vz,Vec vw,Vec vp,Vec vq,Vec vc, Vec vd,Vec vg0,Vec vh0,Vec vg1,Vec vh1,//
		                 Vec vs, Vec va1, Vec vb1, Vec ve,Vec vf,Vec vm,Vec vn, Vec vu, PetscInt normtype,PetscScalar beta0,//
				 PetscScalar alpha0, PetscScalar beta1, PetscScalar alpha1, PetscScalar *lambda)
{ 
  PetscScalar       *PETSC_RESTRICT px, *PETSC_RESTRICT pr, *PETSC_RESTRICT pz, *PETSC_RESTRICT pw;
  PetscScalar       *PETSC_RESTRICT pp, *PETSC_RESTRICT pq;
  PetscScalar       *PETSC_RESTRICT pc, *PETSC_RESTRICT pd, *PETSC_RESTRICT pg0, *PETSC_RESTRICT ph0,//
		    *PETSC_RESTRICT pg1,*PETSC_RESTRICT ph1,*PETSC_RESTRICT ps,*PETSC_RESTRICT pa1,*PETSC_RESTRICT pb1,//
		    *PETSC_RESTRICT pe,*PETSC_RESTRICT pf,*PETSC_RESTRICT pm,*PETSC_RESTRICT pn, *PETSC_RESTRICT pu;
  PetscInt          j, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(vx,(PetscScalar**)&px);CHKERRQ(ierr);
  ierr = VecGetArray(vr,(PetscScalar**)&pr);CHKERRQ(ierr);
  ierr = VecGetArray(vz,(PetscScalar**)&pz);CHKERRQ(ierr);
  ierr = VecGetArray(vw,(PetscScalar**)&pw);CHKERRQ(ierr);
  ierr = VecGetArray(vp,(PetscScalar**)&pp);CHKERRQ(ierr);
  ierr = VecGetArray(vq,(PetscScalar**)&pq);CHKERRQ(ierr);
  ierr = VecGetArray(vc,(PetscScalar**)&pc);CHKERRQ(ierr);
  ierr = VecGetArray(vd,(PetscScalar**)&pd);CHKERRQ(ierr);
  ierr = VecGetArray(vg0,(PetscScalar**)&pg0);CHKERRQ(ierr);
  ierr = VecGetArray(vh0,(PetscScalar**)&ph0);CHKERRQ(ierr);
  ierr = VecGetArray(vg1,(PetscScalar**)&pg1);CHKERRQ(ierr);
  ierr = VecGetArray(vh1,(PetscScalar**)&ph1);CHKERRQ(ierr);
  ierr = VecGetArray(vs,(PetscScalar**)&ps);CHKERRQ(ierr);
  ierr = VecGetArray(va1,(PetscScalar**)&pa1);CHKERRQ(ierr);
  ierr = VecGetArray(vb1,(PetscScalar**)&pb1);CHKERRQ(ierr);
  ierr = VecGetArray(ve,(PetscScalar**)&pe);CHKERRQ(ierr);
  ierr = VecGetArray(vf,(PetscScalar**)&pf);CHKERRQ(ierr);
  ierr = VecGetArray(vm,(PetscScalar**)&pm);CHKERRQ(ierr);
  ierr = VecGetArray(vn,(PetscScalar**)&pn);CHKERRQ(ierr);
  ierr = VecGetArray(vu,(PetscScalar**)&pu);CHKERRQ(ierr);

  ierr = VecGetLocalSize(vx,&n);CHKERRQ(ierr);
  for(j=0; j<10; j++) lambda[j] = 0.0;
  
  if(normtype==KSP_NORM_PRECONDITIONED) {
    #pragma _CRI ivdep
    for(j=0; j<n; j++) {
      pz[j] = pn[j];
      pq[j] = pm[j];
      ps[j] = pw[j];
      pp[j] = pu[j];
      pc[j] = pg0[j];
      pd[j] = ph0[j];
      pa1[j] = pe[j];
      pb1[j] = pf[j];
     
      px[j] = px[j] + alpha0 * pp[j];
      pr[j] = pr[j] - alpha0 * ps[j];
      pu[j] = pu[j] - alpha0 * pq[j];
      pw[j] = pw[j] - alpha0 * pz[j];
      pm[j] = pm[j] - alpha0 * pc[j];
      pn[j] = pn[j] - alpha0 * pd[j];
      pg0[j] = pg0[j] - alpha0 * pa1[j];
      ph0[j] = ph0[j] - alpha0 * pb1[j];

      pg1[j] = pg0[j];
      ph1[j] = ph0[j];

      pz[j] = pn[j] + beta1 * pz[j];
      pq[j] = pm[j] + beta1 * pq[j];
      ps[j] = pw[j] + beta1 * ps[j];
      pp[j] = pu[j] + beta1 * pp[j];
      pc[j] = pg0[j] + beta1 * pc[j];
      pd[j] = ph0[j] + beta1 * pd[j];
      
      px[j] = px[j] + alpha1 * pp[j];
      pr[j] = pr[j] - alpha1 * ps[j];
      pu[j] = pu[j] - alpha1 * pq[j];
      pw[j] = pw[j] - alpha1 * pz[j];
      pm[j] = pm[j] - alpha1 * pc[j];
      pn[j] = pn[j] - alpha1 * pd[j];

      lambda[0] += pu[j]  * ps[j];  lambda[1] += pw[j]  * pm[j];
      lambda[2] += pw[j]  * pq[j];  lambda[3] += ps[j]  * pq[j]; 
      lambda[4] += pn[j]  * pm[j];  lambda[5] += pn[j] * pq[j]; 
      lambda[6] += pz[j] * pq[j];   lambda[7] += pr[j]  * pu[j];
      lambda[8] += pu[j] * pw[j];   lambda[9] += pu[j]  * pu[j];
    }
  } else if(normtype==KSP_NORM_UNPRECONDITIONED) {
      #pragma _CRI ivdep
      for(j=0; j<n; j++) {
         pz[j] = pn[j];
         pq[j] = pm[j];
         ps[j] = pw[j];
         pp[j] = pu[j];
         pc[j] = pg0[j];
         pd[j] = ph0[j];
         pa1[j] = pe[j];
         pb1[j] = pf[j];

         px[j] = px[j] + alpha0 * pp[j];
         pr[j] = pr[j] - alpha0 * ps[j];
         pu[j] = pu[j] - alpha0 * pq[j];
         pw[j] = pw[j] - alpha0 * pz[j];
         pm[j] = pm[j] - alpha0 * pc[j];
         pn[j] = pn[j] - alpha0 * pd[j];
         pg0[j] = pg0[j] - alpha0 * pa1[j];
         ph0[j] = ph0[j] - alpha0 * pb1[j];

         pg1[j] = pg0[j];
         ph1[j] = ph0[j];

         pz[j] = pn[j] + beta1 * pz[j];
         pq[j] = pm[j] + beta1 * pq[j];
         ps[j] = pw[j] + beta1 * ps[j];
         pp[j] = pu[j] + beta1 * pp[j];
         pc[j] = pg0[j] + beta1 * pc[j];
         pd[j] = ph0[j] + beta1 * pd[j];

         px[j] = px[j] + alpha1 * pp[j];
         pr[j] = pr[j] - alpha1 * ps[j];
         pu[j] = pu[j] - alpha1 * pq[j];
         pw[j] = pw[j] - alpha1 * pz[j];
         pm[j] = pm[j] - alpha1 * pc[j];
         pn[j] = pn[j] - alpha1 * pd[j];
    
         lambda[0] += pu[j]  * ps[j];  lambda[1] += pw[j]  * pm[j];
         lambda[2] += pw[j]  * pq[j];  lambda[3] += ps[j]  * pq[j];
         lambda[4] += pn[j]  * pm[j];  lambda[5] += pn[j] * pq[j];
         lambda[6] += pz[j] * pq[j];   lambda[7] += pr[j]  * pu[j];
         lambda[8] += pu[j] * pw[j];   lambda[9] += pr[j]  * pr[j];
    }
  } else if(normtype==KSP_NORM_NATURAL) {
      #pragma _CRI ivdep
      for(j=0; j<n; j++) {
        pz[j] = pn[j];
        pq[j] = pm[j];
        ps[j] = pw[j];
        pp[j] = pu[j];
        pc[j] = pg0[j];
        pd[j] = ph0[j];
        pa1[j] = pe[j];
        pb1[j] = pf[j];

        px[j] = px[j] + alpha0 * pp[j];
        pr[j] = pr[j] - alpha0 * ps[j];
        pu[j] = pu[j] - alpha0 * pq[j];
        pw[j] = pw[j] - alpha0 * pz[j];
        pm[j] = pm[j] - alpha0 * pc[j];
        pn[j] = pn[j] - alpha0 * pd[j];
        pg0[j] = pg0[j] - alpha0 * pa1[j];
        ph0[j] = ph0[j] - alpha0 * pb1[j];

        pg1[j] = pg0[j];
        ph1[j] = ph0[j];

        pz[j] = pn[j] + beta1 * pz[j];
        pq[j] = pm[j] + beta1 * pq[j];
        ps[j] = pw[j] + beta1 * ps[j];
        pp[j] = pu[j] + beta1 * pp[j];
        pc[j] = pg0[j] + beta1 * pc[j];
        pd[j] = ph0[j] + beta1 * pd[j];

        px[j] = px[j] + alpha1 * pp[j];
        pr[j] = pr[j] - alpha1 * ps[j];
        pu[j] = pu[j] - alpha1 * pq[j];
        pw[j] = pw[j] - alpha1 * pz[j];
        pm[j] = pm[j] - alpha1 * pc[j];
        pn[j] = pn[j] - alpha1 * pd[j];

        lambda[0] += pu[j]  * ps[j];  lambda[1] += pw[j]  * pm[j];
        lambda[2] += pw[j]  * pq[j];  lambda[3] += ps[j]  * pq[j];
        lambda[4] += pn[j]  * pm[j];  lambda[5] += pn[j] * pq[j];
        lambda[6] += pz[j] * pq[j];   lambda[7] += pr[j]  * pu[j];
        lambda[8] += pu[j] * pw[j];  
    }
    lambda[9] = lambda[7];
  }

  ierr = VecRestoreArray(vx,(PetscScalar**)&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(vr,(PetscScalar**)&pr);CHKERRQ(ierr);
  ierr = VecRestoreArray(vz,(PetscScalar**)&pz);CHKERRQ(ierr);
  ierr = VecRestoreArray(vw,(PetscScalar**)&pw);CHKERRQ(ierr);
  ierr = VecRestoreArray(vp,(PetscScalar**)&pp);CHKERRQ(ierr);
  ierr = VecRestoreArray(vq,(PetscScalar**)&pq);CHKERRQ(ierr);
  ierr = VecRestoreArray(vc,(PetscScalar**)&pc);CHKERRQ(ierr);
  ierr = VecRestoreArray(vd,(PetscScalar**)&pd);CHKERRQ(ierr);
  ierr = VecRestoreArray(vg0,(PetscScalar**)&pg0);CHKERRQ(ierr);
  ierr = VecRestoreArray(vh0,(PetscScalar**)&ph0);CHKERRQ(ierr);
  ierr = VecRestoreArray(vg1,(PetscScalar**)&pg1);CHKERRQ(ierr);
  ierr = VecRestoreArray(vh1,(PetscScalar**)&ph1);CHKERRQ(ierr);
  ierr = VecRestoreArray(vs,(PetscScalar**)&ps);CHKERRQ(ierr);
  ierr = VecRestoreArray(va1,(PetscScalar**)&pa1);CHKERRQ(ierr);
  ierr = VecRestoreArray(vb1,(PetscScalar**)&pb1);CHKERRQ(ierr);
  ierr = VecRestoreArray(ve,(PetscScalar**)&pe);CHKERRQ(ierr);
  ierr = VecRestoreArray(vf,(PetscScalar**)&pf);CHKERRQ(ierr);
  ierr = VecRestoreArray(vm,(PetscScalar**)&pm);CHKERRQ(ierr);
  ierr = VecRestoreArray(vn,(PetscScalar**)&pn);CHKERRQ(ierr);
  ierr = VecRestoreArray(vu,(PetscScalar**)&pu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*   VecMergedOps function merges the dot products, AXPY and SAXPY operations for all vectors for iteration > 0  */
PetscErrorCode VecMergedOps(Vec vx,Vec vr,Vec vz,Vec vw,Vec vp,Vec vq,Vec vc, Vec vd,Vec vg0,Vec vh0,Vec vg1,Vec vh1,//
		            Vec vs, Vec va1, Vec vb1, Vec ve,Vec vf,Vec vm,Vec vn, Vec vu, PetscInt normtype,PetscScalar beta0,//
			    PetscScalar alpha0, PetscScalar beta1, PetscScalar alpha1, PetscScalar *lambda, PetscScalar alphaold)
{  
  PetscScalar       *PETSC_RESTRICT px, *PETSC_RESTRICT pr, *PETSC_RESTRICT pz, *PETSC_RESTRICT pw;
  PetscScalar       *PETSC_RESTRICT pp, *PETSC_RESTRICT pq;
  PetscScalar       *PETSC_RESTRICT pc,  *PETSC_RESTRICT pd, *PETSC_RESTRICT pg0,  *PETSC_RESTRICT ph0,  *PETSC_RESTRICT pg1, //
		    *PETSC_RESTRICT ph1,*PETSC_RESTRICT ps, *PETSC_RESTRICT pa1,*PETSC_RESTRICT pb1,*PETSC_RESTRICT pe,*PETSC_RESTRICT pf,//
		    *PETSC_RESTRICT pm,*PETSC_RESTRICT pn, *PETSC_RESTRICT pu;
  PetscInt          j, n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(vx,(PetscScalar**)&px);CHKERRQ(ierr);
  ierr = VecGetArray(vr,(PetscScalar**)&pr);CHKERRQ(ierr);
  ierr = VecGetArray(vz,(PetscScalar**)&pz);CHKERRQ(ierr);
  ierr = VecGetArray(vw,(PetscScalar**)&pw);CHKERRQ(ierr);
  ierr = VecGetArray(vp,(PetscScalar**)&pp);CHKERRQ(ierr);
  ierr = VecGetArray(vq,(PetscScalar**)&pq);CHKERRQ(ierr);
  ierr = VecGetArray(vc,(PetscScalar**)&pc);CHKERRQ(ierr);
  ierr = VecGetArray(vd,(PetscScalar**)&pd);CHKERRQ(ierr);
  ierr = VecGetArray(vg0,(PetscScalar**)&pg0);CHKERRQ(ierr);
  ierr = VecGetArray(vh0,(PetscScalar**)&ph0);CHKERRQ(ierr);
  ierr = VecGetArray(vg1,(PetscScalar**)&pg1);CHKERRQ(ierr);
  ierr = VecGetArray(vh1,(PetscScalar**)&ph1);CHKERRQ(ierr);
  ierr = VecGetArray(vs,(PetscScalar**)&ps);CHKERRQ(ierr);
  ierr = VecGetArray(va1,(PetscScalar**)&pa1);CHKERRQ(ierr);
  ierr = VecGetArray(vb1,(PetscScalar**)&pb1);CHKERRQ(ierr);
  ierr = VecGetArray(ve,(PetscScalar**)&pe);CHKERRQ(ierr);
  ierr = VecGetArray(vf,(PetscScalar**)&pf);CHKERRQ(ierr);
  ierr = VecGetArray(vm,(PetscScalar**)&pm);CHKERRQ(ierr);
  ierr = VecGetArray(vn,(PetscScalar**)&pn);CHKERRQ(ierr);
  ierr = VecGetArray(vu,(PetscScalar**)&pu);CHKERRQ(ierr);
  
  ierr = VecGetLocalSize(vx,&n);CHKERRQ(ierr);
  for(j=0; j<10; j++) lambda[j] = 0.0;
 
  if(normtype==KSP_NORM_PRECONDITIONED) {
    #pragma _CRI ivdep
    for(j=0; j<n; j++) {
      pa1[j] = (pg1[j] - pg0[j])/alphaold;
      pb1[j] = (ph1[j] - ph0[j])/alphaold;      
     
      pz[j] = pn[j] + beta0 * pz[j]; 
      pq[j] = pm[j] + beta0 * pq[j];
      ps[j] = pw[j] + beta0 * ps[j];
      pp[j] = pu[j] + beta0 * pp[j];
      pc[j] = pg0[j] + beta0 * pc[j];
      pd[j] = ph0[j] + beta0 * pd[j];
      pa1[j] = pe[j] + beta0 * pa1[j];
      pb1[j] = pf[j] + beta0 * pb1[j];
      
      px[j] = px[j] + alpha0 * pp[j];
      pr[j] = pr[j] - alpha0 * ps[j];
      pu[j] = pu[j] - alpha0 * pq[j];
      pw[j] = pw[j] - alpha0 * pz[j];
      pm[j] = pm[j] - alpha0 * pc[j];
      pn[j] = pn[j] - alpha0 * pd[j];
      pg0[j] = pg0[j] - alpha0 * pa1[j];
      ph0[j] = ph0[j] - alpha0 * pb1[j];

      pg1[j] = pg0[j];
      ph1[j] = ph0[j];

      pz[j] = pn[j] + beta1 * pz[j]; 
      pq[j] = pm[j] + beta1 * pq[j]; 
      ps[j] = pw[j] + beta1 * ps[j];
      pp[j] = pu[j] + beta1 * pp[j]; 
      pc[j] = pg0[j] + beta1 * pc[j];
      pd[j] = ph0[j] + beta1 * pd[j];
      
      px[j] = px[j] + alpha1 * pp[j];
      pr[j] = pr[j] - alpha1 * ps[j];
      pu[j] = pu[j] - alpha1 * pq[j];
      pw[j] = pw[j] - alpha1 * pz[j];
      pm[j] = pm[j] - alpha1 * pc[j];
      pn[j] = pn[j] - alpha1 * pd[j];

      lambda[0] += pu[j]  * ps[j];  lambda[1] += pw[j]  * pm[j];
      lambda[2] += pw[j]  * pq[j];  lambda[3] += ps[j]  * pq[j]; 
      lambda[4] += pn[j]  * pm[j];  lambda[5] += pn[j] * pq[j]; 
      lambda[6] += pz[j] * pq[j];   lambda[7] += pr[j]  * pu[j];
      lambda[8] += pu[j] * pw[j];   lambda[9] += pu[j]  * pu[j];
    }
  } else if(normtype==KSP_NORM_UNPRECONDITIONED) { 
      #pragma _CRI ivdep
      for(j=0; j<n; j++) {
        pa1[j] = (pg1[j] - pg0[j])/alphaold;
        pb1[j] = (ph1[j] - ph0[j])/alphaold;

        pz[j] = pn[j] + beta0 * pz[j];
        pq[j] = pm[j] + beta0 * pq[j];
        ps[j] = pw[j] + beta0 * ps[j];
        pp[j] = pu[j] + beta0 * pp[j];
        pc[j] = pg0[j] + beta0 * pc[j];
        pd[j] = ph0[j] + beta0 * pd[j];
        pa1[j] = pe[j] + beta0 * pa1[j];
        pb1[j] = pf[j] + beta0 * pb1[j];

	px[j] = px[j] + alpha0 * pp[j];
        pr[j] = pr[j] - alpha0 * ps[j];
        pu[j] = pu[j] - alpha0 * pq[j];
        pw[j] = pw[j] - alpha0 * pz[j];
        pm[j] = pm[j] - alpha0 * pc[j];
        pn[j] = pn[j] - alpha0 * pd[j];
        pg0[j] = pg0[j] - alpha0 * pa1[j];
        ph0[j] = ph0[j] - alpha0 * pb1[j];

        pg1[j] = pg0[j];
        ph1[j] = ph0[j];

        pz[j] = pn[j] + beta1 * pz[j];
        pq[j] = pm[j] + beta1 * pq[j];
        ps[j] = pw[j] + beta1 * ps[j];
        pp[j] = pu[j] + beta1 * pp[j];
        pc[j] = pg0[j] + beta1 * pc[j];
        pd[j] = ph0[j] + beta1 * pd[j];

        px[j] = px[j] + alpha1 * pp[j];
        pr[j] = pr[j] - alpha1 * ps[j];
        pu[j] = pu[j] - alpha1 * pq[j];
        pw[j] = pw[j] - alpha1 * pz[j];
        pm[j] = pm[j] - alpha1 * pc[j];
        pn[j] = pn[j] - alpha1 * pd[j];

        lambda[0] += pu[j]  * ps[j];  lambda[1] += pw[j]  * pm[j];
        lambda[2] += pw[j]  * pq[j];  lambda[3] += ps[j]  * pq[j];
        lambda[4] += pn[j]  * pm[j];  lambda[5] += pn[j] * pq[j];
        lambda[6] += pz[j] * pq[j];   lambda[7] += pr[j]  * pu[j];
        lambda[8] += pu[j] * pw[j];   lambda[9] += pr[j]  * pr[j];
    }
  } else if(normtype==KSP_NORM_NATURAL) {
      #pragma _CRI ivdep
      for(j=0; j<n; j++) {
        pa1[j] = (pg1[j] - pg0[j])/alphaold;
        pb1[j] = (ph1[j] - ph0[j])/alphaold;

        pz[j] = pn[j] + beta0 * pz[j];
        pq[j] = pm[j] + beta0 * pq[j];
        ps[j] = pw[j] + beta0 * ps[j];
        pp[j] = pu[j] + beta0 * pp[j];
        pc[j] = pg0[j] + beta0 * pc[j];
        pd[j] = ph0[j] + beta0 * pd[j];
        pa1[j] = pe[j] + beta0 * pa1[j];
        pb1[j] = pf[j] + beta0 * pb1[j];

        px[j] = px[j] + alpha0 * pp[j];
        pr[j] = pr[j] - alpha0 * ps[j];
        pu[j] = pu[j] - alpha0 * pq[j];
        pw[j] = pw[j] - alpha0 * pz[j];
        pm[j] = pm[j] - alpha0 * pc[j];
        pn[j] = pn[j] - alpha0 * pd[j];
        pg0[j] = pg0[j] - alpha0 * pa1[j];
        ph0[j] = ph0[j] - alpha0 * pb1[j];

        pg1[j] = pg0[j];
        ph1[j] = ph0[j];

        pz[j] = pn[j] + beta1 * pz[j];
        pq[j] = pm[j] + beta1 * pq[j];
        ps[j] = pw[j] + beta1 * ps[j];
        pp[j] = pu[j] + beta1 * pp[j];
        pc[j] = pg0[j] + beta1 * pc[j];
        pd[j] = ph0[j] + beta1 * pd[j];

        px[j] = px[j] + alpha1 * pp[j];
        pr[j] = pr[j] - alpha1 * ps[j];
        pu[j] = pu[j] - alpha1 * pq[j];
        pw[j] = pw[j] - alpha1 * pz[j];
        pm[j] = pm[j] - alpha1 * pc[j];
        pn[j] = pn[j] - alpha1 * pd[j];

        lambda[0] += pu[j]  * ps[j];  lambda[1] += pw[j]  * pm[j];
        lambda[2] += pw[j]  * pq[j];  lambda[3] += ps[j]  * pq[j];
        lambda[4] += pn[j]  * pm[j];  lambda[5] += pn[j] * pq[j];
        lambda[6] += pz[j] * pq[j];   lambda[7] += pr[j]  * pu[j];
        lambda[8] += pu[j] * pw[j]; 
    }
    lambda[9] = lambda[7];
  }

  ierr = VecRestoreArray(vx,(PetscScalar**)&px);CHKERRQ(ierr); 
  ierr = VecRestoreArray(vr,(PetscScalar**)&pr);CHKERRQ(ierr);
  ierr = VecRestoreArray(vz,(PetscScalar**)&pz);CHKERRQ(ierr);
  ierr = VecRestoreArray(vw,(PetscScalar**)&pw);CHKERRQ(ierr);
  ierr = VecRestoreArray(vp,(PetscScalar**)&pp);CHKERRQ(ierr);
  ierr = VecRestoreArray(vq,(PetscScalar**)&pq);CHKERRQ(ierr);
  ierr = VecRestoreArray(vc,(PetscScalar**)&pc);CHKERRQ(ierr);
  ierr = VecRestoreArray(vd,(PetscScalar**)&pd);CHKERRQ(ierr);
  ierr = VecRestoreArray(vg0,(PetscScalar**)&pg0);CHKERRQ(ierr);
  ierr = VecRestoreArray(vh0,(PetscScalar**)&ph0);CHKERRQ(ierr);
  ierr = VecRestoreArray(vg1,(PetscScalar**)&pg1);CHKERRQ(ierr);
  ierr = VecRestoreArray(vh1,(PetscScalar**)&ph1);CHKERRQ(ierr);
  ierr = VecRestoreArray(vs,(PetscScalar**)&ps);CHKERRQ(ierr);
  ierr = VecRestoreArray(va1,(PetscScalar**)&pa1);CHKERRQ(ierr);
  ierr = VecRestoreArray(vb1,(PetscScalar**)&pb1);CHKERRQ(ierr);
  ierr = VecRestoreArray(ve,(PetscScalar**)&pe);CHKERRQ(ierr);
  ierr = VecRestoreArray(vf,(PetscScalar**)&pf);CHKERRQ(ierr);
  ierr = VecRestoreArray(vm,(PetscScalar**)&pm);CHKERRQ(ierr);
  ierr = VecRestoreArray(vn,(PetscScalar**)&pn);CHKERRQ(ierr);
  ierr = VecRestoreArray(vu,(PetscScalar**)&pu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

