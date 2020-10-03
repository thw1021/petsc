#include <petsc/private/dmdaimpl.h>  /*I   "petscdmda.h"   I*/

/*@
  DMDACreatePatchIS - Creates an index set corresponding to a patch of the DA.

  Collective

  Input Parameters:
+  da - the DMDA
.  lower - a matstencil with i, j and k corresponding to the lower corner of the patch
-  upper - a matstencil with i, j and k corresponding to the upper corner of the patch

  Output Parameters:
.  is - the IS corresponding to the patch

  Level: developer

.seealso: DMDACreateDomainDecomposition(), DMDACreateDomainDecompositionScatters()
@*/
PetscErrorCode DMDACreatePatchIS(DM da,MatStencil *lower,MatStencil *upper,IS *is)
{
  PetscInt       ms=0,ns=0,ps=0;
  PetscInt       mw=0,nw=0,pw=0;
  PetscInt       me=1,ne=1,pe=1;
  PetscInt       ii,jj,kk;
  PetscInt       si,sj,sk;
  PetscInt       i,j,k,l,idx;
  PetscInt       base;
  PetscInt       xm=1,ym=1,zm=1;
  PetscInt       ox,oy,oz;
  PetscInt       M,N,P,dof;
  PetscInt       nindices;
  PetscInt       *indices;
  DM_DA          *dd = (DM_DA*)da->data;
  PetscBool      skip_i=PETSC_TRUE, skip_j=PETSC_TRUE, skip_k=PETSC_TRUE;
  PetscBool      valid_j=PETSC_FALSE, valid_k=PETSC_FALSE; /* DMDA has at least 1 dimension */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  M = dd->M; N = dd->N; P = dd->P;
  dof = dd->w;
  ierr = DMDAGetOffset(da,&ox,&oy,&oz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &ms, &ns, &ps, &mw, &nw, &pw);CHKERRQ(ierr);

  nindices = -1;
  if (PetscLikely(upper->i - lower->i)) {
    nindices = nindices*(upper->i - lower->i);
    skip_i=PETSC_FALSE;
  }
  if (N>1) {
    valid_j = PETSC_TRUE;
    if (PetscLikely(upper->j - lower->j)) {
      nindices = nindices*(upper->j - lower->j);
      skip_j=PETSC_FALSE;
    }
  }
  if (P>1) {
    valid_k = PETSC_TRUE;
    if (PetscLikely(upper->k - lower->k)) {
      nindices = nindices*(upper->k - lower->k);
      skip_k=PETSC_FALSE;
    }
  }
  if (PetscLikely(nindices<0)) {
    nindices = nindices*(-1);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_WRONG,"Lower and Upper stencils are identical! Please check inputs.");
  }

  ierr = PetscMalloc1(nindices,&indices);CHKERRQ(ierr);
  idx = 0;
  me = ms + mw;
  if (N>1) ne = ns + nw;
  if (P>1) pe = ps + pw;

  if (!valid_k) {
    k = 0;
    upper->k=0;
    lower->k=0;
  }
  if (!valid_j) {
    j = 0;
    upper->j=0;
    lower->j=0;
  }

  /* do while loops to ensure the block gets entered once,
     regardless of control condition being met, necessary for
     cases when skip_i/j/k is true
     */
  if (skip_k) k = upper->k-oz; else k = lower->k-oz;
  do {
    if (skip_j) j = upper->j-oy; else j = lower->j-oy;
    do {
      if (skip_i) i = upper->i-ox; else i = lower->i-ox;
      do {
        /* "actual" indices rather than ones outside of the domain */
        ii = i;
        jj = j;
        kk = k;
        if (kk>=ps && kk<=pe-1) {
          if (jj>=ns && jj<=ne-1) {
            if (ii>=ms && ii<=me-1) {
              /* compute the vector base on owning processor */
              xm = me - ms;
              ym = ne - ns;
              zm = pe - ps;
              base = ms*ym*zm + ns*M*zm + ps*M*N;
              /* compute the local coordinates on owning processor */
              si = ii - ms;
              sj = jj - ns;
              sk = kk - ps;
              for (l=0;l<dof;l++) {
                indices[idx] = l + dof*(base + si + xm*sj + xm*ym*sk);
                idx++;
              }
            }
          }
        }
        i++;
      } while (i<upper->i-ox);
      j++;
    } while (j<upper->j-oy);
    k++;
  } while (k<upper->k-oz);

  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)da),idx,indices,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  ierr = PetscRealloc((size_t)(idx*sizeof(PetscInt)), (void*)&indices);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode DMDASubDomainDA_Private(DM dm, PetscInt *nlocal, DM **sdm)
{
  DM             *da;
  PetscInt       dim,size,i,j,k,idx;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       xsize,ysize,zsize;
  PetscInt       xo,yo,zo;
  PetscInt       xs,ys,zs;
  PetscInt       xm=1,ym=1,zm=1;
  PetscInt       xol,yol,zol;
  PetscInt       m=1,n=1,p=1;
  PetscInt       M,N,P;
  PetscInt       pm,mtmp;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAGetOverlap(dm,&xol,&yol,&zol);CHKERRQ(ierr);
  ierr = DMDAGetNumLocalSubDomains(dm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc1(size,&da);CHKERRQ(ierr);

  if (nlocal) *nlocal = size;

  dim = info.dim;

  M = info.xm;
  N = info.ym;
  P = info.zm;

  if (dim == 1) {
    m = size;
  } else if (dim == 2) {
    m = (PetscInt)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)N)));
    while (m > 0) {
      n = size/m;
      if (m*n*p == size) break;
      m--;
    }
  } else if (dim == 3) {
    n = (PetscInt)(0.5 + PetscPowReal(((PetscReal)N*N)*((PetscReal)size)/((PetscReal)P*M),(PetscReal)(1./3.))); if (!n) n = 1;
    while (n > 0) {
      pm = size/n;
      if (n*pm == size) break;
      n--;
    }
    if (!n) n = 1;
    m = (PetscInt)(0.5 + PetscSqrtReal(((PetscReal)M)*((PetscReal)size)/((PetscReal)P*n)));
    if (!m) m = 1;
    while (m > 0) {
      p = size/(m*n);
      if (m*n*p == size) break;
      m--;
    }
    if (M > P && m < p) {mtmp = m; m = p; p = mtmp;}
  }

  zs = info.zs;
  idx = 0;
  for (k = 0; k < p; k++) {
    ys = info.ys;
    for (j = 0; j < n; j++) {
      xs = info.xs;
      for (i = 0; i < m; i++) {
        if (dim == 1) {
          xm = M/m + ((M % m) > i);
        } else if (dim == 2) {
          xm = M/m + ((M % m) > i);
          ym = N/n + ((N % n) > j);
        } else if (dim == 3) {
          xm = M/m + ((M % m) > i);
          ym = N/n + ((N % n) > j);
          zm = P/p + ((P % p) > k);
        }

        xsize = xm;
        ysize = ym;
        zsize = zm;
        xo = xs;
        yo = ys;
        zo = zs;

        ierr = DMDACreate(PETSC_COMM_SELF,&(da[idx]));CHKERRQ(ierr);
        ierr = DMSetOptionsPrefix(da[idx],"sub_");CHKERRQ(ierr);
        ierr = DMSetDimension(da[idx], info.dim);CHKERRQ(ierr);
        ierr = DMDASetDof(da[idx], info.dof);CHKERRQ(ierr);

        ierr = DMDASetStencilType(da[idx],info.st);CHKERRQ(ierr);
        ierr = DMDASetStencilWidth(da[idx],info.sw);CHKERRQ(ierr);

        if (info.bx == DM_BOUNDARY_PERIODIC || (xs != 0)) {
          xsize += xol;
          xo    -= xol;
        }
        if (info.by == DM_BOUNDARY_PERIODIC || (ys != 0)) {
          ysize += yol;
          yo    -= yol;
        }
        if (info.bz == DM_BOUNDARY_PERIODIC || (zs != 0)) {
          zsize += zol;
          zo    -= zol;
        }

        if (info.bx == DM_BOUNDARY_PERIODIC || (xs+xm != info.mx)) xsize += xol;
        if (info.by == DM_BOUNDARY_PERIODIC || (ys+ym != info.my)) ysize += yol;
        if (info.bz == DM_BOUNDARY_PERIODIC || (zs+zm != info.mz)) zsize += zol;

        if (info.bx != DM_BOUNDARY_PERIODIC) {
          if (xo < 0) {
            xsize += xo;
            xo = 0;
          }
          if (xo+xsize > info.mx-1) {
            xsize -= xo+xsize - info.mx;
          }
        }
        if (info.by != DM_BOUNDARY_PERIODIC) {
          if (yo < 0) {
            ysize += yo;
            yo = 0;
          }
          if (yo+ysize > info.my-1) {
            ysize -= yo+ysize - info.my;
          }
        }
        if (info.bz != DM_BOUNDARY_PERIODIC) {
          if (zo < 0) {
            zsize += zo;
            zo = 0;
          }
          if (zo+zsize > info.mz-1) {
            zsize -= zo+zsize - info.mz;
          }
        }

        ierr = DMDASetSizes(da[idx], xsize, ysize, zsize);CHKERRQ(ierr);
        ierr = DMDASetNumProcs(da[idx], 1, 1, 1);CHKERRQ(ierr);
        ierr = DMDASetBoundaryType(da[idx], DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED);CHKERRQ(ierr);

        /* set up as a block instead */
        ierr = DMSetUp(da[idx]);CHKERRQ(ierr);

        /* nonoverlapping region */
        ierr = DMDASetNonOverlappingRegion(da[idx],xs,ys,zs,xm,ym,zm);CHKERRQ(ierr);

        /* this alters the behavior of DMDAGetInfo, DMDAGetLocalInfo, DMDAGetCorners, and DMDAGetGhostedCorners and should be used with care */
        ierr = DMDASetOffset(da[idx],xo,yo,zo,info.mx,info.my,info.mz);CHKERRQ(ierr);
        xs += xm;
        idx++;
      }
      ys += ym;
    }
    zs += zm;
  }
  *sdm = da;
  PetscFunctionReturn(ierr);
}

/*
 Fills the local vector problem on the subdomain from the global problem.

 Right now this assumes one subdomain per processor.

*/
PetscErrorCode DMCreateDomainDecompositionScatters_DA(DM dm,PetscInt nsubdms,DM *subdms,VecScatter **iscat,VecScatter **oscat, VecScatter **lscat)
{
  PetscErrorCode ierr;
  DMDALocalInfo  info,subinfo;
  DM             subdm;
  MatStencil     upper,lower;
  IS             idis,isis,odis,osis,gdis;
  Vec            svec,dvec,slvec;
  PetscInt       xm,ym,zm,xs,ys,zs;
  PetscInt       i;

  PetscFunctionBegin;
  /* allocate the arrays of scatters */
  if (iscat) {ierr = PetscMalloc1(nsubdms,iscat);CHKERRQ(ierr);}
  if (oscat) {ierr = PetscMalloc1(nsubdms,oscat);CHKERRQ(ierr);}
  if (lscat) {ierr = PetscMalloc1(nsubdms,lscat);CHKERRQ(ierr);}

  ierr  = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  for (i = 0; i < nsubdms; i++) {
    subdm = subdms[i];
    ierr  = DMDAGetLocalInfo(subdm,&subinfo);CHKERRQ(ierr);
    ierr = DMDAGetNonOverlappingRegion(subdm,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

    /* create the global and subdomain index sets for the inner domain */
    lower.i = xs;
    lower.j = ys;
    lower.k = zs;
    upper.i = xs+xm;
    upper.j = ys+ym;
    upper.k = zs+zm;
    ierr    = DMDACreatePatchIS(dm,&lower,&upper,&idis);CHKERRQ(ierr);
    ierr    = DMDACreatePatchIS(subdm,&lower,&upper,&isis);CHKERRQ(ierr);

    /* create the global and subdomain index sets for the outer subdomain */
    lower.i = subinfo.xs;
    lower.j = subinfo.ys;
    lower.k = subinfo.zs;
    upper.i = subinfo.xs+subinfo.xm;
    upper.j = subinfo.ys+subinfo.ym;
    upper.k = subinfo.zs+subinfo.zm;
    ierr    = DMDACreatePatchIS(dm,&lower,&upper,&odis);CHKERRQ(ierr);
    ierr    = DMDACreatePatchIS(subdm,&lower,&upper,&osis);CHKERRQ(ierr);

    /* global and subdomain ISes for the local indices of the subdomain */
    /* todo - make this not loop over at nonperiodic boundaries, which will be more involved */
    lower.i = subinfo.gxs;
    lower.j = subinfo.gys;
    lower.k = subinfo.gzs;
    upper.i = subinfo.gxs+subinfo.gxm;
    upper.j = subinfo.gys+subinfo.gym;
    upper.k = subinfo.gzs+subinfo.gzm;

    ierr = DMDACreatePatchIS(dm,&lower,&upper,&gdis);CHKERRQ(ierr);

    /* form the scatter */
    ierr = DMGetGlobalVector(dm,&dvec);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(subdm,&svec);CHKERRQ(ierr);
    ierr = DMGetLocalVector(subdm,&slvec);CHKERRQ(ierr);

    if (iscat) {ierr = VecScatterCreate(dvec,idis,svec,isis,&(*iscat)[i]);CHKERRQ(ierr);}
    if (oscat) {ierr = VecScatterCreate(dvec,odis,svec,osis,&(*oscat)[i]);CHKERRQ(ierr);}
    if (lscat) {ierr = VecScatterCreate(dvec,gdis,slvec,NULL,&(*lscat)[i]);CHKERRQ(ierr);}

    ierr = DMRestoreGlobalVector(dm,&dvec);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(subdm,&svec);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(subdm,&slvec);CHKERRQ(ierr);

    ierr = ISDestroy(&idis);CHKERRQ(ierr);
    ierr = ISDestroy(&isis);CHKERRQ(ierr);

    ierr = ISDestroy(&odis);CHKERRQ(ierr);
    ierr = ISDestroy(&osis);CHKERRQ(ierr);

    ierr = ISDestroy(&gdis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDASubDomainIS_Private(DM dm,PetscInt n,DM *subdm,IS **iis,IS **ois)
{
  PetscErrorCode ierr;
  PetscInt       i;
  DMDALocalInfo  info,subinfo;
  MatStencil     lower,upper;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  if (iis) {ierr = PetscMalloc1(n,iis);CHKERRQ(ierr);}
  if (ois) {ierr = PetscMalloc1(n,ois);CHKERRQ(ierr);}

  for (i = 0;i < n; i++) {
    ierr = DMDAGetLocalInfo(subdm[i],&subinfo);CHKERRQ(ierr);
    if (iis) {
      /* create the inner IS */
      lower.i = info.xs;
      lower.j = info.ys;
      lower.k = info.zs;
      upper.i = info.xs+info.xm;
      upper.j = info.ys+info.ym;
      upper.k = info.zs+info.zm;
      ierr = DMDACreatePatchIS(dm,&lower,&upper,&(*iis)[i]);CHKERRQ(ierr);
    }

    if (ois) {
      /* create the outer IS */
      lower.i = subinfo.xs;
      lower.j = subinfo.ys;
      lower.k = subinfo.zs;
      upper.i = subinfo.xs+subinfo.xm;
      upper.j = subinfo.ys+subinfo.ym;
      upper.k = subinfo.zs+subinfo.zm;
      ierr    = DMDACreatePatchIS(dm,&lower,&upper,&(*ois)[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode DMCreateDomainDecomposition_DA(DM dm,PetscInt *len,char ***names,IS **iis,IS **ois,DM **subdm)
{
  PetscErrorCode ierr;
  DM             *sdm;
  PetscInt       n,i;

  PetscFunctionBegin;
  ierr = DMDASubDomainDA_Private(dm,&n,&sdm);CHKERRQ(ierr);
  if (names) {
    ierr = PetscMalloc1(n,names);CHKERRQ(ierr);
    for (i=0;i<n;i++) (*names)[i] = NULL;
  }
  ierr = DMDASubDomainIS_Private(dm,n,sdm,iis,ois);CHKERRQ(ierr);
  if (subdm) *subdm = sdm;
  else {
    for (i=0;i<n;i++) {
      ierr = DMDestroy(&sdm[i]);CHKERRQ(ierr);
    }
  }
  if (len) *len = n;
  PetscFunctionReturn(ierr);
}
