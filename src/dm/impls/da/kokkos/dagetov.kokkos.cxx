
#include <petsc/private/dmdaimpl.h>
#include <petscdmda_kokkos.hpp>

PETSC_STATIC_INLINE PetscErrorCode DMDAVecGetShape(DM da,Vec vec,
                                                   PetscInt* _xs, PetscInt* _ys, PetscInt* _zs, PetscInt* _xm, PetscInt* _ym, PetscInt* _zm,
                                                   PetscInt *_gxs,PetscInt* _gys,PetscInt* _gzs,PetscInt* _gxm,PetscInt* _gym,PetscInt* _gzm,
                                                   PetscInt* _N,  PetscInt* _dim,PetscInt* _dof)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,&dim,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  /* Handle case where user passes in global vector as opposed to local */
  ierr = VecGetLocalSize(vec,&N);CHKERRQ(ierr);
  if (N == xm*ym*zm*dof) {
    gxm = xm; gym = ym; gzm = zm;
    gxs = xs; gys = ys; gzs = zs;
  } else if (N != gxm*gym*gzm*dof) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMDA local sizes %D %D\n",N,xm*ym*zm*dof,gxm*gym*gzm*dof);

  if (dim != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"KokkosOffsetView is 1D but DMDA is %DD\n",dim);
  *_xs  = xs;  *_ys  = ys;  *_zs  = zs;  *_xm  = xm;  *_ym  = ym;  *_zm  = zm;
  *_gxs = gxs; *_gys = gys; *_gzs = gzs; *_gxm = gxm; *_gym = gym; *_gzm = gzm;
  *_N   = N;   *_dim = dim; *_dof = dof;
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecGetKokkosOffsetView(DM da,Vec vec,PetscScalarKokkosOffsetView1DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  ierr = DMDAVecGetShape(da,vec,&xs,&ys,&zs,&xm,&ym,&zm,&gxs,&gys,&gzs,&gxm,&gym,&gzm,&N,&dim,&dof);CHKERRQ(ierr);
  ierr = VecGetKokkosView(vec,&kv,overwrite);CHKERRQ(ierr);
  *ov  = PetscScalarKokkosOffsetView1DType<MemorySpace>(kv,{gxs*dof}); /* View to OffsetView by giving the start. The extent is already known. */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,PetscScalarKokkosOffsetView1DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ov->view(); /* OffsetView to View */
  ierr = VecRestoreKokkosView(vec,&kv,overwrite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecGetKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  ierr = DMDAVecGetShape(da,vec,&xs,&ys,&zs,&xm,&ym,&zm,&gxs,&gys,&gzs,&gxm,&gym,&gzm,&N,&dim,&dof);CHKERRQ(ierr);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView1DType<MemorySpace>(kv,{gxs*dof}); /* View to OffsetView */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView1DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ov->view();
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecGetKokkosOffsetView(DM da,Vec vec,PetscScalarKokkosOffsetView2DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  PetscScalarKokkosViewType<MemorySpace>       kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  ierr = DMDAVecGetShape(da,vec,&xs,&ys,&zs,&xm,&ym,&zm,&gxs,&gys,&gzs,&gxm,&gym,&gzm,&N,&dim,&dof);CHKERRQ(ierr);
  ierr = VecGetKokkosView(vec,&kv,overwrite);CHKERRQ(ierr);
  *ov  = PetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(), {gxs*dof,(gxs+gxm)*dof}, {gys*dof,(gys+gym)*dof}); /* View to OffsetView */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,PetscScalarKokkosOffsetView2DType<MemorySpace> *ov,PetscBool overwrite)
{
  PetscErrorCode                             ierr;
  PetscScalarKokkosViewType<MemorySpace>     kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  // kv   = ov->view(); /* 2D OffsetView => 2D View => 1D View. Why does it not work? */
  kv   = PetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1));
  ierr = VecRestoreKokkosView(vec,&kv,overwrite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecGetKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  PetscInt                                     xs,ys,zs,xm,ym,zm,gxs,gys,gzs,gxm,gym,gzm,N,dim,dof;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  ierr = DMDAVecGetShape(da,vec,&xs,&ys,&zs,&xm,&ym,&zm,&gxs,&gys,&gzs,&gxm,&gym,&gzm,&N,&dim,&dof);CHKERRQ(ierr);
  ierr = VecGetKokkosView(vec,&kv);CHKERRQ(ierr);
  *ov  = ConstPetscScalarKokkosOffsetView2DType<MemorySpace>(kv.data(), {gxs*dof,(gxs+gxm)*dof}, {gys*dof,(gys+gym)*dof}); /* View to OffsetView */
  PetscFunctionReturn(0);
}

template<class MemorySpace>
PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM da,Vec vec,ConstPetscScalarKokkosOffsetView2DType<MemorySpace> *ov)
{
  PetscErrorCode                               ierr;
  ConstPetscScalarKokkosViewType<MemorySpace>  kv;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(da,DM_CLASSID,1,DMDA);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidPointer(ov,3);
  kv   = ConstPetscScalarKokkosViewType<MemorySpace>(ov->data(),ov->extent(0)*ov->extent(1));
  ierr = VecRestoreKokkosView(vec,&kv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Function template explicit instantiation */
template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecGetKokkosOffsetView    (DM,Vec,PetscScalarKokkosOffsetView1D*,PetscBool);
template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM,Vec,PetscScalarKokkosOffsetView1D*,PetscBool);

template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecGetKokkosOffsetView    (DM,Vec,ConstPetscScalarKokkosOffsetView1D*);
template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView1D*);

template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecGetKokkosOffsetView    (DM,Vec,PetscScalarKokkosOffsetView2D*,PetscBool);
template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM,Vec,PetscScalarKokkosOffsetView2D*,PetscBool);

template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecGetKokkosOffsetView    (DM,Vec,ConstPetscScalarKokkosOffsetView2D*);
template PETSC_VISIBILITY_PUBLIC PetscErrorCode  DMDAVecRestoreKokkosOffsetView(DM,Vec,ConstPetscScalarKokkosOffsetView2D*);
