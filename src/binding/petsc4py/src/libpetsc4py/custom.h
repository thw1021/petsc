#include "petsc/private/vecimpl.h"
#include "petsc/private/matimpl.h"
#include "petsc/private/pcimpl.h"
#include "petsc/private/kspimpl.h"
#include "petsc/private/snesimpl.h"
#include "petsc/private/tsimpl.h"
#include "petsc/private/taoimpl.h"

PETSC_EXTERN PetscErrorCode (*PetscPythonMonitorSet_C)(PetscObject,const char*);

static inline
PetscErrorCode PetscObjectComposedDataGetIntPy(PetscObject o, PetscInt id, PetscInt *v, PetscBool *exist)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposedDataGetInt(o,id,*v,*exist));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode PetscObjectComposedDataSetIntPy(PetscObject o, PetscInt id, PetscInt v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposedDataSetInt(o,id,v));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode PetscObjectComposedDataRegisterPy(PetscInt *id)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposedDataRegister(id));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode KSPLogHistory(KSP ksp,PetscReal rnorm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode SNESLogHistory(SNES snes,PetscReal rnorm,PetscInt lits)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  CHKERRQ(SNESLogConvergenceHistory(snes,rnorm,lits));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode KSPConverged(KSP ksp,
                            PetscInt iter,PetscReal rnorm,
                            KSPConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (reason) PetscValidPointer(reason,2);
  if (!iter) ksp->rnorm0 = rnorm;
  if (!iter) {
    ksp->reason = KSP_CONVERGED_ITERATING;
    ksp->ttol = PetscMax(rnorm*ksp->rtol,ksp->abstol);
  }
  if (ksp->converged) {
    CHKERRQ((*ksp->converged)(ksp,iter,rnorm,&ksp->reason,ksp->cnvP));
  } else {
    CHKERRQ(KSPConvergedSkip(ksp,iter,rnorm,&ksp->reason,NULL));
    /*CHKERRQ(KSPConvergedDefault(ksp,iter,rnorm,&ksp->reason,NULL));*/
  }
  ksp->rnorm = rnorm;
  if (reason) *reason = ksp->reason;
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode SNESConverged(SNES snes,
                             PetscInt iter,PetscReal xnorm,PetscReal ynorm,PetscReal fnorm,
                             SNESConvergedReason *reason)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (reason) PetscValidPointer(reason,2);
  if (!iter) {
    snes->reason = SNES_CONVERGED_ITERATING;
    snes->ttol = fnorm*snes->rtol;
  }
  if (snes->ops->converged) {
    CHKERRQ((*snes->ops->converged)(snes,iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
  } else {
    CHKERRQ(SNESConvergedSkip(snes,iter,xnorm,ynorm,fnorm,&snes->reason,0));
    /*CHKERRQ(SNESConvergedDefault(snes,iter,xnorm,ynorm,fnorm,&snes->reason,0));*/
  }
  snes->norm = fnorm;
  if (reason) *reason = snes->reason;
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoRegisterCustom(const char sname[], PetscErrorCode (*function)(Tao))
{
#if !defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  PetscFunctionBegin;
  CHKERRQ(TaoRegister(sname, function));
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  PetscFunctionReturn(0);
#endif
}

static inline
PetscErrorCode TaoConverged(Tao tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (tao->ops->convergencetest) {
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
  } else {
    CHKERRQ(TaoDefaultConvergenceTest(tao,tao->cnvP));
  }
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoCheckReals(Tao tao, PetscReal f, PetscReal g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(g),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER,"User provided compute function generated Inf or NaN");
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoCreateDefaultKSP(Tao tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  CHKERRQ(KSPDestroy(&tao->ksp));
  CHKERRQ(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->ksp,(PetscObject)tao,1));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoCreateDefaultLineSearch(Tao tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  CHKERRQ(TaoLineSearchDestroy(&tao->linesearch));
  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch,(PetscObject)tao,1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch,TAOLINESEARCHMT));
  CHKERRQ(TaoLineSearchUseTaoRoutines(tao->linesearch,tao));
  CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoHasGradientRoutine(Tao tao, PetscBool* flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg = (PetscBool)(tao->ops->computegradient || tao->ops->computeobjectiveandgradient);
  PetscFunctionReturn(0);
}

#if 0
static inline
PetscErrorCode TaoHasHessianRoutine(Tao tao, PetscBool* flg)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  *flg = tao->ops->computehessian;
  PetscFunctionReturn(0);
}
#endif

static inline
PetscErrorCode TaoComputeUpdate(Tao tao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (tao->ops->update) {
    CHKERRQ((*tao->ops->update)(tao,tao->niter,tao->user_update));
  }
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoGetVecs(Tao tao, Vec *X, Vec *G, Vec *S)
{
  PetscBool has_g;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  CHKERRQ(TaoHasGradientRoutine(tao,&has_g));
  if (X) *X = tao->solution;
  if (G) {
    if (has_g && !tao->gradient) {
      CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
    }
    *G = has_g ? tao->gradient : NULL;
  }
  if (S) {
    if (has_g && !tao->stepdirection) {
      CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
    }
    *S = has_g ? tao->stepdirection : NULL;
  }
  PetscFunctionReturn(0);
}

static inline
PetscErrorCode TaoApplyLineSearch(Tao tao, PetscReal* f, PetscReal *s)
{
  TaoLineSearchConvergedReason ls_reason;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidRealPointer(f,2);
  PetscValidRealPointer(s,3);
  ierr = TaoLineSearchApply(tao->linesearch,tao->solution,f,tao->gradient,tao->stepdirection,s,&ls_reason);CHKERRQ(ierr);
  PetscCheck((ls_reason == TAOLINESEARCH_SUCCESS) || (ls_reason == TAOLINESEARCH_SUCCESS_USER),PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Linesearch failed");
  ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifndef PETSC_ERR_PYTHON
#define PETSC_ERR_PYTHON ((PetscErrorCode)(-1))
#endif

#define PetscERROR(comm,FUNCT,n,t,msg,arg) \
        PetscError(comm,__LINE__,FUNCT,__FILE__,n,t,msg,arg)

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initlibpetsc4py(void);
#else
#undef  CYTHON_PEP489_MULTI_PHASE_INIT
#define CYTHON_PEP489_MULTI_PHASE_INIT 0
PyMODINIT_FUNC PyInit_libpetsc4py(void);
static void initlibpetsc4py(void)
{
  PyObject *M, *m;
  M = PyImport_GetModuleDict();
  if (!M) return;
  m = PyInit_libpetsc4py();
  if (!m) return;
  PyDict_SetItemString(M, "libpetsc4py", m);
  Py_DECREF(m);
}
#endif
