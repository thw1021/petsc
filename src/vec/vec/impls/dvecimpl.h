/*
   This should not be included in users code.

  Includes definition of structure for sequential vectors

  These are shared by dvec1.c dvec2.c dvec3.c bvec1.c bvec2.c pvec.c pbvec.c
*/

#if !defined(__DVECIMPL)
#define __DVECIMPL

#include <petsc/private/vecimpl.h>

typedef struct {
  VECHEADER
} Vec_Seq;

PETSC_INTERN PetscErrorCode VecMDot_Seq(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMTDot_Seq(Vec,PetscManagedInt,const Vec[],PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMin_Seq(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSet_Seq(Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMAXPY_Seq(Vec,PetscManagedInt,PetscManagedScalar,Vec*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecAYPX_Seq(Vec,PetscManagedScalar,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecWAXPY_Seq(Vec,PetscManagedScalar,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_Seq(Vec,PetscManagedScalar,PetscManagedScalar,PetscManagedScalar,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMaxPointwiseDivide_Seq(Vec,Vec,PetscManagedReal,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPlaceArray_Seq(Vec,const PetscScalar*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecResetArray_Seq(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecReplaceArray_Seq(Vec,const PetscScalar*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDot_Seq(Vec,Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecTDot_Seq(Vec,Vec,PetscManagedScalar,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecScale_Seq(Vec,PetscManagedScalar,PetscDeviceContext);
// exposed by VecSeq_CUPM to MatAIJ_CUSPARSE
PETSC_SINGLE_LIBRARY_INTERN PetscErrorCode VecAXPY_Seq(Vec,PetscManagedScalar,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecAXPBY_Seq(Vec,PetscManagedScalar,PetscManagedScalar,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMax_Seq(Vec,PetscManagedInt,PetscManagedReal,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecNorm_Seq(Vec,NormType,PetscManagedReal,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDestroy_Seq(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDuplicate_Seq(Vec,Vec*,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSetOption_Seq(Vec,VecOption,PetscBool);
PETSC_INTERN PetscErrorCode VecGetValues_Seq(Vec,PetscInt,const PetscInt*,PetscScalar*);
PETSC_INTERN PetscErrorCode VecSetValues_Seq(Vec,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
PETSC_INTERN PetscErrorCode VecSetValuesBlocked_Seq(Vec,PetscInt,const PetscInt*,const PetscScalar*,InsertMode);
PETSC_INTERN PetscErrorCode VecGetSize_Seq(Vec,PetscInt*);
PETSC_INTERN PetscErrorCode VecCopy_Seq(Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSwap_Seq(Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecConjugate_Seq(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSetRandom_Seq(Vec,PetscRandom,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseMult_Seq(Vec,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseMax_Seq(Vec,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseMaxAbs_Seq(Vec,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseMin_Seq(Vec,Vec,Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_Seq(Vec,Vec,Vec,PetscDeviceContext);

PETSC_EXTERN PetscErrorCode VecCreate_Seq(Vec,PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecCreate_Seq_Private(Vec,const PetscScalar[],PetscDeviceContext);
#endif
