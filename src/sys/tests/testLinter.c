#include <petsc.h>

PetscErrorCode testValidPointers(void *a, char *b, PetscInt *c, PetscMPIInt *d, PetscInt64 *e, PetscBool *f, PetscScalar *g, PetscReal *h, int *i)
{
  PetscFunctionBegin;
  /* incorrect */
  PetscValidCharPointer(a,2);
  PetscValidIntPointer(b,3);
  PetscValidBoolPointer(c,4);
  PetscValidRealPointer(d,5);
  PetscValidScalarPointer(e,6);
  PetscValidIntPointer(f,7);
  PetscValidRealPointer(g,8);
  PetscValidScalarPointer(h,9);
  PetscValidBoolPointer(i,10);

  /* correct */
  PetscValidPointer(a,1);
  PetscValidCharPointer(b,2);
  PetscValidIntPointer(c,3);
  PetscValidIntPointer(d,4);
  PetscValidIntPointer(e,5);
  PetscValidBoolPointer(f,6);
  PetscValidScalarPointer(g,7);
  PetscValidRealPointer(h,8);
  PetscValidIntPointer(i,9);
  PetscFunctionReturn(0);
}

PetscErrorCode testValidLogicalCollective(PetscInt *a, PetscMPIInt *b, PetscInt64 *c, PetscBool *d, PetscScalar *e, PetscReal *f, int *g)
{
  PetscFunctionBegin;
  /* incorrect */
  PetscValidLogicalCollectiveInt(d,2);
  PetscValidLogicalCollectiveEnum(e,3);
  PetscValidLogicalCollectiveMPIInt(f,4);
  PetscValidLogicalCollectiveBool(g,5);
  PetscValidLogicalCollectiveScalar(a,6);
  PetscValidLogicalCollectiveReal(b,7);

  /* correct */
  PetscValidLogicalCollectiveInt(a,1);
  PetscValidLogicalCollectiveMPIInt(b,2);
  PetscValidLogicalCollectiveInt(c,3);
  PetscValidLogicalCollectiveBool(d,4);
  PetscValidLogicalCollectiveScalar(e,5);
  PetscValidLogicalCollectiveReal(f,6);
  PetscValidLogicalCollectiveInt(g,7);
  PetscFunctionReturn(0);
}

PetscErrorCode testValidHeaders(Mat m, Vec v, KSP k, SNES s)
{
  PetscFunctionBegin;
  /* incorrect */
  PetscValidHeaderSpecificType(m,VEC_CLASSID,0,DMDA);
  PetscValidHeaderSpecificType(v,KSP_CLASSID,0,DMDA);
  PetscValidHeaderSpecificType(k,SNES_CLASSID,0,DMDA);
  PetscValidHeaderSpecificType(s,MAT_CLASSID,0,DMDA);

  /* correct */
  PetscValidHeaderSpecificType(m,MAT_CLASSID,1,DMDA);
  PetscValidHeaderSpecificType(v,VEC_CLASSID,2,DMDA);
  PetscValidHeaderSpecificType(k,KSP_CLASSID,3,DMDA);
  PetscValidHeaderSpecificType(s,SNES_CLASSID,4,DMDA);

  /* incorrect */
  PetscValidHeaderSpecific(m,KSP_CLASSID,0);
  PetscValidHeaderSpecific(v,SNES_CLASSID,0);
  PetscValidHeaderSpecific(k,MAT_CLASSID,0);
  PetscValidHeaderSpecific(s,VEC_CLASSID,0);

  /* correct */
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidHeaderSpecific(k,KSP_CLASSID,3);
  PetscValidHeaderSpecific(s,SNES_CLASSID,4);

  /* incorrect */
  PetscValidHeader(m,55);
  PetscValidHeader(v,56);
  PetscValidHeader(k,57);
  PetscValidHeader(s,58);

  /* correct */
  PetscValidHeader(m,1);
  PetscValidHeader(v,2);
  PetscValidHeader(k,3);
  PetscValidHeader(s,4);
  PetscFunctionReturn(0);
}

PetscErrorCode testTypes(Mat m, Vec v, KSP k, SNES s)
{
  PetscFunctionBegin;
  /* incorrect */
  PetscValidType(m,-1);
  PetscCheckSameType(m,-1,v,-1);
  PetscCheckSameComm(k,-2,s,-2);
  PetscCheckSameTypeAndComm(m,-3,s,-3);

  /* correct */
  PetscValidType(m,1);
  PetscCheckSameType(m,1,v,2);
  PetscCheckSameComm(k,3,s,4);
  PetscCheckSameTypeAndComm(m,1,s,4);
  PetscFunctionReturn(0);
}
