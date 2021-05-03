#include <petscsnes.h>

void testValidHeaders(Mat m, Vec v, KSP k, SNES s)
{
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
  return;
}
