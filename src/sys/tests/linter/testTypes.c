#include <petscsnes.h>

void testTypes(Mat m, Vec v, KSP k, SNES s)
{
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
  return;
}
