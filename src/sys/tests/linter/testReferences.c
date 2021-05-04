#include <petscsnes.h>

/* for access to private vec members */
#include <petsc/private/vecimpl.h>

/* foward declare */
void extractFunc(Vec,void**);

void testOutOfLineReference(Vec v, Mat m, KSP k, SNES s)
{
  /* linter should be able to connect all of these to v */
  void *foo = v->data,*bar,*baz;

  bar = v->data;
  extractFunc(v,&baz);

  /* incorrect */
  PetscValidPointer(foo,-1);
  PetscValidPointer(bar,-2);
  PetscValidPointer(baz,-3);

  /* correct */
  PetscValidPointer(foo,1);
  PetscValidPointer(bar,1);
  PetscValidPointer(baz,1);
  return;
}
