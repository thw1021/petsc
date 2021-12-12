#include <petscsys.h>

typedef int testType;

/*@C
  testWellFormedFunctionDocString - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
  incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia
  deserunt mollit anim id est laborum.

  Not Collective, Synchronous

  Input Parameters:
+ viewer - a PetscViewer
- x      - an int

  Output Parameter:
+ viewer - a PetscViewer
- y      - a pointer

  Level: beginner

  References:
  Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

.seealso: testIllFormedFunctionDocString(), testType
C@*/
PetscErrorCode testWellFormedFunctionDocString(PetscViewer viewer, PetscInt x, PetscScalar *y)
{
  return 0;
}

/*@C Lorem ipsum dolor sit amet
  someOtherFunctionName - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
  eiusmod tempor incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat
  non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

  Not Collective, Synchronous

   Input Parameters:
+ viewer - a PetscViewer

  Output Parameter:
+ y - a pointer
- z - a nonexistent parameter

  level: Lorem ipsum dolor sit amet

  Level:
  Beginner

  References: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

.seealso: testNonExistentFunction(), testNonExistentType
@*/

PetscErrorCode testIllFormedFunctionDocString(PetscViewer viewer, PetscInt x, PetscScalar *y)
{
  return 0;
}

/*
  Not Collective, Synchronous

  References:
  Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
.seealso: testNonExistentFunction(), testNonExistentType
*/
PetscErrorCode testIllFormedMinimalDocString(void)
{
  return 0;
}

/*@C
  testTerbleSpelingDocstring - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
  eiusmod tempor incididunt ut labore et dolore magna aliqua.

  input prametirs:
+ viewer - a PetsViewer
- x      - a PetscInt

  output Psrammmetrs:
. y - a PetscScalar pointer

  lvl: itnmediate

.zeeakso: testNonExistentFunction(), testNonExistentType
C@*/
PetscErrorCode testTerribleSpellingDocString(PetscViewer viewer, PetscInt x, PetscScalar *y)
{
  return 0;
}

/* a random comment above a funciton */
void function();
