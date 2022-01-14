#include <petscsys.h>

typedef int testType;

/*@C
  testWellFormedFunctionDocString - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
  incididunt ut labore et dolore magna aliqua.

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
- y          - a pointer
+ cnd           - a boolean
. z - a nonexistent parameter

  level: Lorem ipsum dolor sit amet

  Level:
  Beginner

  Developer Notes:
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
  labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa
  qui officia deserunt mollit anim id est laborum as follows:

  Notes Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
  incididunt ut labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident,
  sunt in culpa qui officia deserunt mollit anim id est laborum example.

  Fortran Notes:
  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
  labore et dolore magna aliqua. Excepteur sint occaecat cupidatat non proident, sunt in culpa
  qui officia deserunt mollit anim id est laborum instance:

  References: Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

.seealso:                                                  testNonExistentFunction(), testNonExistentType,
testIllFormedFunctionDocString(), testNonExistentFunction(), testIllFormedMinimalDocString()
@*/

PetscErrorCode testIllFormedFunctionDocString(PetscViewer viewer, PetscInt x, PetscScalar *y, PetscBool cond)
{
  return 0;
}

/*
  Not Collective, Synchronous

  input parms:
. foo

  Output params:
+ bar -

  References:
  Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
   .seealso: testNonExistentFunction(), testNonExistentType,testNonExistentFunction()
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
- x - a PetscInt

  output Psrammmetrs:
. y - a PetscScalar pointer

  optnS dtaaSE:
- -option_a     - foo
- -option_b [filename][:[~]<foo,bar,baz>[:[~]bop]] - descr
  lvl: itnmediate

.zeeakso:
C@*/
PetscErrorCode testTerribleSpellingDocString(PetscViewer viewer, PetscInt x, PetscScalar *y)
{
  return 0;
}

/*@ asdadsadasdas
  testCustomFortranInterfaceDocString - Lorem ipsum dolor sit amet, consectetur adipiscing elit

  Input Parameters:
+ string -  a char pointer
- function_ptr - a function pointer

  Level:

.seealso: Lorem(), ipsum(), dolor(), sit(), amet(), consectetur(), adipiscing(), elit()
@*/
PetscErrorCode testCustomFortranInterfaceDocString(char *******string, PetscErrorCode (*function_ptr)(PetscInt))
{
  return 0;
}

/* a random comment above a function */
void function() { }

PETSC_INTERN PetscErrorCode testInternFunction();

/*@
  testInternFunction - an internal function

  Level: developer

.seealso: function()
@*/
PetscErrorCode testInternFunction()
{
  return 0;
}

/*@
  testStaticFunction - an internal function

  Level: developer

.seealso: function()
@*/
static PetscErrorCode testStaticFunction()
{
  return 0;
}

/*@
  testAllParamsUndocumented - lorem

  Level: beginner developer

  Example Usage:
.vb
  int a;
  double multiline;
  char codeBlock;
.ve

.seealso:
@*/
PetscErrorCode testAllParamsUndocumented(PetscInt a, PetscInt b)
{
  return testStaticFunction();
}

/*@
  testParameterGrouping ipsum

  Input parameters:
- a,b - some params
+ nonExistentParam - this param does not exist
. ... - variadic arguments

  Level dev

.see also: testStaticFunction()
@*/
PetscErrorCode testParameterGrouping(PetscInt a, PetscInt b,...)
{
  return 0;
}

/*@
  testScatteredVerbatimBlocks - bla

  Input Parameters:
+ alpha - an alpha
.vb
  int a_code_block;
.ve
- beta - a beta

  Level: beginner

.seealso: Foo()
@*/
PetscErrorCode testScatteredVerbatimBlocks(PetscInt alpha, PetscInt beta)
{
  return 0;
}

/*@
  testBadParamListDescrSep - foo

  Input Parameters:
+ alpha, an alpha
- beta = a beta

  Level: beginner

.seealso: Foo()
@*/
PetscErrorCode testBadParamListDescrSep(PetscInt alpha, PetscInt beta)
{
  return 0;
}

/*@
  testBadMidSentenceColons - Lorem:

  Notes:
  Lorem ipsum dolor sit amet:, consectetur adipiscing elit: sed do: eiusmod tempor: incididunt ut
  labore et dolore: magna aliqua: Excepteur: sint occaecat cupidatat non proident, sunt: in culpa
  qui officia: deserunt mollit: anim id est: laborum as follows:

  Level: beginner

.seealso: Foo()
@*/
PetscErrorCode testBadMidSentenceColons(void)
{
  return 0;
}
