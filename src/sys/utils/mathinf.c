#include <petscmacros.h>

#if !PetscDefined(SKIP_COMPLEX)
  #define PETSC_SKIP_COMPLEX
#endif

#include <petscsys.h>
/*@
  PetscIsNormalReal - Returns `PETSC_TRUE` if the input value satisfies `isnormal()`

  Input Parameter:
. a - the `PetscReal` Value

  Level: beginner

  Developer Notes:
  Uses the C99 standard `isnormal()` on systems where they exist.

  Uses `isnormalq()` with `__float128`

  Otherwise always returns true

.seealso: `PetscIsInfReal()`, `PetscIsNanReal()`
@*/
#if PetscDefined(USE_REAL___FLOAT128) || PetscDefined(USE_REAL___FP16)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return PETSC_TRUE;
}
#elif PetscDefined(HAVE_ISNORMAL)
PetscBool PetscIsNormalReal(PetscReal a)
{
  return (bool)(isnormal(a) != 0);
}
#else
PetscBool PetscIsNormalReal(PetscReal a)
{
  return PETSC_TRUE;
}
#endif

#if PetscDefined(HAVE_NO_FINITE_MATH_ONLY)
  #define PETSC_FORCE_NO_FINITE_MATH_ONLY __attribute__((optimize("no-finite-math-only")))
#else
  #define PETSC_FORCE_NO_FINITE_MATH_ONLY
#endif

/*@
  PetscIsInfReal - Returns whether the `PetscReal` input is an infinity value.

  Input Parameter:
. a - the floating point number

  Level: beginner

  Developer Notes:
  Uses the C99 standard `isinf()` on systems where it exists.

  Otherwise uses (a && a/2 == a), note that some optimizing compilers compile out this form, thus removing the check.

.seealso: `PetscIsNormalReal()`, `PetscIsNanReal()`
@*/
#if PetscDefined(USE_REAL___FLOAT128)
PetscBool PetscIsInfReal(PetscReal a)
{
  return (bool)(isinfq(a) != 0);
}
#elif PetscDefined(HAVE_ISINF)
PETSC_FORCE_NO_FINITE_MATH_ONLY PetscBool PetscIsInfReal(PetscReal a)
{
  return (bool)(isinf(a) != 0);
}
#elif PetscDefined(HAVE__FINITE)
  #if PetscDefined(HAVE_FLOAT_H)
    #include <float.h> /* Microsoft Windows defines _finite() in float.h */
  #endif
  #if PetscDefined(HAVE_IEEEFP_H)
    #include <ieeefp.h> /* Solaris prototypes these here */
  #endif
PetscBool PetscIsInfReal(PetscReal a)
{
  return (bool)(_finite(a) == 0);
}
#else
PetscBool PetscIsInfReal(PetscReal a)
{
  return (bool)(a && a / 2 == a);
}
#endif

/*@
  PetscIsNanReal - Returns whether the `PetscReal` input is a Not-a-Number (NaN) value.

  Input Parameter:
. a - the floating point number

  Level: beginner

  Developer Notes:
  Uses the C99 standard `isnan()` on systems where it exists.

  Otherwise uses (a != a), note that some optimizing compilers compile
  out this form, thus removing the check.

.seealso: `PetscIsNormalReal()`, `PetscIsInfReal()`
@*/
#if PetscDefined(USE_REAL___FLOAT128)
PetscBool PetscIsNanReal(PetscReal a)
{
  return (bool)(isnanq(a) != 0);
}
#elif PetscDefined(HAVE_ISNAN)
PETSC_FORCE_NO_FINITE_MATH_ONLY PetscBool PetscIsNanReal(PetscReal a)
{
  return (bool)(isnan(a) != 0);
}
#elif PetscDefined(HAVE__ISNAN)
  #if PetscDefined(HAVE_FLOAT_H)
    #include <float.h> /* Microsoft Windows defines _isnan() in float.h */
  #endif
  #if PetscDefined(HAVE_IEEEFP_H)
    #include <ieeefp.h> /* Solaris prototypes these here */
  #endif
PetscBool PetscIsNanReal(PetscReal a)
{
  return (bool)(_isnan(a) != 0);
}
#else
PetscBool PetscIsNanReal(PetscReal a)
{
  return (bool)(a != a);
}
#endif
