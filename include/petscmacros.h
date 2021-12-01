#ifndef PETSC_PREPROCESSOR_MACROS_H
#define PETSC_PREPROCESSOR_MACROS_H

#include <petscconf.h>
#include <petscconf_poison.h> /* for PetscDefined() error checking */

/*MC
  PetscHasAttribute - Determine whether a particular __attribute__ is supported by the compiler

  Synopsis:
  #include <petscmacros.h>
  boolean PetscHasAttribute(name)

  Input Parameter:
. name - the name of the attribute to test

  Notes:
  name should be identical to what you might pass to the __attribute__ declaration itself --
  plain, unbroken text.

  As PetscHasAttribute() is wrapper over the function-like macro __has_attribute(), the exact
  type and value returned is implementation defined. In practice however, it usually returns
  the integer literal 1 if the attribute is supported, and integer literal 0 if the attribute
  is not supported.

  Example Usage:
  Typical usage is using the preprocessor

.vb
  #if PetscHasAttribute(always_inline)
  #  define MY_ALWAYS_INLINE __attribute__((always_inline))
  #else
  #  define MY_ALWAYS_INLINE
  #endif

  void foo(void) MY_ALWAYS_INLINE;
.ve

  but it can also be used in regular code

.vb
  if (PetscHasAttribute(some_attribute)) {
    foo();
  } else {
    bar();
  }
.ve

  Level: advanced

.seealso: PetscDefined(), PetscLikely(), PetscUnlikely()
M*/
#if defined(__has_attribute)
#  define PetscHasAttribute(name) __has_attribute(name)
#else
#  define PetscHasAttribute(name) 0
#endif

#define PETSCPP_STRINGIZE_(a) #a
#define PETSCPP_STRINGIZE(a)  PETSCPP_STRINGIZE_(a)

/*MC
  PETSCPP_CONCAT - Concatenate two tokens

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_CONCAT(x, y)

  Input Parameters:
+ x - First token
- y - Second token

  Notes:
  Not available from Fortran

  PETSCPP_CONCAT() will expand both arguments before pasting them together, use PETSCPP_CONCAT_()
  if you don't want to expand them

  Example usage:
.vb
  PETSCPP_CONCAT(hello,there) -> hellothere

  #define HELLO hello
  PETSCPP_CONCAT(HELLO,there)  -> hellothere
  PETSCPP_CONCAT_(HELLO,there) -> HELLOthere
.ve

  Level: beginner

.seealso: PETSCPP_IF(), PETSCPP_IF_PETSC_DEFINED()
MC*/
#define PETSCPP_CONCAT_(x,y) x ## y
#define PETSCPP_CONCAT(x,y)  PETSCPP_CONCAT_(x,y)

#define PETSCPP_INTERNAL_COMPL_0 1
#define PETSCPP_INTERNAL_COMPL_1 0

/*MC
  PETSCPP_COMPL - Expands to the integer complement of its argument

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_COMPL(b)

  Input Parameter:
. b - Preprocessor variable, must expand to either 0 or 1

  Notes:
  Not available from Fortran

  Expands to integer literal 0 if b expands to 1, or integer literal 1 if b expands to
  0. PETSCPP_COMPL() will expand its argument before returning the complement

  Example usage:
.vb
  #define MY_VAR 1
  PETSCPP_COMPL(MY_VAR) -> 0

  #define MY_VAR 0
  PETSCPP_COMPL(MY_VAR) -> 1
.ve

  Level: beginner

.seealso: PETSCPP_IF(), PETSCPP_CONCAT(), PETSCPP_IF_PETSC_DEFINED()
MC*/
#define PETSCPP_COMPL(b) PETSCPP_CONCAT_(PETSCPP_INTERNAL_COMPL_,b)

#if defined(__cplusplus)
/* icc (and for that matter any windows compiler) is only fully compliant to the letter of the
 * standard up to C++98, so they always set __cplusplus to 199711L (even if they secretly
 * support the vast majority of later standards features). Hence we can't just use the value of
 * __cplusplus
 */
#  if (__cplusplus >= 201103L) || defined(PETSC_HAVE_CXX_DIALECT_CXX11) /* C++11 */
#    define PETSC_HAVE_VARIADIC_MACROS 1
#  endif
#elif defined(__STDC_VERSION__)
#  if __STDC_VERSION__ >= 199901L /* C99 */
#    define PETSC_HAVE_VARIADIC_MACROS 1
#  endif
#endif

#if defined(PETSC_SKIP_VARIADIC_MACROS)
#  undef PETSC_HAVE_VARIADIC_MACROS
#endif

#if defined(PETSC_HAVE_VARIADIC_MACROS)
/*MC
  PetscDefined - Determine whether a boolean macro is defined

  Synopsis:
  #include <petscmacros.h>
  int PetscDefined(def)

  Input Parameter:
. def - PETSc-style preprocessor variable (without PETSC_ prepended!)

  Outut Parameter:
. <return-value> either integer literal 0 or integer literal 1

  Notes:
  PetscDefined() returns 1 if and only if "PETSC_ ## def" is defined (but empty) or defined to
  integer literal 1. In all other cases, PetscDefined() returns integer literal 0. Therefore
  this macro should not be used if its argument may be defined to a non-empty value other than
  1.

  The prefix "PETSC_" is automatically prepended to def. To avoid prepending "PETSC_", say to
  add custom checks in user code, one should use PetscDefined_(),

$ #define FooDefined(d) PetscDefined_(PETSCPP_CONCAT(FOO_,d))

  Developer Notes:
  Getting something that works in C and CPP for an arg that may or may not be defined is
  tricky. Here, if we have "#define PETSC_HAVE_BOOGER 1" we match on the placeholder define,
  insert the "0," for arg1 and generate the triplet (0, 1, 0). Then the last step cherry picks
  the 2nd arg (a one). When PETSC_HAVE_BOOGER is not defined, we generate a (... 1, 0) pair,
  and when the last step cherry picks the 2nd arg, we get a zero.

  Our extra expansion via PetscDefined__take_second_expand() is needed with MSVC, which has a
  nonconforming implementation of variadic macros.

  Example Usage:
  Suppose you would like to call either "foo()" or "bar()" depending on whether PETSC_USE_DEBUG
  is defined then
.vb
  #if PetscDefined(USE_DEBUG)
    foo();
  #else
    bar();
  #endif
.ve
  or alternatively within normal code
.vb
  if (PetscDefined(USE_DEBUG)) {
    foo();
  } else {
    bar();
  }
.ve
  are equivalent to
.vb
  #if defined(PETSC_USE_DEBUG)
  #  if MY_DETECT_EMPTY_MACRO(PETSC_USE_DEBUG)
       foo();
  #   elif PETSC_USE_DEBUG == 1
       foo();
  #   else
       bar();
  #  endif
  #else
  bar();
  #endif
.ve

  Level: advanced

.seealso: PetscHasAttribute(), PetscUnlikely(), PetscLikely(), PETSCPP_CONCAT(),
PETSCPP_EXPAND_TO_NOTHING()
MC*/
#define PetscDefined_arg_1 shift,
#define PetscDefined_arg_  shift,
#define PetscDefined__take_second_expanded(ignored, val, ...) val
#define PetscDefined__take_second_expand(args) PetscDefined__take_second_expanded args
#define PetscDefined__take_second(...) PetscDefined__take_second_expand((__VA_ARGS__))
#define PetscDefined__(arg1_or_junk)   PetscDefined__take_second(arg1_or_junk 1, 0, at_)
#define PetscDefined_(value)           PetscDefined__(PETSCPP_CONCAT_(PetscDefined_arg_,value))
#define PetscDefined(def)              PetscDefined_(PETSCPP_CONCAT(PETSC_,def))
/* what do we do if we dont have variadic macros -> PetscDefined()?????? */


/*MC
  PETSCPP_EXPAND_TO_NOTHING - Expands to absolutely nothing at all

  Synopsis:
  #include <petscmacros.h>
  void PETSCPP_EXPAND_TO_NOTHING(...)

  Input Parameter:
. __VA_ARGS__ - Anything at all

  Notes:
  Not available from Fortran, requires variadic macro support

  Must have at least 1 parameter

  Example usage:
.vb
  #if defined(PETSC_HAVE_VARIADIC_MACROS)
  PETSCPP_EXPAND_TO_NOTHING(a,b,c) -> *nothing*
  #endif
.ve

  Level: advanced

.seealso: PETSCPP_IF(), PETSCPP_CONCAT(), PetscDefined()
MC*/
#define PETSCPP_EXPAND_TO_NOTHING(...)

#define PETSCPP_IF_INTERNAL_0(result_if_true,...) __VA_ARGS__
#define PETSCPP_IF_INTERNAL_1(result_if_true,...) result_if_true

/*MC
  PETSCPP_IF - Conditionally expand to the second or remaining args

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_IF(cond, result_if_true, ...)

  Input Parameters:
+ cond           - Preprocessor conditional, must be defined and expand to either 0 or 1
. result_if_true - Result of macro expansion if cond expands to 1
- __VA_ARGS__    - Result of macro expansion if cond expands to 0

  Notes:
  Not available from Fortran, requires variadic macro support

  Must have at least 1 argument for __VA_ARGS__

  Example usage:
.vb
  #if defined(PETSC_HAVE_VARIADIC_MACROS)
  void myFunction(int,char*);
  #define MY_VAR 1
  PETSCPP_IF(MY_VAR,"hello","goodbye") -> "hello"
  PETSCPP_IF(MY_VAR,myFunction,PETSCPP_EXPAND_TO_NOTHING)(1,"hello") -> myFunction(1,"hello")

  #define MY_VAR 0
  PETSCPP_IF(MY_VAR,"hello",func<type1,type2>()) -> func<type1,type2>()
  PETSCPP_IF(MY_VAR,myFunction,PETSCPP_EXPAND_TO_NOTHING)(1,"hello") -> *nothing*
  #endif
.ve

  Level: advanced

.seealso: PETSCPP_IF_PETSC_DEFINED(), PETSCPP_CONCAT(), PETSCPP_EXPAND_TO_NOTHING(), PETSCPP_NOT()
MC*/
#define PETSCPP_IF(cond,result_if_true,...) PETSCPP_CONCAT_(PETSCPP_IF_INTERNAL_,cond)(result_if_true,__VA_ARGS__)

/*MC
  PETSCPP_IF_PETSC_DEFINED - Like PETSCPP_IF(), but passes cond through PetscDefined() first

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_IF_PETSC_DEFINED(cond, result_if_true, ...)

  Input Parameters:
+ cond           - Condition passed to PetscDefined(), may be undefined
. result_if_true - Result of macro expansion if PetscDefined(cond) expands to 1
- __VA_ARGS__    - Result of macro expansion if PetscDefined(cond) expands to 0

  Notes:
  Not available in Fortran, requires variadic macro support

  Example usage:
.vb
  #if defined(PETSC_HAVE_VARIADIC_MACROS)
  #define PETSC_HAVE_THING 1
  PETSCPP_IF_PETSC_DEFINED(HAVE_THING,"have thing!","don't have thing") -> "have thing!"

  #undef PETSC_HAVE_THING
  PETSCPP_IF_PETSC_DEFINED(HAVE_THING,"have thing!","don't have thing") -> "don't have thing"
  #endif
.ve

  Level: advanced

.seealso: PETSCPP_IF(), PetscDefined(), PETSCPP_CONCAT(), PETSCPP_NOT()
MC*/
#define PETSCPP_IF_PETSC_DEFINED(cond,result_if_true,...) PETSCPP_IF(PetscDefined(cond),result_if_true,__VA_ARGS__)

#endif /* PETSC_HAVE_VARIADIC_MACROS */

#endif /* PETSC_PREPROCESSOR_MACROS_H */
