#ifndef PETSC_MACROS_H
#define PETSC_MACROS_H

#include <petscsys.h>

#if defined(__cplusplus)
#  if __cplusplus >= 201103L /* C++11 */
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

/*MC
  PETSCPP_CONCAT - Concatenate two tokens

  Synopsis:
  #include <petsc/private/macros.h>
  <macro-expansion> PETSCPP_CONCAT(x, y)

  Input parameters:
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
  #include <petsc/private/macros.h>
  <macro-expansion> PETSCPP_COMPL(b)

  Input parameter:
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

#if defined(PETSC_HAVE_VARIADIC_MACROS)
/*MC
  PETSCPP_EXPAND_TO_NOTHING - Expands to absolutely nothing at all

  Synopsis:
  #include <petsc/private/macros.h>
  void PETSCPP_EXPAND_TO_NOTHING(...)

  Input parameter:
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

.seealso: PETSCPP_IF(), PETSCPP_CONCAT()
MC*/
#define PETSCPP_EXPAND_TO_NOTHING(...)

#define PETSCPP_IF_INTERNAL_0(result_if_true,...) __VA_ARGS__
#define PETSCPP_IF_INTERNAL_1(result_if_true,...) result_if_true

/*MC
  PETSCPP_IF - Conditionally expand to the second or remaining args

  Synopsis:
  #include <petsc/private/macros.h>
  <macro-expansion> PETSCPP_IF(cond, result_if_true, ...)

  Input parameters:
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
  #include <petsc/private/macros.h>
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

#endif /* PETSC_MACROS_H */
