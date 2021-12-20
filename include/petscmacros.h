#ifndef PETSC_PREPROCESSOR_MACROS_H
#define PETSC_PREPROCESSOR_MACROS_H

#include <petscconf.h>
#include <petscconf_poison.h> /* for PetscDefined() error checking */

/* ========================================================================== */
/* This facilitates using the C version of PETSc from C++ and the C++ version from C. */
#if defined(__cplusplus)
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_CXX
#else
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_C
#endif

/* ========================================================================== */
/* Since PETSc manages its own extern "C" handling users should never include PETSc include
 * files within extern "C". This will generate a compiler error if a user does put the include
 * file within an extern "C".
 */
#if defined(__cplusplus)
void assert_never_put_petsc_headers_inside_an_extern_c(int); void assert_never_put_petsc_headers_inside_an_extern_c(double);
#endif

#if defined(__cplusplus)
#  define PETSC_RESTRICT PETSC_CXX_RESTRICT
#else
#  define PETSC_RESTRICT PETSC_C_RESTRICT
#endif

#if defined(__cplusplus)
#  define PETSC_INLINE PETSC_CXX_INLINE
#else
#  define PETSC_INLINE PETSC_C_INLINE
#endif

#define PETSC_STATIC_INLINE static PETSC_INLINE

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES) /* For Win32 shared libraries */
#  define PETSC_DLLEXPORT __declspec(dllexport)
#  define PETSC_DLLIMPORT __declspec(dllimport)
#  define PETSC_VISIBILITY_INTERNAL
#elif defined(__cplusplus) && defined(PETSC_USE_VISIBILITY_CXX)
#  define PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define PETSC_DLLIMPORT __attribute__((visibility ("default")))
#  define PETSC_VISIBILITY_INTERNAL __attribute__((visibility ("hidden")))
#elif !defined(__cplusplus) && defined(PETSC_USE_VISIBILITY_C)
#  define PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define PETSC_DLLIMPORT __attribute__((visibility ("default")))
#  define PETSC_VISIBILITY_INTERNAL __attribute__((visibility ("hidden")))
#else
#  define PETSC_DLLEXPORT
#  define PETSC_DLLIMPORT
#  define PETSC_VISIBILITY_INTERNAL
#endif

#if defined(petsc_EXPORTS) /* CMake defines this when building the shared library */
#  define PETSC_VISIBILITY_PUBLIC PETSC_DLLEXPORT
#else  /* Win32 users need this to import symbols from petsc.dll */
#  define PETSC_VISIBILITY_PUBLIC PETSC_DLLIMPORT
#endif

/* Functions tagged with PETSC_EXTERN in the header files are always defined as extern "C" when
 * compiled with C++ so they may be used from C and are always visible in the shared libraries
 */
#if defined(__cplusplus)
#  define PETSC_EXTERN         extern "C" PETSC_VISIBILITY_PUBLIC
#  define PETSC_EXTERN_TYPEDEF extern "C"
#  define PETSC_INTERN         extern "C" PETSC_VISIBILITY_INTERNAL
#else
#  define PETSC_EXTERN         extern PETSC_VISIBILITY_PUBLIC
#  define PETSC_EXTERN_TYPEDEF
#  define PETSC_INTERN         extern PETSC_VISIBILITY_INTERNAL
#endif

#if defined(PETSC_USE_SINGLE_LIBRARY)
#  define PETSC_SINGLE_LIBRARY_INTERN PETSC_INTERN
#else
#  define PETSC_SINGLE_LIBRARY_INTERN PETSC_EXTERN
#endif

/* C++11 features */
#if defined(__cplusplus) && defined(PETSC_HAVE_CXX_DIALECT_CXX11)
#  define PETSC_NULLPTR             nullptr
#  define PETSC_CONSTEXPR           constexpr
#  define PETSC_NOEXCEPT            noexcept
#  define PETSC_NOEXCEPT_ARG(cond_) noexcept(cond_)
#else
#  define PETSC_NULLPTR             NULL
#  define PETSC_CONSTEXPR
#  define PETSC_NOEXCEPT
#  define PETSC_NOEXCEPT_ARG(cond_)
#endif

/* C++14 features */
#if defined(PETSC_HAVE_CXX_DIALECT_CXX14)
#  define PETSC_CONSTEXPR_14 PETSC_CONSTEXPR
#else
#  define PETSC_CONSTEXPR_14
#endif

/* C++17 features */
/* We met cases that the host CXX compiler (say mpicxx) supports C++17, but nvcc does not
 * agree, even with -ccbin mpicxx! */
#if defined(__cplusplus) && defined(PETSC_HAVE_CXX_DIALECT_CXX17) && (!defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_CUDA_DIALECT_CXX17))
#  define PETSC_NODISCARD    [[nodiscard]]
#  define PETSC_CONSTEXPR_17 PETSC_CONSTEXPR
#else
#  define PETSC_NODISCARD
#  define PETSC_CONSTEXPR_17
#endif

#include <petscversion.h>
#define PETSC_AUTHOR_INFO  "       The PETSc Team\n    petsc-maint@mcs.anl.gov\n https://petsc.org/\n"

/*MC
  PetscHasAttribute - Determine whether a particular __attribute__ is supported by the compiler

  Synopsis:
  #include <petscmacros.h>
  boolean PetscHasAttribute(name)

  Input Parameter:
. name - The name of the attribute to test

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

  Level: intermediate

.seealso: PetscDefined(), PetscLikely(), PetscUnlikely()
M*/
#if defined(__has_attribute)
#  define PetscHasAttribute(name) __has_attribute(name)
#else
#  define PetscHasAttribute(name) 0
#endif

/*MC
  PetscUnlikely - Hints the compiler that the given condition is usually FALSE

  Synopsis:
  #include <petscmacros.h>
  bool PetscUnlikely(bool cond)

  Not Collective

  Input Parameter:
. cond - Boolean expression

  Notes:
  Not available from fortran.

  This returns the same truth value, it is only a hint to compilers that the result of cond is
  unlikely to be true.

  Example usage:
.vb
  if (PetscUnlikely(cond)) {
    foo(); // cold path
  } else {
    bar(); // hot path
  }
.ve

  Level: advanced

.seealso: PetscLikely(), PetscUnlikelyDebug(), CHKERRQ, PetscDefined(), PetscHasAttribute()
M*/

/*MC
  PetscLikely - Hints the compiler that the given condition is usually TRUE

  Synopsis:
  #include <petscmacros.h>
  bool PetscLikely(bool cond)

  Not Collective

  Input Parameter:
. cond - Boolean expression

  Notes:
  Not available from fortran.

  This returns the same truth value, it is only a hint to compilers that the result of cond is
  likely to be true.

  Example usage:
.vb
  if (PetscLikely(cond)) {
    foo(); // hot path
  } else {
    bar(); // cold path
  }
.ve

  Level: advanced

.seealso: PetscUnlikely(), PetscDefined(), PetscHasAttribute()
M*/
#if defined(PETSC_HAVE_BUILTIN_EXPECT)
#  define PetscUnlikely(cond) __builtin_expect(!!(cond),0)
#  define PetscLikely(cond)   __builtin_expect(!!(cond),1)
#else
#  define PetscUnlikely(cond) (cond)
#  define PetscLikely(cond)   (cond)
#endif

#if defined(__GNUC__)
/* GCC 4.8+, Clang, Intel and other compilers compatible with GCC (-std=c++0x or above) */
#  define PetscUnreachable_() __builtin_unreachable()
#elif defined(_MSC_VER) /* MSVC */
#  define PetscUnreachable_() __assume(0)
#else /* ??? */
#  define PetscUnreachable_() SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Code path explicitly marked as unreachable executed")
#endif

/*MC
  PetscUnreachable() - Indicate to the compiler that a code-path is logically unreachable

  Synopsis:
  #include <petscmacros.h>
  void PetscUnreachable(void)

  Notes:
  Indicates to the compiler (usually via some built-in) that a particular code path is always
  unreachable. Behavior is undefined if this function is ever executed, the user can expect an
  unceremonious crash.

  Example usage:
  Useful in situations such as switches over enums where not all enumeration values are
  explicitly covered by the switch

.vb
  typedef enum {RED, GREEN, BLUE} Color;

  int foo(Color c)
  {
    // it is known to programmer (or checked previously) that c is either RED or GREEN
    // but compiler may not be able to deduce this and/or emit spurious warnings
    switch (c) {
      case RED:
        return bar();
      case GREEN:
        return baz();
      default:
        PetscUnreachable(); // program is ill-formed if executed
    }
  }
.ve

  Level: advanced

.seealso: SETERRABORT(), PETSCABORT()
MC*/
#define PetscUnreachable() PetscUnreachable_()

/*MC
  PETSCPP_EXPAND - Expand arguments

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_EXPAND(x)

  Input Paramter:
. x - The preprocessor token to expand

.seealso: PETSCPP_STRINGIZE(), PETSCPP_CONCAT()
MC*/
#define PETSCPP_EXPAND(x) x

/*MC
  PETSCPP_STRINGIZE - Stringize a token

  Synopsis:
  #include <petscmacros.h>
  const char* PETSCPP_STRINGIZE(x)

  Input Parameter:
. x - The token you would like to stringize

  Output Parameter:
. <return-value> - The string representation of x

  Notes:
  Not available from Fortran.

  PETSCPP_STRINGIZE() expands x before stringizing it, if you do not wish to do so, use
  PETSCPP_STRINGIZE_() instead.

  Example Usage:
.vb
  #define MY_OTHER_VAR hello there
  #define MY_VAR       MY_OTHER_VAR

  PETSCPP_STRINGIZE(MY_VAR)  -> "hello there"
  PETSCPP_STRINGIZE_(MY_VAR) -> "MY_VAR"

  int foo;
  PETSCPP_STRINGIZE(foo)  -> "foo"
  PETSCPP_STRINGIZE_(foo) -> "foo"
.ve

  Level: beginner

.seealso: PETSCPP_CONCAT(), PETSCPP_EXPAND_TO_NOTHING(), PETSCPP_IF(), PETSCPP_EXPAND()
MC*/
#define PETSCPP_STRINGIZE_(x) #x
#define PETSCPP_STRINGIZE(x)  PETSCPP_STRINGIZE_(x)

/*MC
  PETSCPP_CONCAT - Concatenate two tokens

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_CONCAT(x, y)

  Input Parameters:
+ x - First token
- y - Second token

  Notes:
  Not available from Fortran.

  PETSCPP_CONCAT() will expand both arguments before pasting them together, use PETSCPP_CONCAT_()
  if you don't want to expand them.

  Example usage:
.vb
  PETSCPP_CONCAT(hello,there) -> hellothere

  #define HELLO hello
  PETSCPP_CONCAT(HELLO,there)  -> hellothere
  PETSCPP_CONCAT_(HELLO,there) -> HELLOthere
.ve

  Level: beginner

.seealso: PETSCPP_IF(), PETSCPP_IF_PETSC_DEFINED(), PETSCPP_STRINGIZE(), PETSCPP_EXPAND()
MC*/
#define PETSCPP_CONCAT_(x,y) x ## y
#define PETSCPP_CONCAT(x,y)  PETSCPP_CONCAT_(x,y)

#define PETSCPP_INTERNAL_COMPL_0 1
#define PETSCPP_INTERNAL_COMPL_1 0

/*MC
  PETSCPP_COMPL - Expands to the integer complement of its argument

  Synopsis:
  #include <petscmacros.h>
  int PETSCPP_COMPL(b)

  Input Parameter:
. b - Preprocessor variable, must expand to either integer literal 0 or 1

  Output Paramter:
. <return-value> - Either integer literal 0 or 1

  Notes:
  Not available from Fortran.

  Expands to integer literal 0 if b expands to 1, or integer literal 1 if b expands to
  0. Behaviour is undefined if b expands to anything else. PETSCPP_COMPL() will expand its
  argument before returning the complement.

  This macro can be useful for negating PetscDefined() inside macros e.g.

$ #define PETSC_DONT_HAVE_FOO PETSCPP_COMPL(PetscDefined(HAVE_FOO))

  Example usage:
.vb
  #define MY_VAR 1
  PETSCPP_COMPL(MY_VAR) -> 0

  #undef  MY_VAR
  #define MY_VAR 0
  PETSCPP_COMPL(MY_VAR) -> 1
.ve

  Level: beginner

.seealso: PETSCPP_IF(), PETSCPP_CONCAT(), PETSCPP_IF_PETSC_DEFINED(), PetscDefined()
MC*/
#define PETSCPP_COMPL(b) PETSCPP_CONCAT_(PETSCPP_INTERNAL_COMPL_,PETSCPP_EXPAND(b))

#if !defined(PETSC_SKIP_VARIADIC_MACROS)
/*MC
  PetscDefined - Determine whether a boolean macro is defined

  Synopsis:
  #include <petscmacros.h>
  int PetscDefined(def)

  Input Parameter:
. def - PETSc-style preprocessor variable (without PETSC_ prepended!)

  Outut Parameter:
. <return-value> - Either integer literal 0 or 1

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  PetscDefined() returns 1 if and only if "PETSC_ ## def" is defined (but empty) or defined to
  integer literal 1. In all other cases, PetscDefined() returns integer literal 0. Therefore
  this macro should not be used if its argument may be defined to a non-empty value other than
  1.

  The prefix "PETSC_" is automatically prepended to def. To avoid prepending "PETSC_", say to
  add custom checks in user code, one should use PetscDefined_().

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

  // or alternatively within normal code
  if (PetscDefined(USE_DEBUG)) {
    foo();
  } else {
    bar();
  }
.ve

  is equivalent to

.vb
  #if defined(PETSC_USE_DEBUG)
  #  if MY_DETECT_EMPTY_MACRO(PETSC_USE_DEBUG) // assuming you have such a macro
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

  Level: intermediate

.seealso: PetscHasAttribute(), PetscUnlikely(), PetscLikely(), PETSCPP_CONCAT(),
PETSCPP_EXPAND_TO_NOTHING(), PETSCPP_COMPL()
MC*/
#define PetscDefined_arg_1 shift,
#define PetscDefined_arg_  shift,
#define PetscDefined__take_second_expanded(ignored, val, ...) val
#define PetscDefined__take_second_expand(args) PetscDefined__take_second_expanded args
#define PetscDefined__take_second(...) PetscDefined__take_second_expand((__VA_ARGS__))
#define PetscDefined__(arg1_or_junk)   PetscDefined__take_second(arg1_or_junk 1, 0, at_)
#define PetscDefined_(value)           PetscDefined__(PETSCPP_CONCAT_(PetscDefined_arg_,value))
#define PetscDefined(def)              PetscDefined_(PETSCPP_CONCAT(PETSC_,def))

/*MC
  PetscUnlikelyDebug - Hints the compiler that the given condition is usually FALSE, eliding
  the check in optimized mode

  Synopsis:
  #include <petscmacros.h>
  bool PetscUnlikelyDebug(bool cond)

  Not Collective

  Input Parameters:
. cond - Boolean expression

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  This returns the same truth value, it is only a hint to compilers that the result of cond is
  likely to be false. When PETSc is compiled in optimized mode this will always return
  false. Additionally, cond is guaranteed to not be evaluated when PETSc is compiled in
  optimized mode.

  Example usage:
  This routine is shorthand for checking both the condition and whether PetscDefined(USE_DEBUG)
  is true. So

.vb
  if (PetscUnlikelyDebug(cond)) {
    foo();
  } else {
    bar();
  }
.ve

  is equivalent to

.vb
  if (PetscDefined(USE_DEBUG)) {
    if (PetscUnlikely(cond)) {
      foo();
    } else {
      bar();
    }
  } else {
    bar();
  }
.ve

  Level: advanced

.seealso: PetscUnlikely(), PetscLikely(), CHKERRQ, SETERRQ
M*/
#define PetscUnlikelyDebug(cond) (PetscDefined(USE_DEBUG) && PetscUnlikely(cond))

/*MC
  PETSCPP_EXPAND_TO_NOTHING - Expands to absolutely nothing at all

  Synopsis:
  #include <petscmacros.h>
  void PETSCPP_EXPAND_TO_NOTHING(...)

  Input Parameter:
. __VA_ARGS__ - Anything at all

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  Must have at least 1 parameter.

  Example usage:
.vb
  PETSCPP_EXPAND_TO_NOTHING(a,b,c) -> *nothing*
.ve

  Level: beginner

.seealso: PETSCPP_IF(), PETSCPP_CONCAT(), PetscDefined(), PETSCPP_STRINGIZE(), PETSCPP_EXPAND()
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
+ cond           - Preprocessor conditional
. result_if_true - Result of macro expansion if cond expands to 1
- __VA_ARGS__    - Result of macro expansion if cond expands to 0

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  cond must be defined and expand (not evaluate!) to either integer literal 0 or 1. Must have
  at least 1 argument for __VA_ARGS__, but it may expand empty.

  Example usage:
.vb
  void myFunction(int,char*);
  #define MY_VAR 1
  PETSCPP_IF(MY_VAR,"hello","goodbye") -> "hello"
  PETSCPP_IF(MY_VAR,myFunction,PETSCPP_EXPAND_TO_NOTHING)(1,"hello") -> myFunction(1,"hello")

  #define MY_VAR 0
  PETSCPP_IF(MY_VAR,"hello",func<type1,type2>()) -> func<type1,type2>()
  PETSCPP_IF(MY_VAR,myFunction,PETSCPP_EXPAND_TO_NOTHING)(1,"hello") -> *nothing*
.ve

  Level: intermediate

.seealso: PETSCPP_IF_PETSC_DEFINED(), PETSCPP_CONCAT(), PETSCPP_EXPAND_TO_NOTHING(), PETSCPP_COMPL()
MC*/
#define PETSCPP_IF(cond,result_if_true,...) PETSCPP_CONCAT_(PETSCPP_IF_INTERNAL_,cond)(result_if_true,__VA_ARGS__)

/*MC
  PETSCPP_IF_PETSC_DEFINED - Like PETSCPP_IF(), but passes cond through PetscDefined() first

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PETSCPP_IF_PETSC_DEFINED(cond, result_if_true, ...)

  Input Parameters:
+ cond           - Condition passed to PetscDefined()
. result_if_true - Result of macro expansion if PetscDefined(cond) expands to 1
- __VA_ARGS__    - Result of macro expansion if PetscDefined(cond) expands to 0

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  cond must satisfy all conditions for PetscDefined(). Must have at least 1 argument for
  __VA_ARGS__, but it may expand empty.

  Example usage:
.vb
  #define PETSC_HAVE_FOO 1
  PETSCPP_IF_PETSC_DEFINED(HAVE_FOO,foo,bar) -> foo

  #undef PETSC_HAVE_FOO
  PETSCPP_IF_PETSC_DEFINED(HAVE_FOO,foo,bar,baz,bop) -> bar,baz,bop
.ve

  Level: intermediate

.seealso: PETSCPP_IF(), PetscDefined(), PETSCPP_CONCAT(), PETSCPP_EXPAND(), PETSCPP_COMPL()
MC*/
#define PETSCPP_IF_PETSC_DEFINED(cond,result_if_true,...) PETSCPP_IF(PetscDefined(cond),result_if_true,__VA_ARGS__)
#endif /* !PETSC_SKIP_VARIADIC_MACROS */

#endif /* PETSC_PREPROCESSOR_MACROS_H */
