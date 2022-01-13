#ifndef PETSC_COMPILER_GUARD_H
#define PETSC_COMPILER_GUARD_H

/*
 This header must be the first inclusion to any top-level PETSc headerfile. A top-level
 headerfile is defined as any headerfile that directly includes petscconf.h
*/

#if defined(__cplusplus)
/*
 This is a 2-part problem:
 1. Intel compilers define __cplusplus to 1 instead of the C++ version (?????????????????)
 2. Intel cannot make their mind up on which of these gives the version of the compiler
*/
#  if defined(__INTEL_COMPILER)
#    define PETSC_INTEL_CXX_VERSION __INTEL_COMPILER
#  elif defined(__ICL)
#    define PETSC_INTEL_CXX_VERSION __ICL
#  elif defined(__ICC)
#    define PETSC_INTEL_CXX_VERSION __ICC
#  elif defined(__ECC)
#    define PETSC_INTEL_CXX_VERSION __ECC
#  endif
#  if defined(PETSC_INTEL_CXX_VERSION)
#    if PETSC_INTEL_CXX_VERSION < 1500 /* ICC/L V 15.0 */
#      error "PETSc requires an Intel Compiler of at least version 15 for C++11"
#    endif
#    undef PETSC_INTEL_CXX_VERSION /* cleanup */
#  elif __cplusplus < 201103L /* C++11 */
#    error "PETSc requires a C++ compiler that defines __cplusplus >= 201103L (C++11)"
#  endif
/*
 Visual Studio C compiler does not define __STDC__ unless it is in C89 mode, even though it
 may support an acceptable subset of C99. So it gets special treatment here.
*/
#elif defined(_MSC_VER)
#  if _MSC_VER < 1900
#    error "PETSc requires a Visual Studio C compiler that defines _MSC_VER >= 1900 (Visual Studio 2015)"
#  endif
#elif defined(__STDC__)
#  if defined(__STDC_VERSION__) /* __STDC_VERSION__ since C94 */
#    if (__STDC__ != 1) || (__STDC_VERSION__ < 199901L) /* C99 */
#      error "PETSc requires a C compiler that defines __STDC__ = 1 and __STDC__VERSION__ >= 199901L (C99)"
#    endif
#  else /* __STDC__VERSION__ */
#    error "PETSc requires a C compiler that defines __STDC__ = 1 and __STDC__VERSION__ >= 199901L (C99)"
#  endif
#endif

#endif /* PETSC_COMPILER_GUARD_H */
