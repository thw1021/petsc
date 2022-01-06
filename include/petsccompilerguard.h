#ifndef PETSC_COMPILER_GUARD_H
#define PETSC_COMPILER_GUARD_H

/* This header must be the first inclusion to any top-level PETSc headerfile. A top-level
 * headerfile is defined as any headerfile that directly includes petscconf.h
 */

#if defined(__STDC__) && defined(__STDC_VERSION__)
#  if (__STDC__ != 1) || (__STDC_VERSION__ < 199901L) /* C99 */
#    error "PETSc requires a C compiler that defines __STDC__ = 1 and __STDC__VERSION__ >= 199901L (C99)"
#  endif
#elif defined(__cplusplus)
#  if __cplusplus < 201103L /* C++11 */
#    error "PETSc requires a C++ compiler that defines __cplusplus >= 201103L (C++11)"
#  endif
#else /* ??? */
#  error "PETSc requires either a C compiler that defines __STDC__ = 1 and __STDC__VERSION__ >= 199901L (C99) or a C++ compiler that defines __cplusplus >= 201103L (C++11)"
#endif

#endif /* PETSCCOMPILERGUARD_H */
