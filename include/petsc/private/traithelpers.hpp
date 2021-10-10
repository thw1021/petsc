#ifndef PETSCTRAITHELPERS_HPP
#define PETSCTRAITHELPERS_HPP

#include <petsc/private/petscimpl.h>

#if defined(__cplusplus)

// A useful template to serve as a function wrapper factory. Given a function "foo" which
// you'd like to thinly wrap as "bar", simply doing:
//
// ALIAS_FUNCTION(bar,foo);
//
// essentially creates
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argn)
// { return foo(arg1,arg2,...,argn);}
//
// for you. You may then call bar exactly as you would foo.
#define PETSC_ALIAS_FUNCTION(alias,original) PETSC_ALIAS_FUNCTION_(alias,original)
#if PetscDefined(HAVE_CXX_DIALECT_CXX14)
// decltype(auto) is c++14
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD decltype(auto) alias(Args&&... args)                  \
  { return original(std::forward<Args>(args)...);}
#else
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    -> decltype(original(std::forward<Args>(args)...))                  \
  { return original(std::forward<Args>(args)...);}
#endif // PetscDefined(HAVE_CXX_DIALECT_CXX14)

#define PETSC_DEFINE_AUTO_DECLTYPE(type) decltype(type) type

#endif /* __cplusplus */

#endif /* PETSCTRAITHELPERS_HPP */
