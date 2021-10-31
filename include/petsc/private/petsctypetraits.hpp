#ifndef PETSCTYPETRAITS_HPP
#define PETSCTYPETRAITS_HPP

#include <petsc/private/petscimpl.h> // for PETSC_NODISCARD

#if defined(__cplusplus)

#if __cplusplus >= 201402L // decltype(auto) is c++14
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD decltype(auto) alias(Args&&... args)                  \
  {                                                                     \
    return original(std::forward<Args>(args)...);                       \
  }
#else
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    -> decltype(original(std::forward<Args>(args)...))                  \
  {                                                                     \
    return original(std::forward<Args>(args)...);                       \
  }
#endif // c++14

// A useful template to serve as a function wrapper factory. Given a function "foo" which
// you'd like to thinly wrap as "bar", simply doing:
//
// PETSC_ALIAS_FUNCTION(bar,foo);
//
// creates
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argn)
// {
//   return foo(arg1,arg2,...,argn);
// }
//
// for you. You may then call bar exactly as you would foo.
#define PETSC_ALIAS_FUNCTION(alias,original) PETSC_ALIAS_FUNCTION_(alias,original)

#if __cplusplus >= 201103L // c++11
#include <type_traits>

namespace Petsc
{

namespace util
{

#if __cplusplus >= 201402L // c++14
using std::conditional_t;
using std::remove_const_t;
using std::underlying_type_t;
#else // c++14
template <bool B, class T, class F>
using conditional_t     = typename std::conditional<B,T,F>::type;
template <class T>
using remove_const_t    = typename std::remove_const<T>::type;
template <class T>
using underlying_type_t = typename std::underlying_type<T>::type;
#endif // c++14

template <typename T>
PETSC_STATIC_INLINE constexpr underlying_type_t<T> integral_value(T value)
{
  static_assert(std::is_enum<T>::value,"");
  return static_cast<underlying_type_t<T>>(value);
}

} // namespace util

} // namespace Petsc

#endif // c++11

#endif // __cplusplus

#endif // PETSCTYPETRAITS_HPP
