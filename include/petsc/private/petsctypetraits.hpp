#ifndef PETSCTYPETRAITS_HPP
#define PETSCTYPETRAITS_HPP

#include <petsc/private/petscimpl.h> // for PETSC_NODISCARD

#if defined(__cplusplus)

#if __cplusplus >= 201103L // c++11
#include <type_traits>
#include <tuple>

namespace Petsc
{

namespace util
{

#if __cplusplus >= 201402L // c++14
using std::conditional_t;
using std::remove_const_t;
using std::underlying_type_t;
using std::make_index_sequence;
#else // c++14
namespace detail
{

// index sequence only
template <std::size_t...> struct index_sequence { };

template <std::size_t N, std::size_t... rest>
struct index_sequence_impl : index_sequence_impl<N-1U,N-1U,rest...>
{ };

template <std::size_t... rest>
struct index_sequence_impl<0U,rest...>
{
  using type = index_sequence<rest...>;
};

} // namespace detail

template <bool B, class T, class F>
using conditional_t     = typename std::conditional<B,T,F>::type;
template <class T>
using remove_const_t    = typename std::remove_const<T>::type;
template <class T>
using underlying_type_t = typename std::underlying_type<T>::type;
template <std::size_t N>
using make_index_sequence = typename detail::index_sequence_impl<N>::type;
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

// A useful template to serve as a function wrapper factory. Given a function "foo" which
// you'd like to thinly wrap as "bar", then:
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

#define PETSC_ALIAS_FUNCTION(alias,original) PETSC_ALIAS_FUNCTION_(alias,original)

// Similar to PETSC_ALIAS_FUNCTION() this macro creates a thin wrapper which passes all
// arguments to the target function ~except~ the last N arguments. So
//
// PETSC_ALIAS_FUNCTION_GOBBLE_NTH_ARGS(bar,foo,3);
//
// creates a function with the effect of
//
// returnType bar(argType1 arg1, argType2 arg2, ..., argTypeN argN)
// {
//   IGNORE(argN);
//   IGNORE(argN-1);
//   IGNORE(argN-2);
//   return foo(arg1,arg2,...,argN-3);
// }
//
// for you.
#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,N)    \
  template <typename Tuple, std::size_t... idx>                         \
  auto original ## _gobbler__(Tuple &&tuple,                            \
                              detail::index_sequence<idx...>)           \
    -> decltype(original(std::get<idx>(tuple)...))                      \
  {                                                                     \
    return original(std::get<idx>(tuple)...);                           \
  };                                                                    \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    -> decltype(original ## _gobbler__(                                 \
                  std::forward_as_tuple(args...),                       \
                  detail::make_index_sequence<sizeof...(Args)-(N)>))    \
  {                                                                     \
    using seq = detail::make_index_sequence<sizeof...(Args)-(N)>;       \
    using std::forward_as_tuple;                                        \
    return original ## _gobbler__(forward_as_tuple(args...),seq());     \
  }

#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(alias,original,N)  \
  PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,N)

#endif // __cplusplus

#endif // PETSCTYPETRAITS_HPP
