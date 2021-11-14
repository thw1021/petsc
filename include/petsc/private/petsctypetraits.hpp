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
using std::remove_pointer_t;
using std::index_sequence;
using std::make_index_sequence;
#else // c++14
template <bool B, class T, class F> using conditional_t = typename std::conditional<B,T,F>::type;
template <class T> using remove_const_t    = typename std::remove_const<T>::type;
template <class T> using underlying_type_t = typename std::underlying_type<T>::type;
template <class T> using remove_pointer_t  = typename std::remove_pointer<T>::type;

// index sequence only
template <std::size_t... idx> struct index_sequence
{
  using value_type = std::size_t;

  static constexpr value_type size() noexcept { return sizeof...(idx); }
};

namespace detail
{

template <std::size_t N, std::size_t... rest>
struct index_sequence_impl : index_sequence_impl<N-1U,N-1U,rest...>
{ };

template <std::size_t... rest>
struct index_sequence_impl<0U,rest...>
{
  using type = index_sequence<rest...>;
};

} // namespace detail

template <std::size_t N> using make_index_sequence = typename detail::index_sequence_impl<N>::type;
#endif // c++14

namespace detail
{

template <typename T, typename U = _p_PetscObject>
struct is_petsc_object_impl : std::false_type { };

template <typename T> struct is_petsc_object_impl<T,PetscObject> : std::true_type { };

template <typename T>
struct is_petsc_object_impl<T,decltype(T::hdr)>
  : conditional_t<
  !std::is_pointer<T>::value && std::is_class<T>::value && std::is_standard_layout<T>::value,
  std::true_type,
  std::false_type
  >
{ };

} // namespace detail

template <typename T> using is_petsc_object = detail::is_petsc_object_impl<remove_pointer_t<T>>;

template <typename T>
PETSC_STATIC_INLINE constexpr underlying_type_t<T> integral_value(T value) noexcept
{
  static_assert(std::is_enum<T>::value,"");
  return static_cast<underlying_type_t<T>>(value);
}

} // namespace util

// define this outside namespace util since it can be universally used
template <typename T>
PETSC_STATIC_INLINE constexpr PetscObject& PetscObjectCast(T& object) noexcept
{
  static_assert(util::is_petsc_object<T>::value,"");
  return reinterpret_cast<PetscObject&>(object);
}

template <typename T>
PETSC_STATIC_INLINE util::remove_const_t<T>& PetscRemoveConstCast(T& object) noexcept
{
  return const_cast<util::remove_const_t<T>&>(object);
}

template <typename T>
PETSC_STATIC_INLINE T& PetscRemoveConstCast(const T& object) noexcept
{
  return const_cast<T&>(object);
}

template <typename T>
PETSC_STATIC_INLINE T*& PetscRemoveConstCast(const T*& object) noexcept
{
  return const_cast<T*&>(object);
}

template <typename T>
PETSC_STATIC_INLINE constexpr const T& PetscAddConstCast(T& object) noexcept
{
  static_assert(!std::is_const<T>::value,"");
  return const_cast<const T&>(std::forward<T>(object));
}

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
#define PETSC_ALIAS_FUNCTION_(alias,original)                           \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    -> decltype(original(std::forward<Args>(args)...))                  \
  {                                                                     \
    return original(std::forward<Args>(args)...);                       \
  }

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
  template <typename tuple_type, std::size_t... idx>                    \
  auto original ## _gobble__(tuple_type &&tuple,                        \
                              Petsc::util::index_sequence<idx...>)      \
    -> decltype(original(std::get<idx>(tuple)...))                      \
  {                                                                     \
    return original(std::get<idx>(tuple)...);                           \
  };                                                                    \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args) -> decltype(               \
    original ## _gobble__(                                              \
      std::forward_as_tuple(args...),                                   \
      Petsc::util::make_index_sequence<sizeof...(Args)-(N)>             \
    )                                                                   \
  )                                                                     \
  {                                                                     \
    static_assert(std::is_integral<decltype(N)>::value && (N) >= 0,""); \
    using seq = Petsc::util::make_index_sequence<sizeof...(Args)-(N)>;  \
    return original ## _gobble__(std::forward_as_tuple(args...),seq()); \
  }

#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(alias,original,N)  \
  PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,N)

// helper macros when declaring class member functions that should be callable from C. Member
// functions need to be static to be callable from C otherwise they get an implicit 'this'
// pointer as the first argument
#define PETSC_CXX_COMPAT_DECL(...) PETSC_NODISCARD static __VA_ARGS__ PETSC_NOEXCEPT
// A corresponding out-of-line definition macro to the one above
#define PETSC_CXX_COMPAT_DEFN(...) PETSC_INLINE __VA_ARGS__ PETSC_NOEXCEPT

#endif // __cplusplus

#endif // PETSCTYPETRAITS_HPP
