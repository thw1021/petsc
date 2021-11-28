#ifndef PETSCTYPETRAITS_HPP
#define PETSCTYPETRAITS_HPP

#include <petsc/private/petscimpl.h> // for PETSC_NODISCARD

// PETSC_CONCAT() - Concatenate two tokens
//
// input params:
// x - first token
// y - second token
//
// notes:
// PETSC_CONCAT() will expand both arguments before pasting them together, use PETSC_CONCAT_()
// if you don't want to expand them
//
// example usage:
// PETSC_CONCAT(hello,there) -> hellothere
//
// #define HELLO hello
//
// PETSC_CONCAT(HELLO,there)  -> hellothere
// PETSC_CONCAT_(HELLO,there) -> HELLOthere
#define PETSC_CONCAT_(x,y) x ## y
#define PETSC_CONCAT(x,y)  PETSC_CONCAT_(x,y)

#if defined(__cplusplus)

#if __cplusplus >= 201103L // C++11
#include <type_traits>
#include <tuple>

// PETSC_EXPAND_TO_NOTHING() - Expands to absolutely nothing at all
//
// input params:
// __VA_ARGS__ - anything at all
//
// notes:
// must have at least 1 argument
//
// example usage:
// PETSC_EXPAND_TO_NOTHING(a,b,c) -> *nothing*
#define PETSC_EXPAND_TO_NOTHING(...)

#define PETSC_IF_INTERNAL_0(result_if_true,...) __VA_ARGS__
#define PETSC_IF_INTERNAL_1(result_if_true,...) result_if_true

// PETSC_IF() - Conditionally expand to the second or remaining args
//
// input params:
// cond           - preprocessor conditional, must expand to either 0 or 1
// result_if_true - result of macro expansion if cond expands to 1
// __VA_ARGS__    - result of macro expansion if cond expands to 0
//
// example usage:
// #define MY_VAR 1
// PETSC_IF(MY_VAR,"hello","goodbye") -> "hello"
//
// #define MY_VAR 0
// PETSC_IF(MY_VAR,"hello",func<type1,type2>()) -> func<type1,type2>()
#define PETSC_IF(cond,result_if_true,...) PETSC_CONCAT(PETSC_IF_INTERNAL_,cond)(result_if_true,__VA_ARGS__)

// PETSC_IF_PETSC_DEFINED() - Like PETSC_IF(), but passes cond through PetscDefined() first
//
// example usage:
// #define PETSC_HAVE_THING 1
// PETSC_IF_PETSC_DEFINED(HAVE_THING,"have thing!","don't have thing") -> "have thing!"
//
// #undef PETSC_HAVE_THING
// PETSC_IF_PETSC_DEFINED(HAVE_THING,"have thing!","don't have thing") -> "don't have thing"
#define PETSC_IF_PETSC_DEFINED(cond,result_if_true,...) PETSC_IF(PetscDefined(cond),result_if_true,__VA_ARGS__)

namespace Petsc
{

namespace util
{

#if __cplusplus >= 201402L // C++14
using std::conditional_t;
using std::remove_const_t;
using std::add_const_t;
using std::underlying_type_t;
using std::remove_pointer_t;
using std::add_pointer_t;
using std::index_sequence;
using std::make_index_sequence;
using std::decay_t;
using std::tuple_element_t;
#if __cplusplus >= 201703L
using std::void_t;
#else // C++17
template <class... T> using void_t = void;
#endif // C++17
#else // C++14
template <bool B, class T, class F> using conditional_t = typename std::conditional<B,T,F>::type;
template <class T> using remove_const_t    = typename std::remove_const<T>::type;
template <class T> using add_const_t       = typename std::add_const<T>::type;
template <class T> using underlying_type_t = typename std::underlying_type<T>::type;
template <class T> using remove_pointer_t  = typename std::remove_pointer<T>::type;
template <class T> using add_pointer_t     = typename std::add_pointer<T>::type;
template <class T> using decay_t           = typename std::decay<T>::type;
template <class... T> using void_t = void;
template <std::size_t I, class T> using tuple_element_t = typename std::tuple_element<I,T>::type;
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
  (!std::is_pointer<T>::value) && (std::is_class<T>::value) && (std::is_standard_layout<T>::value),
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

namespace detail
{

struct can_call_test
{
  template<typename F, typename... A>
  static decltype(std::declval<F>()(std::declval<A>()...),std::true_type()) f(int);

  template<typename F, typename... A> static std::false_type f(...);
};

} // namespace detail

template <typename F, typename... A>
struct can_call : decltype(detail::can_call_test::f<F,A...>(0)) { };

template <typename... A, typename F>
PETSC_STATIC_INLINE constexpr can_call<F,A...> is_callable_with(F&&) noexcept
{
  return can_call<F,A...>{};
}

template <typename... T> struct always_false : std::false_type { };

} // namespace util

template <typename T>
PETSC_STATIC_INLINE constexpr util::remove_const_t<T>& PetscRemoveConstCast(T& object) noexcept
{
  return const_cast<util::remove_const_t<T>&>(object);
}

template <typename T>
PETSC_STATIC_INLINE constexpr T& PetscRemoveConstCast(const T& object) noexcept
{
  return const_cast<T&>(object);
}

template <typename T>
PETSC_STATIC_INLINE constexpr T*& PetscRemoveConstCast(const T*& object) noexcept
{
  return const_cast<T*&>(object);
}

template <typename T>
PETSC_STATIC_INLINE constexpr util::add_const_t<T>& PetscAddConstCast(T& object) noexcept
{
  return const_cast<util::add_const_t<T>&>(std::forward<T>(object));
}

template <typename T>
PETSC_STATIC_INLINE constexpr util::add_const_t<T>*& PetscAddConstCast(T*& object) noexcept
{
  static_assert(!std::is_const<T>::value,"");
  return const_cast<util::add_const_t<T>*&>(std::forward<T>(object));
}


// PetscObjectCast() - Cast an object to PetscObject
//
// input param:
// object - the object to cast
//
// output param:
// [return value] - The resulting PetscObject
//
// notes:
// This function checks that the object passed in is in fact a PetscObject, and hence requires
// the full definition of the object. This means you must include the appropriate header
// containing the _p_<object> struct definition
//
//   not available from Fortran
template <typename T>
PETSC_STATIC_INLINE constexpr PetscObject& PetscObjectCast(T& object) noexcept
{
  static_assert(util::is_petsc_object<T>::value,"Did you forget to include the private header?");
  return reinterpret_cast<PetscObject&>(object);
}

template <typename T>
PETSC_STATIC_INLINE constexpr PetscObject& PetscObjectCast(const T& object) noexcept
{
  return PetscObjectCast(PetscRemoveConstCast(object));
}

} // namespace Petsc

#endif // C++11

#if __cplusplus >= 201103L // C++11
#define PETSC_ALIAS_FUNCTION_(alias,orig,dispatch)                      \
  template <typename... Args>                                           \
  PETSC_STATIC_INLINE auto dispatch(int,Args&&... args)                 \
    noexcept(noexcept(orig(std::forward<Args>(args)...)))               \
    -> decltype(orig(std::forward<Args>(args)...))                      \
  {                                                                     \
    return orig(std::forward<Args>(args)...);                           \
  };                                                                    \
  template <typename... Args>                                           \
  PETSC_STATIC_INLINE int dispatch(char,Args&&... args)                 \
  {                                                                     \
    static_assert(                                                      \
      Petsc::util::is_callable_with<Args...>(orig) &&                   \
      Petsc::util::always_false<Args...>::value,                        \
      PetscStringize(orig) "() is not callable with given arguments"    \
    );                                                                  \
    return EXIT_FAILURE;                                                \
  };                                                                    \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    noexcept(noexcept(dispatch(0,std::forward<Args>(args)...)))         \
    -> decltype(dispatch(0,std::forward<Args>(args)...))                \
  {                                                                     \
    return dispatch(0,std::forward<Args>(args)...);                     \
  }

// PETSC_ALIAS_FUNCTION() - Alias a function
//
// input params:
// alias    - the new name for the function
// original - the name of the function you would like to alias
//
// notes:
// Using this macro in effect creates
//
// template <typename... T>
// auto alias(T&&... args)
// {
//   return original(std::forward<T>(args)...);
// }
//
// meaning it will transparently work for any kind of alias (including overloads).
//
// example usage:
// PETSC_ALIAS_FUNCTION(bar,foo);
#define PETSC_ALIAS_FUNCTION(alias,original)                            \
  PETSC_ALIAS_FUNCTION_(alias,original,PETSC_CONCAT(PETSC_CONCAT(original,_petsc_dispatch_),__LINE__))
#else
#define PETSC_ALIAS_FUNCTION(alias,original)                    \
  static_assert(0,"PETSC_ALIAS_FUNCTION() requires C++11")
#endif

#if __cplusplus >= 201103L // C++11

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
#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,gobblefn,N) \
  template <typename TupleT, std::size_t... idx> PETSC_STATIC_INLINE    \
  auto gobblefn(TupleT&& tuple, Petsc::util::index_sequence<idx...>)    \
    noexcept(noexcept(original(std::get<idx>(tuple)...)))               \
    -> decltype(original(std::get<idx>(tuple)...))                      \
  {                                                                     \
    return original(std::get<idx>(tuple)...);                           \
  };                                                                    \
  template <typename... Args>                                           \
  PETSC_NODISCARD auto alias(Args&&... args)                            \
    noexcept(noexcept(gobblefn(std::forward_as_tuple(args...),Petsc::util::make_index_sequence<sizeof...(Args)-(N)>{}))) \
    -> decltype(gobblefn(std::forward_as_tuple(args...),Petsc::util::make_index_sequence<sizeof...(Args)-(N)>{})) \
  {                                                                     \
    static_assert(std::is_integral<decltype(N)>::value,"");             \
    static_assert((N) >= 0,"");                                         \
    using seq = Petsc::util::make_index_sequence<sizeof...(Args)-(N)>;  \
    return gobblefn(std::forward_as_tuple(args...),seq{});              \
  }

#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(alias,original,N)     \
  PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS_(alias,original,PETSC_CONCAT(petsc_private_gobble_,original),N)
#else
#define PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS(alias,original,N)     \
  static_assert(0,"PETSC_ALIAS_FUNCTION_GOBBLE_NTH_LAST_ARGS() requires C++11")
#endif // C++11

// PETSC_CXX_COMPAT_DECL() - Helper macro to declare a C++ class member function or
// free-standing function guaranteed to be compatible with C
//
// input params:
// __VA_ARGS__ - the function declaration
//
// notes:
// Normally member functions of C++ structs or classes are not callable from C as they have an
// implicit "this" parameter tacked on the front (analogous to Pythons "self"). Static
// functions on the other hand do not have this restriction. This macro applies static to the
// function declaration as well as noexcept (as C++ exceptions escaping the C++ boundary is
// undefined behavior anyways) and [[nodiscard]].
//
// Note that the user should take care that function arguments and return type are also C
// compatible.
//
// example usage:
// class myclass
// {
// public:
//   PETSC_CXX_COMPAT_DECL(PetscErrorCode foo(int,Vec,char));
// };
//
// use this to define inline as well
//
// class myclass
// {
// public:
//   PETSC_CXX_COMPAT_DECL(PetscErrorCode foo(int a, Vec b, charc))
//   {
//     ...
//   }
// };
//
// or to define a free-standing function
//
// PETSC_CXX_COMPAT_DECL(bool bar(int x, int y))
// {
//   ...
// }
#define PETSC_CXX_COMPAT_DECL(...) PETSC_NODISCARD static __VA_ARGS__ PETSC_NOEXCEPT

// PETSC_CXX_COMPAT_DEFN() - Corresponding macro to define a C++ member function declared using
// PETSC_CXX_COMPAT_DECL()
//
// input params:
// __VA_ARGS__ - the function prototype (not the body!)
//
// notes:
// prepends inline and appends noexcept to the function
//
// example usage:
// PETSC_CXX_COMPAT_DEFN(PetscErrorCode myclass::foo(int a, Vec b, char c))
// {
//   ...
// }
#define PETSC_CXX_COMPAT_DEFN(...) PETSC_INLINE __VA_ARGS__ PETSC_NOEXCEPT

#endif // __cplusplus

#endif // PETSCTYPETRAITS_HPP
